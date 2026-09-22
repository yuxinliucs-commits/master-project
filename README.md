# master project


## Layout

```
planning_artifacts/    architecture diagram / file tree / config notes
analyzing_artifacts/   per-module implementation analysis (formula -> code location)
config.yaml            every hyperparameter (with provenance comments from the paper)
train_stage1.py        Stage 1: 3DGS reconstruction (15K iters, budget densification)
prepare_masks.py       per-view SAM masks (score >= 0.5; or reuse an existing raw_sam_mask)
associate_masks.py     multi-cue mask association (run once, before training)
train_stage2.py        Stage 2: identity encoding training (10K iters, geometry frozen)
render.py              render RGB / identity label maps
edit.py                remove / recompose / inpaint / colourise (CLI operation: colorize)
evaluate.py            PSNR/SSIM/LPIPS, IoU + optimal linear assignment, IoU drop, CLIP directional similarity
```

## Installation

```bash
conda create -n mcggs python=3.10 -y && conda activate mcggs
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Rasteriser (with identity channels) and simple-knn: build from the submodules of the
# local reference repository (do NOT copy its sources into this project)
pip install ../gaussian-grouping-main/submodules/diff-gaussian-rasterization
pip install ../gaussian-grouping-main/submodules/simple-knn
```

> The rasteriser must be the one bundled with the reference repository: its Python interface
> accepts `sh_objs` of shape (N, 1, 16) and returns `render_objects` of shape (16, H, W),
> which is exactly the extended channel set required for identity-encoding rendering.

## Usage (Replica / self-captured data)

```bash
# 0) Data preparation: COLMAP or transforms.json (for unpacked Replica see Gaga-main/data/replica)
# 1) Stage 1 reconstruction (15K iters; budget = budget_multiplier x number of SfM points, 5x for Replica / 2x for small scenes)
python train_stage1.py --config config.yaml
# 2) SAM masks (skip if raw_sam_mask already exists)
python prepare_masks.py --config config.yaml --sam_checkpoint /path/to/sam_vit_h.pth
# 3) Multi-cue mask association (run once; add --no_dinov2 if DINOv2 is unavailable)
python associate_masks.py --config config.yaml
# 4) Stage 2 identity encoding (10K iters)
python train_stage2.py --config config.yaml
# 5) Rendering and evaluation
python render.py --config config.yaml --mode both
python evaluate.py --mode render
python evaluate.py --mode seg --gt_label_dir /path/to/gt_panoptic
# 6) Editing (no retraining required)
python edit.py --operation remove    --target 3
python edit.py --operation recompose --target 2 --target_b 5
python edit.py --operation inpaint   --target 3
python edit.py --operation colourise --target 4 --colour 0.9 0.2 0.1
```

## Data conventions

- Associated mask directory: `<source_path>/sam_mask/` (uint8/uint16 label PNGs, 0 = background),
  produced by step 3.
- Panoptic GT evaluation directory: label PNGs named after the corresponding views
  (Replica panoptic annotations).
- Outputs: `<model_path>/point_cloud/iteration_*/point_cloud.ply` (carries `identity_0..15`
  channels in the PLY) and `embed_head.pth` (the 16->256 linear head).

## Hyperparameter quick reference (full details in the config.yaml comments)

Identity encoding 16-D; linear head 16->256; lambda_2D = 0.56; lambda_3D = 2.0; encoding lr 2e-3,
linear head lr 4e-4; 3D regularisation k = 5, n = 1000; SAM confidence threshold 0.5; IOCUR
threshold 0.1; front percentage 20%; 32 patches; merge score weights 0.625/0.25/0.125;
Stage 1 15K iters, Stage 2 10K iters; densification every 500 iterations.

## Known limitations

- The rasteriser / simple_knn / SAM / LaMa / DINOv2 / CLIP are external dependencies; when any
  module is missing the corresponding feature raises a clear error (if DINOv2 is absent the
  semantic cue gracefully degrades to zero while the depth/boundary cues keep working).
- Stage 1 densification rebuilds the Adam state after structural changes (the same engineering
  trade-off as the reference implementations).
- All code in this repository is an original implementation and contains no source or licence
  text from the reference repositories; please comply with the licences of those repositories
  when using their rasteriser submodules.
