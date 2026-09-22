"""Stage 1: vanilla 3DGS reconstruction with budget-controlled densification
(paper Sec. 3.7 / 4.1: 15K iterations)."""
import os
import sys
import uuid
from random import randint
from argparse import ArgumentParser, Namespace

import torch
from tqdm import tqdm

from mcggs.config import load_config
from mcggs.scene import Scene
from mcggs.scene.gaussian_model import GroupedGaussianModel
from mcggs.gaussian_renderer import render
from mcggs.utils.loss_utils import l1_loss, ssim
from mcggs.utils.image_utils import psnr
from mcggs.utils.general_utils import set_seed
from mcggs.densification.schedule import parabolic_budget_schedule, resolve_budget


def training(cfg):
    data = cfg["data"]
    s1 = cfg["stage1"]
    dens = cfg["densification"]

    model_path = data["model_path"]
    os.makedirs(model_path, exist_ok=True)
    with open(os.path.join(model_path, "cfg_stage1.json"), "w") as f:
        import json
        json.dump(cfg, f, indent=2, default=str)

    gaussians = GroupedGaussianModel(data["sh_degree"])
    scene = Scene(cfg, gaussians)
    gaussians.training_setup(cfg, stage="stage1")

    bg_color = [1, 1, 1] if data["white_background"] else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    pipe = cfg["pipeline"]

    # budget-controlled densification schedule (paper Sec. 3.7)
    start_count = gaussians.get_xyz.shape[0]
    budget = resolve_budget(start_count, {"mode": dens["budget_mode"],
                                          "value": dens["budget_multiplier"]})
    schedule = parabolic_budget_schedule(start_count, budget,
                                         s1["densify_from_iter"], s1["densify_until_iter"],
                                         dens["densification_interval"]) \
        if dens["enabled"] else None
    print("[stage1] start=%d gaussians, budget=%d, densification steps=%d"
          % (start_count, budget, len(schedule or [])))

    viewpoint_stack = None
    ema = 0.0
    progress = tqdm(range(1, s1["iterations"] + 1), desc="stage1")
    for iteration in progress:
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        render_pkg = render(cam, gaussians, pipe, background, use_identity=False)
        image = render_pkg["render"]
        gt_image = cam.original_image.cuda()

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - s1["lambda_dssim"]) * Ll1 + s1["lambda_dssim"] * (1.0 - ssim(image, gt_image))
        loss.backward()

        with torch.no_grad():
            ema = 0.4 * loss.item() + 0.6 * ema
            if iteration % 10 == 0:
                progress.set_postfix({"Loss": "%.7f" % ema, "N": gaussians._xyz.shape[0]})

            vis = render_pkg["visibility_filter"]
            radii = render_pkg["radii"]
            viewspace = render_pkg["viewspace_points"]

            if dens["enabled"] and iteration < s1["densify_until_iter"]:
                gaussians.max_radii2D[vis] = torch.max(gaussians.max_radii2D[vis], radii[vis].float())
                gaussians.add_densification_stats(viewspace, vis)

                if iteration > s1["densify_from_iter"] and \
                        (iteration - s1["densify_from_iter"]) % dens["densification_interval"] == 0:
                    step_idx = (iteration - s1["densify_from_iter"]) // dens["densification_interval"] - 1
                    target = schedule[min(step_idx, len(schedule) - 1)]
                    scores = gaussians.compute_importance_scores(
                        scene.getTrainCameras(), render, background, dens["score_weights"])
                    gaussians.densify_by_score(target, scores)
                    size_threshold = 20 if iteration > s1["opacity_reset_interval"] else None
                    gaussians.standard_prune(size_threshold=size_threshold)
                    # keep optimizer params in sync after structure change
                    gaussians.training_setup(cfg, stage="stage1")
                    gaussians.optimizer.zero_grad(set_to_none=True)

                if iteration % s1["opacity_reset_interval"] == 0 or \
                        (data["white_background"] and iteration == s1["densify_from_iter"]):
                    from mcggs.utils.general_utils import inverse_sigmoid
                    op_new = inverse_sigmoid(
                        torch.min(gaussians.get_opacity, torch.ones_like(gaussians.get_opacity) * 0.01))
                    gaussians._opacity.data = op_new

            if iteration < s1["iterations"]:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in cfg["log"]["test_iterations"] and len(scene.getTestCameras()):
                torch.cuda.empty_cache()
                ps = 0.0
                for tcam in scene.getTestCameras()[:10]:
                    with torch.no_grad():
                        im = render(tcam, gaussians, pipe, background, use_identity=False)["render"]
                        ps += psnr(im, tcam.original_image.cuda()).mean().double()
                print("\n[iter %d] test PSNR %.2f" % (iteration, ps / min(10, len(scene.getTestCameras()))))
                torch.cuda.empty_cache()

            if iteration in cfg["log"]["save_iterations"] or iteration == s1["iterations"]:
                scene.save(iteration)
                print("\n[iter %d] saved to %s" % (iteration, model_path))

    print("\nStage-1 training complete. Final gaussian count:", gaussians._xyz.shape[0])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    set_seed(args.seed)
    training(load_config(args.config))
