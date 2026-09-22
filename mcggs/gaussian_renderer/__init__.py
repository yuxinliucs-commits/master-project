import math
import torch

try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    _RASTERIZER_OK = True
    _RASTERIZER_ERROR = ""
except Exception as _e:  # pragma: no cover
    _RASTERIZER_OK = False
    _RASTERIZER_ERROR = str(_e)


def _require_rasterizer():
    if not _RASTERIZER_OK:
        raise ImportError(
            "diff_gaussian_rasterization (with identity/object channels) is required.\n"
            "Build it from the submodules of your local reference repositories:\n"
            "  pip install <path-to>/gaussian-grouping-main/submodules/diff-gaussian-rasterization\n"
            f"Original import error: {_RASTERIZER_ERROR}")


def render(viewpoint_camera, pc, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
           override_color=None, use_identity: bool = True):
    """Render RGB and (optionally) the per-pixel identity feature map.

    The identity encoding of each Gaussian (`pc.get_identity`, shape (N,1,16)) is passed
    to the rasterizer via `sh_objs` (same convention as the reference implementation),
    returning `render_objects` with shape (16,H,W) (paper Sec. 3.1/3.5.2).
    """
    _require_rasterizer()

    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype,
                                          requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=(pipe.get("debug", False) if isinstance(pipe, dict) else pipe.debug),
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    scales = None
    rotations = None
    cov3D_precomp = None
    compute_cov3D_python = pipe.get("compute_cov3D_python", False) if isinstance(pipe, dict) else pipe.compute_cov3D_python
    convert_SHs_python = pipe.get("convert_SHs_python", False) if isinstance(pipe, dict) else pipe.convert_SHs_python
    if compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs = None
    colors_precomp = None
    if override_color is None:
        if convert_SHs_python:
            from mcggs.utils.sh_utils import eval_sh
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    kwargs = dict(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )
    if use_identity:
        kwargs["sh_objs"] = pc.get_identity

    outputs = rasterizer(**kwargs)
    if use_identity:
        rendered_image, radii, rendered_objects = outputs
    else:
        rendered_image, radii = outputs[0], outputs[1]
        rendered_objects = None

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "render_object": rendered_objects,
        "blend_weights": None,  # populated only by rasterizers that expose per-pixel blending
    }


def render_visibility_counts(viewpoint_cameras, pc, proj_device="cuda"):
    """Count, per pixel, how many Gaussians contribute across a set of views by projecting
    Gaussian centers (used for inpainting hole detection, paper Sec. 4.5.2).

    Returns a list of (H,W) count maps (float, normalized by max)."""
    results = []
    with torch.no_grad():
        xyz = pc.get_xyz
        ones = torch.ones(xyz.shape[0], 1, device=proj_device)
        for cam in viewpoint_cameras:
            proj = cam.full_proj_transform  # (4,4) transposed convention (world->clip)
            p_hom = torch.cat([xyz, ones], dim=1) @ proj
            z = p_hom[:, 3:].clamp(min=1e-8)
            p_ndc = p_hom[:, :3] / z
            H, W = cam.image_height, cam.image_width
            u = ((p_ndc[:, 0] + 1.0) * W - 1.0) * 0.5
            v = ((p_ndc[:, 1] + 1.0) * H - 1.0) * 0.5
            infront = p_hom[:, 3] > 0.01
            inside = (u >= 0) & (u < W) & (v >= 0) & (v < H) & infront
            ui = u[inside].long().clamp(0, W - 1)
            vi = v[inside].long().clamp(0, H - 1)
            count = torch.zeros(H * W, device=proj_device)
            count.index_add_(0, vi * W + ui, torch.ones_like(ui, dtype=torch.float))
            results.append(count.view(H, W))
    return results
