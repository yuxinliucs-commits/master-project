import os
import numpy as np
import torch
from random import randint

from mcggs.scene.dataset_readers import sceneLoadPathCallbacks
from mcggs.scene.gaussian_model import GroupedGaussianModel
from mcggs.utils.image_utils import save_label_png


class Scene:
    """Camera container + SfM initialization points + model path management."""

    gaussians: GroupedGaussianModel = None

    def __init__(self, cfg, gaussians, load_iteration=None, shuffle=True,
                 mask_dir=None, resolution_scale=1.0):
        self.cfg = cfg
        data = cfg["data"]
        self.model_path = data["model_path"]
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration is not None:
            self.loaded_iter = load_iteration if isinstance(load_iteration, int) else -1
            all_iters = sorted(
                [int(x.split("_")[-1]) for x in os.listdir(os.path.join(self.model_path, "point_cloud"))
                 if x.startswith("iteration_")])
            if self.loaded_iter == -1:
                self.loaded_iter = all_iters[-1]
        elif os.path.exists(os.path.join(self.model_path, "point_cloud")):
            all_iters = sorted(
                [int(x.split("_")[-1]) for x in os.listdir(os.path.join(self.model_path, "point_cloud"))
                 if x.startswith("iteration_")])
            self.loaded_iter = all_iters[-1] if all_iters else None
        else:
            self.loaded_iter = None

        self.train_cameras = {}
        self.test_cameras = {}

        scene_type, loader = sceneLoadPathCallbacks(data["source_path"], int(data["resolution"]),
                                                    data["data_device"], bool(data["eval"]))
        self.scene_type = scene_type
        all_cams, xyz, rgb = loader(data["images"], mask_dir)

        if shuffle:
            rng = np.random.RandomState(0)
            order = rng.permutation(len(all_cams))
            all_cams = [all_cams[i] for i in order]

        if bool(data["eval"]) and len(all_cams) > 1:
            self.test_cameras[1.0] = [all_cams[i] for i in range(0, len(all_cams), 8)]
            self.train_cameras[1.0] = [all_cams[i] for i in range(len(all_cams)) if i % 8 != 0]
        else:
            self.train_cameras[1.0] = all_cams
            self.test_cameras[1.0] = []

        if self.loaded_iter is None:
            if xyz is None or len(xyz) == 0:
                # Blender-style scenes without SfM: initialize points in a 5 m sphere
                rng = np.random.RandomState(0)
                xyz = rng.uniform(-2.5, 2.5, (20000, 3)).astype(np.float32)
                rgb = rng.rand(20000, 3).astype(np.float32)
                print("[scene] no SfM points found; using random init on a 5m sphere")
            self.gaussians.create_from_pcd(xyz, rgb)
        else:
            self.gaussians.load_ply(os.path.join(
                self.model_path, "point_cloud", "iteration_%d" % self.loaded_iter, "point_cloud.ply"))

    def save(self, iteration, extra=None):
        out = os.path.join(self.model_path, "point_cloud", "iteration_%d" % iteration)
        os.makedirs(out, exist_ok=True)
        self.gaussians.save_ply(os.path.join(out, "point_cloud.ply"))
        if extra:
            extra(out)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
