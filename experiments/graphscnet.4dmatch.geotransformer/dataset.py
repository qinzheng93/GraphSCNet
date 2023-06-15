import os.path as osp
import random
from typing import Callable

import numpy as np

from vision3d.array_ops import (
    apply_transform,
    compose_transforms,
    furthest_point_sample,
    get_transform_from_rotation_translation,
    inverse_transform,
    random_sample_small_transform,
)
from vision3d.datasets.registration import FourDMatchPairDataset
from vision3d.utils.collate import SimpleRegistrationCollateFnPackMode
from vision3d.utils.dataloader import build_dataloader
from vision3d.utils.profiling import profile_cpu_runtime


class TransformFunction(Callable):
    """TransformFunction function.

    1. Read corr data.
    2. Sample nodes with FPS. Compared with RS, FPS tends to push the nodes to the boundary.
    """

    def __init__(self, cfg, subset, use_augmentation):
        subset = subset.split("-")[0]
        self.corr_dir = osp.join(cfg.data.dataset_dir, "correspondences", subset)
        self.use_augmentation = use_augmentation
        self.aug_noise = 0.002
        self.node_coverage = cfg.model.deformation_graph.node_coverage

    def __call__(self, data_dict):
        # read corr data
        corr_dict = np.load(osp.join(self.corr_dir, data_dict["filename"]))
        data_dict["src_corr_points"] = corr_dict["src_corr_points"].astype(np.float32)
        data_dict["tgt_corr_points"] = corr_dict["tgt_corr_points"].astype(np.float32)
        data_dict["corr_scene_flows"] = corr_dict["corr_scene_flows"].astype(np.float32)
        data_dict["corr_labels"] = corr_dict["corr_labels"].astype(np.int64)

        # augmentation
        if self.use_augmentation:
            src_points = data_dict["src_points"]
            tgt_points = data_dict["tgt_points"]
            scene_flows = data_dict["scene_flows"]
            transform = data_dict["transform"]
            src_corr_points = data_dict["src_corr_points"]
            tgt_corr_points = data_dict["tgt_corr_points"]
            corr_scene_flows = data_dict["corr_scene_flows"]

            deformed_src_points = src_points + scene_flows
            deformed_src_corr_points = src_corr_points + corr_scene_flows
            aug_transform = random_sample_small_transform()
            if random.random() > 0.5:
                tgt_center = tgt_points.mean(axis=0)
                subtract_center = get_transform_from_rotation_translation(
                    None, -tgt_center
                )
                add_center = get_transform_from_rotation_translation(None, tgt_center)
                aug_transform = compose_transforms(
                    subtract_center, aug_transform, add_center
                )
                tgt_points = apply_transform(tgt_points, aug_transform)
                tgt_corr_points = apply_transform(tgt_corr_points, aug_transform)
                transform = compose_transforms(transform, aug_transform)
            else:
                src_center = src_points.mean(axis=0)
                subtract_center = get_transform_from_rotation_translation(
                    None, -src_center
                )
                add_center = get_transform_from_rotation_translation(None, src_center)
                aug_transform = compose_transforms(
                    subtract_center, aug_transform, add_center
                )
                src_points = apply_transform(src_points, aug_transform)
                src_corr_points = apply_transform(src_corr_points, aug_transform)
                deformed_src_points = apply_transform(
                    deformed_src_points, aug_transform
                )
                deformed_src_corr_points = apply_transform(
                    deformed_src_corr_points, aug_transform
                )
                inv_aug_transform = inverse_transform(aug_transform)
                transform = compose_transforms(inv_aug_transform, transform)

            src_points += (
                np.random.rand(src_points.shape[0], 3) - 0.5
            ) * self.aug_noise
            tgt_points += (
                np.random.rand(tgt_points.shape[0], 3) - 0.5
            ) * self.aug_noise
            src_corr_points += (
                np.random.rand(src_corr_points.shape[0], 3) - 0.5
            ) * self.aug_noise
            tgt_corr_points += (
                np.random.rand(tgt_corr_points.shape[0], 3) - 0.5
            ) * self.aug_noise
            scene_flows = deformed_src_points - src_points
            corr_scene_flows = deformed_src_corr_points - src_corr_points

            data_dict["src_points"] = src_points.astype(np.float32)
            data_dict["tgt_points"] = tgt_points.astype(np.float32)
            data_dict["scene_flows"] = scene_flows.astype(np.float32)
            data_dict["src_corr_points"] = src_corr_points.astype(np.float32)
            data_dict["tgt_corr_points"] = tgt_corr_points.astype(np.float32)
            data_dict["corr_scene_flows"] = corr_scene_flows.astype(np.float32)
            data_dict["transform"] = transform.astype(np.float32)

        # sample nodes
        src_points = data_dict["src_points"]
        src_node_indices = furthest_point_sample(
            src_points, min_distance=self.node_coverage
        )
        data_dict["node_indices"] = src_node_indices.astype(np.int64)
        return data_dict


def train_valid_data_loader(cfg):
    train_dataset = FourDMatchPairDataset(
        cfg.data.dataset_dir,
        "val",
        transform_fn=TransformFunction(cfg, "val", cfg.train.use_augmentation),
        use_augmentation=False,
        return_corr_indices=cfg.train.return_corr_indices,
    )

    train_loader = build_dataloader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        collate_fn=SimpleRegistrationCollateFnPackMode(),
    )

    valid_dataset = FourDMatchPairDataset(
        cfg.data.dataset_dir,
        "4DMatch",
        transform_fn=TransformFunction(cfg, "4DMatch", False),
        use_augmentation=False,
        return_corr_indices=cfg.test.return_corr_indices,
    )
    valid_loader = build_dataloader(
        valid_dataset,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        collate_fn=SimpleRegistrationCollateFnPackMode(),
    )

    return train_loader, valid_loader


def test_data_loader(cfg, benchmark):
    test_dataset = FourDMatchPairDataset(
        cfg.data.dataset_dir,
        benchmark,
        transform_fn=TransformFunction(cfg, benchmark, False),
        use_augmentation=False,
        return_corr_indices=cfg.test.return_corr_indices,
        shape_names=cfg.test.shape_names,
    )

    test_loader = build_dataloader(
        test_dataset,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        collate_fn=SimpleRegistrationCollateFnPackMode(),
    )

    return test_loader


def run_test():
    import numpy as np
    from config import make_cfg
    from tqdm import tqdm

    from vision3d.array_ops import apply_transform
    from vision3d.utils.open3d import (
        draw_geometries,
        get_color,
        make_open3d_point_cloud,
    )
    from vision3d.utils.tensor import tensor_to_array

    def visualize(points_f, points_c):
        pcd = make_open3d_point_cloud(points_f.detach().cpu().numpy())
        pcd.paint_uniform_color([0, 0, 1])
        ncd = make_open3d_point_cloud(points_c.detach().cpu().numpy())
        ncd.paint_uniform_color([1, 0, 0])
        draw_geometries(pcd, ncd)

    cfg = make_cfg()
    train_loader, val_loader = train_valid_data_loader(cfg)

    node_coverage = cfg.model.deformation_graph.node_coverage

    sel_loader = val_loader

    pbar = tqdm(enumerate(sel_loader), total=len(sel_loader))
    for i, data_dict in pbar:
        data_dict = tensor_to_array(data_dict)
        points = data_dict["points"]
        src_length = data_dict["lengths"][0]
        src_points = points[:src_length]
        tgt_points = points[src_length:]
        scene_flows = data_dict["scene_flows"]
        transform = data_dict["transform"]
        deformed_src_points = src_points + scene_flows
        aligned_src_points = apply_transform(deformed_src_points, transform)

        with profile_cpu_runtime("fps"):
            fps_node_indices = furthest_point_sample(
                src_points, min_distance=node_coverage
            )
        # fps_node_indices = data_dict["fps_node_indices"]
        fps_src_nodes = src_points[fps_node_indices]

        # rnd_node_indices = data_dict["rnd_node_indices"]
        # rnd_src_nodes = src_points[rnd_node_indices]

        src_pcd = make_open3d_point_cloud(src_points)
        src_pcd.paint_uniform_color(get_color("blue"))
        fps_src_ncd = make_open3d_point_cloud(fps_src_nodes)
        fps_src_ncd.paint_uniform_color(get_color("red"))
        draw_geometries(src_pcd, fps_src_ncd)
        # rnd_src_ncd = make_open3d_point_cloud(rnd_src_nodes)
        # rnd_src_ncd.paint_uniform_color(get_color("lime"))
        # draw_geometries(src_pcd, fps_src_ncd, rnd_src_ncd)


if __name__ == "__main__":
    run_test()
