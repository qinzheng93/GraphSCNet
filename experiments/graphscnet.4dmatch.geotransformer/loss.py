import ipdb
import torch
import torch.nn as nn

from vision3d.loss import SigmoidFocalLossWithLogits
from vision3d.ops import apply_deformation, apply_transform
from vision3d.ops.metrics import (
    compute_nonrigid_feature_matching_recall,
    compute_scene_flow_accuracy,
    compute_scene_flow_outlier_ratio,
    evaluate_binary_classification,
)


class LossFunction(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.focal_loss = SigmoidFocalLossWithLogits(reduction="mean")
        self.f_loss_weight = cfg.loss.focal_loss.weight
        self.c_loss_weight = cfg.loss.consistency_loss.weight

    def forward(self, data_dict, output_dict):
        # focal loss
        logits = output_dict["corr_logits"]
        labels = data_dict["corr_labels"].float()
        f_loss = self.focal_loss(logits, labels) * self.f_loss_weight

        # feature consistency loss
        fc_mat = output_dict["feature_consistency"]
        local_corr_indices = output_dict["local_corr_indices"]
        local_corr_masks = output_dict["local_corr_masks"]
        local_corr_labels = labels[local_corr_indices]
        fc_labels = local_corr_labels.unsqueeze(2) * local_corr_labels.unsqueeze(1)
        fc_masks = torch.logical_and(local_corr_masks.unsqueeze(2), local_corr_masks.unsqueeze(1))
        loss_mat = (fc_mat - fc_labels).pow(2)
        c_loss = loss_mat[fc_masks].mean() * self.c_loss_weight

        # total loss
        loss = f_loss + c_loss

        return {"loss": loss, "f_loss": f_loss, "c_loss": c_loss}


class EvalFunction(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.acceptance_score = cfg.eval.acceptance_score
        self.acceptance_radius = cfg.eval.acceptance_radius
        self.distance_limit = cfg.eval.distance_limit

    def forward(self, data_dict, output_dict):
        result_dict = {}

        # inlier/outlier classification
        scores = output_dict["corr_scores"]
        labels = data_dict["corr_labels"].float()
        precision, recall = evaluate_binary_classification(
            scores, labels, positive_threshold=self.acceptance_score, use_logits=False
        )

        corr_masks = output_dict["corr_masks"]
        hit_ratio = corr_masks.float().mean().nan_to_num_()

        result_dict["precision"] = precision
        result_dict["recall"] = recall
        result_dict["hit_ratio"] = hit_ratio

        # non-rigid inlier ratio, non-rigid feature matching recall
        src_points = data_dict["src_points"]
        scene_flows = data_dict["scene_flows"]
        transform = data_dict["transform"]
        src_corr_points = data_dict["src_corr_points"]
        tgt_corr_points = data_dict["tgt_corr_points"]
        corr_masks = torch.gt(scores, self.acceptance_score)
        if corr_masks.sum() > 0:
            src_corr_points = src_corr_points[corr_masks]
            tgt_corr_points = tgt_corr_points[corr_masks]

        if "test_indices" in data_dict:
            test_indices = data_dict["test_indices"]
            nfmr = compute_nonrigid_feature_matching_recall(
                src_corr_points,
                tgt_corr_points,
                src_points,
                scene_flows,
                test_indices,
                transform=transform,
                acceptance_radius=self.acceptance_radius,
                distance_limit=self.distance_limit,
            )
            result_dict["NFMR"] = nfmr

        # overlap coverage
        gt_src_corr_indices = data_dict["gt_src_corr_indices"]
        src_overlap_indices = torch.unique(gt_src_corr_indices)
        coverage = compute_nonrigid_feature_matching_recall(
            src_corr_points,
            tgt_corr_points,
            src_points,
            scene_flows,
            src_overlap_indices,
            transform=transform,
            acceptance_radius=self.acceptance_radius,
            distance_limit=self.distance_limit,
        )
        result_dict["coverage"] = coverage

        if "embedded_deformation_transforms" in output_dict:
            nodes = output_dict["embedded_deformation_nodes"]  # (M, 3)
            node_transforms = output_dict["embedded_deformation_transforms"]  # (M, 4, 4)
            anchor_indices = output_dict["anchor_indices"]
            anchor_weights = output_dict["anchor_weights"]

            warped_src_points = apply_deformation(src_points, nodes, node_transforms, anchor_indices, anchor_weights)
            aligned_src_points = apply_transform(src_points + scene_flows, transform)
            warped_scene_flows = warped_src_points - src_points
            aligned_scene_flows = aligned_src_points - src_points
            epe = torch.linalg.norm(warped_scene_flows - aligned_scene_flows, dim=1).mean()
            acc_s = compute_scene_flow_accuracy(warped_scene_flows, aligned_scene_flows, 0.025, 0.025)
            acc_r = compute_scene_flow_accuracy(warped_scene_flows, aligned_scene_flows, 0.05, 0.05)
            outlier_ratio = compute_scene_flow_outlier_ratio(warped_scene_flows, aligned_scene_flows, None, 0.3)

            result_dict["EPE"] = epe
            result_dict["AccS"] = acc_s
            result_dict["AccR"] = acc_r
            result_dict["OR"] = outlier_ratio

        result_dict["nCorr"] = src_corr_points.shape[0]

        return result_dict
