import torch
import torch.nn as nn
import torch.nn.functional as F

from vision3d.layers import ConvBlock, NonRigidICP
from vision3d.ops import (
    apply_deformation,
    build_euclidean_deformation_graph,
    index_select,
    pairwise_distance,
)

# isort: split
from graphsc import GraphSCModule


class GraphSCNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.max_local_correspondences = cfg.model.max_local_correspondences
        self.min_local_correspondences = cfg.model.min_local_correspondences
        self.num_anchors = cfg.model.deformation_graph.num_anchors
        self.node_coverage = cfg.model.deformation_graph.node_coverage

        self.acceptance_score = cfg.eval.acceptance_score

        self.encoder = GraphSCModule(
            cfg.model.transformer.input_dim,
            cfg.model.transformer.output_dim,
            cfg.model.transformer.hidden_dim,
            cfg.model.transformer.num_heads,
            cfg.model.transformer.num_blocks,
            cfg.model.transformer.num_layers_per_block,
            cfg.model.transformer.sigma_d,
            embedding_k=cfg.model.transformer.embedding_k,
            embedding_dim=cfg.model.transformer.embedding_dim,
            dropout=cfg.model.transformer.dropout,
            act_cfg=cfg.model.transformer.activation_fn,
        )

        self.classifier = nn.Sequential(
            ConvBlock(
                in_channels=cfg.model.classifier.input_dim,
                out_channels=cfg.model.classifier.input_dim // 2,
                kernel_size=1,
                conv_cfg="Conv1d",
                norm_cfg="GroupNorm",
                act_cfg="LeakyReLU",
                dropout=cfg.model.classifier.dropout,
            ),
            ConvBlock(
                in_channels=cfg.model.classifier.input_dim // 2,
                out_channels=cfg.model.classifier.input_dim // 4,
                kernel_size=1,
                conv_cfg="Conv1d",
                norm_cfg="GroupNorm",
                act_cfg="LeakyReLU",
                dropout=cfg.model.classifier.dropout,
            ),
            ConvBlock(
                in_channels=cfg.model.classifier.input_dim // 4,
                out_channels=1,
                kernel_size=1,
                conv_cfg="Conv1d",
                norm_cfg="None",
                act_cfg="None",
            ),
        )

        self.sigma_d = cfg.model.transformer.sigma_d
        self.sigma_f = nn.Parameter(torch.as_tensor(1.0))

        self.registration = NonRigidICP(
            corr_lambda=cfg.model.nicp.corr_lambda,
            arap_lambda=cfg.model.nicp.arap_lambda,
            lm_lambda=cfg.model.nicp.lm_lambda,
            num_iterations=cfg.model.nicp.num_iterations,
        )

    def forward(self, data_dict):
        output_dict = {}

        # 1. unpack data
        src_points = data_dict["src_points"]  # (Ns, 3)
        tgt_points = data_dict["tgt_points"]  # (Nt, 3)

        src_corr_points = data_dict["src_corr_points"]  # (C,)
        tgt_corr_points = data_dict["tgt_corr_points"]  # (C,)
        num_correspondences = src_corr_points.shape[0]

        node_indices = data_dict["node_indices"]  # (M,)
        src_nodes = src_points[node_indices]  # (M, 3)
        num_nodes = src_nodes.shape[0]

        output_dict["src_points"] = src_points
        output_dict["tgt_points"] = tgt_points
        output_dict["src_nodes"] = src_nodes

        # 2. build deformation graph
        corr_anchor_indices, corr_anchor_weights = build_euclidean_deformation_graph(
            src_corr_points,
            src_nodes,
            self.num_anchors,
            self.node_coverage,
            return_node_graph=False,
        )  # (C, Ka)

        # 2. compute node-to-correspondence weights
        anchor_masks = torch.ne(corr_anchor_indices, -1)  # (C, Ka)
        anchor_corr_indices, anchor_col_indices = torch.nonzero(
            anchor_masks, as_tuple=True
        )  # (S,), (S,)
        anchor_node_indices = corr_anchor_indices[
            anchor_corr_indices, anchor_col_indices
        ]  # (S,)
        anchor_weights = corr_anchor_weights[
            anchor_corr_indices, anchor_col_indices
        ]  # (S,)
        node_to_corr_weights = torch.zeros(
            size=(num_nodes, num_correspondences)
        ).cuda()  # (M, C)
        node_to_corr_weights[
            anchor_node_indices, anchor_corr_indices
        ] = anchor_weights  # (M, C)

        # 3. assign correspondences to nodes
        max_local_correspondences = (
            torch.gt(node_to_corr_weights, 0.0).sum(dim=1).max().item()
        )
        max_local_correspondences = min(
            max_local_correspondences, self.max_local_correspondences
        )
        local_corr_weights, local_corr_indices = node_to_corr_weights.topk(
            k=max_local_correspondences, dim=1, largest=True
        )  # (M, k), (M, k)
        local_corr_masks = torch.gt(local_corr_weights, 0.0)  # (M, k)

        # 4. remove small nodes
        local_corr_counts = local_corr_masks.sum(dim=-1)  # (M,)
        node_masks = torch.gt(local_corr_counts, self.min_local_correspondences)  # (M,)
        local_corr_indices = local_corr_indices[node_masks]  # (M', k)
        local_corr_weights = local_corr_weights[node_masks]  # (M', k)
        local_corr_masks = local_corr_masks[node_masks]  # (M', k)

        output_dict["local_corr_indices"] = local_corr_indices
        output_dict["local_corr_weights"] = local_corr_weights
        output_dict["local_corr_masks"] = local_corr_masks

        # 5. transformer encoder
        corr_feats, corr_masks = self.encoder(
            src_corr_points,
            tgt_corr_points,
            local_corr_indices,
            local_corr_weights,
            local_corr_masks,
        )  # (C, d) (C,)

        corr_feats_norm = F.normalize(corr_feats, p=2, dim=1)  # (C, d)
        output_dict["corr_feats"] = corr_feats_norm
        output_dict["sigma_f"] = self.sigma_f

        # 6. classifier
        corr_feats = corr_feats.transpose(0, 1).unsqueeze(0)  # (1, d, C)
        corr_logits = self.classifier(corr_feats)
        corr_logits = corr_logits.flatten()  # (C,)
        corr_scores = torch.sigmoid(corr_logits)

        # oracle
        # corr_scores = data_dict["corr_labels"].float()
        # all
        # corr_scores = torch.ones_like(corr_scores)

        output_dict["corr_logits"] = corr_logits
        output_dict["corr_scores"] = corr_scores
        output_dict["corr_masks"] = corr_masks

        # 8. feature consistency
        local_corr_feats_norm = index_select(
            corr_feats_norm, local_corr_indices, dim=0
        )  # (M', k, d)
        local_affinity_mat = pairwise_distance(
            local_corr_feats_norm, local_corr_feats_norm, normalized=True, squared=False
        )  # (M', k, k)
        local_fc_mat = torch.relu(1.0 - local_affinity_mat.pow(2) / self.sigma_f.pow(2))

        output_dict["feature_consistency"] = local_fc_mat

        if data_dict.get("registration", False):
            (
                anchor_indices,
                anchor_weights,
                edges_indices,
                edge_weights,
            ) = build_euclidean_deformation_graph(
                src_points, src_nodes, self.num_anchors, self.node_coverage
            )
            edge_weights = torch.ones_like(
                edge_weights
            )  # use the same weights for all edges

            output_dict["anchor_indices"] = anchor_indices
            output_dict["anchor_weights"] = anchor_weights

            # 0/1 weighting
            # corr_logits = torch.sigmoid(corr_logits)
            # corr_logits = torch.gt(corr_logits, self.acceptance_score).float()
            # corr_anchor_weights = corr_anchor_weights * corr_logits.unsqueeze(1)
            # corr_masks = torch.gt(corr_anchor_weights.sum(1), 0)
            # src_corr_points = src_corr_points[corr_masks]
            # tgt_corr_points = tgt_corr_points[corr_masks]
            # corr_anchor_indices = corr_anchor_indices[corr_masks]
            # corr_anchor_weights = corr_anchor_weights[corr_masks]

            # dynamic weighting
            corr_masks = torch.gt(corr_scores, self.acceptance_score)
            src_corr_points = src_corr_points[corr_masks]
            tgt_corr_points = tgt_corr_points[corr_masks]
            corr_anchor_indices = corr_anchor_indices[corr_masks]
            corr_anchor_weights = corr_anchor_weights[corr_masks]
            corr_scores = corr_scores[corr_masks]

            transforms = self.registration(
                src_nodes,
                src_corr_points,
                tgt_corr_points,
                corr_anchor_indices,
                corr_anchor_weights,
                edges_indices,
                # corr_weights=corr_weights,
                # edge_weights=edge_weights,
            )

            output_dict["embedded_deformation_nodes"] = src_nodes
            output_dict["embedded_deformation_transforms"] = transforms

            warped_src_points = apply_deformation(
                src_points, src_nodes, transforms, anchor_indices, anchor_weights
            )
            output_dict["warped_src_points"] = warped_src_points

        return output_dict


def create_model(cfg):
    model = GraphSCNet(cfg)
    return model


def main():
    from config import make_cfg

    cfg = make_cfg()
    model = create_model(cfg)
    print(model.state_dict().keys())
    print(model)


if __name__ == "__main__":
    main()
