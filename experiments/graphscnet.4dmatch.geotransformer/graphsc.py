from typing import Optional, Union

import ipdb
import torch
import torch.nn as nn
from torch import Tensor

from vision3d.layers import ConvBlock, FourierEmbedding, TransformerLayer
from vision3d.ops import index_select, spatial_consistency


class GraphSCModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_blocks: int,
        num_layers_per_block: int,
        sigma_d: float,
        embedding_k: int = 0,
        embedding_dim: int = 10,
        dropout: Optional[float] = None,
        act_cfg: Union[str, dict] = "ReLU",
    ):
        super().__init__()

        self.sigma_d = sigma_d
        self.eps = 1e-8

        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.num_layers_per_block = num_layers_per_block

        self.embedding = FourierEmbedding(length=embedding_dim, k0=embedding_k, use_pi=False, use_input=True)

        self.in_proj = nn.Sequential(
            ConvBlock(
                in_channels=input_dim * (2 * embedding_dim + 1),
                out_channels=hidden_dim,
                kernel_size=1,
                conv_cfg="Conv1d",
                norm_cfg="GroupNorm",
                act_cfg="LeakyReLU",
            ),
            ConvBlock(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=1,
                conv_cfg="Conv1d",
                norm_cfg="GroupNorm",
                act_cfg="LeakyReLU",
            ),
            ConvBlock(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=1,
                conv_cfg="Conv1d",
                norm_cfg="None",
                act_cfg="None",
            ),
        )
        self.out_proj = nn.Linear(hidden_dim, output_dim)

        layers = []
        for i in range(num_layers_per_block * num_blocks):
            layers.append(TransformerLayer(hidden_dim, num_heads, dropout=dropout, act_cfg=act_cfg))
        self.transformer = nn.ModuleList(layers)

    def forward(
        self,
        src_corr_points: Tensor,
        tgt_corr_points: Tensor,
        local_corr_indices: Tensor,
        local_corr_weights: Tensor,
        local_corr_masks: Tensor,
    ):
        """LOCSC Transformer Module.

        Args:
            src_corr_points (Tensor): The correspondence points in source point cloud (C, 3).
            tgt_corr_points (Tensor): The correspondence points in target point cloud (C, 3).
            local_corr_indices (LongTensor): The local indices for the correspondences (M, k).
            local_corr_weights (Tensor): The local weights for the correspondences (M, k).
            local_corr_masks (BoolTensor): The local masks for the correspondences (M, k).
        """
        num_correspondences = src_corr_points.shape[0]

        # 1. input projection
        src_corr_points_norm = src_corr_points - src_corr_points.mean(dim=0, keepdim=True)  # (C, 3)
        tgt_corr_points_norm = tgt_corr_points - tgt_corr_points.mean(dim=0, keepdim=True)  # (C, 3)
        src_corr_embeddings = self.embedding(src_corr_points_norm)  # (C, 6L)
        tgt_corr_embeddings = self.embedding(tgt_corr_points_norm)  # (C, 6L)
        corr_embeddings = torch.cat([src_corr_embeddings, tgt_corr_embeddings], dim=-1)  # (C, 12L)
        corr_feats = self.in_proj(corr_embeddings.transpose(0, 1).unsqueeze(0)).squeeze(0).transpose(0, 1)  # (C, d)

        # 2. spatial consistency
        local_src_corr_points = index_select(src_corr_points_norm, local_corr_indices, dim=0)  # (M, k, 3)
        local_tgt_corr_points = index_select(tgt_corr_points_norm, local_corr_indices, dim=0)  # (M, k, 3)
        sc_weights = spatial_consistency(local_src_corr_points, local_tgt_corr_points, self.sigma_d)  # (M, k, k)

        # 3. prepare for aggregation
        flat_local_corr_indices = local_corr_indices.view(-1)  # (Mxk)
        flat_local_corr_weights = local_corr_weights.view(-1)  # (Mxk)
        corr_sum_weights = torch.zeros(size=(num_correspondences,)).cuda()  # (C,)
        corr_sum_weights.scatter_add_(dim=0, index=flat_local_corr_indices, src=flat_local_corr_weights)  # (C,)
        flat_local_corr_sum_weights = corr_sum_weights[flat_local_corr_indices]  # (Mxk)
        flat_local_corr_weights = flat_local_corr_weights / (flat_local_corr_sum_weights + self.eps)  # (Mxk)
        flat_local_corr_weights = flat_local_corr_weights.unsqueeze(1).expand(-1, self.hidden_dim)  # (Mxk, d)
        flat_local_corr_indices = flat_local_corr_indices.unsqueeze(1).expand(-1, self.hidden_dim)  # (Mxk, d)

        # 4. transformer module
        local_corr_masks = torch.logical_not(local_corr_masks)
        for block_idx in range(self.num_blocks):
            # 4.1 grouping
            local_corr_feats = index_select(corr_feats, local_corr_indices, dim=0)  # (M, k, d)

            # 4.2 transformer
            for layer_idx in range(self.num_layers_per_block):
                index = block_idx * self.num_layers_per_block + layer_idx
                local_corr_feats = self.transformer[index](
                    local_corr_feats,
                    local_corr_feats,
                    local_corr_feats,
                    qk_weights=sc_weights,
                    k_masks=local_corr_masks,
                )

            # 4.3 aggregate
            flat_local_corr_feats = local_corr_feats.view(-1, self.hidden_dim)  # (Mxk, d)
            flat_local_corr_feats = flat_local_corr_feats * flat_local_corr_weights  # (Mxk, d)
            corr_feats = torch.zeros(size=(num_correspondences, self.hidden_dim)).cuda()  # (C, d)
            corr_feats.scatter_add_(dim=0, index=flat_local_corr_indices, src=flat_local_corr_feats)  # (C, d)

        # 5. output projection
        corr_feats = self.out_proj(corr_feats)
        corr_masks = torch.gt(corr_sum_weights, 0.0)  # (C,)

        return corr_feats, corr_masks
