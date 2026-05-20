"""
GAN discriminator for HFM.

Takes the decoded frame plus the feature and system token hierarchies and
outputs a real/fake logit.  Feature and system tokens are included as
conditioning context — no direct value targets are imposed on them; the
only supervision signal is the binary real/fake classification.
"""

import torch
import torch.nn as nn
from typing import List

from .config import HFMConfig


class _FrameBranch(nn.Module):
    """Strided-conv encoder that maps a pixel frame to a fixed-size vector."""

    def __init__(self, in_channels: int, out_dim: int):
        super().__init__()
        channels = [in_channels, 64, 128, 256, 512]
        layers: list = []
        for i in range(len(channels) - 1):
            layers.append(
                nn.Conv2d(channels[i], channels[i + 1], 4, stride=2, padding=1,
                          bias=(i == 0))
            )
            if i > 0:
                layers.append(nn.InstanceNorm2d(channels[i + 1], affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(512, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.pool(self.conv(x)).flatten(1))   # [B, out_dim]


class HFMDiscriminator(nn.Module):
    """
    Discriminator for real vs. generated fluid frames.

    Inputs
    ------
    frame : [B, C, H, W]
        Decoded pixel-space frame (model output or ground truth).
    feat  : list of [B, S_l, S_l, d_feat_l]
        Feature token hierarchy from HFM.  Included as conditioning context;
        no direct value targets are imposed on these tokens.
    sys   : list of [B, S_l, S_l, d_sys_l]
        System token hierarchy.

    Output
    ------
    [B, 1]  un-normalised real/fake logit (use BCEWithLogitsLoss).
    """

    def __init__(self, cfg: HFMConfig):
        super().__init__()
        d = cfg.disc_dim

        self.frame_branch = _FrameBranch(cfg.in_channels, d)

        # GAP + project each hierarchy level to a common dim
        self.feat_projs = nn.ModuleList([
            nn.Linear(d_feat, d) for d_feat in cfg.d_feat
        ])
        self.sys_projs = nn.ModuleList([
            nn.Linear(d_sys, d) for d_sys in cfg.d_sys
        ])

        n = cfg.n_levels
        fuse_dim = d * (1 + n + n)   # frame + all feat levels + all sys levels

        self.head = nn.Sequential(
            nn.Linear(fuse_dim, d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(d * 2, d),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(d, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        frame: torch.Tensor,
        feat: List[torch.Tensor],
        sys: List[torch.Tensor],
    ) -> torch.Tensor:
        frame_vec = self.frame_branch(frame)                             # [B, d]

        feat_vecs = [
            self.feat_projs[l](feat[l].mean(dim=(1, 2)))                # [B, d]
            for l in range(len(feat))
        ]
        sys_vecs = [
            self.sys_projs[l](sys[l].mean(dim=(1, 2)))                  # [B, d]
            for l in range(len(sys))
        ]

        x = torch.cat([frame_vec] + feat_vecs + sys_vecs, dim=-1)       # [B, fuse_dim]
        return self.head(x)                                              # [B, 1]
