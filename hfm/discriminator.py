"""
GAN discriminator for HFM.

Conditional discriminator: given the previous frame and system embeddings
(both derived from ground-truth data), classify whether the next frame is
real or generated.
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from typing import List

from .config import HFMConfig


class HFMDiscriminator(nn.Module):
    """
    Conditional discriminator for real vs. generated fluid frames.

    Inputs
    ------
    frame : [B, C, H, W]
        The frame being evaluated (model output or ground truth).
    x_prev : [B, C, H, W]
        The ground-truth previous frame (identical for real and fake calls).
    sys : list of [B, S_l, S_l, d_sys_l]
        System token hierarchy built from ground-truth warmup frames.

    Output
    ------
    [B, 1]  un-normalised real/fake logit (use BCEWithLogitsLoss).
    """

    def __init__(self, cfg: HFMConfig):
        super().__init__()
        d = cfg.disc_dim

        # x_prev and frame are concatenated along channels
        channels = [cfg.in_channels * 2, 64, 128, 256, 512]
        layers: list = []
        for i in range(len(channels) - 1):
            layers.append(
                spectral_norm(nn.Conv2d(channels[i], channels[i + 1], 4,
                                        stride=2, padding=1, bias=(i == 0)))
            )
            if i > 0:
                layers.append(nn.InstanceNorm2d(channels[i + 1], affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv_proj = spectral_norm(nn.Linear(512, d))

        # Project each sys level to d
        self.sys_projs = nn.ModuleList([
            spectral_norm(nn.Linear(d_sys, d)) for d_sys in cfg.d_sys
        ])

        fuse_dim = d * (1 + cfg.n_levels)   # conv + all sys levels

        self.head = nn.Sequential(
            spectral_norm(nn.Linear(fuse_dim, d * 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            spectral_norm(nn.Linear(d * 2, d)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(d, 1)),
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
        x_prev: torch.Tensor,
        sys: List[torch.Tensor],
    ) -> torch.Tensor:
        # Concatenate previous frame and current frame along channel dim
        x = torch.cat([x_prev, frame], dim=1)              # [B, 2C, H, W]
        conv_vec = self.conv_proj(
            self.pool(self.conv(x)).flatten(1)             # [B, 512] → [B, d]
        )

        sys_vecs = [
            self.sys_projs[l](sys[l].mean(dim=(1, 2)))    # [B, d]
            for l in range(len(sys))
        ]

        x = torch.cat([conv_vec] + sys_vecs, dim=-1)       # [B, fuse_dim]
        return self.head(x)                                 # [B, 1]
