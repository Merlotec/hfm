"""
GAN discriminator for HFM.

Conditional discriminator: given the previous frame and the context tokens
from ContextEncoder (which encode the PDE dynamics), classify whether the
next frame is real or generated.
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
    frame   : [B, C, H, W]       frame being evaluated (model output or ground truth)
    x_prev  : [B, C, H, W]       ground-truth previous frame
    context : [B, K, d_ctx]      context tokens from ContextEncoder

    Output
    ------
    [B, 1]  un-normalised real/fake logit (use BCEWithLogitsLoss).
    """

    def __init__(self, cfg: HFMConfig):
        super().__init__()
        d = cfg.disc_dim

        # Image branch: x_prev and frame concatenated along channels
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

        self.conv      = nn.Sequential(*layers)
        self.pool      = nn.AdaptiveAvgPool2d(1)
        self.conv_proj = spectral_norm(nn.Linear(512, d))

        # Context branch: mean-pool K tokens then project
        self.ctx_proj = spectral_norm(nn.Linear(cfg.d_ctx, d))

        self.head = nn.Sequential(
            spectral_norm(nn.Linear(d * 2, d * 2)),
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
        context: torch.Tensor,
    ) -> torch.Tensor:
        # Image branch
        x = torch.cat([x_prev, frame], dim=1)              # [B, 2C, H, W]
        conv_vec = self.conv_proj(
            self.pool(self.conv(x)).flatten(1)              # [B, 512] → [B, d]
        )

        # Context branch: mean over K tokens
        ctx_vec = self.ctx_proj(context.mean(dim=1))        # [B, d]

        x = torch.cat([conv_vec, ctx_vec], dim=-1)          # [B, 2d]
        return self.head(x)                                  # [B, 1]
