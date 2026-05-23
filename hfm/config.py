from dataclasses import dataclass
from typing import Tuple


@dataclass
class HFMConfig:
    # --- image / patch ---
    img_size: int = 256
    in_channels: int = 4
    patch_px: int = 4

    # --- patch token dim ---
    d_patch: int = 64

    # --- feature hierarchy ---
    feat_sizes: Tuple[int, ...] = (32, 64, 128, 256)
    d_feat: Tuple[int, ...] = (512, 256, 128, 64)

    # --- attention ---
    patch_local_radius: int = 3
    feat_window_size: int = 8
    n_heads_patch: int = 4
    n_heads_feat: Tuple[int, ...] = (8, 4, 2, 1)

    # --- transformer depth ---
    n_layers: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    # --- hierarchical bottleneck ---
    n_coarse_levels: int = 2
    n_fine_layers: int = 2

    # --- context encoder ---
    n_context_frames: int = 5      # frames fed to ContextEncoder per step
    ctx_patch_px: int = 16         # patch size in context encoder (coarser)
    d_ctx: int = 256               # context token dimension
    n_ctx_tokens: int = 64         # summary tokens output by context encoder
    n_ctx_layers: int = 4          # transformer layers in context encoder
    n_ctx_heads: int = 8

    # --- GAN discriminator ---
    disc_dim: int = 128
    disc_adv_weight: float = 0.02
    disc_lr: float = 1e-4

    # --- memory ---
    gradient_checkpointing: bool = False

    @property
    def n_patch(self) -> int:
        return self.img_size // self.patch_px

    @property
    def n_levels(self) -> int:
        return len(self.feat_sizes)
