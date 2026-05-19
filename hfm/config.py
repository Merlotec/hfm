from dataclasses import dataclass
from typing import Tuple


@dataclass
class HFMConfig:
    # --- image / patch ---
    img_size: int = 256
    in_channels: int = 4
    patch_px: int = 4            # each patch covers patch_px × patch_px pixels

    # --- patch / residual token dims (low) ---
    d_patch: int = 64
    d_resid: int = 64

    # --- feature hierarchy ---
    # Spatial sizes from coarse→fine: 32, 64, 128, 256
    feat_sizes: Tuple[int, ...] = (32, 64, 128, 256)
    # Embedding dims at each level (coarse→fine, decreasing)
    d_feat: Tuple[int, ...] = (512, 256, 128, 64)
    # System embedding dims (same levels, lower dim)
    d_sys: Tuple[int, ...] = (256, 128, 64, 32)

    # --- attention ---
    patch_local_radius: int = 3   # patches attend to (2r+1)^2 neighbourhood
    feat_window_size: int = 8     # window size for local self-attn on fine levels
    n_heads_patch: int = 4
    n_heads_feat: Tuple[int, ...] = (8, 4, 2, 1)  # per level
    n_heads_sys: Tuple[int, ...] = (4, 2, 1, 1)
    n_heads_resid: int = 4

    # --- transformer depth ---
    n_layers: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    # --- training loop ---
    n_warmup_frames: int = 5     # frames used to build up system embeddings

    # --- memory ---
    gradient_checkpointing: bool = False   # recompute activations during backward

    @property
    def n_patch(self) -> int:
        return self.img_size // self.patch_px   # patches per side (64)

    @property
    def n_levels(self) -> int:
        return len(self.feat_sizes)
