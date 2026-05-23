from dataclasses import dataclass


@dataclass
class HFMConfig:
    # --- image / patch ---
    img_size: int = 256
    in_channels: int = 4
    patch_px: int = 16

    # --- patch token dim ---
    d_patch: int = 256

    # --- skip encoder (for overlapping decoder) ---
    skip_ch: int = 32

    # --- global capacity tokens (appended to patch tokens every layer) ---
    n_global_tokens: int = 16

    # --- attention ---
    n_heads: int = 8

    # --- input noise (training only) ---
    noise_std: float = 0.05

    # --- transformer depth ---
    n_layers: int = 9
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    # --- context encoder ---
    n_context_frames: int = 5
    ctx_patch_px: int = 16
    d_ctx: int = 256
    n_ctx_tokens: int = 64
    n_ctx_layers: int = 4
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
