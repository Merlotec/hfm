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

    # --- rollout training ---
    # Autoregressive steps unrolled per training step (backprop-through-time).
    # >1 forces the model to propagate dynamics and stops the context collapsing
    # to a constant.  Needs seq_len = n_context_frames + 1 + rollout_horizon.
    rollout_horizon: int = 4

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

    # --- fixed multilevel quadtree ---
    # Set False to fall back to the flat fixed-grid HFM.
    use_quadtree: bool = True
    # A complete quadtree of `ml_levels` levels tiles the image.  Level 0 is the
    # finest (`ml_finest_px`×`ml_finest_px` blocks, encoded losslessly); each level
    # up doubles the block side.  For a 256px image with finest_px=4 and 4 levels:
    #   L0 4px  → 64×64,  L1 8px  → 32×32,  L2 16px → 16×16,  L3 32px → 8×8.
    ml_levels: int = 4
    ml_finest_px: int = 4                    # smallest unit side (encoded losslessly)
    # Per-level token dim — a bell curve peaking at level 1 (see model docstring).
    # Each dim must be divisible by n_heads, and dim//n_heads divisible by 4 (2-D RoPE).
    ml_dims: tuple = (64, 384, 256, 96)
    # Per-level self-attn pass budget, non-increasing (pyramid).  The last entry is
    # the "number of layers" = how many times level `ml_levels-1` iterates.  Coarse
    # budgets exhaust first, so late passes touch only fine levels (pyramidal tail).
    ml_passes: tuple = (8, 6, 5, 4)

    # --- legacy dynamic-quadtree fields (kept for config/checkpoint compat) ---
    qt_base_grid: int = 4
    qt_rounds: int = 3
    qt_split_k: int = 24
    qt_sample_px: int = 8
    qt_out_px: int = 8
    qt_rope_octaves: float = 6.0
    qt_scale_freqs: int = 6
    # Feed the geometry mask to the decoder's convs.  Set False to load
    # checkpoints written before this channel was added.
    mask_aware_decoder: bool = True

    # Predict the frame-to-frame DELTA (out = x + f(x)) instead of the absolute
    # next frame.  Consecutive frames differ by ~2%, so an absolute model must
    # reconstruct the whole field more accurately than that just to match
    # copying its input.  Set False to reproduce pre-residual checkpoints.
    residual_prediction: bool = True

    # --- memory ---
    gradient_checkpointing: bool = False

    @property
    def n_patch(self) -> int:
        return self.img_size // self.patch_px
