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

    # --- dynamic quadtree ---
    # Set False to fall back to the flat fixed-grid HFM.
    use_quadtree: bool = True
    qt_base_grid: int = 4        # root tiling is qt_base_grid² cells at depth 0
    qt_rounds: int = 3           # refinement rounds (max depth); overridable at inference
    qt_split_k: int = 24         # leaves split per round → +3k tokens per round
    qt_sample_px: int = 8        # every cell is resampled to this crop, at any depth
    qt_out_px: int = 8           # pixels each leaf predicts, in its own local frame
    qt_rope_octaves: float = 6.0 # octaves spanned by the 2-D RoPE frequency band
    qt_scale_freqs: int = 6      # Fourier features on relative depth
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
