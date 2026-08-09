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

    # --- multi-timestep training ---
    # Temporal strides sampled per training step.  With stride s the context is
    # built from every s-th frame and the model predicts s frames ahead — the
    # timestep is parameterised purely by the context (the encoder sees how far
    # the flow moves between its input frames), never as an explicit input.
    # Needs seq_len = (n_context_frames + rollout_horizon) * max(stride) + 1.
    # (1,) by default: the timestep axis is now carried by save_t, which the
    # solver draws per segment over 0.01..0.2, so strides are no longer needed to
    # produce a spread of dt and would only shrink the usable windows per 30-frame
    # segment (stride 2 needs 19 frames instead of 10, i.e. 12 windows not 21).
    # The machinery is kept because it is the only way to reach dt above
    # max(save_t), and the only source of same-state/different-dt pairs; setting
    # e.g. (1, 2) here switches it back on with no code change.
    time_strides: tuple = (1,)
    # Weight of the auxiliary stride-classification loss on the context tokens.
    # A small probe must recover the sampled stride from the context alone —
    # direct evidence (and gradient pressure) that the context actually encodes
    # the timestep rather than collapsing to a constant.  0.5: at 0.1 the probe
    # never learned stride 4 (sacc(s=4) pinned at ~0) and the model hedged with
    # an average-magnitude delta.
    stride_cls_weight: float = 0.5

    # Normalise each rollout loss term by the persistence baseline for the same
    # target (no-grad).  The raw loss scales with the delta size, so stride-4
    # batches out-gradient stride-1 batches ~4:1 and the fine small-stride task
    # is starved; normalised, every stride contributes O(1) gradient and the
    # objective literally becomes "beat persistence".
    persist_norm_loss: bool = True

    # --- context encoder ---
    # Feed each context frame together with its difference from the previous
    # frame ([f_t, f_t - f_{t-1}]).  The encoder embeds frames independently, so
    # without this, inter-frame motion — the ONLY carrier of the timestep —
    # exists just as pattern-entangled differences between token embeddings,
    # which no probe (and, in practice, not the trunk either) can linearise.
    # With explicit diffs, stride classification goes 0.33 -> 0.98 in isolation.
    ctx_temporal_diffs: bool = True
    n_context_frames: int = 5
    ctx_patch_px: int = 16
    d_ctx: int = 256
    n_ctx_tokens: int = 16
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
    # Level 0 is the ONLY level the decoder reads, so it must be wide enough to carry
    # both its own 4x4 patch encoding and everything routed down from the coarse
    # levels.  At dim 64 (= the raw 4x4x4 patch size) it had zero headroom and was the
    # model's output bottleneck; 192 gives the decoded level real capacity.
    ml_dims: tuple = (192, 384, 256, 96)
    # Per-level self-attn pass budget, non-increasing (pyramid).  The last entry is
    # the "number of layers" = how many times level `ml_levels-1` iterates.  Coarse
    # budgets exhaust first, so late passes touch only fine levels (pyramidal tail).
    ml_passes: tuple = (8, 6, 5, 4)
    # Weight tying across a level's repeated passes.  By default each level reuses ONE
    # block for all its passes (parameter-efficient iterative refinement).  Set an entry
    # >1 to give that level that many DISTINCT blocks, cycled round-robin across its
    # passes (e.g. 2 blocks over 6 passes → block order 0,1,0,1,0,1).  Each entry is
    # clamped to [1, ml_passes[l]].
    ml_blocks_per_level: tuple = (1, 1, 1, 1)
    # Convenience flag: fully untie — one distinct block per pass.  Overrides
    # ml_blocks_per_level (sets it to ml_passes).  Multiplies self-attn parameters.
    ml_untie_passes: bool = False

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
