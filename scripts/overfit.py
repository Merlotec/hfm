"""
Overfit sanity check for HFM.

Two stages:
  1. Single-batch overfit  — fix one sequence of frames, train until loss → 0.
     Verifies the model can memorise and that gradients are healthy.

  2. Residual overfit      — same batch but with the residual path enabled.
     Verifies that the system embedding grows and that the residual cross-attn
     actually helps (loss should drop faster / lower than stage 1).

Run from the hfm/ root:
    python scripts/overfit.py
"""

import sys
import json
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn as nn

# Make hfm importable when running from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hfm import HFMConfig, HFM
from hfm.trainer import warmup_system, FluidLoss
from hfm.data import FVMDataModule, build_renderer, FVMSequenceDataset, load_pixel_mask

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_ROOT   = Path(__file__).resolve().parents[2] / 'fvm_model' / 'data'
OVERFIT_DIR = DATA_ROOT / 'fvm_gen_overfit'
FULL_DIR    = DATA_ROOT / 'fvm_gen_datasets-bk'

# ---------------------------------------------------------------------------
# Config — smaller than production to run quickly on CPU/MPS
# ---------------------------------------------------------------------------

CFG = HFMConfig(
    img_size       = 256,
    in_channels    = 4,
    patch_px       = 4,
    d_patch        = 32,
    d_resid        = 32,
    # 3 levels only — skip 256×256 to keep activation memory tractable on MPS
    feat_sizes     = (32, 64, 128),
    d_feat         = (128, 64, 32),
    d_sys          = (64, 32, 16),
    n_heads_patch  = 2,
    n_heads_feat   = (4, 2, 1),
    n_heads_sys    = (2, 1, 1),
    n_heads_resid  = 2,
    feat_window_size = 8,
    n_layers       = 2,
    mlp_ratio      = 4.0,
    dropout        = 0.0,
    n_warmup_frames = 3,
)

N_WARMUP    = CFG.n_warmup_frames
SEQ_LEN     = N_WARMUP + 2        # warmup frames + prediction input + target
FIRST_FRAME = 20
LR          = 1e-4
N_STEPS_1   = 2000                # stage 1: no residual
N_STEPS_2   = 2000                # stage 2: with residual
LOG_EVERY   = 10
CKPT_EVERY  = 500                 # save a checkpoint every N steps (each stage)
CKPT_DIR    = Path(__file__).resolve().parents[1] / 'checkpoints'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_fixed_batch(device: torch.device) -> tuple[list[torch.Tensor], torch.Tensor]:
    """
    Render and cache all frames from the overfit run, then pick a fixed
    window of SEQ_LEN consecutive frames as the single training batch.
    Returns (frames, pixel_mask) where frames is a list of SEQ_LEN tensors
    each [1, C, H, W] and pixel_mask is (1, 1, H, W) bool on device.
    """
    print(f'Loading overfit data from {OVERFIT_DIR}')

    renderer = build_renderer(OVERFIT_DIR, (256, 256), device='cpu')

    # Load or compute normalisation stats from the overfit run
    stats_path = OVERFIT_DIR / 'hfm_input_stats.json'
    if stats_path.exists():
        with open(stats_path) as f:
            s = json.load(f)
        mean = torch.tensor(s['mean'])
        std  = torch.tensor(s['std'])
        print('Loaded normalisation stats from cache')
    else:
        # Fall back to fvm_foundation stats if available, else compute
        foundation_stats = Path(__file__).resolve().parents[2] / \
                           'fvm_model' / 'fvm_foundation' / 'input_stats.json'
        if foundation_stats.exists():
            with open(foundation_stats) as f:
                s = json.load(f)
            mean = torch.tensor(s['mean'])
            std  = torch.tensor(s['std'])
            print('Using fvm_foundation normalisation stats')
        else:
            from hfm.data import compute_normalisation_stats
            sim_dirs = [p for p in OVERFIT_DIR.iterdir() if p.is_dir()]
            mean, std = compute_normalisation_stats(sim_dirs, renderer,
                                                     first_frame=FIRST_FRAME)
            with open(stats_path, 'w') as f:
                json.dump({'mean': mean.tolist(), 'std': std.tolist()}, f)
            print('Computed and saved normalisation stats')

    sim_dir = sorted([p for p in OVERFIT_DIR.iterdir() if p.is_dir()])[0]
    ds = FVMSequenceDataset.with_cache(
        sim_dir, renderer, SEQ_LEN, mean, std, first_frame=FIRST_FRAME
    )
    print(f'  Run: {sim_dir.name}  —  {len(ds)} sequences available')

    pixel_mask = load_pixel_mask(OVERFIT_DIR, renderer, (256, 256)).to(device)

    # Use the middle of the sequence for stability
    mid = len(ds) // 2
    seq = ds[mid]                               # [T, C, H, W]
    frames = [seq[t:t+1].to(device) * pixel_mask for t in range(SEQ_LEN)]
    return frames, pixel_mask


def sys_emb_norm(sys: list[torch.Tensor]) -> float:
    return sum(s.norm().item() for s in sys) / len(sys)


def save_checkpoint(model: HFM, opt: torch.optim.Optimizer,
                    stage: str, step: int, loss: float):
    CKPT_DIR.mkdir(exist_ok=True)
    path = CKPT_DIR / f'overfit_{stage}_step{step:05d}.pt'
    torch.save({
        'stage':            stage,
        'step':             step,
        'loss':             loss,
        'cfg':              CFG,
        'model_state':      model.state_dict(),
        'optimizer_state':  opt.state_dict(),
    }, path)
    print(f'  [ckpt] saved → {path.name}')


# ---------------------------------------------------------------------------
# Stage 1: overfit without residual
# ---------------------------------------------------------------------------

def stage1(model: HFM, frames: list[torch.Tensor], pixel_mask: torch.Tensor,
           criterion: nn.Module):
    print('\n' + '='*60)
    print('STAGE 1 — overfit without residual path')
    print('='*60)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    x_target = frames[N_WARMUP + 1]
    loss = torch.tensor(float('nan'))

    for step in range(1, N_STEPS_1 + 1):
        opt.zero_grad()

        sys = warmup_system(model, frames[:N_WARMUP + 1], n_warmup=N_WARMUP,
                            pixel_mask=pixel_mask)
        pred, _ = model(frames[N_WARMUP], sys_emb=sys, resid=None, freeze_sys=True)
        pred = pred * pixel_mask

        loss = criterion(pred, x_target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % LOG_EVERY == 0:
            snorm = sys_emb_norm(sys)
            print(f'  step {step:4d}  loss={loss.item():.5f}  sys_norm={snorm:.4f}')

        if step % CKPT_EVERY == 0 and step < N_STEPS_1:
            save_checkpoint(model, opt, 'stage1', step, loss.item())

    save_checkpoint(model, opt, 'stage1', N_STEPS_1, loss.item())
    print(f'Stage 1 final loss: {loss.item():.6f}')


# ---------------------------------------------------------------------------
# Stage 2: overfit with residual path
# ---------------------------------------------------------------------------

def stage2(model: HFM, frames: list[torch.Tensor], pixel_mask: torch.Tensor,
           criterion: nn.Module, device: torch.device):
    print('\n' + '='*60)
    print('STAGE 2 — overfit with residual path')
    print('='*60)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    x_target = frames[N_WARMUP + 1]
    loss = torch.tensor(float('nan'))

    for step in range(1, N_STEPS_2 + 1):
        opt.zero_grad()

        B = frames[0].shape[0]
        sys = model.init_sys_emb(B, device)

        for t in range(N_WARMUP):
            pred_t, sys = model(frames[t], sys_emb=sys, resid=None)
            pred_t = pred_t * pixel_mask
            err_t = (frames[t + 1] - pred_t).detach()
            _, sys = model(frames[t], sys_emb=sys, resid=err_t)

        pred, _ = model(frames[N_WARMUP], sys_emb=sys, resid=None, freeze_sys=True)
        pred = pred * pixel_mask

        loss = criterion(pred, x_target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % LOG_EVERY == 0:
            snorm = sys_emb_norm(sys)
            print(f'  step {step:4d}  loss={loss.item():.5f}  sys_norm={snorm:.4f}')

        if step % CKPT_EVERY == 0 and step < N_STEPS_2:
            save_checkpoint(model, opt, 'stage2', step, loss.item())

    save_checkpoint(model, opt, 'stage2', N_STEPS_2, loss.item())
    print(f'Stage 2 final loss: {loss.item():.6f}')


# ---------------------------------------------------------------------------
# Encoder round-trip check
# ---------------------------------------------------------------------------

def roundtrip_check(model: HFM, device: torch.device):
    print('\n' + '='*60)
    print('Encoder round-trip check: encode(decode(z)) ≈ z')
    print('='*60)
    with torch.no_grad():
        z    = torch.randn(1, CFG.d_patch, CFG.n_patch, CFG.n_patch, device=device)
        z_hw = z.permute(0, 2, 3, 1)                    # [B, P, P, d]
        err  = (model.patch_embed(model.decoder(z_hw)) - z_hw).abs()
        print(f'  mean |encode(decode(z)) - z|: {err.mean().item():.2e}')
        print(f'  max  |encode(decode(z)) - z|: {err.max().item():.2e}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    device = get_device()
    print(f'Device: {device}')

    frames, pixel_mask = load_fixed_batch(device)
    model     = HFM(CFG).to(device)
    criterion = FluidLoss(l1_weight=0.1, pixel_mask=pixel_mask).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'Model: {n_params:.1f}M parameters')

    roundtrip_check(model, device)
    stage1(model, frames, pixel_mask, criterion)
    stage2(model, frames, pixel_mask, criterion, device)

    print('\nDone. If stage 1 loss reached < 0.01 and stage 2 loss is lower,')
    print('the model and training loop are working correctly.')
