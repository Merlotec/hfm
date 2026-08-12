"""
Train the conditional flow-matching refiner against a FROZEN HFM base.

Two-stage by design: the base model + context encoder are loaded from an
existing checkpoint, frozen, and used only to generate (pred, target) pairs
on the fly.  The refiner learns to sample the detail residual d = target - pred
(see hfm/refiner.py for the objective).  Single-process, single-device — the
refiner is ~8.6M params and the frozen forwards are no-grad.

    python scripts/train_refiner.py --checkpoint checkpoints/ckpt-step309000.ckpt
    python scripts/train_refiner.py --checkpoint ... --resume checkpoints/refiner_step004000.pt

A refiner is only valid for the base it was trained against; the base path,
step, and normalisation stats are recorded in every refiner checkpoint and
verified at inference (infer.py --refine).
"""

import argparse
import copy
import csv
import json
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hfm import build_model, ContextEncoder
from hfm.data import FVMDataModule
from hfm.refiner import (EMA, RefinerConfig, RefinerUNet, cfm_loss,
                         sample_detail)

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = _ROOT.parent / 'data' / 'fvm_gen_datasets'
CKPT_DIR = _ROOT / 'checkpoints'


# ---------------------------------------------------------------------------
# Frozen base loading
# ---------------------------------------------------------------------------

def load_base(path: str, device: torch.device, allow_missing_stats: bool):
    """Load and freeze the base HFM + ContextEncoder from a .pt or .ckpt.

    Both GANTrainer .pt and Lightning .ckpt checkpoints carry native
    model/context_encoder/cfg keys (on_save_checkpoint injects them); fall back
    to prefix-stripping a raw Lightning state_dict just in case."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ckpt['cfg']

    if 'model' in ckpt and 'context_encoder' in ckpt:
        sd_model, sd_ctx = ckpt['model'], ckpt['context_encoder']
    else:
        sd = ckpt['state_dict']
        sd_model = {k[len('model.'):]: v for k, v in sd.items()
                    if k.startswith('model.')}
        sd_ctx = {k[len('context_encoder.'):]: v for k, v in sd.items()
                  if k.startswith('context_encoder.')}

    model = build_model(cfg)
    model.load_state_dict(sd_model)
    ce = ContextEncoder(cfg)
    ce.load_state_dict(sd_ctx)
    # eval() also matters for correctness: the base injects input noise only in
    # train mode, and the refiner must see exactly the inference-time base.
    model.eval().requires_grad_(False)
    ce.eval().requires_grad_(False)

    mean, std = ckpt.get('norm_mean'), ckpt.get('norm_std')
    if (mean is None or std is None) and not allow_missing_stats:
        raise SystemExit(
            'Base checkpoint carries no normalisation stats — the dataset-dir '
            'fallback could silently mismatch base training and poison sigma_d. '
            'Pass --allow-missing-stats to override.')

    return (model.to(device), ce.to(device), cfg, ckpt,
            None if mean is None else torch.tensor(mean),
            None if std is None else torch.tensor(std))


# ---------------------------------------------------------------------------
# Pair generation (mirrors GANTrainer.step's stride subsampling)
# ---------------------------------------------------------------------------

@torch.no_grad()
def make_pair(frames, mesh_ids, mesh_masks, model, ce, cfg, device,
              amp, rollout_mix_prob: float, rollout_max_k: int,
              force_k: Optional[int] = None, force_s: Optional[int] = None):
    """
    frames: [B, T, C, H, W] on device.  Samples a stride s and a rollout depth
    k, runs the frozen base k teacher-forced/AR steps, and returns
    (x_in, pred, target, mask, ctx_vec, s, k) where x_in is the input of the
    LAST base step (rollout-drifted for k > 1, matching inference).
    """
    nc = cfg.n_context_frames
    strides = tuple(getattr(cfg, 'time_strides', (1,)) or (1,))
    T = frames.shape[1]

    if force_k is not None:
        k = force_k
    elif rollout_max_k > 1 and float(torch.rand(())) < rollout_mix_prob:
        k = int(torch.randint(2, rollout_max_k + 1, (1,)))
    else:
        k = 1

    # force_s pins the stride so every rank does the SAME amount of frozen-base
    # work per step; otherwise ranks with a deeper rollout become stragglers that
    # the whole allreduce waits on.
    s = force_s if force_s is not None else strides[int(torch.randint(len(strides), (1,)))]
    need = (nc + k) * s + 1
    if need > T:                       # window too short for this (s, k)
        s = strides[0]
        need = (nc + k) * s + 1
        k = min(k, (T - 1) // s - nc)
    off = int(torch.randint(T - need + 1, (1,)))
    sub = frames[:, off: off + need: s]          # [B, nc+k+1, C, H, W]

    mask = (mesh_masks[mesh_ids.to(mesh_masks.device).long()]
            if (mesh_ids is not None and mesh_masks is not None)
            else torch.ones(1, 1, frames.shape[3], frames.shape[4], device=device))

    ctx_frames = [sub[:, t] for t in range(nc)]
    with amp():
        context = ce(ctx_frames, pixel_mask=mask)
    ctx_vec = context.float()               # ALL tokens [B, K, d_ctx]

    x_cur = sub[:, nc] * mask
    x_in = x_cur
    with amp():
        for _ in range(k):
            x_in = x_cur
            pred = model(x_cur, context, pixel_mask=mask)
            pred = pred.float() * mask
            x_cur = pred
    target = sub[:, nc + k] * mask
    return x_in, pred, target, mask, ctx_vec, s, k


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Declared up front: Python forbids using a name before its `global`
    # statement in the same scope, and --ckpt-dir's help text prints the default.
    # Rebinding here makes every later CKPT_DIR use follow with no other edits.
    global CKPT_DIR
    p = argparse.ArgumentParser(description='Train the flow-matching refiner')
    p.add_argument('--checkpoint', type=str, required=True,
                   help='Frozen base HFM checkpoint (.pt or .ckpt)')
    p.add_argument('--data', type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--max-steps', type=int, default=0, help='0 = unlimited')
    p.add_argument('--cosine-t-max', type=int, default=30_000)
    p.add_argument('--sigma-calib-batches', type=int, default=200)
    p.add_argument('--rollout-mix-prob', type=float, default=0.25)
    p.add_argument('--rollout-max-k', type=int, default=4)
    p.add_argument('--ema-decay', type=float, default=0.999)
    p.add_argument('--base-width', type=int, default=64)
    p.add_argument('--blocks-per-level', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--eval-every', type=int, default=1000)
    p.add_argument('--ckpt-every', type=int, default=1000)
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--allow-missing-stats', action='store_true')
    p.add_argument('--cache-frames', action='store_true',
                   help='Pre-render all frames in memory (small overfit runs)')
    p.add_argument('--ckpt-dir', type=str, default=None,
                   help=f'Directory for refiner checkpoints + loss log '
                        f'(default: {CKPT_DIR}). --resume paths are given '
                        f'explicitly, so this only affects where NEW ones land.')
    args = p.parse_args()

    if args.ckpt_dir:
        CKPT_DIR = Path(args.ckpt_dir).resolve()
    print(f'Checkpoints: {CKPT_DIR}')

    from hfm.distributed import (init_distributed, allreduce_grads,
                                 allreduce_stats, is_main, barrier, cleanup)
    # Single-process behaviour is unchanged: with world==1 every collective below
    # is a documented no-op, so this stays the same script it was.
    rank, world, local, device = init_distributed()
    main = is_main()

    def bcast0(t: torch.Tensor) -> torch.Tensor:
        """Broadcast a tensor from rank 0 via the HOST.

        Dawn has no working device collectives (oneCCL ZE + native XCCL both
        fault on its driver), so the process group is gloo over host memory --
        every collective has to be staged through the CPU, exactly as
        allreduce_grads does for gradients."""
        if world == 1:
            return t
        h = t.detach().to('cpu')
        torch.distributed.broadcast(h, 0)
        return h.to(t.device, t.dtype)

    amp_enabled = device.type in ('cuda', 'xpu')
    amp = (lambda: torch.autocast(device_type=device.type, dtype=torch.bfloat16)) \
        if amp_enabled else nullcontext
    if main:
        print(f'Device: {device}  (autocast bf16: {amp_enabled})  '
              f'world={world}' + (f' [rank {rank}]' if world > 1 else ''))

    # ---- frozen base ----
    print(f'Loading frozen base: {args.checkpoint}')
    model, ce, cfg, base_ckpt, mean, std = load_base(
        args.checkpoint, device, args.allow_missing_stats)
    base_step = base_ckpt.get('global_step', '?')
    print(f'  base global_step={base_step}  use_quadtree={cfg.use_quadtree}  '
          f'residual={getattr(cfg, "residual_prediction", "?")}')

    # ---- data (same window formula as the base trainer) ----
    nc = cfg.n_context_frames
    max_stride = max(getattr(cfg, 'time_strides', (1,)) or (1,))
    seq_len = (nc + args.rollout_max_k) * max_stride + 1
    dm = FVMDataModule(
        data_dir=args.data, seq_len=seq_len, batch_size=args.batch_size,
        num_workers=args.num_workers, mean=mean, std=std,
        return_mesh_id=True, cache_frames=args.cache_frames,
    )
    dm.setup()
    mesh_masks = dm.mesh_masks.to(device) if dm.mesh_masks is not None else None

    # ---- refiner ----
    rcfg = RefinerConfig(
        in_channels=cfg.in_channels,
        cond_channels=3 * cfg.in_channels + 1,
        base_width=args.base_width,
        blocks_per_level=args.blocks_per_level,
        dropout=args.dropout,
        d_ctx=cfg.d_ctx,
        # Full context conditioning: the refiner's FiLM embedding sees ALL
        # summary tokens (flattened in slot order), not the mean-pooled vector.
        n_ctx_tokens=cfg.n_ctx_tokens,
    )
    refiner = RefinerUNet(rcfg).to(device)
    # Torch seeds its default generator from OS entropy per process, so every rank
    # would otherwise start from DIFFERENT random weights and gradient averaging
    # would blend distinct models.  The DDP wrapper does this for the base
    # trainer; the manual host-staged path has to do it explicitly.
    if world > 1:
        with torch.no_grad():
            for q in refiner.parameters():
                q.copy_(bcast0(q))
            for b in refiner.buffers():
                b.copy_(bcast0(b))
    if main:
        print(f'Refiner: {sum(q.numel() for q in refiner.parameters()) / 1e6:.2f}M params')
    opt = torch.optim.AdamW(refiner.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.cosine_t_max)
    ema = EMA(refiner, args.ema_decay)
    eval_net = copy.deepcopy(refiner)     # receives EMA weights at eval time
    global_step = 0
    sigma_d = None

    if args.resume:
        rck = torch.load(args.resume, map_location='cpu', weights_only=False)
        refiner.load_state_dict(rck['refiner'])
        ema.load_state_dict(rck['refiner_ema'])
        opt.load_state_dict(rck['optimizer'])
        sched.load_state_dict(rck['scheduler'])
        global_step = rck['global_step']
        sigma_d = rck['sigma_d'].to(device)   # NEVER recalibrate on resume
        if main:
            print(f'Resumed {args.resume} at step {global_step}')

    def batch_iter():
        epoch = 0
        while True:
            dl = dm.train_dataloader()
            # DistributedSampler shards deterministically from its epoch seed, so
            # without set_epoch every pass would replay the SAME order per rank.
            if world > 1 and getattr(dl, 'sampler', None) is not None \
                    and hasattr(dl.sampler, 'set_epoch'):
                dl.sampler.set_epoch(epoch)
            epoch += 1
            for batch in dl:
                mesh_b = None
                if isinstance(batch, (tuple, list)):
                    mesh_b = batch[1] if len(batch) > 1 else None
                    batch = batch[0]
                yield batch.to(device), mesh_b

    it = batch_iter()

    # ---- sigma_d calibration (teacher-forced k=1 only; fixed for the model's life) ----
    if sigma_d is None:
        if main:
            print(f'Calibrating sigma_d over {args.sigma_calib_batches} batches '
                  f'(data-loader bound: expect minutes, GPU mostly idle)...',
                  flush=True)
        ssum = torch.zeros(cfg.in_channels, device=device)
        cnt = torch.zeros(cfg.in_channels, device=device)
        for _i in range(args.sigma_calib_batches):
            if main and _i and _i % 20 == 0:
                print(f'  calib {_i}/{args.sigma_calib_batches}', flush=True)
            frames, mesh_b = next(it)
            x_in, pred, target, mask, _, s, _ = make_pair(
                frames, mesh_b, mesh_masks, model, ce, cfg, device, amp,
                0.0, args.rollout_max_k, force_k=1)
            d = (target - pred) * mask
            ssum += d.pow(2).sum(dim=(0, 2, 3))
            cnt += mask.expand_as(d).sum(dim=(0, 2, 3))
        # Each rank calibrated over its OWN shard; sigma_d is fixed for the
        # model's life, so the ranks must agree on one value or they would train
        # against different objectives.  Sum both accumulators before dividing.
        if world > 1:
            tot = allreduce_stats(*[float(v) for v in ssum],
                                  *[float(v) for v in cnt])
            n = cfg.in_channels
            ssum = torch.tensor(tot[:n], device=device)
            cnt = torch.tensor(tot[n:], device=device)
        sigma_d = (ssum / cnt.clamp_min(1.0)).sqrt().clamp_min(1e-6)
        if main:
            print(f'  sigma_d = {[round(float(v), 5) for v in sigma_d]}')

    # ---- logging ----
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = CKPT_DIR / 'refiner_loss_log.csv'
    if main and (not log_path.exists() or not args.resume):
        with open(log_path, 'w', newline='') as f:
            csv.writer(f).writerow(['step', 'loss', 's', 'k', 'lr'])

    def save(step):
        # Ranks hold identical weights (broadcast init + averaged grads), so one
        # writer avoids 8 processes racing on the same path.
        if not main:
            return
        torch.save({
            'refiner': refiner.state_dict(),
            'refiner_ema': ema.state_dict(),
            'refiner_cfg': rcfg,
            'sigma_d': sigma_d.cpu(),
            'base_checkpoint': str(args.checkpoint),
            'base_global_step': base_step,
            'base_cfg': cfg,
            'norm_mean': None if mean is None else mean.tolist(),
            'norm_std': None if std is None else std.tolist(),
            'optimizer': opt.state_dict(),
            'scheduler': sched.state_dict(),
            'global_step': step,
            'train_args': vars(args),
        }, CKPT_DIR / f'refiner_step{step:06d}.pt')
        print(f'  [ckpt] refiner_step{step:06d}.pt')

    # ---- training loop ----
    if main:
        print(f'Training (seq_len={seq_len}, batch={args.batch_size})'
              + (f'  x{world} ranks -> global batch '
                 f'{args.batch_size * world}' if world > 1 else '') + '...')
    steps_per_epoch = max(1, len(dm.train_dataloader()))
    total = args.max_steps or args.epochs * steps_per_epoch
    t0 = time.time()
    run_loss, run_n = 0.0, 0
    while global_step < total:
        frames, mesh_b = next(it)
        # Draw (s, k) ONCE on rank 0 and share it: differing rollout depth would
        # mean differing frozen-base work per rank, so every step would wait on
        # the deepest one.
        f_s = f_k = None
        if world > 1:
            sel = torch.tensor([0.0, 0.0])
            if main:
                strides = tuple(getattr(cfg, 'time_strides', (1,)) or (1,))
                kk = (int(torch.randint(2, args.rollout_max_k + 1, (1,)))
                      if args.rollout_max_k > 1
                      and float(torch.rand(())) < args.rollout_mix_prob else 1)
                sel = torch.tensor([float(strides[int(torch.randint(len(strides), (1,)))]),
                                    float(kk)])
            sel = bcast0(sel)
            f_s, f_k = int(sel[0].item()), int(sel[1].item())
        x_in, pred, target, mask, ctx_vec, s, k = make_pair(
            frames, mesh_b, mesh_masks, model, ce, cfg, device, amp,
            args.rollout_mix_prob, args.rollout_max_k,
            force_k=f_k, force_s=f_s)

        with amp():
            loss = cfm_loss(refiner, target, pred, x_in, mask, ctx_vec, sigma_d)
        # The skip MUST be unanimous: a rank that `continue`s never reaches the
        # allreduce below while the others block inside it, and the job hangs
        # until the wall-clock limit rather than failing.
        bad = 0.0 if torch.isfinite(loss) else 1.0
        if world > 1:
            bad = allreduce_stats(bad)[0]
        if bad:
            if main:
                print(f'  [warn] non-finite loss at step {global_step} '
                      f'(on {int(bad)}/{world} rank(s)); skipping')
            opt.zero_grad()
            continue
        opt.zero_grad()
        loss.backward()
        allreduce_grads([refiner])          # no-op when world == 1
        torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
        opt.step()
        sched.step()
        ema.update(refiner)
        global_step += 1
        run_loss += loss.item(); run_n += 1

        if global_step % args.log_every == 0:
            # Report the loss over the GLOBAL batch, not rank 0's shard, so the
            # curve is comparable across world sizes.
            tot_loss, tot_n = run_loss, run_n
            if world > 1:
                tot_loss, tot_n = allreduce_stats(run_loss, float(run_n))
            avg = tot_loss / max(1.0, tot_n)
            rate = run_n / (time.time() - t0)          # per-rank step rate
            if main:
                print(f'step {global_step:6d} | loss={avg:.4f}  s={s} k={k}  '
                      f'lr={sched.get_last_lr()[0]:.2e}  {rate:.2f} it/s'
                      + (f' x{world} ranks' if world > 1 else ''), flush=True)
                with open(log_path, 'a', newline='') as f:
                    csv.writer(f).writerow([global_step, f'{avg:.5f}', s, k,
                                            f'{sched.get_last_lr()[0]:.3e}'])
            run_loss, run_n, t0 = 0.0, 0, time.time()

        if args.eval_every > 0 and global_step % args.eval_every == 0:
            ema.copy_to(eval_net)
            eval_net.eval()
            with torch.no_grad():
                d_hat = sample_detail(eval_net, pred, x_in, mask, ctx_vec,
                                      sigma_d, n_steps=6)
                refined = (pred + d_hat) * mask
                w = mask.expand_as(pred)
                mse_p = ((pred - target).pow(2) * w).sum() / w.sum()
                mse_r = ((refined - target).pow(2) * w).sum() / w.sum()
                # high-pass energy: how much fine structure did we add back?
                def hf(x):
                    X = torch.fft.rfft2(x * mask)
                    return X.abs().pow(2)[..., 32:].sum()
                if main:
                    print(f'  [eval] mse pred={mse_p:.5f} refined={mse_r:.5f} '
                          f'(healthy: refined slightly worse)  '
                          f'HF energy pred/gt={hf(pred) / hf(target):.3f} '
                          f'refined/gt={hf(refined) / hf(target):.3f}')

        if args.ckpt_every > 0 and global_step % args.ckpt_every == 0:
            save(global_step)

    save(global_step)
    barrier()                 # rank 0 must finish writing before anyone exits
    if main:
        print('Done.')
    cleanup()


if __name__ == '__main__':
    main()
