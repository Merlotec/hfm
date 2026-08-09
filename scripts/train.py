"""
Full-dataset training for HFM (GANTrainer).

Two-stage curriculum, managed automatically:
  Stage 1 (steps 0 → gan_start_step):   reconstruction only
  Stage 2 (steps gan_start_step → end): GAN active
    adv_weight ramps 0 → cfg.disc_adv_weight over gan_ramp_steps steps.

Run from the hfm/ root:
    python scripts/train.py
    python scripts/train.py --data /path/to/dataset
    python scripts/train.py --resume checkpoints/train_step010000.pt
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hfm import HFMConfig, GANTrainer
from hfm.data import FVMDataModule, build_renderer, load_pixel_mask
from hfm.distributed import barrier, cleanup, init_distributed, is_main

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT      = Path(__file__).resolve().parents[1]
_DATA_ROOT = _ROOT.parent / 'data'

DEFAULT_DATA_DIR = _DATA_ROOT / 'fvm_gen_datasets'
CKPT_DIR         = _ROOT / 'checkpoints'
HYPERPARAMS      = _ROOT / 'hyperparams.json'

# ---------------------------------------------------------------------------
# GAN curriculum constants (not in hyperparams.json)
# ---------------------------------------------------------------------------

GAN_START_STEP        = 2_500   # step at which adversarial loss switches on
GAN_RAMP_STEPS        = 2_000   # adv_weight ramps 0 → disc_adv_weight over this
DISC_UPDATE_THRESHOLD = 0.5     # skip disc update when d_loss <= this
SELF_INPUT_PROB       = 0.5     # scheduled sampling: P(input = own prediction)

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def apply_distributed_env_defaults() -> None:
    """Seed the comms/DDP env vars from hyperparams.json so they are version-controlled
    defaults, not shell-only flags.  Only setdefault — anything already exported (by
    scripts/dawn_task.sh or `sbatch --export`) wins.  MUST run before init_distributed(),
    which is where the collective backend is chosen.

    Precedence: real env  >  hyperparams.json 'training'  >  code default.
    """
    try:
        with open(HYPERPARAMS) as f:
            t = json.load(f).get('training', {})
    except Exception:
        return
    if t.get('try_xccl'):
        # Native xccl device collectives instead of the host-staged gloo path.
        os.environ.setdefault('HFM_DDP_BACKEND', 'xccl')
    if t.get('grad_allreduce_bf16'):
        os.environ.setdefault('HFM_GRAD_ALLREDUCE_BF16', '1')


def load_config() -> tuple[HFMConfig, dict]:
    with open(HYPERPARAMS) as f:
        hp = json.load(f)
    m = hp['model']
    t = hp['training']
    # Fixed-quadtree (QuadtreeHFM) knobs: pass through only the keys actually present
    # so anything omitted keeps its HFMConfig default (no duplicated defaults here).
    # JSON arrays arrive as lists; HFMConfig tuple fields tolerate them (the model
    # does list(...) on each).  '_'-prefixed keys are comments and are skipped.
    ml_keys = ('use_quadtree', 'ml_levels', 'ml_finest_px', 'ml_dims', 'ml_passes',
               'ml_blocks_per_level', 'ml_untie_passes',
               'residual_prediction', 'mask_aware_decoder', 'persist_norm_loss',
               'ctx_temporal_diffs')
    ml_kwargs = {k: m[k] for k in ml_keys if k in m}
    cfg = HFMConfig(
        img_size               = m['img_size'],
        in_channels            = m['in_channels'],
        patch_px               = m['patch_px'],
        d_patch                = m['d_patch'],
        skip_ch                = m.get('skip_ch', 32),
        n_global_tokens        = m.get('n_global_tokens', 16),
        n_heads                = m.get('n_heads', 8),
        noise_std              = m.get('noise_std', 0.05),
        n_layers               = m['n_layers'],
        mlp_ratio              = m['mlp_ratio'],
        dropout                = m['dropout'],
        rollout_horizon        = m.get('rollout_horizon', 4),
        time_strides           = tuple(m.get('time_strides',
                                          HFMConfig.time_strides)),
        stride_cls_weight      = m.get('stride_cls_weight', 0.5),
        n_context_frames       = m['n_context_frames'],
        ctx_patch_px           = m['ctx_patch_px'],
        d_ctx                  = m['d_ctx'],
        n_ctx_tokens           = m['n_ctx_tokens'],
        n_ctx_layers           = m['n_ctx_layers'],
        n_ctx_heads            = m['n_ctx_heads'],
        disc_dim               = m['disc_dim'],
        disc_adv_weight        = m['disc_adv_weight'],
        disc_lr                = m['disc_lr'],
        gradient_checkpointing = m.get('gradient_checkpointing', False),
        **ml_kwargs,
    )
    return cfg, t


def get_device() -> torch.device:
    """Single-process device pick.  Multi-rank runs go through
    hfm.distributed.init_distributed() instead, which also selects XPU."""
    from hfm.distributed import pick_device
    return pick_device()


def _save_loss_plot(log_path: Path, plot_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    epochs, train_r, val_r = [], [], []
    with open(log_path) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row['epoch']))
            def _f(key):
                v = row.get(key)
                return float(v) if v and v != 'nan' else float('nan')
            # Plot the RATIOS: the raw losses are on incomparable scales.  Older
            # logs only carry train_loss/val_loss, so fall back to those.
            train_r.append(_f('train_ratio') if 'train_ratio' in row else _f('train_loss'))
            val_r.append(_f('val_ratio') if 'val_ratio' in row else _f('val_loss'))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_r, label='train', linewidth=1.5)
    if any(math.isfinite(v) for v in val_r):
        ax.plot(epochs, val_r, label='val (held-out runs)', linewidth=1.5)
    # 1.0 == "no better than copying the input", the line the model must stay under.
    ax.axhline(1.0, color='0.6', linestyle='--', linewidth=1, label='persistence')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss / persistence  (ratio)')
    ax.set_title('HFM training loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train HFM on fluid simulation data')
    parser.add_argument('--data',       type=Path, default=DEFAULT_DATA_DIR,
                        help='Dataset directory containing simulation subdirs')
    parser.add_argument('--settle-time', type=float, default=None,
                        help='Override settle_time from hyperparams.json: sim-time (s) '
                             'discarded from the front of COLD-START runs only, where '
                             'an impulsive start makes the field change ~140x faster '
                             'than developed flow. Pass 0 to keep everything.')
    parser.add_argument('--val-fraction', type=float, default=0.05,
                        help='Fraction of RUNS held out of training for validation '
                             '(0 disables). Split is deterministic per run name.')
    parser.add_argument('--resume',     type=str, default=None, nargs='?', const='latest',
                        help='Checkpoint to resume from. Omit value to pick the latest train_step checkpoint.')
    parser.add_argument('--epochs',     type=int,  default=None,
                        help='Override n_epochs from hyperparams.json')
    parser.add_argument('--log-every',  type=int,  default=50,
                        help='Print a log line every N steps')
    parser.add_argument('--ckpt-every', type=int,  default=500,
                        help='Save train_step<N>.pt every N steps (0 disables). '
                             'These are what `--resume latest` looks for.')
    args = parser.parse_args()

    # Let fp32 matmuls use TF32 tensor cores (Ampere+/CUDA, and the XPU equivalent).
    # bf16 autocast already covers most matmuls; this picks up the ones that stay fp32.
    torch.set_float32_matmul_precision('high')

    # Seed comms env from hyperparams.json BEFORE the process group is built (the
    # backend is chosen inside init_distributed).  Real env still overrides.
    apply_distributed_env_defaults()

    # Raw torch DDP (NOMAD's launcher).  Lightning is not usable on Dawn: its
    # accelerator registry has no XPU entry, so accelerator='auto' picks CPU on a
    # PVC node.  init_distributed() imports IPEX/oneCCL, selects xpu:<local_rank>
    # and brings up the 'ccl' process group.  No-op when world_size == 1.
    rank, world, local, device = init_distributed()
    if is_main():
        print(f'World size: {world}   device: {device}')

    cfg, train_hp = load_config()
    n_epochs = args.epochs or train_hp['n_epochs']
    # (context frames + rollout targets) at the LARGEST stride, + 1: each step
    # subsamples every s-th frame from this window (see GANTrainer.step).
    max_stride = max(getattr(cfg, 'time_strides', (1,)) or (1,))
    seq_len  = (cfg.n_context_frames + getattr(cfg, 'rollout_horizon', 1)) * max_stride + 1

    # Gradient accumulation: N micro-batches per optimizer step.  Env var overrides the
    # hyperparams.json value so it can be toggled per-job on the cluster without an edit.
    # Cuts the host-staged allreduce count N× at the cost of N× the effective batch.
    accum_steps = int(os.environ.get('HFM_ACCUM_STEPS') or train_hp.get('accum_steps', 1))

    # Complete GAN toggle: use_gan=false → no discriminator is built or applied
    # (pure reconstruction training).  Env HFM_USE_GAN=0/1 overrides hyperparams.json.
    _use_gan_env = os.environ.get('HFM_USE_GAN')
    use_gan = (_use_gan_env not in ('0', 'false', 'False')) if _use_gan_env is not None \
        else bool(train_hp.get('use_gan', True))

    # ---- training data ----
    num_workers = train_hp.get('num_workers', 8)
    # Random causal context block, unless temporal striding is on.  The two are
    # mutually exclusive: with a random context the returned tensor is two blocks
    # spliced together, and the trainer's stride subsample (frames[off::s]) would
    # cut across the splice and silently build a nonsense sequence.  Striding is
    # retired now that dt comes from save_t, so this only bites if it is turned
    # back on, and then it says so rather than corrupting the batch.
    ctx_n = cfg.n_context_frames
    if max_stride > 1:
        print(f'  [WARN] time_strides={cfg.time_strides}: random context frames '
              f'disabled (incompatible with temporal striding)')
        ctx_n = None
    dm = FVMDataModule(
        data_dir    = args.data,
        seq_len     = seq_len,
        batch_size  = train_hp['batch_size'],
        num_workers = num_workers,
        return_mesh_id = True,
        val_fraction   = args.val_fraction,
        n_context      = ctx_n,
        settle_time    = (args.settle_time if args.settle_time is not None
                          else train_hp.get('settle_time', 0.0)),
    )
    dm.setup()

    mesh_dirs = []
    if (args.data / 'shared_mesh.pkl').exists():
        mesh_dirs.append(args.data)
    else:
        for p in args.data.iterdir():
            if p.is_dir() and (p / 'shared_mesh.pkl').exists():
                mesh_dirs.append(p)
                
    if not mesh_dirs:
        raise RuntimeError(f'No shared_mesh.pkl found in {args.data} or its subdirectories')
        
    first_mdir = mesh_dirs[0]
    renderer   = build_renderer(first_mdir, (cfg.img_size, cfg.img_size))
    pixel_mask = load_pixel_mask(
        first_mdir, renderer, (cfg.img_size, cfg.img_size)
    ).to(device)

    # ---- validation: a held-out slice of the TRAINING corpus ----
    # Previously this was a separate directory (data/test) of old-solver runs, all
    # at save_t=0.01.  Training now spans save_t ~ U(0.01, 0.2) with a median near
    # 0.115, so that val set scored one extreme corner of the dt axis and its rising
    # loss could not be told apart from ordinary overfitting.  The split here is by
    # RUN and deterministic (see data.is_val_run), so every rank agrees on it and it
    # survives restarts; the held-out runs are never trained on.
    val_dl = dm.val_dataloader()
    val_pixel_mask = None
    if val_dl is not None:
        assert dm._val_dataset is not None
        print(f'Validation:      {len(dm._val_dataset)} sequences from held-out runs '
              f'({dm.val_fraction:.0%} of the corpus)\n')
    else:
        print('Validation:      disabled (val_fraction=0) — val loss will not be computed\n')

    # ---- trainer ----
    trainer = GANTrainer(
        cfg,
        lr                    = train_hp['lr'],
        weight_decay          = train_hp['weight_decay'],
        l1_weight             = train_hp['l1_weight'],
        gan_start_step        = GAN_START_STEP,
        gan_ramp_steps        = GAN_RAMP_STEPS,
        disc_update_threshold = DISC_UPDATE_THRESHOLD,
        pixel_mask            = pixel_mask,
        # Scheduled sampling MUST be off with the random context block: it rolls
        # frames[n_context-1] one step forward as a stand-in for the input, which
        # is only the preceding frame when the context is contiguous.  With
        # ctx_random the last context frame sits at an arbitrary earlier time, so
        # the branch fed a state advanced from the wrong time and supervised it
        # against a target that does not follow it -- 50% of steps had corrupted
        # input/target pairs, which caps training at a hedged, blurred optimum.
        # The horizon-4 BPTT rollout is the real exposure-bias mechanism anyway.
        self_input_prob       = (0.0 if ctx_n is not None else SELF_INPUT_PROB),
        cosine_t_max          = train_hp.get('cosine_t_max', 10_000),
        accum_steps           = accum_steps,
        use_gan               = use_gan,
        # Context-probe label spaces: geometry classes and BC-summary dimension.
        probe_bc_dim          = getattr(dm, 'bc_dim', 0),
        probe_ctx_dim         = getattr(dm, 'ctx_dim', 0),
        probe_n_visc_models   = getattr(dm, 'n_visc_models', 0),
    )

    # Per-sample masks: a batch mixes geometries, so ONE shared mask would be wrong
    # for every mesh but the first (masks differ by ~13% of the frame between meshes).
    if dm.mesh_masks is not None:
        masks = dm.mesh_masks.to(device)
        trainer.set_mesh_tables(masks)
        # The val runs are drawn from the SAME corpus, so they index the same mesh
        # table.  With a separate directory they did not, and a stale val table
        # silently scored each sample against another geometry's collider.
        trainer.set_val_mesh_tables(masks)
        print(f'Per-sample masks: {dm.mesh_masks.shape[0]} geometries (train + val)')
    # Pin the training normalisation into every checkpoint this run writes, so
    # inference can't silently normalise with different stats (see infer.load_stats).
    trainer.norm_mean, trainer.norm_std = dm.mean, dm.std
    trainer.to(device)

    if args.resume == 'latest':
        candidates = sorted(CKPT_DIR.glob('train_step*.pt'))
        if not candidates:
            print(f'No train_step checkpoints found in {CKPT_DIR}')
            sys.exit(1)
        args.resume = str(candidates[-1])
        print(f'Auto-selected checkpoint: {candidates[-1].name}')
    if args.resume:
        print(f'Resuming from {args.resume}')
        trainer.load(str(args.resume))

    # AFTER load(): checkpoints are stored unwrapped, and DDP reuses the same
    # Parameter objects so the optimizers built in GANTrainer.__init__ stay valid.
    trainer.wrap_ddp(device)

    n_gen  = sum(p.numel() for p in trainer.model.parameters())           / 1e6
    n_ctx  = sum(p.numel() for p in trainer.context_encoder.parameters()) / 1e6
    n_disc = (sum(p.numel() for p in trainer.discriminator.parameters()) / 1e6
              if trainer.discriminator is not None else 0.0)
    print(f'Generator:       {n_gen:.1f}M params')
    print(f'ContextEncoder:  {n_ctx:.1f}M params')
    print(f'Discriminator:   {n_disc:.1f}M params'
          + ('' if use_gan else '  [DISABLED: use_gan=false — reconstruction only]'))
    assert dm._dataset is not None
    # Under DDP the sampler shards the dataset, so each rank takes len/(B*world)
    # optimizer steps per epoch — forgetting `world` here made resume land on the
    # wrong epoch.
    # global_step counts OPTIMIZER steps, so divide the micro-batches/epoch by accum_steps.
    steps_per_epoch = math.ceil(
        len(dm._dataset) / (train_hp['batch_size'] * max(1, world) * accum_steps))
    start_epoch     = trainer.global_step // steps_per_epoch
    print(f'Dataset:         {len(dm._dataset)} sequences  (seq_len={seq_len})')
    print(f'Curriculum:      GAN activates at step {GAN_START_STEP}')
    if start_epoch > 0:
        print(f'Resuming:        epoch {start_epoch} (step {trainer.global_step})')
    print()

    loss_csv = loss_writer = None
    loss_log_path  = CKPT_DIR / 'loss_log.csv'
    loss_plot_path = CKPT_DIR / 'loss_plot.png'
    if is_main():                       # only rank 0 writes logs/plots/checkpoints
        CKPT_DIR.mkdir(exist_ok=True)
        # Append mode — safe for resume; write header only when starting fresh
        write_header = not loss_log_path.exists()
        loss_csv    = open(loss_log_path, 'a', newline='')
        loss_writer = csv.writer(loss_csv)
        if write_header:
            loss_writer.writerow(['epoch', 'train_ratio', 'val_ratio',
                                  'train_loss', 'val_loss', 'val_persist'])
            loss_csv.flush()

    nan_streak = 0
    # Build the loader ONCE.  Rebuilding it per-epoch would hand each epoch a fresh
    # DistributedSampler stuck at epoch=0, i.e. the identical shuffle order every
    # epoch on every rank; set_epoch below is what actually reshuffles.
    train_dl = dm.train_dataloader()

    for epoch in range(start_epoch, n_epochs):
        epoch_recon_sum = 0.0
        epoch_ratio_sum = 0.0
        epoch_recon_cnt = 0
        if hasattr(train_dl.sampler, 'set_epoch'):
            train_dl.sampler.set_epoch(epoch)       # reshuffle this rank's shard

        for batch in train_dl:            # (frames, mesh_ids, bc, save_t) / ... / frames
            mesh_b = bc_b = save_t_b = ctx_b = ctx_cls_b = None
            if isinstance(batch, (tuple, list)):
                mesh_b    = batch[1] if len(batch) > 1 else None
                bc_b      = batch[2] if len(batch) > 2 else None
                save_t_b  = batch[3] if len(batch) > 3 else None
                ctx_b     = batch[4] if len(batch) > 4 else None
                ctx_cls_b = batch[5] if len(batch) > 5 else None
                batch     = batch[0]
            frames = [batch[:, t].to(device) for t in range(batch.shape[1])]

            recon, disc = trainer.step(frames, pixel_mask=pixel_mask,
                                       mesh_ids=mesh_b, bc=bc_b, save_t=save_t_b,
                                       ctx=ctx_b, ctx_cls=ctx_cls_b)
            info = trainer.training_info()
            step = info['global_step']

            if not math.isfinite(recon):
                nan_streak += 1
                if is_main():
                    print(f'  [WARN] step {step}: NaN/inf loss (streak={nan_streak})')
                if nan_streak >= 10:
                    if is_main():
                        path = CKPT_DIR / f'train_EMERGENCY_step{step:06d}.pt'
                        trainer.save(str(path))
                        print('10 consecutive NaN steps — saved emergency checkpoint and stopping.')
                        if loss_csv is not None:
                            loss_csv.close()
                    sys.exit(1)
                continue
            nan_streak = 0

            epoch_recon_sum += recon
            # Accumulate the RATIO too, since that is what validation reports and
            # what the generator actually minimises under persist_norm_loss.  The
            # raw loss is not comparable epoch to epoch: it scales with the run's
            # dt and with how transient the sampled windows were.
            _p = info.get('persist', float('nan'))
            epoch_ratio_sum += (recon / _p) if _p > 0 else float('nan')
            epoch_recon_cnt += 1

            if is_main() and step % args.log_every == 0:
                # Persistence baseline, matched to the SAME rollout `recon` averages:
                # "predict no change" scored against every horizon target.  A one-step
                # baseline would be wrong — the field drifts further from the input each
                # step, so an EXACT persistence model (what a zero-init residual model
                # is) scores ~4.9x the 1-step loss at horizon 4.  Matched, ratio == 1
                # means "as good as persistence" and < 1 means it genuinely beats it.
                # Persistence is computed inside trainer.step at the SAMPLED stride,
                # so the ratio stays honest when the timestep varies.
                persist = info.get('persist', float('nan'))
                ratio = recon / persist if persist > 0 else float('inf')
                stride_bit = f"s={info.get('stride', 1)} dt={info.get('dt', 0):.3g}"
                # Both probe heads, reported separately: they are summed into one
                # term before the weight is applied, so a single number cannot say
                # which of the two the context is failing to carry.  dt_loss is MSE
                # on the standardised log-timestep, bc_loss is MSE on the z-scored
                # BC summary, and both are therefore O(1) at init and -> 0 when the
                # context is informative.  dt_rel is the same dt error expressed as
                # a fraction of dt, which is the interpretable one.
                probe_bit = ''
                if 'dt_mse' in info:
                    probe_bit += (f"  dt_loss={info['dt_mse']:.4f}"
                                  f" (rel {info.get('dt_rel_err', float('nan')):.2f})")
                if 'bc_mse' in info:
                    probe_bit += f"  bc_loss={info['bc_mse']:.4f}"
                # The head that matters most: the hidden per-segment physics.
                # dt and bc are both inferable from things other than the dynamics
                # (frame spacing, trajectory identity); ctx is not.
                if 'ctx_mse' in info:
                    probe_bit += f"  ctx_loss={info['ctx_mse']:.4f}"
                if 'visc_acc' in info:
                    probe_bit += f"  visc_acc={info['visc_acc']:.2f}"
                print(
                    f'epoch {epoch:3d}  step {step:6d} | '
                    f'recon={recon:.4f}  persist={persist:.4f}  ratio={ratio:.2f}  '
                    f'{stride_bit}{probe_bit}  '
                    f'disc={disc:.4f}  '
                    f'adv_w={info["adv_weight"]:.3f}'
                )

            # Step-based checkpoints.  An epoch over 18k sequences is long, so
            # epoch-only saves risk losing hours to a crash — and `--resume latest`
            # globs train_step*.pt, which nothing was writing until now.
            if is_main() and args.ckpt_every > 0 and step > 0 and step % args.ckpt_every == 0:
                path = CKPT_DIR / f'train_step{step:06d}.pt'
                trainer.save(str(path))
                print(f'  [ckpt] {path.name}  (step {step})')

        n = max(1, epoch_recon_cnt)
        train_loss  = epoch_recon_sum / n if epoch_recon_cnt else float('nan')
        train_ratio = epoch_ratio_sum / n if epoch_recon_cnt else float('nan')

        # Validation on rank 0 only: val_dl has no DistributedSampler, so every rank
        # would redundantly score the whole set.  Cheaper to do it once.
        val = {'recon': float('nan'), 'persist': float('nan'), 'ratio': float('nan')}
        if val_dl is not None and is_main():
            val = trainer.validate(val_dl, pixel_mask=val_pixel_mask)

        if is_main():
            # Ratios first: they are the comparable pair.  train_ratio vs val_ratio
            # is the generalisation gap; the raw losses beside them are on different
            # scales (different dt, different window transience) and cannot be
            # subtracted from one another.
            print(
                f'  [epoch {epoch:3d}] train_ratio={train_ratio:.3f}  '
                f'val_ratio={val["ratio"]:.3f}   '
                f'(raw: train={train_loss:.4f}  val={val["recon"]:.4f}  '
                f'val_persist={val["persist"]:.4f})'
            )
            if loss_writer is not None and loss_csv is not None:
                loss_writer.writerow([
                    epoch, f'{train_ratio:.6f}', f'{val["ratio"]:.6f}',
                    f'{train_loss:.6f}', f'{val["recon"]:.6f}',
                    f'{val["persist"]:.6f}'])
                loss_csv.flush()
            _save_loss_plot(loss_log_path, loss_plot_path)

            if (epoch + 1) % 2 == 0:
                path = CKPT_DIR / f'train_epoch{epoch:03d}.pt'
                trainer.save(str(path))
                print(f'  [ckpt] {path.name}')
        # Long timeout: rank 0 may spend many minutes validating and writing,
        # and the quadtree model costs ~2.4x a flat forward, so the default
        # 300 s collective timeout is not a safe bound here.
        barrier(long=True)   # keep ranks together while rank 0 validates / writes

    if is_main() and loss_csv is not None:
        loss_csv.close()
    cleanup()


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    main()
