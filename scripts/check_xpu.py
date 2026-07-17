"""Dawn environment probe — run this on a PVC node BEFORE debugging training.

    srun --partition=pvc9 --nodes=1 --ntasks=1 --time=00:05:00 \
         --account=<acct> scripts/dawn_task_probe.sh

or inside an existing allocation, after activating the venv + modules:

    python scripts/check_xpu.py

Tells you (a) whether torch can see the XPUs at all, and (b) whether Lightning
is capable of selecting them.  If (a) is false, nothing else matters.
"""
from __future__ import annotations

print('--- torch ---')
import torch
print('torch version      :', torch.__version__)

print('\n--- IPEX (registers the XPU backend on older torch/IPEX stacks) ---')
try:
    import intel_extension_for_pytorch as ipex
    print('ipex version       :', ipex.__version__)
except Exception as e:
    print('ipex IMPORT FAILED :', type(e).__name__, e)

print('\n--- XPU visibility ---')
has_xpu = hasattr(torch, 'xpu')
print('torch.xpu exists   :', has_xpu)
if has_xpu:
    try:
        print('xpu.is_available() :', torch.xpu.is_available())
        print('xpu.device_count() :', torch.xpu.device_count())
        for i in range(torch.xpu.device_count()):
            print(f'  xpu:{i} ->', torch.xpu.get_device_name(i))
    except Exception as e:
        print('xpu probe FAILED   :', type(e).__name__, e)

print('\n--- a real op on xpu:0 ---')
try:
    t = torch.ones(8, 8, device='xpu:0', dtype=torch.bfloat16)
    print('bf16 matmul on xpu :', (t @ t).sum().item())
except Exception as e:
    print('xpu op FAILED      :', type(e).__name__, e)

print('\n--- Lightning accelerator support ---')
try:
    import lightning as L
    print('lightning version  :', L.__version__)
    from lightning.pytorch.accelerators import AcceleratorRegistry
    print('registered accels  :', sorted(AcceleratorRegistry.available_accelerators()))
    print("  'xpu' supported  :", 'xpu' in AcceleratorRegistry.available_accelerators())
except Exception as e:
    print('lightning probe FAILED:', type(e).__name__, e)

print('\n--- oneCCL (needed for multi-rank DDP on XPU) ---')
for mod in ('oneccl_bindings_for_pytorch', 'torch_ccl'):
    try:
        m = __import__(mod)
        print(f'{mod:28s}: OK', getattr(m, '__version__', ''))
    except Exception as e:
        print(f'{mod:28s}: MISSING ({type(e).__name__})')
try:
    import torch.distributed as dist
    print('ccl backend available:', dist.is_available() and torch.distributed.is_ccl_available())
except Exception as e:
    print('ccl backend probe failed:', e)
