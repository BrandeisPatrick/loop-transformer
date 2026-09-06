"""Import this BEFORE `from lotus import Lotus`.

scripts/lotus.py and scripts/eval.py call torch.cuda.synchronize() and the
torch.cuda memory-stat APIs unconditionally (lotus.py lines 532/553/1174 inside
forward(), 1540/1605 inside generate(); eval.py calls reset_peak_memory_stats /
max_memory_allocated / max_memory_reserved).  On a Mac these raise
"Torch not compiled with CUDA enabled".  We reroute them to the MPS equivalents
so the author's own latency numbers stay meaningful.
"""
import torch

def _pick_device(prefer="mps"):
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def patch_cuda_calls(device):
    is_mps = (getattr(device, "type", str(device)) == "mps")

    def _sync(*a, **k):
        if is_mps:
            torch.mps.synchronize()

    torch.cuda.synchronize = _sync
    torch.cuda.reset_peak_memory_stats = lambda *a, **k: None
    # torch.mps reports driver allocations; good enough for a peak-RSS proxy.
    torch.cuda.max_memory_allocated = (
        (lambda *a, **k: torch.mps.current_allocated_memory()) if is_mps else (lambda *a, **k: 0)
    )
    torch.cuda.max_memory_reserved = (
        (lambda *a, **k: torch.mps.driver_allocated_memory()) if is_mps else (lambda *a, **k: 0)
    )
