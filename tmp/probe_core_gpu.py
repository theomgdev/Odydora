"""Measure the shared core on real hardware: stability over a long carry,
throughput against the per-token loop, and peak memory vs window size."""
import time, torch
from odydora.core import RecurrentCore

dev = "cuda"
torch.manual_seed(0)
print(f"device: {torch.cuda.get_device_name(0)}\n")

print("=== 1. state norm over a 4000-step carry (the explode/vanish question) ===")
for n in (32, 64, 128):
    core = RecurrentCore(n=n, width=2048, window=4).to(dev)
    torch.nn.init.xavier_uniform_(core.up.weight)      # open the gate fully
    s, norms = None, []
    with torch.no_grad():
        for chunk in range(8):                          # 8 x 500 = 4000 steps
            x = torch.randn(1, 500, 2048, device=dev)
            y, s = core(x, s)
            norms.append(s.norm().item())
    print(f"  N={n:4d}  state norm by chunk: {' '.join(f'{v:7.3f}' for v in norms)}"
          f"   finite={all(torch.isfinite(torch.tensor(v)) for v in norms)}")

print("\n=== 2. peak backward memory vs chunk length, at fixed window=4 ===")
for T in (256, 512, 1024, 2048):
    core = RecurrentCore(n=64, width=2048, window=4).to(dev)
    torch.nn.init.xavier_uniform_(core.up.weight)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    x = torch.randn(1, T, 2048, device=dev, requires_grad=True)
    y, _ = core(x); y.sum().backward()
    print(f"  T={T:5d}  peak {torch.cuda.max_memory_allocated()/2**20:8.1f} MiB")

print("\n=== 3. peak backward memory vs window, at fixed T=1024 ===")
for w in (1, 2, 4, 8, 0):
    core = RecurrentCore(n=64, width=2048, window=w).to(dev)
    torch.nn.init.xavier_uniform_(core.up.weight)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    x = torch.randn(1, 1024, 2048, device=dev, requires_grad=True)
    y, _ = core(x); y.sum().backward()
    label = "full BPTT" if w == 0 else f"window={w}"
    print(f"  {label:12s} peak {torch.cuda.max_memory_allocated()/2**20:8.1f} MiB")

print("\n=== 4. throughput of the per-token loop (the risk named in the plan) ===")
for n in (32, 64, 128):
    core = RecurrentCore(n=n, width=2048, window=4).to(dev)
    torch.nn.init.xavier_uniform_(core.up.weight)
    x = torch.randn(1, 1024, 2048, device=dev)
    for _ in range(2):                                   # warm up
        with torch.no_grad(): core(x)
    torch.cuda.synchronize(); t0 = time.time()
    reps = 5
    for _ in range(reps):
        with torch.no_grad(): core(x)
    torch.cuda.synchronize()
    per_call = (time.time() - t0) / reps
    print(f"  N={n:4d}  {per_call*1000:7.1f} ms per 1024-token chunk "
          f"-> {1024/per_call:9,.0f} core tok/s (forward only, 1 layer)")
