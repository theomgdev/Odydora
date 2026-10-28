"""Is the per-token loop launch-bound, and does compile/batching fix it?"""
import time, torch
from odydora.core import RecurrentCore
dev = "cuda"

def bench(fn, reps=5, warm=2):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(reps): fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / reps

core = RecurrentCore(n=64, width=2048, window=4).to(dev)
torch.nn.init.xavier_uniform_(core.up.weight)

print("=== launch-bound test: does cost scale with batch? ===")
for b in (1, 4, 16, 64):
    x = torch.randn(b, 1024, 2048, device=dev)
    t = bench(lambda: core(x) if False else torch.no_grad().__enter__() or core(x))
    print(f"  batch={b:3d}  {t*1000:7.1f} ms/chunk  -> {b*1024/t:10,.0f} tok/s")

print("\n=== per-step cost breakdown at batch=1 ===")
s = torch.randn(1, 64, device=dev); W = core.W.detach()
p = torch.randn(1, 1024, 64, device=dev)
print(f"  1024x (s@W)            {bench(lambda: [s@W for _ in range(1024)])*1000:7.1f} ms")
print(f"  1024x (s@W + slice)    {bench(lambda: [p[:,i]+s@W for i in range(1024)])*1000:7.1f} ms")
print(f"  1024x full step        {bench(lambda: [core.norm(p[:,i]+s@W) for i in range(1024)])*1000:7.1f} ms")

print("\n=== torch.compile on the step function ===")
try:
    step = torch.compile(lambda prev, inp: core.norm(inp + prev @ core.W), dynamic=False)
    def run_compiled():
        st = torch.zeros(1, 64, device=dev)
        with torch.no_grad():
            for i in range(1024): st = step(st, p[:, i])
    t = bench(run_compiled, reps=3, warm=2)
    print(f"  compiled loop          {t*1000:7.1f} ms -> {1024/t:10,.0f} tok/s")
except Exception as e:
    print(f"  compile FAILED: {type(e).__name__}: {str(e)[:160]}")

print("\n=== cost if attached to all 28 layers of Qwen3-1.7B ===")
x = torch.randn(1, 1024, 2048, device=dev)
one = bench(lambda: torch.no_grad().__enter__() or core(x))
print(f"  1 layer  {one*1000:7.1f} ms   28 layers {one*28*1000:8.1f} ms per 1024-token chunk")
print(f"  -> core alone caps throughput at {1024/(one*28):,.0f} tok/s with 28 call sites")
for k in (2, 4, 7):
    print(f"  every {28//k:2d}th layer ({k:2d} sites): {one*k*1000:7.1f} ms -> {1024/(one*k):9,.0f} tok/s")
