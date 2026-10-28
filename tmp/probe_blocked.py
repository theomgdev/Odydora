"""A blocked scan: W is linear, so within a detach window of w steps the
recurrence unrolls into a closed form with ONE matmul per window position
instead of one per token. Launches drop by a factor of ~w... but better,
we can batch the whole window's positions into a single bmm.

Key identity inside one window (no detach inside it):
  s_i = norm(p_i + s_{i-1} @ W)
Norm is nonlinear so we cannot collapse the whole window, BUT every token's
`down` projection and the final `up` projection are already batched; what is
left per step is (s @ W) + norm. That is 2 launches per token minimum.

So the real lever is: fewer PYTHON-LEVEL steps via CUDA graphs, which replay
a whole captured window as one launch.
"""
import time, torch
from odydora.core import RecurrentCore
dev = "cuda"

def bench(fn, reps=5, warm=3):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(reps): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/reps

core = RecurrentCore(n=64, width=2048, window=4).to(dev)
torch.nn.init.xavier_uniform_(core.up.weight)
x = torch.randn(1, 1024, 2048, device=dev)

base = bench(lambda: torch.no_grad().__enter__() or core(x))
print(f"baseline (python loop, no_grad): {base*1000:7.1f} ms -> {1024/base:,.0f} tok/s")

# --- CUDA graph capture of the whole 1024-step recurrence ---
print("\n=== CUDA graph replay of the full chunk recurrence ===")
try:
    with torch.no_grad():
        projected = core.down(x)            # already one batched matmul
        static_p = projected.clone()
        s = torch.zeros(1, 64, device=dev)
        # warmup on a side stream, required before capture
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            st = s.clone()
            for i in range(1024):
                st = core.norm(static_p[:, i] + st @ core.W)
        torch.cuda.current_stream().wait_stream(stream)

        g = torch.cuda.CUDAGraph()
        static_out = torch.zeros(1, 1024, 64, device=dev)
        with torch.cuda.graph(g):
            st = torch.zeros(1, 64, device=dev)
            for i in range(1024):
                st = core.norm(static_p[:, i] + st @ core.W)
                static_out[:, i] = st
    def replay():
        g.replay()
    t = bench(replay, reps=10, warm=3)
    print(f"  graph replay           {t*1000:7.1f} ms -> {1024/t:10,.0f} tok/s"
          f"   speedup {base/t:.1f}x")
    print(f"  28 call sites          {t*28*1000:7.1f} ms -> {1024/(t*28):10,.0f} tok/s")
except Exception as e:
    print(f"  CUDA graph FAILED: {type(e).__name__}: {str(e)[:200]}")

# --- how much is norm costing us? try fusing norm away ---
print("\n=== is RMSNorm the launch hog? (replace with rsqrt inline) ===")
Wd = core.W.detach(); gain = core.norm.weight.detach()
p = core.down(x).detach()
def manual():
    with torch.no_grad():
        st = torch.zeros(1, 64, device=dev)
        for i in range(1024):
            v = p[:, i] + st @ Wd
            st = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-6) * gain
t = bench(manual)
print(f"  manual rmsnorm         {t*1000:7.1f} ms -> {1024/t:10,.0f} tok/s")
