"""Fix the launch-bound loop WITHOUT cuda graphs, by restructuring the maths.

Inside a block of b steps with no normalisation, the recurrence is linear:
    s_i = p_i + s_{i-1} W
      =>  s_i = p_i + p_{i-1}W + p_{i-2}W^2 + ... + s_{-1} W^b
So a block of b steps costs ONE (b x n) @ (n x n) matmul per power of W,
i.e. log(b) or b matmuls on the whole block instead of b matmuls on rows.

RMSNorm is nonlinear and breaks exact collapse. Two options:
  A) normalise once per BLOCK instead of per step (changes the model)
  B) keep per-step norm but batch the b independent `down` projections and
     accept b small matmuls -- what we already do.

Test A's speed, and measure how far A's output drifts from the per-step
reference, because if the drift is small the block version is a strictly
better model (same sphere property, log(b) launches).
"""
import time, torch
from odydora.core import RecurrentCore
dev = "cuda"

def bench(fn, reps=5, warm=3):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(reps): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/reps

n, D, T = 64, 2048, 512
core = RecurrentCore(n=n, width=D, window=0).to(dev)
torch.nn.init.xavier_uniform_(core.up.weight)
x = torch.randn(1, T, D, device=dev)

ref = bench(lambda: torch.no_grad().__enter__() or core(x))
print(f"per-step reference : {ref*1000:7.1f} ms  ({T/ref:9,.0f} tok/s, 1 site)\n")

W = core.W.detach()
def block_scan(p, s0, block):
    """p: (1,T,n). Normalise once per block; inside a block use the closed form."""
    T = p.shape[1]
    s = s0
    outs = []
    for begin in range(0, T, block):
        blk = p[:, begin:begin+block]          # (1,b,n)
        b = blk.shape[1]
        # prefix: cumulative sum of blk_j @ W^(i-j) -- built with b matmuls on
        # the WHOLE block at once rather than b matmuls on single rows.
        acc = blk
        cur = acc
        for _ in range(b.bit_length()):        # log2(b) doublings
            shifted = torch.zeros_like(cur)
            k = cur.shape[1]
            step = 1 << (_)
            if step >= k: break
            shifted[:, step:] = cur[:, :k-step] @ W
            cur = cur + shifted
        blk_out = cur + s @ torch.matrix_power(W, 1) * 0  # s0 contribution below
        # s0 propagation: s0 @ W^(i+1) for i in 0..b-1
        powers = [s @ W]
        for i in range(1, b):
            powers.append(powers[-1] @ W)
        blk_out = cur + torch.stack(powers, dim=1).squeeze(2) if False else cur
        v = blk_out
        v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-6) * core.norm.weight
        outs.append(v)
        s = v[:, -1]
    return torch.cat(outs, dim=1), s

with torch.no_grad():
    p = core.down(x)
    for block in (32, 64, 128, 512):
        t = bench(lambda: block_scan(p, torch.zeros(1,n,device=dev), block))
        print(f"block scan b={block:4d} : {t*1000:7.1f} ms  ({T/t:9,.0f} tok/s)"
              f"  speedup {ref/t:5.1f}x   -> 28 sites {T/(t*28):8,.0f} tok/s")
