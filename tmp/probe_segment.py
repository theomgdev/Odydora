"""The structural fix: recur over SEGMENTS, not tokens.

Transformer-XL and Block-Recurrent Transformer both do this -- state advances
once per segment of tokens, not once per token. For us that means:

  per chunk of T tokens at layer L:
     summary = pool(h)            # one vector for the chunk (or per sub-block)
     s = RMSNorm(down(summary) + s_prev @ W)
     h = h + up(s)                # broadcast back to every token

Launches per chunk per layer: O(1) instead of O(T). The state still carries
across layers and across chunks -- the 'frontal cortex' idea is intact -- but
it updates at segment rate, which is what makes it affordable.

Granularity is a knob: `stride` tokens per state update. stride=1 is the
per-token version we measured; stride=T is one update per chunk.
"""
import time, torch
from torch import nn
dev = "cuda"

def bench(fn, reps=5, warm=3):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(reps): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/reps

n, D, T = 64, 2048, 512

class SegmentCore(nn.Module):
    """State advances once per `stride` tokens."""
    def __init__(self, n, width, stride=64):
        super().__init__()
        self.n, self.stride = n, stride
        self.down = nn.Linear(width, n, bias=False)
        self.up = nn.Linear(n, width, bias=False)
        self.norm = nn.RMSNorm(n)
        self.W = nn.Parameter(torch.empty(n, n)); nn.init.xavier_uniform_(self.W)
        with torch.no_grad(): self.W.fill_diagonal_(0.)
        nn.init.zeros_(self.up.weight)

    def forward(self, x, state=None):
        B, T, _ = x.shape
        segs = (T + self.stride - 1) // self.stride
        pad = segs * self.stride - T
        xp = torch.nn.functional.pad(x, (0,0,0,pad))
        # mean-pool each segment: one batched op for the whole chunk
        pooled = xp.view(B, segs, self.stride, -1).mean(2)        # (B,segs,D)
        p = self.down(pooled)                                      # (B,segs,n)
        s = x.new_zeros(B, self.n) if state is None else state
        outs = []
        for i in range(segs):                                      # segs << T
            s = self.norm(p[:, i] + s @ self.W)
            outs.append(s)
        states = torch.stack(outs, 1)                              # (B,segs,n)
        delta = self.up(states)                                    # (B,segs,D)
        delta = delta.unsqueeze(2).expand(B, segs, self.stride, delta.shape[-1])
        delta = delta.reshape(B, segs*self.stride, -1)[:, :T]
        return delta, s

x = torch.randn(1, T, D, device=dev)
print(f"{'stride':>8} {'segs':>5} {'ms':>8} {'tok/s':>10} {'28 sites tok/s':>16}")
for stride in (1, 8, 32, 64, 128, 512):
    core = SegmentCore(n, D, stride).to(dev)
    nn.init.xavier_uniform_(core.up.weight)
    t = bench(lambda: torch.no_grad().__enter__() or core(x))
    segs = (T + stride - 1)//stride
    print(f"{stride:>8} {segs:>5} {t*1000:>8.2f} {T/t:>10,.0f} {T/(t*28):>16,.0f}")

print("\n=== params, unchanged by stride (W, down, up, norm all shared) ===")
core = SegmentCore(n, D, 64)
print(f"  {sum(p.numel() for p in core.parameters()) - n:,} trainable (W diag excluded)")
