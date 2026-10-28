"""Verify the convex mix across seeds, N, and very long carries, and check
that a LEARNABLE mixing rate stays in range. a is the one new parameter."""
import torch
from torch import nn

def influence(n, steps, a, seed):
    torch.manual_seed(seed)
    W = torch.empty(n, n); nn.init.xavier_uniform_(W); W.fill_diagonal_(0.)
    norm = nn.RMSNorm(n)
    p0 = torch.randn(1, n, requires_grad=True)
    s = p0.clone()
    for _ in range(steps):
        s = (1 - a) * s + a * norm(torch.randn(1, n) + s @ W)
    return torch.autograd.grad(s.abs().sum(), p0)[0].abs().mean().item(), s.norm().item()

print("=== a=0.02, across seeds and N, at 4096 steps (a full 50k trace in 64-token segments) ===")
print(f"{'N':>5} " + " ".join(f"{'seed '+str(s):>12}" for s in (0,1,2)) + f" {'|s|':>8}")
for n in (32, 64, 128):
    vals = [influence(n, 4096, 0.02, s) for s in (0,1,2)]
    print(f"{n:>5} " + " ".join(f"{v[0]:>12.2e}" for v in vals) + f" {vals[0][1]:>8.2f}")

print("\n=== plain norm (a=1.0) at the same depth, for contrast ===")
for n in (32, 64, 128):
    v = influence(n, 4096, 1.0, 0)
    print(f"  N={n:4d}  influence {v[0]:.2e}   |s| {v[1]:.2f}")

print("\n=== learnable a = sigmoid(alpha): what alpha init gives a=0.02? ===")
import math
for a_target in (0.02, 0.05, 0.1):
    alpha = math.log(a_target / (1 - a_target))
    print(f"  a={a_target:.3f} -> alpha={alpha:+.3f}  (sigmoid check {1/(1+math.exp(-alpha)):.4f})")

print("\n=== gradient reaches alpha? ===")
torch.manual_seed(0)
n = 64
W = nn.Parameter(torch.empty(n,n)); nn.init.xavier_uniform_(W)
alpha = nn.Parameter(torch.tensor(-3.892))   # sigmoid -> 0.02
norm = nn.RMSNorm(n)
s = torch.zeros(1,n)
for _ in range(32):
    a = torch.sigmoid(alpha)
    s = (1-a)*s + a*norm(torch.randn(1,n) + s @ W)
s.abs().sum().backward()
print(f"  d/d alpha = {alpha.grad.item():+.4e}   d/d W sum = {W.grad.abs().sum().item():.4e}")
