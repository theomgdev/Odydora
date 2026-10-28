"""Memory vanishes because RMSNorm renormalises old state away. Which
mechanism actually keeps influence alive over hundreds of steps?

Arms:
  A  plain                s = norm(p + sW)
  B  decay (brief 04)     s = norm(p + 0.9 sW)
  C  residual carry       s = norm(p + sW) ; s = s + s_prev   -- skip in state
  D  convex mix (GRU-ish) s = (1-a)*s_prev + a*norm(p + sW)
  E  no norm, decay only  s = p + 0.9 sW
"""
import torch
from torch import nn
torch.manual_seed(0)
n = 64

def influence(kind, steps, a=0.1, gamma=0.9):
    W = torch.empty(n, n); nn.init.xavier_uniform_(W); W.fill_diagonal_(0.)
    norm = nn.RMSNorm(n)
    p0 = torch.randn(1, n, requires_grad=True)
    s = p0.clone()
    for _ in range(steps):
        inp = torch.randn(1, n)
        if kind == "A":   s = norm(inp + s @ W)
        elif kind == "B": s = norm(inp + gamma * (s @ W))
        elif kind == "C": s = norm(inp + s @ W) + s
        elif kind == "D": s = (1 - a) * s + a * norm(inp + s @ W)
        elif kind == "E": s = inp + gamma * (s @ W)
    g = torch.autograd.grad(s.abs().sum(), p0)[0]
    return g.abs().mean().item(), s.norm().item()

names = {"A":"plain norm", "B":"decay 0.9", "C":"residual carry",
         "D":"convex mix a=0.1", "E":"no norm, decay"}
print(f"{'arm':>20} " + " ".join(f"{s:>11}" for s in ("k=16","k=64","k=256","|s| @256")))
for kind in "ABCDE":
    vals = [influence(kind, k)[0] for k in (16, 64, 256)]
    norm_at_256 = influence(kind, 256)[1]
    print(f"{names[kind]:>20} " + " ".join(f"{v:>11.2e}" for v in vals)
          + f" {norm_at_256:>11.2f}")

print("\n=== convex mix: sweep the mixing rate ===")
print(f"{'a':>8} {'k=64':>12} {'k=256':>12} {'k=1024':>12} {'|s|':>10}")
for a in (0.02, 0.05, 0.1, 0.3, 0.5):
    vals = [influence("D", k, a=a)[0] for k in (64, 256, 1024)]
    print(f"{a:>8.2f} " + " ".join(f"{v:>12.2e}" for v in vals)
          + f" {influence('D',256,a=a)[1]:>10.2f}")
