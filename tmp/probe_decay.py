"""Does the sphere property destroy memory, and does decay fix it?

RMSNorm pins |s| to sqrt(N) every step. That stops explosion, but it also
means the input's contribution is renormalised away: after k steps, how much
of s still depends on token t? Measure the influence decay directly.
"""
import torch
from torch import nn
dev = "cpu"
torch.manual_seed(0)
n, D = 64, 256

def influence(gamma, steps=64, norm_on=True):
    """d|s_k| / d p_0 : how much the state still remembers the first input."""
    W = torch.empty(n, n); nn.init.xavier_uniform_(W)
    W.fill_diagonal_(0.); norm = nn.RMSNorm(n)
    p0 = torch.randn(1, n, requires_grad=True)
    s = p0.clone()
    for k in range(steps):
        inp = torch.randn(1, n) * 1.0
        v = inp + gamma * (s @ W)
        s = norm(v) if norm_on else v
    g = torch.autograd.grad(s.abs().sum(), p0)[0]
    return g.abs().mean().item()

print("influence of the FIRST input on the state after k steps")
print(f"{'steps':>6} {'gamma=1.0':>12} {'gamma=0.95':>12} {'gamma=0.88':>12} {'no norm':>12}")
for steps in (4, 16, 64, 256):
    row = [influence(1.0, steps), influence(0.95, steps),
           influence(0.88, steps), influence(1.0, steps, norm_on=False)]
    print(f"{steps:>6} " + " ".join(f"{v:>12.3e}" for v in row))

print("\n=== does the state stay on the sphere WITH decay? ===")
for gamma in (1.0, 0.95, 0.88):
    W = torch.empty(n, n); nn.init.xavier_uniform_(W); W.fill_diagonal_(0.)
    norm = nn.RMSNorm(n); s = torch.zeros(1, n)
    with torch.no_grad():
        for _ in range(200):
            s = norm(torch.randn(1, n) + gamma * (s @ W))
    print(f"  gamma={gamma:.2f}  |s|={s.norm().item():.4f}  (sqrt(N)={n**0.5:.4f})")
