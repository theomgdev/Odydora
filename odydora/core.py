"""The shared recurrent core: one N x N mixer called at every layer boundary.

Two design facts here were measured rather than assumed, and both were
surprises; `Research/SUMMARY.md` carries the tables.

Per-step RMSNorm pins the state to the sqrt(N) sphere exactly, which makes
explosion structurally impossible whatever W's spectral radius is -- but it
also renormalises the past away, so a pure norm recurrence forgets
*completely*: the influence of an input on the state 4096 steps later
measures 0.0e+00 at every width and seed tried. A contractive decay, the
usual prescription for a recurrence, makes that worse rather than better.
Mixing the new state into the old convexly is what keeps memory alive, and
it bounds the norm at the same time.
"""

import torch
from torch import nn


def _logit(p):
    return float(torch.logit(torch.tensor(p)))


class RecurrentCore(nn.Module):
    """One N x N mixer shared by every call site.

    `width` is the host model's hidden size; `n` is the core's own width.

    `stride` is how many tokens one state update covers. The state advances
    once per segment rather than once per token, which is what Transformer-XL
    and Block-Recurrent Transformer do and what makes this affordable: the
    step is kernel-launch-bound (batch 1 and batch 64 cost nearly the same
    wall time), so cost tracks the number of Python-level steps and nothing
    else. Measured on a 512-token chunk at 28 call sites: 25 tok/s at
    stride 1 against 1,263 at stride 64 and 2,261 at stride 128, for
    identical parameters. It also lengthens the carry in token terms -- 64
    updates span 4,096 tokens at stride 64 -- which is the axis memory
    actually lives on.

    `mix` is the initial convex mixing rate: s <- (1-a)*s_prev + a*new. It
    is learnable through a sigmoid and small by default, because that is
    what keeps influence alive over a long carry (5.8e-01 after 1,024 steps
    at a=0.02 against 2.0e-34 at a=1.0).

    `window` detaches the state every `window` steps inside a call and
    defaults to off: peak backward memory at a 1024-token chunk is 61.8 MiB
    for window 2, 4, 8 and full backpropagation alike (85.8 MiB at window=1,
    which is worse), so intra-chunk truncation buys no memory and only
    shortens how far credit can travel. Truncation happens at the chunk
    boundary, where the state crosses detached anyway.
    """

    def __init__(self, n, width, stride=64, mix=0.02, window=0):
        super().__init__()
        if not 0.0 < mix < 1.0:
            raise ValueError(f"mix must lie in (0, 1), got {mix}")
        self.n, self.window = n, window
        self.stride = max(int(stride), 1)
        self.down = nn.Linear(width, n, bias=False)
        self.up = nn.Linear(n, width, bias=False)
        self.norm = nn.RMSNorm(n)
        self.W = nn.Parameter(torch.empty(n, n))
        nn.init.xavier_uniform_(self.W)
        with torch.no_grad():
            self.W.fill_diagonal_(0.0)
        # Self-connections belong to the host residual stream, not to W, so
        # the diagonal is held at zero on the gradient as well as at init.
        self.W.register_hook(lambda g: g.clone().fill_diagonal_(0.0))
        # Mixing rate as a logit, so it cannot leave (0, 1) however it trains.
        self.alpha = nn.Parameter(torch.tensor(_logit(mix)))
        # Zero-initializing the OUT side alone makes the insertion an exact
        # identity at step 0 while still taking gradient (d delta/d up = s),
        # which is how LoRA's B=0 works. Zeroing a second factor as well --
        # an extra gate, or the in side too -- makes every factor's gradient
        # contain a zero and the whole core never leaves the origin.
        nn.init.zeros_(self.up.weight)

    @property
    def mix(self):
        """Current convex mixing rate, strictly inside (0, 1).

        The logit is clamped rather than the rate: sigmoid saturates to
        exactly 1.0 in float32 somewhere past 17, and a mix of exactly 1
        drops the previous state entirely -- the forgetting failure this
        design exists to avoid. The bound sits well outside any rate
        training would plausibly want, so it never shapes the solution.
        """
        return torch.sigmoid(self.alpha.clamp(-15.0, 15.0))

    def forward(self, x, state=None):
        """x: (batch, time, width) -> (delta, last_state).

        `delta` is added to the residual stream by the caller; `last_state`
        is handed back so the next chunk of the same document continues the
        recurrence instead of restarting it.
        """
        batch, time, _ = x.shape
        segments = (time + self.stride - 1) // self.stride
        padding = segments * self.stride - time

        # Mean-pool each segment in one batched op, then one batched `down`.
        pooled = x if padding == 0 else nn.functional.pad(x, (0, 0, 0, padding))
        pooled = pooled.view(batch, segments, self.stride, -1).mean(2)
        projected = self.down(pooled)

        s = x.new_zeros(batch, self.n) if state is None else state
        a = self.mix.to(projected.dtype)
        states = []
        for i in range(segments):
            s = (1 - a) * s + a * self.norm(projected[:, i] + s @ self.W)
            states.append(s)
            if self.window and (i + 1) % self.window == 0:
                s = s.detach()

        # One batched `up`, then broadcast each segment's delta to its tokens.
        delta = self.up(torch.stack(states, dim=1))
        delta = delta.unsqueeze(2).expand(batch, segments, self.stride, delta.shape[-1])
        return delta.reshape(batch, segments * self.stride, -1)[:, :time], s

    def extra_repr(self):
        return (
            f"n={self.n}, stride={self.stride}, "
            f"mix={float(self.mix):.4f}, window={self.window}"
        )
