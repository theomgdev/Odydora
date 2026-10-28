"""The shared recurrent core: one N x N mixer called at every layer boundary."""

import torch
from torch import nn


class RecurrentCore(nn.Module):
    """One N x N mixer shared by every call site.

    `width` is the host model's hidden size; `n` is the core's own width and
    the only size knob. `window` is how many recurrent steps a backward pass
    may span: the state is detached every `window` tokens, which makes peak
    activation memory independent of sequence length. `window=0` disables
    detaching, which is what makes chunked carry exactly equal to one long
    pass and is used by the equivalence test.
    """

    def __init__(self, n, width, window=4):
        super().__init__()
        self.n, self.window = n, window
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
        # Zero-initializing the OUT side alone makes the insertion an exact
        # identity at step 0 while still taking gradient (d delta/d up = s),
        # which is how LoRA's B=0 works. Zeroing a second factor as well --
        # an extra gate, or the in side too -- makes every factor's gradient
        # contain a zero and the whole core never leaves the origin.
        nn.init.zeros_(self.up.weight)

    def forward(self, x, state=None):
        """x: (batch, time, width) -> (delta, last_state).

        `delta` is added to the residual stream by the caller; `last_state`
        is handed back so the next chunk of the same document continues the
        recurrence instead of restarting it.
        """
        b, t, _ = x.shape
        s = x.new_zeros(b, self.n) if state is None else state
        projected = self.down(x)
        outs = []
        for i in range(t):
            s = self.norm(projected[:, i] + s @ self.W)
            outs.append(s)
            if self.window and (i + 1) % self.window == 0:
                s = s.detach()
        return self.up(torch.stack(outs, dim=1)), s

    def extra_repr(self):
        return f"n={self.n}, window={self.window}"
