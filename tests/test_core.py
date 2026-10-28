import torch
from torch import nn

from odydora.core import RecurrentCore


def test_diagonal_stays_zero_and_state_is_bounded():
    torch.manual_seed(0)
    core = RecurrentCore(n=32, width=64)
    x = torch.randn(2, 40, 64)
    y, state = core(x)
    assert y.shape == x.shape
    assert torch.equal(core.W.diagonal(), torch.zeros(32))
    assert state.shape == (2, 32)
    assert torch.isfinite(y).all()


def test_zero_init_is_identity_yet_the_core_can_leave_the_origin():
    torch.manual_seed(4)
    core = RecurrentCore(n=16, width=32, window=4)
    x = torch.randn(1, 12, 32)
    y, _ = core(x)
    assert torch.equal(y, torch.zeros_like(y)), "step 0 must be an exact identity"

    # Only the out side is zeroed, so `up` takes gradient immediately. W and
    # `down` see grad_out @ up.weight, which IS zero at step 0 -- the same
    # behaviour as LoRA's B=0 blocking grad_A on the first step. They come
    # alive as soon as `up` has moved off zero, which is what this asserts.
    y.sum().backward()
    assert core.up.weight.grad.abs().sum() > 0, "out side must take gradient"

    # A target-based loss, as real training has: d/d out is non-zero at
    # out == 0, so `up` moves on the first step and the path into W opens.
    target = torch.randn_like(x)
    opt = torch.optim.SGD(core.parameters(), lr=0.5)
    for _ in range(3):
        opt.zero_grad()
        out, _ = core(x)
        (out - target).pow(2).mean().backward()
        opt.step()
    assert core.W.grad.abs().sum() > 0, "W must be learning once up has moved"
    assert core.down.weight.grad.abs().sum() > 0


def test_long_sequence_stays_finite_with_an_open_gate():
    torch.manual_seed(1)
    core = RecurrentCore(n=32, width=64, window=4)
    nn.init.xavier_uniform_(core.up.weight)
    y, s = core(torch.randn(1, 300, 64))
    assert torch.isfinite(y).all() and y.abs().max() < 100
    assert torch.isfinite(s).all()


def test_carry_across_chunks_equals_one_long_pass():
    torch.manual_seed(2)
    core = RecurrentCore(n=16, width=32, window=0)
    x = torch.randn(1, 24, 32)
    whole, _ = core(x)
    a, s = core(x[:, :12])
    b, _ = core(x[:, 12:], state=s)
    assert torch.allclose(whole, torch.cat([a, b], dim=1), atol=1e-5)


def test_gradient_does_not_cross_the_window_boundary():
    torch.manual_seed(3)
    core = RecurrentCore(n=8, width=16, window=2)
    nn.init.xavier_uniform_(core.up.weight)
    x = torch.randn(1, 6, 16, requires_grad=True)
    y, _ = core(x)
    y[:, -1].sum().backward()
    per_token = x.grad.abs().sum(dim=-1)[0]
    assert per_token[-1] > 0
    assert per_token[0] == 0
