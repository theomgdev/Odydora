import warnings

import torch
from torch import nn

from odydora.core import RecurrentCore


def test_diagonal_stays_zero_and_shapes_round_trip():
    torch.manual_seed(0)
    core = RecurrentCore(n=32, width=64, stride=8)
    x = torch.randn(2, 40, 64)
    y, state = core(x)
    assert y.shape == x.shape, "delta must match the residual stream it joins"
    assert torch.equal(core.W.diagonal(), torch.zeros(32))
    assert state.shape == (2, 32)
    assert torch.isfinite(y).all()


def test_zero_init_is_identity_yet_the_core_can_leave_the_origin():
    torch.manual_seed(4)
    core = RecurrentCore(n=16, width=32, stride=4)
    x = torch.randn(1, 12, 32)
    y, _ = core(x)
    assert torch.equal(y, torch.zeros_like(y)), "step 0 must be an exact identity"

    # Only the out side is zeroed, so `up` takes gradient immediately. W and
    # `down` see grad_out @ up.weight, which IS zero at step 0 -- the same
    # behaviour as LoRA's B=0 blocking grad_A on the first step. They come
    # alive as soon as `up` has moved off zero, which is what this asserts.
    y.sum().backward()
    assert core.up.weight.grad.abs().sum() > 0, "out side must take gradient"

    target = torch.randn_like(x)
    opt = torch.optim.SGD(core.parameters(), lr=0.5)
    for _ in range(3):
        opt.zero_grad()
        out, _ = core(x)
        (out - target).pow(2).mean().backward()
        opt.step()
    assert core.W.grad.abs().sum() > 0, "W must be learning once up has moved"
    assert core.down.weight.grad.abs().sum() > 0
    assert core.alpha.grad is not None and core.alpha.grad.abs() > 0


def test_long_carry_stays_finite_with_an_open_output():
    torch.manual_seed(1)
    core = RecurrentCore(n=32, width=64, stride=8)
    nn.init.xavier_uniform_(core.up.weight)
    s = None
    with torch.no_grad():
        for _ in range(12):
            y, s = core(torch.randn(1, 256, 64), s)
            assert torch.isfinite(y).all() and y.abs().max() < 100
            assert torch.isfinite(s).all()


def test_state_norm_is_bounded_however_hostile_W_is():
    """The convex mix keeps the state bounded without any clamp: the new
    state is on the sqrt(N) sphere and the old one is inside it, so the
    mixture can never leave the ball. Holds with W scaled far past any
    sane init, which is why no spectral constraint is needed.
    """
    torch.manual_seed(5)
    for n in (16, 32, 64):
        core = RecurrentCore(n=n, width=64, stride=4, mix=0.1)
        core.W.data.mul_(8.0)
        s = None
        with torch.no_grad():
            for _ in range(20):
                _, s = core(torch.randn(1, 128, 64), s)
        assert s.norm().item() <= n**0.5 + 1e-3, (
            f"N={n}: |s|={s.norm().item():.3f} escaped the sqrt(N) ball"
        )


def test_the_convex_mix_keeps_old_inputs_influential():
    """A pure RMSNorm recurrence forgets completely -- influence of an early
    input measures 0.0 after a few thousand steps, because the norm
    renormalises the past away. A small mixing rate is what preserves it,
    and this asserts the ordering rather than a fragile absolute value.
    """
    def influence(mix, steps=256, n=32):
        torch.manual_seed(7)
        core = RecurrentCore(n=n, width=64, stride=1, mix=mix)
        first = torch.randn(1, 1, 64, requires_grad=True)
        _, s = core(first)
        for _ in range(steps):
            _, s = core(torch.randn(1, 1, 64), s)
        return torch.autograd.grad(s.abs().sum(), first)[0].abs().mean().item()

    small, large = influence(0.02), influence(0.95)
    assert small > large, f"mix=0.02 ({small:.2e}) must outlast mix=0.95 ({large:.2e})"
    assert small > 1e-12, f"a small mix should keep real influence, got {small:.2e}"


def test_carry_across_chunks_equals_one_long_pass():
    torch.manual_seed(2)
    core = RecurrentCore(n=16, width=32, stride=4, window=0)
    nn.init.xavier_uniform_(core.up.weight)
    x = torch.randn(1, 24, 32)
    whole, s_whole = core(x)
    a, s = core(x[:, :12])
    b, s_chunked = core(x[:, 12:], state=s)
    assert whole.abs().sum() > 0, "up must be open so delta comparison is non-trivial"
    assert torch.allclose(whole, torch.cat([a, b], dim=1), atol=1e-5), (
        "chunked carry must equal one pass, or every later measurement is void"
    )
    assert torch.allclose(s_whole, s_chunked, atol=1e-5)


def test_partial_segment_mean_is_not_diluted_by_zero_padding():
    """A partial segment (such as time=1 in autoregressive decoding or a ragged
    tail chunk) must divide by its true token count, not by `stride`.
    """
    torch.manual_seed(3)
    core = RecurrentCore(n=16, width=32, stride=16, mix=0.2)
    nn.init.xavier_uniform_(core.up.weight)
    s_prev = torch.randn(1, 16)

    token = torch.randn(1, 1, 32) * 2.0
    full_seg = token.expand(1, 16, 32).contiguous()

    delta_one, s_one = core(token, state=s_prev)
    delta_full, s_full = core(full_seg, state=s_prev)
    assert torch.allclose(s_one, s_full, atol=1e-5), (
        f"time=1 state diverged from full segment: {(s_one - s_full).norm():.4f}"
    )
    assert torch.allclose(delta_one[:, 0], delta_full[:, 0], atol=1e-5)

    # Also check multi-segment ragged tail (1 full segment + 3 identical tail tokens)
    head = torch.randn(1, 16, 32)
    ragged = torch.cat([head, token.expand(1, 3, 32)], dim=1)
    padded_equiv = torch.cat([head, full_seg], dim=1)
    _, s_ragged = core(ragged, state=s_prev)
    _, s_equiv = core(padded_equiv, state=s_prev)
    assert torch.allclose(s_ragged, s_equiv, atol=1e-5)


def test_stride_sets_the_number_of_state_updates_not_the_parameters():
    """The whole point of segment-rate recurrence: cost drops with stride
    while the parameter count is untouched.
    """
    counts = []
    for stride in (1, 8, 64):
        core = RecurrentCore(n=32, width=64, stride=stride)
        counts.append(sum(p.numel() for p in core.parameters()))
    assert len(set(counts)) == 1, f"stride changed the parameter count: {counts}"

    core = RecurrentCore(n=32, width=64, stride=16)
    x = torch.randn(1, 64, 64)
    y, _ = core(x)
    assert y.shape == x.shape, "a strided delta must still cover every token"


def test_a_ragged_tail_is_padded_and_trimmed_back():
    core = RecurrentCore(n=8, width=16, stride=10)
    for time in (1, 7, 10, 11, 23):
        y, s = core(torch.randn(1, time, 16))
        assert y.shape == (1, time, 16), f"time={time} came back as {tuple(y.shape)}"
        assert torch.isfinite(s).all()


def test_mix_is_rejected_outside_the_open_unit_interval():
    for bad in (0.0, 1.0, -0.1, 1.5):
        try:
            RecurrentCore(n=8, width=16, mix=bad)
        except ValueError:
            continue
        raise AssertionError(f"mix={bad} should have been rejected")


def test_mix_cannot_leave_zero_one_however_alpha_trains():
    core = RecurrentCore(n=8, width=16, mix=0.02)
    for value in (-50.0, 50.0):
        core.alpha.data.fill_(value)
        assert 0.0 < float(core.mix) < 1.0, f"alpha={value} escaped the interval"


def test_repr_does_not_touch_the_autograd_graph():
    """Printing the module must not convert a grad-tracking tensor to a scalar.

    `mix` is a property derived from the live `alpha` parameter, so it carries
    requires_grad, and the warning torch raises for that recommends detaching.

    Honest limitation: this test passes on torch 2.7.1 even with the old
    `float(self.mix)` code, because 2.7 does not warn there -- the warning was
    observed on Colab's newer torch. So it is a guard against regression on
    newer runtimes, not a test that failed here first.
    """
    core = RecurrentCore(n=8, width=16, mix=0.02)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        text = repr(core)
    assert "mix=0.0200" in text
    assert not core.mix.detach().requires_grad
