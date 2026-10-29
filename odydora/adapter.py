"""Insert one shared RecurrentCore at chosen decoder-layer boundaries.

The host model stays frozen; the core is added to each selected block's
output hidden state through a forward hook, so the block object itself is
untouched and every attribute the transformers internals read off it
(`attention_type`, `layer_idx`, `self_attn`, ...) keeps working. One core
instance serves every call site, so the mechanism costs one parameter set
no matter how many layers use it; the per-layer state lives here rather
than in the core, which keeps the core stateless and therefore shareable.
"""

from contextlib import contextmanager, nullcontext

from torch import nn

from .core import RecurrentCore

# Decoder-stack attribute names, in the order we look for them.
_STACK_PATHS = (
    "model.layers",  # Llama / Qwen2 / Qwen3 / Mistral / Phi3
    "transformer.h",  # GPT-2 / GPT-J / Falcon
    "model.decoder.layers",  # OPT
    "gpt_neox.layers",  # GPT-NeoX
)


def _resolve_stack(model):
    """Return (path, ModuleList) for the model's decoder blocks.

    Searched by structure rather than by a fixed path list, because the
    path moves with the wrapper: a bare Qwen3 keeps its blocks at
    `model.layers`, while `get_peft_model` nests the host twice and the
    same blocks live at `base_model.model.model.layers`. A decoder block is
    identified by carrying one of the attention attributes every supported
    family uses, so a new wrapper or a new family needs no new entry here.
    The literal paths are still tried first, since that is the common case
    and an exact hit avoids walking the whole module tree.
    """
    for path in _STACK_PATHS:
        node = model
        for part in path.split("."):
            node = getattr(node, part, None)
            if node is None:
                break
        if isinstance(node, nn.ModuleList) and len(node) > 0:
            return path, node

    for name, module in model.named_modules():
        if (
            isinstance(module, nn.ModuleList)
            and len(module) > 0
            and any(hasattr(module[0], a) for a in ("self_attn", "attn", "attention"))
        ):
            return name, module

    raise ValueError(
        f"could not find a decoder stack on {type(model).__name__}: no "
        "ModuleList whose first entry has self_attn/attn/attention"
    )


class _Carry:
    """Per-layer recurrent state, carried across chunks of one document."""

    def __init__(self, indices):
        self.indices = tuple(indices)
        self.calls = 0
        self.reset()

    def reset(self):
        self.state = {i: None for i in self.indices}

    def detach(self):
        for key, value in self.state.items():
            if value is not None:
                self.state[key] = value.detach()


def _make_hook(core, carry, index):
    def hook(_module, _args, output):
        if getattr(core, "enabled", True) is False:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        delta, state = core(hidden, carry.state.get(index))
        # Detached so a later chunk reads history without backpropagating
        # into it; `window` inside the core bounds the live graph.
        carry.state[index] = state.detach()
        carry.calls += 1
        hidden = hidden + delta
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    return hook


def attach_core(model, n, layers=None, stride=64, mix=0.02, window=0, freeze_base=True):
    """Attach one shared RecurrentCore at each layer boundary in `layers`.

    `layers` defaults to every decoder block. `freeze_base=True` freezes the
    host model, leaving only the core (plus any adapter applied separately)
    trainable. Returns the same model object, carrying `odydora_core`,
    `odydora_layers`, `odydora_reset_state()`, `odydora_detach_state()`,
    `odydora_state()`, `odydora_calls()`, `odydora_bypass()` and `without_odydora()`.

    `n=0` or `n=None` creates an adapter-only control arm: no hooks are registered,
    core parameter count is 0, and the recurrent lifecycle methods become no-ops.
    """
    if not n:
        model.odydora_core = None
        model.odydora_layers = ()
        model.odydora_handles = []
        model.odydora_reset_state = lambda: None
        model.odydora_detach_state = lambda: None
        model.odydora_state = lambda: {}
        model.odydora_calls = lambda: 0
        model.odydora_bypass = lambda: nullcontext()
        model.without_odydora = lambda: without_odydora(model)
        if freeze_base:
            model.requires_grad_(False)
        return model

    path, stack = _resolve_stack(model)
    indices = tuple(range(len(stack))) if layers is None else tuple(layers)
    for i in indices:
        if not 0 <= i < len(stack):
            raise ValueError(f"layer {i} out of range for {path} of {len(stack)}")

    width = getattr(model.config, "hidden_size", None) or model.config.n_embd
    if freeze_base:
        model.requires_grad_(False)

    reference = next(model.parameters())
    core = RecurrentCore(n=n, width=width, stride=stride, mix=mix, window=window).to(
        device=reference.device, dtype=reference.dtype
    )
    carry = _Carry(indices)
    handles = [
        stack[i].register_forward_hook(_make_hook(core, carry, i)) for i in indices
    ]

    # Assigning the module registers it, so the core's parameters appear in
    # model.parameters() and are saved alongside the adapter.
    model.odydora_core = core
    model.odydora_layers = indices
    model.odydora_handles = handles
    model.odydora_reset_state = carry.reset
    model.odydora_detach_state = carry.detach
    model.odydora_state = lambda: dict(carry.state)
    model.odydora_calls = lambda: carry.calls
    model.odydora_bypass = lambda: bypass_core(model)
    model.without_odydora = lambda: without_odydora(model)
    return model


@contextmanager
def bypass_core(model):
    """Temporarily bypass the RecurrentCore without unregistering forward hooks."""
    core = getattr(model, "odydora_core", None)
    if core is None:
        yield
        return
    prev = getattr(core, "enabled", True)
    core.enabled = False
    try:
        yield
    finally:
        core.enabled = prev


@contextmanager
def without_odydora(model):
    """Temporarily disable both the RecurrentCore and any attached adapter."""
    core = getattr(model, "odydora_core", None)
    prev = getattr(core, "enabled", True) if core is not None else None
    if core is not None:
        core.enabled = False
    has_peft = hasattr(model, "disable_adapter")
    try:
        if has_peft:
            with model.disable_adapter():
                yield
        else:
            yield
    finally:
        if core is not None:
            core.enabled = prev


def detach_core(model):
    """Remove the hooks, returning the host model to its stock behaviour."""
    for handle in model.odydora_handles:
        handle.remove()
    model.odydora_handles = []
    return model


def count_parameters(model):
    """(adapter_params, core_params) — the numerator of the record claim.

    Anything trainable that is not the core counts as adapter, which is what
    makes a matched-parameter comparison against an adapter-only arm possible.
    """
    core_mod = getattr(model, "odydora_core", None)
    if core_mod is None:
        adapter = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return adapter, 0

    core_ids = {id(p) for p in core_mod.parameters()}
    core = sum(p.numel() for p in core_mod.parameters() if p.requires_grad)
    # W's diagonal is pinned at zero and never trains, so it is not a parameter.
    if hasattr(core_mod, "W") and core_mod.W.requires_grad:
        core -= core_mod.n
    adapter = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad and id(p) not in core_ids
    )
    return adapter, core
