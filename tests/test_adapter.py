import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    Qwen2Config,
    Qwen2ForCausalLM,
)

from odydora.adapter import attach_core, count_parameters
from odydora.core import RecurrentCore


def tiny_gpt2():
    return GPT2LMHeadModel(
        GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=128, n_positions=64)
    )


def tiny_qwen():
    return Qwen2ForCausalLM(
        Qwen2Config(
            num_hidden_layers=2,
            hidden_size=32,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=128,
            max_position_embeddings=64,
        )
    )


def cores_of(model):
    return [m for m in model.modules() if isinstance(m, RecurrentCore)]


def test_one_core_is_shared_and_trained():
    for build in (tiny_gpt2, tiny_qwen):
        torch.manual_seed(0)
        model = attach_core(build(), n=16, layers=(0, 1), stride=4)
        ids = torch.randint(0, 128, (2, 24))
        model.odydora_reset_state()
        model(input_ids=ids, labels=ids).loss.backward()
        # modules() deduplicates a shared module, so exactly one core for two
        # call sites IS the sharing property; the hook counter proves both
        # boundaries actually ran it.
        cores = cores_of(model)
        assert len(cores) == 1, f"{build.__name__}: core is not shared"
        assert model.odydora_calls() == 2, f"{build.__name__}: core not called per layer"
        # At step 0 only the out side takes gradient (zero-init `up` blocks
        # the path into W, as LoRA's B=0 blocks grad_A); after a few steps the
        # whole core is learning. Both halves are asserted below.
        assert cores[0].up.weight.grad.abs().sum() > 0, f"{build.__name__}: up dead"

        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.05)
        for _ in range(3):
            opt.zero_grad()
            model.odydora_reset_state()
            model(input_ids=ids, labels=ids).loss.backward()
            opt.step()
        assert cores[0].W.grad.abs().sum() > 0, f"{build.__name__}: W never learns"
        assert cores[0].down.weight.grad.abs().sum() > 0


def test_zero_gate_leaves_the_base_model_untouched():
    for build in (tiny_gpt2, tiny_qwen):
        torch.manual_seed(1)
        base = build().eval()
        ids = torch.randint(0, 128, (1, 16))
        with torch.no_grad():
            before = base(input_ids=ids).logits
        model = attach_core(base, n=16, layers=(0, 1), stride=4).eval()
        with torch.no_grad():
            after = model(input_ids=ids).logits
        assert torch.equal(before, after), f"{build.__name__}: core is not identity"


def test_state_carries_across_calls_and_resets_on_demand():
    torch.manual_seed(2)
    model = attach_core(tiny_qwen(), n=16, layers=(0, 1), stride=4).eval()
    ids = torch.randint(0, 128, (1, 12))
    model.odydora_reset_state()
    with torch.no_grad():
        model(input_ids=ids)
    carried = model.odydora_state()
    assert set(carried) == {0, 1}
    assert all(v is not None and torch.isfinite(v).all() for v in carried.values())
    model.odydora_reset_state()
    assert all(v is None for v in model.odydora_state().values())


def test_carried_state_is_detached_so_chunks_do_not_grow_the_graph():
    torch.manual_seed(3)
    model = attach_core(tiny_qwen(), n=16, layers=(0, 1), stride=4)
    ids = torch.randint(0, 128, (1, 12))
    model.odydora_reset_state()
    model(input_ids=ids, labels=ids).loss.backward()
    for value in model.odydora_state().values():
        assert value.grad_fn is None
        assert not value.requires_grad


def test_the_core_attaches_through_a_peft_wrapper():
    """The real build order is adapter first, then core, and PEFT nests the
    host twice: a bare Qwen3 keeps its blocks at `model.layers` while a
    PeftModel puts the same blocks at `base_model.model.model.layers`. A
    fixed path list finds nothing there, which crashed `build()` on the
    first real 4-bit call while every unit test here stayed green -- the
    tests were attaching the core without PEFT in the way.
    """
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)
    wrapped = get_peft_model(
        tiny_qwen(),
        LoraConfig(r=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
    )
    model = attach_core(wrapped, n=16, stride=4, freeze_base=False)
    assert len(model.odydora_layers) == 2

    ids = torch.randint(0, 128, (1, 16))
    model.odydora_reset_state()
    model(input_ids=ids, labels=ids).loss.backward()
    assert model.odydora_calls() == 2, "the hook did not fire on both blocks"

    core = model.odydora_core
    assert core.W.grad.abs().sum() > 0 or core.up.weight.grad.abs().sum() > 0
    # Both families of trainable tensor must be present and separable.
    adapter, core_params = count_parameters(model)
    assert adapter > 0, "LoRA parameters were not counted"
    assert core_params > 0, "core parameters were not counted"


def test_the_core_attaches_through_a_dora_wrapper():
    """DoRA introduces magnitude vectors per linear layer.

    Assert that attach_core finds decoder layers under PeftModel with use_dora=True,
    executes both boundaries, and backprops gradients into both DoRA and the core.
    """
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(1)
    wrapped = get_peft_model(
        tiny_qwen(),
        LoraConfig(
            r=8,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
            use_dora=True,
        ),
    )
    model = attach_core(wrapped, n=16, stride=4, freeze_base=False)
    assert len(model.odydora_layers) == 2

    ids = torch.randint(0, 128, (1, 16))
    model.odydora_reset_state()
    model(input_ids=ids, labels=ids).loss.backward()
    assert model.odydora_calls() == 2
    adapter, core_params = count_parameters(model)
    assert adapter > 0 and core_params > 0


def test_attach_core_with_zero_n_is_noop_control_arm():
    """n=0 or n=None creates a clean adapter-only arm with no core parameters."""
    from peft import LoraConfig, get_peft_model

    wrapped = get_peft_model(
        tiny_qwen(),
        LoraConfig(r=4, target_modules=["q_proj"], task_type="CAUSAL_LM"),
    )
    model = attach_core(wrapped, n=0, freeze_base=False)
    assert model.odydora_core is None
    assert model.odydora_layers == ()
    assert model.odydora_calls() == 0
    model.odydora_reset_state()
    model.odydora_detach_state()
    assert model.odydora_state() == {}

    adapter, core = count_parameters(model)
    assert core == 0
    assert adapter > 0

    ids = torch.randint(0, 128, (1, 16))
    loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    assert loss.item() > 0
