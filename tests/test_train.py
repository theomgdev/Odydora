import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from odydora.adapter import attach_core, count_parameters
from odydora.train import (
    configure_optimizer,
    lora_config,
    matched_rank,
    restore,
    save,
    train_steps,
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


def batches(count=6, chunk=8):
    torch.manual_seed(0)
    out = []
    for i in range(count):
        ids = torch.randint(0, 128, (1, chunk))
        out.append((ids, ids.clone(), i % 3 == 0))
    return out


def test_dora_is_requested_and_lora_is_the_fallback():
    assert lora_config(use_dora=True).use_dora is True
    assert lora_config(use_dora=False).use_dora is False
    assert lora_config(r=16).r == 16
    assert lora_config(r=16).lora_alpha == 32


def test_parameter_count_splits_adapter_from_core_and_excludes_the_diagonal():
    model = attach_core(tiny_qwen(), n=16, layers=(0, 1), stride=4)
    adapter, core = count_parameters(model)
    assert adapter == 0, "no adapter applied yet"
    # W minus its diagonal, down, up, the RMSNorm gain.
    # W minus diagonal, down, up, RMSNorm gain, and the mixing logit.
    expected = (16 * 16 - 16) + 16 * 32 + 32 * 16 + 16 + 1
    assert core == expected, f"core count {core} != {expected}"


def test_matched_rank_finds_a_rank_within_tolerance():
    model = attach_core(tiny_qwen(), n=16, layers=(0, 1), stride=4)
    _, core = count_parameters(model)
    rank, got = matched_rank(tiny_qwen, core, use_dora=True)
    assert rank >= 1
    assert abs(got - core) / core < 0.35, f"rank {rank} gave {got} against {core}"


def test_training_moves_the_core_and_the_loss():
    torch.manual_seed(1)
    model = attach_core(tiny_qwen(), n=16, layers=(0, 1), stride=4)
    core = model.odydora_core
    before = core.W.detach().clone()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.05)
    log = list(train_steps(model, batches(), opt, log_every=2))
    assert len(log) == 3, f"expected 3 log rows, got {len(log)}"
    assert not torch.equal(before, core.W), "W never moved"
    assert log[0]["loss"] != log[-1]["loss"], "loss never changed"
    for row in log:
        assert torch.isfinite(torch.tensor(row["state_norm"])), "state went non-finite"
        assert row["tok_s"] > 0


def test_state_resets_on_a_trace_start_and_carries_otherwise():
    torch.manual_seed(2)
    model = attach_core(tiny_qwen(), n=16, layers=(0, 1), stride=4)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.0)
    seen = []
    # Must be consumed: train_steps is a generator, so an un-drained call runs
    # no steps at all and this test would pass on an empty `seen`.
    list(train_steps(model, batches(count=4), opt, log_every=1,
                     on_step=lambda i, start, state: seen.append((start, state))))
    assert len(seen) == 4, f"expected 4 steps, got {len(seen)}"
    assert seen[0][0] is True
    # A trace start must reset, so the state entering a start chunk is fresh.
    assert seen[0][1] is None or torch.count_nonzero(seen[0][1]) == 0
    assert seen[1][1] is not None, "state must carry into a non-start chunk"


def test_checkpoint_round_trips_weights_and_the_corpus_cursor(tmp_path):
    torch.manual_seed(3)
    model = attach_core(tiny_qwen(), n=16, layers=(0, 1), stride=4)
    model.odydora_core.W.data.add_(0.123)
    path = tmp_path / "ckpt.pt"
    save(model, path, step=17, traces=5, tokens=4096)
    assert not (tmp_path / "ckpt.pt.tmp").exists(), "atomic save left .tmp behind"
    fresh = attach_core(tiny_qwen(), n=16, layers=(0, 1), stride=4)
    meta = restore(fresh, path)
    assert meta["step"] == 17 and meta["traces"] == 5 and meta["tokens"] == 4096
    assert torch.allclose(fresh.odydora_core.W, model.odydora_core.W)


def test_configure_optimizer_separates_families_and_scales_core_lr():
    """Modulation parameters (alpha, RMSNorm gain, DoRA magnitudes) must have
    weight_decay=0.0 because zero is not their neutral point; the shared core
    receives `lr * core_lr_scale` and `alpha` receives `5x` more to offset the
    sigmoid Jacobian.
    """
    from peft import get_peft_model

    wrapped = get_peft_model(
        tiny_qwen(), lora_config(use_dora=True, r=4, targets=("q_proj", "v_proj"))
    )
    model = attach_core(wrapped, n=16, stride=4, freeze_base=False)
    opt = configure_optimizer(model, lr=2e-4, core_lr_scale=5.0, weight_decay=0.01)

    by_name = {g["group_name"]: g for g in opt.param_groups}
    assert set(by_name) == {
        "adapter_decay",
        "adapter_no_decay",
        "core_decay",
        "core_no_decay",
        "core_alpha",
    }
    assert by_name["adapter_decay"]["lr"] == 2e-4
    assert by_name["adapter_decay"]["weight_decay"] == 0.01
    assert by_name["adapter_no_decay"]["weight_decay"] == 0.0
    assert abs(by_name["core_decay"]["lr"] - 1e-3) < 1e-12
    assert by_name["core_decay"]["weight_decay"] == 0.01
    assert by_name["core_no_decay"]["weight_decay"] == 0.0
    assert abs(by_name["core_alpha"]["lr"] - 5e-3) < 1e-12
    assert by_name["core_alpha"]["weight_decay"] == 0.0


def test_train_steps_tracks_actual_traces_consumed():
    """Each `is_trace_start=True` increments `traces` in the yielded row so an
    interrupted run saves the true trace cursor rather than the target count.
    """
    model = attach_core(tiny_qwen(), n=16, layers=(0, 1), stride=4)
    opt = configure_optimizer(model, lr=1e-3)
    # 6 chunks where chunks 0 and 3 start new traces -> 2 traces total
    rows = list(train_steps(model, batches(count=6, chunk=8), opt, log_every=2))
    assert [r["traces"] for r in rows] == [1, 2, 2]


def test_package_root_exports_public_api():
    import odydora

    for symbol in (
        "RecurrentCore",
        "attach_core",
        "build",
        "configure_optimizer",
        "train_steps",
        "evaluate",
        "save",
        "restore",
        "without_odydora",
    ):
        assert hasattr(odydora, symbol), f"odydora.{symbol} missing from top-level package"
