import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from odydora.adapter import attach_core, count_parameters
from odydora.train import (
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
    model = attach_core(tiny_qwen(), n=16, layers=(0, 1), window=4)
    adapter, core = count_parameters(model)
    assert adapter == 0, "no adapter applied yet"
    # W minus its diagonal, down, up, the RMSNorm gain.
    expected = (16 * 16 - 16) + 16 * 32 + 32 * 16 + 16
    assert core == expected, f"core count {core} != {expected}"


def test_matched_rank_finds_a_rank_within_tolerance():
    model = attach_core(tiny_qwen(), n=16, layers=(0, 1), window=4)
    _, core = count_parameters(model)
    rank, got = matched_rank(tiny_qwen, core, use_dora=True)
    assert rank >= 1
    assert abs(got - core) / core < 0.35, f"rank {rank} gave {got} against {core}"


def test_training_moves_the_core_and_the_loss():
    torch.manual_seed(1)
    model = attach_core(tiny_qwen(), n=16, layers=(0, 1), window=4)
    core = model.odydora_core
    before = core.W.detach().clone()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.05)
    log = train_steps(model, batches(), opt, log_every=2)
    assert len(log) == 3, f"expected 3 log rows, got {len(log)}"
    assert not torch.equal(before, core.W), "W never moved"
    assert log[0]["loss"] != log[-1]["loss"], "loss never changed"
    for row in log:
        assert torch.isfinite(torch.tensor(row["state_norm"])), "state went non-finite"
        assert row["tok_s"] > 0


def test_state_resets_on_a_trace_start_and_carries_otherwise():
    torch.manual_seed(2)
    model = attach_core(tiny_qwen(), n=16, layers=(0, 1), window=4)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.0)
    seen = []
    train_steps(model, batches(count=4), opt, log_every=1,
                on_step=lambda i, start, state: seen.append((start, state)))
    assert seen[0][0] is True
    # A trace start must reset, so the state entering a start chunk is fresh.
    assert seen[0][1] is None or torch.count_nonzero(seen[0][1]) == 0
    assert seen[1][1] is not None, "state must carry into a non-start chunk"


def test_checkpoint_round_trips_weights_and_the_corpus_cursor(tmp_path):
    torch.manual_seed(3)
    model = attach_core(tiny_qwen(), n=16, layers=(0, 1), window=4)
    model.odydora_core.W.data.add_(0.123)
    path = tmp_path / "ckpt.pt"
    save(model, path, step=17, traces=5, tokens=4096)
    fresh = attach_core(tiny_qwen(), n=16, layers=(0, 1), window=4)
    meta = restore(fresh, path)
    assert meta["step"] == 17 and meta["traces"] == 5 and meta["tokens"] == 4096
    assert torch.allclose(fresh.odydora_core.W, model.odydora_core.W)
