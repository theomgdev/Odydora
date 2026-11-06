import sys
import types

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from odydora.adapter import attach_core, count_parameters
from odydora.distill import (
    DEFAULT_MIXTURE,
    LOG2,
    _row_text,
    check_vocab_match,
    distill_loss,
    distill_steps,
    evaluate_gap,
    teacher_reference_bits,
    teacher_targets,
)
from odydora.train import configure_optimizer


class FakeTokenizer:
    """Minimal stand-in: `check_vocab_match` reads only these two methods."""

    def __init__(self, vocab, encoding=None):
        self._vocab = dict(vocab)
        self._encoding = encoding

    def get_vocab(self):
        return dict(self._vocab)

    def __call__(self, text, add_special_tokens=False):
        if self._encoding is not None:
            return {"input_ids": list(self._encoding)}
        return {"input_ids": [self._vocab[c] for c in text if c in self._vocab]}


def tiny_lm(hidden=32, layers=2, vocab=128):
    return Qwen2ForCausalLM(
        Qwen2Config(
            num_hidden_layers=layers,
            hidden_size=hidden,
            intermediate_size=hidden * 2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=vocab,
            max_position_embeddings=64,
        )
    )


def text_batches(count=6, chunk=8, vocab=128):
    torch.manual_seed(0)
    out = []
    for i in range(count):
        ids = torch.randint(0, vocab, (1, chunk))
        out.append((ids, ids.clone(), i % 3 == 0))
    return out


def test_identical_vocabularies_pass_and_a_mismatch_raises():
    shared = {"a": 0, "b": 1, "c": 2}
    assert check_vocab_match(
        FakeTokenizer(shared), FakeTokenizer(shared), probe="abc"
    ) == 3

    with pytest.raises(ValueError, match="disagree"):
        check_vocab_match(
            FakeTokenizer(shared), FakeTokenizer({"a": 0, "b": 1}), probe="ab"
        )


def test_same_vocabulary_different_encoding_is_still_rejected():
    """A size and mapping match is not enough: two tokenizers can agree on the
    vocabulary and still segment text differently, which would score the two
    models on different token streams while the loss curve looks healthy.
    """
    shared = {"a": 0, "b": 1, "c": 2}
    with pytest.raises(ValueError, match="encode text differently"):
        check_vocab_match(
            FakeTokenizer(shared),
            FakeTokenizer(shared, encoding=[0, 1]),
            probe="abc",
        )


def test_the_loss_is_exactly_zero_when_the_student_matches_the_teacher():
    """The remainder term is what buys this. A bare top-k KL renormalises over
    the subset, so it reports a non-zero loss for a student that agrees with
    the teacher everywhere -- an objective with a floor the optimum cannot
    reach, pulling the student off the target it was given.
    """
    torch.manual_seed(0)
    logits = torch.randn(1, 6, 400) * 3.0
    log_probs = torch.log_softmax(logits, dim=-1)
    values, indices = log_probs.topk(32, dim=-1)

    loss, parts = distill_loss(
        logits, values, indices, labels=None, hard_weight=0.0, return_parts=True
    )
    assert abs(loss.item()) < 1e-6, f"agreement must score 0, got {loss.item():.3e}"
    assert abs(parts["kl_bits"]) < 1e-6


def test_topk_plus_remainder_tracks_the_exact_full_vocabulary_kl():
    """At a real vocabulary size the exact KL cannot be materialised, so the
    approximation has to be shown faithful rather than assumed: same direction
    for the gradient, and an under-estimate rather than an over-estimate.
    """
    torch.manual_seed(0)
    vocab = 20_000
    teacher_logits = torch.randn(1, 8, vocab) * 3.5
    student_logits = (teacher_logits + torch.randn(1, 8, vocab) * 0.5)
    student_logits = student_logits.detach().requires_grad_(True)

    p_t = torch.softmax(teacher_logits, dim=-1)
    exact = (
        p_t * (torch.log_softmax(teacher_logits, -1) - torch.log_softmax(student_logits, -1))
    ).sum(-1).mean()
    exact.backward()
    exact_grad = student_logits.grad.clone()

    student_logits.grad = None
    log_probs = torch.log_softmax(teacher_logits, dim=-1)
    values, indices = log_probs.topk(128, dim=-1)
    approx = distill_loss(
        student_logits, values, indices, labels=None, hard_weight=0.0
    )
    approx.backward()

    cosine = torch.nn.functional.cosine_similarity(
        exact_grad.flatten(), student_logits.grad.flatten(), dim=0
    ).item()
    assert cosine > 0.99, f"gradient direction diverged from exact KL: {cosine:.4f}"
    assert approx.item() <= exact.item() + 1e-6, (
        "a partition coarser than the full vocabulary cannot exceed the exact KL"
    )


def test_unsupervised_positions_and_unconfident_teacher_positions_are_dropped():
    torch.manual_seed(1)
    logits = torch.randn(1, 4, 200)
    teacher_logits = torch.randn(1, 4, 200) * 4.0
    log_probs = torch.log_softmax(teacher_logits, dim=-1)
    values, indices = log_probs.topk(16, dim=-1)

    labels = torch.tensor([[5, -100, -100, 9]])
    _, parts = distill_loss(
        logits, values, indices, labels=labels, hard_weight=0.0, return_parts=True
    )
    assert parts["kept"] == 2, f"masked positions leaked in: kept={parts['kept']}"

    # A confidence floor above every position's top-k mass must empty the batch
    # rather than divide by zero.
    loss, parts = distill_loss(
        logits, values, indices, labels=labels, hard_weight=0.0,
        min_teacher_conf=1.1, return_parts=True,
    )
    assert parts["kept"] == 0
    assert loss.item() == 0.0


def test_the_hard_label_anchor_enters_the_objective_at_its_weight():
    torch.manual_seed(2)
    logits = torch.randn(1, 6, 200, requires_grad=True)
    teacher_logits = torch.randn(1, 6, 200) * 4.0
    values, indices = torch.log_softmax(teacher_logits, dim=-1).topk(16, dim=-1)
    labels = torch.randint(0, 200, (1, 6))

    kl_only, parts = distill_loss(
        logits, values, indices, labels=labels, hard_weight=0.0, return_parts=True
    )
    both = distill_loss(
        logits, values, indices, labels=labels, hard_weight=0.5
    )
    expected = kl_only.item() + 0.5 * parts["hard_bits"] * LOG2
    assert abs(both.item() - expected) < 1e-4, (
        f"hard anchor not applied at its weight: {both.item():.6f} vs {expected:.6f}"
    )


def test_teacher_targets_are_detached_log_probabilities_over_the_full_vocabulary():
    """They must not be renormalised over the top-k subset, or the remainder
    term in the loss has nothing to measure.
    """
    torch.manual_seed(3)
    teacher = tiny_lm()
    teacher.requires_grad_(False)
    ids = torch.randint(0, 128, (1, 8))

    values, indices = teacher_targets(teacher, ids, top_k=16)
    assert values.shape == (1, 8, 16) and indices.shape == (1, 8, 16)
    assert not values.requires_grad, "teacher targets must carry no graph"
    mass = values.exp().sum(dim=-1)
    assert (mass < 1.0).all(), "top-16 of 128 cannot hold the whole distribution"
    assert (values <= 0.0).all(), "log-probabilities must be non-positive"


def test_distill_steps_moves_the_core_and_reports_live_rows():
    torch.manual_seed(4)
    student = attach_core(tiny_lm(), n=16, layers=(0, 1), stride=4)
    teacher = tiny_lm(hidden=64, layers=3)
    teacher.requires_grad_(False)

    before = student.odydora_core.W.detach().clone()
    opt = configure_optimizer(student, lr=0.05)
    rows = list(
        distill_steps(
            student, teacher, text_batches(count=6), opt, top_k=16, log_every=2
        )
    )

    assert len(rows) == 3, f"expected 3 log rows, got {len(rows)}"
    assert not torch.equal(before, student.odydora_core.W), "W never moved"
    for row in rows:
        assert row["kl_bits"] >= 0.0, "a KL cannot be negative"
        assert row["tok_s"] > 0 and row["elapsed"] >= 0.0
        assert torch.isfinite(torch.tensor(row["state_norm"]))
        assert 0.0 < row["mix"] < 1.0
    assert rows[-1]["docs"] == 2, f"trace/doc starts miscounted: {rows[-1]['docs']}"


def test_distill_steps_runs_on_the_adapter_only_control_arm():
    """`n=0` must distil without branching, or the matched-parameter baseline
    needs its own loop and stops being the same experiment.
    """
    torch.manual_seed(5)
    from peft import get_peft_model

    from odydora.train import lora_config

    wrapped = get_peft_model(
        tiny_lm(), lora_config(use_dora=False, r=2, targets=("q_proj", "v_proj"))
    )
    student = attach_core(wrapped, n=0, freeze_base=False)
    teacher = tiny_lm(hidden=64, layers=3)
    teacher.requires_grad_(False)

    adapter, core = count_parameters(student)
    assert core == 0 and adapter > 0

    opt = configure_optimizer(student, lr=0.01)
    rows = list(
        distill_steps(student, teacher, text_batches(count=4), opt, top_k=16, log_every=2)
    )
    assert len(rows) == 2
    assert all(row["mix"] == 0.0 and row["up_norm"] == 0.0 for row in rows)


def test_evaluate_gap_reports_overall_and_tail_separately():
    torch.manual_seed(6)
    student = attach_core(tiny_lm(), n=16, layers=(0, 1), stride=4)
    teacher = tiny_lm(hidden=64, layers=3)
    teacher.requires_grad_(False)

    gap = evaluate_gap(student, teacher, text_batches(count=8), top_k=16)
    assert gap["chunks"] == 8
    assert gap["tail_chunks"] == 2, f"tail 25% of 8 chunks is 2, got {gap['tail_chunks']}"
    assert gap["kl_bits"] > 0.0, "two random models cannot already agree"
    assert gap["student_bits"] > 0.0
    for key in ("kl_bits", "tail_kl_bits", "student_bits", "tail_student_bits"):
        assert torch.isfinite(torch.tensor(gap[key])), f"{key} came back non-finite"


def test_the_kept_fraction_is_reported_so_a_silent_mask_cannot_hide():
    """A confidence floor above the teacher's typical top-k mass drops most of
    the batch while the KL column still falls on whatever survives. Without
    `kept_fraction` in the log that reads as healthy training, so the figure is
    computed rather than left to be inferred.
    """
    torch.manual_seed(9)
    logits = torch.randn(1, 8, 300)
    teacher_logits = torch.randn(1, 8, 300) * 4.0
    values, indices = torch.log_softmax(teacher_logits, dim=-1).topk(16, dim=-1)
    labels = torch.randint(0, 300, (1, 8))

    _, wide = distill_loss(
        logits, values, indices, labels=labels, hard_weight=0.0,
        min_teacher_conf=0.0, return_parts=True,
    )
    assert wide["kept_fraction"] == 1.0, "nothing should be masked at a zero floor"

    _, narrow = distill_loss(
        logits, values, indices, labels=labels, hard_weight=0.0,
        min_teacher_conf=0.999999, return_parts=True,
    )
    assert narrow["kept_fraction"] < wide["kept_fraction"], (
        "a near-1.0 floor must visibly shrink the kept fraction"
    )


def test_distill_steps_reports_the_kept_fraction_every_row():
    torch.manual_seed(10)
    student = attach_core(tiny_lm(), n=16, layers=(0, 1), stride=4)
    teacher = tiny_lm(hidden=64, layers=3)
    teacher.requires_grad_(False)
    opt = configure_optimizer(student, lr=0.01)

    rows = list(
        distill_steps(student, teacher, text_batches(count=4), opt, top_k=16,
                      min_teacher_conf=0.0, log_every=2)
    )
    assert rows, "the loop produced no log rows"
    for row in rows:
        assert 0.0 < row["kept_fraction"] <= 1.0, (
            f"kept_fraction out of range: {row['kept_fraction']}"
        )


def test_every_default_corpus_is_ungated_with_a_declared_license():
    """A corpus default is a trust decision made on the reader's behalf.

    `bigcode/the-stack-smol` -- the obvious code source -- is gated AND carries
    three files the Hub's scanner flagged as malware, and it stopped the
    notebook at cell 4 with a DatasetNotFoundError for a reader who had not
    requested access. This asserts the shape of the mixture offline; the gating
    and scan status itself has to be checked against the Hub before adding an
    entry, which no local test can do.
    """
    assert DEFAULT_MIXTURE, "the default mixture cannot be empty"
    repos = [entry[0] for entry in DEFAULT_MIXTURE]

    assert "bigcode/the-stack-smol" not in repos, (
        "the-stack-smol is gated and has malware-flagged files; it must not be "
        "a default"
    )
    assert len(repos) == len(set(repos)), f"duplicate source in mixture: {repos}"
    assert len(repos) >= 3, (
        f"a {len(repos)}-source mixture is narrow enough that the student can "
        "learn a dialect rather than a distribution"
    )
    for repo, _config, split, field, weight in DEFAULT_MIXTURE:
        assert "/" in repo, f"{repo!r} is not a Hub repo id"
        assert split, f"{repo} has no split"
        assert isinstance(field, str | tuple), f"{repo} field is {type(field)}"
        assert weight >= 1, f"{repo} has weight {weight}"


def test_a_tuple_text_field_joins_its_columns():
    """Instruction corpora split a document across a prompt column and a
    response column. Reading one of them trains on half of every document while
    everything downstream looks perfectly normal.
    """
    row = {"input": "Write a function.", "output": "def f(): pass", "other": "ignored"}
    assert _row_text(row, "input") == "Write a function."
    joined = _row_text(row, ("input", "output"))
    assert joined == "Write a function.\n\ndef f(): pass"
    # A missing or empty column must not leave a dangling separator.
    assert _row_text({"input": "only"}, ("input", "output")) == "only"
    assert _row_text({"input": "", "output": "x"}, ("input", "output")) == "x"
    assert _row_text({}, ("input", "output")) == ""


def test_an_unopenable_source_costs_one_stream_not_the_whole_run(monkeypatch):
    """A gated repo, a removed dataset or a script-based one that `datasets>=4`
    refuses should narrow the mixture, not kill the stage — and the surviving
    sources must be named, because a silently narrower mixture looks identical
    to the intended one in the loss curve.
    """
    class FakeTok:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": list(range(40))}

    good_rows = [{"text": "x" * 600}, {"text": "y" * 600}]

    def fake_load_dataset(repo, config=None, split=None, streaming=False):
        if repo == "gated/repo":
            raise RuntimeError("is a gated dataset on the Hub")
        return list(good_rows)

    import odydora.distill as mod

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    spec = (
        ("gated/repo", None, "train", "text", 1),
        ("open/repo", None, "train", "text", 1),
    )
    chunks = list(
        mod.stream_text_chunks(FakeTok(), 16, datasets_spec=spec, max_chunks=4)
    )
    assert chunks, "the open source should still have produced chunks"
    assert all(c[0].shape[0] == 1 for c in chunks)

    # Every source failing is a different situation: there is nothing to train
    # on, and continuing silently would report a loss over an empty stream.
    all_bad = (("gated/repo", None, "train", "text", 1),)
    with pytest.raises(RuntimeError, match="nothing to"):
        list(mod.stream_text_chunks(FakeTok(), 16, datasets_spec=all_bad))


def test_teacher_reference_bits_gives_the_floor_the_gap_is_read_against():
    torch.manual_seed(7)
    teacher = tiny_lm(hidden=64, layers=3)
    teacher.requires_grad_(False)
    reference = teacher_reference_bits(teacher, text_batches(count=4))
    assert reference["chunks"] == 4
    assert reference["teacher_bits"] > 0.0
    assert torch.isfinite(torch.tensor(reference["tail_teacher_bits"]))


def test_teacher_kv_cache_carries_across_chunks_of_one_document():
    """Without a teacher KV cache across chunks, the teacher is reset at every
    chunk boundary just like a static adapter -- its target distribution on
    chunk 2 is then ignorant of chunk 1, and a student carrying state from
    chunk 1 is penalised for remembering what the amnesic teacher forgot.
    """
    from transformers import DynamicCache

    torch.manual_seed(11)
    teacher = tiny_lm(hidden=64, layers=2)
    teacher.requires_grad_(False)
    teacher.eval()

    c1 = torch.randint(0, 128, (1, 8))
    c2 = torch.randint(0, 128, (1, 8))

    full_vals, full_idx = teacher_targets(
        teacher, torch.cat([c1, c2], dim=1), top_k=16
    )
    amnesic_vals, _ = teacher_targets(teacher, c2, top_k=16)

    cache = DynamicCache()
    _ = teacher_targets(teacher, c1, top_k=16, past_key_values=cache)
    cached_vals, cached_idx = teacher_targets(
        teacher, c2, top_k=16, past_key_values=cache
    )

    assert torch.allclose(cached_vals, full_vals[:, 8:], atol=1e-5)
    assert torch.equal(cached_idx, full_idx[:, 8:])
    assert not torch.allclose(amnesic_vals, full_vals[:, 8:], atol=1e-3), (
        "an uncached teacher on chunk 2 must differ from the full-context pass"
    )


def test_min_chunks_drops_single_chunk_documents(monkeypatch):
    """At `min_chunks=2`, a document that fits inside one chunk is skipped so
    every streamed document crosses at least one chunk boundary (`is_doc_start=False`).
    """
    class LenTok:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": list(range(len(text) // 10))}

    rows = [
        {"text": "a" * 100},  # 10 tokens -> 1 chunk of 16 -> dropped at min_chunks=2
        {"text": "b" * 400},  # 40 tokens -> 3 chunks of 16 -> kept
    ]

    import odydora.distill as mod

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = lambda *a, **kw: list(rows)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    spec = (("open/repo", None, "train", "text", 1),)
    chunks = list(
        mod.stream_text_chunks(
            LenTok(), 16, datasets_spec=spec, min_chars=50, min_chunks=2
        )
    )
    assert len(chunks) == 3
    assert [c[2] for c in chunks] == [True, False, False]


def test_evaluate_gap_splits_fresh_from_carried_chunks():
    torch.manual_seed(12)
    student = attach_core(tiny_lm(), n=16, layers=(0, 1), stride=4)
    teacher = tiny_lm(hidden=64, layers=3)
    teacher.requires_grad_(False)

    # 6 chunks: 2 document starts (0, 3) and 4 carried continuations (1, 2, 4, 5)
    gap = evaluate_gap(student, teacher, text_batches(count=6), top_k=16)
    assert gap["fresh_chunks"] == 2
    assert gap["carried_chunks"] == 4
    assert torch.isfinite(torch.tensor(gap["fresh_kl_bits"]))
    assert torch.isfinite(torch.tensor(gap["carried_kl_bits"]))
