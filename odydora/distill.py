"""Distil a larger teacher's next-token distribution into the student's core.

The claim this module exists to test: a small frozen host steered by a shared
cross-layer recurrent state can match a *larger* model's output distribution
far more cheaply, in trainable parameters, than any spatial adapter of the
same budget. So the target is not a corpus label, it is the teacher's whole
distribution over the vocabulary: the teacher gives probabilities, the student
learns probabilities.

Three facts shape everything here.

**The vocabularies must be identical, and that has to be checked rather than
assumed.** Qwen3-0.6B through Qwen3-14B share one tokenizer (verified: same
151,669 entries, byte-identical ids on a probe string), so a token index means
the same thing in both models and the KL is well defined position by position.
Across families it is not, and a silent mismatch trains the student toward
noise with a plausible-looking loss curve, so `check_vocab_match` is a
precondition and not a courtesy.

**The full-vocabulary KL cannot be materialised at this scale.** One
`(1, 1024, 151936)` fp32 logit tensor is 622 MB, and the exact KL needs the
teacher's *and* the student's, plus the softmax intermediates. Instead the
teacher's distribution is compressed to its top `k` entries plus one lumped
remainder, and the student is scored on exactly that partition: `k` matched
probabilities and `1 - sum(p_k)` for everything else. Measured against the
exact full-vocabulary KL on peaked synthetic logits at `V=151,936`, `k=128`
reproduces the gradient direction at cosine similarity `0.998`, and the loss
is exactly `0.0` when student and teacher agree -- which a bare top-k KL
without the remainder term is not, because it renormalises over a subset and
reports a false zero. The teacher's `(k,)` values and indices are also all
that has to cross from teacher to student, so the teacher can be evicted from
memory between batches on a single-GPU runtime.

**Only positions the teacher is confident about are worth copying.** The
remainder mass is the teacher's own uncertainty on that token; when it is
large the teacher is guessing, and copying a guess spends student capacity on
noise. `min_teacher_conf` drops those positions from the loss.

The hard-label cross-entropy stays in the objective at a small weight. Pure
distillation lets the student drift to a distribution that is close to the
teacher's in KL while being wrong about the actual next token, and the hard
label is the cheapest available anchor against that.
"""

import time

import torch
import torch.nn.functional as F

LOG2 = 0.6931471805599453

# Expository web text, synthetic textbook prose, mathematics, code. Weights are
# relative row counts per round, not token fractions -- fineweb-edu rows are
# short and cosmopedia's are long, so equal weights would skew the token mix
# toward the synthetic half.
#
# Every entry is deliberately **ungated and malware-clean**. A default that
# sends a reader to an access request wall stops the notebook at cell 4 with a
# `DatasetNotFoundError`, and `bigcode/the-stack-smol` -- the obvious code
# source, and what this used to name -- is both gated (`gated="auto"`) and
# carries three files Hugging Face's scanner flagged as malware. A corpus
# default is a trust decision made on the reader's behalf, so the bar is
# ungated plus a clean scan plus a declared license, checked before shipping
# rather than after a user hits the wall.
#
# The `text_field` entry may be a tuple, in which case those columns are joined
# with a blank line: instruction corpora split the prompt and the response
# across two columns, and taking only one of them trains on half a document.
#
# Defined here, above every function, because `stream_text_chunks` reads it as a
# default and the notebook imports the name directly: a module-level constant
# placed after its consumers exists at call time but not at import time, which
# fails as an ImportError rather than at the call site.
DEFAULT_MIXTURE = (
    ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text", 3),
    ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", "train", "text", 2),
    ("open-web-math/open-web-math", None, "train", "text", 1),
    # NVIDIA, CC-BY-4.0, ungated. Replaces bigcode/the-stack-smol, which is
    # gated and has three files flagged as malware by the Hub's scanner.
    ("nvidia/OpenCodeInstruct", None, "train", ("input", "output"), 1),
)


def check_vocab_match(teacher_tokenizer, student_tokenizer, probe=None):
    """Raise unless the two tokenizers agree, so a KL across them is defined.

    Compares the vocabulary mapping itself, not just its size: two tokenizers
    can hold the same number of entries while disagreeing about which string
    each id denotes, and that disagreement is invisible in the loss curve --
    the student simply learns to predict the wrong token confidently.
    """
    teacher_vocab, student_vocab = teacher_tokenizer.get_vocab(), student_tokenizer.get_vocab()
    if teacher_vocab != student_vocab:
        only_teacher = set(teacher_vocab) - set(student_vocab)
        only_student = set(student_vocab) - set(teacher_vocab)
        raise ValueError(
            "teacher and student tokenizers disagree, so a per-token KL is "
            f"undefined: {len(teacher_vocab)} vs {len(student_vocab)} entries, "
            f"{len(only_teacher)} teacher-only, {len(only_student)} student-only"
        )
    probe = probe or "Hello <think> world! def f(x): return x**2\n\u00e7\u0131kt\u0131"
    teacher_ids = teacher_tokenizer(probe, add_special_tokens=False)["input_ids"]
    student_ids = student_tokenizer(probe, add_special_tokens=False)["input_ids"]
    if list(teacher_ids) != list(student_ids):
        raise ValueError(
            "tokenizers share a vocabulary but encode text differently "
            f"({len(teacher_ids)} vs {len(student_ids)} ids on the probe), so "
            "the two models would be scored on different token streams"
        )
    return len(teacher_vocab)


@torch.no_grad()
def teacher_targets(teacher, ids, top_k=128, attention_mask=None):
    """Compress the teacher's next-token distribution to (values, indices).

    Returns `log_probs` and `indices` of shape `(batch, time, top_k)`, in
    which `log_probs` are true log-probabilities over the *whole* vocabulary
    (`log_softmax` then `gather`), not renormalised over the top-k subset.
    Keeping them un-renormalised is what lets the student be scored on the
    lumped remainder as well, so the objective reaches zero exactly when the
    two distributions agree.

    This is the only thing the teacher contributes, it is `(batch, time, k)`
    rather than `(batch, time, V)` -- 128 against 151,936 -- and it is
    detached, so a runtime that cannot hold both models can produce these,
    free the teacher and then train.
    """
    teacher.eval()
    out = teacher(input_ids=ids, attention_mask=attention_mask)
    log_probs = torch.log_softmax(out.logits.float(), dim=-1)
    values, indices = log_probs.topk(top_k, dim=-1)
    return values.detach(), indices.detach()


def distill_loss(
    student_logits,
    teacher_log_probs,
    teacher_indices,
    labels=None,
    temperature=1.0,
    hard_weight=0.1,
    min_teacher_conf=0.0,
    return_parts=False,
):
    """Top-k-plus-remainder KL from teacher to student, in bits per token.

    The partition is `k + 1` events: each of the teacher's top-`k` tokens, and
    "anything else". Both sides are evaluated on it, so

        KL = sum_k p_t(k) * log(p_t(k) / p_s(k)) + p_t(rest) * log(p_t(rest) / p_s(rest))

    with the student's `p_s` taken against its *full* `logsumexp`, which is
    what makes the remainder term meaningful and the whole expression exactly
    `0.0` at agreement. Positions are masked out when `labels` says they are
    unsupervised (`-100`) or when the teacher's own top-`k` mass falls below
    `min_teacher_conf`, since the teacher is then guessing and the guess is
    not worth the student's capacity.

    Reported in bits so it is directly comparable with the bits/token the
    evaluation path already prints. `temperature` softens both sides
    symmetrically; the gradient is *not* rescaled by `T^2` because the loss
    here is reported, not just minimised, and the scale has to stay readable.
    """
    student_logits = student_logits.float()
    if temperature != 1.0:
        student_logits = student_logits / temperature
        teacher_log_probs = torch.log_softmax(teacher_log_probs / temperature, dim=-1)

    p_teacher = teacher_log_probs.exp()
    teacher_mass = p_teacher.sum(dim=-1)

    keep = torch.ones_like(teacher_mass, dtype=torch.bool)
    if labels is not None:
        keep &= labels != -100
    if min_teacher_conf > 0.0:
        keep &= teacher_mass >= min_teacher_conf

    kept = int(keep.sum())
    if kept == 0:
        # Every key the non-empty path returns must also appear here, or a
        # consumer that reads `kept_fraction` to notice this exact situation
        # raises a KeyError precisely when the mask has eaten the batch.
        zero = student_logits.sum() * 0.0
        empty = {
            "kl_bits": float("nan"),
            "hard_bits": float("nan"),
            "kept": 0,
            "kept_fraction": 0.0,
            "teacher_conf": float("nan"),
        }
        return (zero, empty) if return_parts else zero

    student_lse = torch.logsumexp(student_logits, dim=-1, keepdim=True)
    student_log_k = torch.gather(student_logits, -1, teacher_indices) - student_lse

    head = (p_teacher * (teacher_log_probs - student_log_k)).sum(dim=-1)

    # The lumped remainder. Clamped because both masses are sums of exponentials
    # and can round to slightly above 1.0 in float32, which would otherwise take
    # the log of a negative number at exactly the point the two agree.
    rest_teacher = (1.0 - teacher_mass).clamp(min=1e-9)
    rest_student = (1.0 - student_log_k.exp().sum(dim=-1)).clamp(min=1e-9)
    tail = rest_teacher * (rest_teacher.log() - rest_student.log())

    kl_nats = ((head + tail) * keep).sum() / kept
    loss = kl_nats
    hard_nats = torch.zeros((), device=student_logits.device, dtype=student_logits.dtype)

    # The hard cross-entropy is computed whenever labels exist, not only when it
    # is weighted in: it is also the diagnostic the training log prints, and a
    # reported figure that silently reads 0.0 because the weight is 0 is worse
    # than no figure at all -- it looks like a student that predicts perfectly.
    if labels is not None and (hard_weight > 0.0 or return_parts):
        shift_logits = student_logits[:, :-1].reshape(-1, student_logits.shape[-1])
        shift_labels = labels[:, 1:].reshape(-1)
        hard_nats = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
        if torch.isnan(hard_nats):
            hard_nats = torch.zeros_like(kl_nats)

    if hard_weight > 0.0 and labels is not None:
        # The anchor: KL alone admits a student that tracks the teacher's shape
        # while being wrong about the realised token.
        loss = loss + hard_weight * hard_nats

    if return_parts:
        return loss, {
            "kl_bits": kl_nats.item() / LOG2,
            "hard_bits": hard_nats.item() / LOG2,
            "kept": kept,
            # The fraction of positions that survived the mask. A confidence
            # floor that quietly discards most of the batch looks exactly like
            # healthy training in the loss column while almost nothing trains,
            # so the training log prints this rather than inferring it.
            "kept_fraction": kept / max(teacher_mass.numel(), 1),
            "teacher_conf": (teacher_mass * keep).sum().item() / kept,
        }
    return loss


def stream_text_chunks(
    tokenizer,
    chunk,
    datasets_spec=None,
    max_chunks=None,
    skip_chunks=0,
    min_chars=512,
    log_every=0,
):
    """Yield `(ids, labels, is_doc_start)` windows from a mixture of text corpora.

    The corpus choice is itself part of the experiment. Distilling on one
    narrow domain would measure domain transfer rather than capacity, and the
    student is a pretrained model whose breadth is the thing at risk -- so the
    default mixture is deliberately spread: expository web text, synthetic
    textbook prose, mathematics, and code. The round-robin interleave means no
    stage of training sees a long run of one domain, which is what turns a
    mixture into sequential fine-tuning with its usual forgetting.

    `datasets_spec` is `[(repo, config, split, text_field, weight), ...]`, where
    `text_field` is a column name or a tuple of them to join with a blank line.
    Every window is fully supervised: in distillation the teacher labels every
    position, so there is no mask to build and no unsupervised span to drop.
    `is_doc_start` resets the recurrent state at a document boundary, exactly
    as a trace boundary does in the agentic path.

    A source that cannot be opened is skipped with a named reason rather than
    killing the run: a gated repository, a removed dataset or a script-based one
    that `datasets>=4` no longer loads should cost the mixture one stream, not
    the whole stage. The surviving sources are listed so the reader knows what
    the run actually trained on, since a silently narrower mixture looks exactly
    like the intended one in the loss curve.
    """
    from datasets import load_dataset

    spec = datasets_spec or DEFAULT_MIXTURE
    streams, skipped = [], []
    for repo, config, split, field, weight in spec:
        label = f"{repo}" + (f" [{config}]" if config else "")
        try:
            rows = load_dataset(repo, config, split=split, streaming=True) if config \
                else load_dataset(repo, split=split, streaming=True)
            streams.append({"name": label, "it": iter(rows), "field": field,
                            "weight": max(int(weight), 1), "done": False})
        except Exception as exc:
            reason = type(exc).__name__
            skipped.append(f"{label} ({reason})")
            print(f"  corpus: SKIPPING {label} -- {reason}: {str(exc)[:120]}",
                  flush=True)

    if not streams:
        raise RuntimeError(
            "every corpus in the mixture failed to open, so there is nothing to "
            f"train on: {'; '.join(skipped)}"
        )
    if skipped:
        print(f"  corpus: training on {len(streams)} of {len(spec)} sources "
              f"({', '.join(s['name'] for s in streams)})", flush=True)

    produced, passed = 0, 0
    while any(not s["done"] for s in streams):
        for stream in streams:
            if stream["done"]:
                continue
            for _ in range(stream["weight"]):
                try:
                    row = next(stream["it"])
                except StopIteration:
                    stream["done"] = True
                    break
                text = _row_text(row, stream["field"])
                if len(text) < min_chars:
                    continue
                ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                for begin in range(0, len(ids), chunk):
                    window = ids[begin : begin + chunk]
                    # A window of one token has no predictable position after
                    # the causal shift, so it would contribute 0/0 to the hard
                    # cross-entropy exactly as an all-masked chunk does.
                    if len(window) <= 1:
                        continue
                    if passed < skip_chunks:
                        passed += 1
                        continue
                    tensor = torch.tensor([window], dtype=torch.long)
                    yield tensor, tensor.clone(), begin == 0
                    produced += 1
                    if log_every and produced % log_every == 0:
                        print(f"  corpus: {produced} chunks "
                              f"({stream['name']})", flush=True)
                    if max_chunks and produced >= max_chunks:
                        return


def _row_text(row, field):
    """One row's text, joining several columns when `field` is a tuple.

    Instruction corpora split a document across a prompt column and a response
    column; taking only one of them trains the student on half of every
    document while everything downstream looks normal.
    """
    if isinstance(field, str):
        return row.get(field) or ""
    parts = [str(row.get(name) or "").strip() for name in field]
    return "\n\n".join(part for part in parts if part)


def distill_steps(
    student,
    teacher,
    batches,
    optimizer,
    top_k=128,
    temperature=1.0,
    hard_weight=0.1,
    min_teacher_conf=0.0,
    log_every=10,
    grad_accum=1,
    clip=1.0,
    on_step=None,
):
    """Yield one log row every `log_every` chunks while distilling.

    A generator for the same reason the supervised loop is one: a caller
    printing these sees the first line after `log_every` steps rather than
    after the whole stage, and a Colab cell that prints nothing for twenty
    minutes is indistinguishable from a hung one.

    The teacher is queried under `no_grad` on the same chunk the student sees,
    so the two are always scored on identical token streams -- which matters
    more than it looks, because the student's recurrent state depends on the
    chunk order and a mismatched stream would silently decorrelate the two.
    The state is reset at a document boundary and carried detached otherwise,
    exactly as in the agentic path.
    """
    student.train()
    started = began = time.time()
    tokens, docs = 0, 0
    kl_window, hard_window, conf_window, kept_window = [], [], [], []

    for step, (ids, labels, is_doc_start) in enumerate(batches):
        device = next(student.parameters()).device
        ids, labels = ids.to(device), labels.to(device)

        if is_doc_start:
            student.odydora_reset_state()
            docs += 1
        if on_step is not None:
            on_step(step, is_doc_start)

        t_log_probs, t_indices = teacher_targets(teacher, ids, top_k=top_k)

        out = student(input_ids=ids)
        student.odydora_detach_state()

        loss, parts = distill_loss(
            out.logits,
            t_log_probs,
            t_indices,
            labels=labels,
            temperature=temperature,
            hard_weight=hard_weight,
            min_teacher_conf=min_teacher_conf,
            return_parts=True,
        )
        if parts["kept"] == 0 or torch.isnan(loss):
            continue

        (loss / grad_accum).backward()

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in student.parameters() if p.requires_grad], clip
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        kl_window.append(parts["kl_bits"])
        hard_window.append(parts["hard_bits"])
        conf_window.append(parts["teacher_conf"])
        kept_window.append(parts["kept_fraction"])
        tokens += ids.numel()

        if (step + 1) % log_every == 0:
            core = getattr(student, "odydora_core", None)
            layers = getattr(student, "odydora_layers", ())
            state = student.odydora_state().get(layers[-1]) if layers else None
            elapsed = max(time.time() - began, 1e-9)
            yield {
                "step": step + 1,
                "kl_bits": sum(kl_window) / len(kl_window),
                "hard_bits": sum(hard_window) / len(hard_window),
                "teacher_conf": sum(conf_window) / len(conf_window),
                "kept_fraction": sum(kept_window) / len(kept_window),
                "tok_s": tokens / elapsed,
                "tokens": tokens,
                "docs": docs,
                "state_norm": 0.0 if state is None else state.norm().item(),
                "mix": core.mix.detach().item() if core is not None else 0.0,
                "up_norm": core.up.weight.detach().norm().item() if core is not None else 0.0,
                "elapsed": time.time() - started,
            }
            kl_window, hard_window, conf_window, kept_window = [], [], [], []


@torch.no_grad()
def evaluate_gap(
    student,
    teacher,
    batches,
    top_k=128,
    tail_fraction=0.25,
    log_every=0,
    desc="gap",
):
    """How far the student's distribution sits from the teacher's, in bits.

    This is the number the whole claim turns on, and it is deliberately not
    the student's own cross-entropy: a student can lower its perplexity by
    sharpening toward the corpus while drifting *away* from the teacher, and
    perplexity cannot tell the two apart. The KL to the teacher can.

    Three figures come back. `kl_bits` is the mean gap. `tail_kl_bits` is the
    gap over the trailing chunks, where the recurrent state has history behind
    it -- a mechanism that works must show a larger improvement there than
    overall, and one that only improves the mean is doing something else.
    `student_bits` is the student's own hard cross-entropy, kept so the gap can
    be read against an absolute scale.

    `batches` must be a materialised list, since it is iterated once here and
    again by every arm being compared; a generator would silently hand the
    second arm an empty stream.
    """
    student.eval()
    rows = []
    total = len(batches) if hasattr(batches, "__len__") else None

    for step, (ids, labels, is_doc_start) in enumerate(batches):
        device = next(student.parameters()).device
        ids, labels = ids.to(device), labels.to(device)
        if is_doc_start:
            student.odydora_reset_state()

        t_log_probs, t_indices = teacher_targets(teacher, ids, top_k=top_k)
        out = student(input_ids=ids)
        _, parts = distill_loss(
            out.logits, t_log_probs, t_indices, labels=labels,
            hard_weight=1.0, return_parts=True,
        )
        if parts["kept"]:
            rows.append((parts["kl_bits"], parts["hard_bits"], parts["kept"]))

        if log_every and (step + 1) % log_every == 0:
            target = f"/{total}" if total else ""
            print(f"  {desc} chunk {step + 1}{target}...", flush=True)

    def weighted(selected, index):
        mass = sum(row[index] * row[2] for row in selected)
        seen = sum(row[2] for row in selected)
        return mass / seen if seen else float("nan")

    cut = int(len(rows) * (1 - tail_fraction))
    return {
        "kl_bits": weighted(rows, 0),
        "tail_kl_bits": weighted(rows[cut:], 0),
        "student_bits": weighted(rows, 1),
        "tail_student_bits": weighted(rows[cut:], 1),
        "chunks": len(rows),
        "tail_chunks": len(rows[cut:]),
    }


@torch.no_grad()
def teacher_reference_bits(teacher, batches, tail_fraction=0.25, log_every=0):
    """The teacher's own cross-entropy on the same chunks, in bits per token.

    The floor of the experiment. A student whose `student_bits` approaches this
    has matched the teacher's predictive quality on this corpus; the distance
    remaining is what the trainable budget has not yet bought. Without it,
    "the gap shrank" has no scale -- shrank toward what.
    """
    teacher.eval()
    rows = []
    for step, (ids, labels, _) in enumerate(batches):
        device = next(teacher.parameters()).device
        ids, labels = ids.to(device), labels.to(device)
        out = teacher(input_ids=ids)
        logits = out.logits.float()[:, :-1].reshape(-1, out.logits.shape[-1])
        targets = labels[:, 1:].reshape(-1)
        loss = F.cross_entropy(logits, targets, ignore_index=-100)
        kept = int((targets != -100).sum())
        if kept and not torch.isnan(loss):
            rows.append((loss.item() / LOG2, kept))
        if log_every and (step + 1) % log_every == 0:
            print(f"  teacher reference chunk {step + 1}...", flush=True)

    def weighted(selected):
        mass = sum(bits * kept for bits, kept in selected)
        seen = sum(kept for _, kept in selected)
        return mass / seen if seen else float("nan")

    cut = int(len(rows) * (1 - tail_fraction))
    return {
        "teacher_bits": weighted(rows),
        "tail_teacher_bits": weighted(rows[cut:]),
        "chunks": len(rows),
    }
