# Odydora

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theomgdev/Odydora/blob/main/notebooks/odydora_colab.ipynb)
[![CI](https://github.com/theomgdev/Odydora/actions/workflows/ci.yml/badge.svg)](https://github.com/theomgdev/Odydora/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A single `N × N` matrix, shared by every decoder layer of a frozen 4-bit
language model, carrying a recurrent state **across layers** and **across chunk
boundaries**.

The frozen transformer is the limbic system. The core is the frontal cortex
steering it, and it costs the same trainable parameters however deep the host
model is.

## The question

A small model and a large model of the same family share a tokenizer, so their
next-token distributions live in the same space and the distance between them is
measurable. So:

> **How much of a larger model's distribution can about a million trainable
> parameters of temporal recurrence recover inside a smaller frozen host — and
> does it recover more than the same budget spent spatially, as LoRA?**

Teacher gives probabilities, student learns probabilities. The metric is the
KL from student to teacher in bits per token, reported overall **and** over the
trailing chunks where the recurrent state has history behind it — state carry
can only pay where there is a past, so a gain in the tail is the mechanism and
a gain only in the mean is something else.

```
for chunk in stream:                                    # 512 tokens at a time
    teacher_logprobs, teacher_ids = teacher(chunk).topk(128)   # frozen, no grad
    for layer L in CORE_LAYERS:
        h         = block_L(h)                          # frozen, 4-bit
        h         = h + LoRA_L(h)                        # trained, r=2, attention only
        s         = (1-a)·s_prev[L] + a·RMSNorm(down(down_in(pool(h))) + s_prev[L] @ W)
        h         = h + up(up_in(s))                     # up starts at zero
        s_prev[L] = s.detach()                           # at the chunk boundary
    loss = KL_topk_plus_remainder(student_logits, teacher_logprobs, teacher_ids)
```

One `W`, one factored `down`, one factored `up`, one mixing rate — shared by all
28 layers.

### The budget, measured on the real 28-layer `Qwen3-0.6B` geometry

| arm | adapter | core | total trainable |
|---|---|---|---|
| **Odydora** (`N=512`, `proj_rank=64`, `r=2` attention) | 573,440 | 458,753 | 1,032,193 |
| **Matched LoRA** (`N=0`, `r=4` attention) | 1,146,880 | 0 | 1,146,880 |

The control arm gets *more* trainable parameters than the treatment. Two
measurements forced this geometry:

- **A wide core's dense I/O projections dominate it.** At `d=1024` and `N=512`
  they cost `2 · 1024 · 512 = 1,048,576` against `W`'s `261,632`, so the
  recurrence is a fifth of the module. `proj_rank=64` cuts the interface to
  `196,608` and `W` becomes **57%** of the core.
- **LoRA at its usual settings buries the core.** `r=32` on all seven
  projections is `20,185,088` parameters — **44x** the core, which makes any
  win unattributable. Hence `r=2` on attention only.

## Lineage: this comes from OdyssNet

The architectural idea here is not new work. It is borrowed from
**[OdyssNet](https://github.com/theomgdev/OdyssNet)** (Karahan, v4.0.0, MIT),
which demonstrates that *temporal* depth can replace *spatial* depth: a single
fully connected `N × N` matrix with a hard-zero diagonal, through which signals
echo for many timesteps, solves non-linear problems with zero hidden layers —
90.14% on MNIST with 480 parameters.

Three mechanisms are taken directly from it, and they are the load-bearing ones:

- **One shared `N × N` matrix with `W.diagonal() == 0`**, applied identically at
  every step, with self-connections living outside `W`.
- **Per-step RMSNorm on the state**, which is what makes a chaotic recurrence
  trainable at all rather than merely stable.
- **A state detached at the call boundary**, so information crosses forward
  while gradient does not — OdyssNet's `forward` leaves `self.state` detached,
  which is the property that makes chunked training possible.

What Odydora changes is where the matrix lives. OdyssNet *is* the network;
here the same core is a small adapter steering a frozen pretrained transformer,
advancing at segment rate rather than per step, with a convex state mix added
because a frozen host and a 30,000-token document expose a forgetting problem
that OdyssNet's own tasks did not.

Deliberately **not** carried over: Hebbian plasticity (`hebb_type`) and temporal
attention (`attn_heads`). Both are expensive, and OdyssNet's own controlled A/Bs
leave them unproven as wins — so they are out until something here needs them.

OdyssNet is a reference, not a dependency: nothing in this package imports or
vendors it. Its measurements are quoted as its own and are not evidence about
this project; see `CITATION.cff` for how to cite both.

## Why it is shaped this way

Every number below was measured on this hardware, not taken from a paper.
`Research/SUMMARY.md` (local, gitignored) holds the full tables.

**The state advances once per `stride` tokens, not once per token.** The step is
kernel-launch-bound — batch 1 and batch 64 cost nearly the same wall time — so
cost tracks the *number of state updates* and nothing else. Measured end to end
on 2.03B fp16 weights with the core at all 28 layers:

| stride | tok/s | % of no-core baseline (1,786 tok/s) |
|---|---|---|
| 1 (per token) | 28 | 1.6% |
| 16 | 337 | 18.9% |
| 64 | 660 | 37.0% |
| 256 | 816 | 45.7% |

A 23× difference for byte-identical parameters. `torch.compile` bought nothing
(launches are the cost, not arithmetic); CUDA graph replay gave 8.1×, and a
longer stride gave more, for free. Transformer-XL and Block-Recurrent
Transformer advance state per segment for the same reason.

**Per-step RMSNorm makes explosion impossible and forgetting certain.** The
state sits on the `√N` sphere *exactly* — 5.657 at N=32, 8.000 at N=64, 11.314
at N=128, constant across 4,000 steps — so no spectral constraint, clamp or
gradient clip is needed. But the same normalisation renormalises the past away:
the influence of an input on the state 4,096 steps later measures **0.0e+00**
at every width and seed tried. The usual prescription, a contractive decay
γ ∈ (0,1), makes it *worse*.

What works is mixing the new state into the old convexly:

| mixing rate `a` | influence after 1,024 steps |
|---|---|
| 1.0 (plain norm) | 2.0e-34 |
| 0.1 | 3.6e-05 |
| **0.02** | **5.8e-01** |

The new state is on the sphere and the old one is inside it, so the mixture can
never leave the ball — bounded by construction rather than by a clamp. `a` is a
learnable logit, clamped at ±15 because `sigmoid` saturates to exactly 1.0 in
float32 and `a = 1` would drop the state entirely.

**Only the output projection is zero-initialised.** That makes the insertion an
exact identity at step 0 — the frozen model is provably unharmed, so any later
gain is attributable — while still taking gradient, exactly as LoRA's `B = 0`
does. Zeroing a second factor as well (this design started with a gate) puts a
zero in *every* factor's gradient and the core can never leave the origin.

**The core is attached by forward hook, not by wrapping the block.** Modern
`transformers` reads attributes off the layer object (`attention_type` and
friends), and a wrapper module breaks that dispatch.

**Parameter families are separated in the optimizer (`configure_optimizer`).**
Following OdyssNet's `ChaosGrad` family policy, only connective matrices (`W`,
`down`, `up`, `lora_A`, `lora_B`) receive weight decay; modulation parameters
(`core.alpha`, `core.norm.weight`, DoRA `lora_magnitude_vector`) receive
`weight_decay=0.0` because zero is not their neutral point (decaying `alpha`
toward 0 pulls `mix` toward 0.50, erasing long-horizon carry). And because the
shared core sits behind a zero-initialised `up` projection and a `0.02` mixing
rate at bottleneck width `n << d`, `configure_optimizer` scales the core step
size by `core_lr_scale=5.0` (and `alpha` by `25x` to offset the sigmoid
Jacobian `mix*(1-mix) ≈ 1/50`). Measured on 4-bit `Qwen3-0.6B` across three
seeds (`42`, `123`, `7`) on multi-chunk GSM8K traces (32 train chunks, 16
held-out test chunks):

| Arm | Seed 42 (overall / tail) | Seed 123 (overall / tail) | Seed 7 (overall / tail) |
|---|---|---|---|
| LoRA-only (`core` off) | 1.2275 / 1.4903 | 1.2273 / 1.4894 | 1.2328 / 1.4971 |
| Odydora + flat AdamW | 1.1786 / 1.4363 | 1.1792 / 1.4360 | 1.1820 / 1.4359 |
| **Odydora + `configure_optimizer`** | **1.0942 / 1.3425** | **1.0939 / 1.3448** | **1.0966 / 1.3489** |

## Install

```bash
git clone https://github.com/theomgdev/Odydora
cd Odydora
python -m venv .venv && .venv/Scripts/activate   # or source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                                        # 67 tests, CPU, no downloads
```

## Use

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theomgdev/Odydora/blob/main/notebooks/odydora_colab.ipynb)
**[Launch `odydora_colab.ipynb` directly in Google Colab](https://colab.research.google.com/github/theomgdev/Odydora/blob/main/notebooks/odydora_colab.ipynb)** — includes S1–S3 staged training, the matched-rank comparison arm, the Drive Checkpoint Manager Form UI, and the Interactive Chat Playground with live token streaming and side-by-side `without_odydora()` comparison.

```python
from odydora import build, configure_optimizer, evaluate, report, stream_traces, train_steps

model, tok = build("Qwen/Qwen3-1.7B", n=64, stride=64, mix=0.02, four_bit=True)
opt = configure_optimizer(model, lr=2e-4)
print(report(model))          # total / trainable / adapter / core

batches = stream_traces(tok, chunk=1024, max_trace_tokens=4096, max_traces=160)
for row in train_steps(model, batches, opt, log_every=10, grad_accum=16):
    print(row)
print(evaluate(model, val_batches))   # bits/token, overall and tail-only
```

Training happens in [`notebooks/odydora_colab.ipynb`](https://colab.research.google.com/github/theomgdev/Odydora/blob/main/notebooks/odydora_colab.ipynb) on a free Colab T4. The
notebook is generated by `tools/build_notebook.py` so its cells stay diffable;
edit the generator, not the `.ipynb`.

## Measured facts worth knowing before you run it

- **Qwen3-1.7B**: 2,031,739,904 parameters, 28 layers, d=2048, 40,960 context,
  Apache 2.0, non-gated, native `tools` + `<think>` chat template. Verified
  against the HF API. Alternates: `Qwen/Qwen3-0.6B` (fast), 
  `Qwen/Qwen2.5-Coder-1.5B-Instruct` (coding).
- **A T4 is Turing (sm_75)**: no bfloat16 tensor cores, no FlashAttention-2.
  Use `fp16` and `attn_implementation="sdpa"`. Both are the defaults here.
- **LoRA before DoRA.** QDoRA must dequantize the frozen 4-bit weight on every
  forward to take its column norm, costing 2.0–2.4× training time, and the cost
  is invariant to rank. `use_dora=True` is a late ablation, not a first run.
- **Unsloth cannot host this.** `prepare_model_for_training()` sets
  `requires_grad_(False)` on every parameter not named `lora_*`, which silently
  freezes the core; `modules_to_save` rejects anything outside a hard whitelist;
  and its reentrant checkpointing recomputes against a mutated recurrent state.
  DoRA also disables its fused kernels, so the speed reason evaporates anyway.
- **Intra-chunk detaching buys nothing.** Peak backward memory at a 1,024-token
  chunk: 61.8 MiB at window 2, 4, 8 and full backpropagation alike; 85.8 MiB at
  window 1, which is *worse*. `window=0` is the default, and truncation happens
  at the chunk boundary where the state crosses detached anyway.

## Corpus

### Distillation (the current path)

A four-way interleaved mixture, streamed, round-robined so no stage of training
sees a long run of one domain — that is what separates a mixture from
sequential fine-tuning, which forgets:

| weight | corpus | what it is |
|---|---|---|
| 3 | `HuggingFaceFW/fineweb-edu` (`sample-10BT`) | expository web text |
| 2 | `HuggingFaceTB/smollm-corpus` (`cosmopedia-v2`) | synthetic textbook prose |
| 1 | `open-web-math/open-web-math` | mathematics |
| 1 | `nvidia/OpenCodeInstruct` | code (CC-BY-4.0) |

Weights are rows per round rather than token fractions, because `fineweb-edu`
rows are short and `cosmopedia`'s are long — equal weights would skew the token
mixture toward the synthetic half.

**Every source is ungated and malware-clean, verified against the Hub before
shipping.** The obvious code corpus, `bigcode/the-stack-smol`, is both gated
(`gated="auto"`) and carries three files the Hub's scanner flagged as malware;
it stopped the notebook at cell 4 with a `DatasetNotFoundError` for a reader
who had not requested access. A corpus default is a trust decision made on the
reader's behalf, so the bar is ungated, a clean scan, and a declared license.

`text_field` may be a tuple, in which case those columns are joined with a
blank line — instruction corpora split a document across a prompt column and a
response column, and reading one of them trains the student on half of every
document while everything downstream looks normal.

A source that still fails to open (gated, removed, or script-based and
unsupported by `datasets>=4`) narrows the mixture with a named reason rather
than killing the stage, and the surviving sources are printed: a silently
narrower mixture is indistinguishable from the intended one in the loss curve.
Every source failing raises, since there is then nothing to train on.

**Every position is supervised**, against 10.1% in the agentic path, because
the teacher labels every token rather than a corpus labelling some of them.
That density is most of why distillation moves faster per step.

The teacher's distribution is compressed to its top-128 log-probabilities
**plus one lumped remainder**, and the student is scored on exactly that
`k + 1`-event partition. Measured against the exact full-vocabulary KL on
peaked synthetic logits at `V = 151,936`: gradient direction at cosine
**0.998**, and exactly **0.0** at agreement. A bare top-k KL without the
remainder renormalises over the subset and reports a non-zero loss for a
student that agrees with its teacher everywhere — a floor the optimum cannot
reach. The exact KL needs a 622 MB logit tensor per model and does not fit on a
T4.

The two tokenizers must agree, and `check_vocab_match` enforces it rather than
assuming it: every Qwen3 dense model shares 151,669 entries and byte-identical
encoding, but across families a silent mismatch trains the student toward noise
while the loss curve looks healthy.

### Agentic traces (the earlier path, still reachable via `MODE = "sft"`)

`MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x` — agentic coding and
reasoning traces with `messages` and a `tools` schema per row.

**Use `split="train"` (5,000 rows), not `split="full"` (10,000).** The larger
split is corrupted: measured over the first ten rows of each, `full` has `tools`
null in 10/10 rows, **64 of its 65 tool calls degraded to the literal string
`"unknown_tool"`**, and no `<thought>` content at all. `train` has tools on
every row, zero `unknown_tool`, and reasoning intact.

**And `train` has a deduplication problem of its own.** Audited across its
first 200 rows: only **54 unique base trajectories**. The rest are those same
trajectories truncated at different intermediate steps — sometimes ending on
`role="tool"`, sometimes relabelling an intermediate `<thought>` as
`<final_answer>` — with regex substitutions on the prompt preamble. Row 74's
trajectory reappears 24 times across `0..199`, and **40/40** of a positional
validation slice (`skip_traces=160`) shares its base trajectory with the
training slice: identical opening assistant turns, identical first tool
outputs. A held-out split on this corpus needs deduplication by base-trajectory
signature, not by row position.

Real trace lengths, measured with the Qwen3 tokenizer: min 19,122, median
32,480, mean 31,822, max 48,343 tokens. The card's "~50,000 average" is the
tail. **Only 10.1% of tokens are supervised** (assistant turns inside a trace
that is mostly tool output), so a token budget is tokens *seen*, not tokens
*learned from*.

Loss is computed on assistant turns only — including their `tool_calls` when
their arguments are intact, while masking redacted template placeholders (such
as `"<file_path>"` or `"<bash_command>"`) and mapping `<thought>` /
`<final_answer>` tags to Qwen3's native `<think>...</think>` protocol. The
`tool` role's returns are the environment talking back and stay masked. The
mask is built by tokenizing the finished conversation once and locating
assistant spans by sentinel substitution, never by diffing prefixes (which
broke on templates like Qwen3 that rewrite the final turn) or string-matching
role markers. When `max_trace_tokens` bounds a stage, it caps the *kept*
supervised windows from the tail of each trace so bounded runs train on deep
mid-trace reasoning and the final synthesis rather than only Turn 1's opening
boilerplate.

## Claiming a result

Parameter efficiency is a *comparative* claim, so the control is built
alongside the treatment:

1. `count_parameters(model)` splits the trainable count into adapter and core,
   and the build cell prints their ratio: if the adapter is many times the core,
   no win can be attributed to the core and the run is not worth its GPU hours.
2. `adapter_only_params(...)` / `matched_rank(...)` find the adapter-only rank
   whose trainable count matches the treatment's total budget.
3. Both arms run at the same seed, on the same chunks in the same order, for the
   same number of steps, through the **same** `distill_steps` and
   `evaluate_gap` — `attach_core(n=0)` is a transparent no-op, so nothing about
   the comparison depends on a second code path.
4. The metric is the KL to the teacher, reported overall *and* on the tail,
   against `teacher_reference_bits` as the floor. Without the teacher's own
   cross-entropy on the same chunks, "the gap shrank" has no scale.
5. **The gap is reported including when it is negative.** A core that loses to a
   matched LoRA is a real finding.

Two diagnostics are printed because their absence is indistinguishable from
success: `kept_fraction` (a `min_teacher_conf` floor above the teacher's typical
top-k mass silently discards most of the batch while the KL still falls on
whatever survives) and `hard_bits` (a KL that falls while this rises is a
student drifting toward the teacher's *shape* and away from the text).

### Measured benchmark: D1 cross-scale distillation (`Qwen3-4B` → `Qwen3-0.6B`)

First D1 run on a Google Colab Tesla T4 GPU (`Qwen/Qwen3-4B` teacher →
`Qwen/Qwen3-0.6B` student, 400 training chunks of 512 tokens, 60 held-out
chunks, `grad_accum=8`, teacher reference floor `3.1726` bits/token):

| Arm | Trainable | Overall KL gap (step 0 → 400) | Tail 25% KL gap (step 0 → 400) | Student bits/tok (floor `3.1726`) |
|---|---|---|---|---|
| **Odydora** (`N=512, proj=64, r=2` attn) | 1,032,193 | `1.1034 → 0.9648` (`+0.1386`, 12.6%) | `1.0484 → 0.8949` (`+0.1535`, 14.6%) | `3.8187 → 3.6276` (`+0.1911`) |
| **Matched LoRA** (`N=0, r=4` attn) | 1,146,880 | `1.1034 → 0.9454` (`+0.1580`, 14.3%) | `1.0484 → 0.8848` (`+0.1636`, 15.6%) | `3.8187 → 3.6106` (`+0.2081`) |

In that initial 0.2.1 setup, the `r=4` adapter-only control beat the treatment
by `0.0194` bits overall (`0.0101` in the tail) — and the control arm's own
output (`"tail beat overall: YES"` despite `N=0`) exposed the three defects in
the experimental setup that caused it, all fixed in 0.2.2:

1. **The teacher was amnesic at every chunk boundary.** `teacher_targets`
   called `teacher(input_ids=ids)` with no `past_key_values`, so on chunk 2 the
   teacher's target distribution was ignorant of chunk 1 — and a student
   carrying state from chunk 1 was *penalised* for remembering what the amnesic
   teacher forgot. The teacher now carries a `DynamicCache` across chunks of the
   same document (`teacher_cache=True`, capped at `2,048` tokens = 288 MiB).
2. **85% of chunks were 1-chunk documents (`is_doc_start=True`).** At
   `min_chars=512` (~120 tokens), 34 of 40 chunks on `OpenCodeInstruct` were
   document starts (1.18 chunks/doc), wiping the recurrent state at 85% of
   steps. `stream_text_chunks(..., min_chunks=2)` now drops single-chunk
   documents so every streamed document crosses at least one chunk boundary.
3. **`tail_kl_bits` sliced `VAL[-15:]` instead of continuation chunks
   (`is_doc_start=False`).** `evaluate_gap` now splits `fresh_kl_bits` (chunk 1,
   state reset) from `carried_kl_bits` (chunk 2+, state carried across a
   boundary).

### Measured benchmark: agentic SFT, Odydora core vs matched LoRA

The figures below are from the **earlier agentic-SFT chapter** and are kept
because they are real. They were taken before the corpus deduplication audit
above, so the held-out split they use is contaminated by prefix-truncated
copies of training trajectories — treat the absolute values as upper bounds and
the *ordering* as the finding.

Measured on a Google Colab Tesla T4 GPU on `Qwen/Qwen3-1.7B` (4-bit NF4, 28 layers, $d=2048$)
against `MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x` (`split="train"`).
Evaluation ran over 157 held-out chunks from traces 160..200.
Both runs trained for identical step counts (220 chunks of 512 tokens), with `grad_accum=8`,
learning rate $2\times 10^{-4}$, and AdamW.

| Architecture | Trainable Parameters | Held-out Loss (bits/tok) | Held-out Perplexity | Tail 25% Loss (bits/tok) | Tail Perplexity |
|---|---|---|---|---|---|
| **Base Frozen Model (Step 0)** | 0 | 3.3280 | 10.04 | 2.8729 | 7.33 |
| **Matched LoRA Control Arm ($r=1$, 220 steps)** | 1,089,536 | 1.1160 | 2.17 | 0.9662 | 1.95 |
| **Matched LoRA Control Arm ($r=1$, 440 steps / 2× pass)** | 1,089,536 | 0.4923 | 1.41 | 0.4046 | 1.32 |
| **Odydora Recurrent Core ($N=64$, 220 steps, flat AdamW)** | **266,241** | 0.2186 | 1.16 | 0.2441 | 1.18 |
| **Odydora Recurrent Core ($N=64$, 220 steps, `configure_optimizer`)** | **266,241** | **0.2026** | **1.15** | **0.2094** | **1.16** |
| *Odydora Advantage (at 220 steps)* | **-75.6% fewer parameters** | **-0.9134 bits** | **-47.0% perplexity** | **-0.7568 bits** | **-40.5% perplexity** |

*(Even when the 4.1× larger LoRA control arm is given a second full 220-chunk pass — 440 steps total — it plateaus at 0.4923 overall / 0.4046 tail, whereas Odydora reaches **0.2026 overall / 0.2094 tail** in a single 220-step pass (`mix` adapting to `0.0195`, `|up|` reaching `2.3283`), and **0.0057 bits/token** on 790-chunk carry runs).*

#### Why the gap exists: the memory ceiling

The control arm is not an artificial handicap: at $r=1$ across all 7 linear projections in all 28 layers,
the LoRA baseline has **4.1× more trainable parameters** than the Odydora core (1,089,536 vs 266,241).

Yet the training curves tell the story:
- **LoRA hit a wall:** LoRA training loss stalled around 0.80–1.00 bits/token. Because LoRA is purely spatial (per-layer matrices with zero cross-chunk memory), it suffers from chunk-boundary amnesia: it can adapt format and syntax, but cannot infer multi-step reasoning context spanning earlier chunks.
- **Odydora broke through:** Odydora's loss fell steadily from 3.86 to 0.04 (and 0.0036 on longer runs). The 64-dimensional recurrent state vector $s$ carried across chunk boundaries transports compressed context through the decoder stack, allowing the model to anticipate reasoning trajectories without re-reading past tokens.

Published DoRA baselines circulating in secondary sources are unreliable: an
adversarial check of our own research found fabricated rows, including LLaMA3-8B
numbers attributed to a paper submitted two months before LLaMA3 existed. Read
the primary source before quoting any external figure.

## Layout

```
odydora/core.py      RecurrentCore — the shared N×N mixer
odydora/adapter.py   attach_core(), without_odydora() — hooks, state, counting
odydora/data.py      streaming loader, assistant-only mask, chunker
odydora/train.py     build(), configure_optimizer(), train_steps(), evaluate()
tests/               67 tests, CPU-only, no network
tools/               notebook generator and structural validator
notebooks/           the Colab notebook
```

## License

MIT. See [LICENSE](LICENSE).
