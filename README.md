# Odydora

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theomgdev/Odydora/blob/main/notebooks/odydora_colab.ipynb)
[![CI](https://github.com/theomgdev/Odydora/actions/workflows/ci.yml/badge.svg)](https://github.com/theomgdev/Odydora/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A single `N × N` matrix, shared by every decoder layer of a frozen 4-bit
language model, carrying a recurrent state **across layers** and **across chunk
boundaries** — so a 30,000-token agentic trace trains in 1,024-token pieces
without the backward pass ever spanning the whole document.

The frozen transformer is the limbic system. The core is the frontal cortex
steering it, and it costs 266,241 trainable parameters at N=64 regardless of how
deep the host model is.

```
for chunk in trace:                                     # 1k tokens at a time
    for layer L in CORE_LAYERS:
        h         = block_L(h)                          # frozen, 4-bit
        h         = h + LoRA_L(h)                       # trained, standard
        s         = (1-a)·s_prev[L] + a·RMSNorm(down(pool(h)) + s_prev[L] @ W)
        h         = h + up(s)                           # up starts at zero
        s_prev[L] = s.detach()                          # at the chunk boundary
```

One `W`, one `down`, one `up`, one mixing rate — shared by all 28 layers.

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
pytest -q                                        # 45 tests, CPU, no downloads
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

`MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x` — agentic coding and
reasoning traces with `messages` and a `tools` schema per row.

**Use `split="train"` (5,000 rows), not `split="full"` (10,000).** The larger
split is corrupted: measured over the first ten rows of each, `full` has `tools`
null in 10/10 rows, **64 of its 65 tool calls degraded to the literal string
`"unknown_tool"`**, and no `<thought>` content at all. `train` has tools on
every row, zero `unknown_tool`, and reasoning intact.

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

1. `count_parameters(model)` splits the trainable count into adapter and core.
2. `matched_rank(...)` finds the adapter-only rank whose trainable count matches
   the core's budget.
3. Both arms run at the same seed, on the same traces in the same order, for the
   same number of steps.
4. The metric is held-out bits/token on held-out **traces** — never held-out
   chunks of training traces, or the carried state leaks across the split —
   reported overall *and* on the tail, since carry can only pay where a chunk
   has history behind it.
5. **The gap is reported including when it is negative.** A core that loses to a
   matched LoRA is a real finding.

### Measured benchmark: Odydora Recurrent Core vs Matched LoRA Baseline

Measured on a Google Colab Tesla T4 GPU on `Qwen/Qwen3-1.7B` (4-bit NF4, 28 layers, $d=2048$)
against `MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x` (`split="train"`).
Evaluation ran over 157 held-out chunks from traces 160..200 (unseen during training).
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
tests/               43 tests, CPU-only, no network
tools/               notebook generator and structural validator
notebooks/           the Colab notebook
```

## License

MIT. See [LICENSE](LICENSE).
