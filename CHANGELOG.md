# Changelog

Versions are tagged (`v0.1.0`, `v0.2.0`, ...) so a number quoted from this
project can be traced to the exact code that produced it. The Colab badge
tracks `main`, so the notebook a reader opens is always the current one —
a tag is the fixed reference point, not the moment of availability.

## 0.2.1 — 2026-11-06

**Every corpus in the default mixture is now ungated and malware-clean.**
`bigcode/the-stack-smol` was the obvious code source and shipped in 0.2.0; it
is gated (`gated="auto"`) **and** carries three files the Hub's scanner flagged
as malware, so a reader who had not requested access hit a
`DatasetNotFoundError` at cell 4 — and one who had would have been handed
someone else's trust decision. Replaced with `nvidia/OpenCodeInstruct`
(CC-BY-4.0, ungated, verified streaming).

A corpus default names a dataset on the reader's behalf, so the bar is now
explicit in `AGENTS.md`: check `HfApi().dataset_info(repo).gated` and the Hub's
scan status *before* adding an entry. Checked alongside the replacement and
rejected for the same reasons: `bigcode/the-stack-dedup`, `bigcode/starcoderdata`,
`nampdn-ai/tiny-codes`, `bigcode/the-stack-v2-train-smol-ids` (all gated);
`codeparrot/github-code-clean` and `bigcode/commitpackft` (script-based, which
`datasets>=4` refuses to load).

Two mechanisms came out of it:

- **`text_field` may be a tuple**, joined with a blank line. Instruction
  corpora split a document across a prompt column and a response column, and
  reading one of them trains the student on half of every document while
  everything downstream looks normal.
- **A source that cannot be opened narrows the mixture instead of killing the
  stage**, with the reason named and the surviving sources printed. A gated
  repo, a removed dataset or a script-based one should cost one stream, not the
  whole run — but a silently narrower mixture is indistinguishable from the
  intended one in the loss curve, so the list is printed rather than inferred.
  Every source failing still raises, since there is then nothing to train on.

All three regressions were driven RED against 0.2.0's module from git before
the fix. 64 tests pass.

## 0.2.0 — 2026-11-06

The project's question changed. Up to 0.1.0 it asked whether a shared recurrent
core helps a frozen host learn a fixed corpus. It now asks a sharper one:
**how much of a larger model's next-token distribution can a small frozen host
recover, given about a million trainable parameters of temporal recurrence —
and does it recover more than the same budget spent spatially as LoRA?**

Teacher gives probabilities, student learns probabilities. Three things had to
be built for that, and two of them came from a measurement that contradicted
the obvious design:

- **`odydora/distill.py`: top-k-plus-remainder KL from a larger teacher.** The
  exact full-vocabulary KL needs a 622 MB logit tensor per model at Qwen3's
  `V=151,936` and will not fit on a T4. Compressing the teacher to its top-128
  log-probabilities **plus one lumped remainder** reproduces the exact KL's
  gradient direction at cosine **0.998** on peaked synthetic logits, and is
  exactly **0.0** when the two distributions agree. A bare top-k KL without the
  remainder term renormalises over the subset and reports a non-zero loss at
  agreement — an objective with a floor its own optimum cannot reach.
- **`RecurrentCore(proj_rank=...)`: a factored I/O bottleneck.** At the
  `Qwen3-0.6B` student's `d=1024`, dense projections for an `N=512` core cost
  `1,048,576` parameters against `W`'s `261,632` — the recurrence, the part that
  carries state, is a fifth of the module. `proj_rank=64` cuts the interface to
  `196,608`, so `W` becomes **57%** of a `458,753`-parameter core. Only
  `up.weight` is zeroed, so the module is still an exact identity at step 0
  while every factor takes gradient as soon as `up` leaves the origin.
- **`build(lora_targets=...)`, and a much smaller adapter.** `r=32` on all seven
  projections is `20,185,088` trainable parameters — **44x** the core, which
  makes any win unattributable. Held to `r=2` on attention only, the adapter is
  `573,440` against the core's `458,753`: a ratio of **1.25x**, printed by the
  build cell so it cannot drift unnoticed.

Measured end to end on the real 28-layer `Qwen3-0.6B` geometry with random
weights (no download): adapter `573,440` + core `458,753` = `1,032,193`
trainable, the matched adapter-only control at `r=4` is `1,146,880` — *more*
than the treatment, which is the honest direction for the comparison to lean.

Two diagnostics exist because their absence is indistinguishable from success:

- **`kept_fraction`** in every log row. A `min_teacher_conf` floor set above the
  teacher's typical top-k mass silently discards most of the batch while the KL
  column still falls on whatever survives. The default is now `0.0`.
- **`hard_bits`** computed whenever labels exist, not only when weighted in. A
  KL that falls while this rises is a student drifting toward the teacher's
  *shape* and away from the text, and a figure reading `0.0` because its weight
  is zero looks like a student predicting perfectly.

`evaluate_gap` reports the gap overall **and** in the tail, against
`teacher_reference_bits` as the floor: without the teacher's own cross-entropy
on the same chunks, "the gap shrank" has no scale. The corpus is a four-way
round-robin mixture (web expository text, synthetic textbook prose,
mathematics, code), because a student that overfits one domain has learned a
dialect rather than a distribution.

The agentic-SFT path is unchanged and still reachable through `MODE = "sft"`, so
the 0.1.0 result stays reproducible. 61 tests pass.

### Corpus audit that closed the previous chapter

`MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x` (`split="train"`) turned
out to hold only **54 unique base trajectories** in its first 200 rows; the rest
are the same trajectories truncated at different intermediate steps (sometimes
ending on `role="tool"`, sometimes relabelling an intermediate `<thought>` as
`<final_answer>`) with regex substitutions on the prompt preamble. Row 74's
trajectory reappears 24 times across `0..199`, and **40/40** of a positional
validation slice (`skip_traces=160`) shares its base trajectory with the
training slice. That is why distillation replaced it rather than supplementing
it.

## 0.1.0 — 2026-10-30

First working version: a shared `N × N` recurrent core attached to a frozen
4-bit language model as a parameter-efficient adapter, with the Colab notebook
that trains it.

Three design decisions came out of measurement rather than from the plan they
started in, and each reversed the plan:

- **Segment-rate recurrence.** The step is kernel-launch-bound, so a per-token
  state update cost 98.4% of throughput at 28 call sites (28 tok/s against a
  1,786 baseline on 2.03B fp16 weights). Advancing the state once per `stride`
  tokens — as Transformer-XL and Block-Recurrent Transformer do — gives 660
  tok/s at stride 64 for byte-identical parameters.
- **A convex state mix instead of a decay.** Per-step RMSNorm pins the state to
  the `√N` sphere exactly, making explosion structurally impossible and
  forgetting total: influence of an input 4,096 steps later measured 0.0e+00.
  The literature's contractive decay made it worse. `s ← (1-a)s + a·new` holds
  influence at 5.8e-01 after 1,024 steps at `a = 0.02`.
- **One zero-initialised factor, not two.** The design began with both a gate
  and a zero output projection; with both at zero every factor's gradient
  contains a zero and the core can never leave the origin. The gate was removed.

Also settled by measurement: intra-chunk detaching saves no memory (61.8 MiB at
window 2, 4, 8 and full backpropagation alike) so it is off; the corpus's
`train` split is clean while the larger `full` split has 64 of 65 tool calls
degraded to `"unknown_tool"`; and Unsloth cannot host a foreign module, since
`prepare_model_for_training()` freezes every parameter not named `lora_*`.

Fixed before release: `_resolve_stack` searched fixed attribute paths and so
could not find the decoder stack through a PEFT wrapper, crashing `build()` on
the first real 4-bit call while every unit test stayed green.

### Measured benchmark and subsequent refinements

- **Empirical proof on `Qwen/Qwen3-1.7B` (Colab Tesla T4, 157 held-out chunks):**
  Odydora (`N=64`, 266,241 core params) reached 0.2186 bits/token overall (tail
  0.2441) under flat AdamW and **0.2026 overall (tail 0.2094)** with
  `configure_optimizer()` (`mix` adapting to 0.0195, `|up|` reaching 2.3283, and
  0.0057 over 790 chunks) against 1.1160 overall (tail 0.9662) at 220 steps (and
  0.4923 overall / 0.4046 tail at 440 steps / 2× pass) for the matched-rank LoRA
  control arm (`N=0, r=1`, 1,089,536 params, 4.1× larger).
- **Corpus reasoning-tag alignment and redacted tool-call masking.** In
  `Fable-5.1` (`split="train"`), 97% of `tool_calls` carry redacted placeholder
  arguments (`"<file_path>"`, `"<directory_path>"`, etc.), and intermediate
  reasoning uses `<thought>` rather than Qwen3's `<think>`. `mask_assistant_turns`
  now maps `<thought>` / `<final_answer>` to Qwen3's native `<think>` protocol
  and masks redacted `tool_call` suffixes; `stream_traces` applies
  `max_trace_tokens` to the kept supervised windows (retaining the tail
  synthesis) rather than slicing raw `ids[:max_trace_tokens]` before dropping
  unsupervised tool outputs.
- **`configure_optimizer()` with parameter family separation.** Following
  OdyssNet's `ChaosGrad` policy, modulation parameters (`core.alpha`,
  `core.norm.weight`, DoRA `lora_magnitude_vector`) get `weight_decay=0.0` so
  `alpha` is not pulled toward `0.0` (`mix=0.50`), while the shared core scales
  by `core_lr_scale=5.0` (and `alpha` by `25x` to offset `mix*(1-mix) ≈ 1/50`).
  Measured on 4-bit `Qwen3-0.6B` across seeds 42, 123 and 7: overall held-out
  loss dropped from 1.179 to 1.095 bits/token and tail loss from 1.436 to 1.345.
- **Exact partial-segment pooling.** When `time % stride != 0` (including
  `time=1` during `model.generate(use_cache=True)`), zero-padding to `stride`
  and calling `.mean(2)` divided the segment sum by `stride` instead of its true
  token count, attenuating `down(pooled)` by `64x` at `time=1`. Fast-pathing
  `time <= stride` and averaging ragged tails over their true length removed the
  dilution and cut `time=1` core latency by 31%.
- **Interactive Colab UI (Cells 8 and 9).** Added a Drive Checkpoint Manager
  Form (`inspect_checkpoint`, architecture column, safe active-switching with
  timestamped backups, cold-start archive) and an Interactive Chat Playground
  with `TextStreamer` live output, multi-turn state carry, `unlimited_tokens`
  toggle, and RNG-locked side-by-side comparison via `model.without_odydora()`.
