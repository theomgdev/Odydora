# Changelog

## 0.1.0 — unreleased

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
