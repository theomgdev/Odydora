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
