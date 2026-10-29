# Contributing to Odydora

## The one rule everything else follows

**Every number in this repository was measured, not cited.** The README's
throughput table, the memory figures, the influence-decay numbers — each came
from a script that ran on real hardware. If you add a figure, run it first and
say what you ran it on. If you cannot run it, say so instead; a named gap is
worth more than a plausible estimate, and a predicted number published as a
measured one is the one failure that discredits the measured ones beside it.

This also means a figure that does not reproduce on your machine is a finding
rather than a complaint. Open a *measurement dispute* issue with your numbers.

## Setup

```bash
git clone https://github.com/theomgdev/Odydora
cd Odydora
python -m venv .venv
.venv/Scripts/activate          # Windows
source .venv/bin/activate       # Linux / macOS
pip install -e ".[dev]"
```

Add `.[quant]` for the 4-bit path. It is deliberately optional: `bitsandbytes`
is CUDA-only, and the library plus its whole test suite run without it.

```bash
pytest -q          # 43 tests, CPU-only, no network, a few seconds
ruff check .
```

## What the test suite is and is not

The suite builds models from `config` with **random weights** and downloads
nothing. `GPT2Config(n_layer=2, n_embd=32)` appears in `tests/test_adapter.py`
purely so the shared-`W` invariant can be asserted in milliseconds — no
pretrained GPT-2 is ever trained, evaluated, or reported. Anything that needs
real weights belongs in a notebook or a `tmp/` probe, not in `tests/`.

So the suite proves *wiring*, never learning. A smoke test that asserts a loss
threshold is reading the seed; assert instead that gradient reaches every family
expected to learn and that the tensors actually move.

The one bug class this caught the hard way: every test attached the core
*without* PEFT in the way, so all of them stayed green while `build()` — the
real path, adapter first then core — crashed on the first 4-bit call, because
PEFT nests the host and moves the decoder stack to
`base_model.model.model.layers`. **Cover the call order the user actually
runs**, not the convenient one.

## Design constraints that are not negotiable

The first three are inherited from [OdyssNet](https://github.com/theomgdev/OdyssNet)
(Karahan, v4.0.0, MIT), which this architecture is built on; the rest came out of
measurement here. Undoing one silently breaks the architecture:

- **`W.diagonal()` stays zero**, enforced at init *and* on the gradient.
  Self-connections belong to the host residual stream.
- **Only the output projection is zero-initialised.** Zeroing a second factor
  as well puts a zero in every factor's gradient and the core can never leave
  the origin. This design started with a gate and the gate was the bug.
- **No clamps, no spectral normalisation, no gradient clipping on the core.**
  The convex mix bounds the state by construction — the new state is on the
  `√N` sphere and the old one is inside it, so the mixture cannot leave the
  ball. A clamp here would hide a wrong formulation rather than fix one.
- **The state advances per segment, not per token.** The step is
  kernel-launch-bound; per-token updates measured 1.6% of baseline throughput.
- **The core is attached by forward hook.** Replacing or wrapping the block
  breaks `transformers`' attribute dispatch (`attention_type` and friends).
- **The mixing rate is a clamped logit.** `sigmoid` saturates to exactly 1.0 in
  float32, and a mix of 1 drops the carried state entirely.

If you want to change one of these, bring the measurement that says it should
change. That is a welcome pull request, not a forbidden one.

## Corpus handling

- Loss is on **assistant turns only**, including their `tool_calls` — choosing
  the tool and its arguments is the reasoning being trained. The `tool` role's
  returns stay masked.
- The mask is built by tokenizing the finished conversation once and locating
  each assistant turn's span by **sentinel substitution**, never by diffing
  prefix renders (which breaks when a template rewrites the final turn) or
  string-matching a role marker (which shifts the mask when content contains
  the marker).
- Held-out splits are by **trace**, never by chunk. Carried state leaks across a
  chunk-level split and the eval number becomes meaningless.
- `split="train"` is the clean split. `split="full"` is twice the size and
  corrupted; see the README.

## Making a comparative claim

Parameter efficiency is comparative, so the control ships with the treatment:
`count_parameters()` splits the budget, `matched_rank()` finds the adapter-only
rank that matches it, both arms run at the same seed on the same traces in the
same order, and the metric is bits/token on held-out traces reported overall and
tail-only. **Report the gap including when it is negative.**

Do not quote an external DoRA or LoRA baseline from a blog post or a secondary
summary. An adversarial check of this project's own research notes found
fabricated rows — including LLaMA3-8B numbers attributed to a paper submitted
two months before LLaMA3 existed. Read the primary source.

## The notebook is generated

`notebooks/odydora_colab.ipynb` is built by `tools/build_notebook.py`. Edit the
generator and regenerate; CI fails if the committed notebook differs from what
the generator produces. Hand-editing the JSON makes the cells undiffable and
lets the constants drift from what the package actually accepts.

## Commits and pull requests

- Write what was wrong, not what the diff shows. The reader can see the diff.
- Quote test counts, not adjectives: "43 tests pass", not "tests pass".
- Keep the change surgical. Do not reformat what you scrolled past.
- Do not claim a result you did not observe. If you did not run it, say so.
- If a model assisted, disclose it — a line in the PR or an `Assisted-by:`
  trailer. The author and the accountable party stay human, and an agent must
  never add a `Signed-off-by:`.

## Scratch and research

`tmp/` is for throwaway probes and `Research/` for research notes; both are
gitignored and neither is library code. A probe worth keeping becomes a test.
