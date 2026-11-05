# AGENTS.md

Instructions for any assistant working in this repository. Read it before you
touch anything. `CONTRIBUTING.md` carries the full detail; this is the short
form plus the things an agent specifically gets wrong here.

## Where this architecture comes from

The core is borrowed from [OdyssNet](https://github.com/theomgdev/OdyssNet)
(Cahit Karahan, v4.0.0, MIT): one shared `N x N` matrix with a hard-zero
diagonal, per-step RMSNorm on the state, and the state detached at the call
boundary. Reference only -- never import or vendor it, since Colab cannot see
that repository. Its measurements are its own and are not evidence about this
project. Credit it in anything published from here, and keep `CITATION.cff`'s
`references` entry intact.

OdyssNet also lives locally at `~/Desktop/PROJELER/Python/AI/OdyssNet` with its
own `AGENTS.md`; read that before editing anything inside it.

## Rule one: every number here was measured

The throughput table, the memory figures, the influence-decay numbers — each
came from a script that ran on this hardware. A predicted number published as a
measured one is the single failure that discredits every real figure beside it.
Run it, or say you could not. "Roughly" and "approximately" are not a licence to
estimate; they are a reason to measure.

## What the test suite proves, and the trap in it

64 tests, CPU-only, no network, a few seconds. They build models from `config`
with random weights. `GPT2Config(n_layer=2)` in `tests/test_adapter.py` exists
only to assert the shared-`W` invariant cheaply — **no pretrained GPT-2 is ever
trained or reported, and a result on one would be worthless.**

The suite proves wiring, never learning. Do not add a loss-threshold assertion;
at smoke-test budget it reads the seed, not the model.

**The trap:** every test used to attach the core *without* PEFT, so all 30
stayed green while `build()` — the real order, adapter then core — crashed on
the first 4-bit call, because PEFT moves the decoder stack to
`base_model.model.model.layers`. Cover the call order the user runs.

## Constraints that are load-bearing

Undoing one of these silently breaks the architecture. Each came from a
measurement, which is in the README:

- `W.diagonal()` stays zero, at init and on the gradient.
- Only `up.weight` is zero-initialised. Zeroing a second factor (this design
  started with a gate) puts a zero in every factor's gradient and the core
  never leaves the origin.
- No clamps, no spectral normalisation, no gradient clipping on the core. The
  convex mix bounds the state by construction.
- The state advances per `stride` tokens, not per token: the step is
  kernel-launch-bound and per-token measured 1.6% of baseline throughput.
- The core attaches by forward hook. Wrapping the block breaks `transformers`'
  attribute dispatch.
- `mix` is a clamped logit — `sigmoid` saturates to exactly 1.0 in float32 and
  a mix of 1 drops the carried state.
- Only the **last** factor on the output path (`up.weight`) is zeroed, including
  when `proj_rank` factors the projections. Zeroing `up_in` as well puts a zero
  in every factor's gradient and the core never leaves the origin.
- The distillation KL carries a **lumped remainder term** alongside the
  teacher's top-`k`. Dropping it renormalises over the subset, so the loss is
  non-zero for a student that matches its teacher exactly — a floor the optimum
  cannot reach. `kl_when_equal` is asserted to be 0.0 for exactly this reason.
- `check_vocab_match` is a precondition, not a courtesy. A per-token KL across
  two different tokenizers is undefined and trains the student toward noise with
  a healthy-looking loss curve.
- `kept_fraction` and `hard_bits` are reported unconditionally. A confidence
  floor that eats the batch, or a hard-loss column reading 0.0 because its
  weight is 0, is indistinguishable from success in the log.
- Every corpus in `DEFAULT_MIXTURE` is ungated and malware-clean. A default
  names a dataset on the reader's behalf, so a gated one stops their notebook
  at the data cell and a flagged one hands them someone else's trust decision.
  Check `HfApi().dataset_info(repo).gated` and the Hub's scan status *before*
  adding an entry — `bigcode/the-stack-smol` failed both and shipped once.

Changing one is welcome *with* the measurement that says it should change.

## This machine

- Interpreter: `.venv/Scripts/python.exe` (CUDA torch, `bitsandbytes` 0.50.2).
  There is exactly one venv; do not create a second.
- `pytest -q` and `ruff check .` are the gates. Quote the test count.
- **The line here is 10 Mbit.** Never download model weights to verify
  something. Save a tiny random model to disk and load it back, or run with
  `HF_HUB_OFFLINE=1` against what is already cached. A 1.5 GB pull costs the
  user twenty minutes and a Colab run costs nothing.
- The GPU is the user's. Do not start a training run that was not asked for in
  the current turn.

## The notebook is generated

Edit `tools/build_notebook.py`, then run it. CI fails if the committed
`.ipynb` differs from what the generator produces.

## Scratch

`tmp/` for probes, `Research/` for notes; both gitignored, neither is library
code. A probe worth keeping becomes a test. Never delete a background job's
output file while the job is running — a partial read looks exactly like
completion, and on Windows the handle keeps writing into nothing visible.

## Disclosure

Say a model helped: a line in the pull request or an `Assisted-by:` trailer.
Never add a `Signed-off-by:` — only a person can certify that.
