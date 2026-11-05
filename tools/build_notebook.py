"""Generate notebooks/odydora_colab.ipynb.

The notebook is written from here rather than by hand so the cells stay
diffable and the constants cannot drift from what the package actually
accepts. Run: python tools/build_notebook.py
"""

import json
import pathlib

CELLS = []


def _lines(text):
    """Split into nbformat `source` lines, each keeping its trailing newline.

    nbformat says a multiline string is a list whose elements already end in
    "\\n"; a reader reconstructs the cell with "".join(source). Splitting on
    "\\n" and dropping the separator therefore renders every markdown cell as
    one run-together line -- tables collapse, "###" stops being a heading, and
    stray "~" pairs become strikethrough. Colab shows this; a validator that
    rejoins with "\\n".join() does not, which is how it shipped once already.
    """
    return text.strip("\n").splitlines(keepends=True)


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": _lines(text)})


def code(text, metadata=None):
    CELLS.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata or {},
        "outputs": [],
        "source": _lines(text),
    })


md("""
# Odydora — can temporal depth buy a small model a large model's distribution?

One tiny `N x N` matrix, shared by every decoder layer of a **frozen** host,
carrying a recurrent state across layers **and** across chunk boundaries. The
frozen transformer is the limbic system; the core is the frontal cortex
steering it.

The core architecture is borrowed from
**[OdyssNet](https://github.com/theomgdev/OdyssNet)** (Karahan, v4.0.0, MIT):
one shared `N x N` matrix with a hard-zero diagonal under per-step RMSNorm,
with the state detached at the call boundary. Odydora adapts it to steer a
frozen host at segment rate. Reference, not a dependency — cite both
(`CITATION.cff`).

## The question this notebook asks

A small model and a large model of the same family share a tokenizer, so their
next-token distributions live in the same space and the distance between them
is measurable. **Teacher gives probabilities, student learns probabilities.**

So: how much of a larger model's distribution can **about a million trainable
parameters** of temporal recurrence recover inside a smaller frozen host — and
does it recover more than the same budget spent spatially, as LoRA?

Two numbers decide it, and both are printed by Cell 6:

| | what it means |
|---|---|
| `kl_bits` | distance from the student's distribution to the teacher's, in bits/token. Lower is closer. **0.0 means identical.** |
| `tail_kl_bits` | the same distance over the trailing chunks, where the recurrent state has history behind it. State carry can only pay where there is a past, so a gain here and not overall is the mechanism; a gain overall and not here is something else. |

`teacher_bits` is the floor: the teacher's own cross-entropy on the same text.
Without it, "the gap shrank" has no scale.

### Why distillation rather than more corpus

A fixed corpus caps what the student can learn at whatever the corpus happens
to contain, and a hard label is one sample from a distribution the teacher
knows in full. The teacher's top-128 log-probabilities per position carry far
more signal per token than the single realised id — and the signal is dense:
**every** position is supervised, against roughly 10% in the agentic traces
this project trained on before.

### What is deliberately at risk

A student that overfits the teacher on one domain has learned a dialect, not a
distribution. The corpus is therefore a four-way interleaved mixture — web
expository text, synthetic textbook prose, mathematics, code — round-robined so
no stage sees a long run of one domain. That is what separates a mixture from
sequential fine-tuning, which forgets.

### Numbers measured before you run this, so you can tell a surprise from a bug

- The state sits on the `sqrt(N)` sphere exactly because RMSNorm runs every
  step — OdyssNet's mechanism, and the reason no spectral constraint is needed.
  Explosion is structurally impossible; **forgetting** is the real failure,
  which is what `MIX` exists to prevent (influence after 1,024 steps: 5.8e-01
  at mix 0.02, 2.0e-34 at 1.0).
- The step is kernel-launch-bound, so cost tracks the number of state updates
  and not the arithmetic: 28 tok/s at stride 1 against 660 at stride 64 on an
  RTX 3060 Ti. `STRIDE` is the throughput lever.
- Core parameters are **independent of depth**, because `W`, `down` and `up`
  are all shared across every layer that uses them.
- Top-k-plus-remainder KL against the exact full-vocabulary KL on peaked
  synthetic logits at `V=151,936`: gradient direction at cosine **0.998**, and
  exactly **0.0** at agreement. The exact KL needs a 622 MB logit tensor per
  model and will not fit on a T4.
""")

md("""
## 1 — Setup

Installs, repo, Drive. `datasets>=4.8` streams every corpus here without
materializing it.

`pip` will warn that `gradio`, `gradio-client` and `diffusers` want
`huggingface-hub>=1.16` while this cell resolves `0.36.x`. **Expected and
harmless here:** `transformers 4.x` requires `huggingface-hub<1.0`, so pip pins
the hub down, and the three complaining packages are preinstalled in the Colab
image, while nothing in this notebook imports them. We hold `transformers<5` on
purpose, since 4-bit loading through `peft` and `bitsandbytes` is tested
against the 4.x API.
""")

code("""
!pip install "transformers>=4.46,<5" "peft>=0.13" "bitsandbytes>=0.44" "datasets>=4.8" "accelerate>=1.0"
!git clone https://github.com/theomgdev/Odydora /content/Odydora || git -C /content/Odydora pull

import os, sys, torch

sys.path.insert(0, "/content/Odydora")

from google.colab import drive
drive.mount("/content/drive", force_remount=False)
CKPT_DIR = "/content/drive/MyDrive/odydora_ckpt"
os.makedirs(CKPT_DIR, exist_ok=True)

p = torch.cuda.get_device_properties(0)
# is_bf16_supported() defaults to including_emulation=True, which returns True
# even on a T4 by falling back to a software-emulated bfloat16 tensor. Compute
# capability >= 8.0 (Ampere) is the test for real bf16 tensor cores.
bf16_cores = p.major >= 8
print(f"GPU            {p.name}  {p.total_memory/2**30:.1f} GiB  sm_{p.major}{p.minor}")
print(f"bf16 cores     {bf16_cores}   ({'Ampere or newer' if bf16_cores else 'Turing: emulation only, so fp16'})")
print(f"training in    fp16   checkpoints -> {CKPT_DIR}")
""")

md("""
## 2 — Config

**The only cell you edit.** Values marked MEASURED came from a probe, not from
a guess.

### The budget geometry, which is the whole experiment

A parameter-efficiency claim needs the core to be the *majority* of what
trains, or a win cannot be attributed to it. Two measurements set the defaults
below, both on the `Qwen3-0.6B` student (`d=1024`):

**The core's dense I/O projections dominate a wide core.** At `N=512` they cost
`2 * 1024 * 512 = 1,048,576` parameters against `W`'s `261,632` — the
recurrence, the part that actually carries state, ends up as a fifth of the
module. `PROJ_RANK=64` factors them down to `196,608`, so `W` becomes **57%**
of a 458,753-parameter core instead of 20%.

**LoRA at its usual settings buries the core.** `r=32` on all seven
projections is `20,185,088` trainable parameters — **44x** the core. The
adapter exists to let the frozen host's projections meet the core's delta, not
to do the learning, so it is held to `r=2` on attention only: `573,440`, the
same order as the core.

| arm | adapter | core | total trainable |
|---|---|---|---|
| **Odydora** (`N=512`, `PROJ_RANK=64`, `r=2` attn) | 573,440 | 458,753 | 1,032,193 |
| **Matched LoRA** (`N=0`, `r=4` attn) | 1,146,880 | 0 | 1,146,880 |

The control arm gets *more* trainable parameters than the treatment, which is
the honest direction for a comparison to lean.
""")

code("""
MODE = "distill"        # "distill" teacher->student | "sft" agentic traces (legacy path)
STAGE = "D1"            # D1 first signal | D2 long run | D3 push

# The two models must share a tokenizer, or a per-token KL is undefined. Cell 3
# verifies this rather than trusting it. Every Qwen3 dense model does (151,669
# entries, byte-identical ids), so the pair is free to move along the family.
STUDENT = "Qwen/Qwen3-0.6B"     # 0.75B params, 28 layers, d=1024
TEACHER = "Qwen/Qwen3-4B"       # 4.0B params, 36 layers, d=2560 -- 5.3x the student
                                # Cheaper rung: "Qwen/Qwen3-1.7B". Harder: "Qwen/Qwen3-8B".

N         = 512         # core width. MEASURED: W is 57% of the core at PROJ_RANK=64.
PROJ_RANK = 64          # I/O bottleneck. 0 = dense, and then W is only 20% of the budget.
STRIDE    = 64          # tokens per state update. MEASURED: 28 tok/s at 1, 660 at 64.
MIX       = 0.02        # initial convex mixing rate. MEASURED: smaller remembers longer.
LAYERS    = None        # None = every decoder boundary.
WINDOW    = 0           # intra-chunk detach. MEASURED: no memory saved, so off.

# Attention-only at low rank, so the adapter does not dwarf the core. Cell 3
# prints both numbers; if the adapter is more than about 2x the core, lower RANK.
LORA_ON   = ("q_proj", "k_proj", "v_proj", "o_proj")
RANK      = 2
USE_DORA  = False       # QDoRA dequantizes the frozen 4-bit weight every forward
                        # (2.0-2.4x slower, rank-invariant). A D3 arm, not a default.
LR        = 2e-4
RESUME    = True        # False starts fresh from step 0, with a safety backup

# Distillation knobs.
TOP_K        = 128      # teacher distribution entries kept per position. MEASURED on
                        # synthetic peaked logits at V=151,936: top-k-plus-remainder KL
                        # tracks the exact full-vocabulary KL at cosine 0.998 and is
                        # exactly 0.0 at agreement. The full KL needs a 622 MB logit
                        # tensor per model and will not fit.
TEMPERATURE  = 1.0      # 1.0 keeps the loss readable as bits/token.
HARD_WEIGHT  = 0.1      # anchor on the realised token: KL alone admits a student that
                        # tracks the teacher's shape while being wrong about the token.
MIN_CONF     = 0.0      # skip positions where the teacher's own top-k mass is below
                        # this. Off by default: a floor set above the teacher's
                        # typical top-k mass silently drops most of the batch while
                        # the kl column still looks healthy. Watch the `kept` column
                        # and raise this only once you have seen what `conf` runs at.

STAGES = {
    "D1": dict(chunk=512,  train_chunks=400,   val_chunks=60,  grad_accum=8,
               log_every=10, save_every_s=900),
    "D2": dict(chunk=512,  train_chunks=4000,  val_chunks=120, grad_accum=16,
               log_every=25, save_every_s=900),
    "D3": dict(chunk=1024, train_chunks=12000, val_chunks=120, grad_accum=16,
               log_every=50, save_every_s=900),
    # Legacy agentic-SFT stages, kept so the earlier result stays reproducible.
    "S1": dict(chunk=512,  max_trace_tokens=4096, max_tokens=112_640, traces=160,
               val_traces=40, grad_accum=8, log_every=10, save_every_s=900),
    "S2": dict(chunk=1024, max_trace_tokens=None, max_tokens=None, traces=2000,
               val_traces=50, grad_accum=16, log_every=25, save_every_s=900),
}
CFG = STAGES[STAGE]
SFT_DATASET, SFT_SPLIT = "MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x", "train"

print(f"{MODE}/{STAGE}")
if MODE == "distill":
    print(f"  teacher   {TEACHER}")
    print(f"  student   {STUDENT}  N={N} proj={PROJ_RANK} stride={STRIDE}")
    print(f"  adapter   r={RANK} on {len(LORA_ON)} projections")
    print(f"  chunks    {CFG['train_chunks']} train / {CFG['val_chunks']} val at {CFG['chunk']} tokens")
else:
    print(f"  model     {STUDENT}  N={N} proj={PROJ_RANK}  traces={CFG['traces']}")
""")

md("""
## 3 — Build

Adapter on the projections first, then the core hooked onto the layer
boundaries, so the core sees the adapted hidden state rather than the frozen
one.

Two things to read here. The **tokenizer check** has to pass: a silent
vocabulary mismatch trains the student toward noise while the loss curve looks
healthy. And the **core-to-adapter ratio** is the claim's denominator — if the
adapter is many times the core, lower `RANK` in Cell 2 before spending GPU
hours on a comparison that cannot attribute anything.
""")

code("""
import odydora
from odydora.train import build, build_teacher, report, trainable_state
from odydora.distill import check_vocab_match

print(f"odydora {odydora.__version__}", flush=True)
print(f"loading student {STUDENT} in 4-bit...", flush=True)
model, tok = build(STUDENT, n=N, layers=LAYERS, stride=STRIDE, mix=MIX, window=WINDOW,
                   use_dora=USE_DORA, r=RANK, four_bit=True, proj_rank=PROJ_RANK,
                   lora_targets=LORA_ON)
# Deleting a checkpoint on Drive does not reset weights already in GPU RAM, so
# the step-0 state is snapshotted here and restored by Cell 5 on a fresh run.
_INIT_STATE = {k: v.detach().cpu().clone() for k, v in trainable_state(model).items()}

rows = report(model)
ratio = rows["adapter"] / max(rows["core"], 1)
print(f"total       {rows['total']:>14,}")
print(f"trainable   {rows['trainable']:>14,}   ({rows['trainable_fraction']:.4%})")
print(f"  adapter   {rows['adapter']:>14,}")
print(f"  core      {rows['core']:>14,}   shared by {len(model.odydora_layers)} layer(s)")
print(f"  ratio     adapter is {ratio:.2f}x the core"
      + ("" if ratio <= 2.5 else "   <- too high to attribute a win to the core; lower RANK"))
print(f"core: {model.odydora_core}")

teacher = None
if MODE == "distill":
    print(f"\\nloading teacher {TEACHER} in 4-bit (frozen, no adapter)...", flush=True)
    teacher, teacher_tok = build_teacher(TEACHER, four_bit=True)
    shared = check_vocab_match(teacher_tok, tok)
    print(f"tokenizers agree: {shared:,} shared vocabulary entries, identical encoding")
    t_params = sum(q.numel() for q in teacher.parameters())
    print(f"teacher     {t_params:>14,} packed 4-bit params (logical count is about 2x)")
    print(f"the student trains {rows['trainable']:,} parameters to chase it")
    print(f"\\nVRAM: {torch.cuda.memory_allocated()/2**30:.2f} GiB allocated, "
          f"{torch.cuda.max_memory_allocated()/2**30:.2f} GiB peak")
""")

md("""
## 4 — Data

A four-way interleaved mixture, streamed. Every position is supervised here,
unlike the agentic path where roughly 10% of tokens were — that density is most
of why distillation moves faster per step.

The held-out set is materialised into a list on purpose: both arms of the
comparison must be scored on the *same* chunks in the *same* order, and a
generator would hand the second arm an empty stream.
""")

code("""
from odydora.distill import DEFAULT_MIXTURE, stream_text_chunks
from odydora.data import describe_trace, stream_traces

if MODE == "distill":
    print("corpus mixture (round-robin, weight = rows per round):")
    for repo, config, split, field, weight in DEFAULT_MIXTURE:
        print(f"  {weight}x  {repo}" + (f" [{config}]" if config else ""))

    def chunks(skip=0, limit=None):
        return stream_text_chunks(tok, CFG["chunk"], max_chunks=limit,
                                  skip_chunks=skip, log_every=100)

    # Held out by position in the stream, after every training chunk. The
    # mixture is interleaved, so a tail slice still spans all four domains.
    print(f"\\nbuilding the held-out set (skipping {CFG['train_chunks']} training chunks)", flush=True)
    VAL = []
    for chunk in chunks(skip=CFG["train_chunks"], limit=CFG["val_chunks"]):
        VAL.append(chunk)
        if len(VAL) % 20 == 0:
            print(f"  {len(VAL)}/{CFG['val_chunks']} val chunks", flush=True)
    val_tokens = sum(int(c[0].numel()) for c in VAL)
    print(f"val: {len(VAL)} chunks, {val_tokens:,} tokens, every position supervised")
else:
    import datasets, statistics
    print("probing 10 traces for the length histogram", flush=True)
    probe = datasets.load_dataset(SFT_DATASET, split=SFT_SPLIT, streaming=True)
    lens, sup = [], []
    for i, row in enumerate(probe):
        if i >= 10: break
        d = describe_trace(row, tok)
        lens.append(d["tokens"]); sup.append(d["supervised_fraction"])
        print(f"  trace {i+1:>2}/10  {d['tokens']:>7,} tokens  "
              f"{d['supervised_fraction']:>5.1%} supervised", flush=True)
    print(f"tokens/trace  median {int(statistics.median(lens)):,}")
    print(f"supervised    {statistics.mean(sup):.1%} of tokens")

    def chunks(skip=0, limit=None):
        return stream_traces(tok, CFG["chunk"], dataset=SFT_DATASET, split=SFT_SPLIT,
                             max_trace_tokens=CFG["max_trace_tokens"],
                             max_tokens=CFG.get("max_tokens"),
                             skip_traces=skip, max_traces=limit)

    VAL = list(stream_traces(tok, CFG["chunk"], dataset=SFT_DATASET, split=SFT_SPLIT,
                             max_trace_tokens=CFG["max_trace_tokens"],
                             max_tokens=CFG["val_traces"] * 2000,
                             skip_traces=CFG["traces"]))
    print(f"val: {len(VAL)} chunks from traces {CFG['traces']}..{CFG['traces']+CFG['val_traces']}")
""")

md("""
## 5 — Train

Resumes from Drive if a checkpoint is there. Read `tok_s` on the first log line
— that replaces every throughput estimate in this notebook with your own
number.

What each column means while it runs:

- **`kl`** — bits from the student's distribution to the teacher's. The number
  being minimised. It cannot go below 0.
- **`hard`** — the student's own cross-entropy on the realised token, in bits.
  Printed even when `HARD_WEIGHT` is small, because a KL that falls while this
  rises is a student drifting toward the teacher's *shape* and away from the
  text.
- **`conf`** — the teacher's mean top-`k` mass on the kept positions. Near 1.0
  means the teacher is confident and the target is sharp.
- **`kept`** — the fraction of positions that survived `MIN_CONF`. **Watch this
  one.** A confidence floor that quietly discards most of the batch looks
  identical to healthy training in the `kl` column while almost nothing
  actually trains; if `kept` is below about 0.3, lower `MIN_CONF`.
- **`|s|`** — recurrent state norm. Bounded by `sqrt(N)` = 22.6 at `N=512` by
  construction, so a finite value is expected rather than reassuring; what
  matters is that it is *non-zero*, meaning state really is being carried.
- **`mix`** — the convex mixing rate. It starts at 0.02 and **must move**, or
  the core has not learned its own time constant.
""")

code("""
import time, torch
from odydora.train import configure_optimizer, restore, save
from odydora.distill import distill_steps, evaluate_gap, teacher_reference_bits

tag = f"{MODE[0].upper()}{STAGE}_{STUDENT.split('/')[-1]}_N{N}p{PROJ_RANK}r{RANK}"
CKPT = f"{CKPT_DIR}/{tag}.pt"
start = {"step": 0, "traces": 0, "tokens": 0}

if os.path.exists(CKPT) and RESUME:
    start = restore(model, CKPT)
    print(f"resumed: step {start['step']}, {start['tokens']:,} tokens, cursor {start['traces']}")
else:
    if os.path.exists(CKPT):
        import datetime, shutil
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(CKPT)
        bak = f"{base}_prefresh_bak_{ts}{ext}"
        shutil.copy2(CKPT, bak)
        print(f"RESUME=False: backed up -> {os.path.basename(bak)}")
    if "_INIT_STATE" in globals():
        model.load_state_dict(_INIT_STATE, strict=False)
    model.odydora_reset_state()
    print("fresh run: in-memory weights reset to step 0")

# After restoring or resetting, so AdamW moments match the weights they act on.
opt = configure_optimizer(model, lr=LR)

def fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

if MODE == "distill":
    print("\\nteacher's own cross-entropy on the held-out chunks (the floor)", flush=True)
    FLOOR = teacher_reference_bits(teacher, VAL, log_every=20)
    print(f"teacher    {FLOOR['teacher_bits']:.4f} bits/token (tail {FLOOR['tail_teacher_bits']:.4f})")

    print("\\nstep-0 gap, before any training", flush=True)
    BASE = evaluate_gap(model, teacher, VAL, top_k=TOP_K, log_every=20, desc="baseline gap")
    print(f"gap        {BASE['kl_bits']:.4f} bits (tail {BASE['tail_kl_bits']:.4f})  <- the number to beat")
    print(f"student    {BASE['student_bits']:.4f} bits/token vs teacher's {FLOOR['teacher_bits']:.4f}")

    stream = chunks(skip=start["traces"], limit=CFG["train_chunks"])
    rows, last_save = [], time.time()
    print()
    try:
        for r in distill_steps(model, teacher, stream, opt, top_k=TOP_K,
                               temperature=TEMPERATURE, hard_weight=HARD_WEIGHT,
                               min_teacher_conf=MIN_CONF,
                               log_every=CFG["log_every"], grad_accum=CFG["grad_accum"]):
            rows.append(r)
            print(f"  step {r['step']:>5}  kl {r['kl_bits']:.4f}  hard {r['hard_bits']:.4f}  "
                  f"conf {r['teacher_conf']:.3f}  kept {r['kept_fraction']:.2f}  "
                  f"{r['tok_s']:>6,.0f} tok/s  "
                  f"|s| {r['state_norm']:.2f}  mix {r['mix']:.4f}  "
                  f"[{fmt_time(r['elapsed'])}]", flush=True)
            if CFG["save_every_s"] and (time.time() - last_save) >= CFG["save_every_s"]:
                save(model, CKPT, step=start["step"] + r["step"],
                     traces=start["traces"] + r["step"], tokens=start["tokens"] + r["tokens"])
                last_save = time.time()
                print(f"  [saved -> {os.path.basename(CKPT)} at chunk {start['traces'] + r['step']}]", flush=True)
    except KeyboardInterrupt:
        print("\\n[interrupted] saving checkpoint before stopping...", flush=True)
else:
    from odydora.train import evaluate, train_steps
    BASE = evaluate(model, VAL, log_every=25, desc="baseline eval")
    print(f"step-0 held-out: {BASE['bits_per_token']:.4f} bits/token "
          f"(tail {BASE['tail_bits_per_token']:.4f})")
    stream = chunks(skip=start["traces"], limit=CFG["traces"])
    rows, last_save = [], time.time()
    try:
        for r in train_steps(model, stream, opt, log_every=CFG["log_every"],
                             grad_accum=CFG["grad_accum"]):
            rows.append(r)
            print(f"  step {r['step']:>5}  loss {r['loss']:.4f}  {r['tok_s']:>7,.0f} tok/s  "
                  f"|s| {r['state_norm']:.3f}  |up| {r['up_norm']:.4f}  "
                  f"[{fmt_time(r['elapsed'])}]", flush=True)
            if CFG["save_every_s"] and (time.time() - last_save) >= CFG["save_every_s"]:
                save(model, CKPT, step=start["step"] + r["step"],
                     traces=start["traces"] + r["traces"], tokens=start["tokens"] + r["tokens"])
                last_save = time.time()
    except KeyboardInterrupt:
        print("\\n[interrupted] saving checkpoint before stopping...", flush=True)

last = rows[-1] if rows else {"step": 0, "tokens": 0}
save(model, CKPT, step=start["step"] + last["step"],
     traces=start["traces"] + last.get("traces", last["step"]),
     tokens=start["tokens"] + last["tokens"])
print(f"saved -> {CKPT}")
""")

md("""
## 6 — Did it work

The gap, overall and in the tail, against the teacher's own floor.

Read it in this order:

1. **Did the gap close at all?** `kl_bits` below its step-0 value.
2. **Did it close more in the tail?** If `tail` improved more than `overall`,
   the improvement is coming from carried state — the mechanism this project
   exists to test. If `overall` improved and `tail` did not, something else is
   doing the work and the claim does not hold yet.
3. **How far is the student from the teacher now?** `student_bits` against
   `teacher_bits`. The fraction of the original gap recovered is the headline.
4. **Did `mix` move?** A core still sitting at its initial 0.02 has not learned
   a time constant, whatever the loss did.
""")

code("""
if MODE == "distill":
    FINAL = evaluate_gap(model, teacher, VAL, top_k=TOP_K, log_every=20, desc="final gap")
    d_kl = BASE["kl_bits"] - FINAL["kl_bits"]
    d_tail = BASE["tail_kl_bits"] - FINAL["tail_kl_bits"]
    d_hard = BASE["student_bits"] - FINAL["student_bits"]

    print("{:<22} {:>9} {:>9} {:>10}".format("", "step 0", "now", "delta"))
    print("{:<22} {:>9.4f} {:>9.4f} {:>+10.4f}".format(
        "gap to teacher", BASE["kl_bits"], FINAL["kl_bits"], d_kl))
    print("{:<22} {:>9.4f} {:>9.4f} {:>+10.4f}".format(
        "  tail 25%", BASE["tail_kl_bits"], FINAL["tail_kl_bits"], d_tail))
    print("{:<22} {:>9.4f} {:>9.4f} {:>+10.4f}".format(
        "student bits/token", BASE["student_bits"], FINAL["student_bits"], d_hard))
    print("{:<22} {:>9.4f}".format("teacher bits/token", FLOOR["teacher_bits"]))

    recovered = d_kl / BASE["kl_bits"] if BASE["kl_bits"] > 0 else float("nan")
    tail_recovered = d_tail / BASE["tail_kl_bits"] if BASE["tail_kl_bits"] > 0 else float("nan")
    print(f"\\nrecovered {recovered:.1%} of the distribution gap "
          f"({tail_recovered:.1%} in the tail) on {report(model)['trainable']:,} trainable parameters")
    print("tail beat overall: "
          + ("YES - consistent with carried state doing the work"
             if tail_recovered > recovered
             else "NO - the gain is not coming from state carry"))

    core = getattr(model, "odydora_core", None)
    if core is not None:
        print(f"\\nmix {core.mix.detach().item():.4f} (init {MIX})   "
              f"|up| {core.up.weight.detach().norm():.4f} (init 0.0)")
        print("mix moved: " + ("YES" if abs(core.mix.detach().item() - MIX) > 1e-4
                               else "NO - the core has not learned a time constant"))
    else:
        print("\\ncore: None (adapter-only control arm)")
    print(f"\\nD-gate: {'PASS' if d_kl > 0 else 'FAIL - the gap did not close'}")
else:
    from odydora.train import evaluate
    FINAL = evaluate(model, VAL, log_every=25, desc="final eval")
    delta = BASE["bits_per_token"] - FINAL["bits_per_token"]
    tail_delta = BASE["tail_bits_per_token"] - FINAL["tail_bits_per_token"]
    print("{:<12} {:>8} {:>8} {:>9}".format("", "step 0", "now", "delta"))
    print("{:<12} {:>8.4f} {:>8.4f} {:>+9.4f}".format(
        "overall", BASE["bits_per_token"], FINAL["bits_per_token"], delta))
    print("{:<12} {:>8.4f} {:>8.4f} {:>+9.4f}".format(
        "tail 25%", BASE["tail_bits_per_token"], FINAL["tail_bits_per_token"], tail_delta))
    print(f"\\nS-gate: {'PASS' if delta > 0 else 'FAIL'}")
""")

md("""
## 7 — The comparison arm

A parameter-efficiency claim is a comparison, and this cell computes the rank
whose adapter-only budget matches the treatment's — then tells you exactly what
to change to run it.

**Report the gap including when it is negative.** A core that loses to a
matched LoRA on the same chunks in the same order is a real finding and the
most valuable thing this notebook can produce.

The control arm runs through the *same* `distill_steps` and `evaluate_gap` with
`N=0`, so nothing about the comparison depends on a second code path.
""")

code("""
from odydora.adapter import count_parameters
from odydora.train import adapter_only_params

_, core_budget = count_parameters(model)
total_budget = report(model)["trainable"]

print(f"treatment: core {core_budget:,} + adapter r={RANK} = {total_budget:,} trainable")
print(f"\\nadapter-only ranks on {len(LORA_ON)} projections:")
best = None
for r in range(1, 17):
    got = adapter_only_params(model, r, use_dora=USE_DORA, targets=LORA_ON)
    if best is None or abs(got - total_budget) < abs(best[1] - total_budget):
        best = (r, got)
    print(f"  r={r:<3} {got:>12,}   {got/total_budget:.2f}x the treatment")
    if got > total_budget * 1.6:
        break
print(f"\\nclosest match: r={best[0]} -> {best[1]:,} "
      f"({abs(best[1]-total_budget)/total_budget:.1%} off, "
      f"{'more' if best[1] > total_budget else 'fewer'} parameters than the treatment)")
print("\\nTo run it:")
print(f"  1. Cell 2: set N = 0 and RANK = {best[0]}")
print(f"  2. Run Cells 2-6. Same corpus, same order, same teacher, same steps.")
print(f"  3. Compare tail_kl_bits. That is the whole claim.")
""")

md("""
## 8 — Checkpoint Manager (Drive)

Interactive form to inspect, rename, and manage checkpoints stored in Drive
(`CKPT_DIR`). Select an action below and run the cell.
""")

code("""
# @title 📁 Checkpoint Manager { display-mode: "form" }
# @markdown Select an action and specify target filenames if applicable:

import datetime, glob, os, shutil, torch
from odydora.train import inspect_checkpoint, restore

action = "List Checkpoints"  # @param ["List Checkpoints", "Set as Active", "Archive Active (Start Fresh)", "Inspect Checkpoint", "Rename Checkpoint", "Delete Checkpoint"]
target_checkpoint = ""  # @param {type:"string"}
new_name = ""  # @param {type:"string"}
confirm_delete = False  # @param {type:"boolean"}

os.makedirs(CKPT_DIR, exist_ok=True)
files = sorted(glob.glob(os.path.join(CKPT_DIR, "*.pt")), key=os.path.getmtime, reverse=True)
active_name = f"{tag}.pt" if "tag" in globals() else ""

if action == "List Checkpoints":
    if not files:
        print(f"No .pt checkpoints found in {CKPT_DIR}")
    else:
        if active_name:
            print(f"Active config checkpoint: {active_name}\\n")
        print("{:<9} {:<40} {:<16} {:>9}   {:<19} {:>6} {:>8} {:>12}".format(
            "Status", "Filename", "Arch", "Size", "Last Modified", "Step", "Cursor", "Tokens"
        ))
        print("-" * 128)
        for f in files:
            bn = os.path.basename(f)
            stat = os.stat(f)
            size_mb = f"{stat.st_size / (1024 * 1024):.1f} MB"
            mod = datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            status = "[ACTIVE]" if (active_name and bn == active_name) else ""
            try:
                meta = inspect_checkpoint(f)
                arch = f"N={meta['n']} r={meta['r']}"
                step = str(meta.get("step", "-"))
                cursor = str(meta.get("traces", "-"))
                tokens = f"{meta.get('tokens', 0):,}" if "tokens" in meta else "-"
            except Exception:
                arch, step, cursor, tokens = "err", "err", "err", "err"
            print(f"{status:<9} {bn:<40} {arch:<16} {size_mb:>9}   {mod:<19} {step:>6} {cursor:>8} {tokens:>12}")

elif action == "Set as Active":
    if not active_name:
        print("Error: run Cell 2 and Cell 5 first so the active config is known.")
    else:
        active_path = os.path.join(CKPT_DIR, active_name)
        src = os.path.join(CKPT_DIR, target_checkpoint.strip())
        if not os.path.exists(src):
            print(f"Error: '{target_checkpoint}' not found in {CKPT_DIR}")
            print("Available:", [os.path.basename(f) for f in files])
        elif os.path.abspath(src) == os.path.abspath(active_path):
            print(f"'{target_checkpoint}' is already active for this config.")
        else:
            # Never overwrite the active checkpoint without preserving it.
            if os.path.exists(active_path):
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                base, ext = os.path.splitext(active_name)
                backup_name = f"{base}_bak_{ts}{ext}"
                shutil.copy2(active_path, os.path.join(CKPT_DIR, backup_name))
                print(f"Safety backup created: {backup_name}")
            shutil.copy2(src, active_path)
            print(f"Success: '{os.path.basename(src)}' is now active ({active_name}).")

elif action == "Archive Active (Start Fresh)":
    if not active_name:
        print("Error: run Cell 2 and Cell 5 first so the active config is known.")
    else:
        active_path = os.path.join(CKPT_DIR, active_name)
        if not os.path.exists(active_path):
            print(f"'{active_name}' does not exist on Drive. Already set for a fresh run.")
        else:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base, ext = os.path.splitext(active_name)
            archived_name = f"{base}_archived_{ts}{ext}"
            shutil.move(active_path, os.path.join(CKPT_DIR, archived_name))
            if "model" in globals() and "_INIT_STATE" in globals():
                model.load_state_dict(_INIT_STATE, strict=False)
                model.odydora_reset_state()
            print(f"Archived '{active_name}' -> '{archived_name}'")
            print("Slot clear and in-memory weights reset. Cell 5 starts fresh from step 0.")

elif action == "Inspect Checkpoint":
    target_path = os.path.join(CKPT_DIR, target_checkpoint.strip())
    if not os.path.exists(target_path):
        print(f"Error: '{target_checkpoint}' not found in {CKPT_DIR}")
        print("Available:", [os.path.basename(f) for f in files])
    else:
        blob = torch.load(target_path, map_location="cpu", weights_only=True)
        print(f"=== {os.path.basename(target_path)} ===")
        for k, v in blob.items():
            if k != "trainable":
                print(f"  {k}: {v}")
        trainable = blob.get("trainable", {})
        total_p = sum(t.numel() for t in trainable.values())
        print(f"\\nTrainable tensors ({len(trainable)} tensors, {total_p:,} parameters):")
        for k, v in sorted(trainable.items()):
            print(f"  {k:<58} {str(tuple(v.shape)):<18} {v.numel():>10,}")

elif action == "Rename Checkpoint":
    src = os.path.join(CKPT_DIR, target_checkpoint.strip())
    dst_name = new_name.strip()
    if not dst_name.endswith(".pt"):
        dst_name += ".pt"
    dst = os.path.join(CKPT_DIR, dst_name)
    if not os.path.exists(src):
        print(f"Error: '{target_checkpoint}' not found.")
    elif os.path.exists(dst):
        print(f"Error: '{dst_name}' already exists.")
    else:
        os.rename(src, dst)
        print(f"Renamed: {os.path.basename(src)} -> {dst_name}")

elif action == "Delete Checkpoint":
    target_path = os.path.join(CKPT_DIR, target_checkpoint.strip())
    if not os.path.exists(target_path):
        print(f"Error: '{target_checkpoint}' not found in {CKPT_DIR}")
    elif not confirm_delete:
        print(f"Safety check: tick 'confirm_delete' to delete '{target_checkpoint}'.")
    else:
        os.remove(target_path)
        reset_now = (active_name and os.path.basename(target_path) == active_name
                     and "model" in globals() and "_INIT_STATE" in globals())
        if reset_now:
            model.load_state_dict(_INIT_STATE, strict=False)
            model.odydora_reset_state()
            print(f"Deleted '{target_checkpoint}' and reset in-memory model to step 0")
        else:
            print(f"Deleted '{target_checkpoint}'")
""", metadata={"cellView": "form"})

md("""
## 9 — Interactive Chat (Odydora Playground)

Chat with the student and watch the recurrent core steer it. Tick
`compare_without_odydora` to hear the same prompt from the frozen host with
both the core and the adapter bypassed — same RNG trajectory, so any difference
is the mechanism rather than sampling noise.

`|s|` at the end of each turn is the state norm carried into the next turn.
""")

code("""
# @title 💬 Chat with Odydora { display-mode: "form" }
# @markdown Enter your message and generation settings, then run the cell:

user_message = "Explain why carrying a recurrent state across chunk boundaries beats standard LoRA."  # @param {type:"string"}
system_prompt = "You are a helpful AI assistant. Reason step by step inside <think>...</think> tags, then give a clear, direct answer."  # @param {type:"string"}
compare_without_odydora = False  # @param {type:"boolean"}
unlimited_tokens = True  # @param {type:"boolean"}
max_new_tokens = 512  # @param {type:"slider", min:64, max:4096, step:64}
temperature = 0.7  # @param {type:"slider", min:0.0, max:1.5, step:0.05}
top_p = 0.9  # @param {type:"slider", min:0.1, max:1.0, step:0.05}
clear_history = False  # @param {type:"boolean"}

import os, time as _time, torch
from transformers import TextStreamer

if "CHAT_HISTORY" not in globals() or clear_history:
    CHAT_HISTORY = []
    if "model" in globals() and hasattr(model, "odydora_reset_state"):
        model.odydora_reset_state()
    print("[Chat history cleared and recurrent state reset.]\\n")

# Cold start: someone opened the notebook and came straight here.
if "model" not in globals() or "tok" not in globals():
    print(f"Loading {STUDENT} and the active checkpoint from Drive...")
    from odydora.train import build, inspect_checkpoint, restore
    active_ckpt = f"{CKPT_DIR}/{MODE[0].upper()}{STAGE}_{STUDENT.split('/')[-1]}_N{N}p{PROJ_RANK}r{RANK}.pt"
    build_n, build_r, build_dora = N, RANK, USE_DORA
    if os.path.exists(active_ckpt):
        info = inspect_checkpoint(active_ckpt)
        build_n, build_r, build_dora = info["n"], info["r"] or RANK, info["use_dora"]
    model, tok = build(STUDENT, n=build_n, layers=LAYERS, stride=STRIDE, mix=MIX,
                       window=WINDOW, use_dora=build_dora, r=build_r, four_bit=True,
                       proj_rank=PROJ_RANK, lora_targets=LORA_ON)
    if os.path.exists(active_ckpt):
        meta = restore(model, active_ckpt)
        print(f"Loaded {os.path.basename(active_ckpt)} (step {meta.get('step', '-')}, N={build_n})\\n")
    else:
        print(f"Note: no checkpoint at {active_ckpt}; using a freshly initialized core.\\n")

messages = [{"role": "system", "content": system_prompt}]
messages.extend(CHAT_HISTORY)
messages.append({"role": "user", "content": user_message})

prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
device = next(model.parameters()).device
inputs = tok(prompt_text, return_tensors="pt").to(device)
gen_limit = 4096 if unlimited_tokens else max_new_tokens

print(f"User:\\n{user_message}\\n")
print("Odydora (Recurrent Core):")

streamer = TextStreamer(tok, skip_prompt=True)
model.eval()

# Snapshot RNG so the comparison pass samples the same noise trajectory.
cpu_rng = torch.get_rng_state()
cuda_rng = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

_t0 = _time.time()
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=gen_limit,
        temperature=temperature if temperature > 0 else None,
        top_p=top_p if temperature > 0 else None,
        do_sample=temperature > 0,
        repetition_penalty=1.1,
        stop_strings=["</tool_call>", "<|im_end|>"],
        tokenizer=tok,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        streamer=streamer,
    )
_gen_dt = max(_time.time() - _t0, 1e-6)

new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
assistant_reply = tok.decode(new_tokens, skip_special_tokens=True).strip()

if compare_without_odydora:
    print("\\n" + "-" * 60)
    print("Without Odydora (frozen host, core and adapter bypassed):")
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state(cuda_rng)
    streamer_base = TextStreamer(tok, skip_prompt=True)
    with torch.no_grad(), model.without_odydora():
        model.generate(
            **inputs,
            max_new_tokens=gen_limit,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
            do_sample=temperature > 0,
            repetition_penalty=1.1,
            stop_strings=["</tool_call>", "<|im_end|>"],
            tokenizer=tok,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
            streamer=streamer_base,
        )

CHAT_HISTORY.append({"role": "user", "content": user_message})
CHAT_HISTORY.append({"role": "assistant", "content": assistant_reply})

layers = getattr(model, "odydora_layers", ())
if layers:
    states = [s for s in model.odydora_state().values() if s is not None]
    last_s = model.odydora_state().get(layers[-1])
    s_norm = f"{last_s.norm().item():.3f}" if last_s is not None else "0.0"
    mean_s = f"{sum(s.norm().item() for s in states)/len(states):.3f}" if states else "0.0"
    n_tok = int(new_tokens.numel())
    print(f"\\n[Turn {len(CHAT_HISTORY)//2} | {n_tok} tok ({n_tok/_gen_dt:.1f} tok/s) | "
          f"state |s|: {s_norm} (mean {mean_s} across {len(layers)} layers)]")
""", metadata={"cellView": "form"})

nb = {
    "cells": CELLS,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = pathlib.Path(__file__).resolve().parents[1] / "notebooks" / "odydora_colab.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {out}  ({len(CELLS)} cells, "
      f"{sum(1 for c in CELLS if c['cell_type']=='code')} code)")
