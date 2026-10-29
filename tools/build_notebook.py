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
# Odydora — a shared cross-layer recurrent core as a parameter-efficient adapter

One tiny `N x N` matrix, shared by every decoder layer, carrying state across
layers **and** across chunk boundaries. The frozen 4-bit transformer is the
limbic system; the core is the frontal cortex steering it.

The core architecture is borrowed from
**[OdyssNet](https://github.com/theomgdev/OdyssNet)** (Karahan, v4.0.0, MIT):
one shared `N x N` matrix with a hard-zero diagonal under per-step RMSNorm,
with the state detached at the call boundary. Odydora adapts it to steer a
frozen host at segment rate. Reference, not a dependency — cite both
(`CITATION.cff`).

**Watch one number per stage.** S1: held-out bits/token below the step-0 value.
S2: tail bits/token below the matched-parameter adapter-only arm. Nothing else
matters until those move.

| Stage | What it is | Budget | Gate to advance |
|---|---|---|---|
| **S1** | first signal, 160 traces truncated to 4k tokens | about 30 min | held-out loss < step-0 loss; `state_norm` finite; `mix` moved |
| **S2** | long carry, full traces in 1k chunks, Drive-resumed | 2-3 h / session | tail bits/token beats the adapter-only arm at matched parameters |
| **S3** | push whichever axis S2 names | open | this is where a claim gets written |

### Numbers measured before you run this, so you can tell a surprise from a bug

Taken on an RTX 3060 Ti (sm_86) against real 2.03B fp16 weights, T=512,
gradient checkpointing on. A T4 is slower; these are ratios to expect, not
absolutes.

- Baseline without the core: **1,786 tok/s**. Core at all 28 layers:
  **28 tok/s at stride 1**, **660 at stride 64**, **816 at stride 256** — the
  step is kernel-launch-bound, so cost tracks the number of state updates and
  not the arithmetic. `STRIDE` is the throughput lever.
- The state sits on the `sqrt(N)` sphere exactly (8.000 at N=64, constant over
  4,000 steps) because RMSNorm runs every step — OdyssNet's mechanism, and the
  reason no spectral constraint is needed. Explosion is structurally
  impossible; **forgetting** is the real failure, which is what `MIX` exists to
  prevent (influence after 1,024 steps: 5.8e-01 at mix 0.02, 2.0e-34 at 1.0).
- Core parameters are **independent of depth**: 266,240 at N=64 whether 1 layer
  or 28 use it, because `W`, `down` and `up` are all shared.
- The corpus `train` split is the clean one. `full` has twice the rows and is
  unusable: no `tools`, 64 of 65 tool calls degraded to `"unknown_tool"`.
- Real trace lengths: median 32,480 tokens, mean 31,822 — the card's "about 50,000
  average" is the tail. Only **10.1%** of tokens are supervised, so a token
  budget is tokens *seen*, not tokens *learned from*.
""")

md("""
## 1 — Setup

Installs, repo, Drive. `datasets>=4.8` is enough: this corpus's `Json` feature
type streams fine on 4.8.5, tested directly.

`pip` will warn that `gradio`, `gradio-client` and `diffusers` want
`huggingface-hub>=1.16` while this cell resolves `0.36.x`. **Expected and
harmless here:** `transformers 4.x` requires `huggingface-hub<1.0`, so pip pins
the hub down, and the three complaining packages are preinstalled in the Colab
image, while nothing in this notebook imports them. We hold `transformers<5` on
purpose, since 4-bit loading through `peft` and `bitsandbytes` is tested against
the 4.x API.
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

**The only cell you edit.** Every value has a reason; the ones marked MEASURED
came from a probe, not a guess.
""")

code("""
STAGE = "S1"            # "S1" first signal | "S2" long carry | "S3" push

MODEL = "Qwen/Qwen3-1.7B"       # 2,031,739,904 params, 28 layers, d=2048, Apache 2.0,
                                # non-gated, native `tools` + <think> template.
                                # Fast rung: "Qwen/Qwen3-0.6B".
                                # Coding specialist: "Qwen/Qwen2.5-Coder-1.5B-Instruct".

N       = 64            # core width. 266,240 params at d=2048, depth-independent.
STRIDE  = 64            # tokens per state update. MEASURED: 28 tok/s at 1, 660 at 64.
MIX     = 0.02          # initial convex mixing rate. MEASURED: smaller remembers longer.
LAYERS  = None          # None = every decoder boundary. (6,13,20,27) is the cheap arm.
WINDOW  = 0             # intra-chunk detach. MEASURED: no memory saved, so off.

USE_DORA = False        # LoRA first: QDoRA dequantizes the frozen 4-bit weight every
                        # forward (2.0-2.4x slower, rank-invariant). DoRA is an S3 arm.
RANK     = 32
LR       = 2e-4

STAGES = {
    "S1": dict(chunk=512,  max_trace_tokens=4096, traces=160, val_traces=40,
               grad_accum=8,  log_every=10, save_every_s=900),
    "S2": dict(chunk=1024, max_trace_tokens=None, traces=2000, val_traces=50,
               grad_accum=16, log_every=25, save_every_s=900),
    "S3": dict(chunk=1024, max_trace_tokens=None, traces=5000, val_traces=50,
               grad_accum=16, log_every=50, save_every_s=900),
}
CFG = STAGES[STAGE]
DATASET, SPLIT = "MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x", "train"

print(f"{STAGE}: {MODEL}  N={N} stride={STRIDE} mix={MIX}  "
      f"chunk={CFG['chunk']}  traces={CFG['traces']}")
""")

md("""
## 3 — Build

Adapter on the projections first, then the core hooked onto the layer
boundaries, so the core sees the adapted hidden state. The printed counts are
the parameter-efficiency numerator — check the base really is frozen.
""")

code("""
from odydora.train import build, report

model, tok = build(MODEL, n=N, layers=LAYERS, stride=STRIDE, mix=MIX, window=WINDOW,
                   use_dora=USE_DORA, r=RANK, four_bit=True)

rows = report(model)
print(f"total       {rows['total']:>14,}")
print(f"trainable   {rows['trainable']:>14,}   ({rows['trainable_fraction']:.4%})")
print(f"  adapter   {rows['adapter']:>14,}")
print(f"  core      {rows['core']:>14,}   shared by {len(model.odydora_layers)} layer(s)")
print(f"\\ncore: {model.odydora_core}")
""")

md("""
## 4 — Data

Streams the clean split. The histogram is worth a look: if the median is far
from about 32k tokens the corpus changed under us.
""")

code("""
import datasets, statistics
from odydora.data import describe_trace, stream_traces

# Every long step prints as it goes. A cell that sits silent for minutes is
# indistinguishable from a hung one, and there is no way to tell whether the
# work is progressing or wedged.
print("probing 10 traces for the length histogram", flush=True)
probe = datasets.load_dataset(DATASET, split=SPLIT, streaming=True)
lens, sup = [], []
for i, row in enumerate(probe):
    if i >= 10: break
    d = describe_trace(row, tok)
    lens.append(d["tokens"]); sup.append(d["supervised_fraction"])
    print(f"  trace {i+1:>2}/10  {d['tokens']:>7,} tokens  "
          f"{d['supervised_fraction']:>5.1%} supervised", flush=True)
print(f"tokens/trace  min {min(lens):,}  median {int(statistics.median(lens)):,}  max {max(lens):,}")
print(f"supervised    {statistics.mean(sup):.1%} of tokens (assistant turns only)")

def batches(skip=0, limit=None, max_tokens=None):
    return stream_traces(tok, CFG["chunk"], dataset=DATASET, split=SPLIT,
                         max_trace_tokens=CFG["max_trace_tokens"],
                         max_tokens=max_tokens, skip_traces=skip,
                         max_traces=limit)

# Held-out traces, never held-out chunks: carried state would leak across the split.
print(f"\\nbuilding the held-out set (skipping {CFG['traces']} training traces)", flush=True)
VAL = []
for chunk in batches(skip=CFG["traces"], max_tokens=CFG["val_traces"] * 2000):
    VAL.append(chunk)
    if len(VAL) % 25 == 0:
        print(f"  {len(VAL)} val chunks", flush=True)
print(f"val: {len(VAL)} chunks from traces {CFG['traces']}..{CFG['traces']+CFG['val_traces']}")
""")

md("""
## 5 — Train

Resumes from Drive if a checkpoint is there. Read `tok_s` on the first log line
— that replaces every throughput estimate in this notebook with your own
number. Watch `state_norm` stays finite and `mix` moves off its initial value.
""")

code("""
import itertools, time, torch
from odydora.train import evaluate, restore, save, train_steps

CKPT = f"{CKPT_DIR}/{STAGE}_{MODEL.split('/')[-1]}_N{N}s{STRIDE}.pt"
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)

start = {"step": 0, "traces": 0, "tokens": 0}
if os.path.exists(CKPT):
    start = restore(model, CKPT)
    print(f"resumed: step {start['step']}, {start['tokens']:,} tokens, "
          f"trace cursor {start['traces']}")

base_eval = evaluate(model, VAL, log_every=25, desc="baseline eval")
print(f"step-0 held-out: {base_eval['bits_per_token']:.4f} bits/token "
      f"(tail {base_eval['tail_bits_per_token']:.4f})  <- the number to beat")

def fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

stream = batches(skip=start["traces"], limit=CFG["traces"])
rows = []
try:
    for r in train_steps(model, stream, opt, log_every=CFG["log_every"],
                         grad_accum=CFG["grad_accum"]):
        rows.append(r)
        print(f"  step {r['step']:>5}  loss {r['loss']:.4f}  {r['tok_s']:>7,.0f} tok/s  "
              f"|s| {r['state_norm']:.3f}  |up| {r['up_norm']:.4f}  "
              f"[{fmt_time(r['elapsed'])}]", flush=True)
except KeyboardInterrupt:
    print("\\n[interrupted by user] stopping training early and saving checkpoint...", flush=True)

save(model, CKPT, step=start["step"] + (rows[-1]["step"] if rows else 0),
     traces=start["traces"] + CFG["traces"], tokens=start["tokens"] +
     (rows[-1]["tokens"] if rows else 0))
print(f"saved -> {CKPT}")
""")

md("""
## 6 — Did it work

Overall **and** tail bits/token. The tail is the honest one: carrying state can
only pay where a chunk has history behind it, so a gain that shows up overall
but not in the tail is not the mechanism working.
""")

code("""
final = evaluate(model, VAL, log_every=25, desc="final eval")
delta = base_eval["bits_per_token"] - final["bits_per_token"]
tail_delta = base_eval["tail_bits_per_token"] - final["tail_bits_per_token"]

print(f"            step 0     now      delta")
print(f"overall     {base_eval['bits_per_token']:.4f}   {final['bits_per_token']:.4f}   {delta:+.4f}")
print(f"tail 25%    {base_eval['tail_bits_per_token']:.4f}   {final['tail_bits_per_token']:.4f}   {tail_delta:+.4f}")
print(f"\\nmix {model.odydora_core.mix.detach().item():.4f} (init {MIX})   "
      f"|up| {model.odydora_core.up.weight.detach().norm():.4f} (init 0.0)")
print(f"\\nS1 gate: {'PASS' if delta > 0 else 'FAIL - the core is not helping yet'}")
""")

md("""
## 7 — The comparison arm (run after S2, not before)

A parameter-efficiency claim is a comparison. This trains adapter-only at the
rank whose trainable count matches the core's budget, on the same traces in the
same order. **Report the gap including when it is negative** — a core that loses
to a matched LoRA is a real finding.
""")

code("""
from odydora.adapter import count_parameters
from odydora.train import matched_rank

_, core_budget = count_parameters(model)
rank, matched = matched_rank(model, core_budget, use_dora=USE_DORA)
print(f"core budget {core_budget:,} -> adapter-only rank {rank} gives {matched:,} "
      f"({abs(matched-core_budget)/core_budget:.1%} off)")
print("Now re-run cells 3-6 with N=0 disabled and RANK set to this rank, then")
print("compare tail bits/token. Same seed, same traces, same step count.")
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
from odydora.train import restore

action = "List Checkpoints"  # @param ["List Checkpoints", "Set as Active", "Inspect Checkpoint", "Rename Checkpoint", "Delete Checkpoint"]
target_checkpoint = "S1_Qwen3-1.7B_N64s64.pt"  # @param {type:"string"}
new_name = ""  # @param {type:"string"}
confirm_delete = False  # @param {type:"boolean"}

os.makedirs(CKPT_DIR, exist_ok=True)
files = sorted(glob.glob(os.path.join(CKPT_DIR, "*.pt")), key=os.path.getmtime, reverse=True)
active_name = f"{STAGE}_{MODEL.split('/')[-1]}_N{N}s{STRIDE}.pt" if "STAGE" in globals() and "MODEL" in globals() else ""

if action == "List Checkpoints":
    if not files:
        print(f"No .pt checkpoints found in {CKPT_DIR}")
    else:
        if active_name:
            print(f"Active config checkpoint: {active_name}\\n")
        print(f"{\"Status\":<9} {\"Filename\":<36} {\"Size\":>9}   {\"Last Modified\":<19} {\"Step\":>6} {\"Traces\":>8} {\"Tokens\":>10}")
        print("-" * 105)
        for f in files:
            bn = os.path.basename(f)
            stat = os.stat(f)
            size_mb = f"{stat.st_size / (1024 * 1024):.1f} MB"
            mod = datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            status = "[ACTIVE]" if (active_name and bn == active_name) else ""
            try:
                meta = restore(None, f)
                step = str(meta.get("step", "-"))
                traces = str(meta.get("traces", "-"))
                tokens = f"{meta.get('tokens', 0):,}" if "tokens" in meta else "-"
            except Exception:
                step, traces, tokens = "err", "err", "err"
            print(f"{status:<9} {bn:<36} {size_mb:>9}   {mod:<19} {step:>6} {traces:>8} {tokens:>10}")

elif action == "Set as Active":
    if not active_name:
        print("Error: please run Cell 2 (Config) first to identify the active model configuration.")
    else:
        active_path = os.path.join(CKPT_DIR, active_name)
        src = os.path.join(CKPT_DIR, target_checkpoint.strip())
        if not os.path.exists(src):
            print(f"Error: target checkpoint '{target_checkpoint}' not found in {CKPT_DIR}")
            print("Available files:", [os.path.basename(f) for f in files])
        elif os.path.abspath(src) == os.path.abspath(active_path):
            print(f"'{target_checkpoint}' is already the active checkpoint for current config ({active_name}).")
        else:
            # Safety backup: preserve the existing active checkpoint before overwriting!
            if os.path.exists(active_path):
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                base, ext = os.path.splitext(active_name)
                backup_name = f"{base}_bak_{ts}{ext}"
                backup_path = os.path.join(CKPT_DIR, backup_name)
                shutil.copy2(active_path, backup_path)
                print(f"Safety backup created: {backup_name}")
            shutil.copy2(src, active_path)
            print(f"Success: set '{os.path.basename(src)}' as active checkpoint ({active_name}).")
            print("Cell 5 will resume from this checkpoint on its next run.")

elif action == "Inspect Checkpoint":
    target_path = os.path.join(CKPT_DIR, target_checkpoint.strip())
    if not os.path.exists(target_path):
        print(f"Error: checkpoint '{target_checkpoint}' not found in {CKPT_DIR}")
        print("Available files:", [os.path.basename(f) for f in files])
    else:
        blob = torch.load(target_path, map_location="cpu", weights_only=True)
        print(f"=== Checkpoint Metadata: {os.path.basename(target_path)} ===")
        for k, v in blob.items():
            if k != "trainable":
                print(f"  {k}: {v}")
        trainable = blob.get("trainable", {})
        total_p = sum(t.numel() for t in trainable.values())
        print(f"\\nTrainable tensors ({len(trainable)} tensors, {total_p:,} parameters):")
        for k, v in sorted(trainable.items()):
            print(f"  {k:<55} {str(tuple(v.shape)):<18} {v.numel():>10,} params")

elif action == "Rename Checkpoint":
    src = os.path.join(CKPT_DIR, target_checkpoint.strip())
    dst_name = new_name.strip()
    if not dst_name.endswith(".pt"):
        dst_name += ".pt"
    dst = os.path.join(CKPT_DIR, dst_name)
    if not os.path.exists(src):
        print(f"Error: source '{target_checkpoint}' not found.")
    elif os.path.exists(dst):
        print(f"Error: target '{dst_name}' already exists.")
    else:
        os.rename(src, dst)
        print(f"Renamed: {os.path.basename(src)} -> {dst_name}")

elif action == "Delete Checkpoint":
    target_path = os.path.join(CKPT_DIR, target_checkpoint.strip())
    if not os.path.exists(target_path):
        print(f"Error: '{target_checkpoint}' not found in {CKPT_DIR}")
    elif not confirm_delete:
        print(f"Safety check: tick the 'confirm_delete' checkbox to delete '{target_checkpoint}'.")
    else:
        os.remove(target_path)
        print(f"Deleted: '{target_checkpoint}' from {CKPT_DIR}")
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
