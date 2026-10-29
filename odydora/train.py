"""Build the model, train it in chunks, checkpoint so a disconnect is cheap.

The training step is ordinary next-token cross-entropy over one chunk at a
time. What is not ordinary is the state handling: the recurrent state is
reset at a trace boundary and carried (detached) between chunks of the same
trace, which is what lets a 50k-token document train in 1k-token pieces.
"""

import json
import os
import time

import torch

from .adapter import count_parameters

LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def lora_config(use_dora=True, r=32, dropout=0.05, targets=LORA_TARGETS):
    """DoRA by default; `use_dora=False` is the sanctioned LoRA fallback."""
    from peft import LoraConfig

    return LoraConfig(
        r=r,
        lora_alpha=2 * r,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(targets),
        use_dora=use_dora,
    )


def adapter_only_params(model_or_build, r, use_dora=True, targets=LORA_TARGETS):
    """Trainable count for an adapter-only arm at rank `r`, no core.

    Calculated directly from the target linear projection dimensions without
    calling PEFT's model wrapper dispatchers (which fail on environments with
    stale vendor dependencies like Colab's preinstalled torchao 0.10.0).
    """
    if isinstance(model_or_build, torch.nn.Module):
        model = model_or_build
    elif callable(model_or_build):
        model = model_or_build()
    else:
        raise TypeError(
            f"expected nn.Module or factory callable, got {type(model_or_build)}"
        )

    tot = 0
    for name, module in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in targets:
            base = getattr(module, "base_layer", module)
            if hasattr(base, "in_features") and hasattr(base, "out_features"):
                din, dout = base.in_features, base.out_features
                tot += (din + dout) * r
                if use_dora:
                    tot += dout
    return tot


def matched_rank(model_or_build, budget, use_dora=True, max_rank=512):
    """Smallest rank whose adapter-only count is closest to `budget`.

    This is what makes the parameter-efficiency claim a comparison rather
    than an assertion: the control arm gets the same trainable budget the
    treatment spends, so a win cannot come from simply having more weights.
    """
    if isinstance(model_or_build, torch.nn.Module):
        model = model_or_build
    elif callable(model_or_build):
        model = model_or_build()
    else:
        raise TypeError(
            f"expected nn.Module or factory callable, got {type(model_or_build)}"
        )

    best = (1, adapter_only_params(model, 1, use_dora))
    rank = 1
    while rank <= max_rank:
        count = adapter_only_params(model, rank, use_dora)
        if abs(count - budget) < abs(best[1] - budget):
            best = (rank, count)
        if count > budget:
            break
        rank *= 2
    return best


def build(
    model_id,
    n,
    layers=None,
    stride=64,
    mix=0.02,
    window=0,
    use_dora=False,
    r=32,
    four_bit=True,
    device_map="auto",
    attn_implementation="sdpa",
):
    """Load the base model in 4-bit, apply the adapter, attach the core.

    Order matters: the adapter wraps the projections first, then the core is
    hooked onto the layer boundaries, so the core sees the adapted hidden
    state rather than the frozen one.

    Defaults follow the measurements rather than the plan they came from.
    `use_dora=False`: QDoRA must dequantize the frozen 4-bit weight on every
    forward to take the column norm, which costs 2.0-2.4x training time and
    is invariant to rank, so DoRA belongs in a late ablation rather than in
    the first run. `attn_implementation="sdpa"`: FlashAttention-2 needs
    compute capability >= 8.0 and a T4 is 7.5, while eager wastes memory
    SDPA does not. `window=0`: intra-chunk detaching measured identical peak
    memory and only shortens credit assignment.
    """
    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from .adapter import attach_core

    quant = None
    if four_bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            # fp16 compute: a T4 is Turing and has no bfloat16 tensor cores.
            bnb_4bit_compute_dtype=torch.float16,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant,
        device_map=device_map,
        dtype=torch.float16 if four_bit else None,
        attn_implementation=attn_implementation,
    )
    if four_bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    model = get_peft_model(model, lora_config(use_dora=use_dora, r=r))
    model = attach_core(
        model, n=n, layers=layers, stride=stride, mix=mix, window=window,
        freeze_base=False,
    )
    return model, tokenizer


def configure_optimizer(model, lr=2e-4, core_lr_scale=5.0, weight_decay=0.01):
    """Build an AdamW optimizer with OdyssNet-style parameter family separation.

    Two structural asymmetries separate the shared RecurrentCore from the spatial
    LoRA adapters:
    1. Only connective matrices (`W`, `down`, `up`, `lora_A`, `lora_B`) decay
       toward zero. Modulation parameters (`core.alpha`, `core.norm.weight`,
       DoRA `lora_magnitude_vector`, biases) receive `weight_decay=0.0` because
       zero is not their neutral point: pulling `alpha = logit(0.02)` toward 0
       drives the mixing rate toward 0.50 (destroying long-horizon carry), and
       pulling `norm.weight = 1.0` toward 0 shrinks the sqrt(N) state sphere.
    2. The shared core sits behind a zero-initialised `up` projection and a
       small convex mixing rate `a = 0.02`, so at `core_lr == lr` its weights
       barely move in early steps (`|d mix| <= mix*(1-mix)*lr` per step).
       Scaling the core's step size by `core_lr_scale` (and `alpha` by an
       additional `5x` to offset the sigmoid Jacobian `mix*(1-mix) ~ 1/50`)
       lets the recurrent pathway open alongside the spatial adapters.
    """
    core = getattr(model, "odydora_core", None)
    core_ids = {id(p) for p in core.parameters()} if core is not None else set()

    adapter_decay, adapter_no_decay = [], []
    core_decay, core_no_decay, core_alpha = [], [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in core_ids:
            leaf = name.split(".")[-1]
            if leaf == "alpha":
                core_alpha.append(param)
            elif "norm" in name or param.ndim < 2:
                core_no_decay.append(param)
            else:
                core_decay.append(param)
        elif param.ndim < 2 or "magnitude" in name or "norm" in name or "bias" in name:
            adapter_no_decay.append(param)
        else:
            adapter_decay.append(param)

    groups = []
    if adapter_decay:
        groups.append(
            {
                "params": adapter_decay,
                "lr": lr,
                "weight_decay": weight_decay,
                "group_name": "adapter_decay",
            }
        )
    if adapter_no_decay:
        groups.append(
            {
                "params": adapter_no_decay,
                "lr": lr,
                "weight_decay": 0.0,
                "group_name": "adapter_no_decay",
            }
        )
    if core_decay:
        groups.append(
            {
                "params": core_decay,
                "lr": lr * core_lr_scale,
                "weight_decay": weight_decay,
                "group_name": "core_decay",
            }
        )
    if core_no_decay:
        groups.append(
            {
                "params": core_no_decay,
                "lr": lr * core_lr_scale,
                "weight_decay": 0.0,
                "group_name": "core_no_decay",
            }
        )
    if core_alpha:
        groups.append(
            {
                "params": core_alpha,
                "lr": lr * core_lr_scale * 5.0,
                "weight_decay": 0.0,
                "group_name": "core_alpha",
            }
        )

    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)


def train_steps(model, batches, optimizer, log_every=10, grad_accum=1, on_step=None,
                clip=1.0, scaler=None):
    """Yield one log row every `log_every` steps, as training proceeds.

    A generator, not a list: a caller printing these gets the first line after
    `log_every` steps instead of after the whole run. This used to return the
    finished list, which made a Colab cell sit with no output for the entire
    stage -- indistinguishable from a hang, and the user has no way to tell
    whether the loss is falling or the run is wedged.
    """
    model.train()
    started, tokens, traces, began = time.time(), 0, 0, time.time()
    losses = []
    for step, (ids, labels, is_trace_start) in enumerate(batches):
        device = next(model.parameters()).device
        ids, labels = ids.to(device), labels.to(device)

        if is_trace_start:
            model.odydora_reset_state()
            traces += 1
        layers = getattr(model, "odydora_layers", ())
        state_in = model.odydora_state().get(layers[0]) if layers else None
        if on_step is not None:
            on_step(step, is_trace_start, None if state_in is None else state_in.clone())

        out = model(input_ids=ids, labels=labels)
        # The state already crosses the boundary detached; this is the explicit
        # restatement so a reader does not have to trust the hook.
        model.odydora_detach_state()

        if torch.isnan(out.loss):
            # Guard against unexpected numerical overflow: do not backprop NaN
            # into weights and running optimizer moments.
            continue

        loss = out.loss / grad_accum
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], clip
            )
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        losses.append(out.loss.item())
        tokens += ids.numel()

        if (step + 1) % log_every == 0:
            state = model.odydora_state().get(layers[-1]) if layers else None
            elapsed = max(time.time() - began, 1e-9)
            core_mod = getattr(model, "odydora_core", None)
            up_norm = (
                core_mod.up.weight.detach().norm().item()
                if core_mod is not None
                else 0.0
            )
            row = {
                "step": step + 1,
                "loss": sum(losses) / len(losses),
                "tok_s": tokens / elapsed,
                "tokens": tokens,
                "traces": traces,
                "state_norm": 0.0 if state is None else state.norm().item(),
                "up_norm": up_norm,
                "elapsed": time.time() - started,
            }
            losses = []
            yield row


@torch.no_grad()
def evaluate(model, batches, tail_fraction=0.25, log_every=0, desc="eval"):
    """Mean loss in bits per token, overall and over the trailing chunks.

    The tail number is the one that matters: carrying state across chunks can
    only pay where a chunk has history behind it, so a gain that shows up
    overall but not in the tail is not the mechanism working.
    """
    model.eval()
    rows = []
    total = len(batches) if hasattr(batches, "__len__") else None
    for step, (ids, labels, is_trace_start) in enumerate(batches):
        device = next(model.parameters()).device
        if is_trace_start:
            model.odydora_reset_state()
        out = model(input_ids=ids.to(device), labels=labels.to(device))
        supervised = int((labels != -100).sum())
        rows.append((out.loss.item(), supervised, is_trace_start))
        if log_every and (step + 1) % log_every == 0:
            target = f"/{total}" if total else ""
            print(f"  {desc} chunk {step + 1}{target}...", flush=True)

    def bits(selected):
        total = sum(loss * count for loss, count, _ in selected)
        seen = sum(count for _, count, _ in selected)
        return (total / seen) / 0.6931471805599453 if seen else float("nan")

    cut = int(len(rows) * (1 - tail_fraction))
    return {
        "bits_per_token": bits(rows),
        "tail_bits_per_token": bits(rows[cut:]),
        "chunks": len(rows),
        "tail_chunks": len(rows[cut:]),
    }


def trainable_state(model):
    """Only what trains — the base model's 4-bit weights are not ours to save."""
    return {k: v for k, v in model.state_dict().items() if "odydora" in k or "lora" in k}


def save(model, path, step=0, traces=0, tokens=0, extra=None):
    """Checkpoint weights plus the corpus cursor.

    The cursor is the part people forget: a resume that restores the step
    counter but not the position silently re-reads data the model already
    saw, and nothing in the log shows it.
    """
    payload = {
        "trainable": trainable_state(model),
        "step": step,
        "traces": traces,
        "tokens": tokens,
    }
    if extra:
        payload.update(extra)
    target = str(path)
    tmp = f"{target}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, target)
    return path


def inspect_checkpoint(path):
    """Read checkpoint cursor and inferred architecture without building a model."""
    blob = torch.load(str(path), map_location="cpu", weights_only=True)
    trainable = blob.get("trainable", {})
    w = trainable.get("odydora_core.W")
    n = int(w.shape[0]) if w is not None else 0
    r = 0
    for key, tensor in trainable.items():
        if "lora_A" in key and tensor.ndim == 2:
            r = int(tensor.shape[0])
            break
    use_dora = any("lora_magnitude_vector" in k for k in trainable)
    params = sum(t.numel() for t in trainable.values()) - n
    info = {k: v for k, v in blob.items() if k != "trainable"}
    info.update({"n": n, "r": r, "use_dora": use_dora, "trainable_params": params})
    return info


def restore(model, path):
    """Load a checkpoint, returning its metadata (including the cursor)."""
    # weights_only=True: the payload is only tensors and ints, so there is no
    # reason to let a checkpoint unpickle arbitrary objects.
    blob = torch.load(str(path), map_location="cpu", weights_only=True)
    if model is not None:
        try:
            missing, unexpected = model.load_state_dict(blob["trainable"], strict=False)
        except RuntimeError as exc:
            info = inspect_checkpoint(path)
            core = getattr(model, "odydora_core", None)
            model_n = core.n if core is not None else 0
            raise ValueError(
                f"checkpoint architecture (N={info['n']}, r={info['r']}, "
                f"use_dora={info['use_dora']}) does not match model (N={model_n}): {exc}"
            ) from exc
        if unexpected:
            info = inspect_checkpoint(path)
            core = getattr(model, "odydora_core", None)
            model_n = core.n if core is not None else 0
            raise ValueError(
                f"checkpoint carries unexpected tensors {unexpected[:4]} "
                f"(checkpoint N={info['n']}, r={info['r']} vs model N={model_n})"
            )
    return {k: v for k, v in blob.items() if k != "trainable"}


def report(model, path=None):
    """The parameter budget, printed and optionally written as JSON."""
    adapter, core = count_parameters(model)
    total = sum(p.numel() for p in model.parameters())
    rows = {
        "total": total,
        "trainable": adapter + core,
        "adapter": adapter,
        "core": core,
        "trainable_fraction": (adapter + core) / max(total, 1),
    }
    if path:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)
    return rows
