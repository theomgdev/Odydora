"""Stream the corpus, template it, mask it to assistant turns, chunk it.

The corpus is agentic traces averaging tens of thousands of tokens, so
nothing here ever holds a whole split in memory: a trace is templated once,
masked, cut into fixed-size chunks, and handed on.
"""

import json

import torch

ASSISTANT_ROLES = ("assistant",)
IGNORE = -100

# In MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x (even on split="train"),
# 97% of tool_calls have had their arguments scrubbed by the dataset publisher
# into literal template placeholders ("<file_path>", "<directory_path>", etc.).
# Supervising those redacted tool_call blocks teaches the model to emit literal
# "<file_path>" tool calls in an infinite loop after every thought block.
_REDACTED_TOOL_MARKERS = (
    "<file_path>",
    "<directory_path>",
    "<bash_command>",
    "<search_query>",
    "<file_content>",
    "unknown_tool",
)


def _normalize_reasoning_tags(messages):
    """Map Fable's `<thought>` / `<final_answer>` tags to Qwen3 `<think>` format.

    Qwen3's chat template inspects `if '</think>' in content:` to populate its
    native `<think>...</think>` reasoning block, and otherwise injects an empty
    `<think>\\n\\n</think>` on the final turn. Converting `<thought>` to `<think>`
    and stripping `<final_answer>` wrappers preserves Qwen3's native reasoning
    token protocol across both intermediate thoughts and final synthesis turns.
    """
    normalized = []
    for turn in messages:
        copied = dict(turn)
        if copied.get("role") in ASSISTANT_ROLES and isinstance(copied.get("content"), str):
            text = copied["content"]
            text = text.replace("<thought>", "<think>").replace("</thought>", "</think>")
            text = text.replace("<final_answer>", "").replace("</final_answer>", "").strip()
            copied["content"] = text
        normalized.append(copied)
    return normalized


def normalize_messages(messages):
    """Coerce one row's `messages` into a list of dicts.

    The column's declared feature type is `Json`, and the datasets-server
    hands rows back in more than one shape: a list of dicts, or a single
    JSON string holding the whole list. A row that arrives as a sequence of
    one-character strings is a serialized list that something iterated
    before parsing, so rejoining and parsing it recovers the real turns.
    Guessing wrong here corrupts every mask downstream silently, which is
    why this is explicit rather than an isinstance check at the call site.
    """
    if isinstance(messages, str):
        messages = json.loads(messages)
    elif isinstance(messages, list | tuple) and messages and all(
        isinstance(m, str) for m in messages
    ):
        joined = "".join(messages)
        try:
            messages = json.loads(joined)
        except json.JSONDecodeError:
            messages = [json.loads(m) for m in messages]
    if not isinstance(messages, list):
        raise ValueError(f"messages must be a list, got {type(messages).__name__}")
    bad = [i for i, m in enumerate(messages) if not isinstance(m, dict)]
    if bad:
        raise ValueError(f"messages[{bad[0]}] is not a dict after normalizing")
    return messages


def mask_assistant_turns(messages, tokenizer, tools=None):
    """Return (ids, labels) with everything but assistant content masked.

    The mask is built by tokenizing the finished conversation once with offset
    mappings and locating each assistant turn's character span via NUL-sentinel
    substitution (`_assistant_char_spans`), never by string-matching a role
    marker or diffing consecutive prefix renders.

    An assistant turn is supervised whole, including its `tool_calls` when their
    arguments are intact -- deciding which tool to invoke with which arguments
    is part of the reasoning. If a turn's `tool_calls` contain redacted template
    placeholders (such as `"<file_path>"` or `"<bash_command>"`), only the
    turn's reasoning body is supervised and the redacted `tool_call` suffix is
    masked. The `tool` role's returns are the environment talking back and are
    never supervised.
    """
    messages = _normalize_reasoning_tags(normalize_messages(messages))
    kwargs = {"tokenize": False, "add_generation_prompt": False}
    if tools:
        kwargs["tools"] = tools

    # Tokenize the finished conversation exactly once. Diffing consecutive
    # prefixes -- the obvious approach, and what this used to do -- is wrong
    # for any template that rewrites an earlier turn: Qwen3 emits an empty
    # `<think>` block for the LAST assistant turn and drops it again on the
    # next render, so the prefix shrinks, `encoded[previous:]` slices from the
    # wrong place, and tokens are silently lost. Measured on a 5-turn trace,
    # the stream fed to the model read `A1<|im_end|>\n2<|im_end|>` -- the whole
    # `<|im_start|>user\nU2` header gone and "U2" truncated to "2".
    full = tokenizer.apply_chat_template(messages, **kwargs)
    encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(encoded["input_ids"])
    offsets = encoded["offset_mapping"]

    spans = _assistant_char_spans(messages, tokenizer, kwargs, full)
    labels = [IGNORE] * len(ids)
    for begin, end in spans:
        for position, (start, stop) in enumerate(offsets):
            if stop > start and start >= begin and stop <= end:
                labels[position] = ids[position]
    return ids, labels


# A string the corpus cannot contain: NULs are illegal in JSON text, so a trace
# can never carry one and no amount of crafted content can collide with it.
_SENTINEL = "\x00odydora-span\x00"


def _assistant_char_spans(messages, tokenizer, kwargs, full):
    """[(begin, end)] char spans of each assistant turn inside `full`.

    Found by substitution, never by searching for a role marker. Rendering the
    conversation with one turn's content replaced by a sentinel says exactly
    where that turn's text begins, because every earlier turn is untouched, and
    the length difference against `full` says how long the real text was. A
    trace whose own content contains `<|im_start|>assistant` -- which the
    template renders verbatim, and which tokenizes to the same special id as
    the real marker, so no token-level search can tell them apart -- therefore
    cannot shift the mask.

    The sentinel sits where the turn's rendered body starts, which puts any
    `<think>` block the template emits inside the supervised span. That is
    deliberate: the reasoning is what this project trains.
    """
    starts = []
    for index in range(len(messages)):
        probe = [dict(turn) for turn in messages]
        probe[index] = dict(probe[index])
        probe[index]["content"] = _SENTINEL
        rendered = tokenizer.apply_chat_template(probe, **kwargs)
        position = rendered.find(_SENTINEL)
        if position < 0:
            # The template dropped this turn's content entirely, so there is
            # nothing of it in `full` to supervise.
            starts.append((index, None, None))
            continue
        # Only this turn's body differs between the two renders.
        body = len(full) - (len(rendered) - len(_SENTINEL))
        starts.append((index, position, body))

    spans = []
    for order, (index, position, body) in enumerate(starts):
        if position is None or messages[index].get("role") not in ASSISTANT_ROLES:
            continue
        end = position + max(body, 0)
        # Extend over the turn's closing markers, which are template output and
        # so cannot be spoofed by content: stop at the next turn's body.
        # However, if `tail` contains redacted placeholder tool calls (such as
        # `"<file_path>"`), do NOT extend over `tail`: supervising redacted
        # placeholder arguments trains the model to emit broken `<tool_call>`
        # loops after every thought block.
        following = next(
            (p for _, p, _ in starts[order + 1 :] if p is not None), len(full)
        )
        tail = full[end:following]
        if not any(marker in tail for marker in _REDACTED_TOOL_MARKERS):
            cut = tail.rfind("<|im_end|>")
            if cut >= 0:
                end += cut + len("<|im_end|>")
        spans.append((position, end))
    return spans


def chunk_trace(ids, labels, chunk, drop_unsupervised=False, as_tensor=False):
    """Yield (ids, labels, is_trace_start) windows of at most `chunk` tokens.

    `is_trace_start` tells the training loop when to reset the recurrent
    state instead of carrying it: state carries between chunks of the same
    trace, and only there. With `drop_unsupervised` a window holding no
    supervised token is skipped — it would contribute a zero loss while
    still costing a step — and the start flag moves to the first kept one so
    the state is still reset exactly once per trace.
    """
    started = False
    for begin in range(0, len(ids), chunk):
        window_ids = ids[begin : begin + chunk]
        window_labels = labels[begin : begin + chunk]
        # In causal LM training, shift_labels = labels[..., 1:], so position 0
        # cannot be predicted and is dropped by the loss. A window whose only
        # supervised token is at position 0 produces zero active targets in
        # CrossEntropyLoss, yielding 0/0 = NaN loss. Drop it if drop_unsupervised.
        if drop_unsupervised and (
            len(window_labels) <= 1 or all(m == IGNORE for m in window_labels[1:])
        ):
            continue
        is_start = not started
        started = True
        if as_tensor:
            yield (
                torch.tensor([window_ids], dtype=torch.long),
                torch.tensor([window_labels], dtype=torch.long),
                is_start,
            )
        else:
            yield window_ids, window_labels, is_start


def stream_traces(
    tokenizer,
    chunk,
    dataset="MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x",
    split="train",
    max_tokens=None,
    max_trace_tokens=None,
    skip_traces=0,
    max_traces=None,
    streaming=True,
):
    """Yield (ids, labels, is_trace_start) tensors from the live corpus.

    `split` is "train" (5,000 rows) and not "full" (10,000), despite full
    being larger, because full is corrupted. Measured over the first ten
    rows of each from the datasets-server: full has `tools` null in 10/10
    rows, 64 of its 65 tool calls degraded to the literal name
    "unknown_tool", and no <thought> content at all; train has tools on
    every row, zero unknown_tool, and reasoning intact. Training the tool
    choice -- which is the reasoning this project is about -- is impossible
    on full. The card's "10,000 traces" describes that split, which is why
    the row count and the quality point in opposite directions.

    Streams by default: ~2.55 GB of parquet across both splits, and a Colab
    session has no reason to materialize it. `max_trace_tokens` truncates
    each trace, which is how the fast first-signal stage keeps a long trace
    affordable; `skip_traces` steps a resumed run over what it consumed;
    `max_traces` caps the number of consumed traces for bounded stages.
    """
    from datasets import load_dataset

    rows = load_dataset(dataset, split=split, streaming=streaming)
    consumed = 0
    traces_seen = 0
    for index, row in enumerate(rows):
        if index < skip_traces:
            continue
        ids, labels = mask_assistant_turns(
            row["messages"], tokenizer, row.get("tools")
        )
        windows = list(
            chunk_trace(ids, labels, chunk, drop_unsupervised=True, as_tensor=True)
        )
        if max_trace_tokens and chunk > 0:
            max_windows = max(1, max_trace_tokens // chunk)
            if len(windows) > max_windows:
                # Keep the tail of the kept windows so bounded stages (S1)
                # capture deep mid-trace reasoning and the final synthesis
                # rather than only Turn 1's opening boilerplate.
                windows = windows[-max_windows:]
                first_ids, first_labels, _ = windows[0]
                windows[0] = (first_ids, first_labels, True)
        for window in windows:
            yield window
            consumed += window[0].numel()
            if max_tokens and consumed >= max_tokens:
                return
        traces_seen += 1
        if max_traces and traces_seen >= max_traces:
            return


def describe_trace(row, tokenizer):
    """Token counts for one row, for the notebook's length histogram."""
    ids, labels = mask_assistant_turns(row["messages"], tokenizer, row.get("tools"))
    supervised = sum(1 for m in labels if m != IGNORE)
    return {
        "turns": len(normalize_messages(row["messages"])),
        "tokens": len(ids),
        "supervised": supervised,
        "supervised_fraction": supervised / max(len(ids), 1),
    }
