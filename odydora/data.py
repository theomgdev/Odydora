"""Stream the corpus, template it, mask it to assistant turns, chunk it.

The corpus is agentic traces averaging tens of thousands of tokens, so
nothing here ever holds a whole split in memory: a trace is templated once,
masked, cut into fixed-size chunks, and handed on.
"""

import json

import torch

ASSISTANT_ROLES = ("assistant",)
IGNORE = -100


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
    elif isinstance(messages, (list, tuple)) and messages:
        if all(isinstance(m, str) for m in messages):
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

    The mask is built by templating the conversation prefix by prefix and
    taking the length difference, never by string-matching a role marker:
    a trace whose content happens to contain the marker would otherwise
    silently shift the whole mask. The cost is one template call per turn,
    which is nothing beside tokenizing the text itself.

    An assistant turn is supervised whole, including its `tool_calls` --
    deciding which tool to invoke with which arguments is the reasoning we
    want. The `tool` role's returns are the environment talking back and
    are never supervised.
    """
    messages = normalize_messages(messages)
    kwargs = {"tokenize": False, "add_generation_prompt": False}
    if tools:
        kwargs["tools"] = tools

    ids, labels = [], []
    previous = 0
    for index, message in enumerate(messages):
        rendered = tokenizer.apply_chat_template(messages[: index + 1], **kwargs)
        encoded = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        span = encoded[previous:]
        previous = len(encoded)
        ids.extend(span)
        if message.get("role") in ASSISTANT_ROLES:
            labels.extend(span)
        else:
            labels.extend([IGNORE] * len(span))
    return ids, labels


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
        if drop_unsupervised and all(m == IGNORE for m in window_labels):
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
    affordable; `skip_traces` steps a resumed run over what it consumed.
    """
    from datasets import load_dataset

    rows = load_dataset(dataset, split=split, streaming=streaming)
    consumed = 0
    for index, row in enumerate(rows):
        if index < skip_traces:
            continue
        ids, labels = mask_assistant_turns(
            row["messages"], tokenizer, row.get("tools")
        )
        if max_trace_tokens:
            ids, labels = ids[:max_trace_tokens], labels[:max_trace_tokens]
        for window in chunk_trace(
            ids, labels, chunk, drop_unsupervised=True, as_tensor=True
        ):
            yield window
            consumed += window[0].numel()
            if max_tokens and consumed >= max_tokens:
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
