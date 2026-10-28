"""Stream the corpus, template it, mask it to assistant turns, chunk it.

The corpus is agentic traces averaging tens of thousands of tokens, so
nothing here ever holds a whole split in memory: a trace is templated once,
masked, cut into fixed-size chunks, and handed on.
"""

import torch

ASSISTANT_ROLES = ("assistant",)
IGNORE = -100


def mask_assistant_turns(messages, tokenizer, tools=None):
    """Return (ids, labels) with everything but assistant content masked.

    The mask is built by templating the conversation prefix by prefix and
    taking the length difference, never by string-matching a role marker:
    a trace whose content happens to contain the marker would otherwise
    silently shift the whole mask. The cost is one template call per turn,
    which is nothing beside tokenizing the text itself.
    """
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

    Streams by default: the corpus is gigabytes and a Colab session has no
    reason to materialize it. `max_trace_tokens` truncates each trace, which
    is how the fast first-signal stage keeps a 50k-token trace affordable;
    `skip_traces` is how a resumed run steps over what it already consumed.
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
        "turns": len(row["messages"]),
        "tokens": len(ids),
        "supervised": supervised,
        "supervised_fraction": supervised / max(len(ids), 1),
    }
