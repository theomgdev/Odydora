import json

import torch
from transformers import AutoTokenizer

from odydora.data import chunk_trace, mask_assistant_turns

TRACE = {
    "messages": [
        {"role": "system", "content": "You are an autonomous coding agent."},
        {"role": "user", "content": "Fix the failing test in src/app.py"},
        {"role": "assistant", "content": "I will read the file first."},
        {"role": "user", "content": "Go ahead."},
        {"role": "assistant", "content": "Done: the import was missing."},
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write or edit file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            },
        }
    ],
}


def tokenizer():
    # A real tool-calling template, vendored by the test as a local file so
    # nothing downloads. Qwen2-style: the assistant turn is what we supervise.
    tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    tok.chat_template = (
        "{% for m in messages %}"
        "<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
        "{% endfor %}"
    )
    return tok


def test_only_assistant_content_is_supervised():
    tok = tokenizer()
    ids, labels = mask_assistant_turns(TRACE["messages"], tok)
    assert len(ids) == len(labels)
    assert any(x != -100 for x in labels), "nothing supervised"
    assert any(x == -100 for x in labels), "everything supervised"

    supervised = tok.decode([i for i, m in zip(ids, labels, strict=False) if m != -100])
    assert "import was missing" in supervised
    assert "read the file first" in supervised
    # The two things that must never be trained on.
    assert "autonomous coding agent" not in supervised
    assert "Fix the failing test" not in supervised


def test_supervised_label_equals_the_token_it_predicts():
    tok = tokenizer()
    ids, labels = mask_assistant_turns(TRACE["messages"], tok)
    for i, m in zip(ids, labels, strict=False):
        assert m == -100 or m == i, "labels must be the ids, masked in place"


def test_chunking_covers_the_trace_once_and_flags_the_start():
    tok = tokenizer()
    ids, labels = mask_assistant_turns(TRACE["messages"], tok)
    chunks = list(chunk_trace(ids, labels, chunk=8))
    assert sum(len(c[0]) for c in chunks) == len(ids), "chunks must tile the trace"
    assert [c[2] for c in chunks].count(True) == 1
    assert chunks[0][2] is True
    rebuilt = [t for c in chunks for t in c[0]]
    assert rebuilt == ids, "chunking must preserve order"


def test_a_chunk_with_no_supervised_token_is_dropped():
    ids = list(range(20))
    labels = [-100] * 12 + ids[12:]
    chunks = list(chunk_trace(ids, labels, chunk=6, drop_unsupervised=True))
    assert all(any(m != -100 for m in c[1]) for c in chunks)
    assert chunks[0][2] is True, "the first kept chunk still starts the trace"


def test_chunks_are_tensors_when_asked():
    tok = tokenizer()
    ids, labels = mask_assistant_turns(TRACE["messages"], tok)
    for chunk_ids, chunk_labels, _ in chunk_trace(ids, labels, chunk=8, as_tensor=True):
        assert isinstance(chunk_ids, torch.Tensor)
        assert chunk_ids.shape == chunk_labels.shape
        assert chunk_ids.dim() == 2 and chunk_ids.shape[0] == 1


# The real schema, taken from the datasets-server: an assistant turn carries
# `tool_calls`, a tool return carries `tool_call_id` and `name`.
AGENTIC_TRACE = [
    {"role": "system", "content": "You are an autonomous coding agent."},
    {"role": "user", "content": "Add a seed catalog."},
    {
        "role": "assistant",
        "content": "<thought>\nI will read the specs first.\n</thought>",
        "tool_calls": [
            {
                "id": "call_0",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path": "spec.md"}'},
            }
        ],
    },
    {
        "role": "tool",
        "content": "SPEC CONTENTS HERE",
        "tool_call_id": "call_0",
        "name": "read_file",
    },
    {"role": "assistant", "content": "The spec requires three fields."},
]


def test_messages_arriving_as_one_json_string_are_parsed():
    from odydora.data import normalize_messages

    payload = json.dumps(AGENTIC_TRACE)
    assert normalize_messages(payload) == AGENTIC_TRACE


def test_messages_arriving_split_per_character_are_rejoined():
    """The datasets-server returned rows whose `messages` was a sequence of
    one-character strings -- a serialized list iterated before parsing.
    Treating those as turns would corrupt every mask without raising.
    """
    from odydora.data import normalize_messages

    shredded = list(json.dumps(AGENTIC_TRACE))
    assert normalize_messages(shredded) == AGENTIC_TRACE


def test_a_message_that_is_not_a_dict_raises_instead_of_corrupting():
    from odydora.data import normalize_messages

    try:
        normalize_messages([{"role": "user", "content": "hi"}, 42])
    except ValueError as error:
        assert "not a dict" in str(error)
        return
    raise AssertionError("a non-dict turn must raise, not pass silently")


def test_tool_calls_are_supervised_and_tool_returns_are_not():
    tok = tokenizer()
    # A template that renders tool_calls, as the real Qwen template does.
    tok.chat_template = (
        "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}"
        "{% if m.get('tool_calls') %}{% for c in m['tool_calls'] %}"
        "<tool_call>{{ c['function']['name'] }} {{ c['function']['arguments'] }}"
        "</tool_call>{% endfor %}{% endif %}<|im_end|>\n{% endfor %}"
    )
    ids, labels = mask_assistant_turns(AGENTIC_TRACE, tok)
    supervised = tok.decode([i for i, m in zip(ids, labels, strict=False) if m != -100])
    # Choosing the tool and its arguments is the reasoning we are training.
    assert "read_file" in supervised
    assert "spec.md" in supervised
    assert "three fields" in supervised
    # The environment talking back is never a training target.
    assert "SPEC CONTENTS HERE" not in supervised
    assert "autonomous coding agent" not in supervised


def test_the_default_split_is_the_clean_one_not_the_large_one():
    """`full` has 10,000 rows against `train`'s 5,000 and is unusable:
    measured over the first ten rows of each, full carries no `tools` at
    all, 64 of its 65 tool calls degraded to the literal "unknown_tool",
    and no <thought> content. Training tool choice -- the reasoning this
    project is about -- is impossible there, so the larger split loses.
    """
    import inspect

    from odydora.data import stream_traces

    assert inspect.signature(stream_traces).parameters["split"].default == "train"
