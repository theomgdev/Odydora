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

    supervised = tok.decode([i for i, m in zip(ids, labels) if m != -100])
    assert "import was missing" in supervised
    assert "read the file first" in supervised
    # The two things that must never be trained on.
    assert "autonomous coding agent" not in supervised
    assert "Fix the failing test" not in supervised


def test_supervised_label_equals_the_token_it_predicts():
    tok = tokenizer()
    ids, labels = mask_assistant_turns(TRACE["messages"], tok)
    for i, m in zip(ids, labels):
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
    tok = tokenizer()
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
