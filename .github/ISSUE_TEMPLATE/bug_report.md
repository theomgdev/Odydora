---
name: Bug report
about: Something behaves differently than documented
labels: bug
---

**What happened, and what you expected instead**

**The exact command or cell, and its full output**

```
```

**Environment**
- GPU and VRAM (or CPU-only):
- `python -c "import torch, transformers, peft; print(torch.__version__, transformers.__version__, peft.__version__)"`:
- `bitsandbytes` version, if the 4-bit path is involved:
- Base model id and the config values you changed:

**Did the test suite pass?** `pytest -q` output, if you ran it.
