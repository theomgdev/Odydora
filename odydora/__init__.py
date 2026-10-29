"""Odydora: a shared cross-layer recurrent core as a parameter-efficient adapter.

One N x N matrix, shared by every decoder layer of a frozen quantized language
model, carrying a recurrent state across layers and across chunk boundaries.

The core architecture comes from OdyssNet (Cahit Karahan, v4.0.0, MIT,
https://github.com/theomgdev/OdyssNet) -- one shared zero-diagonal matrix under
per-step RMSNorm, with the state detached at the call boundary. Odydora adapts
it to steer a frozen pretrained host at segment rate. See CITATION.cff to cite
both.
"""

from .adapter import (
    attach_core,
    bypass_core,
    count_parameters,
    detach_core,
    without_odydora,
)
from .core import RecurrentCore
from .data import chunk_trace, describe_trace, mask_assistant_turns, stream_traces
from .train import (
    adapter_only_params,
    build,
    configure_optimizer,
    evaluate,
    inspect_checkpoint,
    lora_config,
    matched_rank,
    report,
    restore,
    save,
    train_steps,
    trainable_state,
)

__version__ = "0.1.0"

__all__ = [
    "RecurrentCore",
    "adapter_only_params",
    "attach_core",
    "build",
    "bypass_core",
    "chunk_trace",
    "configure_optimizer",
    "count_parameters",
    "describe_trace",
    "detach_core",
    "evaluate",
    "inspect_checkpoint",
    "lora_config",
    "mask_assistant_turns",
    "matched_rank",
    "report",
    "restore",
    "save",
    "stream_traces",
    "train_steps",
    "trainable_state",
    "without_odydora",
]
