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
from .distill import (
    check_vocab_match,
    distill_loss,
    distill_steps,
    evaluate_gap,
    stream_text_chunks,
    teacher_reference_bits,
    teacher_targets,
)
from .train import (
    adapter_only_params,
    build,
    build_teacher,
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

__version__ = "0.2.0"

__all__ = [
    "RecurrentCore",
    "adapter_only_params",
    "attach_core",
    "build",
    "build_teacher",
    "bypass_core",
    "check_vocab_match",
    "chunk_trace",
    "configure_optimizer",
    "count_parameters",
    "describe_trace",
    "detach_core",
    "distill_loss",
    "distill_steps",
    "evaluate",
    "evaluate_gap",
    "inspect_checkpoint",
    "lora_config",
    "mask_assistant_turns",
    "matched_rank",
    "report",
    "restore",
    "save",
    "stream_text_chunks",
    "stream_traces",
    "teacher_reference_bits",
    "teacher_targets",
    "train_steps",
    "trainable_state",
    "without_odydora",
]
