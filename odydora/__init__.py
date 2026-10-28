"""Odydora: a shared cross-layer recurrent core as a parameter-efficient adapter.

One N x N matrix, shared by every decoder layer of a frozen quantized language
model, carrying a recurrent state across layers and across chunk boundaries.

The core architecture comes from OdyssNet (Cahit Karahan, v4.0.0, MIT,
https://github.com/theomgdev/OdyssNet) -- one shared zero-diagonal matrix under
per-step RMSNorm, with the state detached at the call boundary. Odydora adapts
it to steer a frozen pretrained host at segment rate. See CITATION.cff to cite
both.
"""
