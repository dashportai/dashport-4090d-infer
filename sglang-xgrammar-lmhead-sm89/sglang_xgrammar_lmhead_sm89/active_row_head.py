"""Python interface & C++-accelerated sparsity dispatch for sm_89 active-row xgrammar lm_head."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

_EXT_LOADED: Optional[bool] = None
_CPP_BUDGET_FN: Optional[Callable[[torch.Tensor, int, int], bool]] = None
_BITWISE_COUNT = getattr(np, "bitwise_count", None)
_POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)


def ensure_extension_loaded() -> bool:
    """Import the compiled _C CUDA/C++ extension cleanly once."""
    global _EXT_LOADED, _CPP_BUDGET_FN
    if _EXT_LOADED is not None:
        return _EXT_LOADED
    try:
        from . import _C

        _ = torch.ops.sglang_xgrammar_lmhead_sm89.active_row_bf16_lm_head
        _CPP_BUDGET_FN = getattr(_C, "active_pairs_within_budget_cpu", None)
        _EXT_LOADED = True
    except Exception as exc:
        logger.warning("[xgrammar_lmhead_sm89] _C extension unavailable: %s", exc)
        _CPP_BUDGET_FN = None
        _EXT_LOADED = False
    return _EXT_LOADED


def get_cpp_coupled_step() -> Optional[Callable[..., Any]]:
    """Return compiled C++ coupled grammar step helper if _C is loaded."""
    if not ensure_extension_loaded():
        return None
    from . import _C

    return getattr(_C, "xgrammar_coupled_step_cpu", None)


def active_row_bf16_lm_head(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    vocab_mask: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute hidden @ weight.T restricted to rows permitted by vocab_mask."""
    assert ensure_extension_loaded(), "sglang_xgrammar_lmhead_sm89._C is not built"
    m, n = int(hidden.shape[0]), int(weight.shape[0])
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=hidden.device)
    torch.ops.sglang_xgrammar_lmhead_sm89.active_row_bf16_lm_head(
        hidden, weight, vocab_mask, out
    )
    return out


def active_row_fp8_lm_head(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    vocab_mask: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute hidden @ (weight * weight_scale[:, None]).T on active vocab_mask rows."""
    assert ensure_extension_loaded(), "sglang_xgrammar_lmhead_sm89._C is not built"
    m, n = int(hidden.shape[0]), int(weight.shape[0])
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=hidden.device)
    torch.ops.sglang_xgrammar_lmhead_sm89.active_row_fp8_lm_head(
        hidden, weight, weight_scale, vocab_mask, out
    )
    return out


def _count_active_pairs(words: np.ndarray) -> int:
    if _BITWISE_COUNT is not None:
        if words.shape[-1] % 2 == 0 and words.strides[-1] == words.itemsize:
            words = words.view(np.uint64)
        return int(_BITWISE_COUNT(words).sum(dtype=np.uint64))
    byte_words = np.ascontiguousarray(words).view(np.uint8)
    return int(_POPCOUNT_LUT[byte_words].sum(dtype=np.uint64))


def is_active_pair_count_within_limit(
    vocab_mask_cpu: torch.Tensor, vocab_size: int, max_pairs: int
) -> bool:
    """Check dense/sparse pair budget using C++ popcnt64 when loaded, or NumPy fallback."""
    if _BITWISE_COUNT is not None and _CPP_BUDGET_FN is not None and vocab_mask_cpu.device.type == "cpu":
        return bool(_CPP_BUDGET_FN(vocab_mask_cpu, int(vocab_size), int(max_pairs)))

    mask_words = (vocab_size + 31) // 32
    words = vocab_mask_cpu.detach().numpy()[..., :mask_words].view(np.uint32)
    if words.ndim == 2 and words.shape[0] > 8:
        max_pairs -= _count_active_pairs(words[:8])
        if max_pairs < 0:
            return False
        words = words[8:]
    return _count_active_pairs(words) <= max_pairs


def union_active_row_fraction(
    vocab_mask_cpu: torch.Tensor, vocab_size: int
) -> tuple[float, int, int]:
    """Return (active_tile_fraction, total_active_pairs, active_tiles) from CPU mask."""
    arr = vocab_mask_cpu.detach().numpy()
    mask_words = (vocab_size + 31) // 32
    if arr.ndim == 1:
        words_u32 = arr[:mask_words].astype(np.uint32, copy=False)
        union_words = words_u32
        total_active_pairs = int(_POPCOUNT_LUT[words_u32.view(np.uint8)].sum())
    else:
        words_u32 = arr[:, :mask_words].view(np.uint32)
        union_words = np.bitwise_or.reduce(words_u32, axis=0)
        total_active_pairs = int(_POPCOUNT_LUT[words_u32.view(np.uint8)].sum())

    n_tiles = (vocab_size + 127) // 128
    padded = np.zeros(n_tiles * 4, dtype=np.uint32)
    padded[: len(union_words)] = union_words
    active_tiles = int(padded.reshape(n_tiles, 4).any(axis=1).sum())
    return active_tiles / max(n_tiles, 1), total_active_pairs, active_tiles
