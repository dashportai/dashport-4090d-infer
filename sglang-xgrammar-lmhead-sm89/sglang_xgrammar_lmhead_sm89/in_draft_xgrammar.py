"""Overlap-safe greedy in-draft xgrammar masking with guaranteed rollback."""

from __future__ import annotations

import os
import threading
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch


class _StepContextTLS(threading.local):
    def __init__(self) -> None:
        super().__init__()
        self.stack: list[Tuple[Any, Any, Optional[Callable[[], None]]]] = []


_STEP_TLS = _StepContextTLS()
_BITMASK_CACHE = threading.local()


def _get_scratch_bitmask(vocab_words: int) -> torch.Tensor:
    buf = getattr(_BITMASK_CACHE, "row", None)
    if buf is None or buf.shape != (1, vocab_words):
        buf = torch.full((1, vocab_words), -1, dtype=torch.int32)
        _BITMASK_CACHE.row = buf
    return buf


def is_token_allowed(mask_words_np: np.ndarray, token_id: int, vocab_size: int) -> bool:
    """Check whether token_id bit is 1 inside a 1D int32/uint32 xgrammar bitmask."""
    if token_id < 0 or token_id >= vocab_size:
        return False
    word_idx = token_id >> 5
    if word_idx >= mask_words_np.shape[0]:
        return False
    return bool((int(mask_words_np[word_idx]) >> (token_id & 31)) & 1)


def find_unique_allowed_token(mask_words_np: np.ndarray, vocab_size: int) -> int:
    """Return the unique token ID if |V_valid| == 1 in mask_words_np, else -1."""
    words_u32 = mask_words_np.view(np.uint32)
    mask_limit = (vocab_size + 31) // 32
    words_u32 = words_u32[:mask_limit]
    nz = np.flatnonzero(words_u32)
    if nz.size != 1:
        return -1
    w_idx = int(nz[0])
    w_val = int(words_u32[w_idx])
    # Zero out any padding bits beyond vocab_size in the tail word.
    tail_bits = vocab_size - w_idx * 32
    if tail_bits < 32:
        w_val &= (1 << tail_bits) - 1
    if w_val == 0 or (w_val & (w_val - 1)) != 0:
        return -1
    tok = w_idx * 32 + (w_val.bit_length() - 1)
    return tok if tok < vocab_size else -1


def refine_draft_block_with_grammar(
    *,
    reqs: list[Any],
    draft_cpu: torch.Tensor,
    cand_cpu: Optional[torch.Tensor],
    vocab_size: int,
    allow_forced_single: bool = True,
) -> bool:
    """Refine greedy draft proposals and roll the grammar back to its committed prefix.

    The caller must pass only an all-greedy verification batch. The returned flag
    indicates whether any proposal in ``draft_cpu`` was changed in place.
    """
    bs, block_size = int(draft_cpu.shape[0]), int(draft_cpu.shape[1])
    vocab_words = (vocab_size + 31) // 32
    scratch_mask = _get_scratch_bitmask(vocab_words)
    mask_np = scratch_mask[0].numpy()

    draft_modified = False

    for i in range(bs):
        req = reqs[i]
        grammar = getattr(req, "grammar", None)
        if grammar is None or grammar.is_terminated():
            continue

        accepted_cnt = 0
        try:
            for pos in range(1, block_size):
                if grammar.is_terminated():
                    break
                scratch_mask.fill_(-1)
                grammar.fill_vocab_mask(scratch_mask, 0)

                cur_tok = int(draft_cpu[i, pos].item())
                chosen = find_unique_allowed_token(mask_np, vocab_size) if allow_forced_single else -1
                if chosen < 0:
                    if is_token_allowed(mask_np, cur_tok, vocab_size):
                        chosen = cur_tok
                    elif cand_cpu is not None:
                        chosen = next(
                            (
                                tok
                                for tok in cand_cpu[i, pos - 1].tolist()
                                if is_token_allowed(mask_np, tok, vocab_size)
                            ),
                            -1,
                        )
                if chosen < 0:
                    break

                grammar.accept_token(chosen)
                accepted_cnt += 1
                if chosen != cur_tok:
                    draft_cpu[i, pos] = chosen
                    draft_modified = True
        finally:
            # Guard accepted_cnt > 0: XGrammarGrammar.rollback(0) slices [:-0] -> []!
            if accepted_cnt > 0:
                grammar.rollback(accepted_cnt)

    return draft_modified


def around_dflash_forward_batch_generation(original_fn, worker, *args, **kwargs):
    """Track current DFlashWorkerV2 batch & grammar_barrier across draft/verify."""
    batch = kwargs["batch"] if "batch" in kwargs else args[0]
    grammar_barrier = kwargs.get("grammar_barrier", None)
    _STEP_TLS.stack.append((worker, batch, grammar_barrier))
    try:
        return original_fn(worker, *args, **kwargs)
    finally:
        _STEP_TLS.stack.pop()


def around_grammar_tree_from_linear_chain(original_fn, cls, verify_ids_2d: torch.Tensor):
    """Apply overlap-safe in-draft xgrammar masking right before verify launches."""
    if not _STEP_TLS.stack:
        return original_fn(cls, verify_ids_2d)

    worker, batch, grammar_barrier = _STEP_TLS.stack[-1]
    if not getattr(batch, "has_grammar", False):
        return original_fn(cls, verify_ids_2d)

    # SGLang chooses sampled verification at the batch level. Rewriting the
    # drawn tokens after sampling can desynchronize them from the proposal q
    # used by verification, including when a greedy row shares that batch.
    if not getattr(getattr(batch, "sampling_info", None), "is_all_greedy", False):
        return original_fn(cls, verify_ids_2d)

    if grammar_barrier is not None:
        grammar_barrier()

    reqs = getattr(batch, "reqs", None)
    bs = len(reqs) if reqs is not None else 0
    if bs <= 0 or verify_ids_2d.shape[1] <= 1:
        return original_fn(cls, verify_ids_2d)

    if not any(
        (g := getattr(r, "grammar", None)) is not None and not g.is_terminated()
        for r in reqs[:bs]
    ):
        return original_fn(cls, verify_ids_2d)

    sampler = getattr(worker, "_draft_sampler", None)
    cand_gpu = getattr(sampler, "candidate_out", None)

    draft_cpu = verify_ids_2d[:bs].detach().to("cpu", non_blocking=False).clone()
    cand_cpu = (
        cand_gpu[:bs].detach().to("cpu", non_blocking=False).clone()
        if cand_gpu is not None
        else None
    )
    vocab_size = int(worker.target_worker.model_runner.model_config.vocab_size)
    allow_forced = os.getenv("SGLANG_XGRAMMAR_JUMP_FORWARD", "1").strip() != "0"

    draft_modified = refine_draft_block_with_grammar(
        reqs=reqs[:bs],
        draft_cpu=draft_cpu,
        cand_cpu=cand_cpu,
        vocab_size=vocab_size,
        allow_forced_single=allow_forced,
    )

    if draft_modified:
        verify_ids_2d[:bs, 1:].copy_(draft_cpu[:, 1:].to(verify_ids_2d.device))

    return original_fn(cls, verify_ids_2d)
