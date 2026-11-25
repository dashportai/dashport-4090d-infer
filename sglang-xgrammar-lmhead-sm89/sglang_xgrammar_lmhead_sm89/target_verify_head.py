"""Deferred Target-Verify lm_head dispatch using sm_89 active-row CUDA kernel when sparse."""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Optional

import torch

from .active_row_head import (
    active_row_bf16_lm_head,
    ensure_extension_loaded,
    is_active_pair_count_within_limit,
)
from .in_draft_xgrammar import _STEP_TLS

logger = logging.getLogger(__name__)


class _TargetVerifyTLS(threading.local):
    def __init__(self) -> None:
        super().__init__()
        self.pending: Optional[dict[str, Any]] = None


_TARGET_TLS = _TargetVerifyTLS()


def _ensure_lp_verify_buffers(
    lp_self, *, cap_rows: int, n_vocab: int, hidden_dim: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    buf = getattr(lp_self, "_sm89_verify_logits_buf", None)
    if (
        buf is None
        or buf.shape[0] < cap_rows
        or buf.shape[1] != n_vocab
        or buf.device != device
    ):
        buf = torch.empty((max(cap_rows, 64), n_vocab), dtype=torch.bfloat16, device=device)
        lp_self._sm89_verify_logits_buf = buf

    hbuf = getattr(lp_self, "_sm89_verify_hidden_buf", None)
    if (
        hbuf is None
        or hbuf.shape[0] < cap_rows
        or hbuf.shape[1] != hidden_dim
        or hbuf.device != device
    ):
        hbuf = torch.empty((max(cap_rows, 64), hidden_dim), dtype=torch.bfloat16, device=device)
        lp_self._sm89_verify_hidden_buf = hbuf
    return buf, hbuf


def after_dflash_worker_init_target_buf(_result, worker, *args, **kwargs) -> None:
    """Pre-allocate static verify logits & hidden buffers before CUDA graph capture."""
    target_worker = getattr(worker, "target_worker", None) or getattr(worker, "_target_worker", None)
    if target_worker is None:
        return
    model = getattr(getattr(target_worker, "model_runner", None), "model", None)
    if model is None:
        return
    lm_head = getattr(model, "lm_head", None)
    weight = getattr(lm_head, "weight", None)
    lp = getattr(worker, "_target_logits_processor", None)
    if weight is None or weight.dtype != torch.bfloat16 or lp is None:
        return
    cap_rows = max(64, 16 * int(getattr(worker, "block_size", 8)))
    _ensure_lp_verify_buffers(
        lp,
        cap_rows=cap_rows,
        n_vocab=int(weight.shape[0]),
        hidden_dim=int(weight.shape[1]),
        device=weight.device,
    )


def around_target_logits_processor_forward(
    original_fn,
    lp_self,
    input_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    lm_head,
    logits_metadata,
    aux_hidden_states=None,
    hidden_states_before_norm=None,
):
    """During TARGET_VERIFY (including CUDA graph capture), skip dense lm_head inside graph."""
    from sglang.srt.layers import layernorm_sp
    from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessorOutput
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    meta = (
        LogitsMetadata.from_forward_batch(logits_metadata)
        if isinstance(logits_metadata, ForwardBatch)
        else logits_metadata
    )

    if (
        not meta.forward_mode.is_target_verify()
        or meta.extend_return_logprob
        or getattr(lp_self, "use_fp32_lm_head", False)
        or getattr(lp_self, "use_fp32_lm_head_output", False)
        or getattr(lp_self, "final_logit_softcapping", None) is not None
        or getattr(lm_head, "weight", None) is None
        or lm_head.weight.dtype != torch.bfloat16
        or getattr(lp_self, "do_tensor_parallel_all_gather", False)
        or getattr(lp_self, "logit_scale", None) is not None
    ):
        return original_fn(
            lp_self,
            input_ids,
            hidden_states,
            lm_head,
            logits_metadata,
            aux_hidden_states=aux_hidden_states,
            hidden_states_before_norm=hidden_states_before_norm,
        )

    hidden_states, hidden_states_before_norm = layernorm_sp.maybe_exit_gather(
        hidden_states=hidden_states,
        hidden_states_before_norm=hidden_states_before_norm,
        input_ids=input_ids,
        forward_mode=meta.forward_mode,
    )
    if getattr(lp_self, "expected_aux_hidden_states", None) is not None:
        lp_self._check_aux_hidden_states(aux_hidden_states, hidden_states, meta)

    (
        pruned_states,
        pruned_states_before_norm,
        aux_pruned_states,
        sample_indices,
        _input_logprob_indices,
        _token_to_seq_idx,
    ) = lp_self._get_pruned_states(
        hidden_states,
        hidden_states_before_norm,
        aux_hidden_states,
        meta,
    )

    hidden_states_to_store = lp_self._get_hidden_states_to_store(
        hidden_states,
        hidden_states_before_norm,
        aux_hidden_states,
        pruned_states,
        pruned_states_before_norm,
        aux_pruned_states,
        sample_indices,
        meta,
    )
    last_hidden_states_to_store = lp_self._get_last_hidden_states_to_store(
        hidden_states, aux_hidden_states, meta
    )

    m = int(pruned_states.shape[0])
    n_vocab = int(lm_head.weight.shape[0])
    hidden_dim = int(pruned_states.shape[1])
    buf = getattr(lp_self, "_sm89_verify_logits_buf", None)
    hbuf = getattr(lp_self, "_sm89_verify_hidden_buf", None)
    if (
        buf is None
        or buf.shape[0] < m
        or buf.shape[1] != n_vocab
        or hbuf is None
        or hbuf.shape[0] < m
        or hbuf.shape[1] != hidden_dim
    ):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("verify logits/hidden buffer missing or too small during CUDA graph capture")
        buf, hbuf = _ensure_lp_verify_buffers(
            lp_self,
            cap_rows=m,
            n_vocab=n_vocab,
            hidden_dim=hidden_dim,
            device=pruned_states.device,
        )

    hbuf[:m].copy_(pruned_states)
    lp_self._sm89_verify_deferred = True
    return LogitsProcessorOutput(
        next_token_logits=buf[:m],
        hidden_states=hidden_states_to_store,
        last_hidden_states=last_hidden_states_to_store,
        mm_input_embeds=meta.mm_input_embeds,
    )


def after_tp_worker_forward_batch_generation(result, tp_worker, *args, **kwargs):
    """Record pending deferred verify lm_head work after graph replay OR eager target verify."""
    if not kwargs.get("is_verify", False) or not _STEP_TLS.stack:
        return result

    dflash_worker = _STEP_TLS.stack[-1][0]
    lp = getattr(dflash_worker, "_target_logits_processor", None)
    if lp is None or not getattr(lp, "_sm89_verify_deferred", False):
        return result

    logits_output = getattr(result, "logits_output", None)
    hbuf = getattr(lp, "_sm89_verify_hidden_buf", None)
    if logits_output is None or hbuf is None or logits_output.next_token_logits is None:
        return result

    model = tp_worker.model_runner.model
    lm_head = getattr(model, "lm_head", None)
    weight = getattr(lm_head, "weight", None)
    if weight is None:
        return result

    m = int(logits_output.next_token_logits.shape[0])
    _TARGET_TLS.pending = {
        "hidden": hbuf[:m],
        "weight": weight,
        "out": logits_output.next_token_logits,
        "sparse_vocab_mask_gpu": None,
        "materialized": False,
    }
    return result


def around_build_grammar_vocab_mask(original_fn, *args, **kwargs):
    """Choose sparse target projection from the CPU bitmask before copying it to the GPU."""
    from sglang.srt.speculative import spec_utils as su

    pending = _TARGET_TLS.pending
    if pending is None:
        return original_fn(*args, **kwargs)

    reqs = kwargs["reqs"]
    tree = kwargs["tree"]
    sampling_info = kwargs["sampling_info"]
    device = kwargs["device"]
    barrier = kwargs.get("barrier")

    if barrier is not None:
        barrier()
    vocab_mask_cpu, grammar = su.generate_token_bitmask(
        reqs,
        *tree.resolve(),
        sampling_info.vocab_size,
    )
    if vocab_mask_cpu is None:
        return None

    max_pairs = int(os.getenv("SGLANG_XGRAMMAR_MAX_ACTIVE_PAIRS", "180000"))
    use_sparse = is_active_pair_count_within_limit(
        vocab_mask_cpu, int(sampling_info.vocab_size), max_pairs
    )
    vocab_mask_gpu = vocab_mask_cpu.to(device, non_blocking=True)
    sampling_info.grammar_mask = None
    pending["sparse_vocab_mask_gpu"] = vocab_mask_gpu if use_sparse else None
    return su.GrammarMask(grammar, vocab_mask_gpu)


def materialize_deferred_target_logits() -> None:
    """Populate next_token_logits via active-row CUDA kernel (sparse) or cuBLAS (dense)."""
    pending = _TARGET_TLS.pending
    if pending is None or pending["materialized"]:
        return
    pending["materialized"] = True
    _TARGET_TLS.pending = None

    hidden = pending["hidden"]
    weight = pending["weight"]
    out = pending["out"]
    vocab_mask_gpu = pending["sparse_vocab_mask_gpu"]

    if vocab_mask_gpu is not None and ensure_extension_loaded():
        active_row_bf16_lm_head(hidden, weight, vocab_mask_gpu, out=out)
    else:
        torch.mm(hidden, weight.T, out=out)


def before_apply_dflash_verify_logits_adjustments(*args, **kwargs) -> None:
    materialize_deferred_target_logits()


def before_dflash_accept_block(*args, **kwargs) -> None:
    materialize_deferred_target_logits()


def clear_target_verify_tls() -> None:
    _TARGET_TLS.pending = None
