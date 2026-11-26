"""Overlap-safe greedy & coupled sampled in-draft xgrammar masking with eager/graph isolation."""

from __future__ import annotations

import os
import threading
import random
from typing import Any, Callable, Optional, Tuple
import torch
import numpy as np

from .active_row_head import get_cpp_coupled_step


class _DraftGrammarTLS(threading.local):
    def __init__(self) -> None:
        super().__init__()
        self.stack: list[Tuple[Any, Any, Optional[Callable[[], None]]]] = []
        self.eager_cands: Optional[torch.Tensor] = None
        self.used_graph: Optional[bool] = None
        self.mask_scratch: Optional[torch.Tensor] = None
        self.pin_draft: Optional[torch.Tensor] = None
        self.pin_cand: Optional[torch.Tensor] = None
        self.pin_q: Optional[torch.Tensor] = None
        self.h2d_done: Optional[Any] = None


_STEP_TLS = _DraftGrammarTLS()
_RNG = random.Random()


def _sampled_enabled() -> bool:
    return os.getenv("SGLANG_DFLASH_XGRAMMAR_SAMPLED", "1").strip().lower() not in (
        "0", "false", "no", "off", ""
    )


def _get_scratch_bitmask(vocab_words: int) -> torch.Tensor:
    buf = _STEP_TLS.mask_scratch
    if buf is None or buf.shape != (1, vocab_words):
        buf = torch.full((1, vocab_words), -1, dtype=torch.int32)
        _STEP_TLS.mask_scratch = buf
    return buf


def _alloc_pinned_view(
    slot_name: str, shape: Tuple[int, ...], dtype: torch.dtype
) -> torch.Tensor:
    tensor = getattr(_STEP_TLS, slot_name)
    need_alloc = (
        tensor is None
        or tensor.dtype != dtype
        or tensor.ndim != len(shape)
        or any(tensor.shape[idx] < shape[idx] for idx in range(len(shape)))
        or (len(shape) > 1 and tensor.shape[1:] != shape[1:])
    )
    if need_alloc:
        cap_shape = (max(int(shape[0]), 64), *shape[1:])
        tensor = torch.empty(cap_shape, dtype=dtype, pin_memory=True)
        setattr(_STEP_TLS, slot_name, tensor)
    return tensor[tuple(slice(0, dim) for dim in shape)]


def is_token_allowed(mask_words_np: np.ndarray, token_id: int, vocab_size: int) -> bool:
    """Check whether token_id bit is 1 inside a 1D int32/uint32 xgrammar bitmask."""
    if token_id < 0 or token_id >= vocab_size:
        return False
    w = token_id >> 5
    if w >= mask_words_np.shape[0]:
        return False
    return bool((int(mask_words_np[w]) >> (token_id & 31)) & 1)


def find_unique_allowed_token(mask_words_np: np.ndarray, vocab_size: int) -> int:
    """Return the unique token ID if |V_valid| == 1 in mask_words_np, else -1."""
    n_words = (vocab_size + 31) // 32
    words = mask_words_np[:n_words].view(np.uint32)
    nonzero = np.flatnonzero(words)
    if nonzero.size == 0:
        return -1

    rem = vocab_size & 31
    tail_mask = ((1 << rem) - 1) if rem else 0xFFFFFFFF
    sole = -1
    for w_np in nonzero:
        wi = int(w_np)
        val = int(words[wi])
        if wi == n_words - 1:
            val &= tail_mask
        if val == 0:
            continue
        if sole >= 0 or (val & (val - 1)) != 0:
            return -1
        sole = (wi << 5) + (val.bit_length() - 1)
    return sole if 0 <= sole < vocab_size else -1


def _find_first_legal(mask_words_np: np.ndarray, vocab_size: int) -> int:
    n_words = (vocab_size + 31) // 32
    words = mask_words_np[:n_words].view(np.uint32)
    rem = vocab_size & 31
    tail_mask = ((1 << rem) - 1) if rem else 0xFFFFFFFF
    for w_np in np.flatnonzero(words):
        wi = int(w_np)
        val = int(words[wi])
        if wi == n_words - 1:
            val &= tail_mask
        if val:
            lsb = val & -val
            tok = (wi << 5) + (lsb.bit_length() - 1)
            return tok if tok < vocab_size else -1
    return -1


def _place_front_no_dup(
    cand_row: Optional[torch.Tensor], q_row: Optional[torch.Tensor], tok: int
) -> None:
    if cand_row is not None:
        pos_hits = (cand_row == tok).nonzero(as_tuple=False)
        if pos_hits.numel() > 0:
            idx = int(pos_hits[0, 0].item())
            if idx != 0:
                prev = int(cand_row[0].item())
                cand_row[0] = tok
                cand_row[idx] = prev
        else:
            cand_row[0] = tok
    if q_row is not None:
        q_row.zero_()
        q_row[0] = 1.0


def _resolve_greedy_per_req(sampling_info: Any, bs: int, reqs: Optional[list[Any]]) -> list[bool]:
    is_greedy_batch = True if sampling_info is None else bool(getattr(sampling_info, "is_all_greedy", True))
    if is_greedy_batch:
        return [True] * bs

    t_tensor = getattr(sampling_info, "temperatures", None)
    k_tensor = getattr(sampling_info, "top_ks", None)
    t_addr = int(t_tensor.data_ptr()) if isinstance(t_tensor, torch.Tensor) else 0
    k_addr = int(k_tensor.data_ptr()) if isinstance(k_tensor, torch.Tensor) else 0

    cache_key = None
    if reqs is not None:
        cache_key = (t_addr, k_addr, bs, tuple(id(item) for item in reqs[:bs]))
        if getattr(sampling_info, "_xgrammar_greedy_sig", None) == cache_key:
            return sampling_info._xgrammar_greedy_flags

    if isinstance(t_tensor, torch.Tensor) and isinstance(k_tensor, torch.Tensor):
        greedy_gpu = (t_tensor.view(-1)[:bs] < 1e-5) | (k_tensor.view(-1)[:bs] == 1)
        result = [bool(flag) for flag in greedy_gpu.detach().to("cpu").tolist()]
    else:
        t_vals = (
            t_tensor.view(-1)[:bs].detach().cpu().tolist()
            if isinstance(t_tensor, torch.Tensor)
            else [1.0] * bs
        )
        k_vals = (
            k_tensor.view(-1)[:bs].detach().cpu().tolist()
            if isinstance(k_tensor, torch.Tensor)
            else [0] * bs
        )
        result = [(float(t_vals[i]) < 1e-5) or (int(k_vals[i]) == 1) for i in range(bs)]

    if cache_key is not None:
        sampling_info._xgrammar_greedy_sig = cache_key
        sampling_info._xgrammar_greedy_flags = result
    return result


def _py_step_refine(
    mask_np: np.ndarray, draft_row: torch.Tensor, cand_row: Optional[torch.Tensor],
    q_row: Optional[torch.Tensor], pos: int, vocab_size: int,
    is_greedy: bool, allow_forced_single: bool,
) -> Tuple[int, bool, bool]:
    cur = int(draft_row[pos].item())
    forced = find_unique_allowed_token(mask_np, vocab_size) if allow_forced_single else -1
    if forced >= 0:
        d_mod = forced != cur
        if d_mod:
            draft_row[pos] = forced
        q_mod = q_row is not None
        if cand_row is not None or q_row is not None:
            _place_front_no_dup(cand_row, q_row, forced)
        return forced, d_mod, q_mod

    if is_greedy or cand_row is None or q_row is None:
        if is_token_allowed(mask_np, cur, vocab_size):
            pick = cur
            d_mod = False
        else:
            pick = -1
            if cand_row is not None:
                for tok in cand_row.tolist():
                    if is_token_allowed(mask_np, int(tok), vocab_size):
                        pick = int(tok)
                        break
            if pick < 0:
                pick = _find_first_legal(mask_np, vocab_size)
            if pick < 0:
                return -1, False, False
            draft_row[pos] = pick
            d_mod = True
        q_mod = q_row is not None
        if cand_row is not None or q_row is not None:
            _place_front_no_dup(cand_row, q_row, pick)
        return pick, d_mod, q_mod

    width = int(cand_row.shape[0])
    legal_bool = torch.tensor(
        [is_token_allowed(mask_np, int(cand_row[i].item()), vocab_size) for i in range(width)], dtype=torch.bool
    )
    q_masked = torch.where(legal_bool, q_row.clamp_min(0.0), torch.zeros_like(q_row))
    z_sum = float(q_masked.sum().item())
    if z_sum > 1e-12:
        q_norm = q_masked / z_sum
        q_row.copy_(q_norm)
        hit = (cand_row == cur).nonzero(as_tuple=False)
        if is_token_allowed(mask_np, cur, vocab_size) and hit.numel() > 0:
            if float(q_norm[int(hit[0, 0].item())].item()) > 0.0:
                return cur, False, True
        draw = int(torch.multinomial(q_norm, num_samples=1).item())
        chosen = int(cand_row[draw].item())
        draft_row[pos] = chosen
        return chosen, True, True

    fb = _find_first_legal(mask_np, vocab_size)
    if fb < 0:
        return -1, False, False
    _place_front_no_dup(cand_row, q_row, fb)
    draft_row[pos] = fb
    return fb, True, True


def _run_coupled_batch(
    reqs: list[Any], draft_cpu: torch.Tensor, cand_cpu: Optional[torch.Tensor],
    q_cpu: Optional[torch.Tensor], greedy_flags: Optional[list[bool]], vocab_size: int,
    allow_forced_single: bool = True,
) -> Tuple[bool, bool, bool]:
    bs, block_len = int(draft_cpu.shape[0]), int(draft_cpu.shape[1])
    words = (vocab_size + 31) // 32
    scratch = _get_scratch_bitmask(words)
    scratch_row = scratch[0]
    mask_np = scratch_row.numpy()

    cpp_step = get_cpp_coupled_step()
    can_cpp = (
        cpp_step is not None
        and draft_cpu.is_contiguous()
        and (cand_cpu is None or cand_cpu.is_contiguous())
        and (q_cpu is None or q_cpu.is_contiguous())
    )
    if can_cpp:
        scratch.fill_(-1)

    d_changed = False
    c_changed = False
    q_changed = False
    u_fn = _RNG.random

    for i in range(bs):
        grammar = getattr(reqs[i], "grammar", None)
        if grammar is None or grammar.is_terminated():
            continue
        is_g = True if (greedy_flags is None or i >= len(greedy_flags)) else greedy_flags[i]
        d_row = draft_cpu[i]
        c_mat = cand_cpu[i] if cand_cpu is not None else None
        q_mat = q_cpu[i] if q_cpu is not None else None
        pushed = 0
        try:
            for pos in range(1, block_len):
                if grammar.is_terminated():
                    break
                slot = pos - 1
                c_row = c_mat[slot] if c_mat is not None else None
                q_row = q_mat[slot] if q_mat is not None else None
                if can_cpp:
                    grammar.fill_vocab_mask(scratch, 0)
                    tok, dm, qm = cpp_step(
                        scratch_row, d_row, c_row, q_row,
                        pos, vocab_size, is_g, allow_forced_single, u_fn(), True
                    )
                else:
                    scratch.fill_(-1)
                    grammar.fill_vocab_mask(scratch, 0)
                    tok, dm, qm = _py_step_refine(
                        mask_np, d_row, c_row, q_row,
                        pos, vocab_size, is_g, allow_forced_single
                    )
                if tok < 0:
                    break
                if dm:
                    d_changed = True
                if qm:
                    q_changed = True
                    if c_row is not None:
                        c_changed = True
                grammar.accept_token(tok)
                pushed += 1
        finally:
            if pushed > 0:
                grammar.rollback(pushed)

    return d_changed, c_changed, q_changed


def refine_draft_block_with_grammar(
    *,
    reqs: list[Any],
    draft_cpu: torch.Tensor,
    cand_cpu: Optional[torch.Tensor],
    vocab_size: int,
    allow_forced_single: bool = True,
    q_cpu: Optional[torch.Tensor] = None,
    greedy_flags: Optional[list[bool]] = None,
) -> bool:
    """Public helper supporting both greedy-only and coupled sampled refinement."""
    d_chg, _, _ = _run_coupled_batch(
        reqs, draft_cpu, cand_cpu, q_cpu, greedy_flags, vocab_size, allow_forced_single
    )
    return d_chg


def _is_two_tensor_pair(obj: Any) -> bool:
    return (
        isinstance(obj, (tuple, list))
        and len(obj) >= 2
        and isinstance(obj[0], torch.Tensor)
        and isinstance(obj[1], torch.Tensor)
    )


def _attach_runner_observer(worker: Any) -> None:
    runner = getattr(worker, "draft_model_runner", None)
    if runner is None or getattr(runner, "_xgrammar_obs", False):
        return
    cls_fwd = getattr(type(runner), "forward", None)
    if getattr(cls_fwd, "_xgrammar_hook_tag", False):
        return
    orig = getattr(runner, "forward", None)
    if orig is None:
        return

    def _wrapped(*f_args, **f_kwargs):
        res = orig(*f_args, **f_kwargs)
        after_draft_runner_forward(res, runner, *f_args, **f_kwargs)
        return res

    runner.forward = _wrapped
    runner._xgrammar_obs = True


def after_compute_candidates(result, *args, **kwargs):
    """Capture eager DFlash2DraftModel.compute_candidates output when active."""
    if _STEP_TLS.stack and result is not None:
        cand = result[0] if isinstance(result, (tuple, list)) else result
        if isinstance(cand, torch.Tensor):
            _STEP_TLS.eager_cands = cand
    return result


def after_draft_runner_forward(result, runner_self, *args, **kwargs):
    """Track whether draft_model_runner used CUDA Graph for the active verify step."""
    if not _STEP_TLS.stack:
        return result
    worker_ref = _STEP_TLS.stack[-1][0]
    if getattr(worker_ref, "draft_model_runner", None) is runner_self:
        _STEP_TLS.used_graph = bool(getattr(result, "can_run_graph", False))
    return result


after_draft_runner_forward._xgrammar_hook_tag = True


def _normalize_eager_cands(
    cands: torch.Tensor, bs: int, n_slots: int
) -> Optional[torch.Tensor]:
    if cands.dim() == 3:
        return cands[:bs, :n_slots] if (cands.shape[0] >= bs and cands.shape[1] >= n_slots) else None
    if cands.dim() == 2:
        if cands.shape[0] == bs and cands.shape[1] == n_slots:
            return cands.unsqueeze(-1)
        if cands.shape[0] >= bs * n_slots:
            return cands[: bs * n_slots].reshape(bs, n_slots, -1)
    if cands.dim() == 1 and cands.numel() >= bs * n_slots:
        return cands[: bs * n_slots].reshape(bs, n_slots, -1)
    return None


def around_dflash_forward_batch_generation(original_fn, worker, *args, **kwargs):
    """Track current DFlashWorkerV2 step and isolate eager vs CUDA Graph buffers."""
    batch = kwargs["batch"] if "batch" in kwargs else args[0]
    grammar_barrier = kwargs.get("grammar_barrier", None)
    _attach_runner_observer(worker)

    prev = (_STEP_TLS.eager_cands, _STEP_TLS.used_graph)
    _STEP_TLS.eager_cands = None
    _STEP_TLS.used_graph = None
    stk = _STEP_TLS.stack
    stk.append((worker, batch, grammar_barrier))
    try:
        out = original_fn(worker, *args, **kwargs)
    finally:
        stk.pop()
        _STEP_TLS.eager_cands, _STEP_TLS.used_graph = prev
    return out


def around_grammar_tree_from_linear_chain(original_fn, cls, verify_ids_2d: torch.Tensor):
    """Apply overlap-safe greedy or Theorem 1C coupled sampled grammar masking."""
    if not _STEP_TLS.stack:
        return original_fn(cls, verify_ids_2d)

    worker, batch, grammar_barrier = _STEP_TLS.stack[-1]
    if not getattr(batch, "has_grammar", False):
        return original_fn(cls, verify_ids_2d)

    sampling_info = getattr(batch, "sampling_info", None)
    all_greedy = bool(getattr(sampling_info, "is_all_greedy", False))
    sampled_on = _sampled_enabled()
    if not sampled_on and not all_greedy:
        return original_fn(cls, verify_ids_2d)

    if grammar_barrier is not None:
        grammar_barrier()

    reqs = getattr(batch, "reqs", None)
    bs = len(reqs) if reqs is not None else 0
    n_slots = int(verify_ids_2d.shape[1]) - 1
    if bs <= 0 or n_slots <= 0:
        return original_fn(cls, verify_ids_2d)
    live_grammar = False
    for r in reqs[:bs]:
        g_obj = getattr(r, "grammar", None)
        if g_obj is not None and not g_obj.is_terminated():
            live_grammar = True
            break
    if not live_grammar:
        return original_fn(cls, verify_ids_2d)

    sampler = getattr(worker, "_draft_sampler", None)
    sel = getattr(worker, "_selector_sample", None)
    has_sel_pair = sampled_on and _is_two_tensor_pair(sel)

    cand_gpu: Optional[torch.Tensor] = None
    q_gpu: Optional[torch.Tensor] = None
    if has_sel_pair:
        cand_gpu, q_gpu = sel[0], sel[1]
    elif sampled_on and not all_greedy and sel is not None:
        # Rollback-safe guard if external test/mock passes sentinel instead of (cand, q) pair.
        return original_fn(cls, verify_ids_2d)
    else:
        if _STEP_TLS.eager_cands is not None:
            cand_gpu = _normalize_eager_cands(_STEP_TLS.eager_cands, bs, n_slots)
        elif (
            _STEP_TLS.used_graph is True
            and sampler is not None
            and getattr(sampler, "candidate_out", None) is not None
        ):
            cand_gpu = sampler.candidate_out[:bs]

    vocab_size = int(worker.target_worker.model_runner.model_config.vocab_size)
    greedy_flags = (
        [True] * bs
        if not sampled_on
        else _resolve_greedy_per_req(sampling_info, bs, reqs)
    )
    allow_forced = os.getenv("SGLANG_XGRAMMAR_JUMP_FORWARD", "1").strip() != "0"

    if verify_ids_2d.is_cuda:
        if _STEP_TLS.h2d_done is not None and not _STEP_TLS.h2d_done.query():
            _STEP_TLS.h2d_done.synchronize()
        draft_cpu = _alloc_pinned_view(
            "pin_draft", (bs, int(verify_ids_2d.shape[1])), verify_ids_2d.dtype
        )
        draft_cpu.copy_(verify_ids_2d[:bs], non_blocking=True)

        cand_cpu = None
        if cand_gpu is not None:
            cs = cand_gpu[:bs]
            cand_cpu = _alloc_pinned_view("pin_cand", tuple(int(d) for d in cs.shape), cs.dtype)
            cand_cpu.copy_(cs, non_blocking=True)

        q_cpu = None
        if has_sel_pair and q_gpu is not None:
            qs = q_gpu[:bs]
            q_cpu = _alloc_pinned_view("pin_q", tuple(int(d) for d in qs.shape), qs.dtype)
            q_cpu.copy_(qs, non_blocking=True)

        torch.cuda.current_stream(verify_ids_2d.device).synchronize()
    else:
        draft_cpu = verify_ids_2d[:bs].detach().to("cpu").clone()
        cand_cpu = cand_gpu[:bs].detach().to("cpu").clone() if cand_gpu is not None else None
        q_cpu = q_gpu[:bs].detach().to("cpu").clone() if (has_sel_pair and q_gpu is not None) else None

    d_chg, c_chg, q_chg = _run_coupled_batch(
        reqs[:bs], draft_cpu, cand_cpu, q_cpu, greedy_flags, vocab_size, allow_forced
    )

    h2d_ran = False
    if d_chg:
        verify_ids_2d[:bs, 1:].copy_(draft_cpu[:, 1:], non_blocking=verify_ids_2d.is_cuda)
        h2d_ran = verify_ids_2d.is_cuda

    if q_chg and has_sel_pair and q_cpu is not None:
        sel[1][:bs].copy_(q_cpu, non_blocking=sel[1].is_cuda)
        h2d_ran = h2d_ran or sel[1].is_cuda
        if c_chg and cand_cpu is not None:
            sel[0][:bs].copy_(cand_cpu, non_blocking=sel[0].is_cuda)

    if h2d_ran:
        ev = _STEP_TLS.h2d_done or torch.cuda.Event()
        _STEP_TLS.h2d_done = ev
        ev.record(torch.cuda.current_stream(verify_ids_2d.device))

    return original_fn(cls, verify_ids_2d)
