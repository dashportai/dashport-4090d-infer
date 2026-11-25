"""GPU numerical parity, CUDA-graph capture, and cold-DRAM microbenchmark for active_row_lm_head_sm89."""

from __future__ import annotations

import os
import statistics
import time

import torch

from sglang_xgrammar_lmhead_sm89.active_row_head import (
    active_row_bf16_lm_head,
    active_row_fp8_lm_head,
)


def _make_sparse_mask(m: int, n: int, density: float, *, device: torch.device) -> torch.Tensor:
    words = (n + 31) // 32
    mask = torch.zeros((m, words), dtype=torch.int32, device="cpu")
    k_active = max(1, int(round(n * density)))
    torch.manual_seed(1234)
    if k_active >= n:
        mask.fill_(-1)
    else:
        for row_m in range(m):
            idx = torch.randperm(n)[:k_active]
            for v in idx.tolist():
                mask[row_m, v >> 5] |= 1 << (v & 31)
    return mask.to(device)


def _reference_bf16(hidden: torch.Tensor, weight: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logits = torch.matmul(hidden, weight.T)
    m, n = logits.shape
    words = mask.shape[1]
    bit_pos = torch.arange(n, device=logits.device, dtype=torch.int32)
    word_idx = bit_pos >> 5
    bit_off = bit_pos & 31
    selected_words = mask[:, word_idx]
    allowed = ((selected_words >> bit_off.unsqueeze(0)) & 1).bool()
    return logits.masked_fill(~allowed, float("-inf"))


def _bench_cold_cuda(fn, *, flush_buf: torch.Tensor, warmup: int = 8, iters: int = 30) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        flush_buf.zero_()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000.0)
    return statistics.median(times)


def _row_index_mask(m: int, n: int, offset: int, device: torch.device) -> torch.Tensor:
    rows = torch.arange(m).unsqueeze(1)
    tokens = torch.arange(n)
    allowed = (tokens == (17 * rows + offset) % n) | ((rows % 3 == offset % 3) & (tokens == n - 1))
    padding = (-n) % 32
    words = torch.nn.functional.pad(allowed, (0, padding)).reshape(m, -1, 32)
    bit_values = 1 << torch.arange(32, dtype=torch.int32)
    return (words.to(torch.int32) * bit_values).sum(dim=-1, dtype=torch.int32).to(device)


def run_row_index_regression() -> None:
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    # A partial vocabulary tile and exact input values isolate indexing from rounding.
    n, k = 259, 512
    w_bf16 = torch.zeros((n, k), dtype=torch.bfloat16, device=device)
    w_bf16[:, 0] = torch.arange(n, device=device) % 7 + 1
    w_fp8 = w_bf16.to(torch.float8_e4m3fn)
    w_scale = (torch.arange(n, device=device) % 2 + 1).float() * 0.5
    w_fp8_dequant = (w_fp8.float() * w_scale.unsqueeze(1)).to(torch.bfloat16)

    for m in (256, 257, 289):
        hidden = torch.zeros((m, k), dtype=torch.bfloat16, device=device)
        hidden[:, 0] = torch.arange(m, device=device) % 31 + 1
        mask = _row_index_mask(m, n, 0, device)
        out_bf16 = torch.full((m, n), 1234, dtype=torch.bfloat16, device=device)
        out_fp8 = torch.full_like(out_bf16, 1234)
        active_row_bf16_lm_head(hidden, w_bf16, mask, out=out_bf16)
        torch.testing.assert_close(out_bf16, _reference_bf16(hidden, w_bf16, mask), rtol=0, atol=0)
        active_row_fp8_lm_head(hidden, w_fp8, w_scale, mask, out=out_fp8)
        torch.testing.assert_close(out_fp8, _reference_bf16(hidden, w_fp8_dequant, mask), rtol=0, atol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        active_row_bf16_lm_head(hidden, w_bf16, mask, out=out_bf16)
        active_row_fp8_lm_head(hidden, w_fp8, w_scale, mask, out=out_fp8)

    mask.copy_(_row_index_mask(m, n, 1, device))
    graph.replay()
    torch.testing.assert_close(out_bf16, _reference_bf16(hidden, w_bf16, mask), rtol=0, atol=0)
    torch.testing.assert_close(out_fp8, _reference_bf16(hidden, w_fp8_dequant, mask), rtol=0, atol=0)
    print("Row index regression PASS (BF16/FP8, M=256,257,289, N=259, CUDA Graph)")


def run_parity_and_bench() -> None:
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    torch.manual_seed(42)

    # Full vocab parity if GPU is dedicated (RUN_FULL_VOCAB=1), else compact 16384 parity check.
    full = os.getenv("RUN_FULL_VOCAB", "0") == "1"
    N = 248320 if full else 16384
    K = 5120

    w_bf16 = torch.randn((N, K), dtype=torch.bfloat16, device=device) * 0.02
    # Quantize copy for FP8 active-row check
    w_f32 = w_bf16.float()
    w_scale = (w_f32.abs().amax(dim=1) / 448.0).clamp_min(1e-12).contiguous()
    w_fp8 = (w_f32 / w_scale.unsqueeze(1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).contiguous()
    del w_f32

    for M in (1, 8, 48, 64):
        hidden = torch.randn((M, K), dtype=torch.bfloat16, device=device) * 0.1
        for density in (0.002, 0.05):
            mask = _make_sparse_mask(M, N, density, device=device)
            ref = _reference_bf16(hidden, w_bf16, mask)
            out_bf16 = active_row_bf16_lm_head(hidden, w_bf16, mask)
            # Active elements match closely; masked elements are exact -inf.
            active_flag = torch.isfinite(ref)
            assert torch.equal(torch.isfinite(out_bf16), active_flag), f"mask mismatch M={M} d={density}"
            max_diff = (out_bf16[active_flag] - ref[active_flag]).abs().max().item()
            assert max_diff <= 0.05, f"BF16 diff too high: {max_diff} (M={M}, d={density})"

            out_fp8 = active_row_fp8_lm_head(hidden, w_fp8, w_scale, mask)
            assert torch.equal(torch.isfinite(out_fp8), active_flag)
            fp8_diff = (out_fp8[active_flag] - ref[active_flag]).abs().max().item()
            assert fp8_diff <= 0.15, f"FP8 diff too high: {fp8_diff} (M={M}, d={density})"

    # CUDA Graph capture + replay parity (changing mask bits dynamically without re-capture).
    M_cg = 8
    h_static = torch.randn((M_cg, K), dtype=torch.bfloat16, device=device) * 0.1
    mask_static = _make_sparse_mask(M_cg, N, 0.01, device=device)
    out_static = torch.empty((M_cg, N), dtype=torch.bfloat16, device=device)
    # Warmup stream
    active_row_bf16_lm_head(h_static, w_bf16, mask_static, out=out_static)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        active_row_bf16_lm_head(h_static, w_bf16, mask_static, out=out_static)

    # Swap mask contents on replay and verify CUDA graph tracks new mask!
    mask_new = _make_sparse_mask(M_cg, N, 0.005, device=device)
    mask_static.copy_(mask_new)
    g.replay()
    ref_new = _reference_bf16(h_static, w_bf16, mask_new)
    assert torch.equal(torch.isfinite(out_static), torch.isfinite(ref_new))
    cg_diff = (out_static[torch.isfinite(ref_new)] - ref_new[torch.isfinite(ref_new)]).abs().max().item()
    assert cg_diff <= 0.05, f"CUDAGraph replay diff too high: {cg_diff}"
    print(f"Parity & CUDAGraph PASS (N={N}, max_bf16_diff={cg_diff:.4f})")

    if full:
        flush_buf = torch.empty((80 * 1024 * 1024 // 4,), dtype=torch.float32, device=device)
        for M in (1, 8, 56, 64):
            h = torch.randn((M, K), dtype=torch.bfloat16, device=device)
            dense_us = _bench_cold_cuda(lambda: torch.matmul(h, w_bf16.T), flush_buf=flush_buf)
            line = [f"M={M:2d} | dense_bf16={dense_us:7.1f}us"]
            for d in (0.001, 0.01, 0.05, 0.10, 0.25):
                msk = _make_sparse_mask(M, N, d, device=device)
                sp_us = _bench_cold_cuda(
                    lambda: active_row_bf16_lm_head(h, w_bf16, msk, out=out_static[:M] if M <= 8 else None),
                    flush_buf=flush_buf,
                )
                line.append(f"d={d*100:4.1f}%:{sp_us:6.1f}us({dense_us/sp_us:4.1f}x)")
            print(" | ".join(line))


if __name__ == "__main__":
    run_row_index_regression()
    if os.getenv("RUN_ROW_INDEX_REGRESSION_ONLY") != "1":
        run_parity_and_bench()
