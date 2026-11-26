# `sglang-xgrammar-lmhead-sm89`

Modular read-only SGLang `0.5.23` extension (`sglang.srt.plugins` entry point `xgrammar_lmhead_sm89 = sglang_xgrammar_lmhead_sm89.plugin:register`) for Ada Lovelace (`sm_89`).

**Promoted Artifact (`xgrammar-2c129d1c`):** `_C.cpython-312-x86_64-linux-gnu.so` SHA-256 `2c129d1c82155503ebe3771a9038534451de864cff832e287901860bb41ae8a4`.

## Components

1. **`kTileV=64` CUDA-Graph-Safe Active-Row `lm_head` Kernel (`csrc/active_row_lm_head_sm89.cu`, `csrc/torch_bindings.cpp`):**
   - Exposes `torch.ops.sglang_xgrammar_lmhead_sm89.active_row_bf16_lm_head` and `active_row_fp8_lm_head` targeting `-gencode=arch=compute_89,code=sm_89` with **0 register spills** (`55` regs FP8 / `48` regs BF16, `844 B` smem) and zero `--use_fast_math`.
   - Uses `kTileV=64` active-row compaction (`uint64_t` warp-ballot), `uint4` tile union fast-exit (`0` weight DRAM reads when inactive), and multi-warp `K` streaming (`148.5x` at `M=1`, `47.8x` at `M=8`, `11.5x` at `M=64` on `N=248,320`).
   - Compiled C++20 `-O3` active-pair budget counter (`active_pairs_within_budget_cpu` using `__builtin_popcountll`) inside `active_row_head.py` (with NumPy fallback).

2. **Overlap-Safe In-Draft `xgrammar` Coupled Resampling & Jump-Forward (`in_draft_xgrammar.py`, `csrc/xgrammar_coupled_cpu.cpp`):**
   - Hooks `DFlashWorkerV2.forward_batch_generation`, `DFlash2DraftModel.compute_candidates`, `ModelRunner.forward`, and `@classmethod GrammarTree.from_linear_chain` with eager vs CUDA-graph isolation (preserving Rule A `_selector_sample` 2-tensor `(cand, q)` checks vs Rule B eager/graph candidate buffers).
   - Executes Theorem 1C lossless coupled sampled conditional resampling (`SGLANG_DFLASH_XGRAMMAR_SAMPLED=1`, instant rollback via `SGLANG_DFLASH_XGRAMMAR_SAMPLED=0`) in C++20 `-O3` (`xgrammar_coupled_step_cpu`), paired with `|V_valid| == 1` jump-forward, pinned D2H staging, event-guarded H2D copy, cached per-batch greedy flags, and `try ... finally: rollback(accepted_cnt)`.

3. **Deferred Target-Verify Sparse/Dense Dispatch (`target_verify_head.py`, `active_row_head.py`):**
   - Hooks `LogitsProcessor.forward` during `TARGET_VERIFY` to capture `pruned_states` into a static buffer, evaluates the C++ active-pair budget inside `build_grammar_vocab_mask`, and materializes `next_token_logits` via `active_row_bf16_lm_head` when `total_active_pairs <= SGLANG_XGRAMMAR_MAX_ACTIVE_PAIRS` (`180000`) or dense `torch.mm` otherwise.

## Environment Toggles

| Variable | Default | Description |
|---|---:|---|
| `SGLANG_XGRAMMAR_LMHEAD_SM89` | `1` | Master plugin enable/disable |
| `SGLANG_DFLASH_XGRAMMAR_DRAFT` | `1` | Enable in-draft `xgrammar` masking & rollback |
| `SGLANG_DFLASH_XGRAMMAR_SAMPLED` | `1` | Enable lossless Theorem 1C coupled sampled resampling (`0` restores greedy-only fallback) |
| `SGLANG_XGRAMMAR_JUMP_FORWARD` | `1` | Inject unique `\|V_valid\| == 1` forced tokens |
| `SGLANG_XGRAMMAR_ACTIVE_ROW_TARGET` | `1` | Enable deferred sparse/dense target verify `lm_head` dispatch |
| `SGLANG_XGRAMMAR_MAX_ACTIVE_PAIRS` | `180000` | Max `(m, row)` active bit pairs for active-row CUDA kernel |
