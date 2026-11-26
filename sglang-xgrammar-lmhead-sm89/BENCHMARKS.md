# `sglang-xgrammar-lmhead-sm89` Benchmarks (`RTX 4090 D sm_89`, `300 W`, `Qwen/Qwen3.8-27B-FP8`)

All measurements below were collected on an `NVIDIA GeForce RTX 4090 D` (48 GB, `sm_89`, `300.00 W` TDP limit) inside `lmsysorg/sglang@sha256:3ce6da3e5ea0fee2dac6a23b528cb294fedd31ca74a55a4511f709c5e01e8652` (`v0.5.23-runtime`).

**Verified Artifact (`xgrammar-2c129d1c`):** `_C.cpython-312-x86_64-linux-gnu.so` SHA-256 `2c129d1c82155503ebe3771a9038534451de864cff832e287901860bb41ae8a4`.

## 1. Build, `ptxas`, & CPU Rollback Smoke Suite

- **Docker Build (`ptxas`):** Zero `nvcc` warnings; **0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads** in `active_row_lm_head_tile64_kernel` (`55` regs FP8, `48` regs BF16, `844 B` smem).
- **`tests/test_cpu_rollback_and_smoke.py` + `tests/test_mask_dispatch.py`:** `11 passed in 6.85s` in container (`11 passed in 8.25s` CPU macOS), verifying `active_pairs_within_budget_cpu`, `xgrammar_coupled_step_cpu`, eager vs graph isolation, `SGLANG_DFLASH_XGRAMMAR_SAMPLED=1` & `0` rollback parity, and `try...finally: rollback(accepted_cnt)`.

## 2. Isolated GPU Operator Gates (`RUN_FULL_VOCAB=1` & `run_isolated_gpu_gate.py`)

- **Full-Vocab `kTileV=64` CUDA Parity & Speedup (`tests/test_active_row_lm_head_gpu.py`, `N=248,320`, `K=5,120`):**
  - Parity PASS (`max_bf16_diff=0.0010`).
  - Cold-DRAM sparse vs dense `torch.mm` speedup: **`148.5x` (`M=1`)**, **`47.8x` (`M=8`)**, **`11.5x` (`M=64`)**.
- **P1 Coupled Grammar Gate (`tests/run_isolated_gpu_gate.py p1 --arm candidate`):**
  - `SGLANG_DFLASH_XGRAMMAR_SAMPLED=1`: **`5/5` `correct=True`** (`0` illegal tokens, `0` duplicate candidates; `sampled bs=8` `278.44 us`, `mixed bs=8` `280.06 us`).
  - `SGLANG_DFLASH_XGRAMMAR_SAMPLED=0` rollback mode: **`5/5` `correct=True`**.
- **P2 Row-Index & CUDA Graph Gate (`tests/run_isolated_gpu_gate.py p2 --arm candidate`):**
  - **`24/24` `correct=True`** (BF16 & FP8 tail parity up to `M=289` and CUDA Graph changing-mask replay).

## 3. Live Serving Promotion Verification (`sglang-chat`, `max_total_num_tokens=172,609`)

- **Container Status:** `Healthy`, `EXT_LOADED: True`, `ACTIVE_PAIRS: True`, `300.00 W`, zero GPU Xid errors, instant rollback preserved at `releases/xgrammar-0f7dd912` + `rollback-overlay.sh`.
- **Synthetic Canary (`sglang_synthetic_canary.py --label candidate --execute-live`, 3× consecutive):** **`45/45` PASSED** (`15/15` in `4.439s`, `1.96s`, `1.89s`).
- **Live Agentic & Sampled JSON (`temp=0.6`, `--max-tokens 256`):**
  - `volatile_first` (`pairs=6, iter=2`): **`12/12` `exact_match`** (`980.1 ms` `traj_p50`).
  - `catalog_first` (`pairs=6, iter=2`): **`12/12` `exact_match`** (`641.7 ms` `traj_p50`, **`1.532 valid tasks/s`**).
  - `bench_e2e_ab` JSON (`temp=0.6`): `c=1` `2/2` (`128.55 tok/s`), `c=8` `8/8` (`478.31 tok/s`).
