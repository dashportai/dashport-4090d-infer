# Dashport 4090 D inference

An SM89 CUDA and XGrammar plugin for SGLang `0.5.23` DFlash2, with isolated CUDA/C++ gates and a reproducible agentic tool-call benchmark. Tested on one 48 GB RTX 4090 D at a 300 W power limit (`Qwen/Qwen3.8-27B-FP8` + `z-lab/Qwen3.8-27B-DFlash2`).

**Current Overlay Build (`xgrammar-2c129d1c`):** `_C.cpython-312-x86_64-linux-gnu.so` SHA-256 `2c129d1c82155503ebe3771a9038534451de864cff832e287901860bb41ae8a4`.

## Highlights (`2c129d1c` Release)

- **`kTileV=64` Active-Row `lm_head` CUDA Kernel (`csrc/active_row_lm_head_sm89.cu`):** Compiled with `-gencode=arch=compute_89,code=sm_89` (**0 register spills**; `55` regs FP8, `48` regs BF16, `844 B` smem) and zero `--use_fast_math`. Achieves **`148.5x` (`M=1`)**, **`47.8x` (`M=8`)**, and **`11.5x` (`M=64`)** vs dense `torch.mm` on `N=248,320` (`max_bf16_diff=0.0010`), plus `24/24` P2 row-index and CUDA Graph replay parity up to `M=289`.
- **C++20 `-O3` Coupled Grammar & Active-Pair Dispatch (`csrc/xgrammar_coupled_cpu.cpp`):** Exposes `xgrammar_coupled_step_cpu` and `__builtin_popcountll`-based `active_pairs_within_budget_cpu` via `sglang_xgrammar_lmhead_sm89._C`, with eager vs CUDA Graph isolation (`compute_candidates`, `ModelRunner.forward`, Rule A/B checks), pinned D2H staging, event-guarded H2D copy, and cached per-batch greedy flags.
- **Lossless Theorem 1C Coupled Sampled Conditional Resampling (`SGLANG_DFLASH_XGRAMMAR_SAMPLED=1`):** Enabled by default (`5/5` P1 `correct=True` with `0` illegal tokens, `sampled bs=8` `278.44 us`, `mixed bs=8` `280.06 us`), with instant env rollback (`SGLANG_DFLASH_XGRAMMAR_SAMPLED=0`, `5/5` P1 `correct=True`).

## Results

| Benchmark / Gate | Scope | Verified Result (`2c129d1c`) |
|---|---|---|
| Full-vocab active-row `lm_head` | `N=248,320`, `K=5,120` | `148.5x` (`M=1`), `47.8x` (`M=8`), `11.5x` (`M=64`), `max_bf16_diff=0.0010` |
| Isolated Gate P1 (`sampled=1` & `0`) | Coupled greedy/sampled/mixed | `5/5` PASS (`sampled=1`), `5/5` PASS (`sampled=0`), `0` illegal tokens |
| Isolated Gate P2 (`M=1..289`) | BF16/FP8 tail parity & CUDAGraph | `24/24` PASS (`0` spills in `ptxas`) |
| CPU Rollback & Mask Dispatch | `pytest` unit suite | `11/11` PASS (`6.85s` container, `8.43s` macOS) |
| Live `sglang-chat` (`172,609` KV) | Synthetic canary + `t=0.6` agentic & JSON | `45/45` canary PASS; `12/12` `catalog_first` `641.7 ms` (`1.532 tasks/s`); `c=1` `128.55 tok/s`, `c=8` `478.31 tok/s` |

See [`sglang-xgrammar-lmhead-sm89/README.md`](sglang-xgrammar-lmhead-sm89/README.md), [`sglang-xgrammar-lmhead-sm89/BENCHMARKS.md`](sglang-xgrammar-lmhead-sm89/BENCHMARKS.md), and prior tagged [raw runs](benchmarks/results/).

## Build and run the operator gate

Use Linux x86-64, Docker with BuildKit and NVIDIA Container Toolkit, and a compute capability 8.9 GPU:

```sh
docker buildx build --platform linux/amd64 --load -t dashport-sm89 sglang-xgrammar-lmhead-sm89
docker run --rm --gpus device=0 --network none \
  -v "$PWD/sglang-xgrammar-lmhead-sm89/tests:/tests:ro" \
  --entrypoint /opt/sglang/bin/python dashport-sm89 \
  /tests/run_isolated_gpu_gate.py p1 --arm candidate
SGLANG_DFLASH_XGRAMMAR_SAMPLED=0 docker run --rm --gpus device=0 --network none \
  -v "$PWD/sglang-xgrammar-lmhead-sm89/tests:/tests:ro" \
  --entrypoint /opt/sglang/bin/python dashport-sm89 \
  /tests/run_isolated_gpu_gate.py p1 --arm candidate
docker run --rm --gpus device=0 --network none \
  -v "$PWD/sglang-xgrammar-lmhead-sm89/tests:/tests:ro" \
  --entrypoint /opt/sglang/bin/python dashport-sm89 \
  /tests/run_isolated_gpu_gate.py p2 --arm candidate
```

The plugin entry point is `xgrammar_lmhead_sm89`. Toggle flags: `SGLANG_XGRAMMAR_LMHEAD_SM89=0` disables the plugin, `SGLANG_DFLASH_XGRAMMAR_DRAFT=0` disables in-draft masking, `SGLANG_DFLASH_XGRAMMAR_SAMPLED=0` restores greedy-only fallback, and `SGLANG_XGRAMMAR_ACTIVE_ROW_TARGET=0` disables deferred projection.

## Reproduce the agentic measurements

```sh
python3 scripts/bench-agentic-prefix.py --base-url http://127.0.0.1:30002 --pairs 6 --concurrency 1 --iterations 2 --seed 25213
python3 scripts/bench-agentic-prefix.py --base-url http://127.0.0.1:30002 --pairs 4 --concurrency 2 --iterations 2 --seed 4417
PYTHONPATH=sglang-xgrammar-lmhead-sm89 python3 scripts/bench-agentic-mask-dispatch.py --repeats 25 --inner 10
PYTHONPATH=sglang-xgrammar-lmhead-sm89 python3 scripts/bench-agentic-mask-dispatch.py --xgrammar-tokenizer ascii --repeats 25 --inner 10
```

## Related work

Prefix reuse comes from SGLang RadixAttention. Grammar-conditioned sparse projection and grammar-aware speculative decoding have published precedents. [SGLang](https://arxiv.org/abs/2312.07104) · [Grammar-Sparse Projection](https://github.com/StealthEyeLLC/grammar-sparse-projection) · [TensorRT-LLM guided decoding](https://nvidia.github.io/TensorRT-LLM/1.4.0rc4/blogs/tech_blog/blog12_Combining_Guided_Decoding_and_Speculative_Decoding.html)
