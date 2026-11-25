# Dashport 4090 D inference

An SM89 CUDA and XGrammar plugin for SGLang DFlash2, with a reproducible agentic tool-call benchmark. Tested on one 48 GB RTX 4090 D at a 300 W power limit. The benchmark includes tagged raw results and can also run against a stock SGLang endpoint.

## Results

The tool-call workload uses a fixed catalog of synthetic tools. Each trajectory makes two requests, one for a schema-constrained tool call and one for the answer to a synthetic tool result. Both arms use the same fields and validate the tool, arguments and final value. A changes the leading request nonce each time. B puts the shared catalog before the nonce. Each pair uses independent cache namespaces, the trial order is balanced, and the cold first group is recorded separately.

| Warm requests | First call TTFT p50 | Two-request trajectory p50 | Cached input tokens per call |
| --- | ---: | ---: | ---: |
| 1 concurrent, six-pair repeat | 439 → 97 ms | 1,060 → 374 ms | 64 → 1,408 of 1,480 |
| 2 concurrent, four pairs | 830 → 113 ms | 1,851 → 419 ms | 64 → 1,408 of 1,475 |

All 68 trajectories passed across the pilot and both repeats. The repeated cold p50 was about 1.09 s in both arms at concurrency 1 and 1.93 s at concurrency 2. A pilot warm outlier is retained in the raw results. TTFT is measured at the client on the first nonempty SSE content frame. SSE frame gaps can cover multiple tokens. These experiments used the same already-running Qwen3.8-27B-FP8 engine in both arms. Their speedup measures prefix locality in the application workload.

The plugin changes were measured independently from the application experiment.

| Isolated gate | Result |
| --- | --- |
| Sampled DFlash proposals | The updated hook preserves draft IDs, candidates and proposal probabilities on sampled and mixed batches. All five scenarios pass in both candidate runs. Sampled batch-8 hook median went from 977 to 2.8 µs by handing verification back to upstream. Greedy repair and grammar rollback remain covered. |
| CUDA row indices | BF16 and FP8 pass all 24 cases in both candidate runs. The old kernel fails on rows past 255. CUDA Graph replay passes with a changing mask at 256, 257 and 289 rows. Across the already-correct shapes, the median candidate/baseline CUDA timing ratio is 1.00. |
| XGrammar mask dispatch | Replaying tool-call masks with the pinned Qwen tokenizer, 64-row CPU dispatch median went from 3,280 to 111 µs on a tool request and from 3,282 to 16 µs on the follow-up. All 12 tested masks preserve the sparse/dense decision, including eight seeded sparse, mixed and dense cases. |

The CPU timing covers the dispatch decision only. Both real grammar traces selected the same dense projection in both arms. The GPU operator benchmark uses a smaller synthetic vocabulary and does not characterize end-to-end throughput. The updated overlay has been built and tested separately from the live model. A full before/after model-serving performance comparison for the combined plugin is pending. [Raw runs](benchmarks/results/)

## Build and run the operator gate

Use Linux x86-64, a Docker installation with BuildKit and NVIDIA Container Toolkit, and a compute capability 8.9 GPU. The Dockerfile pins the SGLang runtime image and builds a platform-specific wheel from source with CUDA 13.0 and PyTorch 2.14.1+cu130. The build itself uses no GPU.

```sh
docker buildx build --platform linux/amd64 --load -t dashport-sm89 sglang-xgrammar-lmhead-sm89
docker run --rm --gpus device=0 --network none \
  -v "$PWD/sglang-xgrammar-lmhead-sm89/tests:/tests:ro" \
  --entrypoint /opt/sglang/bin/python dashport-sm89 \
  /tests/run_isolated_gpu_gate.py p1 --arm candidate
docker run --rm --gpus device=0 --network none \
  -v "$PWD/sglang-xgrammar-lmhead-sm89/tests:/tests:ro" \
  --entrypoint /opt/sglang/bin/python dashport-sm89 \
  /tests/run_isolated_gpu_gate.py p2 --arm candidate
```

For baseline reproduction, make a second checkout, apply [the baseline patch](benchmarks/legacy-p1-p2.patch) there, and build it under a different image tag. Run each arm in a separate process with the corresponding `--arm` value. The included baseline uses the original faulty paths, so the baseline harness records expected correctness failures.

The plugin entry point is `xgrammar_lmhead_sm89`. Use the pinned image with an existing SGLang DFlash2 launch configuration and `--grammar-backend xgrammar`. `SGLANG_XGRAMMAR_LMHEAD_SM89=0` disables the plugin. `SGLANG_DFLASH_XGRAMMAR_DRAFT=0` disables only greedy draft repair, and `SGLANG_XGRAMMAR_ACTIVE_ROW_TARGET=0` disables the deferred projection. The kernel uses BF16 projection with a dense fallback. Its accumulation order can differ from dense matmul.

## Reproduce the agentic measurements

Run the prefix benchmark against an existing local SGLang server. It submits synthetic generations and executes no tools. The default model name is `qwen3.8-27b-fp8`. Use `--model` to match another model name.

```sh
python3 scripts/bench-agentic-prefix.py --base-url http://127.0.0.1:30002 --pairs 6 --concurrency 1 --iterations 2 --seed 25213
python3 scripts/bench-agentic-prefix.py --base-url http://127.0.0.1:30002 --pairs 4 --concurrency 2 --iterations 2 --seed 4417
```

For the CPU experiment, use the same runtime's NumPy and XGrammar. `--xgrammar-tokenizer` accepts a local snapshot with tokenizer and model config files. Omit it for deterministic synthetic masks. No model weights are needed for CPU mask replay. The recorded tokenizer is from [Qwen3.8-27B-FP8 at revision `017b9c7`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8/tree/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a).

```sh
PYTHONPATH=sglang-xgrammar-lmhead-sm89 python3 scripts/bench-agentic-mask-dispatch.py --repeats 25 --inner 10
PYTHONPATH=sglang-xgrammar-lmhead-sm89 python3 scripts/bench-agentic-mask-dispatch.py --xgrammar-tokenizer ascii --repeats 25 --inner 10
```

## Related work

Prefix reuse comes from SGLang RadixAttention. Grammar-conditioned sparse projection and grammar-aware speculative decoding have published precedents. This project provides the SM89 integration, its correctness regressions and an agentic benchmark with cold and warm runs. [SGLang](https://arxiv.org/abs/2312.07104) · [Grammar-Sparse Projection](https://github.com/StealthEyeLLC/grammar-sparse-projection) · [TensorRT-LLM guided decoding](https://nvidia.github.io/TensorRT-LLM/1.4.0rc4/blogs/tech_blog/blog12_Combining_Guided_Decoding_and_Speculative_Decoding.html)