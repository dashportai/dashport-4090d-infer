#!/usr/bin/env python3
"""Replay packed grammar masks to measure CPU projection dispatch, without a model server."""

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sglang-xgrammar-lmhead-sm89"))
from sglang_xgrammar_lmhead_sm89.active_row_head import (  # noqa: E402
    is_active_pair_count_within_limit,
    union_active_row_fraction,
)


VOCAB_SIZE = 248_320
DEFAULT_MAX_PAIRS = 180_000


def build_workloads(rng):
    words = VOCAB_SIZE // 32
    for rows in (8, 64):
        sparse = np.zeros((rows, words), dtype=np.uint32)
        for row in sparse:
            tokens = rng.choice(VOCAB_SIZE, size=VOCAB_SIZE // 1_000, replace=False)
            np.bitwise_or.at(row, tokens // 32, np.uint32(1) << (tokens % 32).astype(np.uint32))
        yield "sparse", torch.from_numpy(sparse.view(np.int32))

        for name, row_index in (("mixed_first", 0), ("mixed_last", rows - 1)):
            mixed = sparse.copy()
            mixed[row_index].fill(np.iinfo(np.uint32).max)
            yield name, torch.from_numpy(mixed.view(np.int32))

        dense = rng.integers(-(2**31), 2**31, size=(rows, words), dtype=np.int32)
        yield "dense", torch.from_numpy(dense)


def build_xgrammar_workloads(tokenizer_path):
    import xgrammar as xgr

    if tokenizer_path == "ascii":
        vocabulary = [chr(index) for index in range(32, 127)]
        token_ids = {token: index for index, token in enumerate(vocabulary)}
        tokenizer_info = xgr.TokenizerInfo(
            [*vocabulary, "<eos>"], vocab_size=VOCAB_SIZE, stop_token_ids=[len(vocabulary)]
        )

        def encode(text):
            return [token_ids[character] for character in text]
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=VOCAB_SIZE)

        def encode(text):
            return tokenizer.encode(text, add_special_tokens=False)

    compiler = xgr.GrammarCompiler(tokenizer_info)
    fixtures = (
        (
            "tool_request",
            {"tool": {"enum": ["lookup_inventory", "reserve_inventory"]}, "sku": {"type": "string"}},
            {"tool": "lookup_inventory", "sku": "WIDGET-42"},
        ),
        (
            "tool_followup",
            {"action": {"enum": ["answer", "retry"]}, "summary": {"type": "string"}},
            {"action": "answer", "summary": "Stock is available in two warehouses."},
        ),
    )
    for name, properties, value in fixtures:
        schema = {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }
        compiled = compiler.compile_json_schema(json.dumps(schema))
        matcher = xgr.GrammarMatcher(compiled)
        tokens = encode(json.dumps(value, separators=(",", ":")))
        if len(tokens) < 8:
            raise RuntimeError(f"{name} has fewer than eight trace positions")
        trace = xgr.allocate_token_bitmask(len(tokens), VOCAB_SIZE)
        for index, token in enumerate(tokens):
            matcher.fill_next_token_bitmask(trace, index)
            if not matcher.accept_token(token):
                raise RuntimeError(f"{name} was rejected at trace position {index}")

        starts = np.linspace(0, len(tokens) - 8, 8, dtype=np.int64)
        windows = [trace[start : start + 8] for start in starts]
        yield name, windows[len(windows) // 2]
        yield name, torch.cat(windows)


def time_mean_us(operation, mask, inner):
    started = time.perf_counter_ns()
    for _ in range(inner):
        operation(mask)
    return (time.perf_counter_ns() - started) / (inner * 1_000)


def summarize(samples):
    p50 = statistics.median(samples)
    p95 = statistics.quantiles(samples, n=20, method="inclusive")[-1]
    return {"p50_us": p50, "p95_us": p95, "samples_us": samples}


def run_case(name, mask, vocab_size, max_pairs, repeats, inner):
    def baseline(tensor):
        return union_active_row_fraction(tensor, vocab_size)[1] <= max_pairs

    def candidate(tensor):
        return is_active_pair_count_within_limit(tensor, vocab_size, max_pairs)

    operations = {"baseline": baseline, "candidate": candidate}
    sparse = baseline(mask)
    if candidate(mask) != sparse:
        raise RuntimeError(f"dispatch decisions differ for {name} with {mask.shape[0]} rows")
    for operation in operations.values():
        time_mean_us(operation, mask, inner)

    measurements = {name: [] for name in operations}
    for repeat in range(repeats):
        order = operations if repeat % 2 == 0 else reversed(operations)
        for arm in order:
            measurements[arm].append(time_mean_us(operations[arm], mask, inner))

    baseline_summary = summarize(measurements["baseline"])
    candidate_summary = summarize(measurements["candidate"])
    return {
        "case": name,
        "shape": list(mask.shape),
        "vocab_size": vocab_size,
        "active_pairs": union_active_row_fraction(mask, vocab_size)[1],
        "route": "sparse" if sparse else "dense",
        "agree": True,
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "p50_speedup": baseline_summary["p50_us"] / candidate_summary["p50_us"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--inner", type=int, default=12, help="Dispatch calls per timed sample")
    parser.add_argument("--max-pairs", type=int, default=DEFAULT_MAX_PAIRS)
    parser.add_argument(
        "--xgrammar-tokenizer",
        metavar="PATH",
        help="Replay real XGrammar masks from a local Hugging Face tokenizer or use 'ascii'",
    )
    args = parser.parse_args()
    if args.repeats < 2 or args.inner < 1:
        parser.error("--repeats must be at least 2 and --inner at least 1")

    rng = np.random.default_rng(args.seed)
    if args.xgrammar_tokenizer:
        workloads = build_xgrammar_workloads(args.xgrammar_tokenizer)
        scope = "CPU dispatch replay on XGrammar masks from synthetic tool-call/follow-up JSON"
    else:
        workloads = build_workloads(rng)
        scope = "CPU projection dispatch on seeded synthetic packed masks"
    cases = [
        run_case(name, mask, VOCAB_SIZE, args.max_pairs, args.repeats, args.inner)
        for name, mask in workloads
    ]
    report = {
        "scope": scope,
        "clock": "perf_counter_ns with alternating AB/BA; each sample is the mean of --inner calls",
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "config": {
            **vars(args),
            "xgrammar_tokenizer": (
                Path(args.xgrammar_tokenizer).name if args.xgrammar_tokenizer else None
            ),
        },
        "cases": cases,
    }
    if args.xgrammar_tokenizer:
        report["xgrammar"] = version("xgrammar")
        tokenizer_json = Path(args.xgrammar_tokenizer) / "tokenizer.json"
        if tokenizer_json.is_file():
            report["tokenizer_sha256"] = hashlib.sha256(tokenizer_json.read_bytes()).hexdigest()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
