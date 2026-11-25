#!/usr/bin/env python3
"""Compare synthetic tool-call trajectories with volatile-first and catalog-first prompts."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import random
import statistics
import time
import urllib.request
import uuid


ARMS = ("volatile_first", "catalog_first")
PHASES = ("call", "answer")
MODEL = "qwen3.8-27b-fp8"
MAX_TOKENS = 64
SYSTEM = (
    "You are a synthetic benchmark agent. The user sends a JSON workflow envelope. "
    "Property order has no meaning. Follow the task and return only the requested JSON. "
    "Use the supplied tool result as the source for the answer."
)
CATALOG = [
    {
        "name": f"lookup_{name}",
        "description": (
            f"Read a single synthetic {name} record from the benchmark registry. "
            "Use its numeric ID when the task names the corresponding record type. "
            "This read-only lookup returns the matching ID and its numeric value. "
            "Report missing data separately from a zero value. Do not invent records."
        ),
        "arguments": {"id": "integer"},
        "result": {"id": "integer", "value": "integer"},
    }
    for name in (
        "account", "invoice", "order", "shipment", "product", "vendor",
        "customer", "ticket", "refund", "warehouse", "inventory", "transfer",
        "payment", "contract", "allocation", "budget", "campaign", "subscription",
    )
]


def object_schema(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


CALL_SCHEMA = object_schema({
    "tool": {"type": "string", "enum": [tool["name"] for tool in CATALOG[:4]]},
    "args": object_schema({"id": {"type": "integer"}}),
})
ANSWER_SCHEMA = object_schema({"value": {"type": "integer"}})


def compact(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True)


def build_payload(model, namespace, arm, nonce, phase, record_id, record_value, tool_name):
    fields = [("nonce", nonce), ("catalog", CATALOG)]
    if arm == "catalog_first":
        fields.reverse()
    envelope = dict(fields)
    if phase == "call":
        envelope["task"] = f"Call {tool_name} with id {record_id}."
        schema = CALL_SCHEMA
        expected = {"tool": tool_name, "args": {"id": record_id}}
    else:
        envelope["task"] = "Read the tool result and return its value."
        envelope["tool_result"] = {
            "tool": tool_name,
            "id": record_id,
            "value": record_value,
        }
        schema = ANSWER_SCHEMA
        expected = {"value": record_value}
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": f"Experiment namespace {namespace}. {SYSTEM}"},
            {"role": "user", "content": compact(envelope)},
        ],
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True,
        "stream_options": {"include_usage": True},
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": phase, "strict": True, "schema": schema},
        },
    }, expected


def execute(url, timeout, payload, expected):
    request = urllib.request.Request(
        url,
        data=compact(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    frames = []
    pieces = []
    usage = {}
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            if event.get("error"):
                raise RuntimeError(f"SSE error: {event['error']}")
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                piece = choice.get("delta", {}).get("content")
                if piece:
                    frames.append(time.perf_counter() - start)
                    pieces.append(piece)
    duration = time.perf_counter() - start
    try:
        actual = json.loads("".join(pieces))
        valid = json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)
    except ValueError:
        actual = None
        valid = False
    details = usage.get("prompt_tokens_details") or {}
    cached_tokens = details.get("cached_tokens", usage.get("cached_tokens"))
    return {
        "valid": valid,
        "actual": actual,
        "ttft_ms": round(frames[0] * 1000, 3) if frames else None,
        "frame_gaps_ms": [round((b - a) * 1000, 3) for a, b in zip(frames, frames[1:])],
        "wall_ms": round(duration * 1000, 3),
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": cached_tokens,
        "completion_tokens": usage.get("completion_tokens"),
    }


def trajectory(args, pair, arm, namespace, iteration, slot):
    rng = random.Random(f"{args.seed}/{pair}/{iteration}/{slot}")
    tool = CATALOG[rng.randrange(4)]["name"]
    record_id = rng.randrange(100, 1000)
    record_value = rng.randrange(1000, 9000)
    start = time.perf_counter()
    phases = {}
    for index, phase in enumerate(PHASES):
        nonce = f"{iteration:02d}-{slot:02d}-{index:02d}"
        payload, expected = build_payload(
            args.model, namespace, arm, nonce, phase, record_id, record_value, tool,
        )
        if args.dry_run:
            phases[phase] = {
                "expected": expected,
                "user_prompt_bytes": len(payload["messages"][-1]["content"]),
            }
        else:
            phases[phase] = execute(args.url, args.timeout, payload, expected)
    return {
        "pair": pair,
        "arm": arm,
        "iteration": iteration,
        "slot": slot,
        "cache_phase": "initial_group" if iteration == 0 else "subsequent_group",
        "wall_ms": round((time.perf_counter() - start) * 1000, 3),
        "phases": phases,
    }


def percentile95(values):
    return round(sorted(values)[(95 * len(values) + 99) // 100 - 1], 3)


def summarize(trials):
    summary = {}
    for arm in ARMS:
        summary[arm] = {}
        for group in ("initial_group", "subsequent_group"):
            subset = [row for row in trials if row["arm"] == arm and row["cache_phase"] == group]
            row = {
                "trajectories": len(subset),
                "valid_trajectories": sum(all(p["valid"] for p in t["phases"].values()) for t in subset),
                "trajectory_p50_ms": statistics.median(t["wall_ms"] for t in subset),
                "trajectory_p95_ms": percentile95([t["wall_ms"] for t in subset]),
            }
            for phase in PHASES:
                steps = [t["phases"][phase] for t in subset]
                row[phase] = {}
                for metric in ("ttft_ms", "cached_tokens", "prompt_tokens", "completion_tokens"):
                    known = [s[metric] for s in steps if s[metric] is not None]
                    row[phase][f"{metric}_p50"] = statistics.median(known) if known else None
                    row[phase][f"{metric}_observed"] = len(known)
                frame_gaps = [gap for s in steps for gap in s["frame_gaps_ms"]]
                row[phase]["sse_frame_gap_p50_ms"] = statistics.median(frame_gaps) if frame_gaps else None
            summary[arm][group] = row
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30002")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--concurrency", type=int, choices=(1, 2), default=1)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1741)
    parser.add_argument("--run-id", default=uuid.uuid4().hex)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.pairs < 1 or args.iterations < 2 or args.iterations > 99 or args.timeout <= 0:
        parser.error("Use --pairs >= 1, --iterations between 2 and 99, and --timeout > 0.")
    args.url = args.base_url.rstrip("/") + "/v1/chat/completions"
    rng = random.Random(args.seed)
    swaps = [False, True] * (args.pairs // 2)
    if args.pairs % 2:
        swaps.append(bool(rng.randrange(2)))
    rng.shuffle(swaps)
    trials = []
    arm_orders = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        for pair, swap in enumerate(swaps):
            order = ARMS[::-1] if swap else ARMS
            arm_orders.append(order)
            # The different equal-width leading tags keep the arms from sharing a catalog cache.
            tags = ["A", "B"]
            rng.shuffle(tags)
            for arm, tag in zip(order, tags):
                namespace = f"{tag}-{args.run_id}-{pair:04d}"
                for iteration in range(args.iterations):
                    tasks = [
                        executor.submit(trajectory, args, pair, arm, namespace, iteration, slot)
                        for slot in range(args.concurrency)
                    ]
                    trials.extend(task.result() for task in tasks)

    report = {
        "config": {
            "model": args.model,
            "pairs": args.pairs,
            "iterations": args.iterations,
            "concurrency": args.concurrency,
            "seed": args.seed,
            "run_id": args.run_id,
            "catalog_bytes": len(compact(CATALOG)),
            "requests": len(trials) * len(PHASES),
            "dry_run": args.dry_run,
        },
        "arm_orders": arm_orders,
        "trials": trials,
    }
    if not args.dry_run:
        report["summary"] = summarize(trials)
    print(json.dumps(report, indent=2))
    if not args.dry_run and any(not p["valid"] for t in trials for p in t["phases"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
