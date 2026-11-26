"""Run compact P1 or P2 gates against the overlay on PYTHONPATH, one arm per process."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from types import SimpleNamespace

import torch


class ToolCallGrammar:
    """Fixed tool name followed by a choice of argument tokens, with a committed prefix."""

    def __init__(self) -> None:
        self.history = [9]
        self.allowed_masks = ((1 << 13), (1 << 13) | (1 << 17))

    def is_terminated(self) -> bool:
        return False

    def fill_vocab_mask(self, mask: torch.Tensor, row: int) -> None:
        mask[row].zero_()
        mask[row, 0] = self.allowed_masks[(len(self.history) - 1) % 2]

    def accept_token(self, token: int) -> None:
        if token not in (13, 17) or not (
            self.allowed_masks[(len(self.history) - 1) % 2] & (1 << token)
        ):
            raise AssertionError(f"Invalid fixture grammar token {token}")
        self.history.append(token)

    def rollback(self, count: int) -> None:
        del self.history[-count:]


def summarize(samples: list[float]) -> dict:
    ordered = sorted(samples)
    return {
        "median_us": statistics.median(samples),
        "p95_us": ordered[(95 * len(samples) + 99) // 100 - 1],
        "samples_us": samples,
    }


def _check_all_draft_legal(ids: torch.Tensor, reqs: list[SimpleNamespace]) -> int:
    illegal = 0
    ids_cpu = ids.detach().cpu()
    for b_i, req in enumerate(reqs):
        g = req.grammar
        pushed = 0
        try:
            for slot in range(1, ids_cpu.shape[1]):
                tok = int(ids_cpu[b_i, slot].item())
                mask_bit = g.allowed_masks[(len(g.history) - 1) % 2]
                if ((mask_bit >> tok) & 1) == 0:
                    illegal += 1
                    break
                g.accept_token(tok)
                pushed += 1
        finally:
            if pushed > 0:
                g.rollback(pushed)
    return illegal


def run_p1(iters: int) -> list[dict]:
    from sglang_xgrammar_lmhead_sm89.in_draft_xgrammar import (
        after_draft_runner_forward,
        around_dflash_forward_batch_generation,
        around_grammar_tree_from_linear_chain,
    )

    sampled_enabled = os.getenv("SGLANG_DFLASH_XGRAMMAR_SAMPLED", "1").strip().lower() not in (
        "0", "false", "no", "off", ""
    )

    def identity(_cls, ids):
        return ids

    fixtures = ((1, "greedy"), (1, "sampled"), (8, "greedy"), (8, "sampled"), (8, "mixed"))
    rows = []
    for batch_size, mode in fixtures:
        reqs = [SimpleNamespace(grammar=ToolCallGrammar()) for _ in range(batch_size)]
        values = [
            0.0 if mode == "greedy" or (mode == "mixed" and i % 2 == 0) else 0.3
            for i in range(batch_size)
        ]
        sampling_info = SimpleNamespace(
            is_all_greedy=mode == "greedy",
            temperatures=torch.tensor(values, device="cuda"),
            top_ks=torch.full((batch_size,), 16, dtype=torch.int32, device="cuda"),
        )
        batch = SimpleNamespace(has_grammar=True, reqs=reqs, sampling_info=sampling_info)
        ids_original = torch.tensor([0, 11, 13, 11, 11, 11, 13, 11], device="cuda").repeat(batch_size, 1)
        candidates_original = torch.tensor([11, 13, 17, *range(19, 32)], device="cuda").repeat(batch_size, 7, 1)
        probs_original = torch.tensor([0.65, 0.25, 0.10, *([0.0] * 13)], device="cuda").repeat(batch_size, 7, 1)
        ids = ids_original.clone()
        candidates = candidates_original.clone()
        probs = probs_original.clone()
        sampler = SimpleNamespace(candidate_out=candidates, q_out=probs)
        runner = SimpleNamespace(forward=lambda: SimpleNamespace(can_run_graph=True))
        worker = SimpleNamespace(
            draft_model_runner=runner,
            _draft_sampler=sampler,
            _selector_sample=(candidates, probs) if mode != "greedy" else None,
            target_worker=SimpleNamespace(model_runner=SimpleNamespace(model_config=SimpleNamespace(vocab_size=256))),
        )
        barrier_calls = 0

        def barrier():
            nonlocal barrier_calls
            barrier_calls += 1

        def forward(w, _batch, grammar_barrier=None):
            res = w.draft_model_runner.forward()
            after_draft_runner_forward(res, w.draft_model_runner)
            return around_grammar_tree_from_linear_chain(identity, None, ids)

        def reset():
            ids.copy_(ids_original)
            candidates.copy_(candidates_original)
            probs.copy_(probs_original)
            torch.cuda.synchronize()

        for _ in range(5):
            reset()
            around_dflash_forward_batch_generation(forward, worker, batch, grammar_barrier=barrier)
        durations = []
        barrier_calls = 0
        for _ in range(iters):
            reset()
            start = time.perf_counter_ns()
            around_dflash_forward_batch_generation(forward, worker, batch, grammar_barrier=barrier)
            torch.cuda.synchronize()
            durations.append((time.perf_counter_ns() - start) / 1000)

        unchanged_ids = torch.equal(ids, ids_original)
        unchanged_candidates = torch.equal(candidates, candidates_original)
        unchanged_q = torch.equal(probs, probs_original)
        rollback_exact = all(req.grammar.history == [9] for req in reqs)
        expected_greedy = torch.tensor([0, 13, 13, 13, 13, 13, 13, 13], device="cuda").expand(batch_size, -1)
        greedy_ids_match = torch.equal(ids, expected_greedy) if mode == "greedy" else None
        sorted_candidates = candidates.sort(dim=-1).values
        duplicate_count = (sorted_candidates[..., 1:] == sorted_candidates[..., :-1]).sum().item()
        illegal_tokens = _check_all_draft_legal(ids, reqs)
        q_sums_ok = bool(((probs.sum(dim=-1) - 1.0).abs() <= 1e-4).all().item())

        if mode == "greedy":
            correct = rollback_exact and greedy_ids_match and duplicate_count == 0 and illegal_tokens == 0
        elif sampled_enabled:
            correct = rollback_exact and duplicate_count == 0 and illegal_tokens == 0 and q_sums_ok
        else:
            correct = rollback_exact and unchanged_q and unchanged_candidates and unchanged_ids and duplicate_count == 0

        rows.append({
            "mode": mode, "sampled_enabled": sampled_enabled, "batch": batch_size,
            "draft_block": ids.shape[1], "correct": correct, "exact_rollback": rollback_exact,
            "greedy_ids_match": greedy_ids_match, "unchanged_ids": unchanged_ids,
            "unchanged_candidates": unchanged_candidates, "unchanged_q": unchanged_q,
            "duplicate_candidates": duplicate_count, "illegal_tokens": illegal_tokens,
            "q_sums_valid": q_sums_ok, "barrier_calls": barrier_calls,
            "first_row_ids": ids[0].tolist(), "timing": summarize(durations),
        })
    return rows


def event_times(operation, iters: int) -> dict:
    for _ in range(5):
        operation()
    torch.cuda.synchronize()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for start, stop in events:
        start.record()
        operation()
        stop.record()
    torch.cuda.synchronize()
    return summarize([start.elapsed_time(stop) * 1000 for start, stop in events])


def run_p2(iters: int) -> list[dict]:
    from sglang_xgrammar_lmhead_sm89.active_row_head import active_row_bf16_lm_head, active_row_fp8_lm_head

    n, k = 2051, 5120
    rng = torch.Generator(device="cpu").manual_seed(92033)
    weights = torch.randint(-1, 2, (n, k), generator=rng, dtype=torch.int8).to(device="cuda")
    weights_bf16 = weights.to(torch.bfloat16) * 0.125
    weights_fp8 = weights.to(torch.float8_e4m3fn)
    del weights
    scale = torch.full((n,), 0.125, dtype=torch.float32, device="cuda")
    rows = []
    for m in (1, 8, 64, 256, 257, 289):
        hidden = torch.randint(-1, 2, (m, k), generator=rng, dtype=torch.int8).to(device="cuda", dtype=torch.bfloat16)
        for active_count in (3, 21):
            active = torch.zeros((m, n), dtype=torch.bool)
            active[:, [0, n - 1]] = True
            mask = torch.zeros((m, (n + 31) // 32), dtype=torch.int32)
            for i in range(m):
                active[i, torch.randperm(n - 2, generator=rng)[:active_count - 2] + 1] = True
                for token in active[i].nonzero().flatten().tolist():
                    bit = token % 32
                    mask[i, token // 32] |= -(1 << 31) if bit == 31 else 1 << bit
            mask = mask.to(device="cuda")
            active = active.to(device="cuda")
            ref = hidden.mm(weights_bf16.T).masked_fill(~active, float("-inf"))
            out = torch.full_like(ref, 23.0)

            def bf16():
                active_row_bf16_lm_head(hidden, weights_bf16, mask, out=out)

            def fp8():
                active_row_fp8_lm_head(hidden, weights_fp8, scale, mask, out=out)

            for dtype, operation in (("bf16", bf16), ("fp8", fp8)):
                out.fill_(23.0)
                timing = event_times(operation, iters)
                equal = out == ref
                bad_logits = int((~equal).sum().item())
                bad_late_logits = int((~equal[256:]).sum().item())
                checksum = hashlib.sha256(out.detach().view(torch.int16).cpu().numpy().tobytes()).hexdigest()
                rows.append({
                    "dtype": dtype, "m": m, "n": n, "k": k, "active_per_row": active_count,
                    "correct": bad_logits == 0, "bad_logits": bad_logits, "bad_logits_row_256_plus": bad_late_logits,
                    "mask_violations": int((out.masked_select(~active) != float("-inf")).sum().item()),
                    "checksum_sha256": checksum, "timing": timing,
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("p1", "p2"))
    parser.add_argument("--arm", required=True, choices=("baseline", "candidate"))
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--fraction", type=float, default=0.012, help="PyTorch CUDA allocator fraction")
    args = parser.parse_args()
    if args.iters < 1 or not 0 < args.fraction <= 0.025:
        parser.error("iters must be positive and fraction must be in (0, 0.025]")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (8, 9):
        parser.error("an SM89 CUDA GPU is required")
    torch.set_num_threads(1)
    torch.manual_seed(18073)
    torch.cuda.set_per_process_memory_fraction(args.fraction, 0)
    rows = run_p1(args.iters) if args.mode == "p1" else run_p2(args.iters)
    report = {
        "mode": args.mode, "arm": args.arm, "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "max_torch_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "max_torch_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        "correct": all(row["correct"] for row in rows), "rows": rows,
    }
    print(json.dumps(report, allow_nan=False))
    if args.arm == "candidate" and not report["correct"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
