"""CPU unit tests for in-draft xgrammar masking, rollback, coupled sampled TV, and eager/graph isolation."""

from __future__ import annotations

import zlib
from types import SimpleNamespace

import numpy as np
import torch

from sglang_xgrammar_lmhead_sm89.active_row_head import union_active_row_fraction
from sglang_xgrammar_lmhead_sm89.in_draft_xgrammar import (
    after_compute_candidates,
    after_draft_runner_forward,
    around_dflash_forward_batch_generation,
    around_grammar_tree_from_linear_chain,
    find_unique_allowed_token,
    is_token_allowed,
    refine_draft_block_with_grammar,
)


class _MockGrammar:
    """Deterministic finite-automaton mock tracking exact accept/rollback invariants."""

    def __init__(self, allowed_steps: list[list[int]], vocab_size: int) -> None:
        self.allowed_steps = allowed_steps
        self.vocab_size = vocab_size
        self.accepted_tokens: list[int] = [999]
        self.cursor = 0

    def is_terminated(self) -> bool:
        return self.cursor >= len(self.allowed_steps)

    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        row = vocab_mask[idx]
        row.zero_()
        for tok in self.allowed_steps[self.cursor]:
            row[tok >> 5] |= 1 << (tok & 31)

    def accept_token(self, token: int) -> None:
        assert token in self.allowed_steps[self.cursor], f"Illegal {token} at step {self.cursor}"
        self.accepted_tokens.append(token)
        self.cursor += 1

    def rollback(self, k: int) -> None:
        assert k > 0, "XGrammarGrammar.rollback(0) bug guard triggered!"
        self.accepted_tokens = self.accepted_tokens[:-k]
        self.cursor -= k


class _MockReq:
    def __init__(self, grammar) -> None:
        self.grammar = grammar


def test_bitmask_helpers_and_density() -> None:
    vocab_size = 248320
    words = (vocab_size + 31) // 32
    m = torch.zeros((1, words), dtype=torch.int32)
    m[0, 1234 >> 5] = 1 << (1234 & 31)
    assert is_token_allowed(m[0].numpy(), 1234, vocab_size)
    assert not is_token_allowed(m[0].numpy(), 1235, vocab_size)
    assert find_unique_allowed_token(m[0].numpy(), vocab_size) == 1234

    m[0, 5000 >> 5] |= 1 << (5000 & 31)
    assert find_unique_allowed_token(m[0].numpy(), vocab_size) == -1

    frac, total_pairs, active_tiles = union_active_row_fraction(m, vocab_size)
    assert total_pairs == 2
    assert active_tiles == 2
    assert frac < 0.002


def test_refine_draft_block_rollback_and_jump_forward() -> None:
    vocab_size = 256
    g0 = _MockGrammar([[42], [10, 20], [77], [88], [88], [88], [88]], vocab_size)
    g1 = _MockGrammar([[100, 101], [5]], vocab_size)
    reqs = [_MockReq(g0), _MockReq(g1)]

    draft_cpu = torch.tensor(
        [
            [999, 7, 99, 1, 2, 3, 4, 5],
            [999, 101, 6, 7, 8, 9, 10, 11],
        ],
        dtype=torch.int64,
    )
    cand_cpu = torch.zeros((2, 7, 4), dtype=torch.int64)
    cand_cpu[0, 1] = torch.tensor([99, 20, 10, 3])
    cand_cpu[1, 1] = torch.tensor([6, 7, 8, 9])

    draft_modified = refine_draft_block_with_grammar(
        reqs=reqs,
        draft_cpu=draft_cpu,
        cand_cpu=cand_cpu,
        vocab_size=vocab_size,
    )
    assert draft_modified is True
    assert draft_cpu[0].tolist() == [999, 42, 20, 77, 88, 88, 88, 88]
    assert draft_cpu[1, 1].item() == 101
    assert g0.cursor == 0 and g0.accepted_tokens == [999]
    assert g1.cursor == 0 and g1.accepted_tokens == [999]


def test_terminated_grammar_is_not_rolled_back() -> None:
    grammar = _MockGrammar([], vocab_size=256)
    draft = torch.tensor([[1, 2, 3]], dtype=torch.int64)
    assert not refine_draft_block_with_grammar(
        reqs=[_MockReq(grammar)],
        draft_cpu=draft,
        cand_cpu=None,
        vocab_size=256,
    )
    assert grammar.cursor == 0 and grammar.accepted_tokens == [999]


def test_sampled_and_mixed_batches_leave_proposals_untouched_when_disabled(monkeypatch) -> None:
    """Instant rollback switch SGLANG_DFLASH_XGRAMMAR_SAMPLED=0 leaves sampled batches untouched."""
    monkeypatch.setenv("SGLANG_DFLASH_XGRAMMAR_SAMPLED", "0")
    grammar0 = _MockGrammar([[13]], vocab_size=256)
    grammar1 = _MockGrammar([[12, 13], [14, 15]], vocab_size=256)
    draft = torch.tensor([[1, 99, 3], [1, 12, 99]], dtype=torch.int64)
    candidates = torch.tensor(
        [[[99, 13, 12], [4, 5, 6]], [[12, 13, 99], [99, 14, 15]]], dtype=torch.int64
    )
    q = torch.tensor(
        [[[0.50, 0.45, 0.05], [0.6, 0.3, 0.1]], [[0.45, 0.05, 0.50], [0.50, 0.45, 0.05]]]
    )
    before = [t.clone() for t in (draft, candidates, q)]
    worker = SimpleNamespace(
        _draft_sampler=SimpleNamespace(candidate_out=candidates, q_out=q),
        _selector_sample=(candidates, q),
        target_worker=SimpleNamespace(
            model_runner=SimpleNamespace(model_config=SimpleNamespace(vocab_size=256))
        ),
    )
    calls = []

    def build_tree(cls, ids):
        calls.append(ids)
        return ids

    def forward(worker, batch, *, grammar_barrier):
        return around_grammar_tree_from_linear_chain(build_tree, object, draft)

    def unexpected_barrier():
        raise AssertionError("disabled sampled mode must skip grammar barrier")

    for greedy_first in (False, True):
        batch = SimpleNamespace(
            has_grammar=True,
            reqs=[_MockReq(grammar0), _MockReq(grammar1)],
            sampling_info=SimpleNamespace(
                is_all_greedy=False,
                is_any_greedy=greedy_first,
                temperatures=torch.tensor([[0.0 if greedy_first else 0.3], [0.3]]),
                top_ks=torch.tensor([3, 3], dtype=torch.int32),
            ),
        )
        result = around_dflash_forward_batch_generation(
            forward, worker, batch, grammar_barrier=unexpected_barrier
        )
        assert result is draft
        for expected, actual in zip(before, (draft, candidates, q)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert grammar0.cursor == 0 and grammar0.accepted_tokens == [999]
        assert grammar1.cursor == 0 and grammar1.accepted_tokens == [999]

    assert len(calls) == 2


def test_greedy_hook_refines_draft_after_barrier() -> None:
    grammar = _MockGrammar([[12, 13]], vocab_size=256)
    draft = torch.tensor([[1, 99]], dtype=torch.int64)
    candidates = torch.tensor([[[99, 13, 12]]], dtype=torch.int64)
    runner = SimpleNamespace(forward=lambda: SimpleNamespace(can_run_graph=True))
    worker = SimpleNamespace(
        draft_model_runner=runner,
        _draft_sampler=SimpleNamespace(candidate_out=candidates),
        target_worker=SimpleNamespace(
            model_runner=SimpleNamespace(model_config=SimpleNamespace(vocab_size=256))
        ),
    )
    batch = SimpleNamespace(
        has_grammar=True,
        reqs=[_MockReq(grammar)],
        sampling_info=SimpleNamespace(is_all_greedy=True),
    )
    barriers = []

    def build_tree(cls, ids):
        assert barriers == [True]
        assert ids.tolist() == [[1, 13]]
        assert grammar.cursor == 0 and grammar.accepted_tokens == [999]
        return ids

    def forward(worker, batch, *, grammar_barrier):
        worker.draft_model_runner.forward()
        return around_grammar_tree_from_linear_chain(build_tree, object, draft)

    assert around_dflash_forward_batch_generation(
        forward, worker, batch, grammar_barrier=lambda: barriers.append(True)
    ) is draft


def test_coupled_sampled_eager_isolation_and_rejection_tv(monkeypatch) -> None:
    """With SAMPLED=1, verify stale sampler isolation + 3-step coupled rejection TV < 0.01."""
    monkeypatch.setenv("SGLANG_DFLASH_XGRAMMAR_SAMPLED", "1")
    seed_val = zlib.crc32(b"dashport_coupled_tv") % 10000
    torch.manual_seed(9000 + seed_val)
    rng = np.random.default_rng(1200 + seed_val)

    vocab_size = 128
    # 1. Poison stale _draft_sampler buffers on an eager sampled step.
    stale_c = torch.tensor([[[99, 98]]], dtype=torch.int64)
    stale_q = torch.tensor([[[0.1, 0.9]]], dtype=torch.float32)
    fresh_c = torch.tensor([[[10, 11]]], dtype=torch.int64)
    fresh_q = torch.tensor([[[0.9, 0.1]]], dtype=torch.float32)

    eager_runner = SimpleNamespace(forward=lambda: SimpleNamespace(can_run_graph=False))
    w_iso = SimpleNamespace(
        draft_model_runner=eager_runner,
        _draft_sampler=SimpleNamespace(candidate_out=stale_c.clone(), q_out=stale_q.clone()),
        _selector_sample=(fresh_c, fresh_q),
        target_worker=SimpleNamespace(model_runner=SimpleNamespace(model_config=SimpleNamespace(vocab_size=vocab_size))),
    )
    g_iso = _MockGrammar([[10, 12, 13]], vocab_size)
    ids_iso = torch.tensor([[1, 11]], dtype=torch.int64)
    b_iso = SimpleNamespace(
        has_grammar=True,
        reqs=[_MockReq(g_iso)],
        sampling_info=SimpleNamespace(
            is_all_greedy=False,
            temperatures=torch.tensor([0.7]),
            top_ks=torch.tensor([2]),
        ),
    )

    def _run_iso(w, b):
        res = w.draft_model_runner.forward()
        after_draft_runner_forward(res, w.draft_model_runner)
        after_compute_candidates(fresh_c)
        return around_grammar_tree_from_linear_chain(lambda _c, x: x, object, ids_iso)

    around_dflash_forward_batch_generation(_run_iso, w_iso, b_iso)
    assert torch.allclose(w_iso._selector_sample[1][0, 0], torch.tensor([1.0, 0.0]))
    assert int(ids_iso[0, 1].item()) == 10
    assert torch.allclose(w_iso._draft_sampler.q_out, stale_q)
    assert torch.equal(w_iso._draft_sampler.candidate_out, stale_c)

    # 2. Full 3-step sequential trajectory TV < 0.01 through coupled sampler + residual verify.
    class _ThreeStepGrammar:
        def __init__(self) -> None:
            self.h: list[int] = []

        @staticmethod
        def legal_for(prefix: tuple[int, ...]) -> tuple[int, ...]:
            if len(prefix) == 0:
                return (1, 2)
            if len(prefix) == 1:
                return (4,) if prefix[0] == 1 else (4, 5)
            return (7, 8) if prefix[1] == 4 else (6, 9)

        def is_terminated(self) -> bool:
            return len(self.h) >= 3

        def fill_vocab_mask(self, m: torch.Tensor, idx: int) -> None:
            m[idx].zero_()
            for t in self.legal_for(tuple(self.h)):
                m[idx, t >> 5] |= 1 << (t & 31)

        def accept_token(self, t: int) -> None:
            self.h.append(int(t))

        def rollback(self, n: int) -> None:
            del self.h[-n:]

    def p_tgt(prefix: tuple[int, ...]) -> np.ndarray:
        scores = np.array([0.3 + 0.4 * ((len(prefix) + k) % 3) for k in range(12)], dtype=np.float64)
        allowed = list(_ThreeStepGrammar.legal_for(prefix))
        mask_arr = np.full(12, -1e18, dtype=np.float64)
        mask_arr[allowed] = scores[allowed]
        e = np.exp(mask_arr - np.max(mask_arr))
        return e / e.sum()

    def q_prop(prefix: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        allowed = list(_ThreeStepGrammar.legal_for(prefix))
        illegal = [k for k in range(12) if k not in allowed]
        cands = np.array((allowed + illegal)[:5], dtype=np.int64)
        w = np.array([1.2, 0.7, 0.9, 1.1, 0.8], dtype=np.float64)
        return cands, w / w.sum()

    exact = {}
    for a in _ThreeStepGrammar.legal_for(()):
        for b in _ThreeStepGrammar.legal_for((a,)):
            for c in _ThreeStepGrammar.legal_for((a, b)):
                exact[(a, b, c)] = float(p_tgt(())[a] * p_tgt((a,))[b] * p_tgt((a, b))[c])

    counts: dict[tuple[int, int, int], int] = {}
    total_n = 56000
    chunk = 8000
    w_mc = SimpleNamespace(
        draft_model_runner=SimpleNamespace(forward=lambda: SimpleNamespace(can_run_graph=False)),
        _draft_sampler=SimpleNamespace(candidate_out=torch.full((chunk, 3, 5), 11, dtype=torch.int64),
                                       q_out=torch.full((chunk, 3, 5), 0.2, dtype=torch.float32)),
        _selector_sample=None,
        target_worker=SimpleNamespace(model_runner=SimpleNamespace(model_config=SimpleNamespace(vocab_size=12))),
    )

    for _ in range(total_n // chunk):
        hists: list[list[int]] = [[] for _ in range(chunk)]
        grams = [_ThreeStepGrammar() for _ in range(chunk)]
        active = list(range(chunk))
        while active:
            m = len(active)
            v_ids = torch.zeros((m, 4), dtype=torch.int64)
            c_t = torch.zeros((m, 3, 5), dtype=torch.int64)
            q_t = torch.zeros((m, 3, 5), dtype=torch.float32)
            u_draw = rng.random((m, 3))
            for ri, req_i in enumerate(active):
                sim = list(hists[req_i])
                if sim:
                    v_ids[ri, 0] = sim[-1]
                for s_i in range(3):
                    hk = tuple(sim) if len(sim) < 3 else ()
                    cands, probs = q_prop(hk)
                    c_t[ri, s_i] = torch.from_numpy(cands)
                    q_t[ri, s_i] = torch.from_numpy(probs.astype(np.float32))
                    u = float(u_draw[ri, s_i])
                    acc_p = 0.0
                    pick = int(cands[-1])
                    for k_i in range(4):
                        acc_p += float(probs[k_i])
                        if u < acc_p:
                            pick = int(cands[k_i])
                            break
                    v_ids[ri, s_i + 1] = pick
                    if len(sim) < 3:
                        sim.append(pick)
            w_mc._selector_sample = (c_t, q_t)
            b_mc = SimpleNamespace(
                has_grammar=True,
                reqs=[_MockReq(grams[idx]) for idx in active],
                sampling_info=SimpleNamespace(
                    is_all_greedy=False,
                    temperatures=torch.ones(m),
                    top_ks=torch.full((m,), 5, dtype=torch.int32),
                ),
            )
            around_dflash_forward_batch_generation(
                lambda w, _b: around_grammar_tree_from_linear_chain(lambda _c, x: x, object, v_ids),
                w_mc, b_mc,
            )
            c_np = c_t.numpy()
            q_np = q_t.numpy().astype(np.float64)
            v_np = v_ids.numpy()
            u_acc = rng.random((m, 3))
            u_res = rng.random((m, 3))
            next_active = []
            for ri, req_i in enumerate(active):
                h = hists[req_i]
                g = grams[req_i]
                sl = 0
                while len(h) < 3 and sl < 3:
                    pm = p_tgt(tuple(h))
                    qe = np.zeros(12, dtype=np.float64)
                    for k_i in range(5):
                        qe[int(c_np[ri, sl, k_i])] += q_np[ri, sl, k_i]
                    dt = int(v_np[ri, sl + 1])
                    alpha = min(1.0, pm[dt] / qe[dt]) if qe[dt] > 0.0 else 0.0
                    if float(u_acc[ri, sl]) < alpha:
                        h.append(dt)
                        g.accept_token(dt)
                        sl += 1
                    else:
                        res = np.maximum(0.0, pm - qe)
                        res /= res.sum()
                        ur = float(u_res[ri, sl])
                        ap = 0.0
                        yr = 11
                        for k_i in range(12):
                            ap += float(res[k_i])
                            if ur < ap:
                                yr = k_i
                                break
                        h.append(yr)
                        g.accept_token(yr)
                        break
                if len(h) < 3:
                    next_active.append(req_i)
                else:
                    tr = (h[0], h[1], h[2])
                    counts[tr] = counts.get(tr, 0) + 1
            active = next_active

    assert all(tr in exact for tr in counts)
    tv = 0.5 * sum(abs((counts.get(tr, 0) / total_n) - pe) for tr, pe in exact.items())
    assert tv < 0.01, f"MC TV={tv:.5f} >= 0.01"


if __name__ == "__main__":
    test_bitmask_helpers_and_density()
    test_refine_draft_block_rollback_and_jump_forward()
    test_terminated_grammar_is_not_rolled_back()
    test_greedy_hook_refines_draft_after_barrier()
    print("CPU_ROLLBACK_AND_DENSITY_TESTS_PASS")
