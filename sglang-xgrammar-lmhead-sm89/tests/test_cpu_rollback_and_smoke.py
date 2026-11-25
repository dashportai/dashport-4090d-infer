"""CPU unit tests for in-draft xgrammar masking, rollback invariants, and density helper."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_xgrammar_lmhead_sm89.active_row_head import union_active_row_fraction
from sglang_xgrammar_lmhead_sm89.in_draft_xgrammar import (
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
        self.accepted_tokens: list[int] = [999]  # pre-committed prefix token
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

    # Two bits -> not unique
    m[0, 5000 >> 5] |= 1 << (5000 & 31)
    assert find_unique_allowed_token(m[0].numpy(), vocab_size) == -1

    frac, total_pairs, active_tiles = union_active_row_fraction(m, vocab_size)
    assert total_pairs == 2
    assert active_tiles == 2
    assert frac < 0.002


def test_refine_draft_block_rollback_and_jump_forward() -> None:
    vocab_size = 256
    # Step 0: forced unique token 42; Step 1: legal set {10, 20}; Step 2: {77}; Step 3..6: {88}
    g0 = _MockGrammar([[42], [10, 20], [77], [88], [88], [88], [88]], vocab_size)
    g1 = _MockGrammar([[100, 101], [5]], vocab_size)
    reqs = [_MockReq(g0), _MockReq(g1)]

    # Draft puts wrong token 7 at pos 1 (should be forced 42), illegal 99 at pos 2 (cand has 20), etc.
    draft_cpu = torch.tensor(
        [
            [999, 7, 99, 1, 2, 3, 4, 5],
            [999, 101, 6, 7, 8, 9, 10, 11],
        ],
        dtype=torch.int64,
    )
    cand_cpu = torch.zeros((2, 7, 4), dtype=torch.int64)
    cand_cpu[0, 1] = torch.tensor([99, 20, 10, 3])  # rank 1 is legal (20)
    cand_cpu[1, 1] = torch.tensor([6, 7, 8, 9])  # none legal -> stops advancing g1 cleanly

    draft_modified = refine_draft_block_with_grammar(
        reqs=reqs,
        draft_cpu=draft_cpu,
        cand_cpu=cand_cpu,
        vocab_size=vocab_size,
    )
    assert draft_modified is True
    assert draft_cpu[0].tolist() == [999, 42, 20, 77, 88, 88, 88, 88]
    assert draft_cpu[1, 1].item() == 101

    # Rollback invariant: cursors & accepted_tokens must be back at committed prefix!
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


def test_sampled_and_mixed_batches_leave_proposals_untouched() -> None:
    # First row would force token 13 into a candidate list which already has it.
    # Second row starts with a legal draw, then has a draw that would change q.
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
        raise AssertionError("sampled fallback must not wait on the grammar barrier")

    for greedy_first in (False, True):
        batch = SimpleNamespace(
            has_grammar=True,
            reqs=[_MockReq(grammar0), _MockReq(grammar1)],
            sampling_info=SimpleNamespace(
                is_all_greedy=False,
                is_any_greedy=greedy_first,
                temperatures=torch.tensor([[0.0 if greedy_first else 0.3], [0.3]]),
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
    worker = SimpleNamespace(
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
        return around_grammar_tree_from_linear_chain(build_tree, object, draft)

    assert around_dflash_forward_batch_generation(
        forward, worker, batch, grammar_barrier=lambda: barriers.append(True)
    ) is draft


if __name__ == "__main__":
    test_bitmask_helpers_and_density()
    test_refine_draft_block_rollback_and_jump_forward()
    test_terminated_grammar_is_not_rolled_back()
    test_sampled_and_mixed_batches_leave_proposals_untouched()
    test_greedy_hook_refines_draft_after_barrier()
    print("CPU_ROLLBACK_AND_DENSITY_TESTS_PASS")
