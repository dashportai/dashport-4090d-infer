"""CPU equivalence checks for the packed-mask dispatch threshold."""

import pytest
import torch

from sglang_xgrammar_lmhead_sm89 import active_row_head


@pytest.mark.parametrize("fallback", [False, True])
def test_dispatch_counts_signed_words_and_preserves_partial_word_bits(
    monkeypatch, fallback: bool
) -> None:
    if fallback:
        monkeypatch.setattr(active_row_head, "_BITWISE_COUNT", None)

    # Vocabulary uses the first 3 words. Keep the high bits of the partial
    # third word in the count, as the original dispatch did, but skip word four.
    mask = torch.tensor([-(1 << 31), 0, -1, -1], dtype=torch.int32)
    assert active_row_head.is_active_pair_count_within_limit(mask, 65, 33)
    assert not active_row_head.is_active_pair_count_within_limit(mask, 65, 32)


@pytest.mark.parametrize("dense_row", [0, 15])
def test_dispatch_counts_dense_rows_before_and_after_the_prefix(dense_row: int) -> None:
    mask = torch.zeros((16, 8), dtype=torch.int32)
    mask[:, 0] = 1
    mask[dense_row].fill_(-1)

    assert active_row_head.is_active_pair_count_within_limit(mask, 256, 271)
    assert not active_row_head.is_active_pair_count_within_limit(mask, 256, 270)


def test_dispatch_matches_original_on_mixed_batch() -> None:
    mask = torch.randint(-(1 << 31), 1 << 31, (64, 7760), dtype=torch.int32)
    mask[:60].zero_()
    _, total_pairs, _ = active_row_head.union_active_row_fraction(mask, 248320)

    assert active_row_head.is_active_pair_count_within_limit(mask, 248320, total_pairs)
    assert not active_row_head.is_active_pair_count_within_limit(mask, 248320, total_pairs - 1)
