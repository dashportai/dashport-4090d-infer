"""SGLang plugin entry point for sglang-xgrammar-lmhead-sm89."""

from __future__ import annotations

import logging
import os

from .active_row_head import ensure_extension_loaded
from .in_draft_xgrammar import (
    around_dflash_forward_batch_generation,
    around_grammar_tree_from_linear_chain,
)
from .target_verify_head import (
    after_tp_worker_forward_batch_generation,
    around_build_grammar_vocab_mask,
    around_target_logits_processor_forward,
    before_apply_dflash_verify_logits_adjustments,
    before_dflash_accept_block,
    clear_target_verify_tls,
)

logger = logging.getLogger(__name__)


def _env_enabled(name: str, *, default: str) -> bool:
    return os.getenv(name, default).strip().lower() not in ("0", "false", "no", "off", "")


def _around_dflash_step_with_cleanup(original_fn, worker, *args, **kwargs):
    try:
        return around_dflash_forward_batch_generation(original_fn, worker, *args, **kwargs)
    finally:
        clear_target_verify_tls()


def register() -> None:
    """Register SGLang HookRegistry hooks for xgrammar_lmhead_sm89."""
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    if not _env_enabled("SGLANG_XGRAMMAR_LMHEAD_SM89", default="1"):
        logger.info("[xgrammar_lmhead_sm89] Disabled via SGLANG_XGRAMMAR_LMHEAD_SM89=0")
        return

    ext_ok = ensure_extension_loaded()
    in_draft = _env_enabled("SGLANG_DFLASH_XGRAMMAR_DRAFT", default="1")
    active_target = _env_enabled("SGLANG_XGRAMMAR_ACTIVE_ROW_TARGET", default="1")

    if in_draft or active_target:
        HookRegistry.register(
            "sglang.srt.speculative.dflash_worker_v2.DFlashWorkerV2.forward_batch_generation",
            _around_dflash_step_with_cleanup,
            HookType.AROUND,
        )

    if in_draft:
        HookRegistry.register(
            "sglang.srt.speculative.dflash_worker_v2.GrammarTree.from_linear_chain",
            around_grammar_tree_from_linear_chain,
            HookType.AROUND,
        )

    if active_target:
        HookRegistry.register(
            "sglang.srt.layers.logits_processor.LogitsProcessor.forward",
            around_target_logits_processor_forward,
            HookType.AROUND,
        )
        HookRegistry.register(
            "sglang.srt.managers.tp_worker.TpModelWorker.forward_batch_generation",
            after_tp_worker_forward_batch_generation,
            HookType.AFTER,
        )
        HookRegistry.register(
            "sglang.srt.speculative.spec_utils.build_grammar_vocab_mask",
            around_build_grammar_vocab_mask,
            HookType.AROUND,
        )
        HookRegistry.register(
            "sglang.srt.speculative.dflash_worker_v2.apply_dflash_verify_logits_adjustments",
            before_apply_dflash_verify_logits_adjustments,
            HookType.BEFORE,
        )
        HookRegistry.register(
            "sglang.srt.speculative.dflash_worker_v2.DFlashWorkerV2._accept_block",
            before_dflash_accept_block,
            HookType.BEFORE,
        )

    logger.info(
        "[xgrammar_lmhead_sm89] Registered hooks (in_draft=%s active_row_target=%s ext_loaded=%s)",
        in_draft,
        active_target,
        ext_ok,
    )
