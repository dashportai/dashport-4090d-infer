#include <ATen/Tensor.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>

#include <cstdint>
#include <cstring>
#include <tuple>

namespace sglang_xgrammar_lmhead_sm89 {
namespace {

inline int64_t popcount_words_raw(const uint32_t* row_u32, int64_t mask_words) {
    int64_t acc = 0;
    int64_t p = 0;
    const int64_t pairs = mask_words / 2;
    for (; p < pairs; ++p) {
        const uint64_t packed = static_cast<uint64_t>(row_u32[p * 2]) |
                                (static_cast<uint64_t>(row_u32[p * 2 + 1]) << 32);
        acc += __builtin_popcountll(packed);
    }
    if ((mask_words & 1) != 0) {
        acc += __builtin_popcount(row_u32[mask_words - 1]);
    }
    return acc;
}

inline bool mask_has_token(const uint32_t* row_u32, int64_t tok, int64_t vocab_size) {
    if (tok < 0 || tok >= vocab_size) return false;
    return ((row_u32[tok >> 5] >> (tok & 31)) & 1u) != 0u;
}

inline int64_t find_unique_or_first_token(
    const uint32_t* row_u32,
    int64_t vocab_size,
    bool require_unique,
    int64_t& out_first) {
    out_first = -1;
    const int64_t total_words = (vocab_size + 31) / 32;
    const int64_t pairs = total_words / 2;
    int64_t active_seen = 0;

    for (int64_t p = 0; p < pairs; ++p) {
        uint64_t val = static_cast<uint64_t>(row_u32[p * 2]) |
                       (static_cast<uint64_t>(row_u32[p * 2 + 1]) << 32);
        if ((p * 2 + 2) == total_words) {
            const int tail64 = static_cast<int>(vocab_size & 63);
            if (tail64 != 0) val &= (1ULL << tail64) - 1ULL;
        }
        if (val == 0ULL) continue;
        if (out_first < 0) out_first = (p << 6) + __builtin_ctzll(val);
        if (!require_unique) return -1;
        active_seen += __builtin_popcountll(val);
        if (active_seen > 1) return -1;
    }

    if ((total_words & 1) != 0) {
        uint32_t w = row_u32[total_words - 1];
        const int tail32 = static_cast<int>(vocab_size & 31);
        if (tail32 != 0) w &= (1u << tail32) - 1u;
        if (w != 0u) {
            if (out_first < 0) out_first = ((total_words - 1) << 5) + __builtin_ctz(w);
            if (!require_unique) return -1;
            active_seen += __builtin_popcount(w);
            if (active_seen > 1) return -1;
        }
    }
    return (require_unique && active_seen == 1) ? out_first : -1;
}

inline void write_front_unique(
    int64_t* cand_ptr,
    float* q_ptr,
    int64_t width,
    int64_t token_id) {
    if (cand_ptr != nullptr && width > 0) {
        int64_t slot = -1;
        for (int64_t k = 0; k < width; ++k) {
            if (cand_ptr[k] == token_id) {
                slot = k;
                break;
            }
        }
        if (slot > 0) {
            cand_ptr[slot] = cand_ptr[0];
            cand_ptr[0] = token_id;
        } else if (slot < 0) {
            cand_ptr[0] = token_id;
        }
    }
    if (q_ptr != nullptr && width > 0) {
        for (int64_t k = 0; k < width; ++k) q_ptr[k] = 0.0f;
        q_ptr[0] = 1.0f;
    }
}

}  // namespace

bool active_pairs_within_budget_cpu(
    const at::Tensor& vocab_mask_cpu,
    int64_t vocab_size,
    int64_t max_pairs) {
    TORCH_CHECK(vocab_mask_cpu.device().is_cpu(), "vocab_mask_cpu must be on CPU");
    TORCH_CHECK(vocab_mask_cpu.scalar_type() == at::kInt, "vocab_mask_cpu must be int32");
    TORCH_CHECK(vocab_size > 0, "vocab_size must be positive");

    const at::Tensor contiguous_mask = vocab_mask_cpu.contiguous();
    const int64_t mask_words = (vocab_size + 31) / 32;
    const int64_t stride_words =
        (contiguous_mask.dim() == 1) ? contiguous_mask.size(0) : contiguous_mask.size(1);
    TORCH_CHECK(stride_words >= mask_words, "vocab_mask_cpu smaller than ceil_div(vocab_size, 32)");

    const int64_t rows = (contiguous_mask.dim() == 1) ? 1 : contiguous_mask.size(0);
    if (rows == 0) return max_pairs >= 0;

    const uint32_t* ptr = reinterpret_cast<const uint32_t*>(contiguous_mask.data_ptr<int32_t>());
    int64_t remaining = max_pairs;
    for (int64_t r = 0; r < rows; ++r) {
        remaining -= popcount_words_raw(ptr + r * stride_words, mask_words);
        if (remaining < 0) return false;
    }
    return true;
}

std::tuple<int64_t, bool, bool> xgrammar_coupled_step_cpu(
    at::Tensor& mask_row_cpu, at::Tensor& draft_row_cpu,
    const c10::optional<at::Tensor>& cand_row_cpu, const c10::optional<at::Tensor>& q_row_cpu,
    int64_t pos, int64_t vocab_size, bool is_greedy, bool allow_forced_single,
    double rand_u, bool clear_mask_after) {
    TORCH_CHECK(mask_row_cpu.is_cpu() && mask_row_cpu.scalar_type() == at::kInt && mask_row_cpu.is_contiguous(),
                "mask_row_cpu invalid");
    TORCH_CHECK(draft_row_cpu.is_cpu() && draft_row_cpu.scalar_type() == at::kLong &&
                draft_row_cpu.is_contiguous() && draft_row_cpu.dim() == 1,
                "draft_row_cpu invalid");
    TORCH_CHECK(vocab_size > 0 && pos >= 0 && pos < draft_row_cpu.size(0), "Invalid vocab_size or pos");

    const int64_t req_words = (vocab_size + 31) / 32;
    const int64_t avail_words = (mask_row_cpu.dim() == 1) ? mask_row_cpu.size(0) : mask_row_cpu.size(1);
    TORCH_CHECK(avail_words >= req_words, "mask_row_cpu width smaller than vocab words");

    int32_t* raw_mask = mask_row_cpu.data_ptr<int32_t>();
    const uint32_t* words_u32 = reinterpret_cast<const uint32_t*>(raw_mask);
    int64_t* draft_ptr = draft_row_cpu.data_ptr<int64_t>();
    const int64_t current_tok = draft_ptr[pos];

    int64_t* cand_ptr = nullptr;
    int64_t k_width = 0;
    if (cand_row_cpu.has_value()) {
        const at::Tensor& ct = *cand_row_cpu;
        TORCH_CHECK(ct.device().is_cpu() && ct.scalar_type() == at::kLong && ct.is_contiguous() && ct.dim() == 1,
                    "cand_row_cpu must be 1D contiguous CPU int64");
        cand_ptr = ct.data_ptr<int64_t>();
        k_width = ct.size(0);
    }

    float* q_ptr = nullptr;
    if (q_row_cpu.has_value()) {
        const at::Tensor& qt = *q_row_cpu;
        TORCH_CHECK(qt.device().is_cpu() && qt.scalar_type() == at::kFloat && qt.is_contiguous() && qt.dim() == 1,
                    "q_row_cpu must be 1D contiguous CPU float32");
        if (cand_ptr != nullptr) TORCH_CHECK(qt.size(0) == k_width, "q_row_cpu length mismatch");
        q_ptr = qt.data_ptr<float>();
    }

    auto done = [&](int64_t chosen, bool draft_mod, bool q_mod) {
        if (clear_mask_after) {
            std::memset(raw_mask, 0xFF, static_cast<size_t>(req_words) * sizeof(int32_t));
        }
        return std::make_tuple(chosen, draft_mod, q_mod);
    };

    int64_t first_tok = -1;
    if (allow_forced_single) {
        const int64_t sole_tok = find_unique_or_first_token(words_u32, vocab_size, true, first_tok);
        if (first_tok < 0) return done(-1, false, false);
        if (sole_tok >= 0) {
            bool draft_mod = false;
            if (sole_tok != current_tok) {
                draft_ptr[pos] = sole_tok;
                draft_mod = true;
            }
            const bool q_mod = (q_ptr != nullptr);
            if (cand_ptr != nullptr || q_ptr != nullptr) {
                write_front_unique(cand_ptr, q_ptr, k_width, sole_tok);
            }
            return done(sole_tok, draft_mod, q_mod);
        }
    }

    if (is_greedy || cand_ptr == nullptr || q_ptr == nullptr) {
        int64_t chosen = -1;
        bool draft_mod = false;
        if (mask_has_token(words_u32, current_tok, vocab_size)) {
            chosen = current_tok;
        } else {
            if (cand_ptr != nullptr) {
                for (int64_t k = 0; k < k_width; ++k) {
                    if (mask_has_token(words_u32, cand_ptr[k], vocab_size)) {
                        chosen = cand_ptr[k];
                        break;
                    }
                }
            }
            if (chosen < 0) {
                if (first_tok < 0) find_unique_or_first_token(words_u32, vocab_size, false, first_tok);
                chosen = first_tok;
            }
            if (chosen < 0) return done(-1, false, false);
            draft_ptr[pos] = chosen;
            draft_mod = true;
        }
        const bool q_mod = (q_ptr != nullptr);
        if (cand_ptr != nullptr || q_ptr != nullptr) {
            write_front_unique(cand_ptr, q_ptr, k_width, chosen);
        }
        return done(chosen, draft_mod, q_mod);
    }

    double legal_mass = 0.0;
    int64_t cur_slot = -1;
    int64_t fallback_slot = -1;
    const bool cur_allowed = mask_has_token(words_u32, current_tok, vocab_size);

    for (int64_t k = 0; k < k_width; ++k) {
        const int64_t tok = cand_ptr[k];
        if (tok == current_tok && cur_slot < 0) cur_slot = k;
        const float prob = q_ptr[k];
        if (prob > 0.0f && mask_has_token(words_u32, tok, vocab_size)) {
            legal_mass += static_cast<double>(prob);
            fallback_slot = k;
        } else {
            q_ptr[k] = 0.0f;
        }
    }

    if (legal_mass > 1e-12) {
        const double inv = 1.0 / legal_mass;
        for (int64_t k = 0; k < k_width; ++k) {
            if (q_ptr[k] > 0.0f) {
                q_ptr[k] = static_cast<float>(static_cast<double>(q_ptr[k]) * inv);
            }
        }
        if (cur_allowed && cur_slot >= 0 && q_ptr[cur_slot] > 0.0f) {
            return done(current_tok, false, true);
        }
        double accum = 0.0;
        int64_t pick_idx = fallback_slot;
        for (int64_t k = 0; k < k_width; ++k) {
            const double p = static_cast<double>(q_ptr[k]);
            if (p <= 0.0) continue;
            accum += p;
            if (rand_u < accum) {
                pick_idx = k;
                break;
            }
        }
        const int64_t chosen = cand_ptr[pick_idx];
        draft_ptr[pos] = chosen;
        return done(chosen, true, true);
    }

    if (first_tok < 0) find_unique_or_first_token(words_u32, vocab_size, false, first_tok);
    if (first_tok < 0) return done(-1, false, false);
    write_front_unique(cand_ptr, q_ptr, k_width, first_tok);
    draft_ptr[pos] = first_tok;
    return done(first_tok, true, true);
}

}  // namespace sglang_xgrammar_lmhead_sm89
