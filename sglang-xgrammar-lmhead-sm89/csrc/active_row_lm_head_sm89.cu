#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace sglang_xgrammar_lmhead_sm89 {
namespace {

constexpr int kRowsPerBlock = 64;
constexpr int kBlockSize = 128;
constexpr int kSharedRowCap = 64;
constexpr int kTokenStripe = 8;

__device__ __forceinline__ __nv_bfloat16 neg_inf_bf16() {
    return __ushort_as_bfloat16(static_cast<uint16_t>(0xFF80u));
}

__device__ __forceinline__ float2 unpack_bf16_u32(uint32_t u) {
    __nv_bfloat162 pair;
    __builtin_memcpy(&pair, &u, sizeof(pair));
    return __bfloat1622float2(pair);
}

__device__ __forceinline__ __half2 unpack_fp8x2_to_h2(uint16_t bits) {
#if __CUDA_ARCH__ >= 890
    uint32_t packed;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(packed) : "h"(bits));
    __half2 res;
    __builtin_memcpy(&res, &packed, sizeof(res));
    return res;
#else
    auto to_f = [](uint8_t byte) -> float {
        int s = byte >> 7;
        int e = (byte >> 3) & 0xF;
        int m = byte & 0x7;
        float val = (e == 0)
            ? ldexpf(static_cast<float>(m) / 8.0f, -6)
            : ((e == 15 && m == 7) ? 0.0f : ldexpf(1.0f + static_cast<float>(m) / 8.0f, e - 7));
        return s ? -val : val;
    };
    return __floats2half2_rn(
        to_f(static_cast<uint8_t>(bits & 0xFFu)),
        to_f(static_cast<uint8_t>(bits >> 8)));
#endif
}

__device__ __forceinline__ uint64_t fetch_row64_mask(
    const int32_t* __restrict__ mask_ptr,
    int tok_row,
    int mask_words,
    int word_off,
    uint64_t valid_tail) {
    const int32_t* base = mask_ptr + static_cast<size_t>(tok_row) * mask_words + word_off;
    uint32_t lo = 0u, hi = 0u;
    if ((word_off + 2 <= mask_words) && ((reinterpret_cast<uintptr_t>(base) & 7u) == 0u)) {
        const uint2 vec = __ldg(reinterpret_cast<const uint2*>(base));
        lo = vec.x;
        hi = vec.y;
    } else {
        if (word_off < mask_words) lo = static_cast<uint32_t>(__ldg(base));
        if (word_off + 1 < mask_words) hi = static_cast<uint32_t>(__ldg(base + 1));
    }
    return (static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32)) & valid_tail;
}

template <bool FP8_WEIGHTS>
__device__ __forceinline__ void warp_gemv_stripe(
    const __nv_bfloat16* __restrict__ hidden,
    const void* __restrict__ weight,
    float inv_scale,
    __nv_bfloat16* __restrict__ out,
    const uint8_t* __restrict__ active_tokens,
    int m_origin,
    int m_start,
    int count,
    int vocab_row,
    int N,
    int K,
    int lane) {
    float sum_reg[kTokenStripe];
    sum_reg[0] = 0.0f;
    sum_reg[1] = 0.0f;
    sum_reg[2] = 0.0f;
    sum_reg[3] = 0.0f;
    sum_reg[4] = 0.0f;
    sum_reg[5] = 0.0f;
    sum_reg[6] = 0.0f;
    sum_reg[7] = 0.0f;

    auto accum = [&](int idx, float v) {
        switch (idx) {
            case 0: sum_reg[0] += v; break;
            case 1: sum_reg[1] += v; break;
            case 2: sum_reg[2] += v; break;
            case 3: sum_reg[3] += v; break;
            case 4: sum_reg[4] += v; break;
            case 5: sum_reg[5] += v; break;
            case 6: sum_reg[6] += v; break;
            default: sum_reg[7] += v; break;
        }
    };

    const size_t w_offset = static_cast<size_t>(vocab_row) * K;
    if constexpr (!FP8_WEIGHTS) {
        const __nv_bfloat16* w_ptr = reinterpret_cast<const __nv_bfloat16*>(weight) + w_offset;
        for (int k = lane * 8; k < K; k += 256) {
            const uint4 w4 = __ldg(reinterpret_cast<const uint4*>(w_ptr + k));
            const float2 w0 = unpack_bf16_u32(w4.x);
            const float2 w1 = unpack_bf16_u32(w4.y);
            const float2 w2 = unpack_bf16_u32(w4.z);
            const float2 w3 = unpack_bf16_u32(w4.w);

            #pragma unroll 1
            for (int j = 0; j < count; ++j) {
                const int m_tok = m_origin + static_cast<int>(active_tokens[m_start + j]);
                const uint4 x4 = __ldg(reinterpret_cast<const uint4*>(
                    hidden + static_cast<size_t>(m_tok) * K + k));
                float2 x0 = unpack_bf16_u32(x4.x);
                float partial = w0.x * x0.x + w0.y * x0.y;
                x0 = unpack_bf16_u32(x4.y);
                partial = fmaf(w1.x, x0.x, fmaf(w1.y, x0.y, partial));
                x0 = unpack_bf16_u32(x4.z);
                partial = fmaf(w2.x, x0.x, fmaf(w2.y, x0.y, partial));
                x0 = unpack_bf16_u32(x4.w);
                partial = fmaf(w3.x, x0.x, fmaf(w3.y, x0.y, partial));
                accum(j, partial);
            }
        }
    } else {
        const uint8_t* w_ptr = reinterpret_cast<const uint8_t*>(weight) + w_offset;
        for (int k = lane * 16; k < K; k += 512) {
            const uint4 w4 = __ldg(reinterpret_cast<const uint4*>(w_ptr + k));
            const __half2 h0 = unpack_fp8x2_to_h2(static_cast<uint16_t>(w4.x & 0xFFFFu));
            const __half2 h1 = unpack_fp8x2_to_h2(static_cast<uint16_t>(w4.x >> 16));
            const __half2 h2 = unpack_fp8x2_to_h2(static_cast<uint16_t>(w4.y & 0xFFFFu));
            const __half2 h3 = unpack_fp8x2_to_h2(static_cast<uint16_t>(w4.y >> 16));
            const __half2 h4 = unpack_fp8x2_to_h2(static_cast<uint16_t>(w4.z & 0xFFFFu));
            const __half2 h5 = unpack_fp8x2_to_h2(static_cast<uint16_t>(w4.z >> 16));
            const __half2 h6 = unpack_fp8x2_to_h2(static_cast<uint16_t>(w4.w & 0xFFFFu));
            const __half2 h7 = unpack_fp8x2_to_h2(static_cast<uint16_t>(w4.w >> 16));

            #pragma unroll 1
            for (int j = 0; j < count; ++j) {
                const int m_tok = m_origin + static_cast<int>(active_tokens[m_start + j]);
                const __nv_bfloat16* x_row = hidden + static_cast<size_t>(m_tok) * K + k;
                float partial = 0.0f;

                const uint4 x_a = __ldg(reinterpret_cast<const uint4*>(x_row));
                float2 wf = __half22float2(h0), xf = unpack_bf16_u32(x_a.x);
                partial = fmaf(wf.x, xf.x, fmaf(wf.y, xf.y, partial));
                wf = __half22float2(h1); xf = unpack_bf16_u32(x_a.y);
                partial = fmaf(wf.x, xf.x, fmaf(wf.y, xf.y, partial));
                wf = __half22float2(h2); xf = unpack_bf16_u32(x_a.z);
                partial = fmaf(wf.x, xf.x, fmaf(wf.y, xf.y, partial));
                wf = __half22float2(h3); xf = unpack_bf16_u32(x_a.w);
                partial = fmaf(wf.x, xf.x, fmaf(wf.y, xf.y, partial));

                const uint4 x_b = __ldg(reinterpret_cast<const uint4*>(x_row + 8));
                wf = __half22float2(h4); xf = unpack_bf16_u32(x_b.x);
                partial = fmaf(wf.x, xf.x, fmaf(wf.y, xf.y, partial));
                wf = __half22float2(h5); xf = unpack_bf16_u32(x_b.y);
                partial = fmaf(wf.x, xf.x, fmaf(wf.y, xf.y, partial));
                wf = __half22float2(h6); xf = unpack_bf16_u32(x_b.z);
                partial = fmaf(wf.x, xf.x, fmaf(wf.y, xf.y, partial));
                wf = __half22float2(h7); xf = unpack_bf16_u32(x_b.w);
                partial = fmaf(wf.x, xf.x, fmaf(wf.y, xf.y, partial));

                accum(j, partial);
            }
        }
    }

    #pragma unroll 1
    for (int j = 0; j < count; ++j) {
        float val = sum_reg[0];
        if (j == 1) val = sum_reg[1];
        else if (j == 2) val = sum_reg[2];
        else if (j == 3) val = sum_reg[3];
        else if (j == 4) val = sum_reg[4];
        else if (j == 5) val = sum_reg[5];
        else if (j == 6) val = sum_reg[6];
        else if (j >= 7) val = sum_reg[7];
        #pragma unroll
        for (int s = 16; s > 0; s >>= 1) {
            val += __shfl_xor_sync(0xFFFFFFFFu, val, s);
        }
        if (lane == 0) {
            const int m_tok = m_origin + static_cast<int>(active_tokens[m_start + j]);
            out[static_cast<size_t>(m_tok) * N + vocab_row] = __float2bfloat16(val * inv_scale);
        }
    }
}

template <bool FP8_WEIGHTS>
__global__ void __launch_bounds__(kBlockSize, 4)
active_row_lm_head_tile64_kernel(
    const __nv_bfloat16* __restrict__ hidden, const void* __restrict__ weight,
    const float* __restrict__ weight_scale, const int32_t* __restrict__ vocab_mask,
    __nv_bfloat16* __restrict__ out, int M, int N, int K, int mask_words) {
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int block_row0 = blockIdx.x * kRowsPerBlock;
    if (block_row0 >= N) return;

    const int rem = N - block_row0;
    const int n_tile_rows = (rem < kRowsPerBlock) ? rem : kRowsPerBlock;
    const uint64_t tail_bits = (rem >= kRowsPerBlock)
        ? ~0ULL
        : ((rem <= 0) ? 0ULL : ((1ULL << rem) - 1ULL));
    const int word_idx0 = blockIdx.x * 2;

    __shared__ uint64_t smem_row_masks[kSharedRowCap];
    __shared__ uint64_t smem_union_mask;
    __shared__ uint8_t smem_active_rows[kRowsPerBlock];
    __shared__ int smem_active_count;
    __shared__ uint8_t smem_warp_tokens[4][kSharedRowCap];

    uint64_t thread_or = 0ULL;
    for (int m = tid; m < M; m += kBlockSize) {
        const uint64_t mask64 = fetch_row64_mask(vocab_mask, m, mask_words, word_idx0, tail_bits);
        if (m < kSharedRowCap) {
            smem_row_masks[m] = mask64;
        }
        thread_or |= mask64;
    }

    #pragma unroll
    for (int d = 16; d > 0; d >>= 1) {
        thread_or |= __shfl_xor_sync(0xFFFFFFFFu, thread_or, d);
    }

    if (tid == 0) smem_union_mask = 0ULL;
    __syncthreads();

    if (lane == 0 && thread_or != 0ULL) {
        atomicOr(reinterpret_cast<unsigned long long*>(&smem_union_mask),
                 static_cast<unsigned long long>(thread_or));
    }
    __syncthreads();

    const uint64_t union64 = smem_union_mask;
    const __nv_bfloat16 fill_val = neg_inf_bf16();

    if (union64 == 0ULL) {
        const int total = M * n_tile_rows;
        for (int idx = tid; idx < total; idx += kBlockSize) {
            const int m = (n_tile_rows == kRowsPerBlock) ? (idx >> 6) : (idx / n_tile_rows);
            const int r = (n_tile_rows == kRowsPerBlock) ? (idx & 63) : (idx - m * n_tile_rows);
            out[static_cast<size_t>(m) * N + block_row0 + r] = fill_val;
        }
        return;
    }

    const bool keep_row = (tid < n_tile_rows) && (((union64 >> tid) & 1ULL) != 0ULL);
    if (tid < n_tile_rows && !keep_row) {
        const int global_row = block_row0 + tid;
        for (int m = 0; m < M; ++m) {
            out[static_cast<size_t>(m) * N + global_row] = fill_val;
        }
    }

    if (warp < 2) {
        const uint32_t ball = __ballot_sync(0xFFFFFFFFu, keep_row);
        const int low_pop = __popc(static_cast<uint32_t>(union64 & 0xFFFFFFFFULL));
        const uint32_t pfx_mask = (lane == 0) ? 0u : ((1u << lane) - 1u);
        const int write_at = (warp == 0 ? 0 : low_pop) + __popc(ball & pfx_mask);
        if (keep_row) smem_active_rows[write_at] = static_cast<uint8_t>(tid);
    }
    if (tid == 0) smem_active_count = __popcll(union64);
    __syncthreads();

    const int active_rows = smem_active_count;
    if (active_rows == 0) return;

    const int block_m_step = (M <= kSharedRowCap) ? kSharedRowCap : 32;
    for (int row_i = warp; row_i < active_rows; row_i += 4) {
        const int local_r = static_cast<int>(smem_active_rows[row_i]);
        const int global_r = block_row0 + local_r;
        const uint64_t bit_r = 1ULL << local_r;
        const float row_sc = FP8_WEIGHTS ? __ldg(weight_scale + global_r) : 1.0f;

        for (int m_orig = 0; m_orig < M; m_orig += block_m_step) {
            int n_tok = 0;
            const uint32_t pfx_mask = (lane == 0) ? 0u : ((1u << lane) - 1u);
            if (M <= kSharedRowCap) {
                const int m0 = lane, m1 = 32 + lane;
                const bool ok0 = (m0 < M) && ((smem_row_masks[m0] & bit_r) != 0ULL);
                const bool ok1 = (m1 < M) && ((smem_row_masks[m1] & bit_r) != 0ULL);
                if (m0 < M && !ok0) out[static_cast<size_t>(m0) * N + global_r] = fill_val;
                if (m1 < M && !ok1) out[static_cast<size_t>(m1) * N + global_r] = fill_val;

                const uint32_t b0 = __ballot_sync(0xFFFFFFFFu, ok0);
                const uint32_t b1 = __ballot_sync(0xFFFFFFFFu, ok1);
                n_tok = __popc(b0) + __popc(b1);
                if (n_tok == 0) continue;

                const int c0 = __popc(b0);
                if (ok0) smem_warp_tokens[warp][__popc(b0 & pfx_mask)] = static_cast<uint8_t>(m0);
                if (ok1) smem_warp_tokens[warp][c0 + __popc(b1 & pfx_mask)] = static_cast<uint8_t>(m1);
            } else {
                const int m_tok = m_orig + lane;
                bool ok = false;
                if (m_tok < M) {
                    const uint64_t mw = (m_tok < kSharedRowCap)
                        ? smem_row_masks[m_tok]
                        : fetch_row64_mask(vocab_mask, m_tok, mask_words, word_idx0, tail_bits);
                    ok = (mw & bit_r) != 0ULL;
                    if (!ok) out[static_cast<size_t>(m_tok) * N + global_r] = fill_val;
                }
                const uint32_t b = __ballot_sync(0xFFFFFFFFu, ok);
                n_tok = __popc(b);
                if (n_tok == 0) continue;

                if (ok) smem_warp_tokens[warp][__popc(b & pfx_mask)] = static_cast<uint8_t>(lane);
            }
            __syncwarp();

            for (int off = 0; off < n_tok; off += kTokenStripe) {
                const int cnt = (off + kTokenStripe <= n_tok) ? kTokenStripe : (n_tok - off);
                warp_gemv_stripe<FP8_WEIGHTS>(
                    hidden, weight, row_sc, out,
                    smem_warp_tokens[warp], m_orig, off, cnt, global_r, N, K, lane);
            }
        }
    }
}

}  // namespace

void launch_active_row_bf16_lm_head(
    const void* hidden,
    const void* weight,
    const int32_t* vocab_mask,
    void* out,
    int M,
    int N,
    int K,
    int mask_words,
    cudaStream_t stream) {
    if (M <= 0 || N <= 0) return;
    const dim3 grid((N + kRowsPerBlock - 1) / kRowsPerBlock, 1, 1);
    const dim3 block(kBlockSize, 1, 1);
    active_row_lm_head_tile64_kernel<false><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(hidden),
        weight,
        nullptr,
        vocab_mask,
        reinterpret_cast<__nv_bfloat16*>(out),
        M, N, K, mask_words);
}

void launch_active_row_fp8_lm_head(
    const void* hidden,
    const void* weight,
    const float* weight_scale,
    const int32_t* vocab_mask,
    void* out,
    int M,
    int N,
    int K,
    int mask_words,
    cudaStream_t stream) {
    if (M <= 0 || N <= 0) return;
    const dim3 grid((N + kRowsPerBlock - 1) / kRowsPerBlock, 1, 1);
    const dim3 block(kBlockSize, 1, 1);
    active_row_lm_head_tile64_kernel<true><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(hidden),
        weight,
        weight_scale,
        vocab_mask,
        reinterpret_cast<__nv_bfloat16*>(out),
        M, N, K, mask_words);
}

}  // namespace sglang_xgrammar_lmhead_sm89
