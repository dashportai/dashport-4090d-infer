#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace sglang_xgrammar_lmhead_sm89 {

namespace {

constexpr int kTileN = 128;
constexpr int kBlockThreads = 128;
constexpr int kMaxSmemM = 64;
constexpr int kMChunk = 8;

__device__ __forceinline__ __nv_bfloat16 bf16_neg_inf() {
    return __ushort_as_bfloat16(static_cast<uint16_t>(0xFF80u));
}

__device__ __forceinline__ void cvt_e4m3x4_to_float4(
    uint32_t packed, float& f0, float& f1, float& f2, float& f3) {
#if __CUDA_ARCH__ >= 890
    uint32_t h01, h23;
    uint16_t p01 = static_cast<uint16_t>(packed & 0xFFFFu);
    uint16_t p23 = static_cast<uint16_t>(packed >> 16);
    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h01) : "h"(p01));
    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h23) : "h"(p23));
    float2 v01 = __half22float2(*reinterpret_cast<const __half2*>(&h01));
    float2 v23 = __half22float2(*reinterpret_cast<const __half2*>(&h23));
    f0 = v01.x;
    f1 = v01.y;
    f2 = v23.x;
    f3 = v23.y;
#else
    auto decode = [](uint8_t b) -> float {
        int s = (b >> 7) & 1;
        int e = (b >> 3) & 0xF;
        int m = b & 0x7;
        float val;
        if (e == 0) {
            val = ldexpf(static_cast<float>(m) / 8.0f, -6);
        } else if (e == 15 && m == 7) {
            val = 0.0f;
        } else {
            val = ldexpf(1.0f + static_cast<float>(m) / 8.0f, e - 7);
        }
        return s ? -val : val;
    };
    f0 = decode(static_cast<uint8_t>(packed & 0xFFu));
    f1 = decode(static_cast<uint8_t>((packed >> 8) & 0xFFu));
    f2 = decode(static_cast<uint8_t>((packed >> 16) & 0xFFu));
    f3 = decode(static_cast<uint8_t>((packed >> 24) & 0xFFu));
#endif
}

__device__ __forceinline__ void load_mask_tile(
    const int32_t* __restrict__ vocab_mask,
    int m,
    int mask_words,
    int w_base,
    uint32_t tail0,
    uint32_t tail1,
    uint32_t tail2,
    uint32_t tail3,
    uint32_t& w0,
    uint32_t& w1,
    uint32_t& w2,
    uint32_t& w3) {
    const int32_t* row_ptr = vocab_mask + static_cast<size_t>(m) * mask_words + w_base;
    if ((w_base + 4 <= mask_words) &&
        ((reinterpret_cast<uintptr_t>(row_ptr) & 15u) == 0u)) {
        uint4 v = __ldg(reinterpret_cast<const uint4*>(row_ptr));
        w0 = v.x & tail0;
        w1 = v.y & tail1;
        w2 = v.z & tail2;
        w3 = v.w & tail3;
    } else {
        w0 = (w_base + 0 < mask_words) ? (static_cast<uint32_t>(__ldg(row_ptr + 0)) & tail0) : 0u;
        w1 = (w_base + 1 < mask_words) ? (static_cast<uint32_t>(__ldg(row_ptr + 1)) & tail1) : 0u;
        w2 = (w_base + 2 < mask_words) ? (static_cast<uint32_t>(__ldg(row_ptr + 2)) & tail2) : 0u;
        w3 = (w_base + 3 < mask_words) ? (static_cast<uint32_t>(__ldg(row_ptr + 3)) & tail3) : 0u;
    }
}

template <bool IS_FP8>
__global__ void __launch_bounds__(kBlockThreads)
active_row_lm_head_sm89_kernel(
    const __nv_bfloat16* __restrict__ hidden,
    const void* __restrict__ weight,
    const float* __restrict__ weight_scale,  // [N] for FP8, nullptr for BF16
    const int32_t* __restrict__ vocab_mask,  // [M, mask_words]
    __nv_bfloat16* __restrict__ out,         // [M, N]
    int M,
    int N,
    int K,
    int mask_words) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int cta_id = blockIdx.x;
    const int row_base = cta_id * kTileN;
    if (row_base >= N) return;

    const int w_base = cta_id * 4;
    auto tail_for_word = [&](int word_idx) -> uint32_t {
        int rem = N - (row_base + word_idx * 32);
        if (rem >= 32) return 0xFFFFFFFFu;
        if (rem <= 0) return 0u;
        return (1u << rem) - 1u;
    };
    const uint32_t tail0 = tail_for_word(0);
    const uint32_t tail1 = tail_for_word(1);
    const uint32_t tail2 = tail_for_word(2);
    const uint32_t tail3 = tail_for_word(3);

    __shared__ uint32_t s_mask[kMaxSmemM][4];
    __shared__ uint32_t s_union[4];
    __shared__ int s_warp_counts[4];
    __shared__ int s_warp_offsets[4];
    __shared__ int s_total_active;
    __shared__ uint8_t s_active_rows[kTileN];
    __shared__ uint8_t s_row_m_lanes[4][32];

    uint32_t local_u0 = 0u, local_u1 = 0u, local_u2 = 0u, local_u3 = 0u;
    for (int m = tid; m < M; m += kBlockThreads) {
        uint32_t w0, w1, w2, w3;
        load_mask_tile(vocab_mask, m, mask_words, w_base, tail0, tail1, tail2, tail3, w0, w1, w2, w3);
        if (m < kMaxSmemM) {
            s_mask[m][0] = w0;
            s_mask[m][1] = w1;
            s_mask[m][2] = w2;
            s_mask[m][3] = w3;
        }
        local_u0 |= w0;
        local_u1 |= w1;
        local_u2 |= w2;
        local_u3 |= w3;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_u0 |= __shfl_xor_sync(0xFFFFFFFFu, local_u0, offset);
        local_u1 |= __shfl_xor_sync(0xFFFFFFFFu, local_u1, offset);
        local_u2 |= __shfl_xor_sync(0xFFFFFFFFu, local_u2, offset);
        local_u3 |= __shfl_xor_sync(0xFFFFFFFFu, local_u3, offset);
    }

    if (tid < 4) s_union[tid] = 0u;
    __syncthreads();

    if (lane == 0) {
        atomicOr(&s_union[0], local_u0);
        atomicOr(&s_union[1], local_u1);
        atomicOr(&s_union[2], local_u2);
        atomicOr(&s_union[3], local_u3);
    }
    __syncthreads();

    const uint32_t u0 = s_union[0];
    const uint32_t u1 = s_union[1];
    const uint32_t u2 = s_union[2];
    const uint32_t u3 = s_union[3];

    const __nv_bfloat16 neg_inf = bf16_neg_inf();
    const int my_row = row_base + tid;

    // Fast exit: zero active rows in this 128-row CTA -> 0 weight DRAM reads!
    if ((u0 | u1 | u2 | u3) == 0u) {
        if (my_row < N) {
            for (int m = 0; m < M; ++m) {
                out[static_cast<size_t>(m) * N + my_row] = neg_inf;
            }
        }
        return;
    }

    // Write -inf for permanently inactive rows in this CTA and compact active ones.
    const uint32_t my_word = s_union[warp_id];
    const bool is_active = (((my_word >> lane) & 1u) != 0u) && (my_row < N);
    if (!is_active && my_row < N) {
        for (int m = 0; m < M; ++m) {
            out[static_cast<size_t>(m) * N + my_row] = neg_inf;
        }
    }

    const unsigned ballot = __ballot_sync(0xFFFFFFFFu, is_active);
    const int prefix = __popc(ballot & ((lane == 0) ? 0u : ((1u << lane) - 1u)));
    if (lane == 0) {
        s_warp_counts[warp_id] = __popc(ballot);
    }
    __syncthreads();

    if (tid == 0) {
        int off = 0;
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            s_warp_offsets[w] = off;
            off += s_warp_counts[w];
        }
        s_total_active = off;
    }
    __syncthreads();

    if (is_active) {
        s_active_rows[s_warp_offsets[warp_id] + prefix] = static_cast<uint8_t>(tid);
    }
    __syncthreads();

    const int n_active = s_total_active;
    if (n_active == 0) return;

    // Per-row active-m compaction: compute ONLY for tokens m whose bitmask permits row!
    for (int idx = warp_id; idx < n_active; idx += 4) {
        const int r_in_cta = static_cast<int>(s_active_rows[idx]);
        const int row = row_base + r_in_cta;
        const int w_word = r_in_cta >> 5;
        const uint32_t r_bit = 1u << (r_in_cta & 31);
        const float row_scale = IS_FP8 ? __ldg(weight_scale + row) : 1.0f;

        // Compact active tokens for this row across blocks of 32 tokens (fast for M <= 64).
        for (int m_blk = 0; m_blk < M; m_blk += 32) {
            const int m = m_blk + lane;
            bool allow_m = false;
            if (m < M) {
                uint32_t mw = (m < kMaxSmemM)
                    ? s_mask[m][w_word]
                    : static_cast<uint32_t>(
                          __ldg(vocab_mask + static_cast<size_t>(m) * mask_words + w_base + w_word));
                allow_m = (mw & r_bit) != 0u;
                if (!allow_m) {
                    out[static_cast<size_t>(m) * N + row] = neg_inf;
                }
            }
            const unsigned mb = __ballot_sync(0xFFFFFFFFu, allow_m);
            const int n_m = __popc(mb);
            if (n_m == 0) continue;

            const int m_pos = __popc(mb & ((lane == 0) ? 0u : ((1u << lane) - 1u)));
            if (allow_m) {
                // Compact within this 32-row block so global indices never narrow.
                s_row_m_lanes[warp_id][m_pos] = static_cast<uint8_t>(lane);
            }
            __syncwarp();

            for (int m0 = 0; m0 < n_m; m0 += kMChunk) {
                const int m_cnt = (m0 + kMChunk <= n_m) ? kMChunk : (n_m - m0);
                int m_idx[kMChunk];
                #pragma unroll
                for (int j = 0; j < kMChunk; ++j) {
                    m_idx[j] = (j < m_cnt) ? m_blk + s_row_m_lanes[warp_id][m0 + j] : 0;
                }

                float acc[kMChunk];
                #pragma unroll
                for (int j = 0; j < kMChunk; ++j) acc[j] = 0.0f;

                if constexpr (!IS_FP8) {
                    const __nv_bfloat16* w_row =
                        reinterpret_cast<const __nv_bfloat16*>(weight) + static_cast<size_t>(row) * K;
                    for (int k = lane * 8; k < K; k += 256) {
                        const uint4 wv = __ldg(reinterpret_cast<const uint4*>(w_row + k));
                        const __nv_bfloat162* wb = reinterpret_cast<const __nv_bfloat162*>(&wv);
                        const float2 wf0 = __bfloat1622float2(wb[0]);
                        const float2 wf1 = __bfloat1622float2(wb[1]);
                        const float2 wf2 = __bfloat1622float2(wb[2]);
                        const float2 wf3 = __bfloat1622float2(wb[3]);

                        for (int j = 0; j < m_cnt; ++j) {
                            const uint4 xv = __ldg(reinterpret_cast<const uint4*>(
                                hidden + static_cast<size_t>(m_idx[j]) * K + k));
                            const __nv_bfloat162* xb = reinterpret_cast<const __nv_bfloat162*>(&xv);
                            const float2 xf0 = __bfloat1622float2(xb[0]);
                            const float2 xf1 = __bfloat1622float2(xb[1]);
                            const float2 xf2 = __bfloat1622float2(xb[2]);
                            const float2 xf3 = __bfloat1622float2(xb[3]);

                            float d = wf0.x * xf0.x + wf0.y * xf0.y;
                            d = fmaf(wf1.x, xf1.x, fmaf(wf1.y, xf1.y, d));
                            d = fmaf(wf2.x, xf2.x, fmaf(wf2.y, xf2.y, d));
                            d = fmaf(wf3.x, xf3.x, fmaf(wf3.y, xf3.y, d));
                            acc[j] += d;
                        }
                    }
                } else {
                    const uint8_t* w_row =
                        reinterpret_cast<const uint8_t*>(weight) + static_cast<size_t>(row) * K;
                    for (int k = lane * 16; k < K; k += 512) {
                        const uint4 wv = __ldg(reinterpret_cast<const uint4*>(w_row + k));
                        const uint32_t ws[4] = {wv.x, wv.y, wv.z, wv.w};
                        float wf[16];
                        #pragma unroll
                        for (int t = 0; t < 4; ++t) {
                            cvt_e4m3x4_to_float4(ws[t], wf[t * 4 + 0], wf[t * 4 + 1], wf[t * 4 + 2], wf[t * 4 + 3]);
                        }

                        for (int j = 0; j < m_cnt; ++j) {
                            const __nv_bfloat16* xm = hidden + static_cast<size_t>(m_idx[j]) * K + k;
                            const uint4 xv0 = __ldg(reinterpret_cast<const uint4*>(xm));
                            const uint4 xv1 = __ldg(reinterpret_cast<const uint4*>(xm + 8));
                            const __nv_bfloat162* xb0 = reinterpret_cast<const __nv_bfloat162*>(&xv0);
                            const __nv_bfloat162* xb1 = reinterpret_cast<const __nv_bfloat162*>(&xv1);

                            float d = 0.0f;
                            #pragma unroll
                            for (int t = 0; t < 4; ++t) {
                                float2 xf0 = __bfloat1622float2(xb0[t]);
                                float2 xf1 = __bfloat1622float2(xb1[t]);
                                d = fmaf(wf[t * 2 + 0], xf0.x, d);
                                d = fmaf(wf[t * 2 + 1], xf0.y, d);
                                d = fmaf(wf[8 + t * 2 + 0], xf1.x, d);
                                d = fmaf(wf[8 + t * 2 + 1], xf1.y, d);
                            }
                            acc[j] += d;
                        }
                    }
                }

                for (int j = 0; j < m_cnt; ++j) {
                    float sum = acc[j];
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1) {
                        sum += __shfl_xor_sync(0xFFFFFFFFu, sum, offset);
                    }
                    if (lane == 0) {
                        out[static_cast<size_t>(m_idx[j]) * N + row] =
                            __float2bfloat16(sum * row_scale);
                    }
                }
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
    dim3 grid((N + kTileN - 1) / kTileN, 1, 1);
    dim3 block(kBlockThreads, 1, 1);
    active_row_lm_head_sm89_kernel<false><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(hidden),
        weight,
        nullptr,
        vocab_mask,
        reinterpret_cast<__nv_bfloat16*>(out),
        M,
        N,
        K,
        mask_words);
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
    dim3 grid((N + kTileN - 1) / kTileN, 1, 1);
    dim3 block(kBlockThreads, 1, 1);
    active_row_lm_head_sm89_kernel<true><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(hidden),
        weight,
        weight_scale,
        vocab_mask,
        reinterpret_cast<__nv_bfloat16*>(out),
        M,
        N,
        K,
        mask_words);
}

}  // namespace sglang_xgrammar_lmhead_sm89
