#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Optional.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <cstdint>
#include <tuple>

namespace sglang_xgrammar_lmhead_sm89 {

void launch_active_row_bf16_lm_head(
    const void* hidden,
    const void* weight,
    const int32_t* vocab_mask,
    void* out,
    int M,
    int N,
    int K,
    int mask_words,
    cudaStream_t stream);

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
    cudaStream_t stream);

bool active_pairs_within_budget_cpu(
    const at::Tensor& vocab_mask_cpu,
    int64_t vocab_size,
    int64_t max_pairs);

std::tuple<int64_t, bool, bool> xgrammar_coupled_step_cpu(
    at::Tensor& mask_row_cpu, at::Tensor& draft_row_cpu,
    const c10::optional<at::Tensor>& cand_row_cpu, const c10::optional<at::Tensor>& q_row_cpu,
    int64_t pos, int64_t vocab_size, bool is_greedy, bool allow_forced_single,
    double rand_u, bool clear_mask_after = false);

}  // namespace sglang_xgrammar_lmhead_sm89

namespace {

void check_common(
    const at::Tensor& hidden,
    const at::Tensor& weight,
    const at::Tensor& vocab_mask,
    const at::Tensor& out,
    int k_align) {
    TORCH_CHECK(hidden.is_cuda() && weight.is_cuda() && vocab_mask.is_cuda() && out.is_cuda(),
                "All tensors must be on CUDA");
    TORCH_CHECK(hidden.is_contiguous() && weight.is_contiguous() && vocab_mask.is_contiguous() && out.is_contiguous(),
                "All tensors must be contiguous");
    TORCH_CHECK(hidden.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16,
                "hidden and out must be bfloat16");
    TORCH_CHECK(vocab_mask.scalar_type() == at::kInt, "vocab_mask must be int32");
    TORCH_CHECK(hidden.dim() == 2 && weight.dim() == 2 && vocab_mask.dim() == 2 && out.dim() == 2,
                "All tensors must be 2D");

    const int64_t M = hidden.size(0);
    const int64_t K = hidden.size(1);
    const int64_t N = weight.size(0);
    TORCH_CHECK(weight.size(1) == K, "weight K mismatch");
    TORCH_CHECK(K % k_align == 0, "K must be a multiple of ", k_align);
    TORCH_CHECK(vocab_mask.size(0) == M && vocab_mask.size(1) >= (N + 31) / 32,
                "vocab_mask shape mismatch");
    TORCH_CHECK(out.size(0) == M && out.size(1) == N, "out shape mismatch");
}

void active_row_bf16_lm_head_cuda(
    const at::Tensor& hidden,
    const at::Tensor& weight,
    const at::Tensor& vocab_mask,
    at::Tensor& out) {
    check_common(hidden, weight, vocab_mask, out, 8);
    TORCH_CHECK(weight.scalar_type() == at::kBFloat16, "weight must be bfloat16");

    const at::cuda::OptionalCUDAGuard device_guard(hidden.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    sglang_xgrammar_lmhead_sm89::launch_active_row_bf16_lm_head(
        hidden.data_ptr(),
        weight.data_ptr(),
        vocab_mask.data_ptr<int32_t>(),
        out.data_ptr(),
        static_cast<int>(hidden.size(0)),
        static_cast<int>(weight.size(0)),
        static_cast<int>(hidden.size(1)),
        static_cast<int>(vocab_mask.size(1)),
        stream);
}

void active_row_fp8_lm_head_cuda(
    const at::Tensor& hidden,
    const at::Tensor& weight,
    const at::Tensor& weight_scale,
    const at::Tensor& vocab_mask,
    at::Tensor& out) {
    check_common(hidden, weight, vocab_mask, out, 16);
    TORCH_CHECK(weight.scalar_type() == at::kFloat8_e4m3fn, "weight must be float8_e4m3fn");
    TORCH_CHECK(weight_scale.is_cuda() && weight_scale.is_contiguous() && weight_scale.scalar_type() == at::kFloat,
                "weight_scale must be contiguous CUDA float32");
    TORCH_CHECK(weight_scale.numel() == weight.size(0), "weight_scale length must equal N");

    const at::cuda::OptionalCUDAGuard device_guard(hidden.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    sglang_xgrammar_lmhead_sm89::launch_active_row_fp8_lm_head(
        hidden.data_ptr(),
        weight.data_ptr(),
        weight_scale.data_ptr<float>(),
        vocab_mask.data_ptr<int32_t>(),
        out.data_ptr(),
        static_cast<int>(hidden.size(0)),
        static_cast<int>(weight.size(0)),
        static_cast<int>(hidden.size(1)),
        static_cast<int>(vocab_mask.size(1)),
        stream);
}

}  // namespace

TORCH_LIBRARY(sglang_xgrammar_lmhead_sm89, m) {
    m.def("active_row_bf16_lm_head(Tensor hidden, Tensor weight, Tensor vocab_mask, Tensor(a!) out) -> ()");
    m.def("active_row_fp8_lm_head(Tensor hidden, Tensor weight, Tensor weight_scale, Tensor vocab_mask, Tensor(a!) out) -> ()");
}

TORCH_LIBRARY_IMPL(sglang_xgrammar_lmhead_sm89, CUDA, m) {
    m.impl("active_row_bf16_lm_head", &active_row_bf16_lm_head_cuda);
    m.impl("active_row_fp8_lm_head", &active_row_fp8_lm_head_cuda);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "sm_89 xgrammar active-row lm_head CUDA and C++ coupled grammar kernels";
    m.def(
        "active_pairs_within_budget_cpu",
        &sglang_xgrammar_lmhead_sm89::active_pairs_within_budget_cpu,
        py::arg("vocab_mask_cpu"),
        py::arg("vocab_size"),
        py::arg("max_pairs"));
    m.def(
        "xgrammar_coupled_step_cpu",
        &sglang_xgrammar_lmhead_sm89::xgrammar_coupled_step_cpu,
        py::arg("mask_row_cpu"), py::arg("draft_row_cpu"),
        py::arg("cand_row_cpu"), py::arg("q_row_cpu"),
        py::arg("pos"), py::arg("vocab_size"), py::arg("is_greedy"),
        py::arg("allow_forced_single"), py::arg("rand_u"), py::arg("clear_mask_after") = false);
}
