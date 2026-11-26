from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="sglang-xgrammar-lmhead-sm89",
    version="0.1.0",
    description="Ada Lovelace (sm_89) CUDA-graph-safe xgrammar active-row lm_head & DFlash2 coupled grammar overlay",
    packages=find_packages(),
    entry_points={
        "sglang.srt.plugins": [
            "xgrammar_lmhead_sm89 = sglang_xgrammar_lmhead_sm89.plugin:register",
        ],
    },
    ext_modules=[
        CUDAExtension(
            name="sglang_xgrammar_lmhead_sm89._C",
            sources=[
                "csrc/torch_bindings.cpp",
                "csrc/xgrammar_coupled_cpu.cpp",
                "csrc/active_row_lm_head_sm89.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20"],
                "nvcc": ["-O3", "-std=c++20", "-gencode=arch=compute_89,code=sm_89",
                         "--ptxas-options=-v", "-Xcudafe", "--diag_suppress=3189"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
