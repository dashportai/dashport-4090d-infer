# syntax=docker/dockerfile:1
FROM lmsysorg/sglang@sha256:3ce6da3e5ea0fee2dac6a23b528cb294fedd31ca74a55a4511f709c5e01e8652 AS upstream

FROM upstream AS build
ENV TORCH_CUDA_ARCH_LIST=8.9
WORKDIR /tmp/sglang-xgrammar-lmhead-source
COPY setup.py ./
COPY csrc/torch_bindings.cpp csrc/active_row_lm_head_sm89.cu ./csrc/
COPY sglang_xgrammar_lmhead_sm89/*.py ./sglang_xgrammar_lmhead_sm89/
RUN /opt/sglang/bin/python -m pip wheel --no-index --no-deps --no-build-isolation \
    --wheel-dir /tmp/wheels .

FROM upstream
RUN --mount=type=bind,from=build,source=/tmp/wheels,target=/tmp/wheels \
    /opt/sglang/bin/python -m pip install --no-index --no-deps /tmp/wheels/*.whl
RUN /opt/sglang/bin/python - <<'PY'
from importlib.metadata import entry_points

import torch
from sglang_xgrammar_lmhead_sm89 import _C

(plugin,) = entry_points(group="sglang.srt.plugins", name="xgrammar_lmhead_sm89")
assert callable(plugin.load())
assert hasattr(torch.ops.sglang_xgrammar_lmhead_sm89, "active_row_bf16_lm_head")
assert hasattr(torch.ops.sglang_xgrammar_lmhead_sm89, "active_row_fp8_lm_head")
PY
