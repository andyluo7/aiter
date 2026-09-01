# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Small-batch BF16 latent-MoE local-tail primitive."""

import functools
import math

import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.utils import is_flydsl_available

_LATENT_DIM = 3584
_HIDDEN_DIM = 7168
_MAX_TOKENS = 14
_ROWS_PER_BLOCK = 14
_WAVES_PER_EU = 4
# Policy 2 bypasses cache levels that would otherwise retain the one-use
# 49 MiB projection matrix. It is the same FlyDSL cache modifier used by
# existing streamed mixed-MoE weight loads.
_B1_WEIGHT_CACHE_MODIFIER = 2
_MULTI_TOKEN_WEIGHT_CACHE_MODIFIER = 0


def _is_gfx950_flydsl_available() -> bool:
    if not is_flydsl_available():
        return False
    try:
        return get_gfx_runtime() == "gfx950"
    except (AssertionError, KeyError, RuntimeError):
        return False


def supports_latent_moe_tail(
    routed: torch.Tensor,
    shared: torch.Tensor,
    rms_weight: torch.Tensor,
    up_weight: torch.Tensor,
    epsilon: float,
) -> bool:
    """Return whether the gfx950 BF16 primitive supports these tensors."""

    tensors = (routed, shared, rms_weight, up_weight)
    num_tokens = routed.shape[0] if routed.dim() == 2 else 0
    return (
        all(tensor.is_cuda for tensor in tensors)
        and len({tensor.device for tensor in tensors}) == 1
        and all(tensor.dtype == torch.bfloat16 for tensor in tensors)
        and all(tensor.is_contiguous() for tensor in tensors)
        and 1 <= num_tokens <= _MAX_TOKENS
        and tuple(routed.shape) == (num_tokens, _LATENT_DIM)
        and tuple(shared.shape) == (num_tokens, _HIDDEN_DIM)
        and tuple(rms_weight.shape) == (_LATENT_DIM,)
        and tuple(up_weight.shape) == (_HIDDEN_DIM, _LATENT_DIM)
        and math.isfinite(epsilon)
        and epsilon > 0.0
        and _is_gfx950_flydsl_available()
    )


@functools.cache
def _compiled_latent_moe_tail(
    num_tokens: int,
    rows_per_block: int,
    waves_per_eu: int,
    normalize_in_kernel: bool,
    elements_per_thread: int,
    use_dot2: bool,
    weight_cache_modifier: int,
):
    from aiter.ops.flydsl.kernels.latent_moe_tail_gfx950 import (
        build_latent_moe_tail_module,
    )

    return build_latent_moe_tail_module(
        num_tokens,
        rows_per_block,
        waves_per_eu,
        normalize_in_kernel,
        elements_per_thread,
        use_dot2,
        weight_cache_modifier,
    )


def _launch_latent_moe_tail(
    routed: torch.Tensor,
    shared: torch.Tensor,
    rms_weight: torch.Tensor,
    up_weight: torch.Tensor,
    epsilon: float,
    *,
    out: torch.Tensor,
    rows_per_block: int,
    waves_per_eu: int,
    normalize_in_kernel: bool = True,
    elements_per_thread: int = 8,
    use_dot2: bool = True,
    weight_cache_modifier: int = 0,
) -> torch.Tensor:
    from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg

    _compiled_latent_moe_tail(
        routed.shape[0],
        rows_per_block,
        waves_per_eu,
        normalize_in_kernel,
        elements_per_thread,
        use_dot2,
        weight_cache_modifier,
    )(
        ptr_arg(routed),
        ptr_arg(shared),
        ptr_arg(rms_weight),
        ptr_arg(up_weight),
        ptr_arg(out),
        float(epsilon),
        stream=torch.cuda.current_stream(routed.device),
    )
    return out


def latent_moe_tail(
    routed: torch.Tensor,
    shared: torch.Tensor,
    rms_weight: torch.Tensor,
    up_weight: torch.Tensor,
    epsilon: float,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fuse BF16 RMSNorm, FP32-accumulated projection, and BF16 shared add."""

    if not supports_latent_moe_tail(routed, shared, rms_weight, up_weight, epsilon):
        raise NotImplementedError(
            "latent_moe_tail requires contiguous gfx950 BF16 tensors with "
            "1-14 tokens and trailing dimensions 3584 and 7168"
        )
    if out is None:
        out = torch.empty_like(shared)
    elif (
        out.device != routed.device
        or out.dtype != torch.bfloat16
        or not out.is_contiguous()
        or tuple(out.shape) != tuple(shared.shape)
    ):
        raise ValueError(
            "out must match the contiguous BF16 shared tensor on the input device"
        )
    num_tokens = routed.shape[0]
    return _launch_latent_moe_tail(
        routed,
        shared,
        rms_weight,
        up_weight,
        epsilon,
        out=out,
        rows_per_block=_ROWS_PER_BLOCK,
        waves_per_eu=_WAVES_PER_EU,
        weight_cache_modifier=(
            _B1_WEIGHT_CACHE_MODIFIER
            if num_tokens == 1
            else _MULTI_TOKEN_WEIGHT_CACHE_MODIFIER
        ),
    )
