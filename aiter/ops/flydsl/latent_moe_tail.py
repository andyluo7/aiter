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


def supports_latent_moe_projection(
    normalized: torch.Tensor,
    up_weight: torch.Tensor,
) -> bool:
    """Return whether the gfx950 BF16 projection primitive is supported."""

    tensors = (normalized, up_weight)
    num_tokens = normalized.shape[0] if normalized.dim() == 2 else 0
    return (
        all(tensor.is_cuda for tensor in tensors)
        and len({tensor.device for tensor in tensors}) == 1
        and all(tensor.dtype == torch.bfloat16 for tensor in tensors)
        and all(tensor.is_contiguous() for tensor in tensors)
        and 1 <= num_tokens <= _MAX_TOKENS
        and tuple(normalized.shape) == (num_tokens, _LATENT_DIM)
        and tuple(up_weight.shape) == (_HIDDEN_DIM, _LATENT_DIM)
        and _is_gfx950_flydsl_available()
    )


def supports_latent_moe_projection_add(
    normalized: torch.Tensor,
    shared: torch.Tensor,
    up_weight: torch.Tensor,
) -> bool:
    """Return whether the gfx950 BF16 projection-add primitive is supported."""

    return (
        supports_latent_moe_projection(normalized, up_weight)
        and shared.is_cuda
        and shared.device == normalized.device
        and shared.dtype == torch.bfloat16
        and shared.is_contiguous()
        and tuple(shared.shape) == (normalized.shape[0], _HIDDEN_DIM)
    )


@functools.cache
def _compiled_latent_moe_tail(
    num_tokens: int,
    tokens_per_block: int,
    rows_per_block: int,
    waves_per_eu: int,
    normalize_in_kernel: bool,
    add_shared: bool,
    elements_per_thread: int,
    use_dot2: bool,
    weight_cache_modifier: int,
):
    from aiter.ops.flydsl.kernels.latent_moe_tail_gfx950 import (
        build_latent_moe_tail_module,
    )

    return build_latent_moe_tail_module(
        num_tokens,
        tokens_per_block,
        rows_per_block,
        waves_per_eu,
        normalize_in_kernel,
        add_shared,
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
    tokens_per_block: int,
    rows_per_block: int,
    waves_per_eu: int,
    normalize_in_kernel: bool = True,
    add_shared: bool = True,
    elements_per_thread: int = 8,
    use_dot2: bool = True,
    weight_cache_modifier: int = 0,
) -> torch.Tensor:
    from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg

    _compiled_latent_moe_tail(
        routed.shape[0],
        tokens_per_block,
        rows_per_block,
        waves_per_eu,
        normalize_in_kernel,
        add_shared,
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


def _validate_tiling(
    num_tokens: int, tokens_per_block: int, rows_per_block: int
) -> None:
    if not 1 <= tokens_per_block <= num_tokens:
        raise ValueError("tokens_per_block must be between 1 and num_tokens")
    if not 2 <= rows_per_block <= 64:
        raise ValueError("rows_per_block must be between 2 and 64")


def _validate_output(
    out: torch.Tensor,
    shape: tuple[int, ...],
    device: torch.device,
) -> None:
    if (
        out.device != device
        or out.dtype != torch.bfloat16
        or not out.is_contiguous()
        or tuple(out.shape) != shape
    ):
        raise ValueError(
            "out must match the contiguous BF16 output shape on the input device"
        )


def latent_moe_tail(
    routed: torch.Tensor,
    shared: torch.Tensor,
    rms_weight: torch.Tensor,
    up_weight: torch.Tensor,
    epsilon: float,
    *,
    out: torch.Tensor | None = None,
    tokens_per_block: int = 1,
    rows_per_block: int = _ROWS_PER_BLOCK,
) -> torch.Tensor:
    """Fuse BF16 RMSNorm, FP32-accumulated projection, and BF16 shared add."""

    if not supports_latent_moe_tail(routed, shared, rms_weight, up_weight, epsilon):
        raise NotImplementedError(
            "latent_moe_tail requires contiguous gfx950 BF16 tensors with "
            "1-14 tokens and trailing dimensions 3584 and 7168"
        )
    num_tokens = routed.shape[0]
    _validate_tiling(num_tokens, tokens_per_block, rows_per_block)
    if out is None:
        out = torch.empty_like(shared)
    else:
        _validate_output(out, tuple(shared.shape), routed.device)
    return _launch_latent_moe_tail(
        routed,
        shared,
        rms_weight,
        up_weight,
        epsilon,
        out=out,
        tokens_per_block=tokens_per_block,
        rows_per_block=rows_per_block,
        waves_per_eu=_WAVES_PER_EU,
        weight_cache_modifier=(
            _B1_WEIGHT_CACHE_MODIFIER
            if num_tokens == 1
            else _MULTI_TOKEN_WEIGHT_CACHE_MODIFIER
        ),
    )


def latent_moe_projection_add(
    normalized: torch.Tensor,
    shared: torch.Tensor,
    up_weight: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    tokens_per_block: int = 1,
    rows_per_block: int = _ROWS_PER_BLOCK,
) -> torch.Tensor:
    """Project normalized BF16 latents and add the BF16 shared output."""

    if not supports_latent_moe_projection_add(normalized, shared, up_weight):
        raise NotImplementedError(
            "latent_moe_projection_add requires contiguous gfx950 BF16 tensors "
            "with 1-14 tokens and trailing dimensions 3584 and 7168"
        )
    num_tokens = normalized.shape[0]
    _validate_tiling(num_tokens, tokens_per_block, rows_per_block)
    if out is None:
        out = torch.empty_like(shared)
    else:
        _validate_output(out, tuple(shared.shape), normalized.device)
    return _launch_latent_moe_tail(
        normalized,
        shared,
        normalized,
        up_weight,
        1.0,
        out=out,
        tokens_per_block=tokens_per_block,
        rows_per_block=rows_per_block,
        waves_per_eu=_WAVES_PER_EU,
        normalize_in_kernel=False,
        add_shared=True,
        weight_cache_modifier=(
            _B1_WEIGHT_CACHE_MODIFIER
            if num_tokens == 1
            else _MULTI_TOKEN_WEIGHT_CACHE_MODIFIER
        ),
    )


def latent_moe_projection(
    normalized: torch.Tensor,
    up_weight: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    tokens_per_block: int = 1,
    rows_per_block: int = _ROWS_PER_BLOCK,
) -> torch.Tensor:
    """Project normalized BF16 latents without the shared-output add."""

    num_tokens = normalized.shape[0] if normalized.dim() == 2 else 0
    if not supports_latent_moe_projection(normalized, up_weight):
        raise NotImplementedError(
            "latent_moe_projection requires contiguous gfx950 BF16 tensors "
            "with 1-14 tokens and trailing dimensions 3584 and 7168"
        )
    if out is None:
        out = torch.empty(
            (num_tokens, _HIDDEN_DIM),
            dtype=torch.bfloat16,
            device=normalized.device,
        )
    else:
        _validate_output(out, (num_tokens, _HIDDEN_DIM), normalized.device)
    _validate_tiling(num_tokens, tokens_per_block, rows_per_block)
    return _launch_latent_moe_tail(
        normalized,
        out,
        normalized,
        up_weight,
        1.0,
        out=out,
        tokens_per_block=tokens_per_block,
        rows_per_block=rows_per_block,
        waves_per_eu=_WAVES_PER_EU,
        normalize_in_kernel=False,
        add_shared=False,
        weight_cache_modifier=(
            _B1_WEIGHT_CACHE_MODIFIER
            if num_tokens == 1
            else _MULTI_TOKEN_WEIGHT_CACHE_MODIFIER
        ),
    )
