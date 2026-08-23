# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Hand-written asm sparse paged prefill attention for DeepSeek-V4 on gfx1250.

The asm counterpart of :mod:`aiter.ops.pa_sparse_prefill_opus`'s fp8 entry.
Same two-region sparse scaled-dot-product attention over a paged prefix source
(``unified_kv_*``) and a flat per-fwd extend source (``kv_*``), same split
fp8-NoPE / bf16-RoPE input layout, same per-head fp32 softmax-denominator sink,
same ``[T, H, 512]`` bf16 output. Only the backend differs: gfx950 runs the
OPUS HIP kernel, gfx1250 runs the ``32mx4_32nx1`` SP3 kernel through this
module. The signature is kept identical so callers and tests can swap one for
the other.

Despite "prefill" in the name there is no causal mask and no notion of query or
KV sequence length -- causality is pre-baked by the caller into the two CSR
index lists, exactly as for the OPUS op.

Constraints of the single kernel variant shipped today:

* ``H == 128`` exactly. One workgroup serves one query token x 128 heads, and
  the kernel's Q address math folds the head-block index in additively, so it
  is only correct while ``gridDim.y == 1``. Other head counts are rejected
  rather than silently mis-read.
* Head dim ``512`` (NoPE row: 448 fp8 values + 14 E8M0 block scales + padding)
  and RoPE dim ``64``.
* KV tile size is ``32``; partial trailing tiles are masked by the kernel, so
  ``nnz`` need not be a multiple of anything.
* Registered as the ``prefill=1`` row of ``hsa/<arch>/mla_v4/mla_v4_asm.csv``
  (kernel ``mla_a8w8_qh128_1tg_32mx4_32nx1_sparse_pfl``).
* Empty CSR rows (``kv_indptr[i] == kv_indptr[i + 1]``) are allowed, and either
  region may be entirely empty -- but the index/indptr tensors themselves must
  still be real allocations, never ``None``.

See ``csrc/py_itfs_cu/asm_mla_sparse_prefill.cu`` for the kernarg ABI.
"""

import torch

from ..jit.core import compile_ops
from ..jit.utils.chip_info import get_gfx_runtime
from ..jit.utils.torch_guard import torch_compile_guard

MD_NAME = "module_mla_sparse_prefill_asm"


# NOTE: ctypes binds positionally off this signature -- the argument order here
# must match `mla_sparse_prefill_fp8_asm_fwd` in
# csrc/py_itfs_cu/asm_mla_sparse_prefill.cu one for one. A silent reorder here
# is a silent wrong-pointer launch, not a compile error.
@compile_ops(MD_NAME, fc_name="mla_sparse_prefill_fp8_asm_fwd", ffi_type="ctypes")
def mla_sparse_prefill_fp8_asm_fwd(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    unified_kv_nope: torch.Tensor,
    unified_kv_rope: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv_nope: torch.Tensor,
    kv_rope: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    out: torch.Tensor,
    softmax_scale: float,
) -> None: ...


def _mla_sparse_prefill_fp8_asm_fake(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    unified_kv_nope: torch.Tensor,
    unified_kv_rope: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv_nope: torch.Tensor,
    kv_rope: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is not None:
        return out
    t, h, _ = q_nope.shape
    return torch.empty((t, h, 512), dtype=torch.bfloat16, device=q_nope.device)


@torch_compile_guard(mutates_args=["out"], gen_fake=_mla_sparse_prefill_fp8_asm_fake)
def mla_sparse_prefill_fp8_asm(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    unified_kv_nope: torch.Tensor,
    unified_kv_rope: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv_nope: torch.Tensor,
    kv_rope: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sparse prefill attention with split fp8 NoPE and bf16 RoPE inputs (asm).

    Signature-compatible with
    :func:`aiter.ops.pa_sparse_prefill_opus.pa_sparse_prefill_fp8_opus`.

    Args:
      q_nope:            ``[T, H, 512]`` fp8 query without positional encoding.
      q_rope:            ``[T, H, 64]`` bf16 query RoPE encoding part.
      unified_kv_nope:   ``[total_pages, 512]`` fp8 prefix KV NoPE source.
      unified_kv_rope:   ``[total_pages, 64]`` bf16 prefix KV RoPE source.
      kv_indices_prefix: ``[total_prefix]`` int32 row indices into the prefix
        sources, concatenated per token.
      kv_indptr_prefix:  ``[T+1]`` int32 CSR row pointers.
      kv_nope:           ``[total_tokens, 512]`` fp8 extend KV NoPE source.
      kv_rope:           ``[total_tokens, 64]`` bf16 extend KV RoPE source.
      kv_indices_extend: ``[total_extend]`` int32 row indices into the extend
        sources, concatenated per token.
      kv_indptr_extend:  ``[T+1]`` int32 CSR row pointers.
      attn_sink:         ``[H]`` fp32 per-head softmax-denom bias.
      softmax_scale:     float scalar applied to the combined QK^T scores.
      out:               Optional ``[T, H, 512]`` bf16 output buffer; allocated
        if ``None``.

    Returns:
      ``out`` (``[T, H, 512]`` bf16).
    """
    gfx = get_gfx_runtime()
    if gfx != "gfx1250":
        raise RuntimeError(f"mla_sparse_prefill_fp8_asm requires gfx1250, got {gfx}")

    if q_nope.dtype != unified_kv_nope.dtype or q_nope.dtype != kv_nope.dtype:
        raise RuntimeError(
            f"NoPE dtype mismatch: q_nope={q_nope.dtype}, "
            f"unified_kv_nope={unified_kv_nope.dtype}, kv_nope={kv_nope.dtype}"
        )
    if q_rope.dtype != torch.bfloat16:
        raise RuntimeError(f"q_rope must be bf16, got {q_rope.dtype}")

    t, h = q_nope.shape[0], q_nope.shape[1]
    if h != 128:
        # Hard constraint, not a dispatch miss -- see the module docstring.
        raise RuntimeError(f"mla_sparse_prefill_fp8_asm requires H == 128, got {h}")
    if out is None:
        out = torch.empty((t, h, 512), dtype=torch.bfloat16, device=q_nope.device)
    elif out.shape != (t, h, 512) or out.dtype != torch.bfloat16:
        raise RuntimeError(
            f"out shape/dtype mismatch: got shape={tuple(out.shape)} dtype={out.dtype}, "
            f"expected shape={(t, h, 512)} dtype={torch.bfloat16}"
        )

    mla_sparse_prefill_fp8_asm_fwd(
        q_nope,
        q_rope,
        unified_kv_nope,
        unified_kv_rope,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv_nope,
        kv_rope,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        out,
        float(softmax_scale),
    )
    return out


__all__ = [
    "mla_sparse_prefill_fp8_asm",
    "mla_sparse_prefill_fp8_asm_fwd",
]
