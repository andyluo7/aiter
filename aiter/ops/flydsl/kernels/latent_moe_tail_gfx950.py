# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Small-batch gfx950 BF16 RMSNorm, GEMV, and add kernel."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.arith import ArithValue, CmpIPredicate
from flydsl.expr.typing import ReductionOp, T

from aiter.ops.flydsl.kernels import buffer_ops, vector
from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    ptr_rsrc,
)

_LATENT_DIM = 3584
_HIDDEN_DIM = 7168
_WAVE_SIZE = 64


def _raw(value):
    return value.ir_value() if hasattr(value, "ir_value") else value


def _lds_load(ptr, index):
    return fx.ptr_load(ptr + fx.Int64(index))


def _lds_store(ptr, value, index):
    fx.ptr_store(value, ptr + fx.Int64(index))


def build_latent_moe_tail_module(
    num_tokens: int,
    tokens_per_block: int = 1,
    rows_per_block: int = 4,
    waves_per_eu: int = 0,
    normalize_in_kernel: bool = True,
    elements_per_thread: int = 8,
    use_dot2: bool = True,
    weight_cache_modifier: int = 0,
):
    """Build a launcher that reuses projection rows across a token tile."""

    if not 1 <= num_tokens <= 14:
        raise ValueError("num_tokens must be between 1 and 14")
    if not 1 <= tokens_per_block <= num_tokens:
        raise ValueError("tokens_per_block must be between 1 and num_tokens")
    if not 2 <= rows_per_block <= 64:
        raise ValueError("rows_per_block must be between 2 and 64")
    if waves_per_eu < 0:
        raise ValueError("waves_per_eu must be non-negative")
    if elements_per_thread not in (8, 16, 32):
        raise ValueError("elements_per_thread must be 8, 16, or 32")
    if weight_cache_modifier not in (0, 1, 2, 3):
        raise ValueError("weight_cache_modifier must be between 0 and 3")
    block_threads = (
        (
            (_LATENT_DIM + elements_per_thread - 1) // elements_per_thread
            + _WAVE_SIZE
            - 1
        )
        // _WAVE_SIZE
        * _WAVE_SIZE
    )
    if tokens_per_block * rows_per_block > block_threads:
        raise ValueError(
            "tokens_per_block * rows_per_block must not exceed block_threads"
        )
    waves = block_threads // _WAVE_SIZE
    vectors_per_thread = elements_per_thread // 8
    token_groups = (num_tokens + tokens_per_block - 1) // tokens_per_block

    @fx.struct
    class TailStorage:
        rms_sums: fx.Array[fx.Float32, tokens_per_block * waves, 16]
        inverse_rms: fx.Array[fx.Float32, tokens_per_block, 16]
        dot_sums: fx.Array[fx.Float32, tokens_per_block * rows_per_block * waves, 16]

    kernel_name = (
        f"latent_moe_tail_m{num_tokens}_bf16_gfx950_r{rows_per_block}"
        f"_t{tokens_per_block}"
        f"_wpe{waves_per_eu}_norm{int(normalize_in_kernel)}"
        f"_ept{elements_per_thread}"
        f"_dot2{int(use_dot2)}"
        f"_wcm{weight_cache_modifier}"
    )

    @flyc.kernel(
        name=kernel_name,
        known_block_size=[block_threads, 1, 1],
    )
    def tail_kernel(
        routed: fx.Pointer,
        shared: fx.Pointer,
        rms_weight: fx.Pointer,
        up_weight: fx.Pointer,
        output: fx.Pointer,
        epsilon: fx.Float32,
    ):
        i32 = T.i32
        f32 = T.f32
        i1 = ir.IntegerType.get_signless(1)
        fm_fast = arith.FastMathFlags.fast
        tid = ArithValue(gpu.thread_idx.x)
        lane = tid % arith.constant(_WAVE_SIZE, type=i32)
        wave = tid // arith.constant(_WAVE_SIZE, type=i32)
        linear_block = ArithValue(gpu.block_idx.x)
        token_group_count = arith.constant(token_groups, type=i32)
        # Adjacent workgroups cover token groups for the same output rows.
        token_group = ArithValue(arith.remui(_raw(linear_block), token_group_count))
        output_tile = ArithValue(arith.divui(_raw(linear_block), token_group_count))
        output_base = output_tile * arith.constant(rows_per_block, type=i32)
        token_base = token_group * arith.constant(tokens_per_block, type=i32)
        k_base = tid * arith.constant(elements_per_thread, type=i32)

        routed_rsrc = ptr_rsrc(routed)
        shared_rsrc = ptr_rsrc(shared)
        rms_weight_rsrc = ptr_rsrc(rms_weight)
        up_weight_rsrc = ptr_rsrc(up_weight)
        output_rsrc = ptr_rsrc(output)
        lds = fx.SharedAllocator().allocate(TailStorage).peek()
        rms_sums = lds.rms_sums.ptr
        inverse_rms = lds.inverse_rms.ptr
        dot_sums = lds.dot_sums.ptr

        zero_f32 = arith.constant(0.0, type=f32)
        one_over_dim = arith.constant(1.0 / _LATENT_DIM, type=f32)
        vec8_bf16 = T.vec(8, T.bf16)
        vec8_f32 = T.vec(8, f32)
        zero_bf16_vec = vector.broadcast(vec8_bf16, arith.constant(0.0, type=T.bf16))

        def load_bf16x8(resource, dword_index, cache_modifier=0):
            dwords = buffer_ops.buffer_load(
                resource,
                dword_index,
                vec_width=4,
                dtype=i32,
                cache_modifier=cache_modifier,
            )
            return vector.bitcast(vec8_bf16, dwords)

        def load_bf16x8_masked(resource, element_index, row_base=None):
            resource_element = (
                element_index if row_base is None else row_base + element_index
            )
            if const_expr(block_threads * elements_per_thread == _LATENT_DIM):
                return load_bf16x8(
                    resource, resource_element // arith.constant(2, type=i32)
                )
            valid = arith.cmpi(
                CmpIPredicate.ult,
                element_index,
                arith.constant(_LATENT_DIM, type=i32),
            )
            load_if = scf.IfOp(valid, results_=[vec8_bf16], has_else=True)
            with ir.InsertionPoint(load_if.then_block):
                loaded = load_bf16x8(
                    resource, resource_element // arith.constant(2, type=i32)
                )
                scf.YieldOp([_raw(loaded)])
            with ir.InsertionPoint(load_if.else_block):
                scf.YieldOp([_raw(zero_bf16_vec)])
            return load_if.results[0]

        def wave_reduce_add(value):
            reduced = _raw(value)
            for offset in (32, 16, 8, 4, 2, 1):
                peer = _raw(
                    ArithValue(reduced).shuffle_xor(
                        arith.constant(offset, type=i32),
                        arith.constant(_WAVE_SIZE, type=i32),
                    )
                )
                reduced = arith.AddFOp(reduced, peer, fastmath=fm_fast).result
            return reduced

        def dot_bf16x8(left, right):
            dot = zero_f32
            for pair_index in range_constexpr(4):
                left_pair = vector.from_elements(
                    T.vec(2, T.bf16),
                    [
                        vector.extract(
                            left,
                            static_position=[pair_index * 2],
                            dynamic_position=[],
                        ),
                        vector.extract(
                            left,
                            static_position=[pair_index * 2 + 1],
                            dynamic_position=[],
                        ),
                    ],
                )
                right_pair = vector.from_elements(
                    T.vec(2, T.bf16),
                    [
                        vector.extract(
                            right,
                            static_position=[pair_index * 2],
                            dynamic_position=[],
                        ),
                        vector.extract(
                            right,
                            static_position=[pair_index * 2 + 1],
                            dynamic_position=[],
                        ),
                    ],
                )
                dot = llvm.call_intrinsic(
                    f32,
                    "llvm.amdgcn.fdot2.f32.bf16",
                    [
                        left_pair,
                        right_pair,
                        dot,
                        arith.constant(False, type=i1),
                    ],
                    [],
                    [],
                )
            return dot

        is_lane_zero = arith.cmpi(CmpIPredicate.eq, lane, arith.constant(0, type=i32))
        routed_bf16_vectors_by_token = []
        routed_f32_vectors_by_token = []
        wave_square_sums = []
        for token_index in range_constexpr(tokens_per_block):
            token = token_base + arith.constant(token_index, type=i32)
            token_in_range = arith.cmpi(
                CmpIPredicate.ult,
                token,
                arith.constant(num_tokens, type=i32),
            )
            safe_token = arith.select(
                token_in_range, token, arith.constant(0, type=i32)
            )
            routed_base = ArithValue(safe_token) * arith.constant(_LATENT_DIM, type=i32)
            routed_bf16_vectors = []
            routed_f32_vectors = []
            for vector_index in range_constexpr(vectors_per_thread):
                element_index = k_base + arith.constant(vector_index * 8, type=i32)
                routed_bf16 = load_bf16x8_masked(
                    routed_rsrc, element_index, routed_base
                )
                routed_bf16_vectors.append(routed_bf16)
                routed_f32_vectors.append(ArithValue(routed_bf16).extf(vec8_f32))
            routed_bf16_vectors_by_token.append(routed_bf16_vectors)
            routed_f32_vectors_by_token.append(routed_f32_vectors)
            if const_expr(normalize_in_kernel):
                local_square_sum = ArithValue(zero_f32)
                for routed_f32 in routed_f32_vectors:
                    local_square_sum = local_square_sum + (
                        routed_f32 * routed_f32
                    ).reduce(ReductionOp.ADD, fastmath=fm_fast)
                wave_square_sums.append(wave_reduce_add(local_square_sum))

        if const_expr(normalize_in_kernel):
            lane_zero_if = scf.IfOp(is_lane_zero)
            with ir.InsertionPoint(lane_zero_if.then_block):
                for token_index in range_constexpr(tokens_per_block):
                    rms_index = arith.constant(token_index * waves, type=i32) + wave
                    _lds_store(rms_sums, wave_square_sums[token_index], rms_index)
                scf.YieldOp([])
            gpu.barrier()

            is_thread_zero = arith.cmpi(
                CmpIPredicate.eq, tid, arith.constant(0, type=i32)
            )
            thread_zero_if = scf.IfOp(is_thread_zero)
            with ir.InsertionPoint(thread_zero_if.then_block):
                for token_index in range_constexpr(tokens_per_block):
                    total_square_sum = ArithValue(zero_f32)
                    for wave_index in range_constexpr(waves):
                        rms_index = arith.constant(
                            token_index * waves + wave_index, type=i32
                        )
                        total_square_sum = total_square_sum + _lds_load(
                            rms_sums, rms_index
                        )
                    variance = total_square_sum * one_over_dim
                    inverse = fmath.rsqrt(
                        variance + ArithValue(epsilon), fastmath=fm_fast
                    )
                    _lds_store(
                        inverse_rms,
                        _raw(inverse),
                        arith.constant(token_index, type=i32),
                    )
                scf.YieldOp([])
            gpu.barrier()

            gamma_f32_vectors = []
            for vector_index in range_constexpr(vectors_per_thread):
                element_index = k_base + arith.constant(vector_index * 8, type=i32)
                gamma_bf16 = load_bf16x8_masked(rms_weight_rsrc, element_index)
                gamma_f32_vectors.append(ArithValue(gamma_bf16).extf(vec8_f32))
            normalized_dot_vectors_by_token = []
            normalized_bf16_vectors_by_token = []
            for token_index in range_constexpr(tokens_per_block):
                inverse = ArithValue(
                    _lds_load(inverse_rms, arith.constant(token_index, type=i32))
                )
                normalized_dot_vectors = []
                normalized_bf16_vectors = []
                for vector_index in range_constexpr(vectors_per_thread):
                    normalized_f32 = (
                        routed_f32_vectors_by_token[token_index][vector_index]
                        * gamma_f32_vectors[vector_index]
                        * inverse
                    )
                    normalized_bf16 = normalized_f32.truncf(vec8_bf16)
                    normalized_bf16_vectors.append(normalized_bf16)
                    normalized_dot_vectors.append(normalized_bf16.extf(vec8_f32))
                normalized_bf16_vectors_by_token.append(normalized_bf16_vectors)
                normalized_dot_vectors_by_token.append(normalized_dot_vectors)
        else:
            normalized_bf16_vectors_by_token = routed_bf16_vectors_by_token
            normalized_dot_vectors_by_token = routed_f32_vectors_by_token

        accumulators_by_token = [[] for _ in range(tokens_per_block)]
        for row_index in range_constexpr(rows_per_block):
            row = output_base + arith.constant(row_index, type=i32)
            row_in_range = arith.cmpi(
                CmpIPredicate.ult,
                row,
                arith.constant(_HIDDEN_DIM, type=i32),
            )
            safe_row = arith.select(row_in_range, row, arith.constant(0, type=i32))
            local_dots = []
            for _ in range_constexpr(tokens_per_block):
                local_dots.append(ArithValue(zero_f32))
            for vector_index in range_constexpr(vectors_per_thread):
                row_element = k_base + arith.constant(vector_index * 8, type=i32)
                weight_element = (
                    safe_row * arith.constant(_LATENT_DIM, type=i32) + row_element
                )
                if const_expr(block_threads * elements_per_thread == _LATENT_DIM):
                    weight_bf16 = load_bf16x8(
                        up_weight_rsrc,
                        weight_element // arith.constant(2, type=i32),
                        weight_cache_modifier,
                    )
                else:
                    valid = arith.cmpi(
                        CmpIPredicate.ult,
                        row_element,
                        arith.constant(_LATENT_DIM, type=i32),
                    )
                    weight_if = scf.IfOp(valid, results_=[vec8_bf16], has_else=True)
                    with ir.InsertionPoint(weight_if.then_block):
                        loaded_weight = load_bf16x8(
                            up_weight_rsrc,
                            weight_element // arith.constant(2, type=i32),
                            weight_cache_modifier,
                        )
                        scf.YieldOp([_raw(loaded_weight)])
                    with ir.InsertionPoint(weight_if.else_block):
                        scf.YieldOp([_raw(zero_bf16_vec)])
                    weight_bf16 = weight_if.results[0]
                if const_expr(not use_dot2):
                    weight_f32 = ArithValue(weight_bf16).extf(vec8_f32)
                for token_index in range_constexpr(tokens_per_block):
                    if const_expr(use_dot2):
                        local_dots[token_index] = local_dots[token_index] + dot_bf16x8(
                            normalized_bf16_vectors_by_token[token_index][vector_index],
                            weight_bf16,
                        )
                    else:
                        local_dots[token_index] = local_dots[token_index] + (
                            normalized_dot_vectors_by_token[token_index][vector_index]
                            * weight_f32
                        ).reduce(ReductionOp.ADD, fastmath=fm_fast)
            for token_index in range_constexpr(tokens_per_block):
                accumulators_by_token[token_index].append(
                    wave_reduce_add(local_dots[token_index])
                )

        lane_zero_if = scf.IfOp(is_lane_zero)
        with ir.InsertionPoint(lane_zero_if.then_block):
            for token_index in range_constexpr(tokens_per_block):
                for row_index in range_constexpr(rows_per_block):
                    index = (
                        arith.constant(
                            (token_index * rows_per_block + row_index) * waves,
                            type=i32,
                        )
                        + wave
                    )
                    _lds_store(
                        dot_sums,
                        accumulators_by_token[token_index][row_index],
                        index,
                    )
            scf.YieldOp([])
        gpu.barrier()

        writes_output = arith.cmpi(
            CmpIPredicate.ult,
            tid,
            arith.constant(tokens_per_block * rows_per_block, type=i32),
        )
        write_token_index = ArithValue(
            arith.divui(_raw(tid), arith.constant(rows_per_block, type=i32))
        )
        write_row_index = ArithValue(
            arith.remui(_raw(tid), arith.constant(rows_per_block, type=i32))
        )
        output_index = output_base + write_row_index
        output_token = token_base + write_token_index
        output_in_range = arith.cmpi(
            CmpIPredicate.ult,
            output_index,
            arith.constant(_HIDDEN_DIM, type=i32),
        )
        token_in_range = arith.cmpi(
            CmpIPredicate.ult,
            output_token,
            arith.constant(num_tokens, type=i32),
        )
        writes_output = arith.andi(writes_output, output_in_range)
        writes_output = arith.andi(writes_output, token_in_range)
        write_if = scf.IfOp(writes_output)
        with ir.InsertionPoint(write_if.then_block):
            dot = ArithValue(zero_f32)
            for wave_index in range_constexpr(waves):
                index = tid * arith.constant(waves, type=i32) + arith.constant(
                    wave_index, type=i32
                )
                dot = dot + _lds_load(dot_sums, index)
            projected_bf16 = arith.trunc_f(T.bf16, _raw(dot))
            projected_f32 = ArithValue(arith.extf(f32, projected_bf16))
            token_output_index = (
                output_token * arith.constant(_HIDDEN_DIM, type=i32) + output_index
            )
            shared_bf16 = buffer_ops.buffer_load(
                shared_rsrc, token_output_index, vec_width=1, dtype=T.bf16
            )
            shared_f32 = ArithValue(arith.extf(f32, shared_bf16))
            result = arith.trunc_f(T.bf16, _raw(projected_f32 + shared_f32))
            buffer_ops.buffer_store(result, output_rsrc, token_output_index)
            scf.YieldOp([])

    @flyc.jit
    def launch_tail(
        routed: fx.Pointer,
        shared: fx.Pointer,
        rms_weight: fx.Pointer,
        up_weight: fx.Pointer,
        output: fx.Pointer,
        epsilon: fx.Float32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        ctx = CompilationContext.get_current()
        if const_expr(waves_per_eu > 0):
            for operation in ctx.gpu_module_body.operations:
                if (
                    hasattr(operation, "attributes")
                    and operation.OPERATION_NAME == "gpu.func"
                ):
                    operation.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                        T.i32, int(waves_per_eu)
                    )
        tail_kernel(
            routed,
            shared,
            rms_weight,
            up_weight,
            output,
            epsilon,
        ).launch(
            grid=(
                ((_HIDDEN_DIM + rows_per_block - 1) // rows_per_block) * token_groups,
                1,
                1,
            ),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    launch_tail.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }
    return launch_tail
