# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Focused correctness and graph tests for the Kimi-K3 radix router."""

import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import checkAllclose

NUM_EXPERTS = 896
TOPK = 16
OUTPUT_STRIDE = 23
GATE_UP_WIDTH = 1536
ROUTED_WIDTH = 3584
FUSED_FRONT_WIDTH = GATE_UP_WIDTH + NUM_EXPERTS + ROUTED_WIDTH


def _make_logits(m: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    backing = torch.randn(
        (m, FUSED_FRONT_WIDTH),
        dtype=dtype,
        device="cuda",
        generator=generator,
    )
    logits = backing[:, GATE_UP_WIDTH : GATE_UP_WIDTH + NUM_EXPERTS]
    assert logits.stride() == (FUSED_FRONT_WIDTH, 1)
    return logits


def _make_bias(dtype: torch.dtype, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    bias = (
        torch.randn(NUM_EXPERTS, dtype=dtype, device="cuda", generator=generator) * 0.1
    )
    first = seed % (NUM_EXPERTS - 2 * TOPK)
    selected = torch.arange(first, first + TOPK, device="cuda")
    bias[selected] += 8.0
    return bias


def _make_outputs(
    m: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    weight_backing = torch.full(
        (m, OUTPUT_STRIDE), float("nan"), dtype=torch.float32, device="cuda"
    )
    id_backing = torch.full((m, OUTPUT_STRIDE), -777, dtype=torch.int32, device="cuda")
    return (
        weight_backing[:, :TOPK],
        id_backing[:, :TOPK],
        weight_backing,
        id_backing,
    )


def _run_biased(
    logits: torch.Tensor,
    bias: torch.Tensor,
    *,
    renormalize: bool = True,
    scale: float = 2.827,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights, ids, weight_backing, id_backing = _make_outputs(logits.shape[0])
    aiter.biased_grouped_topk(
        logits,
        bias,
        weights,
        ids,
        1,
        1,
        renormalize,
        scale,
    )
    torch.cuda.synchronize()
    assert torch.isnan(
        weight_backing[:, TOPK:]
    ).all(), "strided weight padding was overwritten"
    assert bool(
        (id_backing[:, TOPK:] == -777).all()
    ), "strided id padding was overwritten"
    return weights, ids


def _assert_ids_valid(ids: torch.Tensor) -> None:
    assert bool((ids >= 0).all())
    assert bool((ids < NUM_EXPERTS).all())
    sorted_ids = ids.sort(dim=-1).values
    assert bool(
        (sorted_ids[:, 1:] != sorted_ids[:, :-1]).all()
    ), "duplicate expert ids found in at least one complete output row"


def _assert_matches_reference(
    logits: torch.Tensor,
    bias: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
    *,
    renormalize: bool,
    scale: float,
) -> None:
    reference_logits = torch.nan_to_num(logits, nan=-float("inf"))
    reference_bias = torch.nan_to_num(bias, nan=-float("inf"))
    reference_weights, reference_ids = aiter.biased_grouped_topk_torch(
        reference_logits,
        reference_bias,
        TOPK,
        renormalize,
        1,
        1,
    )
    reference_weights *= scale

    _assert_ids_valid(ids)
    assert bool(torch.isfinite(weights).all())
    sorted_ids, order = ids.sort(dim=-1)
    sorted_reference_ids, reference_order = reference_ids.sort(dim=-1)
    torch.testing.assert_close(sorted_ids, sorted_reference_ids, rtol=0, atol=0)
    error_ratio = checkAllclose(
        weights.gather(1, order),
        reference_weights.gather(1, reference_order),
        rtol=2e-3,
        atol=2e-3,
        tol_err_ratio=0.0,
        msg="Kimi-K3 radix weights ",
        printLog=False,
    )
    assert error_ratio == 0, f"Kimi-K3 radix weight mismatch ratio: {error_ratio}"


def test_exact_kimi_k3_contract() -> None:
    cases = (
        (1, torch.bfloat16, True, 2.827),
        (1, torch.float32, True, 2.827),
        (2, torch.bfloat16, False, 1.0),
        (2, torch.float32, False, 1.0),
        (7, torch.bfloat16, True, 2.827),
        (7, torch.float32, True, 2.827),
        (14, torch.bfloat16, True, 1.0),
        (14, torch.float32, True, 1.0),
        (64, torch.float32, False, 2.827),
        (1024, torch.bfloat16, True, 2.827),
    )
    for m, dtype, renormalize, scale in cases:
        logits = _make_logits(m, dtype, seed=1100 + m)
        bias = _make_bias(dtype, seed=1200 + m)
        weights, ids = _run_biased(
            logits,
            bias,
            renormalize=renormalize,
            scale=scale,
        )
        _assert_matches_reference(
            logits,
            bias,
            weights,
            ids,
            renormalize=renormalize,
            scale=scale,
        )


def test_unbiased_contract() -> None:
    for m, dtype in ((7, torch.bfloat16), (14, torch.float32)):
        logits = _make_logits(m, dtype, seed=1700 + m)
        logits[:, :TOPK] += 8.0
        weights, ids, weight_backing, id_backing = _make_outputs(m)
        aiter.grouped_topk(logits, weights, ids, 1, 1, True, False, 2.827)
        torch.cuda.synchronize()
        assert torch.isnan(weight_backing[:, TOPK:]).all()
        assert bool((id_backing[:, TOPK:] == -777).all())
        reference_weights, reference_ids = aiter.grouped_topk_torch(
            logits, TOPK, True, 1, 1, "sigmoid"
        )
        reference_weights *= 2.827
        _assert_ids_valid(ids)
        sorted_ids, order = ids.sort(dim=-1)
        sorted_reference_ids, reference_order = reference_ids.sort(dim=-1)
        torch.testing.assert_close(sorted_ids, sorted_reference_ids, rtol=0, atol=0)
        assert (
            checkAllclose(
                weights.gather(1, order),
                reference_weights.gather(1, reference_order),
                rtol=2e-3,
                atol=2e-3,
                tol_err_ratio=0.0,
                msg="Kimi-K3 unbiased radix weights ",
                printLog=False,
            )
            == 0
        )


def test_nan_and_tie_contract() -> None:
    logits = _make_logits(7, torch.bfloat16, seed=2207)
    bias = _make_bias(torch.bfloat16, seed=2307)
    logits[3, 13] = float("nan")
    bias[17] = float("nan")
    weights, ids = _run_biased(logits, bias)
    _assert_matches_reference(
        logits,
        bias,
        weights,
        ids,
        renormalize=True,
        scale=2.827,
    )
    assert not bool((ids[3] == 13).any()), "NaN logit was selected"
    assert not bool((ids == 17).any()), "NaN correction bias was selected"

    levels = torch.linspace(-2.0, 2.0, 5, dtype=torch.bfloat16, device="cuda")
    tie_logits = levels[
        torch.arange(7 * NUM_EXPERTS, device="cuda").reshape(7, NUM_EXPERTS) % 5
    ]
    tie_bias = torch.zeros(NUM_EXPERTS, dtype=torch.bfloat16, device="cuda")
    tie_weights, tie_ids = _run_biased(tie_logits, tie_bias, scale=1.0)
    _assert_ids_valid(tie_ids)
    scores = tie_logits.float().sigmoid()
    selected_scores = scores.gather(1, tie_ids.to(torch.int64))
    threshold = torch.topk(scores, TOPK, dim=-1).values[:, -1:]
    assert bool((selected_scores >= threshold).all())
    expected_weights = selected_scores / selected_scores.sum(dim=-1, keepdim=True)
    assert (
        checkAllclose(
            tie_weights,
            expected_weights,
            rtol=2e-3,
            atol=2e-3,
            tol_err_ratio=0.0,
            msg="Kimi-K3 radix tied weights ",
            printLog=False,
        )
        == 0
    )


def test_dispatch_order_canary() -> None:
    logits = torch.zeros((1, NUM_EXPERTS), dtype=torch.float32, device="cuda")
    bias = torch.arange(NUM_EXPERTS, dtype=torch.float32, device="cuda")
    _, ids = _run_biased(logits, bias, renormalize=False, scale=1.0)
    if get_gfx() == "gfx950":
        expected = [
            880,
            884,
            888,
            892,
            881,
            885,
            889,
            893,
            882,
            886,
            890,
            894,
            883,
            887,
            891,
            895,
        ]
    else:
        expected = list(range(895, 879, -1))
    torch.testing.assert_close(
        ids[0], torch.tensor(expected, dtype=torch.int32, device="cuda"), rtol=0, atol=0
    )


def _exercise_changed_input_graph_replay(dtype: torch.dtype) -> None:
    m = 7
    static_logits = _make_logits(m, dtype, seed=3307)
    static_bias = _make_bias(dtype, seed=3308)
    weights, ids, _, _ = _make_outputs(m)

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            aiter.biased_grouped_topk(
                static_logits, static_bias, weights, ids, 1, 1, True, 2.827
            )
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        aiter.biased_grouped_topk(
            static_logits, static_bias, weights, ids, 1, 1, True, 2.827
        )

    previous_ids = None
    for seed in (4407, 5507):
        changed_logits = _make_logits(m, dtype, seed=seed)
        changed_bias = _make_bias(dtype, seed=seed + 1)
        static_logits.copy_(changed_logits)
        static_bias.copy_(changed_bias)
        graph.replay()
        torch.cuda.synchronize()
        _assert_matches_reference(
            static_logits,
            static_bias,
            weights,
            ids,
            renormalize=True,
            scale=2.827,
        )
        if previous_ids is not None:
            assert not torch.equal(
                ids, previous_ids
            ), "graph replay ignored changed inputs"
        previous_ids = ids.clone()


def test_changed_input_graph_replay() -> None:
    # Kimi-K3's vLLM GateLinear and correction bias are FP32. Keep BF16 here
    # as coverage for direct AITER callers, but require the actual serving
    # dtype to pass capture and changed-input replay as well.
    for dtype in (torch.bfloat16, torch.float32):
        _exercise_changed_input_graph_replay(dtype)


def main() -> None:
    if not torch.cuda.is_available():
        print("SKIP: grouped_topk radix tests require a GPU")
        return
    arch = get_gfx()
    if arch not in {"gfx942", "gfx950"}:
        print(f"SKIP: grouped_topk radix tests cover gfx942/gfx950, got {arch}")
        return
    test_exact_kimi_k3_contract()
    test_unbiased_contract()
    test_nan_and_tie_contract()
    test_dispatch_order_canary()
    test_changed_input_graph_replay()
    print(f"PASS: grouped_topk radix contract on {arch}")


if __name__ == "__main__":
    main()
