"""Synthetic-only tests for the whole-kernel CUDA candidate cost model.

No test imports torch, initializes CUDA, compiles a kernel, or asserts a real
hardware ranking.  Numeric expectations below are independent hand formulas.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest


_ROOT = Path(__file__).resolve().parents[3]
_MODULE_PATH = _ROOT / "tilelang" / "cuda" / "cost_model.py"
_SPEC = importlib.util.spec_from_file_location("tilelang_cuda_cost_model_synthetic", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
cost_model = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = cost_model
_SPEC.loader.exec_module(cost_model)

CalibrationError = cost_model.CalibrationError
CostModelError = cost_model.CostModelError
CudaCostModel = cost_model.CudaCostModel
KernelWork = cost_model.KernelWork
PhaseWork = cost_model.PhaseWork


def hardware(**overrides):
    device = {
        "name": "synthetic-sm120",
        "compute_capability": "12.0",
        "multi_processor_count": 2,
        "warp_size": 32,
        "max_threads_per_block": 1024,
        "max_threads_per_multiprocessor": 1536,
        "max_registers_per_block": 65536,
        "max_registers_per_multiprocessor": 65536,
        "max_registers_per_thread": None,
        "shared_memory_per_block_bytes": 49152,
        "shared_memory_per_block_optin_bytes": 101376,
        "shared_memory_per_multiprocessor_bytes": 102400,
        "reserved_shared_memory_per_block_bytes": 1024,
        "clock_rate_khz": 1_000_000,
    }
    device.update(overrides)
    return {"schema": cost_model.HARDWARE_SCHEMA, "version": 1, "device": device}


def calibration(**overrides):
    rates = {
        "clock_hz": 1.0e9,
        "mma_f16_flops_per_second": 2.0e12,
        "mma_bf16_flops_per_second": 2.0e12,
        "mma_tf32_flops_per_second": 1.0e12,
        "cuda_fp32_flops_per_second": 2.0e11,
        "dram_bytes_per_second": 2.0e12,
        "l2_bytes_per_second": 4.0e12,
        "shared_bytes_per_second": 8.0e12,
        "lsu_warp_instructions_per_second": 2.0e10,
        "sfu_exp2_operations_per_second": 2.0e11,
        "barrier_ns": 30.0,
        "launch_ns": 4000.0,
    }
    rates.update(overrides)
    return {
        "schema": cost_model.CALIBRATION_SCHEMA,
        "version": 1,
        "status": "measured",
        "rates": rates,
    }


def hand_calibration():
    # Two SMs therefore make every per-SM rate 1e9 units/s.  These are synthetic
    # unit choices, not hardware estimates.
    result = calibration(
        clock_hz=1.0e9,
        mma_f16_flops_per_second=2.0e9,
        mma_bf16_flops_per_second=2.0e9,
        mma_tf32_flops_per_second=2.0e9,
        cuda_fp32_flops_per_second=2.0e9,
        dram_bytes_per_second=2.0e9,
        l2_bytes_per_second=2.0e9,
        shared_bytes_per_second=2.0e9,
        lsu_warp_instructions_per_second=2.0e9,
        sfu_exp2_operations_per_second=2.0e9,
        barrier_ns=1000.0,
        launch_ns=1.0e6,
    )
    return result


def gemm_model(**hw_overrides):
    return CudaCostModel(
        hardware(**hw_overrides), hand_calibration(), "gemm",
        {"M": 64, "N": 64, "K": 64, "dtype": "bf16"},
    )


def gemm_config(**overrides):
    config = {
        "block_M": 32,
        "block_N": 32,
        "block_K": 32,
        "num_stages": 2,
        "threads": 128,
        "registers_per_thread": 32,
        "copy_kind": "cp_async",
    }
    config.update(overrides)
    return config


def gdn_shape():
    return {"B": 1, "T": 64, "H": 2, "Hg": 1, "K": 128, "DV": 128, "dtype": "bf16"}


def assert_finite_json_terms(score, terms):
    assert isinstance(score, float)
    assert math.isfinite(score) and score >= 0
    assert isinstance(terms, dict) and terms
    assert score == terms["predicted_ms"]
    for name, value in terms.items():
        assert isinstance(name, str)
        assert isinstance(value, (int, float)) and not isinstance(value, bool)
        assert math.isfinite(value) and value >= 0, (name, value)
    json.dumps(terms, allow_nan=False, sort_keys=True)


def test_guided_protocol_json_and_detached_model_card():
    model = gemm_model()
    config = gemm_config()
    original = copy.deepcopy(config)
    score, terms = model.predict(config)
    assert config == original
    assert_finite_json_terms(score, terms)
    assert list(terms) == sorted(terms)
    assert model.predict(config) == (score, terms)

    card = model.model_card()
    json.dumps(card, allow_nan=False, sort_keys=True)
    assert card["score"] == {"term": "predicted_ms", "units": "ms", "direction": "minimize"}
    assert card["fallback_used"] == []
    assert card["calibration"]["units"]["mma_bf16_flops_per_second"] == "FLOP/s"
    assert card["calibration"]["units"]["launch_ns"] == "ns"
    card["shape"]["M"] = -1
    assert model.model_card()["shape"]["M"] == 64


def test_from_files_roundtrip_and_operator_descriptor(tmp_path):
    hw_path = tmp_path / "hardware.json"
    cal_path = tmp_path / "calibration.json"
    shape_path = tmp_path / "shape.json"
    op_path = tmp_path / "operator.json"
    hw_path.write_text(json.dumps(hardware()), encoding="utf-8")
    cal_path.write_text(json.dumps(calibration()), encoding="utf-8")
    shape_path.write_text(json.dumps({"shape": {"M": 64, "N": 64, "K": 64, "dtype": "bf16"}}), encoding="utf-8")
    op_path.write_text(json.dumps({"operator": "gemm"}), encoding="utf-8")
    model = CudaCostModel.from_files(hw_path, cal_path, op_path, shape_path)
    score, terms = model.predict(gemm_config())
    assert_finite_json_terms(score, terms)
    assert model.model_card()["hardware"]["source"] == str(hw_path)
    assert model.model_card()["calibration"]["source"] == str(cal_path)


@pytest.mark.parametrize("missing", ["clock_hz", "mma_bf16_flops_per_second", "launch_ns"])
def test_require_measured_missing_critical_rate_fails_at_construction(missing):
    cal = calibration()
    del cal["rates"][missing]
    with pytest.raises(CalibrationError, match=missing):
        CudaCostModel(hardware(), cal, "gemm", {"M": 64, "N": 64, "K": 64})


def test_not_performed_manifest_calibration_is_not_measured():
    cal = calibration()
    cal["status"] = "not_performed"
    with pytest.raises(CalibrationError, match="not_performed"):
        CudaCostModel(hardware(), cal, "gemm", {"M": 64, "N": 64, "K": 64})

    manifest = {"calibration": {"status": "not_performed", "rates": calibration()["rates"]}}
    with pytest.raises(CalibrationError, match="not_performed"):
        CudaCostModel(hardware(), manifest, "gemm", {"M": 64, "N": 64, "K": 64})


def test_non_measured_fallback_is_explicit_and_never_free():
    cal = calibration()
    cal["status"] = "not_performed"
    del cal["rates"]["launch_ns"]
    model = CudaCostModel(
        hardware(), cal, "gemm", {"M": 64, "N": 64, "K": 64, "dtype": "bf16"},
        require_measured=False,
    )
    score, terms = model.predict(gemm_config())
    assert_finite_json_terms(score, terms)
    assert any("launch_ns" in item for item in model.fallback_used)
    assert any("launch_ns" in item for item in model.model_card()["fallback_used"])
    assert terms["launch_ms"] == 1000.0  # Explicit 1e9 ns placeholder, not zero.
    assert terms["fallback_count"] >= 1


def test_calibration_unit_and_field_provenance_validation():
    cal = calibration()
    cal["rates"]["barrier_ns"] = {"value": 30.0, "unit": "us", "source": "measured"}
    with pytest.raises(CalibrationError, match="unit"):
        CudaCostModel(hardware(), cal, "gemm", {"M": 64, "N": 64, "K": 64})

    cal = calibration()
    cal["field_sources"] = {"rates.mma_bf16_flops_per_second": "provided"}
    with pytest.raises(CalibrationError, match="provided"):
        CudaCostModel(hardware(), cal, "gemm", {"M": 64, "N": 64, "K": 64})


def test_gemm_hand_formula_cp_async_and_tma():
    model = gemm_model(max_threads_per_multiprocessor=256)
    score, terms = model.predict(gemm_config())

    # Four CTAs, two K steps each.  Per CTA:
    # MMA = 2*32*32*64 = 131072 FLOP -> 0.131072 ms at 1e9 FLOP/s/SM.
    # A+B+C = 4096+4096+2048 = 10240 B -> max(DRAM,L2)=0.01024 ms.
    # Shared = 2*(A+B) = 16384 B -> 0.016384 ms.
    # cp.async LSU = 8192/512 + 2048/512 + 16384/512 = 52 -> 0.000052 ms.
    # Four barriers -> 0.004 ms.  Occupancy=2, 2 SMs, four CTAs -> one wave.
    assert terms["flops"] == 4 * 131072
    assert terms["useful_flops"] == 2 * 64**3
    assert terms["dram_bytes"] == terms["l2_bytes"] == 4 * 10240
    assert terms["shared_bytes"] == 4 * 16384
    assert terms["lsu_warp_instructions"] == 4 * 52
    assert terms["barrier_count"] == 4 * 4
    assert terms["occupancy_ctas_per_sm"] == 2
    assert terms["waves"] == 1
    assert terms["shared_memory_bytes_per_cta"] == 8192
    assert terms["stage_overlap_ms"] == pytest.approx(0.131072)
    assert score == pytest.approx(1.0 + 0.131072 + 0.000052 + 0.004)

    tma_score, tma = model.predict(gemm_config(copy_kind="tma"))
    # TMA: two descriptors per K step, four total; stores and shared issue stay.
    assert tma["tma_descriptors"] == 4 * 4
    assert tma["lsu_warp_instructions"] == 4 * (4 + 4 + 32)
    assert tma["dram_bytes"] == terms["dram_bytes"]
    assert tma["l2_bytes"] == terms["l2_bytes"]
    assert tma_score == pytest.approx(1.0 + 0.131072 + 0.000040 + 0.004)


def test_single_stage_hand_formula_and_stage_bytes():
    model = gemm_model(max_threads_per_multiprocessor=256)
    _, one = model.predict(gemm_config(num_stages=1))
    _, two = model.predict(gemm_config(num_stages=2))
    assert one["shared_memory_bytes_per_cta"] == 4096
    assert two["shared_memory_bytes_per_cta"] == 2 * one["shared_memory_bytes_per_cta"]
    # No overlap: 0.131072 + 0.01024 + 0.016384 = 0.157696 ms.
    assert one["stage_overlap_ms"] == pytest.approx(0.157696)
    assert one["predicted_ms"] == pytest.approx(1.0 + 0.157696 + 0.000052 + 0.004)
    # This checks formulas only; it is not a claim about which config wins on a GPU.


def test_unknown_registers_use_conservative_occupancy_and_annotation():
    model = gemm_model(max_threads_per_multiprocessor=256)
    config = gemm_config()
    del config["registers_per_thread"]
    _, unknown = model.predict(config)
    _, known = model.predict(gemm_config())
    assert unknown["registers_per_thread"] == 0
    assert unknown["register_occupancy_conservative"] == 1
    assert unknown["occupancy_ctas_per_sm"] == 1
    assert unknown["waves"] == 2
    assert known["register_occupancy_conservative"] == 0
    assert known["occupancy_ctas_per_sm"] == 2
    assert known["waves"] == 1
    assert unknown["fallback_count"] >= 1
    assert any("registers_per_thread" in item for item in model.model_card()["conditional_fallbacks"])


@pytest.mark.parametrize(
    "operator,shape,config,expected",
    [
        (
            "gemm", {"M": 64, "N": 64, "K": 64, "dtype": "bf16"}, gemm_config(),
            {"flops": 524288, "useful_flops": 524288},
        ),
        (
            "conv2d_nhwc",
            {"N": 1, "H": 8, "W": 8, "C": 16, "K": 16, "R": 3, "S": 3, "dtype": "bf16"},
            {"block_M": 8, "block_N": 8, "block_K": 16, "num_stages": 2, "registers_per_thread": 32},
            {"flops": 184320, "useful_flops": 165888},
        ),
        (
            "flash_attention_fwd",
            {"B": 1, "H": 2, "S": 64, "D": 64, "dtype": "bf16"},
            {"block_M": 16, "block_N": 32, "num_stages": 2, "registers_per_thread": 64},
            {"mma_flops": 2097152, "useful_flops": 2097152, "sfu_exp2_operations": 8192},
        ),
        (
            "gdn_fwd", gdn_shape(), {"block_DV": 64, "registers_per_thread": 128},
            {"mma_flops": 17825792},
        ),
        (
            "gdn_bwd", gdn_shape(), {"prepare_h_num_stages": 2, "registers_per_thread": 128},
            {},
        ),
    ],
)
def test_every_operator_has_finite_itemized_predictions(operator, shape, config, expected):
    model = CudaCostModel(hardware(), calibration(), operator, shape)
    score, terms = model.predict(config)
    assert_finite_json_terms(score, terms)
    for name, value in expected.items():
        assert terms[name] == value, (operator, name, terms[name], value)
    for name in (
        "occupancy_ctas_per_sm", "waves", "stage_overlap_ms", "dram_ms", "l2_ms",
        "shared_ms", "lsu_ms", "sfu_exp2_ms", "barrier_ms", "launch_ms",
    ):
        assert name in terms


def test_gdn_forward_backward_stage_identity_not_ranking():
    shape = gdn_shape()
    fwd = CudaCostModel(hardware(), calibration(), "gdn_fwd", shape)
    bwd = CudaCostModel(hardware(), calibration(), "gdn_bwd", shape)
    fwd_score, f = fwd.predict({"block_DV": 64, "registers_per_thread": 128})
    bwd_score, b = bwd.predict({"prepare_h_num_stages": 2, "registers_per_thread": 128})
    assert_finite_json_terms(fwd_score, f)
    assert_finite_json_terms(bwd_score, b)

    assert f["gdn_kkt_ms"] > 0 and f["gdn_solve_ms"] > 0
    assert f["gdn_wy_ms"] > 0 and f["gdn_recurrence_ms"] > 0 and f["gdn_output_ms"] > 0
    assert f["gdn_dhu_ms"] == f["gdn_dqkwg_ms"] == 0
    assert f["gdn_group_reduce_ms"] == f["gdn_reverse_cumsum_ms"] == 0
    assert b["gdn_kkt_ms"] == b["gdn_solve_ms"] == b["gdn_output_ms"] == 0
    assert b["gdn_wy_ms"] > 0 and b["gdn_recurrence_ms"] > 0
    assert b["gdn_dhu_ms"] > 0 and b["gdn_dqkwg_ms"] > 0
    assert b["gdn_group_reduce_ms"] > 0 and b["gdn_reverse_cumsum_ms"] > 0
    assert f["flops"] != b["flops"]
    assert f["launch_count"] == 2
    assert b["launch_count"] == 5  # prepare_h, fused, dq reduce, dk reduce, reverse cumsum
    assert f["predicted_ms"] == pytest.approx(
        f["launch_ms"] + sum(f[name] for name in (
            "gdn_kkt_ms", "gdn_solve_ms", "gdn_wy_ms", "gdn_recurrence_ms", "gdn_output_ms",
        ))
    )
    assert b["predicted_ms"] == pytest.approx(
        b["launch_ms"] + sum(b[name] for name in (
            "gdn_wy_ms", "gdn_recurrence_ms", "gdn_dhu_ms", "gdn_dqkwg_ms",
            "gdn_group_reduce_ms", "gdn_reverse_cumsum_ms",
        ))
    )
    # Deliberately no fwd-vs-bwd or real-hardware faster-than assertion.


def test_gdn_block_dv_and_prepare_h_stage_byte_constraints():
    shape = gdn_shape()
    fwd = CudaCostModel(hardware(), calibration(), "gdn_fwd", shape)
    _, dv64 = fwd.predict({"block_DV": 64, "registers_per_thread": 128})
    _, dv128 = fwd.predict({"block_DV": 128, "registers_per_thread": 128})
    assert dv64["gdn_block_dv"] == 64
    assert dv128["gdn_block_dv"] == 128
    assert dv64["shared_memory_bytes_per_cta"] == 66304
    assert dv128["shared_memory_bytes_per_cta"] == 94976
    assert dv64["cta_count"] != dv128["cta_count"]

    bwd = CudaCostModel(hardware(), calibration(), "gdn_bwd", shape)
    _, stage1 = bwd.predict({"prepare_h_num_stages": 1, "registers_per_thread": 128})
    _, stage2 = bwd.predict({"prepare_h_num_stages": 2, "registers_per_thread": 128})
    # G5 measured dynamic shared bytes: 57344 + 18432 * stages.
    assert stage1["gdn_prepare_h_shared_memory_bytes_per_cta"] == 75776
    assert stage2["gdn_prepare_h_shared_memory_bytes_per_cta"] == 94208
    assert stage2["gdn_prepare_h_shared_memory_bytes_per_cta"] - stage1["gdn_prepare_h_shared_memory_bytes_per_cta"] == 18432
    assert stage1["shared_memory_bytes_per_cta"] == 90880  # max(prepare_h, fused_bwd)
    assert stage2["shared_memory_bytes_per_cta"] == 94208
    assert stage2["gdn_prepare_h_num_stages"] == 2

    # Relax only the synthetic cap to observe the exact stage-3 capacity; the
    # real 101376 B cap must still reject the same 112640 B allocation.
    relaxed = CudaCostModel(
        hardware(
            shared_memory_per_block_optin_bytes=262144,
            shared_memory_per_multiprocessor_bytes=262144,
        ),
        calibration(),
        "gdn_bwd",
        shape,
    )
    _, stage3 = relaxed.predict({"prepare_h_num_stages": 3, "registers_per_thread": 128})
    assert stage3["gdn_prepare_h_shared_memory_bytes_per_cta"] == 112640
    assert stage3["shared_memory_bytes_per_cta"] == 112640
    assert stage3["gdn_prepare_h_num_stages"] == 3
    with pytest.raises(CostModelError, match="gdn_prepare_h shared bytes 112640 exceed per-block opt-in 101376"):
        bwd.predict({"prepare_h_num_stages": 3, "registers_per_thread": 128})

    with pytest.raises(CostModelError, match="only|not gdn_fwd|applies"):
        fwd.predict({"block_DV": 64, "prepare_h_num_stages": 2})
    with pytest.raises(CostModelError, match="no block_DV"):
        bwd.predict({"block_DV": 64})


def test_generic_stage_capacity_and_bad_configs_raise():
    model = gemm_model(
        shared_memory_per_block_optin_bytes=9000,
        shared_memory_per_multiprocessor_bytes=16384,
    )
    model.predict(gemm_config(num_stages=2))
    with pytest.raises(CostModelError, match="shared bytes"):
        model.predict(gemm_config(num_stages=3))
    with pytest.raises(CostModelError, match="num_stages"):
        model.predict(gemm_config(num_stages=0))
    with pytest.raises(CostModelError, match="Unknown config"):
        model.predict(gemm_config(made_up_knob=7))
    with pytest.raises(CostModelError, match="Missing config"):
        model.predict({"block_M": 32})


def test_model_and_plugin_errors_are_not_silently_dropped():
    with pytest.raises(CostModelError, match="Unknown operator"):
        CudaCostModel(hardware(), calibration(), "not_an_operator", {})
    with pytest.raises(TypeError, match="config"):
        gemm_model().predict(None)

    def broken(shape, config):
        raise RuntimeError("plugin boom")

    model = CudaCostModel(
        hardware(), calibration(), "broken", {}, operator_plugins={"broken": broken},
    )
    with pytest.raises(RuntimeError, match="plugin boom"):
        model.predict({})

    def invalid(shape, config):
        return KernelWork("invalid", 1, ())

    model = CudaCostModel(
        hardware(), calibration(), "invalid", {}, operator_plugins={"invalid": invalid},
    )
    with pytest.raises(CostModelError, match="at least one phase"):
        model.predict({})

    def nonfinite(shape, config):
        return KernelWork("nonfinite", 1, (PhaseWork(cuda_flops_per_cta=float("nan")),))

    model = CudaCostModel(
        hardware(), calibration(), "nonfinite", {}, operator_plugins={"nonfinite": nonfinite},
    )
    with pytest.raises(CostModelError, match="finite"):
        model.predict({})


def test_mma_fp16_bf16_tf32_use_distinct_calibrated_rates():
    cal = calibration(
        mma_f16_flops_per_second=2.0e9,
        mma_bf16_flops_per_second=4.0e9,
        mma_tf32_flops_per_second=8.0e9,
    )
    expected = {"f16": 0.008192, "bf16": 0.004096, "tf32": 0.002048}
    for dtype, expected_ms in expected.items():
        model = CudaCostModel(
            hardware(), cal, "gemm", {"M": 16, "N": 16, "K": 16, "dtype": dtype},
        )
        _, terms = model.predict({
            "block_M": 16, "block_N": 16, "block_K": 16,
            "registers_per_thread": 32,
        })
        # One CTA, 2*16^3=8192 FLOP; two SMs make per-SM rates
        # 1e9/2e9/4e9 respectively.  These are independent unit oracles.
        assert terms["mma_ms"] == pytest.approx(expected_ms)
        assert terms["mma_flops"] == 8192


def test_cuda_core_gemm_and_causal_fa_are_explicit_models():
    model = gemm_model()
    _, tensor = model.predict(gemm_config())
    _, cores = model.predict(gemm_config(use_mma=False))
    assert tensor["mma_flops"] == tensor["flops"]
    assert cores["mma_flops"] == 0
    assert cores["cuda_flops"] == tensor["flops"]

    fa = CudaCostModel(
        hardware(), calibration(), "flash_attention_fwd",
        {"B": 1, "H": 1, "S": 32, "D": 32, "dtype": "bf16"},
    )
    with pytest.raises(CostModelError, match="Unknown config"):
        fa.predict({"block_M": 16, "block_N": 16, "use_mma": False})
    _, causal = fa.predict({"block_M": 16, "block_N": 16, "causal": True, "registers_per_thread": 64})
    assert causal["useful_flops"] == 2 * 1 * 1 * 32 * 32 * 33
    assert causal["fallback_count"] >= 1
    with pytest.raises(CostModelError, match="seq_q == seq_kv"):
        CudaCostModel(
            hardware(), calibration(), "flash_attention_fwd",
            {"B": 1, "H": 1, "S": 32, "S_KV": 64, "D": 32, "dtype": "bf16"},
        ).predict({"block_M": 16, "block_N": 16, "causal": True})


def test_gdn_backward_state_io_follows_public_kernel_contract():
    shape = gdn_shape()
    config = {"prepare_h_num_stages": 2, "registers_per_thread": 128}
    without_dht = CudaCostModel(hardware(), calibration(), "gdn_bwd", shape)
    _, base = without_dht.predict(config)

    # prepare_h output_final_state is fixed False in the backward pipeline; the
    # forward-shaped flag must not silently add an ht store to this operator.
    no_final_shape = dict(shape, output_final_state=False)
    _, also_without_dht = CudaCostModel(hardware(), calibration(), "gdn_bwd", no_final_shape).predict(config)
    assert also_without_dht == base

    with_dht_shape = dict(shape, use_dht=True, output_final_state=False)
    _, with_dht = CudaCostModel(hardware(), calibration(), "gdn_bwd", with_dht_shape).predict(config)
    # B*H=2 CTAs; each adds one FP32 K*DV dht read and one dh0 write.
    expected_bytes = 2 * 2 * 128 * 128 * 4
    assert with_dht["l2_bytes"] - base["l2_bytes"] == expected_bytes
    assert with_dht["dram_bytes"] - base["dram_bytes"] == expected_bytes


def test_gdn_backward_real_space_all_nine_configs_are_finite():
    # Stage 3 exceeds the measured SM120 capacity, so use explicitly
    # synthetic larger shared limits to exercise all nine legal space points.
    relaxed_hardware = hardware(
        shared_memory_per_block_optin_bytes=262144,
        shared_memory_per_multiprocessor_bytes=262144,
    )
    model = CudaCostModel(relaxed_hardware, calibration(), "gdn_bwd", gdn_shape())
    seen = set()
    for stages in (1, 2, 3):
        for block in (8, 16, 32):
            score, terms = model.predict({
                "prepare_h_num_stages": stages,
                "group_reduce_block_size": block,
                "registers_per_thread": 128,
            })
            assert_finite_json_terms(score, terms)
            assert terms["gdn_prepare_h_num_stages"] == stages
            assert terms["gdn_group_reduce_block_size"] == block
            assert terms["launch_count"] == 5
            assert terms["gdn_reverse_cumsum_ms"] > 0
            seen.add((stages, block))
    assert seen == {(stages, block) for stages in (1, 2, 3) for block in (8, 16, 32)}


def test_gdn_group_reduce_grid_traffic_fragment_capacity_and_cost():
    model = CudaCostModel(hardware(), calibration(), "gdn_bwd", gdn_shape())
    expected_ctas = {8: 16, 16: 8, 32: 4}  # Includes separate dq/dk launches.
    costs = []
    for block, expected in expected_ctas.items():
        _, terms = model.predict({"group_reduce_block_size": block, "registers_per_thread": 128})
        assert terms["gdn_group_reduce_cta_count"] == expected
        assert terms["gdn_group_reduce_launch_count"] == 2
        assert terms["gdn_group_reduce_cuda_flops"] == 32768
        assert terms["gdn_group_reduce_l2_bytes"] == 98304
        assert terms["gdn_group_reduce_fragment_bytes"] == 655360
        assert terms["gdn_group_reduce_fragment_capacity_bytes_per_cta"] == 1024 * block
        assert terms["gdn_group_reduce_shared_memory_bytes_per_cta"] == 0
        assert terms["gdn_group_reduce_shared_bytes"] == 0
        assert terms["gdn_group_reduce_ms"] > 0
        costs.append(terms["gdn_group_reduce_ms"])
    assert len(set(costs)) > 1  # Formula sensitivity only, not a faster-than claim.

    with pytest.raises(CostModelError, match="fragment-capacity lower bound"):
        model.predict({"group_reduce_block_size": 32, "registers_per_thread": 1})


def test_gdn_group_reduce_is_absent_when_hg_equals_h():
    shape = dict(gdn_shape(), Hg=2)
    model = CudaCostModel(hardware(), calibration(), "gdn_bwd", shape)
    score, terms = model.predict({"group_reduce_block_size": 32, "registers_per_thread": 128})
    assert_finite_json_terms(score, terms)
    assert terms["launch_count"] == 3
    for name in (
        "gdn_group_reduce_ms", "gdn_group_reduce_cta_count", "gdn_group_reduce_launch_count",
        "gdn_group_reduce_cuda_flops", "gdn_group_reduce_l2_bytes",
        "gdn_group_reduce_fragment_bytes", "gdn_group_reduce_fragment_capacity_bytes_per_cta",
    ):
        assert terms[name] == 0, name
    assert terms["gdn_reverse_cumsum_ms"] > 0


def test_gdn_group_block_default_and_direction_validation():
    shape = dict(gdn_shape(), T=65)
    model = CudaCostModel(hardware(), calibration(), "gdn_bwd", shape)
    base_config = {"registers_per_thread": 128}
    default_prediction = model.predict(base_config)
    explicit_prediction = model.predict(dict(base_config, group_reduce_block_size=16))
    assert explicit_prediction == default_prediction
    _, terms = default_prediction
    assert terms["gdn_group_reduce_block_size"] == 16
    assert terms["gdn_group_reduce_cta_count"] == 10
    assert terms["fallback_count"] >= 1  # Padded default-16 tail remains explicit.

    for invalid in (0, -1, 7, 64):
        with pytest.raises(CostModelError, match="group_reduce_block_size"):
            model.predict({"group_reduce_block_size": invalid})
    for block in (8, 32):
        with pytest.raises(CostModelError, match="requires T divisible"):
            model.predict({"group_reduce_block_size": block})
    with pytest.raises(CostModelError, match=r"\{1, 2, 3\}"):
        model.predict({"prepare_h_num_stages": 4})

    fwd = CudaCostModel(hardware(), calibration(), "gdn_fwd", gdn_shape())
    with pytest.raises(CostModelError, match="applies to gdn_bwd, not gdn_fwd"):
        fwd.predict({"group_reduce_block_size": 16})


def test_gdn_reverse_cumsum_work_traffic_and_separate_capacities():
    # Hg==H removes GQA reduce so these totals isolate reverse cumsum.
    shape = dict(gdn_shape(), Hg=2)
    model = CudaCostModel(hardware(), calibration(), "gdn_bwd", shape)
    _, terms = model.predict({"registers_per_thread": 128})
    assert terms["gdn_reverse_cumsum_cta_count"] == 2  # B*ceil(T/32)=2, H stays in-kernel.
    assert terms["gdn_reverse_cumsum_launch_count"] == 1
    assert terms["gdn_reverse_cumsum_cuda_flops"] == 2 * (64 - 2)
    assert terms["gdn_reverse_cumsum_l2_bytes"] == 1024
    assert terms["gdn_reverse_cumsum_shared_bytes"] == 2048
    assert terms["gdn_reverse_cumsum_fragment_bytes"] == 4096
    assert terms["gdn_reverse_cumsum_shared_memory_bytes_per_cta"] == 384
    assert terms["gdn_reverse_cumsum_fragment_capacity_bytes_per_cta"] == 512
    assert terms["fragment_bytes"] == 4096
    assert terms["fragment_capacity_bytes_per_cta"] == 512
    assert terms["gdn_reverse_cumsum_ms"] > 0


def test_gdn_model_card_declares_backward_scope_and_forward_artifacts():
    model = CudaCostModel(hardware(), calibration(), "gdn_bwd", gdn_shape())
    card = model.model_card()
    scope = card["gdn_scope"]
    assert scope["direction"] == "backward"
    assert scope["kernels_in_program_order"] == [
        "prepare_h", "fused_bwd", "gqa_reduce(dq)", "gqa_reduce(dk)", "reverse_dg_cumsum",
    ]
    assert "provided by public forward" in scope["inputs"]
    assert "KKT is not rerun or charged" in scope["inputs"]
    assert scope["group_reduce_block_size"] == {
        "default": 16, "space": [8, 16, 32], "non_default_requires_T_divisible": True,
    }
    assert scope["block_DV"] == {"value": 128, "tunable": False}
    assert any("fragment traffic" in item for item in card["conditional_fallbacks"])
    assert any("torch.zeros_like" in item for item in card["conditional_fallbacks"])
    json.dumps(card, allow_nan=False)
