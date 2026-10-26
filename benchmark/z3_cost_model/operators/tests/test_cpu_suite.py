"""Static and pure-CPU verification for the three operator modules."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
import textwrap
from functools import partial
from pathlib import Path

import pytest
import torch
import z3

from benchmark.z3_cost_model import operators
from benchmark.z3_cost_model.operators import convolution, flash_attention, gemm
from benchmark.z3_cost_model.operators._common import ConstraintModel, CorrectnessError
from tilelang.autotuner.space import Z3Pruner

MODULES = (gemm, convolution, flash_attention)
EXPECTED_COUNTS = {gemm: 288, convolution: 144, flash_attention: 54}
EXPECTED_DOMAINS = {
    gemm: {
        "block_M": [64, 128, 256],
        "block_N": [64, 128, 256],
        "block_K": [32, 64],
        "num_stages": [0, 1, 2, 3],
        "threads": [128, 256],
        "enable_rasterization": [False, True],
    },
    convolution: {
        "block_M": [64, 128, 256],
        "block_N": [64, 128, 256],
        "block_K": [32, 64],
        "num_stages": [0, 1, 2, 3],
        "threads": [128, 256],
    },
    flash_attention: {
        "block_M": [64, 128, 256],
        "block_N": [64, 128, 256],
        "num_stages": [1, 2, 3],
        "threads": [128, 256],
    },
}


def test_raw_counts_exact_domains_and_unique_ids():
    for module in MODULES:
        configs = module.raw_configs()
        assert len(configs) == module.RAW_COUNT == EXPECTED_COUNTS[module]
        assert EXPECTED_DOMAINS[module] == module.DOMAINS
        assert tuple(module.DOMAINS) == module.CONFIG_KEYS
        assert all(tuple(config) == module.CONFIG_KEYS for config in configs)
        assert len({module.config_space().config_id(config) for config in configs}) == len(configs)

        fresh = module.raw_configs()
        assert fresh == configs
        assert fresh is not configs
        assert fresh[0] is not configs[0]
        first_key = module.CONFIG_KEYS[0]
        fresh[0][first_key] = "mutated"
        assert configs[0][first_key] != "mutated"


def test_canonical_id_is_sorted_compact_sha256():
    for module in MODULES:
        config = module.raw_configs()[0]
        reordered = dict(reversed(tuple(config.items())))
        canonical = module.config_space().canonical_json(config)
        assert canonical == json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        assert module.config_space().canonical_json(reordered) == canonical
        assert module.config_space().config_id(reordered) == module.config_space().config_id(config)
        assert module.config_space().config_id(config) == hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_json_manifests_and_provenance():
    aggregate = operators.suite_manifest()
    assert [entry["operator"] for entry in aggregate["operators"]] == [module.OPERATOR for module in MODULES]
    for module, manifest in zip(MODULES, aggregate["operators"], strict=True):
        assert manifest == module.suite_manifest()
        assert {"operator", "shape", "dtype", "raw_count", "constraints", "provenance"} <= set(manifest)
        assert manifest["raw_count"] == EXPECTED_COUNTS[module]
        assert manifest["dtype"] == {"io": "float16", "accumulator": "float32", "reference": "float32"}
        assert manifest["default_pass_configs"] is None
        assert manifest["configuration_keys"] == list(module.CONFIG_KEYS)
        assert manifest["provenance"]["upstream_revision"]
        assert len(manifest["provenance"]["sources"]) == 2
        assert manifest["constraints"]["unknown"]
        json.dumps(manifest, allow_nan=False)

    manifest_path = Path(__file__).resolve().parents[1] / "manifest.json"
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == aggregate
    assert gemm.SHORT_K_SHAPE.to_dict() == {"M": 8192, "N": 8192, "K": 256}
    assert convolution.PRIMARY_SHAPE.OH == convolution.PRIMARY_SHAPE.OW == 17
    assert flash_attention.PRIMARY_SHAPE.to_dict() == {"B": 1, "H": 16, "S": 2048, "D": 64, "is_causal": True}


def test_every_config_key_is_consumed_by_kernel_ir_builder():
    for module in MODULES:
        tree = ast.parse(textwrap.dedent(inspect.getsource(module._build_prim_func)))
        used = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "cfg"
        }
        assert set(module.CONFIG_KEYS) <= used


def test_prim_func_factories_construct_on_cpu_without_compilation():
    for module, parameter_count in ((gemm, 3), (convolution, 3), (flash_attention, 4)):
        shapes = gemm.SHAPES if module is gemm else (module.PRIMARY_SHAPE,)
        configs = module.raw_configs()
        for shape in shapes:
            for config in (configs[0], configs[-1]):
                prim_func = module.kernel_factory(config, shape=shape, compile=False)
                assert type(prim_func).__name__ == "PrimFunc"
                assert len(prim_func.params) == parameter_count


def test_factory_rejects_unconsumed_keys_and_forwards_stock_or_optin_passes(monkeypatch):
    for module in MODULES:
        config = module.raw_configs()[0]
        missing = dict(config)
        missing.pop(module.CONFIG_KEYS[0])
        with pytest.raises(ValueError, match="missing"):
            module.kernel_factory(missing, compile=False)
        with pytest.raises(ValueError, match="extra"):
            module.kernel_factory({**config, "unused_key": 1}, compile=False)

        sentinel = object()
        compiled = object()
        calls = []

        def fake_build(shape, cfg, *, expected_config=config, prim_sentinel=sentinel):
            assert cfg.to_dict() == expected_config
            return prim_sentinel

        def fake_compile(
            prim_func,
            *,
            target,
            pass_configs,
            prim_sentinel=sentinel,
            call_records=calls,
            compiled_sentinel=compiled,
        ):
            assert prim_func is prim_sentinel
            call_records.append((target, pass_configs))
            return compiled_sentinel

        monkeypatch.setattr(module, "_build_prim_func", fake_build)
        monkeypatch.setattr(module, "compile_prim_func", fake_compile)
        assert module.kernel_factory(config) is compiled
        assert calls[-1] == ("auto", None)
        optin = {"example.explicit_pass": True}
        assert module.kernel_factory(config, target="sm_90", pass_configs=optin) is compiled
        assert calls[-1] == ("sm_90", optin)
        assert module.kernel_factory(config, compile=False) is sentinel
        with pytest.raises(ValueError, match="compile=True"):
            module.kernel_factory(config, compile=False, pass_configs=optin)


@pytest.mark.parametrize(
    "model",
    [
        ConstraintModel(),
        ConstraintModel(target="sm_80", source="static upstream source audit"),
        ConstraintModel(target="sm_90", source="static upstream source audit"),
    ],
)
def test_constraints_do_not_prune_raw_or_obvious_tail_configs(model):
    spaces = [gemm.config_space(shape=shape, model=model) for shape in gemm.SHAPES]
    spaces.extend((convolution.config_space(model=model), flash_attention.config_space(model=model)))
    for space in spaces:
        result = Z3Pruner().prune(space)
        assert result.audit["raw"] == len(space)
        assert result.audit["sat"] == len(space)
        assert result.audit["unsat"] == 0
        assert result.audit["unknown"] == 0
        assert result.configs == space.raw_configs()

    # Conv M=4*17*17=1156 and N=F=96 intentionally have tails for many tiles.
    tail_conv = {
        "block_M": 256,
        "block_N": 256,
        "block_K": 64,
        "num_stages": 3,
        "threads": 256,
    }
    assert tail_conv in convolution.config_space(model=model).raw_configs()


def test_no_global_tail_divisibility_or_stage_le_trip_heuristic():
    model = ConstraintModel(target="sm_80", source="static upstream source audit")
    gemm_config = {
        "block_M": 64,
        "block_N": 64,
        "block_K": 64,
        "num_stages": 3,
        "threads": 128,
        "enable_rasterization": False,
    }
    gemm_result = Z3Pruner().prune(
        [gemm_config],
        constraint_fn=partial(gemm.constraint_builder, shape=gemm.GemmShape(M=65, N=67, K=1), model=model),
    )
    assert gemm_result.configs == [gemm_config]
    assert gemm_result.audit["unsat"] == 0

    fa_config = {"block_M": 256, "block_N": 256, "num_stages": 3, "threads": 128}
    fa_result = Z3Pruner().prune(
        [fa_config],
        constraint_fn=partial(
            flash_attention.constraint_builder,
            shape=flash_attention.FAShape(B=1, H=1, S=17, D=64, is_causal=True),
            model=model,
        ),
    )
    assert fa_result.configs == [fa_config]
    assert fa_result.audit["unsat"] == 0


@pytest.mark.parametrize("module", MODULES)
def test_non_warp_thread_count_is_provably_unsat(module):
    config = module.raw_configs()[0]
    config["threads"] = 127
    result = Z3Pruner().prune([config], constraint_fn=module.constraint_builder)
    assert result.configs == []
    assert result.audit["unsat"] == 1
    assert result.audit["unknown"] == 0


def test_target_specific_atom_constraints_are_not_guessed_for_auto():
    config = {
        "block_M": 80,
        "block_N": 64,
        "block_K": 32,
        "num_stages": 0,
        "threads": 128,
        "enable_rasterization": False,
    }
    # SM90 selects WGMMA for this local shape, whose 80-row tile cannot be
    # partitioned into four 16-row warps. SM80's native-MMA partition can.
    hopper = Z3Pruner().prune([config], constraint_fn=partial(gemm.constraint_builder, model=ConstraintModel(target="sm_90")))
    ampere = Z3Pruner().prune([config], constraint_fn=partial(gemm.constraint_builder, model=ConstraintModel(target="sm_80")))
    automatic = Z3Pruner().prune([config], constraint_fn=gemm.constraint_builder)
    assert hopper.audit["unsat"] == 1
    assert ampere.configs == [config]
    assert automatic.configs == [config]


@pytest.mark.parametrize(
    ("module", "lower_bound"),
    [(gemm, 8192), (convolution, 8192), (flash_attention, 16384)],
)
def test_exact_necessary_shared_memory_lower_bound(module, lower_bound):
    config = module.raw_configs()[0]
    # First raw config uses block_M/N=64, block_K=32 where applicable.
    at_limit = ConstraintModel(
        shared_memory_per_block_limit_bytes=lower_bound,
        source="unit-test verified usable limit",
    )
    below_limit = ConstraintModel(
        shared_memory_per_block_limit_bytes=lower_bound - 1,
        source="unit-test verified usable limit",
    )
    kept = Z3Pruner().prune([config], constraint_fn=partial(module.constraint_builder, model=at_limit))
    pruned = Z3Pruner().prune([config], constraint_fn=partial(module.constraint_builder, model=below_limit))
    assert kept.configs == [config]
    assert pruned.configs == []
    assert pruned.audit["unsat"] == 1


def test_solver_unknown_is_retained_not_relabelled_unsat():
    class UnknownSolver:
        def __init__(self):
            self.expressions = ()

        def set(self, *_args):
            return None

        def add(self, *expressions):
            self.expressions = expressions

        def check(self):
            return z3.unknown

        def reason_unknown(self):
            return "unit test resource limit"

        def statistics(self):
            return {}

    config = gemm.raw_configs()[0]
    result = Z3Pruner(solver_factory=lambda _context: UnknownSolver(), z3_module=z3).prune([config], constraint_fn=gemm.constraint_builder)
    assert result.configs == [config]
    assert result.unknown == [config]
    assert result.audit["unsat"] == 0
    assert result.audit["unknown"] == 1


def test_gemm_fixed_seed_and_independent_fp32_reference():
    shape = gemm.GemmShape(M=3, N=4, K=5)
    first = gemm.make_inputs(shape=shape, device="cpu")
    second = gemm.make_inputs(shape=shape, device="cpu")
    different = gemm.make_inputs(shape=shape, device="cpu", seed=1)
    assert tuple(first) == ("A", "B")
    for name in first:
        assert first[name].dtype == torch.float16
        assert torch.equal(first[name], second[name])
        assert not torch.equal(first[name], different[name])

    actual = gemm.reference(first, shape=shape)
    naive = torch.empty((shape.M, shape.N), dtype=torch.float32)
    for m in range(shape.M):
        for n in range(shape.N):
            naive[m, n] = sum(float(first["A"][m, k]) * float(first["B"][n, k]) for k in range(shape.K))
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, naive, atol=1e-5, rtol=1e-5)


def test_convolution_nhwc_fp32_reference_with_padding_and_stride():
    shape = convolution.ConvShape(N=1, C=2, H=4, W=4, F=3, K=3, S=2, D=1, P=1)
    inputs = convolution.make_inputs(shape=shape, device="cpu", seed=7)
    actual = convolution.reference(inputs, shape=shape)
    naive = torch.empty((shape.N, shape.OH, shape.OW, shape.F), dtype=torch.float32)
    data = inputs["data"]
    weight = inputs["kernel"]
    for n in range(shape.N):
        for oh in range(shape.OH):
            for ow in range(shape.OW):
                for f in range(shape.F):
                    value = 0.0
                    for kh in range(shape.K):
                        for kw in range(shape.K):
                            ih = oh * shape.S + kh * shape.D - shape.P
                            iw = ow * shape.S + kw * shape.D - shape.P
                            if 0 <= ih < shape.H and 0 <= iw < shape.W:
                                for c in range(shape.C):
                                    value += float(data[n, ih, iw, c]) * float(weight[kh, kw, c, f])
                    naive[n, oh, ow, f] = value
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, naive, atol=1e-5, rtol=1e-5)


def test_flash_attention_fp32_reference_and_strict_lower_triangle():
    shape = flash_attention.FAShape(B=1, H=2, S=5, D=4, is_causal=True)
    inputs = flash_attention.make_inputs(shape=shape, device="cpu", seed=11)
    actual = flash_attention.reference(inputs, shape=shape)
    naive = torch.empty_like(actual)
    q, k, v = inputs["Q"], inputs["K"], inputs["V"]
    for b in range(shape.B):
        for h in range(shape.H):
            for i in range(shape.S):
                scores = []
                for j in range(i + 1):
                    dot = sum(float(q[b, i, h, d]) * float(k[b, j, h, d]) for d in range(shape.D))
                    scores.append(dot / math.sqrt(shape.D))
                maximum = max(scores)
                weights = [math.exp(score - maximum) for score in scores]
                normalizer = sum(weights)
                for d in range(shape.D):
                    naive[b, i, h, d] = sum(weights[j] * float(v[b, j, h, d]) for j in range(i + 1)) / normalizer
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, naive, atol=1e-4, rtol=1e-4)

    # Changing the final K/V row cannot affect any strictly earlier query.
    modified = dict(inputs)
    modified["K"] = inputs["K"].clone()
    modified["V"] = inputs["V"].clone()
    modified["K"][:, -1] += 3
    modified["V"][:, -1] -= 2
    changed = flash_attention.reference(modified, shape=shape)
    torch.testing.assert_close(actual[:, :-1], changed[:, :-1], atol=0, rtol=0)


@pytest.mark.parametrize("module", MODULES)
def test_strict_validation_records_errors_and_allows_zero_mismatches(module):
    expected = torch.tensor([[0.0, 1.0, -2.0]], dtype=torch.float32)
    config = module.raw_configs()[0]
    report = module.validate(expected.half(), expected, config=config, atol=0, rtol=0)
    assert report.passed
    assert report.max_mismatched_ratio == 0
    assert report.mismatched_elements == 0
    assert report.mismatched_ratio == 0
    assert report.max_abs_error == 0
    assert report.max_rel_error == 0
    assert report.config_id == module.config_space().config_id(config)

    wrong = expected.half()
    wrong[0, 1] += 0.5
    with pytest.raises(CorrectnessError) as caught:
        module.validate(wrong, expected, config=config, atol=0, rtol=0)
    failure = caught.value.report
    assert not failure.passed
    assert failure.mismatched_elements == 1
    assert failure.mismatched_ratio == 1 / 3
    assert failure.max_mismatched_ratio == 0
    assert failure.max_abs_error == 0.5
    assert failure.max_rel_error > 0

    wrong[0, 1] = float("nan")
    with pytest.raises(CorrectnessError):
        module.validate(wrong, expected, config=config)
