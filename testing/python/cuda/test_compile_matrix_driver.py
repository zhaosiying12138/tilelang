"""Pure-CPU tests for the frozen compile-matrix driver.

No test in the driver suite launches a GPU kernel.  The dedicated Z3 compile
fixture is inert unless the runner supplies both its mode and explicit JSON
pass_configs; ordinary pytest execution skips it before importing TileLang.
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER_PATH = REPO_ROOT / "benchmark" / "z3_cost_model" / "compile_matrix.py"
MANIFEST_PATH = REPO_ROOT / "manifest" / "compile_matrix.json"
_SPEC = importlib.util.spec_from_file_location("_tilelang_compile_matrix_driver", RUNNER_PATH)
cm = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = cm
_SPEC.loader.exec_module(cm)


@pytest.fixture(scope="module")
def frozen():
    return cm.load_manifest(MANIFEST_PATH)


def _row(frozen, *, role="operator", layer=None):
    return next(row for row in frozen.rows if row.role == role and (layer is None or row.layer == layer))


def _record(status, *, selected=True, numeric=False, attempted=False, passed=0, collected=0):
    return {
        "status": status,
        "selected": selected,
        "attempted": attempted,
        "numeric_gate": numeric,
        "pytest": {"passed": passed, "collected": collected},
    }


def test_manifest_counts_and_tier_partition(frozen):
    quick = frozen.tier_rows("quick")
    full = frozen.tier_rows("full")
    host = frozen.tier_rows("host")
    assert (len(quick), len(full), len(host)) == (126, 729, 77)
    assert Counter(row.role for row in quick) == {"operator": 49, "host": 77}
    assert Counter(row.role for row in full) == {"operator": 651, "host": 77, "z3_fixture": 1}
    assert Counter(row.role for row in host) == {"host": 77}
    full_operator_ids = {row.id for row in full if row.role == "operator"}
    full_operator_rows = [row for row in full if row.role == "operator"]
    assert sum(row.path.startswith("testing/python/") for row in full_operator_rows) == 627
    assert sum(row.path.startswith("examples/") for row in full_operator_rows) == 24
    assert {row.id for row in quick if row.role == "operator"} <= full_operator_ids


def test_manifest_every_frozen_node_exists_in_current_source(frozen):
    functions_by_path = {}
    for row in frozen.rows:
        if row.path not in functions_by_path:
            source = (REPO_ROOT / row.path).read_text(encoding="utf-8")
            tree = ast.parse(source, filename=row.path)
            functions_by_path[row.path] = {
                node.name
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test")
            }
        assert row.node in functions_by_path[row.path], row.id
        assert row.target == row.id


def test_gdn_is_smoke_and_fa_fla_are_independent(frozen):
    gdn = [row for row in frozen.rows if row.path == "examples/gdn/test_example_gdn_compilation.py"]
    assert len(gdn) == 8
    assert all(row.layer == "G" and not row.numeric_gate for row in gdn)
    assert all("fla" not in row.dependencies and "fla" in row.optional_dependencies for row in gdn)
    fa_rows = {row.id for row in frozen.rows if "flash-attn" in row.dependencies}
    fla_rows = {row.id for row in frozen.rows if "fla" in row.dependencies}
    assert fa_rows and fla_rows and not fa_rows & fla_rows
    assert frozen.data["dependencies"]["flash-attn"]["torch_requirement"] == "<2.14"
    assert frozen.data["dependencies"]["fla"]["independent_from"] == "flash-attn"
    assert frozen.data["dependency_policy"]["install"] is False


def test_status_aggregation_does_not_count_collection_skip_or_smoke_as_numeric():
    records = [
        _record(cm.COLLECTED, collected=3),
        _record(cm.PASS_COMPILE_ONLY, attempted=True, passed=1),
        _record(cm.PASS_NUMERIC, numeric=True, attempted=True, passed=1),
        _record(cm.G_SMOKE, attempted=True, passed=1),
        _record(cm.SKIP_ARCH, numeric=True),
        _record(cm.SKIP_DEP),
        _record(cm.NOT_RUN),
        _record(cm.FAIL_ENV, attempted=True),
    ]
    summary = cm.aggregate_results(records)
    assert summary["denominator"] == 8
    assert summary["passed"] == 2
    assert summary["numeric_passed"] == 1
    assert summary["g_smoke_not_numeric"] == 1
    assert summary["collected_not_passed"] == 1
    assert summary["skipped_not_passed"] == 2
    assert summary["not_run_not_passed"] == 1
    assert summary["failed"] == 1
    assert summary["numeric_gate"] == {"selected_eligible": 2, "passed": 1, "not_satisfied": 1}


def test_only_preserves_frozen_denominator_and_marks_remainder_not_run(frozen):
    rows = frozen.tier_rows("host")
    selected_id = rows[0].id
    selection = cm.select_rows(rows, action="collect", mode="default", only=[selected_id])
    assert len(selection) == 77
    assert sum(selected for _, selected, _ in selection) == 1
    records = [
        _record(cm.COLLECTED if selected else cm.NOT_RUN, selected=selected, collected=int(selected))
        for _, selected, _ in selection
    ]
    summary = cm.aggregate_results(records)
    assert summary["denominator"] == 77
    assert summary["selected"] == 1
    assert summary["collected_not_passed"] == 1
    assert summary["not_run_not_passed"] == 76
    assert summary["passed"] == 0


def test_only_is_exact_and_rejects_fuzzy_selection(frozen):
    rows = frozen.tier_rows("host")
    with pytest.raises(cm.MatrixError, match="wildcard"):
        cm.select_rows(rows, action="collect", mode="default", only=["*scheduler*"])
    with pytest.raises(cm.MatrixError, match="did not match"):
        cm.select_rows(rows, action="collect", mode="default", only=["test_straight_line"])


def test_collect_and_run_commands_use_same_frozen_node(frozen):
    row = _row(frozen)
    collect = cm.build_command(row, action="collect", mode="default", python="python")
    run = cm.build_command(row, action="run", mode="default", python="python")
    assert collect[-1] == run[-1] == row.target
    assert "--collect-only" in collect and "--collect-only" not in run
    assert "--noconftest" in collect and "--noconftest" not in run
    assert "-k" not in collect and "-k" not in run
    assert collect.count(row.target) == run.count(row.target) == 1
    assert collect[:3] == ["python", "-m", "pytest"]
    assert "no:cacheprovider" in collect


def test_z3_opt_in_rejects_ordinary_nodes_and_default_rejects_fixture(frozen):
    ordinary = next(row for row in frozen.tier_rows("quick") if row.role == "operator")
    fixture = _row(frozen, role="z3_fixture")
    with pytest.raises(cm.MatrixError, match="pass_configs"):
        cm.build_command(ordinary, action="run", mode="z3-opt-in")
    with pytest.raises(cm.MatrixError, match="ordinary nodes"):
        cm.select_rows(
            frozen.tier_rows("quick"), action="run", mode="z3-opt-in", only=[ordinary.id]
        )
    with pytest.raises(cm.MatrixError, match="requires z3-opt-in"):
        cm.build_command(fixture, action="run", mode="default")
    with pytest.raises(cm.MatrixError, match="dedicated z3 fixtures"):
        cm.select_rows(
            frozen.tier_rows("full"), action="run", mode="default", only=[fixture.id]
        )


def test_z3_opt_in_without_only_selects_only_dedicated_fixture(frozen):
    selection = cm.select_rows(frozen.tier_rows("full"), action="run", mode="z3-opt-in")
    selected = [row for row, enabled, _ in selection if enabled]
    assert len(selection) == 729
    assert len(selected) == 1 and selected[0].role == "z3_fixture"
    assert all(row.role == "z3_fixture" or not enabled for row, enabled, _ in selection)


def test_default_environment_scrubs_pretend_global_opt_in(frozen, tmp_path):
    row = _row(frozen)
    output = cm.prepare_output(tmp_path / "results", REPO_ROOT)
    cache = cm.isolated_cache_directory(output, "a" * 64)
    env = cm.build_environment(
        row,
        mode="default",
        cache_dir=cache,
        repo_root=REPO_ROOT,
        base_env={
            cm.PASS_CONFIG_ENV: '{"tl.enable_z3_pipeline_planning":true}',
            "PYTEST_ADDOPTS": "-k something",
        },
    )
    assert cm.PASS_CONFIG_ENV not in env
    assert "PYTEST_ADDOPTS" not in env
    assert env[cm.MODE_ENV] == "default"
    for key in (
        "TILELANG_CACHE_DIR",
        "TORCH_EXTENSIONS_DIR",
        "TRITON_CACHE_DIR",
        "CUDA_CACHE_PATH",
        "XDG_CACHE_HOME",
        "TMPDIR",
    ):
        assert Path(env[key]).resolve().is_relative_to(cache)


def test_z3_environment_is_explicit_and_limited_to_fixture(frozen, tmp_path):
    ordinary = _row(frozen)
    fixture = _row(frozen, role="z3_fixture")
    output = cm.prepare_output(tmp_path / "results", REPO_ROOT)
    cache = cm.isolated_cache_directory(output, "b" * 64)
    with pytest.raises(cm.MatrixError, match="ordinary pytest"):
        cm.build_environment(ordinary, mode="z3-opt-in", cache_dir=cache, repo_root=REPO_ROOT, base_env={})
    env = cm.build_environment(
        fixture, mode="z3-opt-in", cache_dir=cache, repo_root=REPO_ROOT, base_env={}
    )
    assert json.loads(env[cm.PASS_CONFIG_ENV]) == {cm.Z3_PIPELINE_CONFIG: True}
    assert env[cm.MODE_ENV] == "z3-opt-in"
    assert env[cm.ROW_ENV] == fixture.id


def test_cache_identity_is_stable_per_node_and_mode(frozen):
    first = _row(frozen)
    second = next(row for row in frozen.rows if row.id != first.id)
    kwargs = {"manifest_sha256": "m" * 64, "source_sha256": "s" * 64, "python": sys.executable}
    one = cm.build_cache_identity(first, mode="default", **kwargs)
    assert one == cm.build_cache_identity(first, mode="default", **kwargs)
    assert one != cm.build_cache_identity(second, mode="default", **kwargs)
    fixture = _row(frozen, role="z3_fixture")
    assert cm.build_cache_identity(fixture, mode="default", **kwargs) != cm.build_cache_identity(
        fixture, mode="z3-opt-in", **kwargs
    )


def test_cache_guard_rejects_escape_and_shared_cache(tmp_path):
    output = cm.prepare_output(tmp_path / "results", REPO_ROOT)
    with pytest.raises(cm.MatrixError, match="repository/home"):
        cm.prepare_output(REPO_ROOT, REPO_ROOT)
    with pytest.raises(cm.MatrixError, match="cache identity"):
        cm.isolated_cache_directory(output, "../shared")
    shared = output / "cache"
    old = os.environ.get("TILELANG_CACHE_DIR")
    os.environ["TILELANG_CACHE_DIR"] = str(shared)
    try:
        with pytest.raises(cm.MatrixError, match="inherited shared"):
            cm.prepare_output(output, REPO_ROOT)
    finally:
        if old is None:
            os.environ.pop("TILELANG_CACHE_DIR", None)
        else:
            os.environ["TILELANG_CACHE_DIR"] = old


def test_each_node_gets_distinct_cache_and_no_shared_clear(frozen, tmp_path):
    output = cm.prepare_output(tmp_path / "results", REPO_ROOT)
    rows = frozen.tier_rows("host")[:2]
    caches = [cm.isolated_cache_directory(output, str(index + 1) * 64) for index in range(2)]
    assert caches[0] != caches[1]
    assert all(cache.parent == output / "cache" for cache in caches)
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "rmtree" not in source
    assert "rm -rf" not in source
    assert "pip install" not in source
    assert "clear_cache" not in source
    assert rows[0].id != rows[1].id


@pytest.mark.parametrize(
    ("diagnostic", "numeric", "expected"),
    [
        ("ModuleNotFoundError: No module named 'fla'", False, cm.FAIL_ENV),
        ("ptxas error : illegal register", False, cm.FAIL_DEVICE_COMPILE),
        ("CUDA_ERROR_MISALIGNED_ADDRESS at kernel launch", True, cm.FAIL_LAUNCH),
        ("Failed to set the allowed dynamic shared memory size to 132352", True, cm.FAIL_LAUNCH),
        ("RuntimeError: #include <tl_templates/cuda/nvrtc_std.h>\nCompilation error:", True, cm.FAIL_DEVICE_COMPILE),
        ("torch.testing.assert_close: Tensor-likes are not close", True, cm.FAIL_NUMERIC),
        ("tvm.error.InternalError in LowerTileOp", False, cm.FAIL_LOWER),
        ("AssertionError: generated source differs", False, cm.FAIL_LOWER),
        ("AssertionError: values differ", True, cm.FAIL_NUMERIC),
    ],
)
def test_failure_classification(diagnostic, numeric, expected):
    assert cm.classify_failure(diagnostic, numeric_gate=numeric) == expected


def test_collect_timeout_and_empty_run_are_not_passes(frozen):
    row = _row(frozen)
    collected = {name: 0 for name in cm.PYTEST_COUNT_NAMES}
    collected["collected"] = 2
    status, _ = cm.classify_result(row, action="collect", exit_code=0, text="", counts=collected)
    assert status == cm.COLLECTED
    assert cm.aggregate_results([_record(status, collected=2)])["passed"] == 0
    status, _ = cm.classify_result(
        row, action="run", exit_code=124, text="", counts=collected, timed_out=True
    )
    assert status == cm.FAIL_INFRA
    empty = {name: 0 for name in cm.PYTEST_COUNT_NAMES}
    status, _ = cm.classify_result(row, action="run", exit_code=0, text="", counts=empty)
    assert status == cm.FAIL_INFRA


def test_partial_skip_cannot_be_promoted_to_numeric_pass(frozen):
    row = _row(frozen, layer="P")
    counts = {name: 0 for name in cm.PYTEST_COUNT_NAMES}
    counts.update(passed=1, skipped=1)
    status, _ = cm.classify_result(
        row,
        action="run",
        exit_code=0,
        text="SKIPPED [1] Requires CUDA compute ge 9.0",
        counts=counts,
    )
    assert status == cm.SKIP_ARCH
    counts["skipped"] = 0
    status, _ = cm.classify_result(row, action="run", exit_code=0, text="", counts=counts)
    assert status == cm.PASS_NUMERIC


def test_gdn_zero_exit_is_g_smoke_not_numeric(frozen):
    row = _row(frozen, layer="G")
    counts = {name: 0 for name in cm.PYTEST_COUNT_NAMES}
    counts["passed"] = 1
    status, _ = cm.classify_result(row, action="run", exit_code=0, text="", counts=counts)
    assert status == cm.G_SMOKE
    summary = cm.aggregate_results([_record(status, attempted=True, passed=1)])
    assert summary["passed"] == 0
    assert summary["g_smoke_not_numeric"] == 1


def test_skip_classification_is_conservative():
    assert cm.classify_skip("SKIPPED Requires CUDA compute ge 9.0")[0] == cm.SKIP_ARCH
    assert cm.classify_skip("SKIPPED pytest.importorskip('fla')")[0] == cm.SKIP_DEP
    assert cm.classify_skip("SKIPPED unsupported fixture detail")[0] == cm.NOT_RUN


def test_pytest_summary_parser_keeps_case_counts_separate():
    counts = cm.parse_pytest_summary("collected 3 items\n2 passed, 1 skipped, 1 warning in 0.1s")
    assert counts["collected"] == 3
    assert counts["passed"] == 2
    assert counts["skipped"] == 1
    assert counts["warnings"] == 1


def test_arch_lane_matching_and_explicit_override():
    assert cm.parse_arch("sm_120") == 120
    assert cm.parse_arch("12.0") == 120
    row = replace(_row_without_manifest(), arch_lane="sm90")
    lane = {"requires_device": True, "exact": 90}
    assert cm.arch_skip_reason(row, lane, 120) is not None
    assert cm.arch_skip_reason(row, lane, 90) is None
    assert cm.arch_skip_reason(row, lane, None) is not None


def _row_without_manifest():
    return cm.Row(
        id="example.py::test_x",
        group_id="example",
        path="example.py",
        node="test_x",
        role="operator",
        tiers=("full",),
        layer="H1",
        arch_lane="host",
        dependencies=(),
        optional_dependencies=(),
        numeric_gate=False,
    )


def test_dependency_checker_does_not_substitute_fla_for_fa(frozen):
    checker = cm.DependencyChecker(frozen.data, REPO_ROOT)
    checker._cache = {
        "flash-attn": {"name": "flash-attn", "available": False},
        "fla": {"name": "fla", "available": True},
    }
    row = replace(_row_without_manifest(), dependencies=("flash-attn",))
    missing, _ = checker.check(row)
    assert missing == ["missing required dependency flash-attn"]
    row = replace(row, dependencies=("fla",))
    assert checker.check(row)[0] == []


def test_parallelism_is_limited_to_host_collection(frozen, tmp_path):
    kwargs = {
        "matrix": frozen,
        "repo_root": REPO_ROOT,
        "tier": "quick",
        "mode": "default",
        "only": [],
        "timeout": 1,
        "output": tmp_path / "results",
        "collect_workers": 2,
    }
    with pytest.raises(cm.MatrixError, match="CPU-only host"):
        cm.execute_matrix(action="collect", **kwargs)
    with pytest.raises(cm.MatrixError, match="always serial"):
        cm.execute_matrix(action="run", **kwargs)


def test_shell_wrapper_delegates_without_fuzzy_pytest_selection():
    script = (REPO_ROOT / "scripts" / "run_compile_matrix.sh").read_text(encoding="utf-8")
    assert 'compile_matrix.py" "$@"' in script
    assert " -k " not in script
    assert "pytest" not in script
    assert "pip install" not in script


def test_z3_default_off_is_a_source_structural_contract(frozen):
    pipeline = (REPO_ROOT / "tilelang" / "cuda" / "pipeline.py").read_text(encoding="utf-8")
    native = (REPO_ROOT / "src" / "transform" / "z3_pipeline_planning.cc").read_text(encoding="utf-8")
    assert "pass_ctx.config.get(_Z3_PIPELINE_CONFIG, False)" in pipeline
    assert "GetConfig<Bool>(kZ3PipelineConfig, Bool(false))" in native
    assert "HasZ3PipelineAnnotation" in native
    assert _row(frozen).z3_opt_in is None


def test_z3_opt_in_pass_configs_compile_fixture():
    """Dedicated future integration point; compile only, never launch a kernel."""

    if os.environ.get(cm.MODE_ENV) != "z3-opt-in":
        pytest.skip("dedicated fixture runs only through compile_matrix.py --mode z3-opt-in")
    raw = os.environ.get(cm.PASS_CONFIG_ENV)
    if raw is None:
        pytest.fail("z3-opt-in fixture requires explicit JSON pass_configs")
    pass_configs = json.loads(raw)
    if pass_configs != {cm.Z3_PIPELINE_CONFIG: True}:
        pytest.fail(f"unexpected pass_configs for this fixture: {pass_configs!r}")

    import tilelang
    from tilelang import language as T
    from tilelang.language.annotations import z3_pipeline_annotations

    @T.prim_func
    def annotated_gemm(
        A: T.Tensor((64, 128), "float16"),
        B: T.Tensor((128, 64), "float16"),
        C: T.Tensor((64, 64), "float32"),
    ):
        with T.Kernel(1, threads=128):
            a_shared = T.alloc_shared((64, 64), "float16")
            b_shared = T.alloc_shared((64, 64), "float16")
            c_local = T.alloc_fragment((64, 64), "float32")
            T.clear(c_local)
            annotations = z3_pipeline_annotations(
                latencies=[1, 1, 1],
                iis=[1, 1, 1],
                resource_flags=[1, 1, 2],
                data_deps=[(0, 2, 0, 1), (1, 2, 0, 1)],
                resource_deps=[(0, 1)],
            )
            for k in T.Pipelined(2, num_stages=2, annotations=annotations):
                T.copy(A[:, k * 64 : (k + 1) * 64], a_shared)
                T.copy(B[k * 64 : (k + 1) * 64, :], b_shared)
                T.gemm(a_shared, b_shared, c_local)
            T.copy(c_local, C)

    kernel = tilelang.compile(
        annotated_gemm,
        target={"kind": "cuda", "arch": "sm_80"},
        out_idx=[2],
        pass_configs=pass_configs,
    )
    assert kernel.get_kernel_source()
