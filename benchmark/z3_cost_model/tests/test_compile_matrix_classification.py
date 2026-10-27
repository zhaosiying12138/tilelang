"""Regression tests for explicit compile-matrix skip classification."""

from types import SimpleNamespace

from benchmark.z3_cost_model import compile_matrix as cm


def _counts(**updates):
    counts = {name: 0 for name in cm.PYTEST_COUNT_NAMES}
    counts.update(updates)
    return counts


def _row():
    return SimpleNamespace(numeric_gate=False, layer="H1")


def test_opt_in_performance_skip_is_not_not_run_or_pass():
    text = (
        "------- Skipped 1 perf test(s). Re-run with --run-perf to include them. --------\n"
        "SKIPPED [1] testing/python/jit/test_tilelang_jit_gemm_cython.py:210: "
        "performance test skipped by default; pass --run-perf to include it\n"
        "1 skipped in 0.55s"
    )
    status, reason = cm.classify_result(
        _row(), action="run", exit_code=0, text=text, counts=_counts(skipped=1)
    )
    assert status == cm.SKIP_PERF
    assert "not counted as passed" in reason
    summary = cm.aggregate_results([
        {"status": status, "selected": True, "numeric_gate": False, "pytest": _counts(skipped=1)}
    ])
    assert summary["status_counts"][cm.SKIP_PERF] == 1
    assert summary["passed"] == 0
    assert summary["failed"] == 0
    assert summary["skipped_not_passed"] == 1
    assert summary["selected_not_run"] == 0


def test_all_skipped_arch_exit_five_is_arch_skip_not_infra():
    for text in (
        "SKIPPED [1] test_atom.py:322: Requires CUDA compute lt 11.0, but have 12.0",
        "SKIPPED [1] test_nvrtc.py:406: Test requires Hopper GPU (compute capability 9.0)",
        "SKIPPED [2] test_warp_sync.py:185: Run-time support for ROCm not present",
    ):
        status, reason = cm.classify_result(
            _row(), action="run", exit_code=5, text=text, counts=_counts()
        )
        assert status == cm.SKIP_ARCH
        assert "exit 5" in reason


def test_unknown_all_skipped_exit_five_remains_infra():
    status, _ = cm.classify_result(
        _row(), action="run", exit_code=5, text="SKIPPED unsupported fixture", counts=_counts(skipped=1)
    )
    assert status == cm.FAIL_INFRA


def test_import_failure_without_skip_line_is_environment_failure():
    status, _ = cm.classify_result(
        _row(), action="run", exit_code=4, text="ModuleNotFoundError: No module named 'fla.ops'", counts=_counts()
    )
    assert status == cm.FAIL_ENV


def test_unknown_zero_exit_skip_remains_not_run():
    status, reason = cm.classify_skip("SKIPPED unsupported fixture")
    assert status == cm.NOT_RUN
    assert "arch/dependency/perf" in reason
