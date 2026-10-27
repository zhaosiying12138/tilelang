"""Complete-compile determinism evidence for opt-in CUDA Z3 planning.

Every snapshot is a fresh ``tilelang.compile`` with the kernel cache disabled
and a run-owned cache directory.  No compiled kernel is launched.
"""

from __future__ import annotations

from contextlib import contextmanager
import difflib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import tilelang
import tilelang.language as T
from tilelang.cuda.transform import z3_scheduler
from tilelang.engine.param import dump_kernel_params
from tilelang.language.annotations import z3_pipeline_annotations


_TARGET = {"kind": "cuda", "arch": "sm_120a"}
_CONFIG_KEY = tilelang.PassConfigKey.TL_ENABLE_Z3_PIPELINE_PLANNING
_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXPECTED_SOLVES = {"default_off": 0, "multi_stage": 1, "bind_clone": 2}
_VOLATILE_SCHEDULE_KEYS = {"timestamp", "schedule_directory"}


def _multi_stage_kernel(debug_dir: Path):
    block = 128
    iterations = 4

    @T.prim_func
    def main(
        A: T.Tensor((block * iterations,), "float32"),
        B: T.Tensor((block * iterations,), "float32"),
        C: T.Tensor((block * iterations,), "float32"),
    ):
        with T.Kernel(1, threads=128):
            a_shared = T.alloc_shared((block,), "float32")
            b_shared = T.alloc_shared((block,), "float32")
            c_fragment = T.alloc_fragment((block,), "float32")
            annotations = z3_pipeline_annotations(
                latencies=[4, 4, 2, 2],
                iis=[1, 1, 1, 1],
                resource_flags=[1, 1, 16, 2],
                data_deps=[(0, 2, 0, 4), (1, 2, 0, 4), (2, 3, 0, 2)],
                resource_deps=[(0, 1)],
                enable_offset=False,
                verbose=True,
                debug_dir=str(debug_dir),
            )
            for k in T.Pipelined(iterations, num_stages=2, annotations=annotations):
                T.copy(A[k * block : (k + 1) * block], a_shared)
                T.copy(B[k * block : (k + 1) * block], b_shared)
                for i in T.Parallel(block):
                    c_fragment[i] = a_shared[i] + 2 * b_shared[i]
                T.copy(c_fragment, C[k * block : (k + 1) * block])

    return main


def _bind_clone_kernel(debug_dir: Path):
    """A cross-stage Bind clone with next-iteration source overwrites."""

    block = 128
    iterations = 12
    source_nbytes = block * 4

    @T.prim_func
    def main(
        A: T.Tensor((block * iterations,), "float32"),
        B: T.Tensor((block * iterations,), "float32"),
    ):
        with T.Kernel(1, threads=1):
            source = T.alloc_shared((block,), "float32")
            annotations = z3_pipeline_annotations(
                latencies=[30, 2, 1],
                iis=[1, 1, 1],
                resource_flags=[0, 0, 0],
                data_deps=[(0, 1, 0, 30), (1, 0, -1, 2)],
                buffer_sizes=[source_nbytes],
                buffer_vars=[source],
                memory_groups=[[2 * source_nbytes, 0]],
                enable_offset=True,
                manual_stages=[0, 0, 1],
                manual_schedule=True,
                recalculate_buffer_versions=True,
                verbose=True,
                debug_dir=str(debug_dir),
            )
            for k in T.Pipelined(iterations, num_stages=3, annotations=annotations):
                T.copy(A[k * block : (k + 1) * block], source)
                value: T.float32 = source[0] + 1.0
                for i in T.Parallel(block):
                    B[k * block + i] = value + i

    return main


@contextmanager
def _isolated_compile_environment(run_root: Path):
    cache_dir = run_root / "cache"
    debug_dir = run_root / "schedule"
    # Import-time cache initialization may already have made this empty root.
    cache_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=False)
    overrides = {
        "TILELANG_CACHE_DIR": str(cache_dir),
        "TILELANG_DISABLE_CACHE": "1",
        "TILELANG_PRINT_ON_COMPILATION": "0",
        "MPLBACKEND": "Agg",
    }
    previous = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield cache_dir, debug_dir
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _normalize_text(text: str, run_root: Path) -> str:
    """Remove only run-owned paths and platform newline differences."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    for path in (str(run_root), run_root.as_posix()):
        normalized = normalized.replace(path, "<RUN_ROOT>")
    return normalized


def _canonical_schedule(value, run_root: Path):
    """Keep semantic schedule fields; drop timestamps and directory names."""

    if isinstance(value, dict):
        return {key: _canonical_schedule(item, run_root) for key, item in value.items() if key not in _VOLATILE_SCHEDULE_KEYS}
    if isinstance(value, list):
        return [_canonical_schedule(item, run_root) for item in value]
    if isinstance(value, str):
        return _normalize_text(value, run_root)
    return value


def _read_canonical_schedules(debug_dir: Path, run_root: Path):
    schedule_files = sorted(
        debug_dir.glob("schedule_*/schedule_info.json"),
        key=lambda path: int(path.parent.name.removeprefix("schedule_")),
    )
    return [_canonical_schedule(json.loads(path.read_text(encoding="utf-8")), run_root) for path in schedule_files]


def _compile_snapshot(case: str, run_root: Path):
    """Run one uncached complete compilation and return path-free artifacts."""

    assert case in _EXPECTED_SOLVES
    run_root = run_root.resolve()
    solve_results = []
    original_solve = z3_scheduler.z3_schedule_loop_python

    def recording_solve(*args, **kwargs):
        starts, versions, minimal_ii = original_solve(*args, **kwargs)
        solve_results.append(
            {
                "start_times": [int(value) for value in starts],
                "buffer_versions": [int(value) for value in versions],
                "minimal_II": int(minimal_ii),
            }
        )
        return starts, versions, minimal_ii

    z3_scheduler.z3_schedule_loop_python = recording_solve
    try:
        with _isolated_compile_environment(run_root) as (cache_dir, debug_dir):
            if case == "bind_clone":
                func = _bind_clone_kernel(debug_dir)
                out_idx = [1]
            else:
                func = _multi_stage_kernel(debug_dir)
                out_idx = [2]

            compile_kwargs = {}
            if case != "default_off":
                # The stock control deliberately omits this option entirely.
                compile_kwargs["pass_configs"] = {_CONFIG_KEY: True}
            kernel = tilelang.compile(
                func,
                target=dict(_TARGET),
                out_idx=out_idx,
                execution_backend="tvm_ffi",
                **compile_kwargs,
            )

            schedules = _read_canonical_schedules(debug_dir, run_root)
            snapshot = {
                "case": case,
                "schedules": schedules,
                "solver_results": solve_results,
                "metadata": {
                    "target": str(kernel.artifact.target),
                    "target_host": str(kernel.artifact.target_host),
                    "execution_backend": kernel.execution_backend,
                    "parameters": json.loads(dump_kernel_params(kernel.artifact.params)),
                    "device_functions": sorted(str(name) for name in kernel.artifact.device_mod.functions),
                },
                "device_tir": _normalize_text(kernel.artifact.device_mod.script(), run_root),
                "host_source": _normalize_text(kernel.get_host_source(), run_root),
                "cuda": _normalize_text(kernel.artifact.kernel_source, run_root),
            }
            assert str(cache_dir) not in json.dumps(snapshot, sort_keys=True)
    finally:
        z3_scheduler.z3_schedule_loop_python = original_solve

    _validate_snapshot(snapshot)
    return snapshot


def _validate_snapshot(snapshot):
    case = snapshot["case"]
    schedules = snapshot["schedules"]
    solver_results = snapshot["solver_results"]
    expected_solves = _EXPECTED_SOLVES[case]
    assert len(solver_results) == expected_solves
    assert len(schedules) == expected_solves
    assert "sm_120a" in snapshot["metadata"]["target"]
    assert snapshot["metadata"]["execution_backend"] == "tvm_ffi"
    assert "__global__" in snapshot["cuda"]
    assert snapshot["device_tir"].strip()
    assert snapshot["host_source"].strip()

    if case == "default_off":
        assert not schedules, "Stock default-off compilation must not emit Z3 schedules"
        return

    for schedule, result in zip(schedules, solver_results):
        assert schedule["fallback"] is False
        assert schedule["minimal_II"] == result["minimal_II"]
        assert schedule["start_times"] == result["start_times"]
        assert schedule["buffer_versions"] == result["buffer_versions"]

    if case == "multi_stage":
        assert solver_results == [
            {
                "start_times": [1, 2, 6, 8],
                "buffer_versions": [],
                "minimal_II": 2,
            }
        ]
        assert "tl::cp_async_gs<4>" in snapshot["cuda"]
    else:
        assert [schedule["num_tasks"] for schedule in schedules] == [3, 4]
        assert schedules[-1]["buffer_versions"] == [2]
        assert len(schedules[-1]["start_times"]) == 4


def _assert_artifacts_equal(reference, actual, label: str):
    """Fail on the first semantic artifact difference with a focused diff."""

    assert reference["case"] == actual["case"]
    for key in ("schedules", "solver_results", "metadata"):
        expected_text = json.dumps(reference[key], indent=2, sort_keys=True)
        actual_text = json.dumps(actual[key], indent=2, sort_keys=True)
        if expected_text != actual_text:
            _raise_text_difference(key, expected_text, actual_text, label)
    for key in ("device_tir", "host_source", "cuda"):
        if reference[key] != actual[key]:
            _raise_text_difference(key, reference[key], actual[key], label)


def _raise_text_difference(artifact: str, expected: str, actual: str, label: str):
    diff = list(
        difflib.unified_diff(
            expected.splitlines(),
            actual.splitlines(),
            fromfile="reference",
            tofile=label,
            lineterm="",
            n=3,
        )
    )
    if len(diff) > 100:
        diff = [*diff[:100], f"... {len(diff) - 100} diff lines omitted ..."]
    raise AssertionError(f"Complete compilation is nondeterministic in {artifact} ({label}):\n" + "\n".join(diff))


def _subprocess_snapshot(case: str, hash_seed: str, run_root: Path):
    output_path = run_root.parent / f"{run_root.name}-snapshot.json"
    env = os.environ.copy()
    env.update(
        {
            "PYTHONHASHSEED": hash_seed,
            "TILELANG_CACHE_DIR": str(run_root / "cache"),
            "TILELANG_DISABLE_CACHE": "1",
            "TILELANG_PRINT_ON_COMPILATION": "0",
            "MPLBACKEND": "Agg",
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--compile-worker",
            case,
            str(run_root),
            str(output_path),
        ],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=240,
    )
    assert completed.returncode == 0, (
        f"Compile worker failed for {case}, PYTHONHASHSEED={hash_seed}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    return json.loads(output_path.read_text(encoding="utf-8"))


def _exercise_complete_compile_determinism(case: str, tmp_path: Path):
    same_process = [_compile_snapshot(case, tmp_path / f"{case}-same-{index}") for index in range(2)]
    _assert_artifacts_equal(same_process[0], same_process[1], "same-process repeat")
    for hash_seed in ("1", "999"):
        separate = _subprocess_snapshot(case, hash_seed, tmp_path / f"{case}-seed-{hash_seed}")
        _assert_artifacts_equal(same_process[0], separate, f"separate process, PYTHONHASHSEED={hash_seed}")


def test_sm120_default_off_complete_compile_is_deterministic(tmp_path):
    """Stock compilation repeats without a Z3 solve or schedule artifact."""

    _exercise_complete_compile_determinism("default_off", tmp_path)


@pytest.mark.parametrize("case", ["multi_stage", "bind_clone"])
def test_sm120_opt_in_complete_compile_is_deterministic(case, tmp_path):
    """Real Z3 solves produce stable schedules, TIR, CUDA, and metadata."""

    _exercise_complete_compile_determinism(case, tmp_path)


if __name__ == "__main__" and len(sys.argv) == 5 and sys.argv[1] == "--compile-worker":
    _, _, worker_case, worker_root, worker_output = sys.argv
    worker_snapshot = _compile_snapshot(worker_case, Path(worker_root))
    Path(worker_output).write_text(json.dumps(worker_snapshot, indent=2, sort_keys=True), encoding="utf-8")
