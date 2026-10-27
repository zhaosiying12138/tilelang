"""CPU-only tests for the CUDA-owned Python Z3 schedulers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest
import tvm_ffi
import z3


os.environ.setdefault("MPLBACKEND", "Agg")
_SCHEDULER_PATH = Path(__file__).parents[3] / "tilelang" / "cuda" / "transform" / "z3_scheduler.py"
_SPEC = importlib.util.spec_from_file_location("_tilelang_cuda_z3_scheduler_under_test", _SCHEDULER_PATH)
z3_scheduler = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = z3_scheduler
_SPEC.loader.exec_module(z3_scheduler)


def _straight(**overrides):
    arguments = {
        "latencies": [1],
        "iis": [1],
        "resource_flags": [0],
        "data_deps": [],
        "resource_deps": [],
        "owner_exclusion_deps": [],
        "pipe_order_deps": [],
    }
    arguments.update(overrides)
    return z3_scheduler.z3_schedule_python(**arguments)


def _loop(**overrides):
    arguments = {
        "num_stages": 2,
        "latencies": [1],
        "iis": [1],
        "resource_flags": [0],
        "data_deps": [],
        "resource_deps": [],
        "owner_exclusion_deps": [],
        "pipe_order_deps": [],
        "buffer_sizes": [],
        "memory_groups": [],
    }
    arguments.update(overrides)
    return z3_scheduler.z3_schedule_loop_python(**arguments)


def _assert_loop_schedule(starts, versions, ii, arguments):
    """Independently check the public constraints on a returned loop model."""
    arguments = {
        "data_deps": [],
        "resource_deps": [],
        "owner_exclusion_deps": [],
        "pipe_order_deps": [],
        "buffer_sizes": [],
        "memory_groups": [],
        "stage_order_deps": [],
        "enable_offset": False,
        **arguments,
    }
    assert ii >= 1
    assert len(starts) == len(arguments["latencies"])
    assert all(start >= 0 for start in starts)
    k_values = [start // ii for start in starts]
    residues = [start % ii for start in starts]

    for source, target, distance, latency in arguments["data_deps"]:
        distance_or_versions = distance if distance >= 0 else versions[-distance - 1]
        assert starts[target] - starts[source] >= latency - ii * distance_or_versions

    for left, right in arguments["resource_deps"]:
        left, right = min(left, right), max(left, right)
        if residues[left] <= residues[right]:
            assert residues[right] - residues[left] >= arguments["iis"][left]
            assert residues[left] - residues[right] + ii >= arguments["iis"][right]
        else:
            assert residues[left] - residues[right] >= arguments["iis"][right]
            assert residues[right] - residues[left] + ii >= arguments["iis"][left]

    for left, right, latency_lr, latency_rl in arguments["owner_exclusion_deps"]:
        if residues[left] <= residues[right]:
            assert residues[right] - residues[left] >= latency_lr
            assert residues[left] - residues[right] + ii >= latency_rl
        else:
            assert residues[left] - residues[right] >= latency_rl
            assert residues[right] - residues[left] + ii >= latency_lr

    for previous, current in arguments["pipe_order_deps"]:
        assert residues[previous] <= residues[current]
    for upstream, downstream in arguments.get("stage_order_deps", []):
        assert k_values[upstream] <= k_values[downstream]

    if not arguments.get("enable_offset", False):
        for left in range(len(starts)):
            for right in range(left + 1, len(starts)):
                if arguments["resource_flags"][left] & arguments["resource_flags"][right]:
                    assert abs(starts[left] - starts[right]) < ii

    for capacity, *members in arguments["memory_groups"]:
        assert sum(versions[member] * arguments["buffer_sizes"][member] for member in members) <= capacity


def test_straight_line_minimizes_makespan_and_stable_tie():
    starts, order = _straight(latencies=[4, 2, 3], iis=[1, 1, 1], resource_flags=[0, 0, 0])
    assert starts == [0, 0, 0]
    assert order == [0, 1, 2]


def test_straight_line_data_latency_and_resource_exclusion():
    starts, order = _straight(
        latencies=[3, 2],
        iis=[2, 1],
        resource_flags=[1, 1],
        data_deps=[(0, 1, 3)],
        resource_deps=[(1, 0)],  # Unordered pairs are honored in either input order.
    )
    assert starts == [0, 3]
    assert order == [0, 1]
    assert max(start + latency for start, latency in zip(starts, [3, 2])) == 5


@pytest.mark.parametrize(
    ("pipe_order", "expected", "expected_order"),
    [
        ([(0, 1)], [0, 4], [0, 1]),
        ([(1, 0)], [2, 0], [1, 0]),
    ],
)
def test_straight_line_owner_exclusion_and_pipe_order(pipe_order, expected, expected_order):
    starts, order = _straight(
        latencies=[4, 4],
        iis=[1, 1],
        resource_flags=[0, 0],
        owner_exclusion_deps=[(0, 1, 4, 2)],
        pipe_order_deps=pipe_order,
    )
    assert starts == expected
    assert order == expected_order


def test_straight_line_unsat_has_dedicated_exception():
    with pytest.raises(z3_scheduler.Z3ScheduleInfeasibleError, match="UNSAT"):
        _straight(data_deps=[(0, 0, 1)])


def test_straight_line_unknown_is_not_reported_as_unsat(monkeypatch):
    monkeypatch.setattr(z3.Optimize, "check", lambda self: z3.unknown)
    monkeypatch.setattr(z3.Optimize, "reason_unknown", lambda self: "injected budget exhaustion")
    with pytest.raises(z3_scheduler.Z3ScheduleUnknownError, match="injected budget exhaustion") as error:
        _straight()
    assert not isinstance(error.value, z3_scheduler.Z3ScheduleInfeasibleError)


def test_loop_start_is_k_ii_plus_r():
    arguments = {
        "latencies": [3, 1],
        "iis": [1, 1],
        "resource_flags": [0, 0],
        "data_deps": [(0, 1, 0, 3)],
        "enable_offset": True,
    }
    starts, versions, ii = _loop(**arguments)
    assert ii == 1
    assert starts == [0, 3]
    assert starts[1] == 3 * ii
    _assert_loop_schedule(starts, versions, ii, arguments)


def test_loop_fixed_cross_iteration_distance():
    starts, _, ii = _loop(latencies=[4], data_deps=[(0, 0, 2, 5)])
    assert starts == [0]
    assert ii == 3  # 0 >= 5 - 2 * II, hence II >= ceil(5 / 2).


def test_loop_negative_distance_recalculates_and_reads_buffer_versions():
    arguments = {
        "num_stages": 4,
        "latencies": [4],
        "data_deps": [(0, 0, -1, 5)],
        "buffer_sizes": [1],
        "memory_groups": [[3, 0]],
    }
    for recalculate in (True, False):
        starts, versions, ii = _loop(**arguments, recalculate_buffer_versions=recalculate)
        assert starts == [0]
        assert ii == 2
        assert versions == [3]
        _assert_loop_schedule(starts, versions, ii, {**arguments, "resource_deps": [], "owner_exclusion_deps": [], "pipe_order_deps": []})


def test_loop_resource_forward_and_wraparound_exclusion():
    arguments = {
        "latencies": [2, 2],
        "iis": [2, 2],
        "resource_flags": [0, 0],
        "resource_deps": [(0, 1)],
    }
    starts, versions, ii = _loop(**arguments)
    assert ii == 4  # Two two-cycle issue intervals must fit in both cyclic directions.
    assert sorted(starts) == [0, 2]
    _assert_loop_schedule(starts, versions, ii, arguments)


def test_loop_owner_forward_and_wraparound_exclusion():
    arguments = {
        "latencies": [3, 3],
        "iis": [1, 1],
        "resource_flags": [0, 0],
        "owner_exclusion_deps": [(0, 1, 1, 2)],
    }
    starts, versions, ii = _loop(**arguments)
    assert ii == 3
    _assert_loop_schedule(starts, versions, ii, arguments)


@pytest.mark.parametrize("owner", [[(0, 1, 0, 2)], [(1, 0, 2, 0)]])
def test_loop_owner_equal_residue_has_explicit_valid_runtime_direction(owner):
    starts, _, ii = _loop(
        latencies=[1, 1],
        iis=[1, 1],
        resource_flags=[0, 0],
        owner_exclusion_deps=owner,
        enable_offset=True,
    )
    assert ii == 2
    assert starts == [0, 0]
    left, right = 0, 1
    left_residue, right_residue = starts[left] % ii, starts[right] % ii
    assert left_residue == right_residue == 0
    assert right_residue - left_residue >= 0
    assert left_residue - right_residue + ii >= 2


def test_loop_owner_unsat_never_returns_unvalidated_serial_fallback():
    with pytest.raises(z3_scheduler.Z3ScheduleInfeasibleError, match="Owner exclusion.*no serial fallback"):
        _loop(
            latencies=[1, 1],
            iis=[1, 1],
            resource_flags=[0, 0],
            owner_exclusion_deps=[(0, 1, 4, 4)],
            enable_offset=True,
        )


def test_loop_stage_order_changes_minimum_ii():
    arguments = {
        "latencies": [2, 2],
        "iis": [1, 1],
        "resource_flags": [0, 0],
        "data_deps": [(1, 0, 0, 1)],
        "stage_order_deps": [(0, 1)],
        "enable_offset": True,
    }
    starts, versions, ii = _loop(**arguments)
    assert ii == 2
    assert starts == [1, 0]
    _assert_loop_schedule(starts, versions, ii, arguments)


def test_loop_enable_offset_controls_cross_stage_resource_spread():
    arguments = {
        "latencies": [3, 1],
        "iis": [1, 1],
        "resource_flags": [1, 1],
        "data_deps": [(0, 1, 0, 3)],
    }
    offset_starts, _, offset_ii = _loop(**arguments, enable_offset=True)
    constrained_starts, constrained_versions, constrained_ii = _loop(**arguments, enable_offset=False)
    assert (offset_starts, offset_ii) == ([0, 3], 1)
    assert constrained_ii == 4
    _assert_loop_schedule(
        constrained_starts,
        constrained_versions,
        constrained_ii,
        arguments,
    )


def test_loop_manual_stages_are_fixed_then_normalized():
    starts, _, ii = _loop(
        latencies=[1, 1],
        iis=[1, 1],
        resource_flags=[0, 0],
        stage_order_deps=[(0, 1)],
        enable_offset=True,
        manual_schedule=True,
        manual_stages=[1, 2],
    )
    assert ii == 1
    assert starts == [0, 1]


def test_loop_manual_pipe_order_infeasible_does_not_fallback():
    with pytest.raises(z3_scheduler.Z3ScheduleInfeasibleError, match="Manual schedule"):
        _loop(
            latencies=[2, 2],
            iis=[1, 1],
            resource_flags=[0, 0],
            data_deps=[(1, 0, 0, 1)],
            pipe_order_deps=[(0, 1)],
            enable_offset=True,
            manual_schedule=True,
            manual_stages=[0, 0],
        )


def test_loop_memory_group_capacity():
    arguments = {
        "num_stages": 3,
        "latencies": [1, 1],
        "iis": [1, 1],
        "resource_flags": [0, 0],
        "buffer_sizes": [2, 3],
        "memory_groups": [[5, 0, 1]],
    }
    starts, versions, ii = _loop(**arguments)
    assert versions == [1, 1]
    _assert_loop_schedule(starts, versions, ii, arguments)


def test_loop_automatic_serial_fallback_only_after_unsat_exhaustion():
    starts, versions, ii = _loop(
        latencies=[2, 3],
        iis=[1, 1],
        resource_flags=[0, 0],
        buffer_sizes=[1],
        memory_groups=[[0, 0]],
    )
    assert starts == [0, 2]
    assert versions == [1]
    assert ii == 5


def test_loop_mocked_unknown_never_falls_back(monkeypatch):
    calls = []

    def unknown_check(self):
        calls.append(self)
        return z3.unknown

    monkeypatch.setattr(z3.Solver, "check", unknown_check)
    monkeypatch.setattr(z3.Solver, "reason_unknown", lambda self: "mock rlimit")
    with pytest.raises(z3_scheduler.Z3ScheduleUnknownError, match="mock rlimit"):
        _loop(latencies=[4])
    assert len(calls) == 1  # It did not treat unknown as UNSAT and try a larger II.


def test_loop_unknown_after_sat_aborts_instead_of_pruning(monkeypatch):
    real_check = z3.Solver.check
    calls = []

    def sat_then_unknown(self):
        calls.append(self)
        if len(calls) == 1:
            return real_check(self)
        return z3.unknown

    monkeypatch.setattr(z3.Solver, "check", sat_then_unknown)
    monkeypatch.setattr(z3.Solver, "reason_unknown", lambda self: "injected unknown")
    with pytest.raises(z3_scheduler.Z3ScheduleUnknownError, match="II=2"):
        _loop(latencies=[4])
    assert len(calls) == 2


def test_private_context_loop_seed_and_rlimit(monkeypatch):
    real_context = z3.Context
    real_solver_set = z3.Solver.set
    real_optimize_set = z3.Optimize.set
    contexts = []
    solver_settings = []
    optimize_settings = []

    def context_factory():
        context = real_context()
        contexts.append(context)
        return context

    def record_solver_set(self, *arguments):
        solver_settings.append(arguments)
        return real_solver_set(self, *arguments)

    def record_optimize_set(self, *arguments):
        optimize_settings.append(arguments)
        return real_optimize_set(self, *arguments)

    monkeypatch.setattr(z3_scheduler.z3, "Context", context_factory)
    monkeypatch.setattr(z3.Solver, "set", record_solver_set)
    monkeypatch.setattr(z3.Optimize, "set", record_optimize_set)
    _loop()
    _straight()
    assert len(contexts) == 2
    assert contexts[0] is not contexts[1]
    assert ("random_seed", 42) in solver_settings
    assert ("rlimit", 1_000_000) in solver_settings
    assert optimize_settings == [("rlimit", 1_000_000)]


def test_repeated_and_threaded_solves_are_deterministic():
    arguments = {
        "num_stages": 3,
        "latencies": [3, 2, 1],
        "iis": [1, 1, 1],
        "resource_flags": [1, 1, 2],
        "data_deps": [(0, 1, 0, 2)],
        "resource_deps": [(0, 1)],
        "enable_offset": True,
    }
    expected = _loop(**arguments)
    assert [_loop(**arguments) for _ in range(4)] == [expected] * 4
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: _loop(**arguments), range(8)))
    assert results == [expected] * 8

    straight_arguments = {
        "latencies": [3, 2, 1],
        "iis": [1, 1, 1],
        "resource_flags": [1, 1, 2],
        "data_deps": [(0, 1, 2)],
        "resource_deps": [(0, 1)],
    }
    assert _straight(**straight_arguments) == _straight(**straight_arguments)


def test_solver_lock_serializes_feasibility_checks_across_entry_points(monkeypatch):
    real_check = z3_scheduler._check_feasibility
    measurement_lock = threading.Lock()
    state = {"active": 0, "maximum": 0, "threads": set()}

    def measured_check(solver, description):
        with measurement_lock:
            state["active"] += 1
            state["maximum"] = max(state["maximum"], state["active"])
            state["threads"].add(threading.get_ident())
        time.sleep(0.001)
        try:
            return real_check(solver, description)
        finally:
            with measurement_lock:
                state["active"] -= 1

    monkeypatch.setattr(z3_scheduler, "_check_feasibility", measured_check)

    def invoke(index):
        if index % 2:
            return _loop(latencies=[2, 1], iis=[1, 1], resource_flags=[0, 0], enable_offset=True)
        return _straight(latencies=[2, 1], iis=[1, 1], resource_flags=[0, 0])

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(invoke, range(12)))
    assert len(results) == 12
    assert state["maximum"] == 1
    assert len(state["threads"]) > 1


def test_schedule_results_are_stable_across_processes():
    straight_arguments = {
        "latencies": [3, 2, 1],
        "iis": [1, 1, 1],
        "resource_flags": [1, 1, 2],
        "data_deps": [(0, 1, 2)],
        "resource_deps": [(0, 1)],
    }
    loop_arguments = {
        "num_stages": 3,
        "latencies": [3, 2, 1],
        "iis": [1, 1, 1],
        "resource_flags": [1, 1, 2],
        "data_deps": [(0, 1, 0, 2)],
        "resource_deps": [(0, 1)],
        "owner_exclusion_deps": [],
        "pipe_order_deps": [],
        "buffer_sizes": [],
        "memory_groups": [],
        "enable_offset": True,
    }
    expected = {
        "straight": _straight(**straight_arguments),
        "loop": _loop(**loop_arguments),
    }
    script = f"""
import importlib.util
import json
spec = importlib.util.spec_from_file_location('scheduler_under_test', {str(_SCHEDULER_PATH)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
straight = module.z3_schedule_python(**{straight_arguments!r})
loop = module.z3_schedule_loop_python(**{loop_arguments!r})
print(json.dumps({{'straight': straight, 'loop': loop}}, sort_keys=True))
"""
    outputs = []
    for hash_seed in ("1", "999"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = hash_seed
        outputs.append(subprocess.check_output([sys.executable, "-c", script], env=env, text=True, timeout=30))
    assert outputs[0] == outputs[1]
    assert json.loads(outputs[0]) == json.loads(json.dumps(expected))


def test_resource_bits_are_configurable_and_checked():
    with pytest.raises(ValueError, match="does not fit in 7 bits"):
        _straight(resource_flags=[128])
    starts, _ = _straight(resource_flags=[128], resource_bits=8)
    assert starts == [0]
    with pytest.raises(ValueError, match="resource_bits"):
        _straight(resource_bits=0)


@pytest.mark.parametrize(
    "arguments",
    [
        {"latencies": [], "iis": [], "resource_flags": []},
        {"latencies": [1, 1]},
        {"latencies": [-1]},
        {"iis": [0]},
        {"resource_flags": [-1]},
        {"data_deps": [(0, 1)]},
        {"data_deps": [(0, 2, 1)]},
        {"data_deps": [(0, 0, -1)]},
        {"resource_deps": [(0, 0)]},
        {"verbose": 1},
    ],
)
def test_straight_line_strict_input_validation(arguments):
    with pytest.raises((TypeError, ValueError)):
        _straight(**arguments)


@pytest.mark.parametrize(
    "arguments",
    [
        {"num_stages": 0},
        {"buffer_sizes": [-1]},
        {"data_deps": [(0, 0, -1, 1)]},
        {"memory_groups": [[]]},
        {"memory_groups": [[-1]]},
        {"buffer_sizes": [1], "memory_groups": [[2, 1]]},
        {"buffer_sizes": [1], "memory_groups": [[2, 0, 0]]},
        {"manual_schedule": True},
        {"manual_schedule": True, "manual_stages": []},
        {"manual_schedule": True, "manual_stages": [-1]},
        {"manual_schedule": True, "manual_stages": [1], "enable_offset": False},
        {"enable_offset": "yes"},
    ],
)
def test_loop_strict_input_validation(arguments):
    with pytest.raises((TypeError, ValueError)):
        _loop(**arguments)


def test_missing_required_inputs_raise_type_error():
    with pytest.raises(TypeError):
        z3_scheduler.z3_schedule_python()
    with pytest.raises(TypeError):
        z3_scheduler.z3_schedule_loop_python()


def test_ffi_registration_conversion_and_return_shapes():
    assert tvm_ffi.get_global_func("tl.cuda.transform.z3_schedule_python") is not None
    assert tvm_ffi.get_global_func("tl.cuda.transform.z3_schedule_loop_python") is not None

    starts = z3_scheduler.z3_schedule_ffi([2, 1], [1, 1], [0, 0], [(0, 1, 2)], [])
    assert list(starts) == [0, 2]
    loop_starts, versions, ii = z3_scheduler.z3_schedule_loop_ffi(
        2,
        [2, 1],
        [1, 1],
        [0, 0],
        [(0, 1, 0, 2)],
        [],
        None,
        None,
        [],
        None,
    )
    assert list(loop_starts) == [0, 2]
    assert list(versions) == []
    assert ii == 1


def test_verbose_writes_json_in_tmp_path(tmp_path):
    starts, versions, ii = _loop(latencies=[2], verbose=True, debug_dir=tmp_path)
    schedule_dir = tmp_path / "schedule_000"
    info_path = schedule_dir / "schedule_info.json"
    assert info_path.is_file()
    info = json.loads(info_path.read_text(encoding="utf-8"))
    assert info["start_times"] == starts
    assert info["buffer_versions"] == versions
    assert info["minimal_II"] == ii
    assert info["random_seed"] == 42
    assert info["rlimit"] == 1_000_000
    assert info["search_history"]
    assert info["schedule_directory"] == str(schedule_dir)


def test_verbose_writes_png_and_pdf_in_tmp_path(tmp_path):
    pytest.importorskip("matplotlib")
    _loop(latencies=[2], verbose=True, debug_dir=tmp_path)
    schedule_dir = tmp_path / "schedule_000"
    assert (schedule_dir / "schedule_timeline.png").stat().st_size > 0
    assert (schedule_dir / "schedule_timeline.pdf").stat().st_size > 0


def test_debug_failure_does_not_change_schedule(monkeypatch, tmp_path, capsys):
    expected = _loop(latencies=[2])

    def fail_debug(*_arguments):
        raise OSError("injected read-only debug directory")

    monkeypatch.setattr(z3_scheduler, "_write_verbose_artifacts", fail_debug)
    actual = _loop(latencies=[2], verbose=True, debug_dir=tmp_path)
    assert actual == expected
    assert "injected read-only debug directory" in capsys.readouterr().out
