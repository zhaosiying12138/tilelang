"""Deterministic Z3 schedulers for CUDA automatic scheduling.

The two public functions mirror the Ascend scheduling model while using
CUDA-owned FFI names.  No CUDA device or compiled TileLang pass is needed to
solve a schedule.
"""

from __future__ import annotations

import json
import operator
import os
from pathlib import Path
import threading
import time

import tvm_ffi

try:
    import z3

    Z3_AVAILABLE = True
except ImportError:
    z3 = None
    Z3_AVAILABLE = False


_Z3_RLIMIT = 1_000_000
_DEFAULT_RESOURCE_BITS = 7
# Z3 is process-global internally, even when expressions use private contexts.
# An RLock lets the FFI wrappers and direct Python entry points share the same
# serialization without deadlocking when the wrapper calls the Python API.
_z3_lock = threading.RLock()


class Z3ScheduleInfeasibleError(RuntimeError):
    """Raised when Z3 proves that the requested schedule is infeasible."""


class Z3ScheduleUnknownError(RuntimeError):
    """Raised when Z3 cannot decide feasibility (for example, at ``rlimit``)."""


def _sequence(values, name: str) -> list:
    if values is None or isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence, got {values!r}")
    try:
        return list(values)
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable sequence") from error


def _integer(value, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got a boolean")
    try:
        return int(operator.index(value))
    except TypeError as error:
        raise TypeError(f"{name} must be an integer, got {value!r}") from error


def _integer_list(values, name: str, minimum: int | None = None) -> list[int]:
    result = []
    for index, value in enumerate(_sequence(values, name)):
        item = _integer(value, f"{name}[{index}]")
        if minimum is not None and item < minimum:
            raise ValueError(f"{name}[{index}] must be >= {minimum}, got {item}")
        result.append(item)
    return result


def _convert_int_tuples(values, arity: int, name: str = "dependency") -> list[tuple[int, ...]]:
    """Convert TVM/Python containers and validate every dependency tuple."""
    if values is None:
        return []
    result = []
    for index, value in enumerate(_sequence(values, name)):
        items = _sequence(value, f"{name}[{index}]")
        if len(items) != arity:
            raise ValueError(f"{name}[{index}] must contain {arity} integers, got {len(items)}")
        result.append(tuple(_integer(item, f"{name}[{index}][{position}]") for position, item in enumerate(items)))
    return result


def _boolean(value, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {value!r}")
    return value


def _task_index(value: int, num_tasks: int, name: str) -> int:
    if not 0 <= value < num_tasks:
        raise ValueError(f"{name} task index {value} is outside [0, {num_tasks})")
    return value


def _validate_common(latencies, iis, resource_flags, resource_bits):
    latencies = _integer_list(latencies, "latencies", minimum=0)
    num_tasks = len(latencies)
    if num_tasks == 0:
        raise ValueError("Z3 scheduling requires at least one task")
    iis = _integer_list(iis, "iis", minimum=1)
    resource_flags = _integer_list(resource_flags, "resource_flags", minimum=0)
    if len(iis) != num_tasks or len(resource_flags) != num_tasks:
        raise ValueError(
            "latencies, iis, and resource_flags must have equal lengths; "
            f"got {num_tasks}, {len(iis)}, and {len(resource_flags)}"
        )
    resource_bits = _integer(resource_bits, "resource_bits")
    if resource_bits < 1:
        raise ValueError("resource_bits must be positive")
    mask_limit = 1 << resource_bits
    for index, mask in enumerate(resource_flags):
        if mask >= mask_limit:
            raise ValueError(f"resource_flags[{index}]={mask} does not fit in {resource_bits} bits")
    return latencies, iis, resource_flags, resource_bits


def _validate_seed(seed):
    if seed is not None:
        seed = _integer(seed, "seed")
        if seed < 0:
            raise ValueError("seed must be non-negative or None")
    return seed


def _validate_debug_dir(debug_dir):
    if debug_dir is None:
        return None
    try:
        debug_dir = os.fspath(debug_dir)
    except TypeError as error:
        raise TypeError("debug_dir must be a string, path-like object, or None") from error
    if not isinstance(debug_dir, str):
        raise TypeError("debug_dir must resolve to a string path")
    return debug_dir


def _validate_pairs(values, num_tasks: int, name: str, arity: int = 2):
    pairs = _convert_int_tuples(values, arity, name)
    for pair_index, pair in enumerate(pairs):
        _task_index(pair[0], num_tasks, f"{name}[{pair_index}][0]")
        _task_index(pair[1], num_tasks, f"{name}[{pair_index}][1]")
    return pairs


def _validate_straight_dependencies(data_deps, resource_deps, owner_exclusion_deps, pipe_order_deps, num_tasks):
    data_deps = _convert_int_tuples(data_deps, 3, "data_deps")
    for index, (source, target, latency) in enumerate(data_deps):
        _task_index(source, num_tasks, f"data_deps[{index}].source")
        _task_index(target, num_tasks, f"data_deps[{index}].target")
        if latency < 0:
            raise ValueError(f"data_deps[{index}] latency must be non-negative")

    resource_deps = _validate_pairs(resource_deps, num_tasks, "resource_deps")
    normalized_resources = []
    for index, (left, right) in enumerate(resource_deps):
        if left == right:
            raise ValueError(f"resource_deps[{index}] cannot exclude a task from itself")
        normalized_resources.append((min(left, right), max(left, right)))

    owner_exclusion_deps = _convert_int_tuples(owner_exclusion_deps, 4, "owner_exclusion_deps")
    normalized_owners = []
    for index, (left, right, latency_lr, latency_rl) in enumerate(owner_exclusion_deps):
        _task_index(left, num_tasks, f"owner_exclusion_deps[{index}].left")
        _task_index(right, num_tasks, f"owner_exclusion_deps[{index}].right")
        if left == right:
            raise ValueError(f"owner_exclusion_deps[{index}] cannot exclude a task from itself")
        if latency_lr < 0 or latency_rl < 0:
            raise ValueError(f"owner_exclusion_deps[{index}] latencies must be non-negative")
        if left < right:
            normalized_owners.append((left, right, latency_lr, latency_rl))
        else:
            normalized_owners.append((right, left, latency_rl, latency_lr))

    pipe_order_deps = _validate_pairs(pipe_order_deps, num_tasks, "pipe_order_deps")
    return data_deps, normalized_resources, normalized_owners, pipe_order_deps


def _require_z3():
    if not Z3_AVAILABLE:
        raise RuntimeError("z3-solver package is required but not installed")


def _unknown_reason(solver) -> str:
    try:
        return str(solver.reason_unknown())
    except Exception:  # A mocked or partially initialized solver may lack it.
        return "no reason supplied"


def _check_feasibility(solver, description: str) -> bool:
    """Return True for sat, False for unsat, and never conflate unknown."""
    verdict = solver.check()
    if verdict == z3.sat:
        return True
    if verdict == z3.unsat:
        return False
    raise Z3ScheduleUnknownError(f"Z3 returned unknown for {description}: {_unknown_reason(solver)}")


def _find_next_schedule_number(base_dir="debug") -> int:
    """Find the first unused non-negative ``schedule_NNN`` suffix."""
    if not os.path.exists(base_dir):
        return 0
    existing = set()
    for item in os.listdir(base_dir):
        path = os.path.join(base_dir, item)
        if item.startswith("schedule_") and os.path.isdir(path):
            try:
                number = int(item[len("schedule_") :])
            except ValueError:
                continue
            if number >= 0:
                existing.add(number)
    number = 0
    while number in existing:
        number += 1
    return number


def _write_verbose_artifacts(info: dict, debug_dir) -> None:
    """Write JSON first, then optional PNG/PDF; callers isolate all failures."""
    base_dir = Path(debug_dir) if debug_dir is not None else Path("debug")
    schedule_dir = base_dir / f"schedule_{_find_next_schedule_number(base_dir):03d}"
    schedule_dir.mkdir(parents=True, exist_ok=True)

    info = dict(info)
    info["timestamp"] = time.time()
    info["schedule_directory"] = str(schedule_dir)
    with (schedule_dir / "schedule_info.json").open("w", encoding="utf-8") as output:
        json.dump(info, output, indent=2, sort_keys=True)

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        print(f"[CUDA Python Z3] WARNING: matplotlib unavailable; skipped PNG/PDF: {error}")
        return

    start_times = info["start_times"]
    latencies = info["latencies"]
    resource_flags = info["resource_flags"]
    num_tasks = len(start_times)
    fig, ax = plt.subplots(figsize=(12, max(3, num_tasks * 0.55)))
    try:
        is_loop = info["schedule_kind"] == "loop"
        max_time = max(start + latency for start, latency in zip(start_times, latencies))
        if is_loop:
            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch

            best_ii = info["minimal_II"]
            for period in range((max_time + best_ii - 1) // best_ii):
                if period % 2 == 0:
                    ax.axvspan(period * best_ii, (period + 1) * best_ii, color="gray", alpha=0.06, zorder=0)
            for boundary in range(0, max_time + best_ii, best_ii):
                ax.axvline(boundary, color="black", linestyle="--", linewidth=0.9, alpha=0.55, zorder=1)
        for index, (start, latency, mask) in enumerate(zip(start_times, latencies, resource_flags)):
            color = plt.cm.tab20(index % 20)
            if is_loop:
                issue = min(info["iis"][index], latency)
                ax.barh(index, issue, left=start, height=0.75, color=color, edgecolor="black", hatch="//", zorder=2)
                if latency > issue:
                    ax.barh(
                        index,
                        latency - issue,
                        left=start + issue,
                        height=0.75,
                        color=color,
                        edgecolor="black",
                        alpha=0.45,
                        zorder=2,
                    )
            else:
                ax.barh(index, latency, left=start, height=0.75, color=color, edgecolor="black")
            ax.text(
                start + max(latency, 1) / 2,
                index,
                f"T{index}\n0b{mask:0{info['resource_bits']}b}",
                ha="center",
                va="center",
                fontsize=8,
                zorder=3,
            )
        ax.invert_yaxis()
        ax.set_yticks(range(num_tasks), [f"Task {index}" for index in range(num_tasks)])
        ax.set_xlabel("Time (cycles)")
        title = f"CUDA Z3 {info['schedule_kind']} schedule"
        if info.get("minimal_II") is not None:
            title += f" (II={info['minimal_II']})"
        if info.get("fallback"):
            title += " [serial fallback]"
        ax.set_title(title)
        ax.grid(True, axis="x", alpha=0.2, zorder=0)
        if is_loop:
            ax.set_xlim(0, max_time)
            ax.legend(
                handles=[
                    Patch(facecolor=plt.cm.tab20(0), hatch="//", edgecolor="black", label="Issue interval"),
                    Patch(facecolor=plt.cm.tab20(0), alpha=0.45, edgecolor="black", label="Completion tail"),
                    Line2D([0], [0], color="black", linestyle="--", label="II boundary"),
                ],
                loc="upper center",
                bbox_to_anchor=(0.5, -0.16),
                ncol=3,
                frameon=False,
            )
        fig.tight_layout()
        fig.savefig(schedule_dir / "schedule_timeline.png", dpi=150, bbox_inches="tight")
        fig.savefig(schedule_dir / "schedule_timeline.pdf", bbox_inches="tight")
    finally:
        plt.close(fig)


def _save_verbose_artifacts(info: dict, debug_dir, verbose: bool) -> None:
    if not verbose:
        return
    try:
        _write_verbose_artifacts(info, debug_dir)
    except Exception as error:
        # Artifact generation is observational and must never change a result.
        print(f"[CUDA Python Z3] WARNING: failed to write schedule artifacts: {error}")


def _resource_mask(mask: int, resource_bits: int) -> str:
    return f"0b{mask:0{resource_bits}b}"


def z3_schedule_python(
    latencies: list[int],
    iis: list[int],
    resource_flags: list[int],
    data_deps: list[tuple[int, int, int]],
    resource_deps: list[tuple[int, int]],
    owner_exclusion_deps: list[tuple[int, int, int, int]] | None = None,
    pipe_order_deps: list[tuple[int, int]] | None = None,
    verbose: bool = False,
    resource_bits: int = _DEFAULT_RESOURCE_BITS,
    debug_dir: str | os.PathLike[str] | None = None,
) -> tuple[list[int], list[int]]:
    """Minimize the makespan of a straight-line CUDA task list.

    Resource pairs constrain issue intervals, owner pairs choose either storage
    lifetime order, and pipe-order pairs retain source issue order.  The second
    return value is stably sorted by ``(start_time, task_index)``.
    """
    with _z3_lock:
        latencies, iis, resource_flags, resource_bits = _validate_common(latencies, iis, resource_flags, resource_bits)
        num_tasks = len(latencies)
        data_deps, resource_deps, owner_exclusion_deps, pipe_order_deps = _validate_straight_dependencies(
            data_deps,
            resource_deps,
            owner_exclusion_deps,
            pipe_order_deps,
            num_tasks,
        )
        verbose = _boolean(verbose, "verbose")
        debug_dir = _validate_debug_dir(debug_dir)
        _require_z3()

        ctx = z3.Context()
        solver = z3.Optimize(ctx=ctx)
        solver.set("rlimit", _Z3_RLIMIT)
        start_vars = [z3.Int(f"start_{index}", ctx) for index in range(num_tasks)]
        for start in start_vars:
            solver.add(start >= 0)

        for source, target, latency in data_deps:
            solver.add(start_vars[target] >= start_vars[source] + latency)
        for pair_index, (left, right) in enumerate(resource_deps):
            left_first = z3.Bool(f"resource_order_{pair_index}", ctx)
            solver.add(z3.Implies(left_first, start_vars[right] >= start_vars[left] + iis[left]))
            solver.add(z3.Implies(z3.Not(left_first), start_vars[left] >= start_vars[right] + iis[right]))
        for pair_index, (left, right, latency_lr, latency_rl) in enumerate(owner_exclusion_deps):
            left_first = z3.Bool(f"owner_order_{pair_index}", ctx)
            solver.add(z3.Implies(left_first, start_vars[right] - start_vars[left] >= latency_lr))
            solver.add(z3.Implies(z3.Not(left_first), start_vars[left] - start_vars[right] >= latency_rl))
        for previous, current in pipe_order_deps:
            solver.add(start_vars[previous] <= start_vars[current])

        makespan = z3.Int("makespan", ctx)
        solver.add(makespan >= 0)
        for index, start in enumerate(start_vars):
            solver.add(makespan >= start + latencies[index])
        solver.minimize(makespan)

        if verbose:
            print(f"[CUDA Python Z3] Scheduling {num_tasks} straight-line tasks")
        if not _check_feasibility(solver, "straight-line makespan optimization"):
            raise Z3ScheduleInfeasibleError("Z3 straight-line scheduling is infeasible (UNSAT)")
        model = solver.model()
        start_times = [model.eval(start, model_completion=True).as_long() for start in start_vars]
        makespan_value = model.eval(makespan, model_completion=True).as_long()
        sorted_indices = sorted(range(num_tasks), key=lambda index: (start_times[index], index))

        if verbose:
            print(f"[CUDA Python Z3] Minimum makespan: {makespan_value}")
            for index in sorted_indices:
                print(
                    f"[CUDA Python Z3] Task {index}: start={start_times[index]}, "
                    f"latency={latencies[index]}, II={iis[index]}, "
                    f"resource={_resource_mask(resource_flags[index], resource_bits)}"
                )
        _save_verbose_artifacts(
            {
                "schedule_kind": "straight-line",
                "num_tasks": num_tasks,
                "latencies": latencies,
                "iis": iis,
                "resource_flags": resource_flags,
                "resource_bits": resource_bits,
                "data_dependencies": data_deps,
                "resource_dependencies": resource_deps,
                "owner_exclusions": owner_exclusion_deps,
                "pipe_order_dependencies": pipe_order_deps,
                "start_times": start_times,
                "sorted_indices": sorted_indices,
                "makespan": makespan_value,
                "rlimit": _Z3_RLIMIT,
                "fallback": False,
            },
            debug_dir,
            verbose,
        )
        return start_times, sorted_indices


def _validate_loop_inputs(
    num_stages,
    latencies,
    iis,
    resource_flags,
    data_deps,
    resource_deps,
    owner_exclusion_deps,
    pipe_order_deps,
    buffer_sizes,
    memory_groups,
    stage_order_deps,
    manual_stages,
    manual_schedule,
    resource_bits,
):
    num_stages = _integer(num_stages, "num_stages")
    if num_stages < 1:
        raise ValueError("num_stages must be positive")
    latencies, iis, resource_flags, resource_bits = _validate_common(latencies, iis, resource_flags, resource_bits)
    num_tasks = len(latencies)
    buffer_sizes = _integer_list(buffer_sizes, "buffer_sizes", minimum=0)

    data_deps = _convert_int_tuples(data_deps, 4, "data_deps")
    for index, (source, target, distance, latency) in enumerate(data_deps):
        _task_index(source, num_tasks, f"data_deps[{index}].source")
        _task_index(target, num_tasks, f"data_deps[{index}].target")
        if latency < 0:
            raise ValueError(f"data_deps[{index}] latency must be non-negative")
        if distance < 0:
            buffer_index = -distance - 1
            if buffer_index >= len(buffer_sizes):
                raise ValueError(
                    f"data_deps[{index}] refers to buffer {buffer_index}, "
                    f"but only {len(buffer_sizes)} buffers were supplied"
                )

    _, resource_deps, owner_exclusion_deps, pipe_order_deps = _validate_straight_dependencies(
        [], resource_deps, owner_exclusion_deps, pipe_order_deps, num_tasks
    )
    stage_order_deps = _validate_pairs(stage_order_deps, num_tasks, "stage_order_deps")

    groups = []
    for group_index, group in enumerate(_sequence(memory_groups, "memory_groups")):
        entries = _integer_list(group, f"memory_groups[{group_index}]", minimum=0)
        if not entries:
            raise ValueError(f"memory_groups[{group_index}] must start with a capacity")
        members = entries[1:]
        if len(set(members)) != len(members):
            raise ValueError(f"memory_groups[{group_index}] contains duplicate buffer indices")
        for member in members:
            if member >= len(buffer_sizes):
                raise ValueError(f"memory_groups[{group_index}] buffer index {member} is out of range")
        groups.append([entries[0], *members])

    manual_schedule = _boolean(manual_schedule, "manual_schedule")
    if manual_stages is not None:
        manual_stages = _integer_list(manual_stages, "manual_stages", minimum=0)
    if manual_schedule:
        if manual_stages is None:
            raise ValueError("manual_schedule=True requires one manual stage per task")
        if len(manual_stages) != num_tasks:
            raise ValueError(f"Manual schedule has {len(manual_stages)} stages for {num_tasks} tasks")
    elif manual_stages is None:
        manual_stages = [0] * num_tasks

    return (
        num_stages,
        latencies,
        iis,
        resource_flags,
        data_deps,
        resource_deps,
        owner_exclusion_deps,
        pipe_order_deps,
        buffer_sizes,
        groups,
        stage_order_deps,
        manual_stages,
        manual_schedule,
        resource_bits,
    )


def z3_schedule_loop_python(
    num_stages: int,
    latencies: list[int],
    iis: list[int],
    resource_flags: list[int],
    data_deps: list[tuple[int, int, int, int]],
    resource_deps: list[tuple[int, int]],
    owner_exclusion_deps: list[tuple[int, int, int, int]] | None,
    pipe_order_deps: list[tuple[int, int]] | None,
    buffer_sizes: list[int],
    memory_groups: list[list[int]],
    stage_order_deps: list[tuple[int, int]] | None = None,
    recalculate_buffer_versions: bool = True,
    enable_offset: bool = False,
    manual_stages: list[int] | None = None,
    manual_schedule: bool = False,
    seed: int | None = 42,
    verbose: bool = False,
    resource_bits: int = _DEFAULT_RESOURCE_BITS,
    debug_dir: str | os.PathLike[str] | None = None,
) -> tuple[list[int], list[int], int]:
    """Find a minimum-II CUDA modulo schedule using ``start = k * II + r``.

    Non-negative dependency distances are fixed iteration distances.  A
    negative distance ``-b-1`` references automatic buffer ``b`` and its Z3
    version variable.  Resource and owner exclusions include both the forward
    and reverse wraparound hand-offs.
    """
    with _z3_lock:
        (
            num_stages,
            latencies,
            iis,
            resource_flags,
            data_deps,
            resource_deps,
            owner_exclusion_deps,
            pipe_order_deps,
            buffer_sizes,
            memory_groups,
            stage_order_deps,
            manual_stages,
            manual_schedule,
            resource_bits,
        ) = _validate_loop_inputs(
            num_stages,
            latencies,
            iis,
            resource_flags,
            data_deps,
            resource_deps,
            owner_exclusion_deps,
            pipe_order_deps,
            buffer_sizes,
            memory_groups,
            stage_order_deps,
            manual_stages,
            manual_schedule,
            resource_bits,
        )
        num_tasks = len(latencies)
        recalculate_buffer_versions = _boolean(recalculate_buffer_versions, "recalculate_buffer_versions")
        enable_offset = _boolean(enable_offset, "enable_offset")
        verbose = _boolean(verbose, "verbose")
        seed = _validate_seed(seed)
        debug_dir = _validate_debug_dir(debug_dir)
        if manual_schedule and not enable_offset and any(stage != 0 for stage in manual_stages):
            raise ValueError("Non-zero manual stages require enable_offset=True")
        _require_z3()

        if verbose:
            print(f"[CUDA Python Z3 Loop] Scheduling {num_tasks} tasks, num_stages={num_stages}")
            print(f"[CUDA Python Z3 Loop] Data dependencies: {data_deps}")
            print(f"[CUDA Python Z3 Loop] Resource dependencies: {resource_deps}")
            print(f"[CUDA Python Z3 Loop] Owner exclusions: {owner_exclusion_deps}")
            print(f"[CUDA Python Z3 Loop] Buffer sizes/groups: {buffer_sizes} / {memory_groups}")

        ctx = z3.Context()
        ii_lower = max(iis)
        ii_upper = max(sum(latencies), sum(iis)) + 1
        best_ii = ii_upper
        best_model = None
        best_start_vars = None
        best_k_vars = None
        best_buffer_vars = None
        search_history = []

        while ii_lower < ii_upper:
            candidate_ii = (ii_lower + ii_upper) // 2
            solver = z3.Solver(ctx=ctx)
            solver.set("rlimit", _Z3_RLIMIT)
            if seed is not None:
                solver.set("random_seed", seed)
            k_vars = [z3.Int(f"k_{index}", ctx) for index in range(num_tasks)]
            r_vars = [z3.Int(f"r_{index}", ctx) for index in range(num_tasks)]
            start_vars = [k_vars[index] * candidate_ii + r_vars[index] for index in range(num_tasks)]
            for k_var, r_var in zip(k_vars, r_vars):
                solver.add(k_var >= 0, r_var >= 0, r_var < candidate_ii)

            if manual_schedule:
                for index, stage in enumerate(manual_stages):
                    solver.add(k_vars[index] == stage)
            for previous, current in pipe_order_deps:
                solver.add(r_vars[previous] <= r_vars[current])

            buffer_vars = [z3.Int(f"buffer_versions_{index}", ctx) for index in range(len(buffer_sizes))]
            for buffer_var in buffer_vars:
                solver.add(buffer_var >= 1, buffer_var <= num_stages)
            for capacity, *members in memory_groups:
                if members:
                    solver.add(z3.Sum([buffer_vars[index] * buffer_sizes[index] for index in members]) <= capacity)

            for source, target, distance, latency in data_deps:
                if distance >= 0:
                    solver.add(start_vars[target] - start_vars[source] >= latency - candidate_ii * distance)
                else:
                    buffer_var = buffer_vars[-distance - 1]
                    solver.add(start_vars[target] - start_vars[source] >= latency - candidate_ii * buffer_var)

            for pair_index, (left, right) in enumerate(resource_deps):
                left_first = z3.Bool(f"resource_order_{pair_index}", ctx)
                solver.add(
                    z3.Implies(left_first, r_vars[right] - r_vars[left] >= iis[left]),
                    z3.Implies(left_first, r_vars[left] - r_vars[right] + candidate_ii >= iis[right]),
                    z3.Implies(z3.Not(left_first), r_vars[left] - r_vars[right] >= iis[right]),
                    z3.Implies(z3.Not(left_first), r_vars[right] - r_vars[left] + candidate_ii >= iis[left]),
                )
            for pair_index, (left, right, latency_lr, latency_rl) in enumerate(owner_exclusion_deps):
                left_first = z3.Bool(f"owner_order_{pair_index}", ctx)
                solver.add(
                    z3.Implies(left_first, r_vars[right] - r_vars[left] >= latency_lr),
                    z3.Implies(left_first, r_vars[left] - r_vars[right] + candidate_ii >= latency_rl),
                    z3.Implies(z3.Not(left_first), r_vars[left] - r_vars[right] >= latency_rl),
                    z3.Implies(z3.Not(left_first), r_vars[right] - r_vars[left] + candidate_ii >= latency_lr),
                )
            for upstream, downstream in stage_order_deps:
                solver.add(k_vars[upstream] <= k_vars[downstream])

            if not enable_offset:
                active_bits = sorted(
                    {bit for mask in resource_flags for bit in range(mask.bit_length()) if mask & (1 << bit)}
                )
                for bit in active_bits:
                    group = [index for index, mask in enumerate(resource_flags) if mask & (1 << bit)]
                    for group_index, left in enumerate(group):
                        for right in group[group_index + 1 :]:
                            solver.add(start_vars[left] - start_vars[right] < candidate_ii)
                            solver.add(start_vars[right] - start_vars[left] < candidate_ii)

            if verbose:
                print(f"[CUDA Python Z3 Loop] Testing II={candidate_ii}")
            if _check_feasibility(solver, f"loop schedule at II={candidate_ii}"):
                search_history.append({"II": candidate_ii, "verdict": "sat"})
                best_ii = candidate_ii
                best_model = solver.model()
                best_start_vars = start_vars
                best_k_vars = k_vars
                best_buffer_vars = buffer_vars
                ii_upper = candidate_ii
            else:
                search_history.append({"II": candidate_ii, "verdict": "unsat"})
                ii_lower = candidate_ii + 1

        fallback = best_model is None
        if fallback:
            if manual_schedule:
                raise Z3ScheduleInfeasibleError(
                    "Manual schedule constraints are infeasible (UNSAT); check per-pipe source order, "
                    "manual stages, dependencies, resource usage, and memory capacity"
                )
            start_times = [0] * num_tasks
            for index in range(1, num_tasks):
                start_times[index] = start_times[index - 1] + latencies[index - 1]
            best_ii = max(sum(latencies), 1)
            buffer_versions = [1] * len(buffer_sizes)
        else:
            k_values = [best_model.eval(k_var, model_completion=True).as_long() for k_var in best_k_vars]
            stage_offset = min(k_values)
            start_times = [
                best_model.eval(start, model_completion=True).as_long() - stage_offset * best_ii
                for start in best_start_vars
            ]
            if recalculate_buffer_versions:
                buffer_versions = [1] * len(buffer_sizes)
                for source, target, distance, latency in data_deps:
                    if distance < 0:
                        buffer_index = -distance - 1
                        numerator = latency + start_times[source] - start_times[target]
                        versions = (numerator + best_ii - 1) // best_ii
                        buffer_versions[buffer_index] = max(buffer_versions[buffer_index], versions)
            else:
                buffer_versions = [
                    best_model.eval(buffer_var, model_completion=True).as_long() for buffer_var in best_buffer_vars
                ]

        sorted_indices = sorted(range(num_tasks), key=lambda index: (start_times[index], index))
        if verbose:
            print(f"[CUDA Python Z3 Loop] Result II={best_ii}, serial_fallback={fallback}")
            for index in sorted_indices:
                print(
                    f"[CUDA Python Z3 Loop] Task {index}: start={start_times[index]}, "
                    f"latency={latencies[index]}, II={iis[index]}, "
                    f"resource={_resource_mask(resource_flags[index], resource_bits)}"
                )
        _save_verbose_artifacts(
            {
                "schedule_kind": "loop",
                "num_stages": num_stages,
                "num_tasks": num_tasks,
                "minimal_II": best_ii,
                "latencies": latencies,
                "iis": iis,
                "resource_flags": resource_flags,
                "resource_bits": resource_bits,
                "data_dependencies": data_deps,
                "resource_dependencies": resource_deps,
                "owner_exclusions": owner_exclusion_deps,
                "pipe_order_dependencies": pipe_order_deps,
                "stage_order_dependencies": stage_order_deps,
                "buffer_sizes": buffer_sizes,
                "memory_groups": memory_groups,
                "buffer_versions": buffer_versions,
                "recalculate_buffer_versions": recalculate_buffer_versions,
                "enable_offset": enable_offset,
                "manual_schedule": manual_schedule,
                "manual_stages": manual_stages,
                "start_times": start_times,
                "sorted_indices": sorted_indices,
                "random_seed": seed,
                "rlimit": _Z3_RLIMIT,
                "search_history": search_history,
                "fallback": fallback,
            },
            debug_dir,
            verbose,
        )
        return start_times, buffer_versions, best_ii


@tvm_ffi.register_global_func("tl.cuda.transform.z3_schedule_python", override=True)
def z3_schedule_ffi(
    latencies,
    iis,
    resource_flags,
    data_deps,
    resource_deps,
    owner_exclusion_deps=None,
    pipe_order_deps=None,
    resource_bits=_DEFAULT_RESOURCE_BITS,
    verbose=False,
    debug_dir=None,
):
    """FFI wrapper converting TVM containers for straight-line scheduling."""
    latencies = _integer_list(latencies, "latencies")
    iis = _integer_list(iis, "iis")
    resource_flags = _integer_list(resource_flags, "resource_flags")
    data_deps = _convert_int_tuples(data_deps, 3, "data_deps")
    resource_deps = _convert_int_tuples(resource_deps, 2, "resource_deps")
    owner_exclusion_deps = _convert_int_tuples(owner_exclusion_deps, 4, "owner_exclusion_deps")
    pipe_order_deps = _convert_int_tuples(pipe_order_deps, 2, "pipe_order_deps")
    with _z3_lock:
        start_times, _ = z3_schedule_python(
            latencies,
            iis,
            resource_flags,
            data_deps,
            resource_deps,
            owner_exclusion_deps,
            pipe_order_deps,
            verbose=verbose,
            resource_bits=resource_bits,
            debug_dir=debug_dir,
        )
    return start_times


@tvm_ffi.register_global_func("tl.cuda.transform.z3_schedule_loop_python", override=True)
def z3_schedule_loop_ffi(
    num_stages,
    latencies,
    iis,
    resource_flags,
    data_deps,
    resource_deps,
    owner_exclusion_deps,
    pipe_order_deps,
    buffer_sizes,
    memory_groups,
    stage_order_deps=None,
    enable_offset=False,
    manual_stages=None,
    manual_schedule=False,
    resource_bits=_DEFAULT_RESOURCE_BITS,
    seed=42,
    recalculate_buffer_versions=True,
    verbose=False,
    debug_dir=None,
):
    """FFI wrapper converting TVM containers for loop modulo scheduling."""
    latencies = _integer_list(latencies, "latencies")
    iis = _integer_list(iis, "iis")
    resource_flags = _integer_list(resource_flags, "resource_flags")
    buffer_sizes = _integer_list(buffer_sizes, "buffer_sizes")
    data_deps = _convert_int_tuples(data_deps, 4, "data_deps")
    resource_deps = _convert_int_tuples(resource_deps, 2, "resource_deps")
    owner_exclusion_deps = _convert_int_tuples(owner_exclusion_deps, 4, "owner_exclusion_deps")
    pipe_order_deps = _convert_int_tuples(pipe_order_deps, 2, "pipe_order_deps")
    stage_order_deps = _convert_int_tuples(stage_order_deps, 2, "stage_order_deps")
    groups = [] if memory_groups is None else [list(group) for group in memory_groups]
    if manual_stages is not None:
        manual_stages = _integer_list(manual_stages, "manual_stages")
    with _z3_lock:
        return z3_schedule_loop_python(
            num_stages,
            latencies,
            iis,
            resource_flags,
            data_deps,
            resource_deps,
            owner_exclusion_deps,
            pipe_order_deps,
            buffer_sizes,
            groups,
            stage_order_deps=stage_order_deps,
            recalculate_buffer_versions=recalculate_buffer_versions,
            enable_offset=enable_offset,
            manual_stages=manual_stages,
            manual_schedule=manual_schedule,
            seed=seed,
            verbose=verbose,
            resource_bits=resource_bits,
            debug_dir=debug_dir,
        )
