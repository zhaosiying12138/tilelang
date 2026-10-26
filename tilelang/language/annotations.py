"""Annotation helpers exposed on the TileLang language surface."""

from collections.abc import Callable
import operator
import os

from tilelang.layout import Fragment, Layout, PartialFragment
from tilelang.utils.language import is_fragment, is_reducer
from tvm.tirx import FloatImm, IntImm, Var, tvm_tuple
from tvm.tirx.script.parser import attr
from tvm.tirx.script.builder.ir import sblock_attr

__all__ = [
    "WSID",
    "Z3_PIPELINE_ANNOTATION",
    "use_swizzle",
    "annotate_layout",
    "annotate_safe_value",
    "annotate_l2_hit_ratio",
    "annotate_restrict_buffers",
    "annotate_min_blocks_per_sm",
    "annotate_ws_pipeline_depth",
    "annotate_ws_schedule",
    "ws_op",
    "z3_pipeline_annotations",
]

# For-annotation marker for the opt-in CUDA Z3 loop adapter.
Z3_PIPELINE_ANNOTATION = "tl_z3_pipeline"

# Annotation key carrying a statement's stable warp-specialization op id
# (kWSOpIdKey in src/transform/common/warp_specialize.h). Shorthand so kernels
# write ``annotations={T.WSID: "copy_A"}`` on tile ops and loops.
WSID = "tl.ws_op_id"


def use_swizzle(panel_size: int, order: str = "row", enable: bool = True):
    """Annotate a kernel to use a specific threadblock swizzle pattern."""
    if order == "row":
        device_func = "rasterization2DRow"
    elif order == "column":
        device_func = "rasterization2DColumn"
    elif order == "mlx":
        device_func = "rasterization2DMLX"
    else:
        raise ValueError(f"Unsupported swizzle order: {order}")
    if not enable:
        return None
    return attr(None, "threadblock_swizzle_pattern", tvm_tuple(device_func, panel_size))


def annotate_layout(layout_map: dict):
    """Annotate the layout of the buffer."""
    _layout_map = {}
    for buffer, layout in layout_map.items():
        if is_reducer(buffer):
            # The reducer's replicas are addends awaiting the finalize
            # collective, not equal copies: plain Fragment semantics would
            # silently misdescribe them.
            assert isinstance(layout, PartialFragment), (
                f"for reducer {buffer} (T.alloc_reducer), the layout must be a "
                f"PartialFragment describing the per-thread partials, but got {type(layout)}"
            )
        elif is_fragment(buffer):
            assert isinstance(layout, Fragment), f"for Fragment {buffer}, layout must be a Fragment, but got {type(layout)}"
        if isinstance(layout, Layout):
            _layout_map[buffer.data] = layout
        elif isinstance(layout, Callable):
            _layout_map[buffer.data] = Layout(buffer.shape, layout)
        else:
            raise ValueError(f"Invalid layout: {layout}")

    return sblock_attr({"layout_map": _layout_map})


def annotate_safe_value(safe_value_map: dict):
    """Annotate the safe value of the buffer."""
    _safe_value_map = {}
    for buffer, safe_value in safe_value_map.items():
        _safe_value_map[buffer.data] = safe_value
    return sblock_attr({"safe_value_map": _safe_value_map})


def annotate_l2_hit_ratio(l2_hit_ratio_map: dict):
    """Annotate the L2 hit ratio of the buffer."""
    _l2_hit_ratio_map = {}
    for buffer, hit_ratio in l2_hit_ratio_map.items():
        assert buffer.scope() == "global", "persistent L2 can only be applied to global buffers"
        _l2_hit_ratio_map[buffer.data] = FloatImm("float32", float(hit_ratio))
    return sblock_attr({"l2_hit_ratio_map": _l2_hit_ratio_map})


def annotate_min_blocks_per_sm(n: int):
    """Annotate the minimum number of thread blocks per SM (multiprocessor).

    When set, this value is passed as the second argument of
    ``__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)`` in
    the generated CUDA kernel.  A larger value hints the compiler to limit
    register usage so that more blocks can reside on each SM simultaneously,
    which can improve occupancy at the cost of potentially more register
    spilling.

    Example
    -------
    >>> @T.prim_func
    ... def my_kernel(...):
    ...     with T.Kernel(...):
                T.annotate_min_blocks_per_sm(2)
    ...         ...
    """
    assert isinstance(n, int) and n > 0, "n must be a positive integer"
    return attr(None, "tl.min_blocks_per_sm", n)


def annotate_restrict_buffers(*buffers):
    """Mark the given buffer parameters as non-restrict.

    This annotation tells codegen to omit the `__restrict__` qualifier for the
    specified kernel buffer parameters. Use this when two (or more) buffers may
    alias, for example overlapping slices from the same base tensor.

    Example
    -------
    >>> @T.prim_func
    ... def buggy_kernel(x: T.Tensor((N,), T.float32),
    ...                  y: T.Tensor((N,), T.float32)):
    ...     T.annotate_restrict_buffers(x, y)
    ...     with T.Kernel(N, threads=32) as pid:
    ...         y[pid] = x[pid] + 1
    """
    if not buffers:
        return None
    data_vars = []
    for buf in buffers:
        try:
            data_vars.append(buf.data)
        except Exception as e:
            raise TypeError(f"annotate_restrict_buffers expects Buffer arguments, got {type(buf)}") from e
    # Also return as block attribute (root block exists by default) for readability/tools.
    return sblock_attr({"tl.non_restrict_params": data_vars})


def annotate_ws_pipeline_depth(depth_map: dict):
    """Declare the depth of the pipelines the automatic scheduler hosts at
    the ENCLOSING scope for the given buffers, asserting each recurrence
    resets per version — otherwise the depth derives from the loop's
    ``num_stages``, and accumulators pin single-buffered.

    ``depth_map`` maps each buffer to its pipeline depth at this scope.
    """
    _map = {}
    for buffer, depth in depth_map.items():
        assert isinstance(depth, int) and depth >= 1, f"pipeline depth must be a positive int, got {depth!r}"
        _map[buffer.data] = IntImm("int32", depth)
    return attr(_map, "tl.ws_pipeline_depth", 0)


def annotate_ws_schedule(schedule):
    """Attach a warp-specialization schedule to the kernel.

    ``schedule`` is a typed :class:`~tilelang.language.warp_specialize.WSSchedule`
    object describing how to transform the straight-line kernel into a
    warp-specialized one: warp roles, pipelines (full/empty barrier pairs
    protecting multi-versioned buffers), and per-role instruction sequences
    per loop scope. It is materialized by the ``MaterializeWSSchedule`` pass;
    see ``examples/aws/gemm.py`` for a complete example.
    """
    from tilelang.language.warp_specialize import WSSchedule

    assert isinstance(schedule, WSSchedule), f"annotate_ws_schedule expects a T.WSSchedule object, got {type(schedule)}"
    return sblock_attr({"tl.ws_schedule": schedule})


def ws_op(op_id: str):
    """Manually annotate the enclosed statement(s) with a stable
    warp-specialization op or scope id.

    Wraps the body in an ``AttrStmt`` with key :data:`WSID`. This is the
    annotation of last resort, for statement forms that cannot carry
    ``annotations=`` themselves. The enclosed statements — a scalar Bind,
    or several statements such as an inlined scheduler method — become
    ONE opaque op:

    >>> with T.ws_op("rescale_vote"):
    ...     should_rescale = T.any_sync(scale_shared[tid] < 1.0)
    >>> with T.ws_op("sched_next"):
    ...     sched.next_tile()

    With a scope id the wrapper encloses a ``while`` loop and opens that
    scope (serial scope loops carry the id in their own annotations):

    >>> with T.ws_op("loop_wave"):
    ...     while sched.valid():
    ...         ...

    Tile ops and loops pass ``annotations={T.WSID: ...}`` directly
    instead.
    """
    assert isinstance(op_id, str) and op_id, "ws_op id must be a non-empty string"
    return attr(None, WSID, op_id)


def _z3_sequence(values, name):
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence, got {values!r}")
    try:
        return list(values)
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable sequence") from error


def _z3_integer(value, name, minimum=None):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got a boolean")
    try:
        result = int(operator.index(value))
    except TypeError as error:
        raise TypeError(f"{name} must be an integer, got {value!r}") from error
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


def _z3_integer_list(values, name, minimum=None):
    return [
        _z3_integer(value, f"{name}[{index}]", minimum)
        for index, value in enumerate(_z3_sequence(values, name))
    ]


def _z3_dependencies(values, arity, name):
    result = []
    for index, value in enumerate(_z3_sequence(values, name)):
        items = _z3_sequence(value, f"{name}[{index}]")
        if len(items) != arity:
            raise ValueError(f"{name}[{index}] must contain {arity} integers, got {len(items)}")
        result.append(tuple(_z3_integer(item, f"{name}[{index}][{position}]") for position, item in enumerate(items)))
    return result


def _z3_boolean(value, name):
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {value!r}")
    return value


def z3_pipeline_annotations(
    *,
    latencies,
    iis,
    resource_flags,
    data_deps=(),
    resource_deps=(),
    owner_exclusion_deps=(),
    pipe_order_deps=(),
    stage_order_deps=(),
    buffer_sizes=(),
    memory_groups=(),
    buffer_vars=(),
    resource_bits=7,
    enable_offset=False,
    manual_stages=None,
    manual_schedule=False,
    seed=42,
    recalculate_buffer_versions=True,
    verbose=False,
    debug_dir=None,
):
    """Build and validate For annotations for CUDA :func:`Z3PipelinePlanning`.

    Pass the returned dictionary to ``T.Pipelined(..., num_stages=N,
    annotations=z3_pipeline_annotations(...))``. One solver task corresponds to
    one scheduled top-level statement after nested ``SeqStmt`` flattening and
    removal of buffer declarations and replayable scalar ``Bind`` statements.

    Dependency tuples use solver task indices. ``data_deps`` entries are
    ``(source, target, distance, latency)``; a negative distance ``-buffer_id-1``
    references the ordered ``buffer_vars``/``buffer_sizes`` entry. Resource,
    pipe-order and stage-order dependencies are pairs, while owner exclusions
    are ``(left, right, latency_lr, latency_rl)``. The current C++ adapter
    rejects owner exclusions and stage-order dependencies explicitly because
    downstream CUDA IR cannot yet preserve their full semantics.

    ``buffer_vars`` accepts Buffer objects or their ``.data`` Vars. Its order is
    the solver buffer-id order used by the returned
    ``tl_pipeline_buffer_versions`` map. ``memory_groups`` entries have the
    solver-native form ``[capacity, buffer_id, ...]``.
    """
    latencies = _z3_integer_list(latencies, "latencies", minimum=0)
    num_tasks = len(latencies)
    if num_tasks == 0:
        raise ValueError("Z3 pipeline planning requires at least one task")
    iis = _z3_integer_list(iis, "iis", minimum=1)
    resource_flags = _z3_integer_list(resource_flags, "resource_flags", minimum=0)
    if len(iis) != num_tasks or len(resource_flags) != num_tasks:
        raise ValueError("latencies, iis, and resource_flags must have equal lengths")

    resource_bits = _z3_integer(resource_bits, "resource_bits", minimum=1)
    for index, mask in enumerate(resource_flags):
        if mask >= 1 << resource_bits:
            raise ValueError(f"resource_flags[{index}]={mask} does not fit in {resource_bits} bits")

    data_deps = _z3_dependencies(() if data_deps is None else data_deps, 4, "data_deps")
    resource_deps = _z3_dependencies(() if resource_deps is None else resource_deps, 2, "resource_deps")
    owner_exclusion_deps = _z3_dependencies(
        () if owner_exclusion_deps is None else owner_exclusion_deps,
        4,
        "owner_exclusion_deps",
    )
    pipe_order_deps = _z3_dependencies(() if pipe_order_deps is None else pipe_order_deps, 2, "pipe_order_deps")
    stage_order_deps = _z3_dependencies(() if stage_order_deps is None else stage_order_deps, 2, "stage_order_deps")

    buffer_sizes = _z3_integer_list(
        () if buffer_sizes is None else buffer_sizes,
        "buffer_sizes",
        minimum=0,
    )
    ordered_buffer_vars = []
    for index, buffer in enumerate(_z3_sequence(() if buffer_vars is None else buffer_vars, "buffer_vars")):
        var = buffer if isinstance(buffer, Var) else getattr(buffer, "data", None)
        if not isinstance(var, Var):
            raise TypeError(f"buffer_vars[{index}] must be a Buffer or Var, got {buffer!r}")
        if any(var.same_as(previous) for previous in ordered_buffer_vars):
            raise ValueError(f"buffer_vars[{index}] duplicates an earlier solver buffer Var")
        ordered_buffer_vars.append(var)
    if len(buffer_sizes) != len(ordered_buffer_vars):
        raise ValueError("buffer_sizes and buffer_vars must have equal lengths")

    def check_task(index, name):
        if not 0 <= index < num_tasks:
            raise ValueError(f"{name} task index {index} is outside [0, {num_tasks})")

    for index, (source, target, distance, latency) in enumerate(data_deps):
        check_task(source, f"data_deps[{index}].source")
        check_task(target, f"data_deps[{index}].target")
        if latency < 0:
            raise ValueError(f"data_deps[{index}] latency must be non-negative")
        if distance < 0 and -distance - 1 >= len(buffer_sizes):
            raise ValueError(f"data_deps[{index}] references an unavailable solver buffer")

    for name, dependencies in (
        ("resource_deps", resource_deps),
        ("pipe_order_deps", pipe_order_deps),
        ("stage_order_deps", stage_order_deps),
    ):
        for index, (left, right) in enumerate(dependencies):
            check_task(left, f"{name}[{index}].left")
            check_task(right, f"{name}[{index}].right")
            if name == "resource_deps" and left == right:
                raise ValueError(f"resource_deps[{index}] cannot exclude a task from itself")

    for index, (left, right, latency_lr, latency_rl) in enumerate(owner_exclusion_deps):
        check_task(left, f"owner_exclusion_deps[{index}].left")
        check_task(right, f"owner_exclusion_deps[{index}].right")
        if left == right:
            raise ValueError(f"owner_exclusion_deps[{index}] cannot exclude a task from itself")
        if latency_lr < 0 or latency_rl < 0:
            raise ValueError(f"owner_exclusion_deps[{index}] latencies must be non-negative")

    groups = []
    for group_index, group in enumerate(_z3_sequence(() if memory_groups is None else memory_groups, "memory_groups")):
        entries = _z3_integer_list(group, f"memory_groups[{group_index}]", minimum=0)
        if not entries:
            raise ValueError(f"memory_groups[{group_index}] must start with a capacity")
        members = entries[1:]
        if len(set(members)) != len(members):
            raise ValueError(f"memory_groups[{group_index}] contains duplicate buffer indices")
        for member in members:
            if member >= len(buffer_sizes):
                raise ValueError(f"memory_groups[{group_index}] buffer index {member} is out of range")
        groups.append(entries)

    enable_offset = _z3_boolean(enable_offset, "enable_offset")
    manual_schedule = _z3_boolean(manual_schedule, "manual_schedule")
    if manual_schedule and manual_stages is None:
        raise ValueError("manual_schedule=True requires one manual stage per task")
    if manual_stages is None:
        manual_stages = []
    else:
        if not manual_schedule:
            raise ValueError("manual_stages requires manual_schedule=True")
        manual_stages = _z3_integer_list(manual_stages, "manual_stages", minimum=0)
        if len(manual_stages) != num_tasks:
            raise ValueError("manual_schedule requires one manual stage per task")
    if manual_schedule and not enable_offset and any(stage != 0 for stage in manual_stages):
        raise ValueError("Non-zero manual stages require enable_offset=True")

    seed = _z3_integer(seed, "seed", minimum=0)
    recalculate_buffer_versions = _z3_boolean(recalculate_buffer_versions, "recalculate_buffer_versions")
    verbose = _z3_boolean(verbose, "verbose")
    if debug_dir is not None:
        try:
            debug_dir = os.fspath(debug_dir)
        except TypeError as error:
            raise TypeError("debug_dir must be a string, path-like object, or None") from error
        if not isinstance(debug_dir, str):
            raise TypeError("debug_dir must resolve to a string path")

    def imms(values):
        return [IntImm("int64", value) for value in values]

    def nested_imms(groups_of_values):
        return [imms(values) for values in groups_of_values]

    annotations = {
        Z3_PIPELINE_ANNOTATION: True,
        "tl_z3_pipeline_latencies": imms(latencies),
        "tl_z3_pipeline_iis": imms(iis),
        "tl_z3_pipeline_resource_flags": imms(resource_flags),
        "tl_z3_pipeline_data_deps": nested_imms(data_deps),
        "tl_z3_pipeline_resource_deps": nested_imms(resource_deps),
        "tl_z3_pipeline_owner_exclusion_deps": nested_imms(owner_exclusion_deps),
        "tl_z3_pipeline_pipe_order_deps": nested_imms(pipe_order_deps),
        "tl_z3_pipeline_stage_order_deps": nested_imms(stage_order_deps),
        "tl_z3_pipeline_buffer_sizes": imms(buffer_sizes),
        "tl_z3_pipeline_memory_groups": nested_imms(groups),
        "tl_z3_pipeline_buffer_vars": ordered_buffer_vars,
        "tl_z3_pipeline_resource_bits": IntImm("int64", resource_bits),
        "tl_z3_pipeline_enable_offset": enable_offset,
        "tl_z3_pipeline_manual_stages": imms(manual_stages),
        "tl_z3_pipeline_manual_schedule": manual_schedule,
        "tl_z3_pipeline_seed": IntImm("int64", seed),
        "tl_z3_pipeline_recalculate_buffer_versions": recalculate_buffer_versions,
        "tl_z3_pipeline_verbose": verbose,
    }
    if debug_dir is not None:
        annotations["tl_z3_pipeline_debug_dir"] = debug_dir
    return annotations
