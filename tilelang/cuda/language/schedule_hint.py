"""Public scheduling boundaries for the opt-in CUDA Z3 pipeline planner."""

from __future__ import annotations

from tvm import tirx
from tvm.tirx import IntImm
from tvm.tirx.script.builder.ir import attr

__all__ = ["Stage", "Task"]


def Task(latency: int | None = None, ii: int | None = None):
    """Group serial statements into one atomic CUDA Z3 scheduler task.

    The complete body is one dependency and scheduling unit. Statements inside
    it issue serially without scheduler-internal synchronization, and must use
    one execution domain. ``latency`` and ``ii`` are exact whole-task overrides;
    either may be omitted and inferred independently. They do not replace the
    task's buffer dependency footprint.

    ``T.Task`` is consumed only inside an opted-in CUDA Z3 pipeline loop. Put
    ``T.Stage`` outside ``T.Task`` when assigning a manual stage::

        with T.Stage(0), T.Task(latency=4, ii=1):
            T.copy(A[k], A_shared)
    """
    if latency is not None and (isinstance(latency, bool) or not isinstance(latency, int) or latency < 0):
        raise ValueError(f"Task latency must be a non-negative integer, got {latency!r}")
    if ii is not None and (isinstance(ii, bool) or not isinstance(ii, int) or ii <= 0):
        raise ValueError(f"Task ii must be a positive integer, got {ii!r}")
    if latency is not None and ii is not None and latency < ii:
        raise ValueError(f"Task latency must be greater than or equal to ii, got latency={latency} and ii={ii}")

    annotations = {}
    if latency is not None:
        annotations["latency"] = IntImm("int64", latency)
    if ii is not None:
        annotations["ii"] = IntImm("int64", ii)
    return attr(annotations, "tl.cuda_z3_task", tirx.const(1, "int32"))


def Stage(stage: int):
    """Assign a non-negative manual stage to complete enclosed Z3 tasks.

    If any direct child has a stage marker, every unannotated sibling defaults
    to stage zero. Per-pipe issue order remains source order. Nonzero stages
    require ``enable_offset=True``. With no positive ``num_stages``, the buffer
    version search bound defaults to ``max(stage) + 1``; an explicit positive
    bound does not truncate stage values.

    ``T.Stage`` must be outside complete tasks and cannot be nested in another
    stage, a task, or a parallel tile-op body. It cannot be combined with the
    old ``manual_stages`` schema or handwritten pipeline order/stage arrays.
    """
    if isinstance(stage, bool) or not isinstance(stage, int) or stage < 0:
        raise ValueError(f"Stage must be a non-negative integer, got {stage!r}")
    return attr(IntImm("int64", stage), "tl.cuda_z3_stage", tirx.const(1, "int32"))
