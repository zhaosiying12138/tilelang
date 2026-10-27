"""CUDA-specific transformation frontends."""

from tvm import tirx
from tvm.target import Target
from tvm.tirx.stmt_functor import ir_transform, post_order_visit
from tvm.tirx.transform import prim_func_pass

from .. import _ffi_api
from . import z3_scheduler  # noqa: F401  # Import registers the CUDA Z3 FFI callbacks.
from .z3_scheduler import (
    Z3ScheduleInfeasibleError,
    Z3ScheduleUnknownError,
    z3_schedule_loop_python,
    z3_schedule_python,
)


def _default_recursive_roles():
    from tilelang import language as T

    # Role IDs are the fixed Worker/Load/MMA/Store convention shared with the
    # native role planner. The builder computes the used subset and warp layout.
    return [
        T.WSRole("Worker", warps_lo=0, warps_hi=4),
        T.WSRole("Load", warps_lo=4, warps_hi=5),
        T.WSRole("MMA", warps_lo=5, warps_hi=6),
        T.WSRole("Store", warps_lo=6, warps_hi=7),
    ]


def _recursive_root_pass(name, rewrite):
    """Apply an SBlock FFI rewrite only to materialized recursive roots."""

    def pass_fn(func, _mod, _ctx):
        def rewrite_root(stmt):
            if not isinstance(stmt, tirx.SBlock) or stmt.name_hint != "tilelang_root":
                return None
            found = False

            def visit(node):
                nonlocal found
                if isinstance(node, tirx.AttrStmt) and node.attr_key == "tl.schedule_unit":
                    found = True

            post_order_visit(stmt.body, visit)
            return rewrite(stmt) if found else None

        body = ir_transform(func.body, None, rewrite_root, ["tirx.SBlock"])
        return func if body.same_as(func.body) else func.with_body(body)

    return prim_func_pass(pass_fn, opt_level=0, name=name)


def NormalizeZ3ControlFlow():
    """Normalize recursive scheduling control flow before unit materialization."""
    return _ffi_api.NormalizeZ3ControlFlow()  # type: ignore


def MaterializeZ3ScheduleUnits():
    """Materialize the explicitly selected recursive Task/Control grammar."""
    return _ffi_api.MaterializeZ3ScheduleUnits()  # type: ignore


def AnalyzeZ3BufferLifetimes(target: Target):
    """Resolve recursive roles/epochs and attach positive lifetime aliases."""
    roles = _default_recursive_roles()

    def analyze(root):
        return _ffi_api.Z3AnalyzeRecursiveLifetimes(root, roles, 0, str(target))

    return _recursive_root_pass("tl.cuda.AnalyzeZ3BufferLifetimes", analyze)


def BuildZ3WSSchedule(target: Target, worker_warps: int = 0, widen_cpasync_load: bool = False):
    """Project a solved recursive tree to WS metadata when multiple roles need it."""
    roles = _default_recursive_roles()

    def build(root):
        built = _ffi_api.Z3BuildWSSchedule(
            root,
            roles,
            0,
            str(target),
            worker_warps,
            widen_cpasync_load,
        )
        schedule = built.annotations.get("tl.ws_schedule")
        if schedule is not None and len(schedule.roles) == 1 and len(schedule.pipelines) == 0:
            # The inspection build proves role/epoch feasibility. A one-role result
            # needs neither WS dispatch nor builder-added IDs, so retain Slice E's
            # original solved tree for the stock-only projection.
            return root
        return built

    return _recursive_root_pass("tl.cuda.BuildZ3WSSchedule", build)


def LowerZ3ScheduleUnits():
    """Project solved recursive units onto the stock prepared-loop contract."""
    return _ffi_api.LowerZ3ScheduleUnits()  # type: ignore


def AutoWarpSpecialization():
    """Derive a warp-specialization schedule with the scheduler named by
    the ``tl.enable_auto_warp_specialization`` pass config (currently
    ``"role_based"``); a no-op when the config is unset.

    Eligible kernels gain stable ``tl.ws_op_id`` markers and a typed
    ``WSSchedule`` block annotation — the same surface a hand-written
    schedule uses — which :func:`MaterializeWSSchedule` then lowers.
    Kernels the scheduler declines are left unchanged.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.AutoWarpSpecialization()  # type: ignore


def AnnotateDeviceBoundTmaCopies():
    """Mark tile copies whose TensorMap base is defined in the device body.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.AnnotateDeviceBoundTmaCopies()  # type: ignore


def ProducerConsumerWarpSpecialized():
    """Producer-consumer warp specialization at the tile-op level.

    This pass runs before LayoutInference and LowerTileOp. It rewrites
    eligible pipelined tile-op loops into warp-specialized producer and
    consumer branches with explicit barrier synchronization.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.ProducerConsumerWarpSpecialized()  # type: ignore


def LowerBlackwell2SM():
    """Lower 2SM TCGEN5MMA and related on Blackwell target

    Returns:
        fpass : tvm.transform.Pass
            The result pass
    """
    return _ffi_api.LowerBlackwell2SM()  # type: ignore


def LowerHopperIntrin():
    """LowerHopperIntrin"""
    if hasattr(_ffi_api, "LowerHopperIntrin"):
        return _ffi_api.LowerHopperIntrin()  # type: ignore
    return lambda f: f


def LowerL2Persistent():
    """LowerL2Persistent"""
    return _ffi_api.LowerL2Persistent()  # type: ignore


def LowerSharedTmem():
    """LowerSharedTmem"""
    return _ffi_api.LowerSharedTmem()  # type: ignore


def LowerSharedBarrier():
    """LowerSharedBarrier"""
    return _ffi_api.LowerSharedBarrier()  # type: ignore


def FuseMBarrierArriveExpectTx():
    """Fuse simple expect_tx -> TMA issue -> arrive back into arrive_and_expect_tx."""
    return _ffi_api.FuseMBarrierArriveExpectTx()  # type: ignore


def LowerLDGSTG():
    """Lower Ramp-based global memory load/store to ldg/stg intrinsics.

    This pass transforms vectorized global memory loads and stores (using Ramp indices)
    into explicit ldg32/64/128/256 and stg32/64/128/256 intrinsics for better codegen.

    Key behaviors:
    - Converts Ramp-based global BufferLoad to ldg intrinsics
    - Converts Ramp-based global BufferStore to stg intrinsics
    - Supports predicated loads (if_then_else with else=0)
    - Supports predicated stores (if in then case)
    - Skips loads in async scope (will be lowered to cp.async)
    - Only enabled for CUDA targets

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LowerLDGSTG()  # type: ignore


def MarkCudaSyncCalls(have_pdl: bool = False):
    """MarkCudaSyncCalls"""
    return _ffi_api.MarkCudaSyncCalls(have_pdl)  # type: ignore


def InjectFenceProxy():
    """InjectFenceProxy

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.InjectFenceProxy()  # type: ignore


def InjectTcgen05Fence():
    """Inject tcgen05.fence::before_thread_sync / after_thread_sync at
    conservative TCGEN05/TMEM synchronization boundaries on Blackwell
    (SM100+) targets.

    The current pass wraps CTA-wide shared-memory syncs and also inserts
    fences around linear mbarrier wait/use and use/arrive handoff patterns.
    It is intentionally conservative and does not try to infer arbitrary
    barrier protocols.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.InjectTcgen05Fence()  # type: ignore


def AnnotateWarpGroupRegAlloc():
    """Inject set_max_nreg calls into warp-specialized functions.

    This pass analyzes the function to collect register hints from set_max_nreg
    and no_set_max_nreg calls, then injects appropriate set_max_nreg calls into
    producer and consumer branches of warp-specialized code.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.AnnotateWarpGroupRegAlloc()  # type: ignore


def MaterializeWSSchedule():
    """Materialize a user-provided warp-specialization schedule.

    Consumes the ``ws_schedule`` block annotation attached by
    ``T.annotate_ws_schedule`` and rewrites the kernel into explicit
    warp-specialized form (role branches, multi-versioned buffers,
    mbarrier synchronization, TMA copies, async tcgen05 MMA).

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.MaterializeWSSchedule()  # type: ignore


def PersistThreadblock():
    """PersistThreadblock"""
    return _ffi_api.PersistThreadblock()  # type: ignore


__all__ = [
    "AnalyzeZ3BufferLifetimes",
    "AnnotateDeviceBoundTmaCopies",
    "AutoWarpSpecialization",
    "AnnotateWarpGroupRegAlloc",
    "BuildZ3WSSchedule",
    "FuseMBarrierArriveExpectTx",
    "InjectFenceProxy",
    "InjectTcgen05Fence",
    "LowerBlackwell2SM",
    "LowerHopperIntrin",
    "LowerLDGSTG",
    "LowerL2Persistent",
    "LowerSharedBarrier",
    "LowerSharedTmem",
    "LowerZ3ScheduleUnits",
    "MarkCudaSyncCalls",
    "MaterializeWSSchedule",
    "MaterializeZ3ScheduleUnits",
    "NormalizeZ3ControlFlow",
    "PersistThreadblock",
    "ProducerConsumerWarpSpecialized",
    "Z3ScheduleInfeasibleError",
    "Z3ScheduleUnknownError",
    "z3_schedule_loop_python",
    "z3_schedule_python",
]
