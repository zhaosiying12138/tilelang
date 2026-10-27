"""Host-only vertical slice for recursive CUDA WS construction.

The main example is deliberately small: one guarded, two-owner, depth-two ring
and one shared counter protocol. Tests inspect IR only. They skip until the
central TileLang library is rebuilt with Slice F; no GPU build or run is done.
"""

from __future__ import annotations

import pytest
import tvm_ffi

import tilelang
import tilelang.testing
from tilelang import language as T
from tilelang import tvm
from tilelang.testing.ir import collect_calls, collect_nodes
from tvm import tirx


_ARCH = "sm_100a"
_CAPACITY = 1 << 20


def _ffi(name):
    function = tvm_ffi.get_global_func(name, allow_missing=True)
    if function is None:
        pytest.skip(f"{name} requires the centrally rebuilt TileLang library")
    return function


def _target(include_owner_capacity=True):
    spec = {"kind": "cuda", "arch": _ARCH}
    if include_owner_capacity:
        spec["max_shared_memory_per_block"] = _CAPACITY
    return tvm.target.Target(spec)


def _materialize(func, include_owner_capacity=True):
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.tirx.transform.BindTarget(_target(include_owner_capacity))(mod)
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    transformed = tilelang.cuda.transform.MaterializeWSSchedule()(mod)["main"]
    for node in collect_nodes(transformed, tvm.runtime.Object):
        assert {"tl.ws_schedule", "tl.ws_op_id"}.isdisjoint(
            getattr(node, "annotations", {})
        )
        if isinstance(node, tirx.AttrStmt):
            assert node.attr_key not in {"tl.ws_schedule", "tl.ws_op_id"}
    return transformed


def _buffers(root, name):
    result = [
        buf
        for block in collect_nodes(root, tirx.SBlock)
        for buf in block.alloc_buffers
        if buf.name == name
    ]
    result.extend(
        buf for buf in (node.buffer for node in collect_nodes(root, tirx.AllocBuffer)) if buf.name == name
    )
    return result


def _buffer(root, name):
    matches = _buffers(root, name)
    assert len(matches) == 1, f"expected one buffer {name}, got {len(matches)}"
    return matches[0]


def _barrier(func, name, counts):
    buffer = _buffer(func, name)
    assert buffer.scope() == "shared.barrier"
    initializers = [
        block.annotations["barrier_init"][buffer.data]
        for block in collect_nodes(func, tirx.SBlock)
        if "barrier_init" in block.annotations
        and buffer.data in block.annotations["barrier_init"]
    ]
    assert len(initializers) == 1
    assert [int(value) for value in initializers[0]] == counts
    assert [int(dim) for dim in buffer.shape] == [len(counts)]


def _role_branches(func):
    (launch,) = [
        attr
        for attr in collect_nodes(func, tirx.AttrStmt)
        if attr.attr_key == "thread_extent"
        and isinstance(attr.node, tirx.IterVar)
        and attr.node.thread_tag == "threadIdx.x"
    ]
    (custom,) = [
        attr
        for attr in collect_nodes(launch.body, tirx.AttrStmt)
        if attr.attr_key == "kCustomWarpSpecialization"
    ]
    branches = []
    lo = 0
    body = custom.body
    while isinstance(body, tirx.IfThenElse):
        hi = int(body.condition.b)
        branches.append((lo, hi, body.then_case))
        lo, body = hi, body.else_case
    branches.append((lo, int(launch.value), body))
    return branches


def _role_body(func, first_thread):
    (body,) = [body for lo, _, body in _role_branches(func) if lo == first_thread]
    return body


def _ring_kernel(*, guarded):
    pipeline = "ring_version_counter_1" if guarded else "ring"

    @T.prim_func
    def kernel(A: T.Tensor((T.dynamic("n"),), T.float32), B: T.Tensor((64,), T.float32)):
        with T.Kernel(1, threads=128) as _:
            ring = T.alloc_shared((4,), T.float32, scope="shared")
            frag = T.alloc_fragment((4,), T.float32)
            active = None
            if guarded:
                with T.ws_op("bind_active"):
                    active = A.shape[0] > 0

            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[
                        T.WSRole("Producer", warps_lo=0, warps_hi=1),
                        T.WSRole("Consumer", warps_lo=4, warps_hi=8),
                    ],
                    pipelines=[
                        T.WSPipeline(
                            pipeline,
                            [ring],
                            depth=2,
                            owner_scopes=("owner0", "owner1"),
                        )
                    ],
                    scopes=[
                        T.WSScope(
                            "owner0",
                            {
                                "Producer": [
                                    T.WSSync.producer_acquire(pipeline, guard=active),
                                    "produce0",
                                    T.WSSync.producer_commit(pipeline, guard=active),
                                ],
                                "Consumer": [
                                    T.WSSync.consumer_wait(pipeline, guard=active),
                                    "consume0",
                                    T.WSSync.consumer_release(pipeline, guard=active),
                                    "output0",
                                ],
                            },
                        ),
                        T.WSScope(
                            "owner1",
                            {
                                "Producer": [
                                    T.WSSync.producer_acquire(pipeline, guard=active),
                                    "produce1",
                                    T.WSSync.producer_commit(pipeline, guard=active),
                                ],
                                "Consumer": [
                                    T.WSSync.consumer_wait(pipeline, guard=active),
                                    "consume1",
                                    T.WSSync.consumer_release(pipeline, guard=active),
                                    "output1",
                                ],
                            },
                        ),
                        T.WSScope(
                            T.WSScope.ROOT,
                            {
                                "Producer": (["bind_active"] if guarded else []) + ["owner0", "owner1"],
                                "Consumer": (["bind_active"] if guarded else []) + ["owner0", "owner1"],
                            },
                        ),
                    ],
                )
            )

            for i in T.serial(2, annotations={T.WSID: "owner0"}):
                if active is None:
                    T.fill(ring, i, annotations={T.WSID: "produce0"})
                    T.copy(ring, frag, annotations={T.WSID: "consume0"})
                    T.copy(frag, B[i * 4], annotations={T.WSID: "output0"})
                else:
                    if active:
                        T.fill(ring, i, annotations={T.WSID: "produce0"})
                        T.copy(ring, frag, annotations={T.WSID: "consume0"})
                        T.copy(frag, B[i * 4], annotations={T.WSID: "output0"})
            for j in T.serial(3, annotations={T.WSID: "owner1"}):
                if active is None:
                    T.fill(ring, j, annotations={T.WSID: "produce1"})
                    T.copy(ring, frag, annotations={T.WSID: "consume1"})
                    T.copy(frag, B[(2 + j) * 4], annotations={T.WSID: "output1"})
                else:
                    if active:
                        T.fill(ring, j, annotations={T.WSID: "produce1"})
                        T.copy(ring, frag, annotations={T.WSID: "consume1"})
                        T.copy(frag, B[(2 + j) * 4], annotations={T.WSID: "output1"})

    return kernel


def _legacy_no_owner_kernel():
    @T.prim_func
    def kernel(B: T.Tensor((8,), T.float32)):
        with T.Kernel(1, threads=128) as _:
            ring = T.alloc_shared((4,), T.float32, scope="shared")
            frag = T.alloc_fragment((4,), T.float32)
            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[
                        T.WSRole("Producer", warps_lo=0, warps_hi=1),
                        T.WSRole("Consumer", warps_lo=4, warps_hi=8),
                    ],
                    pipelines=[T.WSPipeline("buf", [ring], depth=2)],
                    scopes=[
                        T.WSScope(
                            "loop_k",
                            {
                                "Producer": [
                                    T.WSSync.producer_acquire("buf"),
                                    "produce",
                                    T.WSSync.producer_commit("buf"),
                                ],
                                "Consumer": [
                                    T.WSSync.consumer_wait("buf"),
                                    "consume",
                                    T.WSSync.consumer_release("buf"),
                                    "output",
                                ],
                            },
                        ),
                        T.WSScope(
                            T.WSScope.ROOT,
                            {"Producer": ["loop_k"], "Consumer": ["loop_k"]},
                        ),
                    ],
                )
            )
            for i in T.Pipelined(2, num_stages=2, annotations={T.WSID: "loop_k"}):
                T.fill(ring, i, annotations={T.WSID: "produce"})
                T.copy(ring, frag, annotations={T.WSID: "consume"})
                T.copy(frag, B[i * 4], annotations={T.WSID: "output"})

    return kernel


def test_ws_sync_guard_reflection_is_backward_compatible():
    _ffi("tl.WSSyncWithGuard")
    buffer = tirx.decl_buffer((4,), "float32", name="ring")
    predicate = tirx.Var("active", "bool")
    legacy = T.WSSync.producer_acquire("legacy")
    explicit = T.WSSync(T.WSSyncKind.PRODUCER_ACQUIRE, "legacy", 0)
    guarded = T.WSSync.producer_acquire("legacy", guard=predicate)
    assert legacy.guard is None
    assert explicit.guard is None
    assert guarded.guard.same_as(predicate)
    tvm.ir.assert_structural_equal(legacy, explicit)
    assert list(T.WSPipeline("legacy", [buffer], 2).owner_scopes) == []


def test_single_role_no_pipeline_skips_unneeded_lifetime_adapter():
    build = _ffi("tl.cuda.transform.Z3BuildWSSchedule")
    shared = tirx.decl_buffer((8,), "int32", name="shared", scope="shared")
    output = tirx.decl_buffer((8,), "int32", name="output", scope="global")
    iv = tirx.Var("i", "int32")

    def unit(body, source_id, op_id):
        return tirx.AttrStmt(
            {"stage": 0, "source_id": source_id},
            "tl.schedule_unit",
            1,
            tirx.AttrStmt({"ws_op_id": op_id}, "tl.cuda_task", 1, body),
        )

    loop = tirx.For(
        iv,
        0,
        4,
        tirx.ForKind.SERIAL,
        tirx.SeqStmt(
            [
                unit(tirx.BufferStore(shared, iv, [iv]), 1, "write"),
                unit(tirx.BufferStore(output, shared[iv] + 1, [iv]), 2, "read"),
            ]
        ),
        annotations={"tl.ws_op_id": "owner"},
    )
    root = tirx.SBlock(
        [],
        [],
        [],
        "tilelang_root",
        tirx.AttrStmt(
            {"stage": 0, "source_id": 3}, "tl.schedule_unit", 1, loop
        ),
        alloc_buffers=[shared, output],
        annotations={
            "tl.cuda_z3_selected_buffer_versions": {shared.data: 2},
            "tl.buffer_version_mode": {shared.data: "iteration"},
        },
    )
    built = build(
        root,
        [T.WSRole("Worker", warps_lo=0, warps_hi=4)],
        0,
        '{"kind":"cuda","arch":"sm_90"}',
        4,
        False,
    )
    schedule = built.annotations["tl.ws_schedule"]
    assert [str(role.name) for role in schedule.roles] == ["Worker"]
    assert len(schedule.pipelines) == 0
    assert {str(scope.id) for scope in schedule.scopes} == {"owner", T.WSScope.ROOT}


def test_recursive_builder_emits_one_guarded_shared_ring_protocol():
    build = _ffi("tl.cuda.transform.Z3BuildWSSchedule")
    _ffi("tl.WSSyncWithGuard")
    shared = tirx.decl_buffer((4,), "float32", name="ring", scope="shared")
    source = tirx.decl_buffer((4,), "float32", name="source", scope="global")
    output = tirx.decl_buffer((4,), "float32", name="output", scope="global")
    enabled = tirx.Var("enabled", "bool")
    active = tirx.Var("active", "bool")

    def region(buffer, mask):
        indices = [0] * len(buffer.shape)
        return tirx.Call(
            "handle",
            tvm.ir.Op.get("tl.region"),
            [buffer[tuple(indices)], mask, *buffer.shape],
        )

    def copy(src, dst):
        return tirx.Evaluate(
            tirx.Call(
                "handle",
                tvm.ir.Op.get("tl.tileop.copy"),
                [region(src, 1), region(dst, 2)],
            )
        )

    def unit(body, source_id, op_id, *, guard=None):
        stmt = tirx.AttrStmt({"ws_op_id": op_id}, "tl.cuda_task", 1, body)
        if guard is not None:
            stmt = tirx.IfThenElse(guard, stmt, None)
        return tirx.AttrStmt(
            {"stage": 0, "source_id": source_id}, "tl.schedule_unit", 1, stmt
        )

    def owner(index, extent):
        iv = tirx.Var(f"i{index}", "int32")
        children = [
            unit(copy(source, shared), 10 + index * 10, f"load{index}", guard=active),
            unit(copy(shared, output), 11 + index * 10, f"store{index}", guard=active),
        ]
        loop = tirx.For(
            iv,
            0,
            extent,
            tirx.ForKind.SERIAL,
            tirx.SeqStmt(children),
            annotations={"tl.ws_op_id": f"owner{index}"},
        )
        return tirx.AttrStmt(
            {"stage": 0, "source_id": 20 + index * 10}, "tl.schedule_unit", 1, loop
        )

    bind = unit(tirx.Bind(active, enabled), 1, "bind_active")
    root = tirx.SBlock(
        [],
        [],
        [],
        "tilelang_root",
        tirx.SeqStmt([bind, owner(0, 2), owner(1, 3)]),
        alloc_buffers=[shared, source, output],
        annotations={
            "tl.cuda_z3_selected_buffer_versions": {shared.data: 2},
            "tl.buffer_version_mode": {shared.data: "auto"},
        },
    )
    roles = [
        T.WSRole("Load", warps_lo=0, warps_hi=1),
        T.WSRole("Store", warps_lo=1, warps_hi=2),
    ]
    built = build(root, roles, 0, '{"kind":"cuda","arch":"sm_90"}', 0, False)
    schedule = built.annotations["tl.ws_schedule"]
    assert schedule.num_warps == 4
    assert [str(role.name) for role in schedule.roles] == ["Load", "Store"]
    assert len(schedule.pipelines) == 1
    pipeline = schedule.pipelines[0]
    assert pipeline.depth == 2
    assert list(pipeline.owner_scopes) == ["owner0", "owner1"]
    assert str(pipeline.name).startswith("ring_version_counter_")

    scopes = {str(scope.id): scope for scope in schedule.scopes}
    assert set(scopes) == {"owner0", "owner1", T.WSScope.ROOT}
    for owner_id in ("owner0", "owner1"):
        for role, task, open_kind, close_kind in (
            ("Load", f"load{owner_id[-1]}", T.WSSyncKind.PRODUCER_ACQUIRE, T.WSSyncKind.PRODUCER_COMMIT),
            ("Store", f"store{owner_id[-1]}", T.WSSyncKind.CONSUMER_WAIT, T.WSSyncKind.CONSUMER_RELEASE),
        ):
            body = list(scopes[owner_id].bodies[role])
            assert len(body) == 3
            assert body[0].kind == open_kind
            assert body[2].kind == close_kind
            assert str(body[1].id) == task
            assert body[0].guard is not None and body[2].guard is not None
            tvm.ir.assert_structural_equal(body[0].guard, body[2].guard)
    for role in ("Load", "Store"):
        root_body = list(scopes[T.WSScope.ROOT].bodies[role])
        assert [str(instr.id) for instr in root_body] == ["bind_active", "owner0", "owner1"]

    # Slice E consumes the same encoded root without rediscovering order or IDs;
    # its lowering preserves the WS annotation for MaterializeWSSchedule.
    lower = _ffi("tl.cuda.transform.LowerZ3ScheduleUnits")
    func = tirx.PrimFunc([], tirx.SBlockRealize([], True, built)).with_attr(
        "global_symbol", "main"
    )
    lowered = lower()(tvm.IRModule({"main": func}))["main"]
    lowered_root = lowered.body.block
    assert "tl.ws_schedule" in lowered_root.annotations
    assert not any(
        isinstance(node, tirx.AttrStmt) and node.attr_key == "tl.schedule_unit"
        for node in collect_nodes(lowered_root, tirx.AttrStmt)
    )
    assert {"owner0", "owner1"}.issubset(
        {
            str(loop.annotations["tl.ws_op_id"])
            for loop in collect_nodes(lowered_root, tirx.For)
            if "tl.ws_op_id" in loop.annotations
        }
    )

def test_legacy_no_owner_pipeline_does_not_require_owner_capacity_metadata():
    # This target deliberately has no max_shared_memory_per_block. A plain
    # single-scope WSPipeline must follow the pre-owner-ring path unchanged.
    func = _materialize(_legacy_no_owner_kernel(), include_owner_capacity=False)
    ring = _buffer(func, "ring")
    assert [int(dim) for dim in ring.shape] == [2, 4]
    _barrier(func, "buf_full", [32, 32])
    _barrier(func, "buf_empty", [128, 128])
    assert [(lo, hi) for lo, hi, _ in _role_branches(func)] == [
        (0, 32),
        (32, 128),
        (128, 256),
    ]
    for first_thread in (0, 128):
        role = _role_body(func, first_thread)
        assert len(collect_calls(role, op="tl.mbarrier_wait_parity")) == 1
        assert len(collect_calls(role, op="tirx.ptx_arrive_barrier")) == 1


def test_legacy_no_owner_capacity_deadlock_still_rejected():
    @T.prim_func
    def kernel(A: T.Tensor((8, 4), T.float32), B: T.Tensor((8, 4), T.float32)):
        with T.Kernel(1, threads=128) as _:
            s1 = T.alloc_shared((4, 4), T.float32)
            s2 = T.alloc_shared((4, 4), T.float32)
            f1 = T.alloc_fragment((4, 4), T.float32)
            f2 = T.alloc_fragment((4, 4), T.float32)
            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[
                        T.WSRole("Producer", warps_lo=0, warps_hi=1),
                        T.WSRole("Consumer", warps_lo=4, warps_hi=8),
                    ],
                    pipelines=[
                        T.WSPipeline("s1", [s1], depth=2),
                        T.WSPipeline("s2", [s2], depth=2),
                    ],
                    scopes=[
                        T.WSScope(
                            "loop_a",
                            {
                                "Producer": [
                                    T.WSSync.producer_acquire("s1"),
                                    "copy_in1",
                                    T.WSSync.producer_commit("s1"),
                                ],
                                "Consumer": [
                                    T.WSSync.consumer_wait("s1"),
                                    "copy_frag1",
                                    T.WSSync.consumer_release("s1"),
                                    "copy_out1",
                                ],
                            },
                        ),
                        T.WSScope(
                            "loop_b",
                            {
                                "Consumer": [
                                    T.WSSync.producer_acquire("s2"),
                                    "copy_in2",
                                    T.WSSync.producer_commit("s2"),
                                ],
                                "Producer": [
                                    T.WSSync.consumer_wait("s2"),
                                    "copy_frag2",
                                    T.WSSync.consumer_release("s2"),
                                    "copy_out2",
                                ],
                            },
                        ),
                        T.WSScope(
                            T.WSScope.ROOT,
                            {
                                "Producer": ["loop_a", "loop_b"],
                                "Consumer": ["loop_b", "loop_a"],
                            },
                        ),
                    ],
                )
            )
            for _i in T.serial(4, annotations={T.WSID: "loop_a"}):
                T.copy(A[0, 0], s1, annotations={T.WSID: "copy_in1"})
                T.copy(s1, f1, annotations={T.WSID: "copy_frag1"})
                T.copy(f1, B[0, 0], annotations={T.WSID: "copy_out1"})
            for _i in T.serial(4, annotations={T.WSID: "loop_b"}):
                T.copy(A[4, 0], s2, annotations={T.WSID: "copy_in2"})
                T.copy(s2, f2, annotations={T.WSID: "copy_frag2"})
                T.copy(f2, B[4, 0], annotations={T.WSID: "copy_out2"})

    # Owner-only metadata remains unnecessary, but generic parity/capacity
    # deadlock validation must still reject this schedule.
    with pytest.raises(Exception, match="deadlocks"):
        _materialize(kernel, include_owner_capacity=False)


def test_legacy_two_owner_ring_keeps_original_counter_protocol():
    _ffi("tl.WSSyncWithGuard")
    func = _materialize(_ring_kernel(guarded=False))
    ring = _buffer(func, "ring")
    assert [int(dim) for dim in ring.shape] == [2, 4]
    _barrier(func, "ring_full", [32, 32])
    _barrier(func, "ring_empty", [128, 128])
    for first_thread in (0, 128):
        role = _role_body(func, first_thread)
        counter = _buffer(role, "ring_phase")
        stores = [
            store
            for store in collect_nodes(role, tirx.BufferStore)
            if store.buffer.data.same_as(counter.data)
        ]
        assert len(stores) == 3  # one initialization and one advance per owner
        assert len(collect_calls(role, op="tl.mbarrier_wait_parity")) == 2
        assert len(collect_calls(role, op="tirx.ptx_arrive_barrier")) == 2


def test_guarded_ring_before_after_ir_has_exact_sync_and_counter_structure():
    _ffi("tl.WSSyncWithGuard")
    before = _ring_kernel(guarded=True)
    schedule = next(
        block.annotations["tl.ws_schedule"]
        for block in collect_nodes(before, tirx.SBlock)
        if "tl.ws_schedule" in block.annotations
    )
    assert len(schedule.pipelines) == 1
    assert list(schedule.pipelines[0].owner_scopes) == ["owner0", "owner1"]
    assert all(
        instr.guard is not None
        for scope in schedule.scopes
        for body in scope.bodies.values()
        for instr in body
        if isinstance(instr, T.WSSync)
    )

    func = _materialize(before)
    assert tirx.analysis.verify_ssa(func)
    ring = _buffer(func, "ring")
    assert [int(dim) for dim in ring.shape] == [2, 4]
    _barrier(func, "ring_version_counter_1_full", [32, 32])
    _barrier(func, "ring_version_counter_1_empty", [128, 128])

    for first_thread, producer, barrier_name in (
        (0, True, "ring_version_counter_1_empty"),
        (128, False, "ring_version_counter_1_full"),
    ):
        role = _role_body(func, first_thread)
        counter = _buffer(role, "ring_version_counter_1")
        barrier = _buffer(func, barrier_name)
        binds = [bind for bind in collect_nodes(role, tirx.Bind) if bind.var.name.startswith("active")]
        assert len(binds) == 1  # fresh role-local scalar projection
        active = binds[0].var
        owners = [stmt for stmt in role.seq if isinstance(stmt, tirx.For)]
        assert [int(loop.extent) for loop in owners] == [2, 3]
        for owner in owners:
            (epoch,) = [
                stmt
                for stmt in owner.body.seq
                if isinstance(stmt, tirx.IfThenElse) and stmt.condition.same_as(active)
            ] if isinstance(owner.body, tirx.SeqStmt) else [owner.body]
            assert isinstance(epoch, tirx.IfThenElse) and epoch.condition.same_as(active)
            waits = [
                call
                for call in collect_calls(epoch.then_case, op="tl.mbarrier_wait_parity")
                if call.args[0].buffer.data.same_as(barrier.data)
            ]
            assert len(waits) == 1
            phase = tirx.BufferLoad(counter, [0])
            tvm.ir.assert_structural_equal(waits[0].args[0].indices[0], phase % 2)
            parity = tirx.bitwise_and(phase // 2, 1)
            if producer:
                parity = tirx.bitwise_xor(parity, 1)
            tvm.ir.assert_structural_equal(waits[0].args[1], parity)
            assert len(collect_calls(epoch.then_case, op="tirx.ptx_arrive_barrier")) == 1
            advances = [
                store
                for store in collect_nodes(epoch.then_case, tirx.BufferStore)
                if store.buffer.data.same_as(counter.data)
            ]
            assert len(advances) == 1
            tvm.ir.assert_structural_equal(advances[0].value, phase + 1)
            accesses = [
                load
                for load in collect_nodes(epoch.then_case, tirx.BufferLoad)
                if load.buffer.data.same_as(ring.data)
            ]
            assert accesses
            for load in accesses:
                assert len(load.indices) == 2
                tvm.ir.assert_structural_equal(load.indices[0], phase % 2)

        initializers = [
            store
            for store in collect_nodes(role, tirx.BufferStore)
            if store.buffer.data.same_as(counter.data)
            and isinstance(store.value, tirx.IntImm)
            and int(store.value) == 0
        ]
        assert len(initializers) == 1


if __name__ == "__main__":
    tilelang.testing.main()
