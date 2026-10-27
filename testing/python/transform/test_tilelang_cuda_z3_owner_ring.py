"""Host-only tests for the restricted two-owner warp-specialized ring.

The input is explicit typed WSSchedule metadata. These tests do not exercise
owner inference, Z3 planning, CUDA execution, or the stock software pipeline.
"""

from __future__ import annotations

import pytest

import tilelang
import tilelang.testing
from tilelang import language as T
from tilelang import tvm
from tilelang.testing.ir import collect_calls, collect_nodes
from tvm import tirx


_ARCH = "sm_100a"
_LARGE_CAPACITY = 1 << 20


def _target(shared_capacity=_LARGE_CAPACITY):
    return tvm.target.Target(
        {
            "kind": "cuda",
            "arch": _ARCH,
            "max_shared_memory_per_block": shared_capacity,
        }
    )


def _materialize(func, *, shared_capacity=_LARGE_CAPACITY):
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.tirx.transform.BindTarget(_target(shared_capacity))(mod)
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    transformed = tilelang.cuda.transform.MaterializeWSSchedule()(mod)["main"]
    markers = {"tl.ws_schedule", "tl.ws_op_id"}
    for node in collect_nodes(transformed, tvm.runtime.Object):
        assert markers.isdisjoint(getattr(node, "annotations", {}))
        if isinstance(node, tirx.AttrStmt):
            assert node.attr_key not in markers
    return transformed


def _buffers(root, name):
    result = [buf for block in collect_nodes(root, tirx.SBlock) for buf in block.alloc_buffers if buf.name == name]
    result.extend(buf for buf in (node.buffer for node in collect_nodes(root, tirx.AllocBuffer)) if buf.name == name)
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
        if "barrier_init" in block.annotations and buffer.data in block.annotations["barrier_init"]
    ]
    assert len(initializers) == 1, f"expected one initialization site for {name}"
    assert [int(value) for value in initializers[0]] == counts
    assert [int(dim) for dim in buffer.shape] == [len(counts)]
    return buffer


def _role_branches(func):
    (launch,) = [
        attr
        for attr in collect_nodes(func, tirx.AttrStmt)
        if attr.attr_key == "thread_extent"
        and isinstance(attr.node, tirx.IterVar)
        and attr.node.thread_tag == "threadIdx.x"
    ]
    (custom,) = [
        attr for attr in collect_nodes(launch.body, tirx.AttrStmt) if attr.attr_key == "kCustomWarpSpecialization"
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


def _owner_loops(role):
    assert isinstance(role, tirx.SeqStmt)
    loops = [stmt for stmt in role.seq if isinstance(stmt, tirx.For)]
    assert len(loops) == 2
    return loops


def _sync_body(owner, stage=0, read_first=False):
    return {
        "Producer": [
            T.WSSync.producer_acquire("ring", stage=stage),
            *([f"read{owner}"] if read_first else []),
            f"produce{owner}",
            T.WSSync.producer_commit("ring", stage=stage),
        ],
        "Consumer": [
            T.WSSync.consumer_wait("ring", stage=stage),
            f"consume{owner}",
            T.WSSync.consumer_release("ring", stage=stage),
            f"output{owner}",
        ],
    }


def _owner_ring_kernel(
    *,
    depth=2,
    extents=(3, 5),
    owner_scopes=("owner0", "owner1"),
    variant=None,
    extra_shared=False,
):
    bodies0 = _sync_body(0, read_first=variant == "read_first")
    bodies1 = _sync_body(1, stage=1 if variant == "stage_offsets" else 0)
    root = {"Producer": ["owner0", "owner1"], "Consumer": ["owner0", "owner1"]}
    if variant == "external":
        root["Producer"].insert(0, "external")
    if variant == "different_order":
        root["Consumer"].reverse()
    if variant == "nested":
        for role in ("Producer", "Consumer"):
            bodies0[role].append("owner1")
        root = {"Producer": ["owner0"], "Consumer": ["owner0"]}

    roles = [
        T.WSRole("Producer", warps_lo=0, warps_hi=1),
        T.WSRole("Consumer", warps_lo=4, warps_hi=8),
    ]
    if variant == "three_roles":
        roles.insert(1, T.WSRole("Extra", warps_lo=1, warps_hi=2))

    @T.prim_func
    def kernel(A: T.Tensor((64,), T.float32), B: T.Tensor((64,), T.float32)):
        with T.Kernel(1, threads=128) as bx:
            ring = T.alloc_shared((4,), T.float32, scope="shared")
            frag = T.alloc_fragment((4,), T.float32)
            if extra_shared:
                T.alloc_shared((1,), T.float32, scope="shared")
            if variant == "private":
                pipeline_buffer = T.alloc_shared((4,), T.float32, scope="local")
            elif variant == "global":
                pipeline_buffer = T.alloc_shared((4,), T.float32, scope="global")
            elif variant == "cross_cta":
                pipeline_buffer = T.alloc_cluster_barrier(1)
            else:
                pipeline_buffer = ring

            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=roles,
                    pipelines=[T.WSPipeline("ring", [pipeline_buffer], depth=depth, owner_scopes=owner_scopes)],
                    scopes=[
                        T.WSScope("owner0", bodies0),
                        T.WSScope("owner1", bodies1),
                        T.WSScope(T.WSScope.ROOT, root),
                    ],
                )
            )

            if variant == "external":
                T.fill(ring, 0, annotations={T.WSID: "external"})

            if variant == "nested":
                for i in T.serial(extents[0], annotations={T.WSID: "owner0"}):
                    T.fill(ring, i, annotations={T.WSID: "produce0"})
                    T.copy(ring, frag, annotations={T.WSID: "consume0"})
                    T.copy(frag, B[i * 4], annotations={T.WSID: "output0"})
                    for j in T.serial(extents[1], annotations={T.WSID: "owner1"}):
                        T.fill(ring, j, annotations={T.WSID: "produce1"})
                        T.copy(ring, frag, annotations={T.WSID: "consume1"})
                        T.copy(frag, B[(extents[0] + j) * 4], annotations={T.WSID: "output1"})
            else:
                for i in T.serial(extents[0], annotations={T.WSID: "owner0"}):
                    if variant == "async":
                        T.async_copy(A[i * 4], ring, annotations={T.WSID: "produce0"})
                    else:
                        if variant == "read_first":
                            T.copy(ring, frag, annotations={T.WSID: "read0"})
                        T.fill(ring, i, annotations={T.WSID: "produce0"})
                    T.copy(ring, frag, annotations={T.WSID: "consume0"})
                    T.copy(frag, B[i * 4], annotations={T.WSID: "output0"})

                if variant == "guarded":
                    if bx == 0:
                        for j in T.serial(extents[1], annotations={T.WSID: "owner1"}):
                            T.fill(ring, j, annotations={T.WSID: "produce1"})
                            T.copy(ring, frag, annotations={T.WSID: "consume1"})
                            T.copy(frag, B[(extents[0] + j) * 4], annotations={T.WSID: "output1"})
                else:
                    for j in T.serial(extents[1], annotations={T.WSID: "owner1"}):
                        if variant == "async":
                            T.async_copy(A[(extents[0] + j) * 4], ring, annotations={T.WSID: "produce1"})
                        elif variant == "mismatched_arrivals":
                            T.fill(frag, j, annotations={T.WSID: "produce1"})
                        else:
                            T.fill(ring, j, annotations={T.WSID: "produce1"})
                        T.copy(ring, frag, annotations={T.WSID: "consume1"})
                        T.copy(frag, B[(extents[0] + j) * 4], annotations={T.WSID: "output1"})

    return kernel


def _assert_persistent_counter(role, counter, depth, producer, barrier):
    stores = [store for store in collect_nodes(role, tirx.BufferStore) if store.buffer.data.same_as(counter.data)]
    initializers = [store for store in stores if isinstance(store.value, tirx.IntImm) and int(store.value) == 0]
    assert len(initializers) == 1
    assert len(stores) == 3  # one initialization and one update in each owner
    assert isinstance(role.seq[0], tirx.AllocBuffer) and role.seq[0].buffer.data.same_as(counter.data)
    assert role.seq[1].same_as(initializers[0])
    expected_update = tirx.BufferLoad(counter, [0]) + 1
    for store in stores:
        if not store.same_as(initializers[0]):
            tvm.ir.assert_structural_equal(store.value, expected_update)

    phase = tirx.BufferLoad(counter, [0])
    loops = _owner_loops(role)
    for loop in loops:
        (wait,) = [
            call
            for call in collect_calls(loop.body, op="tl.mbarrier_wait_parity")
            if call.args[0].buffer.data.same_as(barrier.data)
        ]
        tvm.ir.assert_structural_equal(wait.args[0].indices[0], phase % depth)
        parity = tirx.bitwise_and(phase // depth, 1)
        if producer:
            parity = tirx.bitwise_xor(parity, 1)
        tvm.ir.assert_structural_equal(wait.args[1], parity)
        assert not any(load.buffer.data.same_as(counter.data) for load in collect_nodes(loop.loop_var, tirx.BufferLoad))


def test_owner_scopes_reflection_and_legacy_constructor():
    buffer = tirx.decl_buffer((4,), "float32", name="ring")
    legacy = T.WSPipeline("legacy", [buffer], depth=2)
    explicit_empty = T.WSPipeline("legacy", [buffer], depth=2, owner_scopes=())
    ring = T.WSPipeline("ring", [buffer], depth=2, owner_scopes=("owner0", "owner1"))
    assert list(legacy.owner_scopes) == []
    assert list(explicit_empty.owner_scopes) == []
    assert list(ring.owner_scopes) == ["owner0", "owner1"]
    tvm.ir.assert_structural_equal(legacy, explicit_empty)


def test_two_owners_one_ring_and_continuous_depth2_counters():
    func = _materialize(_owner_ring_kernel(depth=2, extents=(3, 5)))
    ring = _buffer(func, "pipeline_buffer")
    full = _barrier(func, "ring_full", [32, 32])
    empty = _barrier(func, "ring_empty", [128, 128])
    assert len(_buffers(func, "ring_full")) == 1
    assert len(_buffers(func, "ring_empty")) == 1
    assert [int(dim) for dim in ring.shape] == [2, 4]

    for first_thread, producer, barrier in ((0, True, empty), (128, False, full)):
        role = _role_body(func, first_thread)
        counter = _buffer(role, "ring_phase")
        assert counter.scope() == "local"
        _assert_persistent_counter(role, counter, 2, producer, barrier)
        loops = _owner_loops(role)
        assert [int(loop.extent) for loop in loops] == [3, 5]
        for loop in loops:
            accesses = [load for load in collect_nodes(loop.body, tirx.BufferLoad) if load.buffer.data.same_as(ring.data)]
            assert accesses
            for load in accesses:
                assert len(load.indices) == 2
                tvm.ir.assert_structural_equal(load.indices[0], tirx.BufferLoad(counter, [0]) % 2)
        # No full-CTA barrier is inserted into a role-exclusive path.
        assert all(call.op.name != "tirx.ptx_barrier" for call in collect_nodes(role, tirx.Call))


def test_depth1_parity_and_single_version_dimension():
    func = _materialize(_owner_ring_kernel(depth=1))
    ring = _buffer(func, "pipeline_buffer")
    assert [int(dim) for dim in ring.shape] == [1, 4]
    full = _barrier(func, "ring_full", [32])
    empty = _barrier(func, "ring_empty", [128])
    for first_thread, producer, barrier in ((0, True, empty), (128, False, full)):
        role = _role_body(func, first_thread)
        counter = _buffer(role, "ring_phase")
        _assert_persistent_counter(role, counter, 1, producer, barrier)
        for loop in _owner_loops(role):
            accesses = [load for load in collect_nodes(loop.body, tirx.BufferLoad) if load.buffer.data.same_as(ring.data)]
            assert accesses
            for load in accesses:
                assert len(load.indices) == 2
                tvm.ir.assert_structural_equal(load.indices[0], tirx.IntImm("int32", 0))


def test_arrival_counts_and_allocation_lifetime():
    func = _materialize(_owner_ring_kernel(depth=2))
    for name in ("pipeline_buffer", "ring_full", "ring_empty"):
        buffer = _buffer(func, name)
        allocations = [
            block
            for block in collect_nodes(func, tirx.SBlock)
            if any(buf.data.same_as(buffer.data) for buf in block.alloc_buffers)
        ]
        assert len(allocations) == 1
    for first_thread in (0, 128):
        role = _role_body(func, first_thread)
        counter = _buffer(role, "ring_phase")
        assert isinstance(role.seq[0], tirx.AllocBuffer)
        assert role.seq[0].buffer.data.same_as(counter.data)
        assert all(not isinstance(stmt, tirx.AllocBuffer) for stmt in role.seq[1:])
        assert len(collect_calls(role, op="tl.mbarrier_wait_parity")) == 2
        assert len(collect_calls(role, op="tirx.ptx_arrive_barrier")) == 2


@pytest.mark.parametrize(("depth", "required_bytes"), [(1, 48), (2, 64), (3, 112)])
def test_global_shared_capacity_includes_aligned_allocations_and_barriers(depth, required_bytes):
    # CUDA codegen aligns every static shared allocation to 16 bytes.
    _materialize(_owner_ring_kernel(depth=depth), shared_capacity=required_bytes)
    with pytest.raises(Exception, match="exceeding target capacity"):
        _materialize(_owner_ring_kernel(depth=depth), shared_capacity=required_bytes - 1)
    # One unrelated shared float32 also consumes a 16-byte aligned allocation.
    with pytest.raises(Exception, match="exceeding target capacity"):
        _materialize(_owner_ring_kernel(depth=depth, extra_shared=True), shared_capacity=required_bytes)


@pytest.mark.parametrize(
    ("variant", "owner_scopes", "message"),
    [
        ("legacy_two_scopes", (), "synchronizes exactly one scope"),
        ("three_owners", ("owner0", "owner1", "owner2"), "exactly two ordered owner scopes"),
        ("different_order", ("owner0", "owner1"), "declared order"),
        ("external", ("owner0", "owner1"), "outside both declared owners"),
        ("guarded", ("owner0", "owner1"), "guarded owner-ring epochs"),
        ("read_first", ("owner0", "owner1"), "read-first owner-ring epoch"),
        ("stage_offsets", ("owner0", "owner1"), "stage-zero epochs"),
        ("mismatched_arrivals", ("owner0", "owner1"), "mismatched arrival signature"),
        ("private", ("owner0", "owner1"), "CTA-local shared storage"),
        ("global", ("owner0", "owner1"), "CTA-local shared storage"),
        ("cross_cta", ("owner0", "owner1"), "CTA-local shared storage"),
        ("nested", ("owner0", "owner1"), "ordered siblings"),
        ("async", ("owner0", "owner1"), "asynchronous atom"),
        ("three_roles", ("owner0", "owner1"), "exactly two fixed nonoverlapping warp roles"),
    ],
)
def test_unsupported_owner_ring_cases_rejected(variant, owner_scopes, message):
    owners = () if variant == "legacy_two_scopes" else owner_scopes
    with pytest.raises(Exception, match=message):
        _materialize(_owner_ring_kernel(owner_scopes=owners, variant=variant))


if __name__ == "__main__":
    tilelang.testing.main()
