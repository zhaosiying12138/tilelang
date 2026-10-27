"""Host-only tests for the nested CUDA Z3 schedule IR and shared primitives.

These tests construct canonical scheduled TIR directly. They do not run a CUDA
pass, invoke the solver, compile a kernel, or require a GPU. The internal FFI
inspection entry points become available when the central build includes
src/cuda/transform/z3_schedule_ir.cc.
"""

from __future__ import annotations

import pytest
import tvm_ffi

import tilelang  # noqa: F401  # Load the TileLang shared library.
from tilelang import language as T
from tilelang import tvm
from tilelang.testing.ir import collect_nodes
from tvm import tirx


def _ffi(name):
    function = tvm_ffi.get_global_func(name, allow_missing=True)
    if function is None:
        pytest.skip(f"{name} requires the centrally rebuilt TileLang library")
    return function


def _role_words(mask):
    return (mask & 0xFFFFFFFF, (mask >> 32) & 0xFFFFFFFF)


def _unit(
    body,
    *,
    stage=-1,
    source_id,
    cost=None,
    per_role=False,
    declaration=False,
    solver_eligible=True,
    candidate_roles=0,
    resolved_roles=0,
    ws_op_id=None,
    guard=None,
    assume=None,
):
    metadata = {}
    if cost is not None:
        latency, ii = cost
        if latency is not None:
            metadata["latency"] = tirx.IntImm("int64", latency)
        if ii is not None:
            metadata["ii"] = tirx.IntImm("int64", ii)
    if declaration:
        metadata["declaration"] = tirx.IntImm("int32", 1)
    if not solver_eligible:
        metadata["solver_eligible"] = tirx.IntImm("int32", 0)
    for name, mask in (("candidate_roles", candidate_roles), ("resolved_roles", resolved_roles)):
        if mask:
            low, high = _role_words(mask)
            metadata[f"{name}_lo"] = tirx.IntImm("int64", low)
            metadata[f"{name}_hi"] = tirx.IntImm("int64", high)
    if ws_op_id is not None:
        metadata["ws_op_id"] = ws_op_id
    body = tirx.AttrStmt(metadata, "tl.cuda_per_role_task" if per_role else "tl.cuda_task", 1, body)
    if assume is not None:
        body = tirx.AttrStmt(assume, "tl.assume", 1, body)
    if guard is not None:
        body = tirx.IfThenElse(guard, body, None)
    unit_metadata = {"source_id": tirx.IntImm("int32", source_id)}
    if stage is not None:
        unit_metadata["stage"] = tirx.IntImm("int32", stage)
    return tirx.AttrStmt(unit_metadata, "tl.schedule_unit", 1, body)


def _control(loop_var, extent, children, *, stage, source_id, ws_op_id, step=None):
    loop = tirx.For(
        loop_var,
        0,
        extent,
        tirx.ForKind.SERIAL,
        tirx.SeqStmt(children),
        annotations={"tl.ws_op_id": ws_op_id},
        step=step,
    )
    unit_metadata = {
        "stage": tirx.IntImm("int32", stage),
        "source_id": tirx.IntImm("int32", source_id),
    }
    return tirx.AttrStmt(unit_metadata, "tl.schedule_unit", 1, loop)


def _nested_root():
    storage = tirx.decl_buffer((8,), "int32", name="storage", scope="shared.dyn")
    local = tirx.decl_buffer((2,), "int32", name="local", scope="local.var")
    old = tirx.Var("old", "int32")
    i = tirx.Var("i", "int32")
    j = tirx.Var("j", "int32")

    high_bit = 1 << 63
    bind = _unit(
        tirx.Bind(old, 7),
        stage=0,
        source_id=10,
        cost=(4, 2),
        solver_eligible=False,
        candidate_roles=high_bit | (1 << 2),
        resolved_roles=high_bit,
        ws_op_id="bind",
    )
    declaration = _unit(
        tirx.DeclBuffer(local),
        stage=-1,
        source_id=11,
        declaration=True,
        solver_eligible=False,
    )
    guarded_store = _unit(
        tirx.BufferStore(storage, old + j, [i]),
        stage=-1,
        source_id=12,
        guard=old > 0,
        assume=old >= 0,
        ws_op_id="guarded_store",
    )
    local_store = _unit(
        tirx.BufferStore(local, storage[i], [j]),
        stage=-1,
        source_id=13,
        ws_op_id="local_store",
    )
    inner = _control(j, 2, [guarded_store, local_store], stage=-1, source_id=14, ws_op_id="inner")
    tail = _unit(tirx.BufferStore(storage, i, [0]), stage=-1, source_id=15, ws_op_id="tail")
    outer = _control(i, 3, [declaration, inner, tail], stage=0, source_id=16, ws_op_id="outer")
    return tirx.SBlock(
        [],
        [],
        [],
        "tilelang_root",
        tirx.SeqStmt([bind, outer]),
        alloc_buffers=[storage, local],
        annotations={
            "tl.buffer_versions_map": {storage.data: 2},
            "tl.cuda_z3_selected_buffer_versions": {storage.data: 2},
            "tl.buffer_version_mode": {storage.data: "auto"},
            # Presence with an empty map is a valid positive-alias analysis result.
            "tl.buffer_alias_map": {},
        },
    )


def _schedule_units(root):
    return [node for node in collect_nodes(root, tirx.AttrStmt) if node.attr_key == "tl.schedule_unit"]


def _task_markers(root):
    return [
        node
        for node in collect_nodes(root, tirx.AttrStmt)
        if node.attr_key in ("tl.cuda_task", "tl.cuda_per_role_task")
    ]


def test_codec_round_trip_preserves_nested_tree_and_metadata():
    root = _nested_root()
    after = _ffi("tl.cuda.transform.Z3ScheduleIRRoundTrip")(root)
    tvm.ir.assert_structural_equal(after, root)

    assert after.annotations["tl.buffer_versions_map"][next(iter(after.annotations["tl.buffer_versions_map"]))] == 2
    assert "tl.cuda_z3_selected_buffer_versions" in after.annotations
    assert "tl.buffer_alias_map" in after.annotations
    assert len(after.annotations["tl.buffer_alias_map"]) == 0

    markers = _task_markers(after)
    declaration = [marker for marker in markers if int(marker.node.get("declaration", 0)) == 1]
    replayable = [marker for marker in markers if isinstance(marker.body, tirx.Bind)]
    assert len(declaration) == 1
    assert len(replayable) == 1
    assert int(declaration[0].node["solver_eligible"]) == 0
    assert int(replayable[0].node["solver_eligible"]) == 0

    (bind,) = [marker for marker in markers if marker.node.get("ws_op_id", "") == "bind"]
    assert int(bind.node["candidate_roles_lo"]) == 1 << 2
    assert int(bind.node["candidate_roles_hi"]) == 1 << 31
    assert int(bind.node["resolved_roles_lo"]) == 0
    assert int(bind.node["resolved_roles_hi"]) == 1 << 31
    assert int(bind.node["latency"]) == 4
    assert int(bind.node["ii"]) == 2


def test_clone_is_deep_and_substitution_leaves_source_unchanged():
    root = _nested_root()
    cloned = _ffi("tl.cuda.transform.Z3ScheduleIRCloneRoundTrip")(root)
    tvm.ir.assert_structural_equal(cloned, root)

    substituted = _ffi("tl.cuda.transform.Z3ScheduleIRSubstituteRoundTrip")(
        root, "old", "fresh", True
    )
    original_names = {var.name for var in collect_nodes(root, tirx.Var)}
    substituted_names = {var.name for var in collect_nodes(substituted, tirx.Var)}
    assert "old" in original_names
    assert "old" not in substituted_names
    assert "fresh" in substituted_names
    # Clone-first substitution must not mutate the source tree's guards or Bind.
    assert "old" in {var.name for var in collect_nodes(cloned, tirx.Var)}
    fresh_bind = [marker.body for marker in _task_markers(substituted) if isinstance(marker.body, tirx.Bind)]
    assert len(fresh_bind) == 1
    assert fresh_bind[0].var.name == "fresh"


def test_paths_lca_comparison_and_symbolic_iteration_count():
    summary = _ffi("tl.cuda.transform.Z3ScheduleIRInspect")(_nested_root())
    assert summary["task_count"] == 5  # Four operations plus the declaration task.
    assert summary["task_with_header_count"] == 7
    assert summary["control_count"] == 2
    assert summary["null_root_compare"] == -2
    assert summary["self_compare"] == 0
    assert summary["sibling_compare"] == -1
    assert summary["ancestor_compare"] == -2
    assert summary["descendant_compare"] == 2
    assert summary["path_size"] == 2
    assert summary["child_on_path_valid"] == 1
    assert summary["lca_is_control"] == 1
    assert summary["header_parent_valid"] == 1
    assert summary["control_read_regions"] > 0
    assert summary["control_write_regions"] > 0
    assert summary["body_has_loop_break"] == 0
    assert summary["body_touches_storage"] == 1
    iteration = str(summary["iteration_count"])
    assert "i" in iteration and "j" in iteration
    assert "2" in iteration  # The inner extent is a multiplier, not a dynamic guess.


def test_stage_defaults_are_independent_per_child_list():
    a = tirx.decl_buffer((1,), "int32", name="a")
    first = _unit(tirx.BufferStore(a, 1, [0]), stage=2, source_id=1)
    second = _unit(tirx.BufferStore(a, 2, [0]), stage=-1, source_id=2)
    i = tirx.Var("i", "int32")
    child0 = _unit(tirx.BufferStore(a, i, [0]), stage=-1, source_id=3)
    child1 = _unit(tirx.BufferStore(a, i + 1, [0]), stage=-1, source_id=4)
    control = _control(i, 2, [child0, child1], stage=1, source_id=5, ws_op_id="control")
    root = tirx.SBlock([], [], [], "tilelang_root", tirx.SeqStmt([first, second, control]))
    after = _ffi("tl.cuda.transform.Z3ScheduleIRRoundTrip")(root)

    top_units = list(after.body.seq)
    assert [int(unit.node["stage"]) for unit in top_units] == [2, 0, 1]
    nested_loop = top_units[2].body
    nested_units = list(nested_loop.body.seq)
    assert [int(unit.node["stage"]) for unit in nested_units] == [-1, -1]


def test_declaration_may_omit_stage_and_per_role_marker_round_trips():
    local = tirx.decl_buffer((1,), "int32", name="local", scope="local.var")
    declaration = _unit(
        tirx.DeclBuffer(local),
        stage=None,
        source_id=1,
        declaration=True,
        solver_eligible=False,
    )
    per_role = _unit(tirx.Evaluate(0), stage=0, source_id=2, per_role=True)
    root = tirx.SBlock([], [], [], "tilelang_root", tirx.SeqStmt([declaration, per_role]))
    after = _ffi("tl.cuda.transform.Z3ScheduleIRRoundTrip")(root)
    units = list(after.body.seq)
    assert int(units[0].node["stage"]) == -1
    assert units[1].body.attr_key == "tl.cuda_per_role_task"


def test_latency_only_cost_does_not_acquire_zero_ii():
    buffer = tirx.decl_buffer((1,), "int32", name="buffer")
    task = _unit(tirx.BufferStore(buffer, 0, [0]), stage=0, source_id=1, cost=(3, None))
    root = tirx.SBlock([], [], [], "tilelang_root", task)
    after = _ffi("tl.cuda.transform.Z3ScheduleIRRoundTrip")(root)
    (marker,) = _task_markers(after)
    assert int(marker.node["latency"]) == 3
    assert "ii" not in marker.node


def test_loop_break_state_is_local_to_its_direct_control():
    i = tirx.Var("i", "int32")
    break_task = _unit(tirx.Evaluate(T.loop_break()), stage=-1, source_id=1)
    sibling = _unit(tirx.Evaluate(0), stage=-1, source_id=2)
    control = _control(i, 2, [break_task, sibling], stage=0, source_id=3, ws_op_id="control")
    root = tirx.SBlock([], [], [], "tilelang_root", control)
    summary = _ffi("tl.cuda.transform.Z3ScheduleIRInspect")(root)
    assert summary["body_has_loop_break"] == 1


@pytest.mark.parametrize(
    "operation, first, second, expected",
    [
        ("seq", 1, 3, 3),  # A conditional write does not cover a later read.
        ("seq", 2, 3, 2),  # A guaranteed write determines the first access.
        ("seq", 1, 0, 1),
        ("if_else", 2, 2, 2),
        ("if_else", 2, 0, 1),
        ("if_only", 0, 0, 0),
        ("demote", 2, 0, 1),
        ("demote", 3, 1, 3),
    ],
)
def test_write_first_lattice(operation, first, second, expected):
    actual = _ffi("tl.cuda.transform.Z3WriteFirstLattice")(operation, first, second)
    assert actual == expected


def test_iteration_order_requires_a_strong_completion_edge():
    minimum = _ffi("tl.cuda.transform.Z3MinimumIterationDistance")
    # weak 0->1, strong 1->2, weak 0->2, strong 2->3.
    arguments = (4, [0, 1, 0, 2], [1, 2, 2, 3], [1, 2, 0, 1], [0, 1, 0, 1])
    assert minimum(*arguments, 0, 1) is None
    assert minimum(*arguments, 0, 2) == 3  # weak 0->1 plus strong 1->2; direct 0->2 is weak.
    assert minimum(*arguments, 1, 3) == 3
    assert minimum(*arguments, 0, 3) == 1  # Weak issue order may precede a strong completion edge.
    assert minimum(1, [0], [0], [2], [1], 0, 0) == 2
