"""Host tests for public CUDA Z3 buffer-version requests (P13)."""

from __future__ import annotations

import pytest
import tvm_ffi

import tilelang as tl
from tilelang import tvm
import tilelang.cuda.language as T
from tilelang.cuda.language.annotations import annotate_buffer_versions
from tilelang.language.annotations import z3_pipeline_annotations
import tilelang.testing
from tilelang.testing.ir import collect_nodes


_FFI_NAME = "tl.cuda.transform.z3_schedule_loop_python"
_CONFIG = "tl.enable_z3_pipeline_planning"
_TASK_CONFIG = "tl.enable_z3_task_analysis"
_SM80 = tvm.target.Target({"kind": "cuda", "arch": "sm_80"})
_REQUESTS = "tl.cuda_z3_buffer_version_requests"
_MODE_REQUESTS = "tl.cuda_z3_buffer_version_mode_requests"
_SELECTED = "tl.cuda_z3_buffer_versions_selected"
_INTERNAL = "tl_pipeline_buffer_versions"


@pytest.fixture(autouse=True)
def _restore_solver_ffi():
    previous = tvm_ffi.get_global_func(_FFI_NAME, allow_missing=True)
    yield
    if tvm_ffi.get_global_func(_FFI_NAME, allow_missing=True) is not None:
        tvm_ffi.remove_global_func(_FFI_NAME)
    if previous is not None:
        tvm_ffi.register_global_func(_FFI_NAME, f=previous, override=True)


def _install_solver(result):
    calls = []

    def solver(*args):
        calls.append(args)
        return result(calls[-1]) if callable(result) else result

    tvm_ffi.register_global_func(_FFI_NAME, f=solver, override=True)
    return calls


def _public_annotations(counts=None, modes=None):
    return {
        _REQUESTS: {
            var: tvm.tirx.IntImm("int64", count) for var, count in (counts or {}).items()
        },
        _MODE_REQUESTS: modes or {},
    }


def _make_problem(request=None, *, data_deps=None, capacity_groups=None, infer=False):
    tir = tvm.tirx
    source = tir.decl_buffer((8,), "int32", name="version_source")
    dest = tir.decl_buffer((8,), "int32", name="version_dest")
    target = tir.decl_buffer((4,), "int32", name="version_target", scope="shared")
    counts, modes = request(source, dest, target) if request is not None else (None, None)
    k = tir.Var("k", "int32")
    annotations = z3_pipeline_annotations(
        latencies=[1, 1],
        iis=[1, 1],
        resource_flags=[0, 0],
        data_deps=[(0, 1, -1, 1)] if data_deps is None else data_deps,
        buffer_sizes=[16],
        buffer_vars=[target],
        memory_groups=[] if capacity_groups is None else capacity_groups,
    )
    annotations["num_stages"] = tir.IntImm("int32", 2)
    if infer:
        for key in (
            "tl_z3_pipeline_data_deps",
            "tl_z3_pipeline_buffer_sizes",
            "tl_z3_pipeline_buffer_vars",
        ):
            annotations.pop(key)
    loop = tir.For(
        k,
        0,
        4,
        tir.ForKind.SERIAL,
        tir.SeqStmt(
            [
                tir.BufferStore(target, tir.BufferLoad(source, [k]), [0]),
                tir.BufferStore(dest, tir.BufferLoad(target, [0]), [k]),
            ]
        ),
        annotations=annotations,
    )
    root = tir.SBlock(
        [],
        [],
        [],
        "version_root",
        loop,
        alloc_buffers=[target],
        annotations=_public_annotations(counts, modes),
    )
    func = tir.PrimFunc(
        [source.data, dest.data],
        root,
        buffer_map={source.data: source, dest.data: dest},
    )
    return func.with_attr("global_symbol", "main"), target


def _normalize(func):
    mod = tvm.IRModule.from_expr(func)
    mod = tvm.tirx.transform.BindTarget(_SM80)(mod)
    return tl.transform._ffi_api.NormalizeZ3BufferVersions()(mod)


def _plan(normalized):
    with tvm.transform.PassContext(config={_CONFIG: True}):
        return tl.transform.Z3PipelinePlanning()(normalized)


def _only_loop(func):
    loops = [loop for loop in collect_nodes(func, tvm.tirx.For) if "num_stages" in loop.annotations]
    assert len(loops) == 1
    return loops[0]


def _root(func):
    blocks = [block for block in collect_nodes(func, tvm.tirx.SBlock) if "version_root" == str(block.name_hint)]
    assert len(blocks) == 1
    return blocks[0]


def test_public_mapping_forms_and_staged_maps():
    @T.prim_func
    def kernel():
        with T.Kernel(1, threads=1) as _:
            bare = T.alloc_shared((4,), "int32")
            fixed = T.alloc_shared((4,), "int32")
            automatic = T.alloc_shared((4,), "int32")
            iteration_only = T.alloc_shared((4,), "int32")
            iteration = T.alloc_shared((4,), "int32")
            counter = T.alloc_shared((4,), "int32")
            eligible_one = T.alloc_shared((4,), "int32")
            T.annotate_buffer_versions(
                {
                    bare: 1,
                    fixed: 2,
                    automatic: "auto",
                    iteration_only: "iteration",
                    iteration: (3, "iteration"),
                    counter: "counter",
                    eligible_one: (1, "auto"),
                }
            )

    root = next(
        block
        for block in collect_nodes(kernel, tvm.tirx.SBlock)
        if _REQUESTS in block.annotations
    )
    counts = root.annotations[_REQUESTS]
    modes = root.annotations[_MODE_REQUESTS]
    by_name = {str(buffer.name): buffer.data for buffer in root.alloc_buffers}
    assert int(counts[by_name["bare"]]) == 1
    assert int(counts[by_name["fixed"]]) == 2
    assert int(counts[by_name["iteration"]]) == 3
    assert int(counts[by_name["eligible_one"]]) == 1
    assert by_name["automatic"] not in counts
    assert by_name["iteration_only"] not in counts
    assert by_name["counter"] not in counts
    assert by_name["bare"] not in modes
    assert by_name["fixed"] not in modes
    assert modes[by_name["automatic"]] == "auto"
    assert modes[by_name["iteration_only"]] == "iteration"
    assert modes[by_name["iteration"]] == "iteration"
    assert modes[by_name["counter"]] == "counter"
    assert modes[by_name["eligible_one"]] == "auto"
    assert T.annotate_buffer_versions is annotate_buffer_versions


@pytest.mark.parametrize("version_request", [0, -1, True, 1.5])
def test_public_count_validation(version_request):
    buffer = tvm.tirx.decl_buffer((4,), "int32", scope="shared")
    with pytest.raises((TypeError, ValueError), match="positive integer|must be positive"):
        annotate_buffer_versions({buffer: version_request})


@pytest.mark.parametrize("version_request", [(2,), (2, "auto", 1), (2, "invalid"), "invalid"])
def test_public_tuple_and_mode_validation(version_request):
    buffer = tvm.tirx.decl_buffer((4,), "int32", scope="shared")
    with pytest.raises(ValueError):
        annotate_buffer_versions({buffer: version_request})


def test_public_alias_conflicts_and_consistency():
    tir = tvm.tirx
    data = tir.Var("shared_alias", tvm.ir.PointerType(tvm.ir.PrimType("int32")))
    first = tir.decl_buffer((4,), "int32", data=data, scope="shared")
    second = tir.decl_buffer((4,), "int32", data=data, scope="shared")
    @T.prim_func
    def consistent_alias_kernel():
        with T.Kernel(1, threads=1) as _:
            annotate_buffer_versions({first: 2, second: (2, "counter")})

    assert consistent_alias_kernel is not None
    with pytest.raises(ValueError, match="conflicting buffer version counts"):
        annotate_buffer_versions({first: 2, second: 3})
    with pytest.raises(ValueError, match="conflicting buffer version modes"):
        annotate_buffer_versions({first: "auto", second: "counter"})
    with pytest.raises(ValueError, match="bare-one opt-out"):
        annotate_buffer_versions({first: 1, second: (1, "auto")})


def test_fixed_count_is_exact_and_excluded_from_solver():
    func, target = _make_problem(lambda _source, _dest, buffer: ({buffer.data: 2}, None))
    calls = _install_solver(([0, 1], [], 1))
    planned = _plan(_normalize(func))

    loop = _only_loop(planned["main"])
    assert int(loop.annotations[_INTERNAL][target.data]) == 2
    assert int(_root(planned["main"]).annotations[_SELECTED][target.data]) == 2
    solver = calls[0]
    assert list(solver[8]) == []
    assert list(solver[9]) == []
    assert [list(map(int, dep)) for dep in solver[4]] == [[0, 1, 2, 1]]
    assert int(loop.annotations["tl_z3_audit_version_selected"][target.data]) == 2
    assert int(loop.annotations["tl_z3_audit_version_required"][target.data]) == 1


def test_fixed_request_is_normalized_before_task_inference():
    func, target = _make_problem(
        lambda _source, _dest, buffer: ({buffer.data: 2}, None), infer=True
    )
    calls = _install_solver(([0, 1], [], 1))
    normalized = _normalize(func)
    with tvm.transform.PassContext(config={_CONFIG: True, _TASK_CONFIG: True}):
        inferred = tl.transform.InferZ3PipelineProblem()(normalized)
        planned = tl.transform.Z3PipelinePlanning()(inferred)
    inferred_loop = _only_loop(inferred["main"])
    assert list(inferred_loop.annotations["tl_z3_pipeline_buffer_vars"]) == [target.data]
    assert int(_only_loop(planned["main"]).annotations[_INTERNAL][target.data]) == 2
    assert list(calls[0][8]) == []
    assert [list(map(int, dep)) for dep in calls[0][4]] == [
        [0, 0, 2, 1],
        [0, 1, 0, 1],
        [0, 1, 2, 1],
        [1, 0, 2, 1],
    ]


def test_bare_one_opts_out_but_eligible_one_is_selected():
    func, target = _make_problem(lambda _source, _dest, buffer: ({buffer.data: 1}, None))
    calls = _install_solver(([0, 1], [], 1))
    planned = _plan(_normalize(func))
    loop = _only_loop(planned["main"])
    assert int(loop.annotations[_INTERNAL][target.data]) == 1
    assert target.data not in _root(planned["main"]).annotations[_SELECTED]
    assert [list(map(int, dep)) for dep in calls[0][4]] == [[0, 1, 1, 1]]

    func, target = _make_problem(
        lambda _source, _dest, buffer: ({buffer.data: 1}, {buffer.data: "auto"})
    )
    _install_solver(([0, 1], [], 1))
    planned = _plan(_normalize(func))
    loop = _only_loop(planned["main"])
    assert int(loop.annotations[_INTERNAL][target.data]) == 1
    assert int(_root(planned["main"]).annotations[_SELECTED][target.data]) == 1
    assert loop.annotations["tl_z3_audit_version_mode_request"][target.data] == "auto"


def test_mode_only_request_uses_solver_and_preserves_two_sources():
    func, target = _make_problem(
        lambda _source, _dest, buffer: (None, {buffer.data: "counter"})
    )
    calls = _install_solver(([0, 1], [3], 1))
    planned = _plan(_normalize(func))
    loop = _only_loop(planned["main"])
    assert list(map(int, calls[0][8])) == [16]
    assert [list(map(int, dep)) for dep in calls[0][4]] == [[0, 1, -1, 1]]
    assert int(loop.annotations[_INTERNAL][target.data]) == 3
    assert int(loop.annotations["tl_z3_audit_version_selected"][target.data]) == 3
    assert int(loop.annotations["tl_z3_audit_version_required"][target.data]) == 1
    assert loop.annotations["tl_z3_audit_version_mode_request"][target.data] == "counter"


def test_fixed_under_version_is_rejected_after_schedule():
    func, target = _make_problem(lambda _source, _dest, buffer: ({buffer.data: 2}, None))
    _install_solver(([2, 0], [], 1))
    with pytest.raises(Exception, match="fixed buffer version request.*under-versioned"):
        _plan(_normalize(func))


def test_unknown_and_out_of_scope_requests_are_rejected():
    tir = tvm.tirx
    unknown = tir.decl_buffer((4,), "int32", name="unknown", scope="shared")
    func, _ = _make_problem(lambda _source, _dest, _buffer: ({unknown.data: 2}, None))
    with pytest.raises(Exception, match="unknown Buffer.data Var"):
        _normalize(func)

    func, _ = _make_problem(lambda source, _dest, _buffer: ({source.data: 2}, None))
    with pytest.raises(Exception, match="outside automatic on-chip version scopes"):
        _normalize(func)


def test_request_selected_internal_cleanup_and_second_pass_noop():
    func, target = _make_problem(lambda _source, _dest, buffer: ({buffer.data: 2}, None))
    _install_solver(([0, 1], [], 1))
    planned = _plan(_normalize(func))
    root = _root(planned["main"])
    assert _REQUESTS not in root.annotations
    assert _MODE_REQUESTS not in root.annotations
    assert "tl_z3_audit_version_request" in root.annotations
    assert _SELECTED in root.annotations

    with tvm.transform.PassContext(config={_CONFIG: True}):
        projected = tl.transform.PipelinePlanning()(planned)
        injected = tl.transform.InjectSoftwarePipeline()(projected)
        again = tl.transform.InjectSoftwarePipeline()(injected)
    for loop in collect_nodes(injected["main"], tvm.tirx.For):
        assert _REQUESTS not in loop.annotations
        assert _SELECTED not in loop.annotations
        assert _INTERNAL not in loop.annotations
    tvm.ir.assert_structural_equal(injected, again, True)


if __name__ == "__main__":
    tilelang.testing.main()
