"""Host-only IR tests for explicit software-pipeline buffer versions."""

import pytest

import tilelang as tl
from tilelang import tvm
import tilelang.testing
from tilelang.testing.ir import collect_nodes


_BUFFER_VERSIONS = "tl_pipeline_buffer_versions"


def _versions(**versions):
    def annotate(buffers):
        return {
            buffers[name].data: tvm.tirx.IntImm("int32", count)
            for name, count in versions.items()
        }

    return annotate


def _make_pipeline(*, extent=8, allocation="outer", scope="shared", annotate=None, include_unused=False):
    tir = tvm.tirx
    source = tir.decl_buffer((extent, 4), "float32", name="source")
    dest = tir.decl_buffer((extent, 4), "float32", name="dest")
    target = tir.decl_buffer((4,), "float32", name="target", scope=scope)
    unknown = tir.decl_buffer((4,), "float32", name="unknown", scope=scope)
    unused = tir.decl_buffer((4,), "float32", name="unused", scope=scope)
    k = tir.Var("k", "int32")

    annotations = {
        "software_pipeline_stage": [0, 1],
        "software_pipeline_order": [0, 1],
    }
    buffers = {"target": target, "unknown": unknown, "unused": unused}
    if annotate is not None:
        annotations[_BUFFER_VERSIONS] = annotate(buffers)

    loop_body = []
    if allocation == "local":
        loop_body.append(tir.AllocBuffer(target))
    else:
        assert allocation == "outer"
    loop_body.extend(
        [
            tir.BufferStore(target, tir.BufferLoad(source, [k, 0]), [0]),
            tir.BufferStore(dest, tir.BufferLoad(target, [0]), [k, 0]),
        ]
    )
    loop = tir.For(
        k,
        0,
        extent,
        tir.ForKind.SERIAL,
        tir.SeqStmt(loop_body),
        annotations=annotations,
    )

    if allocation == "outer":
        body = [tir.AllocBuffer(target)]
        if include_unused:
            body.append(tir.AllocBuffer(unused))
        body.append(loop)
        body = tir.SeqStmt(body)
    else:
        assert not include_unused
        body = loop

    func = tir.PrimFunc(
        [source.data, dest.data],
        body,
        buffer_map={source.data: source, dest.data: dest},
    )
    return func.with_attr("global_symbol", "main"), buffers


def _make_multi_buffer_pipeline():
    tir = tvm.tirx
    extent = 8
    source = tir.decl_buffer((extent, 4), "float32", name="source")
    dest = tir.decl_buffer((extent, 4), "float32", name="dest")
    single = tir.decl_buffer((4,), "float32", name="single", scope="shared")
    ring = tir.decl_buffer((4,), "float32", name="ring", scope="local.fragment")
    automatic = tir.decl_buffer((4,), "float32", name="automatic", scope="shared")
    buffers = {"single": single, "ring": ring, "automatic": automatic}
    k = tir.Var("k", "int32")
    annotations = {
        "software_pipeline_stage": [0, 0, 0, 0, 1, 1],
        "software_pipeline_order": [0, 1, 2, 3, 4, 5],
        _BUFFER_VERSIONS: _versions(single=2, ring=3)(buffers),
    }
    statements = [
        # ``single`` is written and read in stage zero, so auto inference keeps
        # one version. The explicit override must raise that to two.
        tir.BufferStore(single, tir.BufferLoad(source, [k, 0]), [0]),
        tir.BufferStore(dest, tir.BufferLoad(single, [0]), [k, 0]),
        tir.BufferStore(ring, tir.BufferLoad(source, [k, 1]), [0]),
        tir.BufferStore(automatic, tir.BufferLoad(source, [k, 2]), [0]),
        tir.BufferStore(dest, tir.BufferLoad(ring, [0]), [k, 1]),
        tir.BufferStore(dest, tir.BufferLoad(automatic, [0]), [k, 2]),
    ]
    loop = tir.For(
        k,
        0,
        extent,
        tir.ForKind.SERIAL,
        tir.SeqStmt(statements),
        annotations=annotations,
    )
    body = tir.SeqStmt([*(tir.AllocBuffer(buf) for buf in buffers.values()), loop])
    func = tir.PrimFunc(
        [source.data, dest.data],
        body,
        buffer_map={source.data: source, dest.data: dest},
    )
    return func.with_attr("global_symbol", "main"), buffers


def _run(func):
    mod = tvm.IRModule.from_expr(func)
    return tl.transform.InjectSoftwarePipeline()(mod)["main"]


def _find_allocation(func, original):
    tir = tvm.tirx
    allocations = [node.buffer for node in collect_nodes(func, tir.AllocBuffer)]
    allocations.extend(
        buf
        for block in collect_nodes(func, tir.SBlock)
        for buf in block.alloc_buffers
    )
    matches = [buf for buf in allocations if str(buf.name) == str(original.name)]
    assert len(matches) == 1, f"Expected one allocation for {original.name}, got {matches}"
    return matches[0]


def _assert_buffer_version(func, original, versions, *, require_modulus=True):
    tir = tvm.tirx
    allocation = _find_allocation(func, original)
    expected_shape = [4] if versions == 1 else [versions, 4]
    assert [int(dim) for dim in allocation.shape] == expected_shape
    assert allocation.scope() == original.scope()

    accesses = [
        node
        for node_type in (tir.BufferLoad, tir.BufferStore)
        for node in collect_nodes(func, node_type)
        if str(node.buffer.name) == str(original.name)
    ]
    assert accesses
    analyzer = tvm.arith.Analyzer()
    moduli = set()
    saw_symbolic_version = False
    for node in accesses:
        assert node.buffer.same_as(allocation)
        assert node.buffer.scope() == original.scope()
        if versions == 1:
            assert len(node.indices) == 1
            continue

        assert len(node.indices) == 2
        version = analyzer.simplify(node.indices[0])
        if isinstance(version, tir.IntImm):
            assert 0 <= int(version) < versions
            continue

        saw_symbolic_version = True
        mods = collect_nodes(version, tir.FloorMod)
        assert mods, f"Expected a version modulo in {version}"
        for mod in mods:
            assert int(mod.b) == versions
            moduli.add(int(mod.b))

    if versions > 1 and require_modulus:
        assert saw_symbolic_version
        assert moduli == {versions}


def _assert_annotation_removed(func):
    for loop in collect_nodes(func, tvm.tirx.For):
        assert _BUFFER_VERSIONS not in loop.annotations


@pytest.mark.parametrize("versions", [1, 2, 3])
@pytest.mark.parametrize("allocation", ["outer", "local"])
@pytest.mark.parametrize("scope", ["shared", "local.fragment"])
def test_explicit_buffer_version_override(versions, allocation, scope):
    func, buffers = _make_pipeline(
        allocation=allocation,
        scope=scope,
        annotate=_versions(target=versions),
    )
    after = _run(func)

    _assert_buffer_version(after, buffers["target"], versions)
    _assert_annotation_removed(after)


def test_multiple_buffers_mix_overrides_with_auto_inference():
    func, buffers = _make_multi_buffer_pipeline()
    after = _run(func)

    _assert_buffer_version(after, buffers["single"], 2)
    _assert_buffer_version(after, buffers["ring"], 3)
    _assert_buffer_version(after, buffers["automatic"], 2)
    _assert_annotation_removed(after)


def test_short_loop_keeps_explicit_ring_and_bounds():
    func, buffers = _make_pipeline(extent=2, annotate=_versions(target=3))
    after = _run(func)

    _assert_buffer_version(after, buffers["target"], 3, require_modulus=False)
    _assert_annotation_removed(after)


def test_default_no_annotation_is_unchanged_and_matches_empty_override():
    func, buffers = _make_pipeline()
    after = _run(func)

    # This two-stage RAW pipeline is double-buffered by the existing heuristic.
    _assert_buffer_version(after, buffers["target"], 2)
    _assert_annotation_removed(after)

    empty_func, _ = _make_pipeline(annotate=lambda _: {})
    empty_after = _run(empty_func)
    tvm.ir.assert_structural_equal(after, empty_after, True)


def test_annotation_is_consumed_and_second_pass_does_not_reapply_override():
    func, buffers = _make_pipeline(annotate=_versions(target=3))
    after = _run(func)
    _assert_buffer_version(after, buffers["target"], 3)
    _assert_annotation_removed(after)

    again = _run(after)
    _assert_buffer_version(again, buffers["target"], 3)
    _assert_annotation_removed(again)
    tvm.ir.assert_structural_equal(after, again, True)


def test_unknown_buffer_is_rejected():
    func, _ = _make_pipeline(annotate=_versions(unknown=2))
    with pytest.raises(tvm.TVMError, match="unknown Buffer.data Var"):
        _run(func)


def test_buffer_outside_current_pipeline_is_rejected():
    func, _ = _make_pipeline(
        include_unused=True,
        annotate=_versions(unused=2),
    )
    with pytest.raises(tvm.TVMError, match="not an allocated buffer in the current pipeline"):
        _run(func)


def test_ambiguous_current_buffer_mapping_is_rejected():
    tir = tvm.tirx
    source = tir.decl_buffer((8, 4), "float32", name="source")
    dest = tir.decl_buffer((8, 4), "float32", name="dest")
    data = tir.Var("shared_target_data", tvm.ir.PointerType(tvm.ir.PrimType("float32")))
    first = tir.decl_buffer((4,), "float32", name="first", data=data, scope="shared")
    second = tir.decl_buffer((4,), "float32", name="second", data=data, scope="shared")
    k = tir.Var("k", "int32")
    loop = tir.For(
        k,
        0,
        8,
        tir.ForKind.SERIAL,
        tir.SeqStmt(
            [
                tir.BufferStore(first, tir.BufferLoad(source, [k, 0]), [0]),
                tir.BufferStore(dest, tir.BufferLoad(second, [0]), [k, 0]),
            ]
        ),
        annotations={
            "software_pipeline_stage": [0, 1],
            "software_pipeline_order": [0, 1],
            _BUFFER_VERSIONS: {data: tir.IntImm("int32", 2)},
        },
    )
    func = tir.PrimFunc(
        [source.data, dest.data],
        tir.SeqStmt([tir.AllocBuffer(first), tir.AllocBuffer(second), loop]),
        buffer_map={source.data: source, dest.data: dest},
    ).with_attr("global_symbol", "main")

    with pytest.raises(tvm.TVMError, match="duplicate or ambiguous buffer mappings"):
        _run(func)


@pytest.mark.parametrize("versions", [0, -2])
def test_nonpositive_version_is_rejected(versions):
    func, _ = _make_pipeline(annotate=_versions(target=versions))
    with pytest.raises(tvm.TVMError, match="must be positive"):
        _run(func)


@pytest.mark.parametrize(
    "kind,error",
    [
        ("container", "must be Map<Var, IntImm>"),
        ("key", "non-Var key"),
        ("value", "IntImm version count"),
    ],
)
def test_invalid_override_mapping_is_rejected(kind, error):
    if kind == "container":
        annotate = lambda buffers: [buffers["target"].data]
    elif kind == "key":
        annotate = lambda _: {"target": tvm.tirx.IntImm("int32", 2)}
    else:
        annotate = lambda buffers: {buffers["target"].data: tvm.tirx.StringImm("two")}

    func, _ = _make_pipeline(annotate=annotate)
    with pytest.raises(tvm.TVMError, match=error):
        _run(func)


if __name__ == "__main__":
    tilelang.testing.main()
