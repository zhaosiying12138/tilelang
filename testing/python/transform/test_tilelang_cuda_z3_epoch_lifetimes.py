"""Host-only proofs for CUDA Z3 epoch sync graphs and generation aliases.

The generic model is intentionally independent of WSSync and stock-pipeline IR.
Slice F will project those concrete sync sites into this contract. No build or
GPU is performed here; tests skip until the central library exports Slice H.
"""

from __future__ import annotations

import pytest
import tvm_ffi

import tilelang  # noqa: F401  # Load the TileLang shared library.
from tvm import tirx


def _ffi(name):
    function = tvm_ffi.get_global_func(name, allow_missing=True)
    if function is None:
        pytest.skip(f"{name} requires the centrally rebuilt TileLang library")
    return function


def _generation(begins, ends, *, guard=None, invariant=True):
    result = {
        "begins": [[site, offset] for site, offset in begins],
        "ends": [[site, offset] for site, offset in ends],
        "guard_iteration_invariant": invariant,
    }
    if guard is not None:
        result["guard"] = guard
    return result


def _storage(
    name,
    generations,
    *,
    domain=1,
    domain_kind="lexical",
    periodic=False,
    lexical_period=None,
):
    result = {
        "storage": tirx.Var(name, "handle"),
        "domain": domain,
        "domain_kind": domain_kind,
        "periodic": periodic,
        "generations": generations,
    }
    if lexical_period is not None:
        result["lexical_period"] = lexical_period
    return result


def _model(sites, edges, storages=(), *, domain=1):
    return {
        "sites": [{"domain": domain} for _ in range(sites)],
        "edges": [
            {
                "src": src,
                "dst": dst,
                "distance": distance,
                "strong": strong,
                "domain": domain,
            }
            for src, dst, distance, strong in edges
        ],
        "storages": list(storages),
    }


def _aliases(model):
    result = _ffi("tl.cuda.transform.Z3AnalyzeLifetimeModel")(model)
    return {str(key.name): {str(value.name) for value in values} for key, values in result.items()}


def _local_generations():
    # A: write 0, last read 1. B: write 2, last read 3.
    return _generation([(0, 0)], [(1, 0)]), _generation([(2, 0)], [(3, 0)])


def test_weak_only_dma_issue_order_cannot_authorize_alias():
    a, b = _local_generations()
    model = _model(
        4,
        [(1, 2, 0, False)],
        [_storage("a", [a]), _storage("b", [b])],
    )
    assert _aliases(model) == {"a": set(), "b": set()}


def test_strong_completion_through_another_role_authorizes_alias():
    a, b = _local_generations()
    # DMA issue 1 -> role-local handshake 4 -> completion 5 -> consumer issue 2.
    # Only the middle edge is strong; the composed path is sufficient.
    model = _model(
        6,
        [(1, 4, 0, False), (4, 5, 0, True), (5, 2, 0, False)],
        [_storage("a", [a]), _storage("b", [b])],
    )
    assert _aliases(model) == {"a": {"b"}, "b": {"a"}}


def test_nonperiodic_alias_requires_exactly_one_strong_direction():
    a, b = _local_generations()
    both_directions = _model(
        4,
        [(1, 2, 0, True), (3, 0, 0, True)],
        [_storage("a", [a]), _storage("b", [b])],
    )
    assert _aliases(both_directions) == {"a": set(), "b": set()}


def _periodic_model(edges, *, guards=None, invariant=True):
    a, b = _local_generations()
    if guards is not None:
        a = _generation([(0, 0)], [(1, 0)], guard=guards, invariant=invariant)
        b = _generation([(2, 0)], [(3, 0)], guard=guards, invariant=invariant)
    return _model(
        4,
        edges,
        [
            _storage("a", [a], periodic=True, lexical_period=2),
            _storage("b", [b], periodic=True, lexical_period=2),
        ],
    )


def test_local_ring_generations_use_two_separating_cuts():
    # ends(A(k)) < begins(B(k+1)); ends(B(k)) < begins(A(k)).
    model = _periodic_model([(1, 2, 1, True), (3, 0, 0, True)])
    assert _aliases(model) == {"a": {"b"}, "b": {"a"}}


def test_nested_carried_generation_offsets_participate_in_cut():
    # A is live in from the preceding nested epoch [-1, 0]. B is local. The
    # forward/reverse shifts are zero and one, so the carried lifetime is safe.
    carried = _generation([(0, -1)], [(1, 0)])
    local = _generation([(2, 0)], [(3, 0)])
    model = _model(
        4,
        [(1, 2, 0, True), (3, 0, 0, True)],
        [
            _storage("carried", [carried], periodic=True, lexical_period=2),
            _storage("local", [local], periodic=True, lexical_period=2),
        ],
    )
    assert _aliases(model) == {"carried": {"local"}, "local": {"carried"}}


def test_periodic_changing_guard_cannot_be_shifted_between_generations():
    iv = tirx.Var("i", "int32")
    model = _periodic_model(
        [(1, 2, 1, True), (3, 0, 0, True)], guards=iv >= 0, invariant=False
    )
    assert _aliases(model) == {"a": set(), "b": set()}


def test_iteration_invariant_mutual_exclusion_is_positive_nonperiodic_proof():
    predicate = tirx.Var("p", "int32")
    a = _generation([(0, 0)], [(1, 0)], guard=predicate > 0)
    b = _generation([(2, 0)], [(3, 0)], guard=predicate <= 0)
    model = _model(4, [], [_storage("a", [a]), _storage("b", [b])])
    assert _aliases(model) == {"a": {"b"}, "b": {"a"}}


def _inspect_cuts(edges, lhs, rhs, *, sites=4):
    spec = _model(sites, edges)
    spec.update({"lhs": lhs, "rhs": rhs, "domain": 1})
    return _ffi("tl.cuda.transform.Z3InspectSeparatingCuts")(spec)


def test_direct_separating_cut_valid_sum_and_invalid_greater_than_one():
    a, b = _local_generations()
    assert list(_inspect_cuts([(1, 2, 1, True), (3, 0, 0, True)], a, b)) == [
        1,
        0,
        1,
        True,
    ]
    assert list(_inspect_cuts([(1, 2, 2, True), (3, 0, 0, True)], a, b)) == [
        2,
        0,
        2,
        False,
    ]


def test_direct_separating_cut_rejects_missing_strong_path_and_unknown_distance():
    a, b = _local_generations()
    weak = list(_inspect_cuts([(1, 2, 1, False), (3, 0, 0, True)], a, b))
    assert weak[0] is None and weak[1] == 0 and weak[2] is None and not weak[3]
    unknown = list(_inspect_cuts([(1, 2, 1, True)], a, b))
    assert unknown[0] == 1 and unknown[1] is None and unknown[2] is None and not unknown[3]


def test_direct_separating_cut_rejects_int64_overflow():
    a = _generation([(0, 0)], [(1, (1 << 63) - 1)])
    b = _generation([(2, 0)], [(3, 0)])
    result = list(_inspect_cuts([(1, 2, 1, True), (3, 0, 0, True)], a, b))
    assert result[0] is None and result[2] is None and not result[3]


def test_counter_epoch_cannot_be_assigned_lexical_period():
    storage = _storage(
        "counter_storage",
        [],
        domain_kind="counter",
        periodic=True,
        lexical_period=2,
    )
    with pytest.raises(Exception, match="lexical iteration period|lexical period"):
        _aliases(_model(0, [], [storage]))


def test_present_empty_alias_annotation_disables_unproved_reuse():
    task = tirx.AttrStmt(
        {"stage": 0, "source_id": 1},
        "tl.schedule_unit",
        1,
        tirx.AttrStmt({}, "tl.cuda_task", 1, tirx.Evaluate(0)),
    )
    root = tirx.SBlock([], [], [], "tilelang_root", task)
    after = _ffi("tl.cuda.transform.Z3AttachLifetimeAliases")(root, _model(0, []))
    assert "tl.buffer_alias_map" in after.annotations
    assert len(after.annotations["tl.buffer_alias_map"]) == 0


@pytest.mark.skip(
    reason="Slice F owns WSSync/stock async projection into CudaSyncModel; "
    "the generic strong/weak proof engine is tested above"
)
def test_ws_and_stock_adapters_extract_actual_access_and_live_boundaries():
    """Slice F: connect real sync metadata without changing the proof engine."""
