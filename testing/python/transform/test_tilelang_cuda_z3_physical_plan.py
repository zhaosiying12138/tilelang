"""Host-only exact physical-plan validation for recursive CUDA Z3 schedules.

These tests supply post-layout allocation records directly. Slice G/I will
project final TIR allocations into this contract; legacy allocator behavior is
not changed. No build or GPU execution is performed by this file.
"""

from __future__ import annotations

import pytest
import tvm_ffi

import tilelang  # noqa: F401
from tilelang import tvm
from tvm import tirx


def _ffi(name):
    function = tvm_ffi.get_global_func(name, allow_missing=True)
    if function is None:
        pytest.skip(f"{name} requires the centrally rebuilt TileLang library")
    return function


def _target(capacity=4096):
    return tvm.target.Target(
        {
            "kind": "cuda",
            "arch": "sm_90",
            "max_shared_memory_per_block": capacity,
        }
    )


def _expr(value):
    return value if isinstance(value, (tirx.PrimExpr, tirx.Var)) else tirx.IntImm("int64", value)


def _expectation(name, allocation_id, *, versions=1, unversioned_bytes=4):
    return {
        "storage": tirx.Var(name, "handle"),
        "allocation_id": allocation_id,
        "versions": versions,
        "unversioned_bytes": unversioned_bytes,
    }


def _allocation(
    name,
    allocation_id,
    *,
    kind="selected",
    extents=(1,),
    offset=0,
    byte_size=4,
    alignment=1,
    contracts=None,
):
    return {
        "storage": tirx.Var(name, "handle"),
        "allocation_id": allocation_id,
        "kind": kind,
        "leading_version_extents": list(extents),
        "byte_offset": _expr(offset),
        "byte_size": _expr(byte_size),
        "required_alignment": alignment,
        "alias_contracts": contracts or {},
    }


def _validate(expected, allocations, *, arena_bytes, aliases=None, capacity=4096):
    return _ffi("tl.transform.ValidateZ3PhysicalAllocationContracts")(
        expected,
        aliases or {},
        {"allocations": allocations, "arena_bytes": _expr(arena_bytes)},
        _target(capacity),
    )


def test_holes_sub_byte_rounding_fixed_and_barrier_allocations_fit_exact_capacity():
    expected = [_expectation("selected", 10, versions=2, unversioned_bytes=3)]
    allocations = [
        _allocation("selected_flat", 10, extents=(2,), byte_size=6, alignment=16),
        _allocation("fixed", 11, kind="fixed", extents=(), offset=16, byte_size=5),
        _allocation("barrier", 12, kind="barrier", extents=(), offset=32, byte_size=8),
    ]
    # Holes [6,16) and [21,32) are legal. The arena total includes every kind.
    _validate(expected, allocations, arena_bytes=40, capacity=40)
    with pytest.raises(Exception, match="capacity|exceeding"):
        _validate(expected, allocations, arena_bytes=40, capacity=39)


def test_stable_allocation_id_not_final_var_identity_is_provenance_key():
    expected = [_expectation("logical_storage", 7)]
    # Layout/flattening may replace Buffer.data. Stable ID 7 still resolves the
    # original positive-alias and version contract.
    allocations = [_allocation("remapped_physical_storage", 7)]
    _validate(expected, allocations, arena_bytes=4)


@pytest.mark.parametrize(
    "actual_id,message",
    [(99, "Unknown or stale|provenance"), (1, "Missing|provenance")],
)
def test_reject_unknown_stale_and_missing_provenance(actual_id, message):
    expected = [_expectation("a", 1)]
    allocations = [] if actual_id == 1 else [_allocation("stale", actual_id)]
    with pytest.raises(Exception, match=message):
        _validate(expected, allocations, arena_bytes=4)


def test_reject_duplicate_physical_allocation_ids():
    expected = [_expectation("a", 1)]
    allocations = [_allocation("a0", 1), _allocation("a1", 1)]
    with pytest.raises(Exception, match="Duplicate physical allocation ID"):
        _validate(expected, allocations, arena_bytes=8)


@pytest.mark.parametrize("offset,alignment", [(3, 4), (6, 16)])
def test_reject_unaligned_offsets(offset, alignment):
    expected = [_expectation("a", 1)]
    allocations = [_allocation("a", 1, offset=offset, alignment=alignment)]
    with pytest.raises(Exception, match="unaligned offset|alignment"):
        _validate(expected, allocations, arena_bytes=16)


def test_reject_out_of_range_and_overflowing_allocation_ranges():
    expected = [_expectation("a", 1)]
    with pytest.raises(Exception, match="outside arena"):
        _validate(expected, [_allocation("a", 1, offset=3, byte_size=4)], arena_bytes=6)
    with pytest.raises(Exception, match="overflows int64"):
        _validate(
            expected,
            [_allocation("a", 1, offset=(1 << 63) - 2, byte_size=4)],
            arena_bytes=(1 << 63) - 1,
        )


@pytest.mark.parametrize(
    "versions,extents,byte_size",
    [(2, (2, 2), 16), (2, (1,), 4), (2, (3,), 12)],
)
def test_reject_double_under_and_over_version_expansion(versions, extents, byte_size):
    expected = [_expectation("a", 1, versions=versions, unversioned_bytes=4)]
    allocations = [_allocation("a", 1, extents=extents, byte_size=byte_size)]
    with pytest.raises(Exception, match="version-dimension product"):
        _validate(expected, allocations, arena_bytes=byte_size)


def test_extent_one_pseudo_version_cannot_be_silently_dropped():
    expected = [_expectation("a", 1, versions=1)]
    with pytest.raises(Exception, match="no leading version dimension"):
        _validate(expected, [_allocation("a", 1, extents=())], arena_bytes=4)


def test_reject_incorrect_physical_size_even_when_version_product_matches():
    expected = [_expectation("a", 1, versions=2, unversioned_bytes=3)]
    with pytest.raises(Exception, match="Physical byte size"):
        _validate(expected, [_allocation("a", 1, extents=(2,), byte_size=7)], arena_bytes=7)


def test_reject_dynamic_unknown_size_instead_of_converting_it_to_zero():
    expected = [_expectation("a", 1)]
    dynamic = tirx.Var("dynamic_size", "int64")
    with pytest.raises(Exception, match="dynamic values"):
        _validate(expected, [_allocation("a", 1, byte_size=dynamic)], arena_bytes=4)


def test_reject_unproved_overlap():
    expected = [_expectation("a", 1), _expectation("b", 2)]
    allocations = [_allocation("a", 1), _allocation("b", 2)]
    with pytest.raises(Exception, match="Unproved overlap"):
        _validate(expected, allocations, arena_bytes=4)


def _generation(begins, ends):
    return {"begins": begins, "ends": ends, "guard_iteration_invariant": True}


def _cut_alias_proof(strong):
    a = tirx.Var("a", "handle")
    b = tirx.Var("b", "handle")
    generations = [
        _generation([[0, 0]], [[1, 0]]),
        _generation([[2, 0]], [[3, 0]]),
    ]
    storages = []
    for storage, generation in zip((a, b), generations):
        storages.append(
            {
                "storage": storage,
                "domain": 1,
                "domain_kind": "lexical",
                "periodic": True,
                "lexical_period": 2,
                "generations": [generation],
            }
        )
    model = {
        "sites": [{"domain": 1} for _ in range(4)],
        "edges": [
            {"src": 1, "dst": 2, "distance": 1, "strong": strong, "domain": 1},
            {"src": 3, "dst": 0, "distance": 0, "strong": strong, "domain": 1},
        ],
        "storages": storages,
    }
    aliases = _ffi("tl.cuda.transform.Z3AnalyzeLifetimeModel")(model)
    return a, b, aliases


def test_positive_separating_cut_and_matching_contract_permit_reuse():
    a, b, aliases = _cut_alias_proof(strong=True)
    expected = [
        {
            "storage": a,
            "allocation_id": 1,
            "versions": 1,
            "unversioned_bytes": 4,
        },
        {
            "storage": b,
            "allocation_id": 2,
            "versions": 1,
            "unversioned_bytes": 4,
        },
    ]
    allocations = [
        _allocation("a_flat", 1, contracts={2: "cuts:forward=1,reverse=0"}),
        _allocation("b_flat", 2, contracts={1: "cuts:forward=1,reverse=0"}),
    ]
    _validate(expected, allocations, arena_bytes=4, aliases=aliases)


def test_same_program_with_weak_only_order_is_rejected():
    a, b, aliases = _cut_alias_proof(strong=False)
    assert len(aliases) == 2 and all(len(values) == 0 for values in aliases.values())
    expected = [
        {"storage": a, "allocation_id": 1, "versions": 1, "unversioned_bytes": 4},
        {"storage": b, "allocation_id": 2, "versions": 1, "unversioned_bytes": 4},
    ]
    allocations = [
        _allocation("a_flat", 1, contracts={2: "weak-is-not-a-proof"}),
        _allocation("b_flat", 2, contracts={1: "weak-is-not-a-proof"}),
    ]
    with pytest.raises(Exception, match="Unproved overlap"):
        _validate(expected, allocations, arena_bytes=4, aliases=aliases)


def test_positive_alias_map_does_not_hide_mismatched_actual_generation_contracts():
    a, b, aliases = _cut_alias_proof(strong=True)
    expected = [
        {"storage": a, "allocation_id": 1, "versions": 1, "unversioned_bytes": 4},
        {"storage": b, "allocation_id": 2, "versions": 1, "unversioned_bytes": 4},
    ]
    allocations = [
        _allocation("a_flat", 1, contracts={2: "generation-7"}),
        _allocation("b_flat", 2, contracts={1: "generation-8"}),
    ]
    with pytest.raises(Exception, match="Mismatched actual generation/lifetime"):
        _validate(expected, allocations, arena_bytes=4, aliases=aliases)


@pytest.mark.skip(
    reason="Slice G/I owns final TIR allocation/layout projection into "
    "SharedMemoryArenaPlan; the exact validator is tested directly above"
)
def test_final_merged_tir_consumes_physical_contract_after_layout_and_flattening():
    """Slice G/I: thread stable IDs through all physical rewrites and invoke once."""
