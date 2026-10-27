"""Public CUDA scheduling annotations."""

from __future__ import annotations

from collections.abc import Mapping
import operator

from tvm.tirx import IntImm, Var
from tvm.tirx.script.builder.ir import sblock_attr

__all__ = ["annotate_buffer_versions"]

_BUFFER_VERSION_REQUESTS = "tl.cuda_z3_buffer_version_requests"
_BUFFER_VERSION_MODE_REQUESTS = "tl.cuda_z3_buffer_version_mode_requests"
_VERSION_MODES = frozenset(("auto", "iteration", "counter"))


def _positive_count(value, buffer):
    if isinstance(value, bool):
        raise TypeError(f"buffer version count for {buffer!r} must be a positive integer, got a boolean")
    try:
        count = operator.index(value)
    except TypeError as error:
        raise TypeError(f"buffer version count for {buffer!r} must be a positive integer, got {value!r}") from error
    if count <= 0:
        raise ValueError(f"buffer version count for {buffer!r} must be positive, got {count}")
    return count


def _mode(value, buffer):
    if not isinstance(value, str) or value not in _VERSION_MODES:
        raise ValueError(
            f"buffer version mode for {buffer!r} must be 'auto', 'iteration', or 'counter', got {value!r}"
        )
    return value


def annotate_buffer_versions(requests: Mapping):
    """Request fixed or inferred physical versions for CUDA Z3 pipeline buffers.

    The mapping accepts the following forms for each buffer::

        T.annotate_buffer_versions({
            buf: 1,                    # opt out of automatic versioning
            buf: N,                    # fixed N >= 2, automatic clock selection
            buf: "auto",              # infer the count and select a safe clock
            buf: "iteration",         # infer the count, require an affine clock
            buf: "counter",           # infer the count, use active-epoch counting
            buf: (N, "auto"),
            buf: (1, "auto"),        # eligible fixed one-version storage
        })

    Counts are exact. In particular, a fixed request is never raised to repair a
    schedule. A bare ``1`` is the sole opt-out spelling; ``(1, "auto")`` and
    ``(1, "iteration" | "counter")`` remain version-managed storage. Buffer
    aliases are canonicalized by ``Buffer.data`` and must not make conflicting
    requests.
    """
    if not isinstance(requests, Mapping):
        raise TypeError(f"annotate_buffer_versions expects a mapping, got {requests!r}")

    counts = {}
    modes = {}
    canonical = {}

    for buffer, request in requests.items():
        var = getattr(buffer, "data", None)
        if not isinstance(var, Var):
            raise TypeError(f"buffer version request keys must be Buffers with a .data Var, got {buffer!r}")

        count = None
        mode = None
        if isinstance(request, tuple):
            if len(request) != 2:
                raise ValueError(f"buffer version tuple for {buffer!r} must have arity two, got {request!r}")
            count = _positive_count(request[0], buffer)
            mode = _mode(request[1], buffer)
        elif isinstance(request, str):
            mode = _mode(request, buffer)
        else:
            count = _positive_count(request, buffer)

        previous = canonical.get(var)
        if previous is not None:
            old_count, old_mode = previous
            if old_count is not None and count is not None and old_count != count:
                raise ValueError(f"conflicting buffer version counts for aliases of Buffer.data Var {var}")
            if old_mode is not None and mode is not None and old_mode != mode:
                raise ValueError(f"conflicting buffer version modes for aliases of Buffer.data Var {var}")
            # A bare one is an opt-out, while an explicit one/mode tuple is an
            # eligible fixed request. Do not merge those different contracts.
            if old_count == count == 1 and (old_mode is None) != (mode is None):
                raise ValueError(f"bare-one opt-out conflicts with an eligible one-version request for {var}")
            count = old_count if count is None else count
            mode = old_mode if mode is None else mode

        canonical[var] = (count, mode)
        if count is None:
            counts.pop(var, None)
        else:
            counts[var] = IntImm("int64", count)
        if mode is None:
            modes.pop(var, None)
        else:
            modes[var] = mode

    return sblock_attr({
        _BUFFER_VERSION_REQUESTS: counts,
        _BUFFER_VERSION_MODE_REQUESTS: modes,
    })
