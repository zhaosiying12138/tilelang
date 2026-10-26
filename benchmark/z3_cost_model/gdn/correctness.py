"""GPU correctness checks against the vendored pure-PyTorch float64 reference.

The checks intentionally follow upstream ``test_gdr_unit.py``: relative L2 error
with RTOL=0.02 and repeated-run determinism checks.  Additional assertions cover
zero-filled backing-buffer padding, final_state without h0, and dh0 with h0 but
without dht.  No function in this module is used by ``--check-configs``.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    from .config import Direction, GDNConfig, default_config
    from .runner import (
        GDNRunner,
        load_test_support,
        make_fixed_inputs,
    )
except ImportError:  # Direct execution with this directory on sys.path.
    from config import Direction, GDNConfig, default_config
    from runner import GDNRunner, load_test_support, make_fixed_inputs

RTOL = 0.02
DETERMINISM_ITERS = 1000
CORE_CASES = (
    ("dense-no-h0-no-dht", {
        "num_tokens": 65,
        "use_h0": False,
        "use_dht": False,
    }),
    ("dense-h0-no-dht", {
        "num_tokens": 65,
        "use_h0": True,
        "use_dht": False,
    }),
    ("padded-h0-dht", {
        "num_tokens": 96,
        "varlen": True,
        "cu_seqlens_list": [0, 37, 71],
        "use_h0": True,
        "use_dht": True,
    }),
)


def reference_forward(batch):
    gdr_common, ref_gdr = load_test_support()
    return ref_gdr.chunk_gated_delta_rule_fwd(
        q=batch.q.to(gdr_common.REF_DTYPE, copy=True),
        k=batch.k.to(gdr_common.REF_DTYPE, copy=True),
        v=batch.v.to(gdr_common.REF_DTYPE, copy=True),
        g=batch.g.to(gdr_common.REF_DTYPE, copy=True),
        beta=batch.beta.to(gdr_common.REF_DTYPE, copy=True),
        scale=batch.scale,
        initial_state=batch.h0_ref,
        cu_seqlens=batch.cu_seqlens,
        chunk_size=gdr_common.CHUNK_SIZE,
    )


def reference_backward(batch, expected_forward):
    gdr_common, ref_gdr = load_test_support()
    _, _, A_ref, _, _ = expected_forward
    g_ref = expected_forward[0]
    return ref_gdr.chunk_gated_delta_rule_bwd(
        q=batch.q.to(gdr_common.REF_DTYPE, copy=True),
        k=batch.k.to(gdr_common.REF_DTYPE, copy=True),
        v=batch.v.to(gdr_common.REF_DTYPE, copy=True),
        g=g_ref,
        beta=batch.beta.to(gdr_common.REF_DTYPE, copy=True),
        A=A_ref.to(gdr_common.REF_DTYPE, copy=True),
        scale=batch.scale,
        initial_state=batch.h0_ref,
        do=batch.do.to(gdr_common.REF_DTYPE, copy=True),
        dht=batch.dht_ref,
        cu_seqlens=batch.cu_seqlens,
        chunk_size=gdr_common.CHUNK_SIZE,
    )


def _state_for_reference(state, state_v_first):
    return state.transpose(-1, -2) if state_v_first else state


def assert_padded_zeros(batch, tensor, name: str) -> None:
    """Check the complete backing-buffer tail, not merely absence of NaNs."""

    if batch.cu_seqlens is None:
        return
    valid_tokens = int(batch.cu_seqlens[-1].item())
    if valid_tokens == tensor.shape[1]:
        return
    tail = tensor[:, valid_tokens:]
    assert bool((tail == 0).all().item()), (
        f"{name}: backing-buffer padding [{valid_tokens}:{tensor.shape[1]}] "
        "contains a nonzero value"
    )


def assert_forward(batch, actual, expected, *, output_h=False, suffix=""):
    gdr_common, _ = load_test_support()
    _, o_ref, _, h_ref, final_state_ref = expected
    gdr_common.assert_relative(actual.o, o_ref, f"o{suffix}", rtol=RTOL)
    # Unlike the source test, validate final_state even when h0 is absent.
    gdr_common.assert_relative(
        _state_for_reference(actual.final_state, batch.state_v_first),
        final_state_ref,
        f"final_state{suffix}",
        rtol=RTOL,
    )
    if output_h:
        gdr_common.assert_relative(
            _state_for_reference(actual.h, batch.state_v_first),
            h_ref,
            f"h{suffix}",
            rtol=RTOL,
        )
    assert_padded_zeros(batch, actual.o, f"o{suffix}")


def assert_backward(batch, actual, expected, *, suffix=""):
    gdr_common, _ = load_test_support()
    names = ("dq", "dk", "dv", "db", "dg")
    for name, expected_value in zip(names, expected[:5]):
        actual_value = getattr(actual, name)
        gdr_common.assert_relative(
            actual_value, expected_value, f"{name}{suffix}", rtol=RTOL
        )
        assert_padded_zeros(batch, actual_value, f"{name}{suffix}")

    dh0_ref = expected[5]
    if batch.h0_ref is None:
        assert actual.dh0 is None, "dh0 must be None when initial_state is absent"
        assert dh0_ref is None
    else:
        # This deliberately does not depend on dht: h0 without dht still has dh0.
        gdr_common.assert_relative(
            _state_for_reference(actual.dh0, batch.state_v_first),
            dh0_ref,
            f"dh0{suffix}",
            rtol=RTOL,
        )


def check_forward(runner: GDNRunner, batch, *, output_h=True):
    expected = reference_forward(batch)
    actual = runner.forward(batch, output_h=output_h)
    assert_forward(batch, actual, expected, output_h=output_h)
    return actual, expected


def check_backward(runner: GDNRunner, batch):
    expected_forward = reference_forward(batch)
    actual_forward = runner.forward(batch, output_h=False)
    expected = reference_backward(batch, expected_forward)
    actual = runner.backward(batch, actual_forward)
    assert_backward(batch, actual, expected)
    return actual, expected


def check_determinism(
    runner: GDNRunner,
    batch,
    direction: Direction | str,
    *,
    iterations: int = DETERMINISM_ITERS,
) -> None:
    """Repeat the public pipeline and compare every result to float64, as upstream."""

    if iterations < 0:
        raise ValueError("iterations must be >= 0")
    direction = Direction(direction)
    expected_forward = reference_forward(batch)
    if direction is Direction.FWD:
        for index in range(iterations):
            actual = runner.forward(batch, output_h=False)
            assert_forward(
                batch, actual, expected_forward, suffix=f" iter {index}"
            )
        return

    actual_forward = runner.forward(batch, output_h=False)
    expected = reference_backward(batch, expected_forward)
    for index in range(iterations):
        actual = runner.backward(batch, actual_forward)
        assert_backward(batch, actual, expected, suffix=f" iter {index}")


@dataclass(frozen=True)
class CorrectnessSummary:
    directions: tuple[str, ...]
    cases: tuple[str, ...]
    seed: int
    rtol: float
    determinism_iterations: int

    def as_dict(self):
        return {
            "status": "passed",
            "directions": list(self.directions),
            "cases": list(self.cases),
            "seed": self.seed,
            "rtol": self.rtol,
            "reference": "pure PyTorch float64",
            "determinism_iterations": self.determinism_iterations,
            "auto_cp": False,
        }


def run_core_correctness(
    direction: Direction | str,
    config: GDNConfig | None = None,
    *,
    seed: int = 42,
    determinism_iterations: int = 0,
) -> CorrectnessSummary:
    """Run dense, h0/no-dht, and padded cases for one or both directions."""

    requested = direction.value if isinstance(direction, Direction) else str(direction)
    if requested == "all":
        directions = (Direction.FWD, Direction.BWD)
    else:
        directions = (Direction(direction),)
    runner = GDNRunner(default_config(directions[0]) if config is None else config)

    for case_name, case_kwargs in CORE_CASES:
        batch = make_fixed_inputs(seed=seed, **case_kwargs)
        for current in directions:
            if current is Direction.FWD:
                check_forward(runner, batch)
            else:
                check_backward(runner, batch)
            if determinism_iterations:
                check_determinism(
                    runner,
                    batch,
                    current,
                    iterations=determinism_iterations,
                )
        del case_name

    return CorrectnessSummary(
        tuple(current.value for current in directions),
        tuple(name for name, _ in CORE_CASES),
        seed,
        RTOL,
        determinism_iterations,
    )
