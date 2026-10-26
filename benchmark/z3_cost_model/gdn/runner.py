"""Factories for complete public FlashQLA forward and backward orchestration.

Nothing in this module calls a raw fused kernel directly.  Imports of Torch,
TileLang, and the vendored package are lazy so the configuration CLI remains
CPU-only.  The fixed input fixture is the upstream ``tests/gdr_common.py``
generator, including its seed and distributions.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .config import (
        CORE_CONTRACT,
        Direction,
        GDNConfig,
        Status,
        default_config,
        evaluate_config,
    )
except ImportError:  # Direct execution with this directory on sys.path.
    from config import (
        CORE_CONTRACT,
        Direction,
        GDNConfig,
        Status,
        default_config,
        evaluate_config,
    )

GDN_ROOT = Path(__file__).resolve().parent
VENDOR_ROOT = GDN_ROOT / "vendor"
TEST_ROOT = GDN_ROOT / "tests"
_UNSET = object()


def _prepend_path(path: Path) -> None:
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


def _module_within(module, root: Path) -> bool:
    filename = getattr(module, "__file__", None)
    return filename is not None and Path(filename).resolve().is_relative_to(root)


def _import_owned(module_name: str, root: Path):
    """Import from an owned directory, never silently mix an installed copy."""

    _prepend_path(root)
    existing = sys.modules.get(module_name)
    if existing is not None and not _module_within(existing, root):
        raise ImportError(
            f"{module_name!r} was already imported from {existing.__file__!r}; "
            f"start a fresh process so the vendored copy under {root} is used."
        )
    module = importlib.import_module(module_name)
    if not _module_within(module, root):
        raise ImportError(
            f"Resolved {module_name!r} to {module.__file__!r}, outside {root}."
        )
    return module


def load_public_api():
    """Load the documented public low-level API from this directory's vendor tree."""

    module = _import_owned("flash_qla.ops.gated_delta_rule.chunk", VENDOR_ROOT)
    if module.CHUNK_SIZE != CORE_CONTRACT.chunk_size:
        raise RuntimeError(
            f"The core port requires chunk={CORE_CONTRACT.chunk_size}; "
            f"the detected architecture uses chunk={module.CHUNK_SIZE}."
        )
    return module


def load_test_support():
    """Load the unmodified upstream fixture/reference modules after the vendor API."""

    load_public_api()
    gdr_common = _import_owned("gdr_common", TEST_ROOT)
    if gdr_common.CHUNK_SIZE != CORE_CONTRACT.chunk_size:
        raise RuntimeError(
            "The upstream fixture did not detect the SM120 chunk=32 contract; "
            f"got chunk={gdr_common.CHUNK_SIZE}."
        )
    ref_gdr = _import_owned("ref_gdr", TEST_ROOT)
    return gdr_common, ref_gdr


@dataclass
class GDNBatch:
    q: Any
    k: Any
    v: Any
    g: Any
    beta: Any
    do: Any
    h0_ref: Any
    dht_ref: Any
    h0: Any
    dht: Any
    cu_seqlens: Any
    scale: float
    state_v_first: bool = False


@dataclass
class ForwardResult:
    o: Any
    final_state: Any
    g: Any
    A: Any
    h: Any
    cp_cache: Any


@dataclass
class BackwardResult:
    dq: Any
    dk: Any
    dv: Any
    db: Any
    dg: Any
    dh0: Any

    def as_tuple(self):
        return self.dq, self.dk, self.dv, self.db, self.dg, self.dh0


def make_fixed_inputs(
    *,
    batch_size: int = 1,
    num_tokens: int = 128,
    num_k_heads: int = 2,
    num_v_heads: int = 4,
    varlen: bool = False,
    cu_seqlens_list: list[int] | None = None,
    use_h0: bool = True,
    use_dht: bool | None = None,
    state_v_first: bool = False,
    seed: int = 42,
) -> GDNBatch:
    """Create deterministic bf16, K=DV=128 GQA inputs with the upstream fixture.

    The default Hg=2, H=4 layout activates backward group reduction. ``use_h0``
    and ``use_dht`` are independent to cover the upstream test gap where an
    initial state exists but no terminal-state gradient is supplied.
    """

    if use_dht is None:
        use_dht = use_h0
    gdr_common, _ = load_test_support()
    values = gdr_common.make_inputs(
        batch_size,
        num_tokens,
        num_k_heads,
        num_v_heads,
        varlen,
        cu_seqlens_list,
        use_h0 or use_dht,
        state_v_first,
        seed,
    )
    (
        q,
        k,
        v,
        g,
        beta,
        do,
        h0_ref,
        dht_ref,
        h0,
        dht,
        cu_seqlens,
        scale,
    ) = values
    if not use_h0:
        h0_ref = h0 = None
    if not use_dht:
        dht_ref = dht = None
    return GDNBatch(
        q,
        k,
        v,
        g,
        beta,
        do,
        h0_ref,
        dht_ref,
        h0,
        dht,
        cu_seqlens,
        scale,
        state_v_first,
    )


class GDNRunner:
    """Run complete public fwd/bwd pipelines for one explicit tuning config."""

    def __init__(self, config: GDNConfig | None = None):
        self.config = default_config(Direction.FWD) if config is None else config
        evaluation = evaluate_config(self.config)
        if evaluation.structural_status is Status.UNSAT:
            reasons = [
                check.reason
                for check in evaluation.structural_checks
                if check.status is Status.UNSAT
            ]
            raise ValueError("Structurally UNSAT GDN config: " + " ".join(reasons))
        self.api = load_public_api()

    def forward(self, batch: GDNBatch, *, output_h: bool = False) -> ForwardResult:
        g, A, o, h, final_state, cp_cache = self.api.chunk_gated_delta_rule_fwd(
            q=batch.q,
            k=batch.k,
            v=batch.v,
            g=batch.g,
            beta=batch.beta,
            scale=batch.scale,
            initial_state=batch.h0,
            cu_seqlens=batch.cu_seqlens,
            output_final_state=True,
            output_h=output_h,
            auto_cp=False,
            state_v_first=batch.state_v_first,
            enable_fwd_cp_cache=False,
            cp_context=None,
            block_DV=self.config.block_DV,
            prepare_h_num_stages=self.config.prepare_h_num_stages,
        )
        return ForwardResult(o, final_state, g, A, h, cp_cache)

    def backward(
        self,
        batch: GDNBatch,
        forward: ForwardResult | None = None,
        *,
        do=_UNSET,
        dht=_UNSET,
    ) -> BackwardResult:
        """Run public backward; compute its required public forward if omitted."""

        if forward is None:
            forward = self.forward(batch, output_h=False)
        do = batch.do if do is _UNSET else do
        dht = batch.dht if dht is _UNSET else dht
        gradients = self.api.chunk_gated_delta_rule_bwd(
            q=batch.q,
            k=batch.k,
            v=batch.v,
            g=forward.g,
            beta=batch.beta,
            A=forward.A,
            do=do,
            dht=dht,
            scale=batch.scale,
            initial_state=batch.h0,
            cu_seqlens=batch.cu_seqlens,
            state_v_first=batch.state_v_first,
            auto_cp=False,
            cp_cache=forward.cp_cache,
            cp_context=None,
            prepare_h_num_stages=self.config.prepare_h_num_stages,
            group_reduce_block_size=self.config.group_reduce_block_size,
        )
        return BackwardResult(*gradients)

    def run(self, batch: GDNBatch):
        if self.config.direction is Direction.FWD:
            return self.forward(batch)
        return self.backward(batch)


def make_runner(
    direction: Direction | str = Direction.FWD,
    *,
    block_DV: int = 128,
    prepare_h_num_stages: int = 2,
    group_reduce_block_size: int = 16,
) -> GDNRunner:
    direction = Direction(direction)
    if direction is Direction.FWD and (
        prepare_h_num_stages != 2 or group_reduce_block_size != 16
    ):
        raise ValueError(
            "fwd fixes prepare_h_num_stages=2 and group_reduce_block_size=16."
        )
    if direction is Direction.BWD and block_DV != 128:
        raise ValueError("bwd fixes block_DV=128; fused_bwd has no block_DV parameter.")
    return GDNRunner(
        GDNConfig(
            direction,
            block_DV,
            prepare_h_num_stages,
            group_reduce_block_size,
        )
    )
