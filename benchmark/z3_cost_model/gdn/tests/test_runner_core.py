"""SM120 GPU entry points for the ported public-pipeline runner.

These tests exercise the real vendored TileLang implementation when explicitly
run on a supported GPU; no replacement kernel is used.
"""

from __future__ import annotations

import pytest

from config import Direction, GDNConfig, default_config, raw_configs
from correctness import (
    DETERMINISM_ITERS,
    check_backward,
    check_determinism,
    check_forward,
)
from runner import GDNRunner, make_fixed_inputs

pytestmark = [pytest.mark.gpu, pytest.mark.sm120]


def _config_id(config: GDNConfig) -> str:
    if config.direction is Direction.FWD:
        return f"block_DV-{config.block_DV}"
    return (
        f"stages-{config.prepare_h_num_stages}_"
        f"group-{config.group_reduce_block_size}"
    )


def _gqa_batch(**kwargs):
    batch = make_fixed_inputs(
        num_k_heads=2,
        num_v_heads=4,
        **kwargs,
    )
    assert batch.k.shape[2] == 2
    assert batch.v.shape[2] == 4
    return batch


def test_core_final_state_without_h0():
    runner = GDNRunner(default_config(Direction.FWD))
    batch = _gqa_batch(num_tokens=65, use_h0=False, use_dht=False)
    check_forward(runner, batch)


def test_core_h0_without_dht():
    runner = GDNRunner(default_config(Direction.BWD))
    batch = _gqa_batch(num_tokens=65, use_h0=True, use_dht=False)
    check_backward(runner, batch)


def test_core_padded_outputs_and_gradients_are_zero():
    runner = GDNRunner(default_config(Direction.BWD))
    batch = _gqa_batch(
        num_tokens=96,
        varlen=True,
        cu_seqlens_list=[0, 37, 71],
        use_h0=True,
        use_dht=True,
    )
    check_forward(runner, batch)
    check_backward(runner, batch)


@pytest.mark.slow
@pytest.mark.parametrize(
    "config", raw_configs(Direction.FWD), ids=_config_id
)
def test_core_raw_forward_config(config):
    assert config.prepare_h_num_stages == 2
    assert config.group_reduce_block_size == 16
    runner = GDNRunner(config)
    batch = _gqa_batch(num_tokens=128, use_h0=True, use_dht=False)
    check_forward(runner, batch, output_h=False)


@pytest.mark.slow
@pytest.mark.parametrize(
    "config", raw_configs(Direction.BWD), ids=_config_id
)
def test_core_raw_backward_config(config):
    assert config.block_DV == 128
    runner = GDNRunner(config)
    # Hg=2 < H=4 makes every group_reduce_block_size value reach dq/dk reduction.
    batch = _gqa_batch(num_tokens=128, use_h0=True, use_dht=False)
    check_backward(runner, batch)


@pytest.mark.slow
@pytest.mark.parametrize("direction", ("fwd", "bwd"))
def test_core_determinism(direction):
    runner = GDNRunner(default_config(direction))
    batch = _gqa_batch(num_tokens=128, use_h0=True, use_dht=False)
    check_determinism(
        runner,
        batch,
        direction,
        iterations=DETERMINISM_ITERS,
    )
