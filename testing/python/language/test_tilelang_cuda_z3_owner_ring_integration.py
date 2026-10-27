"""Function-level Z3 owner-ring integration and SM120A numerics."""

from __future__ import annotations

import json

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.language.annotations import z3_pipeline_annotations
from tilelang.testing.ir import collect_nodes


def _owner_kernel(depth: int, debug_dir: str):
    extents = (3, 5)

    def owner_bodies(owner: int):
        return {
            "Producer": [
                T.WSSync.producer_acquire("ring"),
                f"produce{owner}",
                T.WSSync.producer_commit("ring"),
            ],
            "Consumer": [
                T.WSSync.consumer_wait("ring"),
                f"consume{owner}",
                T.WSSync.consumer_release("ring"),
                f"output{owner}",
            ],
        }

    @T.prim_func
    def kernel(A: T.Tensor((64,), T.float32), B: T.Tensor((64,), T.float32)):
        with T.Kernel(1, threads=128):
            ring = T.alloc_shared((4,), T.float32, scope="shared")
            frag = T.alloc_fragment((4,), T.float32)
            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[
                        T.WSRole("Producer", warps_lo=0, warps_hi=1),
                        T.WSRole("Consumer", warps_lo=4, warps_hi=8),
                    ],
                    pipelines=[
                        T.WSPipeline(
                            "ring",
                            [ring],
                            depth=depth,
                            owner_scopes=("owner0", "owner1"),
                        )
                    ],
                    scopes=[
                        T.WSScope("owner0", owner_bodies(0)),
                        T.WSScope("owner1", owner_bodies(1)),
                        T.WSScope(
                            T.WSScope.ROOT,
                            {
                                "Producer": ["owner0", "owner1"],
                                "Consumer": ["owner0", "owner1"],
                            },
                        ),
                    ],
                )
            )
            for owner, extent in enumerate(extents):
                annotations = z3_pipeline_annotations(
                    latencies=[4],
                    iis=[1],
                    resource_flags=[1],
                    owner_exclusion_deps=[(0, 1, 1, 2)],
                    enable_offset=False,
                    verbose=True,
                    debug_dir=debug_dir,
                    owner_ring=owner,
                )
                for i in T.Pipelined(
                    extent,
                    num_stages=depth,
                    annotations={**annotations, "tl.ws_op_id": f"owner{owner}"},
                ):
                    T.fill(ring, i + 16 * owner, annotations={T.WSID: f"produce{owner}"})
                    T.copy(ring, frag, annotations={T.WSID: f"consume{owner}"})
                    T.copy(frag, B[(owner * 5 + i) * 4], annotations={T.WSID: f"output{owner}"})

    return kernel


def _function_pass(mod, config_key):
    with tvm.transform.PassContext(config={config_key: True}):
        return tilelang.transform.Z3OwnerRingPlanning()(mod)


def test_function_level_owner_validation_and_materialization(tmp_path):
    config_key = tilelang.PassConfigKey.TL_ENABLE_Z3_PIPELINE_PLANNING
    target = tvm.target.Target(
        {
            "kind": "cuda",
            "arch": "sm_100a",
            "max_shared_memory_per_block": 1 << 20,
        }
    )
    func = _owner_kernel(2, str(tmp_path)).with_attr("global_symbol", "main")
    mod = tvm.IRModule.from_expr(func)
    mod = tvm.tirx.transform.BindTarget(target)(mod)
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    mod = _function_pass(mod, config_key)
    validated = [
        block
        for block in collect_nodes(mod["main"], tvm.tirx.SBlock)
        if "tl_z3_owner_ring_validated" in block.annotations
    ]
    assert len(validated) == 1
    for loop in collect_nodes(mod["main"], tvm.tirx.For):
        # Metadata is retained for audit; the owner-specific pass skips generic planning.
        assert "tl_z3_owner_ring" in loop.annotations
        assert "tl_z3_pipeline" in loop.annotations
        assert "num_stages" in loop.annotations
    mod = tilelang.cuda.transform.MaterializeWSSchedule()(mod)
    schedule_files = list(tmp_path.glob("schedule_*/schedule_info.json"))
    assert len(schedule_files) == 1
    schedule = json.loads(schedule_files[0].read_text())
    assert schedule["fallback"] is False
    assert schedule["minimal_II"] == 3
    left, right = schedule["start_times"]
    assert left % 3 == 0
    assert right % 3 == 1


@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_sm120_owner_ring_forward_wraparound_and_reuse(depth, tmp_path):
    config_key = tilelang.PassConfigKey.TL_ENABLE_Z3_PIPELINE_PLANNING
    target = tvm.target.Target(
        {
            "kind": "cuda",
            "arch": "sm_120a",
            "max_shared_memory_per_block": int(
                torch.cuda.get_device_properties(0).shared_memory_per_block_optin
            ),
        }
    )
    kernel = tilelang.compile(
        _owner_kernel(depth, str(tmp_path)),
        target=target,
        out_idx=None,
        execution_backend="tvm_ffi",
        pass_configs={config_key: True},
    )
    a = torch.arange(64, dtype=torch.float32, device="cuda")
    b = torch.full((64,), -7.0, dtype=torch.float32, device="cuda")
    expected = b.clone()
    for owner, extent in enumerate((3, 5)):
        for index in range(extent):
            output = (owner * 5 + index) * 4
            expected[output : output + 4] = index + 16 * owner
    for _ in range(4):
        kernel(a, b)
        torch.testing.assert_close(b, expected, rtol=0, atol=0)
    schedule_files = list(tmp_path.glob("schedule_*/schedule_info.json"))
    assert len(schedule_files) == 1
    schedule = json.loads(schedule_files[0].read_text())
    assert schedule["fallback"] is False
    assert schedule["minimal_II"] == 3
