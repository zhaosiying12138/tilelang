"""Real CUDA integration for opt-in Z3 pipeline planning on SM120."""

from __future__ import annotations

import json

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.cuda.transform import z3_scheduler
from tilelang.language.annotations import z3_pipeline_annotations


@tilelang.testing.requires_cuda
def test_default_off_and_opt_in_real_solver_and_numerics(tmp_path, monkeypatch):
    block = 128
    iterations = 4

    def make_func():
        @T.prim_func
        def main(
            A: T.Tensor((block * iterations,), "float32"),
            B: T.Tensor((block * iterations,), "float32"),
            C: T.Tensor((block * iterations,), "float32"),
        ):
            with T.Kernel(1, threads=128):
                a_shared = T.alloc_shared((block,), "float32")
                b_shared = T.alloc_shared((block,), "float32")
                c_fragment = T.alloc_fragment((block,), "float32")
                annotations = z3_pipeline_annotations(
                    latencies=[1, 1, 2, 2],
                    iis=[1, 1, 1, 1],
                    resource_flags=[1, 1, 16, 2],
                    data_deps=[(0, 2, 0, 1), (1, 2, 0, 1), (2, 3, 0, 2)],
                    resource_deps=[(0, 1)],
                    enable_offset=False,
                    verbose=True,
                    debug_dir=str(tmp_path),
                )
                for k in T.Pipelined(iterations, num_stages=2, annotations=annotations):
                    T.copy(A[k * block : (k + 1) * block], a_shared)
                    T.copy(B[k * block : (k + 1) * block], b_shared)
                    for i in T.Parallel(block):
                        c_fragment[i] = a_shared[i] + b_shared[i]
                    T.copy(c_fragment, C[k * block : (k + 1) * block])

        return main

    original = z3_scheduler.z3_schedule_loop_python
    calls = []

    def counted_solve(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(z3_scheduler, "z3_schedule_loop_python", counted_solve)
    config_key = tilelang.PassConfigKey.TL_ENABLE_Z3_PIPELINE_PLANNING

    def compile_and_run(enabled):
        func = make_func()
        kernel = tilelang.compile(
            func,
            target="cuda",
            out_idx=[2],
            execution_backend="tvm_ffi",
            pass_configs={config_key: enabled},
        )
        a = torch.arange(block * iterations, dtype=torch.float32, device="cuda") / 100
        b = torch.arange(block * iterations, dtype=torch.float32, device="cuda") / 50
        actual = kernel(a, b)
        torch.testing.assert_close(actual, a + b, rtol=0, atol=0)

    compile_and_run(False)
    assert calls == [], "Default-off compilation must not call the CUDA Z3 solver"

    compile_and_run(True)
    assert len(calls) == 1, "Opt-in compilation must perform exactly one real solve"
    starts, versions, minimal_ii = calls[0]
    assert minimal_ii == 2
    assert versions == []
    assert max(start // minimal_ii for start in starts) + 1 == 3

    schedule_files = list(tmp_path.glob("schedule_*/schedule_info.json"))
    assert len(schedule_files) == 1
    schedule = json.loads(schedule_files[0].read_text())
    assert schedule["fallback"] is False
    assert schedule["minimal_II"] == 2
    assert schedule["start_times"] == starts
