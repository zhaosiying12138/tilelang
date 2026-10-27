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


@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
@pytest.mark.parametrize("iterations", [1, 4, 5, 12])
def test_sm120_multi_async_stage_real_solver_and_numerics(tmp_path, monkeypatch, iterations):
    block = 128

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
                latencies=[4, 4, 2, 2],
                iis=[1, 1, 1, 1],
                resource_flags=[1, 1, 16, 2],
                data_deps=[(0, 2, 0, 4), (1, 2, 0, 4), (2, 3, 0, 2)],
                resource_deps=[(0, 1)],
                enable_offset=False,
                verbose=True,
                debug_dir=str(tmp_path),
            )
            for k in T.Pipelined(iterations, num_stages=2, annotations=annotations):
                T.copy(A[k * block : (k + 1) * block], a_shared)
                T.copy(B[k * block : (k + 1) * block], b_shared)
                for i in T.Parallel(block):
                    c_fragment[i] = a_shared[i] + 2 * b_shared[i]
                T.copy(c_fragment, C[k * block : (k + 1) * block])

    original = z3_scheduler.z3_schedule_loop_python
    calls = []

    def counted_solve(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(z3_scheduler, "z3_schedule_loop_python", counted_solve)
    config_key = tilelang.PassConfigKey.TL_ENABLE_Z3_PIPELINE_PLANNING
    kernel = tilelang.compile(
        main,
        target={"kind": "cuda", "arch": "sm_120a"},
        out_idx=[2],
        execution_backend="tvm_ffi",
        pass_configs={config_key: True},
    )

    assert len(calls) == 1, "The multi-stage compilation must perform exactly one real solve"
    starts, versions, minimal_ii = calls[0]
    assert starts == [1, 2, 6, 8]
    assert minimal_ii == 2
    assert versions == []
    stages = [start // minimal_ii for start in starts]
    assert stages == [0, 1, 3, 4], "Expected producers at 0/1, consumer at 3, and output at 4"
    assert max(stages) + 1 == 5

    schedule_files = list(tmp_path.glob("schedule_*/schedule_info.json"))
    assert len(schedule_files) == 1
    schedule = json.loads(schedule_files[0].read_text())
    assert schedule["fallback"] is False
    assert schedule["minimal_II"] == 2
    assert schedule["start_times"] == starts

    cuda_source = kernel.get_kernel_source()
    assert "tl::cp_async_gs<4>" in cuda_source
    assert "tl::cp_async_commit();" in cuda_source
    assert "tl::cp_async_wait<0>();" in cuda_source, "Every async pipeline must drain its tail"
    if iterations == 12:
        assert cuda_source.count("tl::cp_async_gs<4>") == 10
        assert cuda_source.count("tl::cp_async_commit();") == 10
        assert cuda_source.count("tl::cp_async_wait<1>();") == 2
        assert cuda_source.count("tl::cp_async_wait<0>();") == 3
        assert "buf_dyn_shmem + 1536" in cuda_source, "B starts after three 512-byte A versions"
        assert "((k + 1) % 3)" in cuda_source, "A must use a three-version ring"
        assert "((k + 1) & 1)" in cuda_source, "B must use a two-version ring"

    indices = torch.arange(block * iterations, dtype=torch.float32, device="cuda")
    for repeat in range(4):
        a = indices + 7 * repeat
        b = 2 * indices + 1 - 3 * repeat
        expected = a + 2 * b
        actual = kernel(a, b)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
def test_sm120_bind_clone_next_iteration_overwrite_real_solver_and_numerics(tmp_path, monkeypatch):
    """Later producer iterations must not overwrite a cloned Bind's source snapshot."""

    block = 128
    iterations = 12
    source_nbytes = block * 4

    @T.prim_func
    def main(
        A: T.Tensor((block * iterations,), "float32"),
        B: T.Tensor((block * iterations,), "float32"),
    ):
        # The producer's next-iteration stores provide the later overwrite.
        # One thread keeps the numerical check focused on buffer versioning.
        with T.Kernel(1, threads=1):
            source = T.alloc_shared((block,), "float32")
            annotations = z3_pipeline_annotations(
                latencies=[30, 2, 1],
                iis=[1, 1, 1],
                resource_flags=[0, 0, 0],
                data_deps=[(0, 1, 0, 30), (1, 0, -1, 2)],
                buffer_sizes=[source_nbytes],
                buffer_vars=[source],
                memory_groups=[[2 * source_nbytes, 0]],
                enable_offset=True,
                manual_stages=[0, 0, 1],
                manual_schedule=True,
                recalculate_buffer_versions=True,
                verbose=True,
                debug_dir=str(tmp_path),
            )
            for k in T.Pipelined(iterations, num_stages=3, annotations=annotations):
                T.copy(A[k * block : (k + 1) * block], source)
                value: T.float32 = source[0] + 1.0
                for i in T.Parallel(block):
                    B[k * block + i] = value + i

    original = z3_scheduler.z3_schedule_loop_python
    calls = []

    def counted_solve(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(z3_scheduler, "z3_schedule_loop_python", counted_solve)
    config_key = tilelang.PassConfigKey.TL_ENABLE_Z3_PIPELINE_PLANNING
    kernel = tilelang.compile(
        main,
        target={"kind": "cuda", "arch": "sm_120a"},
        out_idx=[1],
        execution_backend="tvm_ffi",
        pass_configs={config_key: True},
    )

    assert len(calls) == 2, "Cross-stage Bind planning must solve the original and clone-augmented problems"
    original_starts, _, original_ii = calls[0]
    augmented_starts, augmented_versions, augmented_ii = calls[1]
    assert (len(original_starts), len(augmented_starts)) == (3, 4)
    original_stages = [start // original_ii for start in original_starts]
    augmented_stages = [start // augmented_ii for start in augmented_starts]
    assert original_stages == [0, 0, 1]
    assert augmented_stages == [0, 0, 1, 1], "The appended Bind clone must share its consumer's stage"
    assert augmented_versions == [2], "Next-iteration producer overwrites require two source versions"

    schedule_files = sorted(tmp_path.glob("schedule_*/schedule_info.json"))
    assert len(schedule_files) == 2
    original_schedule, augmented_schedule = [json.loads(path.read_text()) for path in schedule_files]
    for schedule, (starts, versions, minimal_ii), task_count, stages in (
        (original_schedule, calls[0], 3, [0, 0, 1]),
        (augmented_schedule, calls[1], 4, [0, 0, 1, 1]),
    ):
        assert schedule["fallback"] is False
        assert schedule["num_tasks"] == task_count
        assert schedule["manual_stages"] == stages
        assert schedule["start_times"] == starts
        assert [start // minimal_ii for start in starts] == stages
        assert schedule["buffer_versions"] == versions
        assert schedule["minimal_II"] == minimal_ii
        assert schedule["buffer_sizes"] == [source_nbytes]
        assert schedule["memory_groups"] == [[2 * source_nbytes, 0]]
    assert original_schedule["latencies"] == [30, 2, 1]
    assert augmented_schedule["latencies"] == [30, 2, 1, 2]
    augmented_deps = {tuple(dep) for dep in augmented_schedule["data_dependencies"]}
    assert (0, 3, 0, 30) in augmented_deps, "The clone must read the producer's current snapshot"
    assert (3, 0, -1, 2) in augmented_deps, "The clone must protect its snapshot from the next producer overwrite"
    assert (
        augmented_starts[3] + augmented_schedule["latencies"][3] <= augmented_starts[2]
    ), "The cloned Bind must finish before its consumer"

    # Every iteration has a distinct source[0], so reading a version overwritten
    # by a later producer iteration changes the exact output.
    a = torch.arange(block * iterations, dtype=torch.float32, device="cuda") - 512
    lanes = torch.arange(block, dtype=torch.float32, device="cuda")
    expected = (a.reshape(iterations, block)[:, :1] + 1 + lanes).reshape(-1)
    for _ in range(4):
        actual = kernel(a)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
