"""Host-only tests; runnable with unittest and without Torch or TileLang."""

from __future__ import annotations

import ast
import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

GDN_ROOT = Path(__file__).resolve().parents[1]
if str(GDN_ROOT) not in sys.path:
    sys.path.insert(0, str(GDN_ROOT))

from cli import main as cli_main  # noqa: E402
from config import (  # noqa: E402
    BLOCK_DV_DOMAIN,
    CORE_CONTRACT,
    GROUP_REDUCE_BLOCK_SIZE_DOMAIN,
    PREPARE_H_NUM_STAGES_DOMAIN,
    RAW_CONFIGS_PER_DIRECTION,
    CheckResult,
    Direction,
    Status,
    combine_status,
    default_config,
    evaluate_config,
    evaluate_configs,
    parameter_consumers,
    raw_config_count,
    raw_configs,
    structural_checks,
)

CHUNK_PATH = (
    GDN_ROOT
    / "vendor/flash_qla/ops/gated_delta_rule/chunk/__init__.py"
)
GROUP_REDUCE_PATH = (
    GDN_ROOT / "vendor/flash_qla/ops/utils/group_reduce.py"
)


def _function(tree, name):
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _keywords(call):
    return {keyword.arg: keyword.value for keyword in call.keywords}


def _load_group_reduce_wrapper(num_tokens):
    """Execute the real wrapper with host doubles, without importing accelerators."""

    tree = ast.parse(GROUP_REDUCE_PATH.read_text())
    nodes = [
        _function(tree, "_validate_group_reduce_block_size"),
        _function(tree, "group_reduce_vector"),
    ]
    calls = {}
    result = object()
    buffer = SimpleNamespace(
        shape=(1, num_tokens, 4, 128), dtype="bf16", device="cpu"
    )

    def empty(shape, **kwargs):
        calls["empty"] = (shape, kwargs)
        return result

    def factory(*args, **kwargs):
        calls["factory"] = (args, kwargs)

        def kernel(actual_buffer, actual_result):
            calls["kernel"] = (actual_buffer, actual_result)

        return kernel

    namespace = {
        "torch": SimpleNamespace(Tensor=object, empty=empty),
        "tilelang_group_reduce_vector": factory,
    }
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(GROUP_REDUCE_PATH), "exec"), namespace)
    return namespace["group_reduce_vector"], buffer, result, calls


def _assert_public_nondivisible_rejection(test, block_size):
    """Call public bwd up to its early GQA validation with shape-only doubles."""

    chunk_tree = ast.parse(CHUNK_PATH.read_text())
    group_tree = ast.parse(GROUP_REDUCE_PATH.read_text())
    nodes = [
        _function(group_tree, "_validate_group_reduce_block_size"),
        _function(chunk_tree, "chunk_gated_delta_rule_bwd"),
    ]
    namespace = {
        "torch": SimpleNamespace(Tensor=object, LongTensor=object),
        "fused_gdr_bwd": object(),
    }
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(CHUNK_PATH), "exec"), namespace)
    public_bwd = namespace["chunk_gated_delta_rule_bwd"]
    k = SimpleNamespace(shape=(1, 127, 2, 128))
    v = SimpleNamespace(shape=(1, 127, 4, 128))
    A = SimpleNamespace(shape=(1, 127, 4, 32))
    with test.assertRaisesRegex(
        ValueError,
        rf"num_tokens=127.*block_size={block_size}",
    ):
        public_bwd(
            q=object(),
            k=k,
            v=v,
            g=object(),
            beta=object(),
            A=A,
            do=object(),
            group_reduce_block_size=block_size,
        )


class ConfigSpaceTest(unittest.TestCase):
    def test_direction_specific_domains_and_counts(self):
        self.assertEqual(BLOCK_DV_DOMAIN, (16, 32, 64, 128))
        self.assertEqual(PREPARE_H_NUM_STAGES_DOMAIN, (1, 2, 3))
        self.assertEqual(GROUP_REDUCE_BLOCK_SIZE_DOMAIN, (8, 16, 32))
        self.assertEqual(RAW_CONFIGS_PER_DIRECTION[Direction.FWD], 4)
        self.assertEqual(RAW_CONFIGS_PER_DIRECTION[Direction.BWD], 9)
        self.assertEqual(raw_config_count(Direction.FWD), 4)
        self.assertEqual(raw_config_count(Direction.BWD), 9)
        self.assertEqual(len(raw_configs(Direction.FWD)), 4)
        self.assertEqual(len(raw_configs(Direction.BWD)), 9)

    def test_forward_space_has_only_block_dimension(self):
        configs = raw_configs(Direction.FWD)
        self.assertEqual(
            [item.block_DV for item in configs], [16, 32, 64, 128]
        )
        self.assertEqual(len(set(configs)), 4)
        for config in configs:
            self.assertIs(config.direction, Direction.FWD)
            self.assertEqual(config.prepare_h_num_stages, 2)
            self.assertEqual(config.group_reduce_block_size, 16)

    def test_backward_space_is_stage_by_group_without_block_dimension(self):
        configs = raw_configs(Direction.BWD)
        expected = [
            (stages, group)
            for stages in (1, 2, 3)
            for group in (8, 16, 32)
        ]
        self.assertEqual(
            [
                (item.prepare_h_num_stages, item.group_reduce_block_size)
                for item in configs
            ],
            expected,
        )
        self.assertEqual(len(set(configs)), 9)
        for config in configs:
            self.assertIs(config.direction, Direction.BWD)
            self.assertEqual(config.block_DV, 128)

    def test_defaults_match_upstream(self):
        for direction in Direction:
            config = default_config(direction)
            self.assertEqual(config.block_DV, 128)
            self.assertEqual(config.prepare_h_num_stages, 2)
            self.assertEqual(config.group_reduce_block_size, 16)
        self.assertFalse(CORE_CONTRACT.auto_cp)
        self.assertEqual(CORE_CONTRACT.dtype, "bf16")
        self.assertEqual(CORE_CONTRACT.head_dim_k, 128)
        self.assertEqual(CORE_CONTRACT.head_dim_v, 128)
        self.assertEqual(CORE_CONTRACT.chunk_size, 32)
        self.assertEqual(CORE_CONTRACT.num_tokens, 128)
        self.assertEqual(CORE_CONTRACT.threads, 512)

    def test_all_raw_configs_are_structurally_sat_and_resource_unknown(self):
        for direction, count in ((Direction.FWD, 4), (Direction.BWD, 9)):
            evaluations = evaluate_configs(direction)
            self.assertEqual(len(evaluations), count)
            for evaluation in evaluations:
                with self.subTest(config=evaluation.config):
                    self.assertIs(evaluation.structural_status, Status.SAT)
                    self.assertIs(evaluation.resource_check.status, Status.UNKNOWN)
                    self.assertIs(evaluation.status, Status.UNKNOWN)
                    self.assertIsNot(evaluation.status, Status.UNSAT)

    def test_block_128_is_not_unsat(self):
        for direction in Direction:
            evaluation = evaluate_config(default_config(direction))
            self.assertEqual(evaluation.config.block_DV, 128)
            self.assertIs(evaluation.structural_status, Status.SAT)
            self.assertIs(evaluation.status, Status.UNKNOWN)

    def test_block_divisibility(self):
        for value in (0, -16, 24, 48, 256, 16.0, True):
            with self.subTest(block_DV=value):
                config = replace(default_config(Direction.FWD), block_DV=value)
                evaluation = evaluate_config(config)
                self.assertIs(evaluation.structural_status, Status.UNSAT)
                self.assertIs(evaluation.status, Status.UNSAT)

    def test_stage_count_must_be_positive_integer(self):
        for value in (0, -1, 1.5, True, None):
            with self.subTest(prepare_h_num_stages=value):
                config = replace(
                    default_config(Direction.BWD), prepare_h_num_stages=value
                )
                evaluation = evaluate_config(config)
                self.assertIs(evaluation.structural_status, Status.UNSAT)
                self.assertIs(evaluation.status, Status.UNSAT)

    def test_group_token_divisibility_only_constrains_nondefault(self):
        core = replace(CORE_CONTRACT, num_tokens=127)
        for block_size in (8, 16, 32):
            config = replace(
                default_config(Direction.BWD),
                group_reduce_block_size=block_size,
            )
            results = {
                check.name: check for check in structural_checks(config, core)
            }
            with self.subTest(group_reduce_block_size=block_size):
                if block_size == 16:
                    self.assertIs(
                        results["group_reduce_token_tiling"].status, Status.SAT
                    )
                else:
                    self.assertIs(
                        results["group_reduce_token_tiling"].status, Status.UNSAT
                    )
        for block_size in (8, 16, 32):
            config = replace(
                default_config(Direction.BWD),
                group_reduce_block_size=block_size,
            )
            self.assertIs(evaluate_config(config).structural_status, Status.SAT)

    def test_group_block_size_must_be_positive_integer(self):
        for value in (0, -8, 8.0, True, None):
            with self.subTest(group_reduce_block_size=value):
                config = replace(
                    default_config(Direction.BWD),
                    group_reduce_block_size=value,
                )
                evaluation = evaluate_config(config)
                self.assertIs(evaluation.structural_status, Status.UNSAT)

    def test_fixed_thread_protocol(self):
        for core in (
            replace(CORE_CONTRACT, threads=384),
            replace(CORE_CONTRACT, warp_size=16),
        ):
            with self.subTest(core=core):
                results = structural_checks(default_config(Direction.FWD), core)
                self.assertIs(combine_status(item.status for item in results), Status.UNSAT)

    def test_fixed_core_contract(self):
        for core in (
            replace(CORE_CONTRACT, auto_cp=True),
            replace(CORE_CONTRACT, dtype="fp16"),
            replace(CORE_CONTRACT, head_dim_k=64),
            replace(CORE_CONTRACT, head_dim_v=64),
            replace(CORE_CONTRACT, chunk_size=64),
            replace(CORE_CONTRACT, num_tokens=127),
        ):
            with self.subTest(core=core):
                results = structural_checks(default_config(Direction.FWD), core)
                self.assertIs(combine_status(item.status for item in results), Status.UNSAT)

    def test_resource_checker_extension_point(self):
        class SatResources:
            def check(self, config, core):
                return CheckResult.sat("test_resource", "test double only")

        evaluation = evaluate_config(
            default_config(Direction.FWD), resource_checker=SatResources()
        )
        self.assertIs(evaluation.status, Status.SAT)

        class InvalidResources:
            def check(self, config, core):
                return None

        with self.assertRaises(TypeError):
            evaluate_config(
                default_config(Direction.FWD), resource_checker=InvalidResources()
            )

    def test_unsat_dominates_unknown(self):
        self.assertIs(combine_status((Status.UNKNOWN, Status.UNSAT)), Status.UNSAT)
        self.assertIs(combine_status((Status.SAT, Status.UNKNOWN)), Status.UNKNOWN)
        self.assertIs(combine_status((Status.SAT, Status.SAT)), Status.SAT)

    def test_direction_applicability_is_explicit(self):
        fwd = parameter_consumers(Direction.FWD)
        bwd = parameter_consumers(Direction.BWD)
        self.assertEqual(fwd["block_DV"], "fused_gdr_fwd")
        self.assertIn("fixed at 2", fwd["prepare_h_num_stages"])
        self.assertIn("no block_DV parameter", bwd["block_DV"])
        self.assertIn("fused_gdr_h", bwd["prepare_h_num_stages"])
        self.assertIn("group_reduce_vector", bwd["group_reduce_block_size"])

    def test_cpu_cli_json_and_direction_counts(self):
        output = io.StringIO()
        with redirect_stdout(output):
            result = cli_main(["--check-configs"])
        self.assertEqual(result, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["raw_configs_per_direction"], {"fwd": 4, "bwd": 9})
        self.assertEqual(payload["core"]["num_tokens"], 128)
        for direction, count in (("fwd", 4), ("bwd", 9)):
            report = payload["directions"][direction]
            self.assertEqual(report["count"], count)
            self.assertEqual(report["structural_status_counts"]["SAT"], count)
            self.assertEqual(report["status_counts"]["UNKNOWN"], count)
            self.assertEqual(report["status_counts"]["UNSAT"], 0)
        fwd = payload["directions"]["fwd"]
        bwd = payload["directions"]["bwd"]
        self.assertEqual(set(fwd["raw_domains"]), {"block_DV"})
        self.assertEqual(
            set(bwd["raw_domains"]),
            {"prepare_h_num_stages", "group_reduce_block_size"},
        )
        self.assertTrue(
            all(
                item["prepare_h_num_stages"] == 2
                and item["group_reduce_block_size"] == 16
                for item in fwd["configs"]
            )
        )
        self.assertTrue(all(item["block_DV"] == 128 for item in bwd["configs"]))

    def test_runner_factory_rejects_wrong_direction_knobs(self):
        from runner import make_runner

        with self.assertRaisesRegex(ValueError, "fwd fixes"):
            make_runner(Direction.FWD, prepare_h_num_stages=3)
        with self.assertRaisesRegex(ValueError, "fwd fixes"):
            make_runner(Direction.FWD, group_reduce_block_size=8)
        with self.assertRaisesRegex(ValueError, "bwd fixes"):
            make_runner(Direction.BWD, block_DV=64)

    def test_fused_bwd_has_no_block_DV_parameter(self):
        for arch in ("hopper", "blackwell", "blackwell_sm120"):
            path = (
                GDN_ROOT
                / f"vendor/flash_qla/ops/gated_delta_rule/chunk/{arch}/fused_bwd.py"
            )
            function = _function(ast.parse(path.read_text()), "fused_gdr_bwd")
            self.assertNotIn("block_DV", [arg.arg for arg in function.args.args])

    def test_every_backward_config_has_two_real_consumer_wirings(self):
        runner_tree = ast.parse((GDN_ROOT / "runner.py").read_text())
        runner_backward = _function(runner_tree, "backward")
        public_call = next(
            call
            for call in ast.walk(runner_backward)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "chunk_gated_delta_rule_bwd"
        )
        runner_keywords = _keywords(public_call)
        self.assertEqual(
            ast.unparse(runner_keywords["prepare_h_num_stages"]),
            "self.config.prepare_h_num_stages",
        )
        self.assertEqual(
            ast.unparse(runner_keywords["group_reduce_block_size"]),
            "self.config.group_reduce_block_size",
        )

        chunk_tree = ast.parse(CHUNK_PATH.read_text())
        public_bwd = _function(chunk_tree, "chunk_gated_delta_rule_bwd")
        calls = [
            call for call in ast.walk(public_bwd) if isinstance(call, ast.Call)
        ]
        prepare = next(
            call for call in calls
            if isinstance(call.func, ast.Name) and call.func.id == "fused_gdr_h"
        )
        self.assertEqual(
            ast.unparse(_keywords(prepare)["num_stages"]),
            "prepare_h_num_stages",
        )
        reductions = [
            call for call in calls
            if isinstance(call.func, ast.Name)
            and call.func.id == "group_reduce_vector"
        ]
        self.assertEqual(len(reductions), 2)
        for call in reductions:
            self.assertEqual(
                ast.unparse(_keywords(call)["block_size"]),
                "group_reduce_block_size",
            )
        for config in raw_configs(Direction.BWD):
            with self.subTest(config=config):
                self.assertIn(config.prepare_h_num_stages, (1, 2, 3))
                self.assertIn(config.group_reduce_block_size, (8, 16, 32))
                self.assertIs(evaluate_config(config).structural_status, Status.SAT)

    def test_group_reduce_default_and_valid_values_reach_factory(self):
        wrapper, buffer, result, calls = _load_group_reduce_wrapper(127)
        self.assertIs(wrapper(buffer, 2), result)
        self.assertEqual(calls["factory"][1]["block_size"], 16)
        self.assertEqual(calls["empty"][0], (1, 127, 2, 128))
        self.assertIn("kernel", calls)

        for block_size in (8, 16, 32):
            with self.subTest(group_reduce_block_size=block_size):
                wrapper, buffer, _, calls = _load_group_reduce_wrapper(128)
                wrapper(buffer, 2, block_size=block_size)
                self.assertEqual(calls["factory"][1]["block_size"], block_size)
                self.assertIn("kernel", calls)

    def test_nondefault_group_reduce_rejects_nondivisible_tokens(self):
        for block_size in (8, 32):
            with self.subTest(group_reduce_block_size=block_size):
                wrapper, buffer, _, calls = _load_group_reduce_wrapper(127)
                with self.assertRaisesRegex(
                    ValueError,
                    rf"num_tokens=127.*block_size={block_size}",
                ):
                    wrapper(buffer, 2, block_size=block_size)
                self.assertNotIn("factory", calls)
                _assert_public_nondivisible_rejection(self, block_size)

    def test_autograd_group_default_and_arity(self):
        tree = ast.parse(CHUNK_PATH.read_text())
        for name in ("chunk_gated_delta_rule_bwd", "chunk_gated_delta_rule"):
            function = _function(tree, name)
            args = [arg.arg for arg in function.args.args]
            defaults = dict(zip(args[-len(function.args.defaults):], function.args.defaults))
            self.assertEqual(ast.unparse(defaults["group_reduce_block_size"]), "16")

        cls = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "ChunkGatedDeltaRuleFunction"
        )
        forward = _function(cls, "forward")
        args = [arg.arg for arg in forward.args.args]
        defaults = dict(zip(args[-len(forward.args.defaults):], forward.args.defaults))
        self.assertEqual(ast.unparse(defaults["group_reduce_block_size"]), "16")
        backward = _function(cls, "backward")
        return_node = next(
            node for node in ast.walk(backward)
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple)
        )
        self.assertEqual(len(return_node.value.elts), 19)
        apply_call = next(
            call for call in ast.walk(tree)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "apply"
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "ChunkGatedDeltaRuleFunction"
        )
        self.assertEqual(len(apply_call.args), 19)


if __name__ == "__main__":
    unittest.main()
