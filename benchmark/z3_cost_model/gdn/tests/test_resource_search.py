"""Pure-CPU tests for GDN resource checks and direction-specific pruning."""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
GDN_ROOT = REPOSITORY_ROOT / "benchmark/z3_cost_model/gdn"
HARDWARE_PATH = REPOSITORY_ROOT / "manifest/hardware-sm120.json"
FWD_EVIDENCE_PATH = REPOSITORY_ROOT / "evidence/G5/gdn-fwd-shared-memory.json"
PREPARE_H_EVIDENCE_PATH = REPOSITORY_ROOT / "evidence/G5/gdn-prepare-h-shared-memory.json"
_TEST_PACKAGE = "_tilelang_gdn_resource_test"


def _install_synthetic_package(name, path):
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    package.__package__ = name
    sys.modules[name] = package


# Load space.py without executing tilelang/__init__.py or any native/GPU path.
_install_synthetic_package(_TEST_PACKAGE, REPOSITORY_ROOT / "tilelang")
_install_synthetic_package(f"{_TEST_PACKAGE}.autotuner", REPOSITORY_ROOT / "tilelang/autotuner")
_space = importlib.import_module(f"{_TEST_PACKAGE}.autotuner.space")

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmark.z3_cost_model.gdn import config as gdn_config  # noqa: E402
from benchmark.z3_cost_model.gdn import resource, search  # noqa: E402


class _RecordingSolver:
    def __init__(self):
        self.settings = {}
        self.clauses = []
        self.check_calls = 0

    def set(self, name, value):
        self.settings[name] = value

    def add(self, *clauses):
        self.clauses.extend(clauses)

    def check(self):
        self.check_calls += 1
        return "unsat" if any(clause is False for clause in self.clauses) else "sat"

    def reason_unknown(self):
        return ""

    def statistics(self):
        return {"checks": self.check_calls}


class GDNResourceSearchTest(unittest.TestCase):
    def setUp(self):
        space_patch = patch.object(search, "_space_module", return_value=_space)
        space_patch.start()
        self.addCleanup(space_patch.stop)

    def test_raw_domains_and_all_json_ids_are_unique(self):
        fwd = search.make_config_space("fwd", HARDWARE_PATH).raw_configs()
        bwd = search.make_config_space("bwd", HARDWARE_PATH).raw_configs()
        expected_keys = {
            "direction",
            "block_DV",
            "prepare_h_num_stages",
            "group_reduce_block_size",
        }

        self.assertEqual(len(fwd), 4)
        self.assertEqual(len(bwd), 9)
        self.assertEqual([config["block_DV"] for config in fwd], [16, 32, 64, 128])
        self.assertTrue(all(config["direction"] == "fwd" for config in fwd))
        self.assertTrue(all(config["direction"] == "bwd" for config in bwd))
        self.assertTrue(all(set(config) == expected_keys for config in fwd + bwd))
        self.assertTrue(all(config["prepare_h_num_stages"] == 2 and config["group_reduce_block_size"] == 16 for config in fwd))
        self.assertTrue(all(config["block_DV"] == 128 for config in bwd))

        ids = [_space.ConfigSpace.config_id(config) for config in fwd + bwd]
        self.assertEqual(len(set(ids)), 13)
        for config, identity in zip(fwd + bwd, ids, strict=True):
            canonical = _space.canonical_json(config)
            self.assertEqual(json.loads(canonical), config)
            self.assertEqual(_space.ConfigSpace.config_sha256(config), identity)

    def test_fwd_prunes_only_block_128_and_checks_only_that_config(self):
        raw = search.make_config_space("fwd", HARDWARE_PATH).raw_configs()
        solvers = []

        def solver_factory(_context):
            solver = _RecordingSolver()
            solvers.append(solver)
            return solver

        pruner = _space.Z3Pruner(seed=17, rlimit=99, solver_factory=solver_factory)
        result = search.prune_direction("fwd", HARDWARE_PATH, pruner=pruner)
        audit = result.audit

        self.assertEqual(result.configs, raw[:3])
        self.assertEqual(result.unknown, raw[:3])
        self.assertEqual(audit["raw"], 4)
        self.assertEqual(audit["unique"], 4)
        self.assertEqual(audit["kept"], 3)
        self.assertEqual(audit["sat"], 0)
        self.assertEqual(audit["unsat"], 1)
        self.assertEqual(audit["unknown"], 3)
        self.assertEqual(audit["checked"], 1)
        self.assertEqual(audit["pruned"], 1)
        self.assertEqual(len(solvers), 1)
        self.assertEqual(sum(solver.check_calls for solver in solvers), 1)
        self.assertEqual(solvers[0].clauses, [False])

        unsat = [record for record in audit["records"] if record["status"] == "unsat"]
        self.assertEqual(len(unsat), 1)
        self.assertEqual(unsat[0]["config"]["block_DV"], 128)
        self.assertEqual(
            [record["config"]["block_DV"] for record in audit["records"]],
            [16, 32, 64, 128],
        )
        for record in audit["records"]:
            if record["status"] == "unknown":
                self.assertEqual(record["stats"], {})
                self.assertIn("remain UNKNOWN", record["reason"])
        record_ids = [record["config_id"] for record in audit["records"]]
        self.assertEqual(len(set(record_ids)), 4)
        self.assertEqual(audit["unknown_config_ids"], record_ids[:3])

    def test_bwd_prunes_only_three_stage3_configs(self):
        raw = search.make_config_space("bwd", HARDWARE_PATH).raw_configs()
        solvers = []

        def solver_factory(_context):
            solver = _RecordingSolver()
            solvers.append(solver)
            return solver

        pruner = _space.Z3Pruner(seed=17, rlimit=99, solver_factory=solver_factory)
        result = search.prune_direction("bwd", HARDWARE_PATH, pruner=pruner)
        audit = result.audit
        expected_kept = [config for config in raw if config["prepare_h_num_stages"] != 3]

        self.assertEqual(len(expected_kept), 6)
        self.assertEqual(result.configs, expected_kept)
        self.assertEqual(result.unknown, expected_kept)
        self.assertEqual(audit["raw"], 9)
        self.assertEqual(audit["unique"], 9)
        self.assertEqual(audit["kept"], 6)
        self.assertEqual(audit["checked"], 3)
        self.assertEqual(audit["sat"], 0)
        self.assertEqual(audit["unsat"], 3)
        self.assertEqual(audit["unknown"], 6)
        self.assertEqual(audit["pruned"], 3)
        self.assertEqual(audit["raw"], audit["sat"] + audit["unsat"] + audit["unknown"] + audit["duplicates"])
        self.assertEqual(audit["kept"], audit["sat"] + audit["unknown"])
        self.assertEqual(len(solvers), 3)
        self.assertEqual(sum(solver.check_calls for solver in solvers), 3)
        self.assertTrue(all(solver.clauses == [False] for solver in solvers))

        unsat = [record["config"] for record in audit["records"] if record["status"] == "unsat"]
        self.assertEqual(
            [(config["prepare_h_num_stages"], config["group_reduce_block_size"]) for config in unsat],
            [(3, 8), (3, 16), (3, 32)],
        )
        for record in audit["records"]:
            if record["status"] == "unknown":
                self.assertEqual(record["stats"], {})
                self.assertIn("bwd prepare_h", record["reason"])
                self.assertIn("remain UNKNOWN", record["reason"])
        record_ids = [record["config_id"] for record in audit["records"]]
        self.assertEqual(len(set(record_ids)), 9)
        self.assertEqual(audit["unknown_config_ids"], record_ids[:6])

    def test_forward_formula_matches_evidence_and_never_promotes_under_cap_to_sat(self):
        evidence = json.loads(FWD_EVIDENCE_PATH.read_text(encoding="utf-8"))
        checker = resource.GDNResourceChecker(HARDWARE_PATH)
        self.assertEqual(checker.cap_bytes, 101376)

        for observation in evidence["observations"]:
            with self.subTest(block_DV=observation["block_DV"]):
                config = gdn_config.GDNConfig(
                    gdn_config.Direction.FWD,
                    block_DV=observation["block_DV"],
                )
                dynamic_bytes = resource.forward_dynamic_shared_bytes(config.block_DV)
                self.assertEqual(dynamic_bytes, observation["formula_bytes"])
                self.assertEqual(dynamic_bytes, observation["actual_dynamic_shared_bytes"])
                evaluation = gdn_config.evaluate_config(config, resource_checker=checker)
                expected = gdn_config.Status.UNSAT if dynamic_bytes > checker.cap_bytes else gdn_config.Status.UNKNOWN
                self.assertIs(evaluation.structural_status, gdn_config.Status.SAT)
                self.assertIs(evaluation.resource_check.status, expected)
                self.assertIs(evaluation.status, expected)
                self.assertIsNot(evaluation.resource_check.status, gdn_config.Status.SAT)

    def test_prepare_h_formula_matches_evidence_and_never_promotes_under_cap_to_sat(self):
        evidence = json.loads(PREPARE_H_EVIDENCE_PATH.read_text(encoding="utf-8"))
        checker = resource.GDNResourceChecker(HARDWARE_PATH)
        self.assertEqual(checker.cap_bytes, evidence["device_optin_limit_bytes"])

        for observation in evidence["observations"]:
            with self.subTest(stages=observation["prepare_h_num_stages"]):
                config = gdn_config.GDNConfig(
                    gdn_config.Direction.BWD,
                    prepare_h_num_stages=observation["prepare_h_num_stages"],
                )
                dynamic_bytes = resource.prepare_h_dynamic_shared_bytes(config.prepare_h_num_stages)
                self.assertEqual(dynamic_bytes, observation["formula_bytes"])
                self.assertEqual(dynamic_bytes, observation["actual_dynamic_shared_bytes"])
                evaluation = gdn_config.evaluate_config(config, resource_checker=checker)
                expected = gdn_config.Status.UNSAT if dynamic_bytes > checker.cap_bytes else gdn_config.Status.UNKNOWN
                self.assertIs(evaluation.structural_status, gdn_config.Status.SAT)
                self.assertIs(evaluation.resource_check.status, expected)
                self.assertIs(evaluation.status, expected)
                self.assertIsNot(evaluation.resource_check.status, gdn_config.Status.SAT)

    def test_missing_or_invalid_cap_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "missing-cap.json"
            missing.write_text(json.dumps({"device": {}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "shared_memory_per_block_optin_bytes"):
                resource.GDNResourceChecker(missing)

            invalid = root / "invalid-cap.json"
            invalid.write_text(
                json.dumps({"device": {"shared_memory_per_block_optin_bytes": None}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "invalid"):
                resource.GDNResourceChecker(invalid)

            with self.assertRaises(FileNotFoundError):
                resource.GDNResourceChecker(root / "absent.json")


if __name__ == "__main__":
    unittest.main()
