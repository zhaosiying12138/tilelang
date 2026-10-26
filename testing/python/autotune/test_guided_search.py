"""Pure-Python tests for backend-neutral guided search infrastructure.

The synthetic package below loads only the two source files under test.  It
never executes tilelang/__init__.py, so this file can be collected without a
TileLang native installation or a GPU.  Z3 is imported lazily only by the real
solver test and by default pruners that actually have constraints.
"""

import hashlib
import importlib
import json
import random
import sys
import types
from pathlib import Path

import pytest


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_TILELANG_DIR = _REPOSITORY_ROOT / "tilelang"
_AUTOTUNER_DIR = _TILELANG_DIR / "autotuner"
_TEST_PACKAGE = "_tilelang_g4_backend_neutral_test"


def _install_synthetic_package(name, path):
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    package.__package__ = name
    sys.modules[name] = package


_install_synthetic_package(_TEST_PACKAGE, _TILELANG_DIR)
_install_synthetic_package(f"{_TEST_PACKAGE}.autotuner", _AUTOTUNER_DIR)
_space = importlib.import_module(f"{_TEST_PACKAGE}.autotuner.space")
_guided = importlib.import_module(f"{_TEST_PACKAGE}.autotuner.guided")

ConfigSpace = _space.ConfigSpace
ConstraintBuildError = _space.ConstraintBuildError
Indeterminate = _space.Indeterminate
Z3Pruner = _space.Z3Pruner
canonical_json = _space.canonical_json
CandidateCostModel = _guided.CandidateCostModel
CostModelCapacityError = _guided.CostModelCapacityError
CostModelError = _guided.CostModelError
GuidedSearch = _guided.GuidedSearch


class _MockSolver:
    def __init__(self, verdict, reason=""):
        self.verdict = verdict
        self.reason = reason
        self.settings = {}
        self.clauses = []
        self.check_calls = 0

    def set(self, name, value):
        self.settings[name] = value

    def add(self, *clauses):
        self.clauses.extend(clauses)

    def check(self):
        self.check_calls += 1
        if isinstance(self.verdict, BaseException):
            raise self.verdict
        return self.verdict

    def reason_unknown(self):
        return self.reason

    def statistics(self):
        return {"checks": self.check_calls, "label": "mock"}


def _mock_pruner(outcomes):
    pending = iter(outcomes)
    solvers = []

    def solver_factory(_context):
        verdict, reason = next(pending)
        solver = _MockSolver(verdict, reason)
        solvers.append(solver)
        return solver

    pruner = Z3Pruner(
        constraint_fn=lambda _config, _z3, _context: True,
        seed=17,
        rlimit=99,
        solver_factory=solver_factory,
    )
    return pruner, solvers


class _ValueModel:
    def __init__(self, score_fn=None):
        self.score_fn = score_fn or (lambda config: float(config["x"]))
        self.calls = []

    def predict(self, config):
        self.calls.append(dict(config))
        score = float(self.score_fn(config))
        return score, {"compute": float(config["x"]), "memory": float(config["x"] + 1)}


class _PairModel:
    def predict(self, config):
        score = float(config["x"] + config["y"])
        return score, {"sum": score}


def test_config_order_and_hash_are_stable():
    space = ConfigSpace([("block", [2, 1]), ("mode", ["b", "a"])])

    expected = [
        {"block": 2, "mode": "b"},
        {"block": 2, "mode": "a"},
        {"block": 1, "mode": "b"},
        {"block": 1, "mode": "a"},
    ]
    assert space.raw_configs() == expected
    assert space.raw_configs() == expected
    assert list(space.domains) == ["block", "mode"]
    assert len(space) == 4

    config = {"mode": "a", "block": 1}
    expected_json = '{"block":1,"mode":"a"}'
    expected_hash = hashlib.sha256(expected_json.encode("utf-8")).hexdigest()
    assert ConfigSpace.canonical_json(config) == expected_json
    assert canonical_json(config) == expected_json
    assert ConfigSpace.config_id(config) == expected_hash
    assert ConfigSpace.config_sha256(config) == expected_hash
    assert ConfigSpace.config_id({"block": 1, "mode": "a"}) == expected_hash
    assert ConfigSpace.config_id({"block": 2, "mode": "a"}) != expected_hash


def test_domain_validation_and_raw_config_detachment():
    with pytest.raises(ValueError, match="duplicate canonical JSON"):
        ConfigSpace({"x": [1, 1]})
    with pytest.raises(TypeError, match="ordered iterable"):
        ConfigSpace({"x": {1, 2}})
    with pytest.raises(ValueError, match="non-finite"):
        ConfigSpace({"x": [float("nan")]})

    space = ConfigSpace({"nested": [[1, 2]], "meta": [{"b": 2, "a": 1}]})
    first = space.raw_configs()
    first[0]["nested"].append(3)
    first[0]["meta"]["a"] = 99
    assert space.raw_configs() == [{"nested": [1, 2], "meta": {"a": 1, "b": 2}}]


def test_coupled_constraints_define_legal_universe_with_real_z3():
    def coupled(config, z3, context):
        x = z3.IntVal(config["x"], ctx=context)
        y = z3.IntVal(config["y"], ctx=context)
        return [x >= 0, y >= 0, x + y <= 2]

    space = ConfigSpace({"x": [0, 1, 2], "y": [0, 1, 2]}, constraint_builder=coupled)
    result = Z3Pruner(seed=42, rlimit=10_000).prune(space)

    # Independent arithmetic oracle: the constraint couples x and y, so this
    # is not the product of independently filtered domains.
    expected = [config for config in space.raw_configs() if config["x"] + config["y"] <= 2]
    assert len(expected) == 6
    assert result.configs == expected
    assert result.unknown == []
    assert result.audit["raw"] == 9
    assert result.audit["checked"] == 9
    assert result.audit["sat"] == 6
    assert result.audit["unsat"] == 3
    assert result.audit["unknown"] == 0

    search = GuidedSearch(space, _PairModel())
    assert len(search.legal_configs()) == 6
    assert len(search.top_k(100)) == 6


def test_pruner_classifies_sat_unsat_and_unknown_conservatively():
    pruner, solvers = _mock_pruner([("sat", ""), ("unsat", ""), ("unknown", "incomplete")])
    configs = [{"x": 0}, {"x": 1}, {"x": 2}]

    result = pruner.prune(configs)

    assert result.configs == [{"x": 0}, {"x": 2}]
    assert result.unknown == [{"x": 2}]
    assert result.audit["sat"] == 1
    assert result.audit["unsat"] == 1
    assert result.audit["unknown"] == 1
    assert result.audit["kept"] == 2
    assert result.audit["pruned"] == 1
    assert result.audit["checked"] == 3
    assert result.audit["stats"]["checks"] == 3
    assert [record["reason"] for record in result.audit["records"]] == ["sat", "unsat", "incomplete"]
    assert all(solver.settings == {"random_seed": 17, "rlimit": 99} for solver in solvers)


def test_unknown_timeouts_and_resource_limits_are_retained_and_itemized():
    pruner, _ = _mock_pruner(
        [
            ("unknown", "resource limits reached"),
            ("unknown", "timeout after 5 ms"),
            (TimeoutError("deadline"), ""),
        ]
    )
    configs = [{"x": 0}, {"x": 1}, {"x": 2}]

    result = pruner.prune(configs)

    assert result.configs == configs
    assert result.unknown == configs
    assert result.audit["unknown"] == 3
    assert result.audit["timeouts"] == 2
    assert result.audit["resource_limited"] == 1
    assert result.audit["unknown_config_ids"] == [ConfigSpace.config_id(config) for config in configs]
    assert result.audit["timeout_config_ids"] == [ConfigSpace.config_id(config) for config in configs[1:]]
    assert result.audit["resource_limited_config_ids"] == [ConfigSpace.config_id(configs[0])]
    assert [record["timeout"] for record in result.audit["records"]] == [False, True, True]


def test_duplicates_have_a_separate_audit_bucket_and_self_consistent_denominator():
    first = {"x": 0}
    unsat = {"x": 1}
    unknown = {"x": 2}
    pruner, solvers = _mock_pruner([("sat", ""), ("unsat", ""), ("unknown", "incomplete")])

    result = pruner.prune([first, dict(first), unsat, dict(unsat), unknown])
    audit = result.audit

    assert len(solvers) == 3
    assert result.configs == [first, unknown]
    assert audit["raw"] == 5
    assert audit["unique"] == 3
    assert audit["duplicates"] == 2
    assert audit["unsat"] == 1
    assert audit["raw"] == audit["sat"] + audit["unsat"] + audit["unknown"] + audit["duplicates"]
    assert audit["unique"] == audit["sat"] + audit["unsat"] + audit["unknown"]
    assert audit["kept"] == audit["sat"] + audit["unknown"]
    assert audit["pruned"] == audit["unsat"]
    assert len(audit["records"]) == audit["raw"]
    assert [record["status"] for record in audit["records"]] == ["sat", "duplicate", "unsat", "duplicate", "unknown"]
    assert audit["records"][1]["duplicate_of"] == ConfigSpace.config_id(first)
    assert audit["records"][3]["duplicate_of"] == ConfigSpace.config_id(unsat)
    assert audit["wall_seconds"] >= 0
    assert all(record["wall_seconds"] >= 0 for record in audit["records"])
    json.dumps(audit, allow_nan=False)


def test_external_indeterminate_skips_solver_and_preserves_denominators():
    config = {"x": 0}

    def forbidden_solver(_context):
        raise AssertionError("external UNKNOWN must not construct a solver")

    pruner = Z3Pruner(
        constraint_fn=lambda _config, _z3, _context: Indeterminate("precomputed resource UNKNOWN"),
        solver_factory=forbidden_solver,
    )
    result = pruner.prune([config, dict(config)])
    audit = result.audit

    assert result.configs == [config]
    assert result.unknown == [config]
    assert audit["raw"] == 2
    assert audit["unique"] == 1
    assert audit["duplicates"] == 1
    assert audit["checked"] == 0
    assert audit["sat"] == 0
    assert audit["unsat"] == 0
    assert audit["unknown"] == 1
    assert audit["kept"] == 1
    assert audit["pruned"] == 0
    assert audit["raw"] == audit["sat"] + audit["unsat"] + audit["unknown"] + audit["duplicates"]
    assert audit["unique"] == audit["sat"] + audit["unsat"] + audit["unknown"]
    assert audit["kept"] == audit["sat"] + audit["unknown"]
    assert [record["status"] for record in audit["records"]] == ["unknown", "duplicate"]
    assert audit["records"][0]["reason"] == "precomputed resource UNKNOWN"
    assert audit["records"][0]["stats"] == {}
    assert audit["stats"] == {}
    json.dumps(audit, allow_nan=False)


def test_none_builder_result_still_errors_without_solver():
    def forbidden_solver(_context):
        raise AssertionError("an invalid builder result must not construct a solver")

    pruner = Z3Pruner(
        constraint_fn=lambda _config, _z3, _context: None,
        solver_factory=forbidden_solver,
    )
    with pytest.raises(ConstraintBuildError, match="returned None"):
        pruner.prune([{"x": 0}])


def test_constraint_builder_error_is_explicit_not_unsat():
    def broken(config, _z3, _context):
        if config["x"] == 1:
            raise RuntimeError("cannot encode")
        return True

    pruner, solvers = _mock_pruner([("sat", ""), ("sat", "")])
    pruner.constraint_fn = broken

    with pytest.raises(ConstraintBuildError, match="cannot encode"):
        pruner.prune([{"x": 0}, {"x": 1}])
    assert len(solvers) == 1
    assert solvers[0].check_calls == 1


def test_equal_budget_raw_random_and_guided_candidates():
    space = ConfigSpace({"x": list(range(6))})
    model = _ValueModel(score_fn=lambda config: -float(config["x"]))
    search = GuidedSearch(space, model, seed=23)
    assert isinstance(model, CandidateCostModel)

    controls = search.control_candidates(4)
    guided = search.candidates(4, "guided")
    assert set(controls) == {"raw", "random"}
    assert len(guided) == len(controls["raw"]) == len(controls["random"]) == 4
    assert controls["raw"] == [{"x": 0}, {"x": 1}, {"x": 2}, {"x": 3}]
    assert guided == [{"x": 5}, {"x": 4}, {"x": 3}, {"x": 2}]
    assert len({ConfigSpace.config_id(config) for config in controls["random"]}) == 4
    assert all(config in search.legal_configs() for config in controls["random"])

    oversized = search.control_candidates(100)
    assert len(oversized["raw"]) == len(oversized["random"]) == len(search.candidates(100)) == 6
    assert search.candidates(0, "raw") == []
    assert search.candidates(0, "random") == []
    assert search.candidates(0, "guided") == []


def test_top_k_tie_break_is_deterministic_and_capped():
    space = ConfigSpace({"x": [2, 0, 1, 3]})

    class TieModel:
        def predict(self, _config):
            return 7.0, {"constant": 1.0}

    search = GuidedSearch(space, TieModel())
    expected = sorted(space.raw_configs(), key=lambda config: (ConfigSpace.config_id(config), canonical_json(config)))

    assert search.top_k(2) == expected[:2]
    assert search.top_k(100) == expected
    assert len(search.top_k(100)) == len(space)
    assert [record["rank"] for record in search.predictions()] == [1, 2, 3, 4]
    assert [record["score"] for record in search.predictions()] == [7.0] * 4


def test_repeated_runs_and_predictions_are_deterministic():
    space = ConfigSpace({"x": list(range(5))})
    first_model = _ValueModel(score_fn=lambda config: float(config["x"] % 2))
    second_model = _ValueModel(score_fn=lambda config: float(config["x"] % 2))
    first = GuidedSearch(space, first_model, seed=31)
    second = GuidedSearch(space, second_model, seed=31)

    assert first.legal_configs() == second.legal_configs()
    assert first.predictions() == second.predictions()
    assert first.predictions() == second.predictions()
    assert first.top_k(3) == second.top_k(3)
    assert first.control_candidates(3) == second.control_candidates(3)
    assert first.candidates(3, "random") == second.candidates(3, "random")
    assert first.predictions_jsonl() == second.predictions_jsonl()
    assert len(first_model.calls) == len(second_model.calls) == 5

    detached = first.predictions()
    detached[0]["config"]["x"] = 99
    assert first.predictions() != detached


def test_random_seeds_are_reproducible_without_global_rng_mutation():
    space = ConfigSpace({"x": list(range(10))})
    raw = space.raw_configs()
    first = GuidedSearch(space, _ValueModel(), seed=7)
    same = GuidedSearch(space, _ValueModel(), seed=7)
    different = GuidedSearch(space, _ValueModel(), seed=8)

    expected = random.Random(7).sample(raw, 5)
    global_state = random.getstate()
    assert first.candidates(5, "random") == expected
    assert random.getstate() == global_state
    assert first.candidates(5, "random") == expected
    assert same.candidates(5, "random") == expected
    assert different.candidates(5, "random") == random.Random(8).sample(raw, 5)
    assert different.candidates(5, "random") != expected


def test_prediction_records_and_jsonl_are_json_serializable(tmp_path):
    search = GuidedSearch(ConfigSpace({"x": [1, 0]}), _ValueModel())

    records = search.predictions()
    payload = search.predictions_jsonl()
    lines = payload.splitlines()

    assert len(records) == len(lines) == 2
    assert payload.endswith("\n")
    assert [json.loads(line) for line in lines] == records
    assert [record["config"] for record in records] == [{"x": 0}, {"x": 1}]
    assert records[0]["terms"] == {"compute": 0.0, "memory": 1.0}
    for record in records:
        assert set(record) == {"rank", "universe_index", "config", "config_id", "score", "terms"}
        json.dumps(record, allow_nan=False)

    output_path = tmp_path / "predictions.jsonl"
    assert search.write_predictions(output_path) == output_path
    assert output_path.read_text(encoding="utf-8") == payload


def test_capacity_unknown_is_last_deterministic_and_available_to_top_k():
    class CapacityModel:
        def __init__(self):
            self.calls = []

        def predict(self, config):
            self.calls.append(config["x"])
            if config["x"] >= 2:
                raise CostModelCapacityError(f"shared capacity is uncertain for x={config['x']}")
            return float(config["x"]), {"known": 1.0}

    space = ConfigSpace({"x": [3, 2, 0, 1]})
    model = CapacityModel()
    search = GuidedSearch(space, model)
    records = search.predictions()
    expected_unknown = sorted(
        [{"x": 2}, {"x": 3}],
        key=lambda config: (ConfigSpace.config_id(config), canonical_json(config)),
    )

    assert [record["config"] for record in records[:2]] == [{"x": 0}, {"x": 1}]
    assert [record["config"] for record in records[2:]] == expected_unknown
    assert [record["rank"] for record in records] == [1, 2, 3, 4]
    for record in records[2:]:
        assert record["unknown"] is True
        assert record["error_type"] == "capacity"
        assert "shared capacity" in record["error"]
        assert record["score"] is None
        assert record["terms"] == {"capacity_error": 1.0, "fallback_count": 1.0}
        json.dumps(record, allow_nan=False)

    assert search.top_k(2) == [{"x": 0}, {"x": 1}]
    assert search.top_k(3) == [{"x": 0}, {"x": 1}, expected_unknown[0]]
    assert search.top_k(4) == [{"x": 0}, {"x": 1}, *expected_unknown]
    assert search.model_unknown_configs() == expected_unknown
    assert search.model_unknown_configs() == expected_unknown
    assert model.calls == [3, 2, 0, 1]
    assert [json.loads(line) for line in search.predictions_jsonl().splitlines()] == records

    repeated = GuidedSearch(space, CapacityModel())
    assert repeated.predictions() == records
    assert repeated.predictions_jsonl() == search.predictions_jsonl()


def test_non_capacity_cost_model_error_is_not_reclassified_as_unknown():
    class BrokenModel:
        def predict(self, _config):
            raise CostModelError("invalid analytical formula")

    search = GuidedSearch(ConfigSpace({"x": [0]}), BrokenModel())
    with pytest.raises(CostModelError, match="legal index 0.*invalid analytical formula"):
        search.predictions()


def test_cost_model_exception_is_explicit_and_not_silently_dropped():
    class BrokenModel:
        def __init__(self):
            self.calls = []

        def predict(self, config):
            self.calls.append(config["x"])
            if config["x"] == 1:
                raise RuntimeError("unknown latency")
            return 1.0, {"known": 1.0}

    model = BrokenModel()
    search = GuidedSearch(ConfigSpace({"x": [0, 1, 2]}), model)

    with pytest.raises(CostModelError, match="legal index 1.*unknown latency"):
        search.predictions()
    assert model.calls == [0, 1]


@pytest.mark.parametrize(
    "bad_prediction",
    [
        (None, {}),
        (True, {}),
        (float("nan"), {}),
        (float("inf"), {}),
        (1.0, None),
        (1.0, {"latency": float("nan")}),
    ],
)
def test_null_or_non_finite_prediction_without_capacity_error_is_an_explicit_failure(bad_prediction):
    class InvalidModel:
        def predict(self, _config):
            return bad_prediction

    search = GuidedSearch(ConfigSpace({"x": [0]}), InvalidModel())
    with pytest.raises(CostModelError):
        search.predictions()


def test_empty_legal_universe_has_empty_top_k_controls_and_jsonl():
    class MustNotRunModel:
        def predict(self, _config):
            raise AssertionError("an empty universe must not be scored")

    search = GuidedSearch(ConfigSpace({"x": []}), MustNotRunModel())

    assert search.legal_configs() == []
    assert search.unknown_configs() == []
    assert search.predictions() == []
    assert search.model_unknown_configs() == []
    assert search.top_k(5) == []
    assert search.candidates(5, "guided") == []
    assert search.control_candidates(5) == {"raw": [], "random": []}
    assert search.predictions_jsonl() == ""
    assert search.pruning_audit()["raw"] == 0
