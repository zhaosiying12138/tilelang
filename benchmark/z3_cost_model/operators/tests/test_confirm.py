"""CPU/mock tests for classic winner selection, confirmation, and analysis.

No Torch, CUDA, profiler, native compiler, or GPU query is invoked.  Production
entry points are exercised with callable candidates and timing dependencies
that are entirely CPU mocks.
"""

from __future__ import annotations

from collections import Counter
import copy
import json
import os
from pathlib import Path
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmark.z3_cost_model.operators import analyze_confirmation, autotune, confirm  # noqa: E402


def _config(suite="gemm_square", variant=0):
    block_m = (64, 128, 256)[variant % 3]
    if suite in ("gemm_square", "gemm_short_k"):
        return {
            "block_M": block_m,
            "block_N": 64,
            "block_K": 32,
            "num_stages": 1,
            "threads": 128,
            "enable_rasterization": False,
        }
    if suite == "conv2d_stride_tail":
        return {"block_M": block_m, "block_N": 64, "block_K": 32, "num_stages": 1, "threads": 128}
    if suite == "flash_attention_causal":
        return {"block_M": block_m, "block_N": 64, "num_stages": 1, "threads": 128}
    raise AssertionError(suite)


def _default_trial_specs(suite, arm):
    del arm
    first = _config(suite, 0)
    second = _config(suite, 1)
    return [(first, [1.0, 5.0, 100.0]), (second, [100.0, 4.0, 6.0])]


def _cold_analysis(root: Path, suite="gemm_square", arm_trials=None):
    root.mkdir(parents=True, exist_ok=True)
    runs = []
    for arm in confirm.ARMS:
        specs = _default_trial_specs(suite, arm) if arm_trials is None else arm_trials[arm]
        for repetition in range(3):
            run_dir = root / suite / arm / f"rep-{repetition}"
            run_dir.mkdir(parents=True)
            summary_path = run_dir / "summary.json"
            events_path = run_dir / "events.jsonl"
            events = [
                {
                    "event": "run_start",
                    "arm": arm,
                    "metadata": {"suite": suite, "cache_track": "cold", "top_k": 10, "repetition": repetition},
                }
            ]
            for index, (config, _latencies) in enumerate(specs):
                events.append({"event": "config_generated", "phase": "selected", "index": index, "config": config})
            events.append({"event": "config_generation_end", "status": "ok"})
            trial_payloads = []
            for index, (config, latencies) in enumerate(specs):
                latency = latencies[repetition]
                trial_payloads.append((config, latency))
                events.append({"event": "trial", "index": index, "status": "ok", "config": config, "latency_ms": latency})
            best_config, best_latency = min(trial_payloads, key=lambda item: (item[1], confirm._canonical(item[0])))
            events.append({"event": "run_end", "status": "ok", "best_config": best_config, "best_latency_ms": best_latency})
            events_path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
            summary = {
                "status": "ok",
                "operator_run": {"suite": suite, "cache_track": "cold", "top_k": 10},
                "experiment": {"arm": arm, "repetition": repetition},
                "metadata": {"suite": suite},
            }
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            runs.append(
                {
                    "path": str(summary_path),
                    "events_path": str(events_path),
                    "suite": suite,
                    "arm": arm,
                    "repetition": repetition,
                    "cache_track": "cold",
                    "top_k": 10,
                    "performance_evidence": True,
                    "cache_hit": False,
                    "oracle_complete": True,
                    "best_config": best_config,
                    "best_latency_ms": best_latency,
                }
            )
    return {"runs": runs}


def _rewrite_events(row, events):
    Path(row["events_path"]).write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")


def _read_events(row):
    return [json.loads(line) for line in Path(row["events_path"]).read_text(encoding="utf-8").splitlines() if line]


def test_candidate_median_differs_from_modal_per_run_winner(tmp_path):
    analysis = _cold_analysis(tmp_path)
    guided_rows = [row for row in analysis["runs"] if row["arm"] == confirm.GUIDED_ARM]
    modal = Counter(confirm._canonical(row["best_config"]) for row in guided_rows).most_common(1)[0][0]
    assert json.loads(modal)["block_M"] == 128
    winners = confirm.candidate_median_winners(analysis, "gemm_square")
    for arm in confirm.ARMS:
        assert winners[arm]["config"]["block_M"] == 64
        assert winners[arm]["median_latency_ms"] == 5.0
        assert winners[arm]["trial_count"] == 3
        assert winners[arm]["measurement_repetitions"] == 3
        assert winners[arm]["latency_scope"] == "search_observed"


def test_candidate_median_canonical_json_breaks_exact_ties(tmp_path):
    arm_trials = {arm: [(_config(variant=1), [2.0, 2.0, 2.0]), (_config(variant=0), [2.0, 2.0, 2.0])] for arm in confirm.ARMS}
    analysis = _cold_analysis(tmp_path, arm_trials=arm_trials)
    winners = confirm.candidate_median_winners(analysis, "gemm_square")
    expected = min((confirm._canonical(_config(variant=0)), confirm._canonical(_config(variant=1))))
    for arm in confirm.ARMS:
        assert confirm._canonical(winners[arm]["config"]) == expected


def test_identical_winners_are_deduplicated_without_fake_pairs(tmp_path):
    assignments = {
        "stock-raw": 0,
        "z3-legal-exhaustive": 0,
        "z3-cm-topk": 0,
        "budget-raw": 1,
        "budget-random": 1,
    }
    arm_trials = {arm: [(_config(variant=variant), [1.0, 1.0, 1.0])] for arm, variant in assignments.items()}
    plan = confirm.confirmation_plan(_cold_analysis(tmp_path, arm_trials=arm_trials), "gemm_square")
    assert len(plan["candidates"]) == 2
    assert len(plan["comparisons"]) == 2
    identity = next(row for row in plan["comparisons"] if row["identical"])
    assert set(identity["identical_arms"]) == {"stock-raw", "z3-legal-exhaustive", "z3-cm-topk"}
    assert identity["alternative_arms"] == []
    alternative = next(row for row in plan["comparisons"] if not row["identical"])
    assert set(alternative["alternative_arms"]) == {"budget-raw", "budget-random"}
    schedule = confirm.measurement_schedule(plan, 7)
    assert len(schedule) == 14
    assert {row["pair_index"] for row in schedule} == {plan["comparisons"].index(alternative)}
    assert all(row["role"] != "identical" for row in schedule)


def test_all_identical_plan_measures_only_one_config(tmp_path):
    arm_trials = {arm: [(_config(), [1.0, 1.0, 1.0])] for arm in confirm.ARMS}
    plan = confirm.confirmation_plan(_cold_analysis(tmp_path, arm_trials=arm_trials), "gemm_square")
    assert len(plan["candidates"]) == 1
    schedule = confirm.measurement_schedule(plan, 7)
    assert len(schedule) == 7
    assert {row["role"] for row in schedule} == {"identical"}
    measurements = [{"pair_index": -1, "round": row["round"], "config_id": row["config_id"], "median_ms": 1.0} for row in schedule]
    result = confirm.evaluate_comparisons(plan, measurements)[0]
    assert result["quality_ratio_guided_over_alternative"] == 1.0
    assert result["paired_round_differences_ms"] == [0.0] * 7


def test_abba_order_and_expanded_round_order():
    assert [confirm.abba_order(index) for index in range(7)] == [
        (0, 1),
        (1, 0),
        (1, 0),
        (0, 1),
        (0, 1),
        (1, 0),
        (1, 0),
    ]
    with pytest.raises(ValueError):
        confirm.abba_order(-1)


def test_event_measurement_mock_counts_warmup_flush_and_samples():
    class FakeEvent:
        def __init__(self):
            self.recorded = False

        def record(self):
            self.recorded = True

        def elapsed_time(self, other):
            assert self.recorded and other.recorded
            return 1.25

    class FakeCuda:
        def __init__(self):
            self.synchronizes = 0

        def Event(self, *, enable_timing):
            assert enable_timing is True
            return FakeEvent()

        def synchronize(self):
            self.synchronizes += 1

    class FakeTorch:
        def __init__(self):
            self.cuda = FakeCuda()

    class FakeCache:
        def __init__(self):
            self.flushes = 0

        def zero_(self):
            self.flushes += 1

    class FakeCandidate:
        def __init__(self):
            self.calls = 0

        def __call__(self, *inputs):
            assert len(inputs) == 2
            self.calls += 1

    torch = FakeTorch()
    cache = FakeCache()
    candidate = FakeCandidate()
    stats, samples = confirm._measure_kernel(candidate, [object(), object()], cache, 50, 200, torch_module=torch)
    assert candidate.calls == 250
    assert cache.flushes == 200
    assert torch.cuda.synchronizes == 2
    assert len(samples) == 200
    assert stats["count"] == 200
    assert stats["median_ms"] == 1.25


def test_cache_is_run_owned_restored_and_nonempty_cache_rejected(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    names = (
        "TILELANG_CACHE_DIR",
        "TILELANG_DISABLE_CACHE",
        "TILELANG_AUTO_TUNING_DISABLE_CACHE",
        "CUDA_CACHE_PATH",
        "CUDA_CACHE_DISABLE",
        "TORCH_EXTENSIONS_DIR",
        "TRITON_CACHE_DIR",
        "XDG_CACHE_HOME",
        "TMPDIR",
    )
    previous = {}
    for index, name in enumerate(names):
        previous[name] = str(tmp_path / f"previous-{index}")
        monkeypatch.setenv(name, previous[name])
    with confirm._confirmation_cache(run_dir):
        for name in names:
            if name in {"TILELANG_DISABLE_CACHE", "TILELANG_AUTO_TUNING_DISABLE_CACHE", "CUDA_CACHE_DISABLE"}:
                assert os.environ[name] == "0"
            else:
                assert Path(os.environ[name]).is_relative_to(run_dir / "cache")
        assert Path(os.environ["TILELANG_CACHE_DIR"]) == run_dir / "cache" / "tilelang"
    for name in names:
        assert os.environ[name] == previous[name]
        assert not Path(previous[name]).exists()

    second = tmp_path / "second"
    (second / "cache").mkdir(parents=True)
    unknown = second / "cache" / "unknown.txt"
    unknown.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty"), confirm._confirmation_cache(second):
        pass
    assert unknown.read_text(encoding="utf-8") == "keep"


def test_nonempty_confirmation_directory_is_refused(tmp_path):
    run_dir = tmp_path / "gemm_square"
    run_dir.mkdir()
    (run_dir / "keep.txt").write_text("unknown", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty"):
        confirm._prepare_run_directory(tmp_path, "gemm_square")
    assert (run_dir / "keep.txt").read_text(encoding="utf-8") == "unknown"


@pytest.mark.parametrize("case", ["missing_events", "missing_config", "nonfinite_latency", "mixed_suite"])
def test_invalid_cold_evidence_is_rejected(tmp_path, case):
    analysis = _cold_analysis(tmp_path)
    row = analysis["runs"][0]
    if case == "missing_events":
        Path(row["events_path"]).unlink()
        expected = "Missing cold-trial events"
    elif case == "missing_config":
        events = _read_events(row)
        next(event for event in events if event["event"] == "trial")["config"] = None
        _rewrite_events(row, events)
        expected = "Missing or invalid config"
    elif case == "nonfinite_latency":
        events = _read_events(row)
        next(event for event in events if event["event"] == "trial")["latency_ms"] = float("nan")
        _rewrite_events(row, events)
        expected = "Invalid trial latency"
    else:
        summary_path = Path(row["path"])
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["operator_run"]["suite"] = "gemm_short_k"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        expected = "Mixed or inconsistent cold summary"
    with pytest.raises(ValueError, match=expected):
        confirm.candidate_median_winners(analysis, "gemm_square")


def test_real_candidate_factory_signature_is_forwarded_with_mocks():
    spec = autotune.get_suite("gemm_square")
    fixture = autotune.OperatorFixture(
        spec,
        device="cpu",
        input_factory=lambda *, shape, device, seed: {"A": object(), "B": object()},
        reference_factory=lambda inputs, *, shape: "expected",
        validate_factory=lambda actual, expected, *, shape: {"passed": True},
    )
    calls = []

    def kernel_factory(config, *, shape, target, pass_configs, compile):
        calls.append(
            {
                "config": copy.deepcopy(config),
                "shape": shape,
                "target": target,
                "pass_configs": copy.deepcopy(pass_configs),
                "compile": compile,
            }
        )
        return lambda *inputs: "actual"

    entries = [{"config": _config(variant=0)}, {"config": _config(variant=1)}]
    candidates = confirm._compile_candidates(
        spec,
        entries,
        fixture=fixture,
        target="mock-target",
        pass_configs={"mock.pass": 7},
        kernel_factory=kernel_factory,
    )
    assert len(candidates) == 2
    assert [call["config"] for call in calls] == [entry["config"] for entry in entries]
    for call in calls:
        assert call["shape"] is fixture.shape
        assert call["target"] == "mock-target"
        assert call["pass_configs"] == {"mock.pass": 7}
        assert call["compile"] is True
    assert all(callable(candidate) for candidate in candidates.values())


class _FakeCandidate:
    def __init__(self, config):
        self.config = config

    def __call__(self, *inputs):
        return "actual"


class _FakeFixture:
    def __init__(self, spec, *, seed, device):
        self.spec = spec
        self.seed = seed
        self.device = device
        names = ("A", "B") if spec.name.startswith("gemm") else ("data", "kernel")
        self.inputs = {name: object() for name in names}
        self.input_names = tuple(self.inputs)

    def reference_program(self, *args):
        assert all(arg is self.inputs[name] for arg, name in zip(args, self.input_names, strict=True))
        return "expected"

    def manual_check_prog(self, actual_outputs, reference_outputs):
        assert actual_outputs == ["actual"]
        assert reference_outputs == ["expected"]
        return {
            "passed": True,
            "mismatched_elements": 0,
            "total_elements": 1,
            "mismatched_ratio": 0.0,
            "max_mismatched_ratio": 0.0,
            "max_abs_error": 0.001,
            "max_rel_error": 0.002,
        }


def _mock_production_dependencies(record_calls=True):
    factory_calls = []
    measure_calls = []

    def builder(spec, **kwargs):
        factory_calls.append((spec, kwargs))

        def factory(**config):
            return _FakeCandidate(config)

        return factory

    def measure(candidate, inputs, cache, warmup, repeat, *, torch_module):
        measure_calls.append((candidate.config, warmup, repeat, cache, torch_module))
        base = 1.0 if candidate.config["block_M"] == 64 else 2.0
        samples = [base + index * 0.001 for index in range(repeat)]
        return confirm._sample_stats(samples), samples

    if not record_calls:

        def failing_measure(*args, **kwargs):
            raise RuntimeError("mock timing failure")

        measure = failing_measure
    return factory_calls, measure_calls, builder, measure


def _mock_run_analysis(root):
    arm_trials = {arm: [(_config(variant=1 if arm == "budget-raw" else 0), [1.0, 1.0, 1.0])] for arm in confirm.ARMS}
    return _cold_analysis(root, arm_trials=arm_trials)


def test_complete_mock_confirmation_preserves_samples_reports_and_abba(tmp_path, monkeypatch):
    analysis = _mock_run_analysis(tmp_path / "cold")
    analysis_path = tmp_path / "cold-analysis.json"
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
    monkeypatch.setattr(confirm, "_source_evidence", lambda: {"file_sha256": {"mock": "abc"}})
    factory_calls, measure_calls, builder, measure = _mock_production_dependencies()
    telemetry_calls = []

    def telemetry():
        telemetry_calls.append(1)
        return "mock-gpu-telemetry"

    summary = confirm.run_confirmation(
        "gemm_square",
        analysis_path,
        tmp_path / "confirmation",
        rounds=2,
        warmup=3,
        repeat=4,
        fixture_factory=_FakeFixture,
        candidate_factory_builder=builder,
        measure_fn=measure,
        telemetry_fn=telemetry,
    )
    assert summary["status"] == "ok"
    assert summary["acceptance"]["correctness_passed"] is True
    assert summary["acceptance"]["performance_report_only"] is True
    assert summary["acceptance"]["passed_scope"] == "execution_and_correctness_only"
    assert len(factory_calls) == 1
    assert factory_calls[0][1]["compile"] is True
    assert len(measure_calls) == 4
    assert all(call[1:] == (3, 4, None, None) for call in measure_calls)
    assert len(summary["measurements"]) == 4
    assert len(summary["validations"]) == 4
    assert {report["phase"] for report in summary["validations"]} == {"before", "after"}
    sample_lines = (tmp_path / "confirmation/gemm_square/samples.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(sample_lines) == 4
    assert all(len(json.loads(line)["samples_ms"]) == 4 for line in sample_lines)
    assert len(telemetry_calls) == 2 + 2 * len(measure_calls)
    assert (tmp_path / "confirmation/gemm_square/before.json").is_file()
    assert (tmp_path / "confirmation/gemm_square/after.json").is_file()


def test_failure_is_recorded_and_never_marked_ok(tmp_path, monkeypatch):
    analysis = _mock_run_analysis(tmp_path / "cold")
    analysis_path = tmp_path / "cold-analysis.json"
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
    monkeypatch.setattr(confirm, "_source_evidence", lambda: {"file_sha256": {"mock": "abc"}})
    _factory_calls, _measure_calls, builder, measure = _mock_production_dependencies(record_calls=False)
    with pytest.raises(RuntimeError, match="mock timing failure"):
        confirm.run_confirmation(
            "gemm_square",
            analysis_path,
            tmp_path / "confirmation",
            rounds=1,
            warmup=1,
            repeat=1,
            fixture_factory=_FakeFixture,
            candidate_factory_builder=builder,
            measure_fn=measure,
            telemetry_fn=lambda: "mock",
        )
    summary = json.loads((tmp_path / "confirmation/gemm_square/summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "error"
    assert "mock timing failure" in summary["error"]
    assert "acceptance" not in summary


def test_cold_failure_happens_before_fixture_or_gpu_telemetry(tmp_path, monkeypatch):
    analysis = _cold_analysis(tmp_path / "cold")
    Path(analysis["runs"][0]["events_path"]).unlink()
    analysis_path = tmp_path / "cold-analysis.json"
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
    monkeypatch.setattr(confirm, "_source_evidence", lambda: {"file_sha256": {"mock": "abc"}})
    forbidden = []
    with pytest.raises(ValueError, match="Missing cold-trial events"):
        confirm.run_confirmation(
            "gemm_square",
            analysis_path,
            tmp_path / "confirmation",
            fixture_factory=lambda *args, **kwargs: forbidden.append("fixture"),
            telemetry_fn=lambda: forbidden.append("telemetry"),
        )
    assert forbidden == []
    summary = json.loads((tmp_path / "confirmation/gemm_square/summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "error"


def _cold_aggregates():
    aggregates = []
    for suite in confirm.SUITE_NAMES:
        for arm_index, arm in enumerate(confirm.ARMS):
            aggregates.append(
                {
                    "suite": suite,
                    "arm": arm,
                    "top_k": 10,
                    "cache_track": "cold",
                    "compile_attempts": 20 + arm_index,
                    "profile_attempts": 10 + arm_index,
                    "end_to_end_wall_seconds": 40.0 + arm_index,
                    "best_latency_ms": 8.0 + arm_index,
                    "search_speedup": 1.0,
                    "solver_wall_seconds": None if arm == "stock-raw" else 1.0,
                    "model_wall_seconds": 2.0 if arm == "z3-cm-topk" else None,
                    "compile_wall_seconds": 20.0,
                    "validation_wall_seconds": 4.0,
                    "profile_wall_seconds": 5.0,
                }
            )
    return {"aggregates": aggregates}


def _comparison(suite, config, *, arms, identical, alternative_latency):
    guided_latency = 4.0
    return {
        "config": config,
        "config_id": confirm._canonical(config),
        "identical": identical,
        "identical_arms": arms if identical else [],
        "alternative_arms": [] if identical else arms,
        "relations": ["same_space_exhaustive"] if identical else ["equal_budget_control"],
        "guided_median_ms": guided_latency,
        "alternative_median_ms": guided_latency if identical else alternative_latency,
        "quality_ratio_guided_over_alternative": 1.0 if identical else guided_latency / alternative_latency,
        "paired_round_differences_ms": [0.0] * 7,
        "requirements": {},
        "passed": True,
    }


def _confirmation_summary(suite):
    variants = {"stock-raw": 0, "z3-legal-exhaustive": 0, "z3-cm-topk": 0, "budget-raw": 1, "budget-random": 2}
    winners = {
        arm: {
            "config": _config(suite, variant),
            "selection_rule": "candidate-median",
            "median_latency_ms": 2.0,
            "trial_count": 3,
            "measurement_repetitions": 3,
        }
        for arm, variant in variants.items()
    }
    candidates = [
        {"config": _config(suite, 0), "arms": ["z3-cm-topk"]},
        {"config": _config(suite, 1), "arms": ["budget-raw"]},
        {"config": _config(suite, 2), "arms": ["budget-random"]},
    ]
    comparisons = [
        _comparison(
            suite,
            _config(suite, 0),
            arms=["stock-raw", "z3-legal-exhaustive", "z3-cm-topk"],
            identical=True,
            alternative_latency=4.0,
        ),
        _comparison(suite, _config(suite, 1), arms=["budget-raw"], identical=False, alternative_latency=4.5),
        _comparison(suite, _config(suite, 2), arms=["budget-random"], identical=False, alternative_latency=4.25),
    ]
    validations = []
    for candidate in candidates:
        for phase in ("before", "after"):
            validations.append(
                {
                    "config_id": confirm._canonical(candidate["config"]),
                    "phase": phase,
                    "status": "ok",
                    "passed": True,
                    "mismatched_ratio": 0.0,
                    "max_mismatched_ratio": 0.0,
                    "max_abs_error": 0.01,
                    "max_rel_error": 0.02,
                }
            )
    return {
        "status": "ok",
        "suite": suite,
        "rounds": 7,
        "plan": {
            "selection_rule": "candidate-median",
            "winners": winners,
            "candidates": candidates,
            "legal_oracle_complete": True,
        },
        "comparisons": comparisons,
        "validations": validations,
        "acceptance": {"passed": True},
    }


def _confirmation_inputs():
    return _cold_aggregates(), {suite: _confirmation_summary(suite) for suite in confirm.SUITE_NAMES}


def test_every_suite_arm_has_one_config_and_one_confirmed_latency():
    cold, confirmations = _confirmation_inputs()
    rows, comparisons = analyze_confirmation.confirmed_records(cold, confirmations)
    assert len(rows) == 20
    assert len(comparisons) == 12
    assert {(row["suite"], row["method"]) for row in rows} == {(suite, arm) for suite in confirm.SUITE_NAMES for arm in confirm.ARMS}
    guided = next(row for row in rows if row["suite"] == "gemm_square" and row["method"] == "z3-cm-topk")
    assert guided["best_latency"] == 4.0
    assert guided["latency_scope"] == "confirmed"
    assert guided["count_scope"] == "search_observed"
    assert guided["search_observed_latency_ms"] == 10.0
    stock = next(row for row in rows if row["suite"] == "gemm_square" and row["method"] == "stock-raw")
    legal = next(row for row in rows if row["suite"] == "gemm_square" and row["method"] == "z3-legal-exhaustive")
    assert stock["search_space"] == "raw-full"
    assert "not a Z3 legal oracle" in stock["reference_semantics"]
    assert legal["search_space"] == "legal"
    assert legal["reference_semantics"] == "complete Z3 legal oracle"


def test_analysis_rejects_duplicate_and_missing_arm_assignments():
    cold, confirmations = _confirmation_inputs()
    duplicate = copy.deepcopy(confirmations)
    duplicate["gemm_square"]["comparisons"].append(
        _comparison("gemm_square", _config("gemm_square", 2), arms=["budget-random"], identical=False, alternative_latency=4.25)
    )
    with pytest.raises(ValueError, match="Duplicate confirmed assignment"):
        analyze_confirmation.confirmed_records(cold, duplicate)

    missing = copy.deepcopy(confirmations)
    identity = missing["gemm_square"]["comparisons"][0]
    identity["identical_arms"].remove("stock-raw")
    with pytest.raises(ValueError, match="Missing confirmed assignment"):
        analyze_confirmation.confirmed_records(cold, missing)


def test_write_analysis_emits_json_csv_and_usable_plots(tmp_path, monkeypatch):
    cold, confirmations = _confirmation_inputs()
    cold_path = tmp_path / "cold-analysis.json"
    cold_path.write_text(json.dumps(cold), encoding="utf-8")
    root = tmp_path / "confirmation"
    for suite, summary in confirmations.items():
        path = root / suite / "summary.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(summary), encoding="utf-8")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    payload = analyze_confirmation.write_analysis(cold_path, root, tmp_path / "analysis")
    assert len(payload["rows"]) == 20
    assert payload["correctness_passed"] is True
    assert payload["performance_report_only"] is True
    assert payload["acceptance_scope"] == "execution_and_correctness_only"
    for name in ("analysis.json", "confirmed-summary.csv", "confirmation-comparisons.csv"):
        assert (tmp_path / "analysis" / name).stat().st_size > 0
    for prefix in ("classic-confirmation", "classic-search-time", "classic-quality-pareto"):
        for suffix in (".png", ".svg"):
            path = tmp_path / "analysis" / f"{prefix}{suffix}"
            assert path.stat().st_size > 100
    written = json.loads((tmp_path / "analysis/analysis.json").read_text(encoding="utf-8"))
    assert written["latency_scopes"]["best_latency"] == "confirmed"
    assert written["latency_scopes"]["compile_profile_counts"] == "search_observed"


def test_internal_pass_configs_are_normalized_only_when_null():
    spec = autotune.get_suite("gemm_square")
    config = _config()
    normalized, canonical = confirm._validated_config(spec, {"__pass_configs__": None, **config}, context="test")
    assert normalized == config
    assert canonical == confirm._canonical(config)
    with pytest.raises(ValueError, match="Per-config pass_configs"):
        confirm._validated_config(spec, {"__pass_configs__": {"any": True}, **config}, context="test")


def test_cli_defaults_match_preregistered_confirmation_protocol():
    args = confirm.build_parser().parse_args(["--suite", "gemm_square", "--analysis", "analysis.json"])
    assert args.rounds == 7
    assert args.warmup == 50
    assert args.repeat == 200
    assert args.seed == 20261026
