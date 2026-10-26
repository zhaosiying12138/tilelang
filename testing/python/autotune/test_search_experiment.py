"""Pure-Python tests for guided experiment directory and cache safety."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import sys
import time
import types

import pytest


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_TILELANG_DIR = _REPOSITORY_ROOT / "tilelang"
_AUTOTUNER_DIR = _TILELANG_DIR / "autotuner"
_ALIAS = "_tilelang_g4_experiment_test"


def _install_package(name, path):
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    package.__package__ = name
    sys.modules[name] = package


_install_package(_ALIAS, _TILELANG_DIR)
_install_package(f"{_ALIAS}.autotuner", _AUTOTUNER_DIR)
_experiment = importlib.import_module(f"{_ALIAS}.autotuner.experiment")
ExperimentSafetyError = _experiment.ExperimentSafetyError
SearchExperiment = _experiment.SearchExperiment


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class _FakeTuner:
    """Cache-aware tuner double; it never imports TileLang or touches a GPU."""

    def __init__(self, calls, outside_cache=False):
        self.calls = calls
        self.outside_cache = outside_cache
        self.last_summary = None

    def run(self, *, jsonl_path, metadata, **_kwargs):
        cache_root = Path(os.environ["TILELANG_CACHE_DIR"])
        disabled = os.environ.get("TILELANG_DISABLE_CACHE") == "1" or os.environ.get("TILELANG_AUTO_TUNING_DISABLE_CACHE") == "1"
        result_dir = cache_root / "namespace" / "autotuner" / "fake-key"
        marker = result_dir / "result.json"
        hit = marker.exists() and not disabled
        self.calls.append(
            {
                "cache_root": str(cache_root),
                "disabled": disabled,
                "hit": hit,
                "metadata": dict(metadata),
            }
        )
        if not disabled and not hit:
            result_dir.mkdir(parents=True)
            marker.write_text('{"cached":true}\n', encoding="utf-8")
        events = [
            {"event": "run_start", "perf_counter": time.perf_counter()},
            {"event": "cache_hit" if hit else "run_end", "perf_counter": time.perf_counter()},
        ]
        Path(jsonl_path).write_text("".join(json.dumps(event, allow_nan=False) + "\n" for event in events), encoding="utf-8")
        self.last_summary = {
            "schema_version": 1,
            "status": "cache_hit" if hit else "ok",
            "R": 1,
            "U": 1,
            "unknown": 0,
            "duplicates": 0,
            "compile_attempts": 0 if hit else (0 if disabled else 1),
            "best_config": {"x": 1},
            "best_latency_ms": 1.0,
            "cache_identity": "fake-key",
            "cache": {
                "identity": "fake-key",
                "directory": str(Path(self.outside_cache) if self.outside_cache else result_dir),
                "hit": hit,
            },
        }
        return {"best_config": {"x": 1}, "cache_hit": hit}


class _FailingTuner:
    def __init__(self):
        self.last_summary = None

    def run(self, *, jsonl_path, metadata, **_kwargs):
        Path(jsonl_path).write_text('{"event":"run_start"}\n', encoding="utf-8")
        self.last_summary = {"status": "error", "cache": {"directory": None}}
        raise RuntimeError("candidate exploded")


def test_enabled_track_has_empty_before_and_complete_after(tmp_path, monkeypatch):
    monkeypatch.setenv("TILELANG_CACHE_DIR", "/original/cache")
    monkeypatch.setenv("TILELANG_DISABLE_CACHE", "original-global")
    monkeypatch.setenv("TILELANG_AUTO_TUNING_DISABLE_CACHE", "original-autotune")
    calls = []
    experiment = SearchExperiment(tmp_path / "experiment")

    outcome = experiment.run(
        lambda: _FakeTuner(calls),
        arm="stock-raw",
        repetition=0,
        cache_mode="enabled",
        metadata={"purpose": "cpu-test"},
    )

    assert outcome.result["cache_hit"] is False
    assert calls[0]["disabled"] is False
    assert Path(calls[0]["cache_root"]) == outcome.run_directory / "cache"
    assert calls[0]["metadata"]["purpose"] == "cpu-test"
    assert os.environ["TILELANG_CACHE_DIR"] == "/original/cache"
    assert os.environ["TILELANG_DISABLE_CACHE"] == "original-global"
    assert os.environ["TILELANG_AUTO_TUNING_DISABLE_CACHE"] == "original-autotune"

    before = _load(outcome.before_path)
    assert before["empty"] is True
    assert before["entries"] == before["cache_entries"] == []
    after = _load(outcome.after_path)
    paths = {entry["path"] for entry in after["entries"]}
    assert {"before.json", "events.jsonl", "summary.json", "cache", "cache/namespace/autotuner/fake-key/result.json"} <= paths
    assert "after.json" not in paths
    assert _load(outcome.run_directory / "summary.json") == outcome.summary
    assert outcome.summary["experiment"]["cache_mode"] == "enabled"
    json.dumps(outcome.summary, allow_nan=False)


def test_no_cache_track_does_not_write_cache(tmp_path):
    calls = []
    outcome = SearchExperiment(tmp_path / "experiment").run(
        lambda: _FakeTuner(calls),
        arm="stock-raw",
        repetition=0,
        cache_mode="no-cache",
    )
    assert calls[0]["disabled"] is True
    assert not (outcome.run_directory / "cache" / "namespace").exists()
    after = _load(outcome.after_path)
    assert [entry["path"] for entry in after["cache_entries"]] == ["cache"]


def test_warm_track_prewarms_then_measures_fresh_tuner(tmp_path):
    calls = []
    outcome = SearchExperiment(tmp_path / "experiment").run(
        lambda: _FakeTuner(calls),
        arm="z3-cm-topk",
        repetition=3,
        cache_mode="warm",
    )
    assert len(calls) == 2
    assert calls[0]["hit"] is False
    assert calls[1]["hit"] is True
    assert outcome.warmup_result["cache_hit"] is False
    assert outcome.result["cache_hit"] is True
    assert outcome.summary["compile_attempts"] == 0
    assert _load(outcome.run_directory / "warmup_summary.json")["status"] == "ok"
    assert (outcome.run_directory / "warmup.jsonl").exists()
    assert (outcome.run_directory / "events.jsonl").exists()


def test_warm_track_reuses_neither_the_same_tuner_nor_its_memory(tmp_path):
    calls = []
    shared = _FakeTuner(calls)
    experiment = SearchExperiment(tmp_path / "experiment")
    with pytest.raises(ExperimentSafetyError, match="fresh tuner"):
        experiment.run(lambda: shared, arm="stock-raw", repetition=0, cache_mode="warm")
    assert len(calls) == 1
    run_dir = tmp_path / "experiment" / "stock-raw__rep-0000"
    assert (run_dir / "failure.json").exists()
    assert (run_dir / "after.json").exists()


def test_each_repetition_gets_a_new_directory_and_repeat_is_rejected(tmp_path):
    experiment = SearchExperiment(tmp_path / "experiment")
    first = experiment.run(lambda: _FakeTuner([]), arm="budget-raw", repetition=0)
    second = experiment.run(lambda: _FakeTuner([]), arm="budget-raw", repetition=1)
    assert first.run_directory != second.run_directory
    assert first.run_directory.name == "budget-raw__rep-0000"
    assert second.run_directory.name == "budget-raw__rep-0001"
    with pytest.raises(ExperimentSafetyError, match="non-empty"):
        experiment.run(lambda: _FakeTuner([]), arm="budget-raw", repetition=0)


def test_non_empty_unfamiliar_run_directory_is_never_cleaned_or_reused(tmp_path):
    root = tmp_path / "experiment"
    target = root / "stock-raw__rep-0000"
    target.mkdir(parents=True)
    unfamiliar = target / "keep.txt"
    unfamiliar.write_text("do not delete", encoding="utf-8")
    called = False

    def factory():
        nonlocal called
        called = True
        return _FakeTuner([])

    with pytest.raises(ExperimentSafetyError, match="non-empty"):
        SearchExperiment(root).run(factory, arm="stock-raw", repetition=0)
    assert called is False
    assert unfamiliar.read_text(encoding="utf-8") == "do not delete"


def test_cleanup_requires_matching_owned_manifest_and_preserves_strangers(tmp_path):
    root = tmp_path / "experiment"
    experiment = SearchExperiment(root)
    outcome = experiment.run(lambda: _FakeTuner([]), arm="stock-raw", repetition=0)
    stranger = root / "unfamiliar"
    stranger.mkdir()
    (stranger / "keep.txt").write_text("safe", encoding="utf-8")

    removed = experiment.cleanup()
    assert removed == [outcome.run_directory]
    assert not outcome.run_directory.exists()
    assert (stranger / "keep.txt").read_text(encoding="utf-8") == "safe"

    owned = experiment.run(lambda: _FakeTuner([]), arm="stock-raw", repetition=1)
    manifest_path = owned.run_directory / ".g4-run.json"
    manifest = _load(manifest_path)
    manifest["run_name"] = "different-name"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ExperimentSafetyError, match="ownership/name mismatch"):
        experiment.cleanup_run(owned.run_directory)
    assert owned.run_directory.exists()


def test_default_tilelang_cache_is_forbidden_as_root():
    with pytest.raises(ExperimentSafetyError, match="default cache"):
        SearchExperiment(Path.home() / ".tilelang" / "g4-experiment")


def test_reported_cache_outside_run_is_an_explicit_failure(tmp_path):
    outside = tmp_path / "outside-cache"
    experiment = SearchExperiment(tmp_path / "experiment")
    with pytest.raises(ExperimentSafetyError, match="outside the experiment"):
        experiment.run(
            lambda: _FakeTuner([], outside_cache=outside),
            arm="stock-raw",
            repetition=0,
        )
    run_dir = tmp_path / "experiment" / "stock-raw__rep-0000"
    assert (run_dir / "failure.json").exists()
    assert (run_dir / "after.json").exists()
    assert not outside.exists()


def test_tuner_exception_is_recorded_and_reraised(tmp_path):
    experiment = SearchExperiment(tmp_path / "experiment")
    with pytest.raises(RuntimeError, match="candidate exploded"):
        experiment.run(_FailingTuner, arm="stock-raw", repetition=0)
    run_dir = tmp_path / "experiment" / "stock-raw__rep-0000"
    assert _load(run_dir / "failure.json")["error"] == "candidate exploded"
    assert _load(run_dir / "summary.json")["status"] == "error"
    assert (run_dir / "after.json").exists()
