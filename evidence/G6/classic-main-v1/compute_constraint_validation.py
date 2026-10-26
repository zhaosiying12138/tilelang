"""Validate the versioned shared-memory revision against real raw outcomes."""

from __future__ import annotations

import collections
import json
from pathlib import Path

from benchmark.z3_cost_model.operators.autotune import (
    DEFAULT_CALIBRATION_PATH,
    DEFAULT_HARDWARE_PATH,
    SUITES,
    constraint_model_from_hardware,
    model_shape,
)
from tilelang.autotuner.guided import GuidedSearch
from tilelang.cuda.cost_model import CudaCostModel


ROOT = Path("evidence/G6/classic-main-v1")
STOCK = ROOT / "gemm_square-rep-0/gemm_square/top-k-0010/stock-raw__rep-0000"
GUIDED = ROOT / "gemm_square-rep-0/gemm_square/top-k-0010/z3-cm-topk__rep-0000"


def canonical(config):
    return json.dumps({key: value for key, value in config.items() if key != "__pass_configs__"}, sort_keys=True, separators=(",", ":"))


def main():
    events = [json.loads(line) for line in (STOCK / "events.jsonl").open(encoding="utf-8")]
    trials = [event for event in events if event.get("event") == "trial"]
    failures = [event for event in events if event.get("event") == "validation_end" and event.get("status") not in ("ok", "passed")]
    failed = {canonical(event["config"]) for event in failures}
    predicted = {
        canonical(event["config"])
        for event in trials
        if max(
            2 * event["config"]["block_K"] * (event["config"]["block_M"] + event["config"]["block_N"]) * max(1, event["config"]["num_stages"]),
            2 * event["config"]["block_M"] * event["config"]["block_N"],
        ) > 101376
    }
    missing = sorted(predicted - failed)
    unexpected = sorted(failed - predicted)

    spec = SUITES["gemm_square"]
    constraint, _ = constraint_model_from_hardware(DEFAULT_HARDWARE_PATH)
    space = spec.module.config_space(shape=spec.shape, model=constraint)
    model = CudaCostModel.from_files(
        DEFAULT_HARDWARE_PATH,
        DEFAULT_CALIBRATION_PATH,
        spec.model_operator,
        model_shape(spec),
        require_measured=True,
    )
    search = GuidedSearch(space, model, seed=20261026)
    legal = search.legal_configs()
    top10 = search.top_k(10)
    top20 = search.top_k(20)
    guided_events = [json.loads(line) for line in (GUIDED / "events.jsonl").open(encoding="utf-8")]
    old_top10 = [event["config"] for event in guided_events if event.get("event") == "config_generated" and event.get("phase") == "selected"]
    old_legal = json.load((GUIDED.parent / "z3-legal-exhaustive__rep-0000/summary.json").open(encoding="utf-8"))["U"]
    raw_best = {
        "block_K": 64,
        "block_M": 128,
        "block_N": 128,
        "enable_rasterization": True,
        "num_stages": 2,
        "threads": 256,
    }
    payload = {
        "schema": "tilelang.classic.shared-constraint-validation/v1",
        "role": "constraint validation, not a new performance run",
        "formula": "max(2*block_K*(block_M+block_N)*max(1,num_stages), 2*block_M*block_N)",
        "hardware_opt_in_bytes": 101376,
        "raw_trials": len(trials),
        "actual_failed_configs": len(failed),
        "predicted_over_capacity_configs": len(predicted),
        "missing_from_failure": missing,
        "unexpected_failure": unexpected,
        "failed_by_num_stages": dict(sorted(collections.Counter(event["config"]["num_stages"] for event in failures).items())),
        "old_legal": old_legal,
        "new_legal": len(legal),
        "old_top10": old_top10,
        "new_top10": top10,
        "top10_unchanged": [canonical(config) for config in old_top10] == [canonical(config) for config in top10],
        "raw_best_config": raw_best,
        "raw_best_in_top10": raw_best in top10,
        "raw_best_in_top20": raw_best in top20,
        "performance_model_coefficients_changed": False,
        "source": {
            "stock_events": str(STOCK / "events.jsonl"),
            "guided_events": str(GUIDED / "events.jsonl"),
            "formula": "benchmark/z3_cost_model/operators/gemm.py::_shared_memory_lower_bound",
        },
    }
    assert not missing and not unexpected
    assert payload["top10_unchanged"]
    assert old_legal == 256 and len(legal) == 240
    output = ROOT / "constraint-validation.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key not in ("old_top10", "new_top10")}, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
