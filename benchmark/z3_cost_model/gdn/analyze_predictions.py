"""Join GDN model scores with measured latencies and plot rank quality."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import statistics


def _key(config, direction):
    return (
        direction,
        config.get("block_DV"),
        config.get("prepare_h_num_stages"),
        config.get("group_reduce_block_size"),
    )


def collect(root, cache_track="cold"):
    if cache_track not in ("cold", "warm", "no-cache"):
        raise ValueError(f"Unknown cache track: {cache_track}")
    scores = {}
    measured = {}
    for path in Path(root).rglob("events.jsonl"):
        direction = None
        include = True
        for line in path.open(encoding="utf-8"):
            event = json.loads(line)
            if event.get("event") == "run_start":
                metadata = event.get("metadata", {})
                direction = metadata.get("direction")
                track = metadata.get("cache_track")
                if track is None:
                    mode = metadata.get("cache_mode")
                    track = "cold" if mode == "enabled" else mode
                include = track == cache_track
            if not include:
                continue
            if event.get("event") == "score":
                key = _key(event["config"], direction)
                scores.setdefault(key, {
                    "suite": direction,
                    "candidate": f"{key[0]}:block_DV={key[1]},stage={key[2]},group={key[3]}",
                    "predicted": event["score"],
                    "config": event["config"],
                    "config_id": event["config_id"],
                    "path": str(path),
                })
            if event.get("event") == "trial" and event.get("status") == "ok":
                key = _key(event["config"], direction)
                measured.setdefault(key, []).append(event["latency_ms"])
    if not scores:
        raise ValueError(f"No score records found under {root}")
    predicted_records = []
    measured_records = []
    for key, score in sorted(scores.items(), key=lambda item: item[1]["predicted"]):
        if key not in measured or not measured[key]:
            raise ValueError(f"No measured latency for candidate {key}: model predictions cannot be silently dropped")
        predicted_records.append({
            "suite": score["suite"],
            "candidate": score["candidate"],
            "predicted": score["predicted"],
            "config_id": score["config_id"],
            "config": score["config"],
            "path": score["path"],
        })
        measured_records.append({
            "suite": score["suite"],
            "candidate": score["candidate"],
            "measured": statistics.median(measured[key]),
            "config_id": score["config_id"],
            "config": score["config"],
            "path": score["path"],
        })
    return predicted_records, measured_records


def _load_visualization(repo_root):
    path = repo_root / "benchmark/z3_cost_model/visualization.py"
    spec = importlib.util.spec_from_file_location("gdn_prediction_visualization", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-track", choices=("cold", "warm", "no-cache"), default="cold")
    args = parser.parse_args(argv)
    predicted, measured = collect(args.root, cache_track=args.cache_track)
    repo_root = Path(__file__).resolve().parents[3]
    viz = _load_visualization(repo_root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    viz.plot_prediction_rank(predicted, measured, output / "prediction-rank", top_k=(1, 2, 3))
    metrics = {}
    for direction in ("fwd", "bwd"):
        p = [row["predicted"] for row in predicted if row["suite"] == direction]
        m = [row["measured"] for row in measured if row["suite"] == direction]
        metrics[direction] = {
            "count": len(p),
            "spearman": viz.spearman_rank_correlation(p, m),
            "kendall_tau_b": viz.kendall_tau_b(p, m),
            "top1_recall": viz.top_k_recall(p, m, 1),
            "top2_recall": viz.top_k_recall(p, m, 2),
            "top3_recall": viz.top_k_recall(p, m, 3),
        }
    (output / "prediction-metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    (output / "predicted.json").write_text(json.dumps(predicted, indent=2, allow_nan=False) + "\n")
    (output / "measured.json").write_text(json.dumps(measured, indent=2, allow_nan=False) + "\n")
    print(json.dumps(metrics, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
