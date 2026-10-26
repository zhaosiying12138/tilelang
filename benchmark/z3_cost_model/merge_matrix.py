"""Merge disjoint compile-matrix shards without losing the frozen denominator."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

try:
    from .compile_matrix import (
        NOT_RUN,
        aggregate_results,
        load_manifest,
        select_rows,
    )
except ImportError:
    from compile_matrix import (
        NOT_RUN,
        aggregate_results,
        load_manifest,
        select_rows,
    )


def _load(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _records(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write(path, value):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def merge_shards(shards, output, *, tier="full", mode="default", manifest_path=None):
    if not shards:
        raise ValueError("at least one shard is required")
    output = Path(output).expanduser().resolve(strict=False)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"refusing a non-empty merge output: {output}")
    matrix = load_manifest(manifest_path) if manifest_path is not None else load_manifest()
    rows = matrix.tier_rows(tier)
    selection = select_rows(rows, action="run", mode=mode)
    eligible = {row.id for row, selected, _ in selection if selected}
    selected_records = {}
    fallback_records = None
    shard_metadata = []
    source_hashes = None
    run_ids = set()

    for shard_value in shards:
        shard = Path(shard_value).expanduser().resolve(strict=False)
        run = _load(shard / "run.json")
        if run["action"] != "run" or run["tier"] != tier or run["mode"] != mode:
            raise ValueError(f"shard action/tier/mode mismatch: {shard}")
        if run["manifest_sha256"] != matrix.sha256:
            raise ValueError(f"manifest hash mismatch: {shard}")
        if source_hashes is None:
            source_hashes = run["source_sha256"]
        elif run["source_sha256"] != source_hashes:
            raise ValueError(f"source hashes changed between shards: {shard}")
        if run["run_id"] in run_ids:
            raise ValueError(f"duplicate shard run_id: {run['run_id']}")
        run_ids.add(run["run_id"])
        records = _records(shard / "results.jsonl")
        if len(records) != len(rows):
            raise ValueError(f"frozen row count mismatch in {shard}: {len(records)} != {len(rows)}")
        if fallback_records is None:
            fallback_records = records
        for record in records:
            if record["manifest_sha256"] != matrix.sha256:
                raise ValueError(f"record manifest hash mismatch in {shard}: {record['id']}")
            if not record["selected"] or record["status"] == NOT_RUN:
                continue
            if record["id"] in selected_records:
                raise ValueError(f"row appears as executed in multiple shards: {record['id']}")
            selected_records[record["id"]] = (record, shard)
        shard_metadata.append({
            "run_id": run["run_id"],
            "only": run["only"],
            "output": str(shard),
            "started_at": run["started_at"],
            "finished_at": run["finished_at"],
        })

    missing = sorted(eligible - set(selected_records))
    if missing:
        raise ValueError(f"incomplete shard set: {len(missing)} eligible rows missing, first={missing[:3]}")
    unexpected = sorted(set(selected_records) - eligible)
    if unexpected:
        raise ValueError(f"shard set contains ineligible rows: {unexpected[:3]}")

    output.mkdir(parents=True)
    merged = []
    fallback_by_id = {record["id"]: record for record in fallback_records}
    for row, selected, _ in selection:
        if selected:
            record, shard = selected_records[row.id]
        else:
            record = dict(fallback_by_id[row.id])
            shard = Path(shard_metadata[0]["output"])
        record = dict(record)
        for key in ("stdout", "stderr", "cache_dir"):
            if key in record:
                record[key] = os.path.relpath(shard / record[key], output)
        record["shard_run_id"] = next(item["run_id"] for item in shard_metadata if Path(item["output"]) == shard)
        record["shard_output"] = os.path.relpath(shard, output)
        merged.append(record)

    summary = aggregate_results(merged)
    summary.update({
        "action": "run",
        "tier": tier,
        "mode": mode,
        "manifest_sha256": matrix.sha256,
        "results": "results.jsonl",
        "merged": True,
        "complete": True,
        "shard_count": len(shard_metadata),
        "shards": shard_metadata,
    })
    with (output / "results.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for record in merged:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n")
    _write(output / "summary.json", summary)
    _write(output / "run.json", {
        "schema": "tilelang.cuda.compile-matrix-merge",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "action": "run",
        "tier": tier,
        "mode": mode,
        "manifest": str(matrix.path),
        "manifest_sha256": matrix.sha256,
        "source_sha256": source_hashes,
        "shards": shard_metadata,
        "summary": summary,
    })
    return {"records": merged, "summary": summary, "output": output}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("quick", "full", "host"), default="full")
    parser.add_argument("--mode", choices=("default", "z3-opt-in"), default="default")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("shards", nargs="+", type=Path)
    args = parser.parse_args(argv)
    result = merge_shards(args.shards, args.output, tier=args.tier, mode=args.mode, manifest_path=args.manifest)
    print(json.dumps({"output": str(result["output"]), "summary": result["summary"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
