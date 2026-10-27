#!/usr/bin/env python3
"""Compare two compile-matrix results.jsonl files without hiding old failures."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

PASS = {"PASS_COMPILE_ONLY", "PASS_NUMERIC"}


def load(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = {row["id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate row ID in {path}")
    return result


def category(before: str, after: str) -> str:
    old_fail = before.startswith("FAIL_")
    new_fail = after.startswith("FAIL_")
    if before == "MISSING" or after == "MISSING":
        return "ROW_SET_CHANGED"
    if old_fail and new_fail:
        return "PREEXISTING_FAILURE_STILL_FAILING"
    if new_fail:
        return "NEW_FAILURE"
    if old_fail and after in PASS:
        return "OLD_FAILURE_NOW_PASS"
    if old_fail:
        return "OLD_FAILURE_NOW_NONPASS"
    if before in PASS and after not in PASS:
        return "PASS_LOST"
    return "OTHER_STATUS_CHANGE"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", required=True, type=Path)
    parser.add_argument("--new", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    old = load(args.old)
    new = load(args.new)
    audit = []
    for row_id in sorted(old.keys() | new.keys()):
        before_row = old.get(row_id)
        after_row = new.get(row_id)
        before = before_row["status"] if before_row else "MISSING"
        after = after_row["status"] if after_row else "MISSING"
        if before == after and not before.startswith("FAIL_") and not after.startswith("FAIL_"):
            continue
        audit.append({
            "id": row_id,
            "category": category(before, after),
            "old_status": before,
            "new_status": after,
            "old_reason": before_row.get("reason") if before_row else None,
            "new_reason": after_row.get("reason") if after_row else None,
            "old_stdout": before_row.get("stdout") if before_row else None,
            "new_stdout": after_row.get("stdout") if after_row else None,
        })
    report = {
        "schema": "tilelang.compile-matrix-status-comparison/v1",
        "old_results": str(args.old),
        "new_results": str(args.new),
        "old_rows": len(old),
        "new_rows": len(new),
        "status_changed_rows": sum(row["old_status"] != row["new_status"] for row in audit),
        "old_failure_rows": sum(row["status"].startswith("FAIL_") for row in old.values()),
        "new_failure_rows": sum(row["status"].startswith("FAIL_") for row in new.values()),
        "new_failure_ids": [row["id"] for row in audit if row["category"] == "NEW_FAILURE"],
        "category_counts": dict(collections.Counter(row["category"] for row in audit)),
        "audit": audit,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "audit"}, indent=2, ensure_ascii=False))
    return 1 if report["new_failure_ids"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
