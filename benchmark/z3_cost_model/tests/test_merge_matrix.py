"""Unit tests for denominator-preserving compile-matrix shard merges."""

from __future__ import annotations

import json
from pathlib import Path
import types

import pytest

from benchmark.z3_cost_model import compile_matrix, merge_matrix


def _record(row, selected):
    numeric = selected and row["id"] == "b"
    return {
        "id": row["id"],
        "selected": selected,
        "attempted": selected,
        "status": (compile_matrix.PASS_NUMERIC if numeric else compile_matrix.PASS_COMPILE_ONLY) if selected else compile_matrix.NOT_RUN,
        "numeric_gate": numeric,
        "manifest_sha256": "manifest",
        "pytest": {name: int(selected) if name == "passed" else 0 for name in compile_matrix.PYTEST_COUNT_NAMES},
        "stdout": f"logs/{row['id']}.out",
        "stderr": f"logs/{row['id']}.err",
        "cache_dir": f"cache/{row['id']}",
    }


def _rows():
    return [{"id": "a"}, {"id": "b"}, {"id": "fixture"}]


def _matrix(rows):
    return types.SimpleNamespace(
        sha256="manifest",
        path=Path("manifest.json"),
        tier_rows=lambda _tier: rows,
    )


def _selection(rows, **_kwargs):
    return [(types.SimpleNamespace(**row), row["id"] != "fixture", None) for row in rows]


def _shard(root, name, rows, selected_id):
    shard = root / name
    shard.mkdir()
    records = [_record(row, row["id"] == selected_id) for row in rows]
    (shard / "results.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    (shard / "run.json").write_text(json.dumps({
        "run_id": name,
        "action": "run",
        "tier": "full",
        "mode": "default",
        "manifest_sha256": "manifest",
        "source_sha256": {"test.py": "source"},
        "only": [selected_id],
        "started_at": "start",
        "finished_at": "finish",
    }), encoding="utf-8")
    (shard / "logs").mkdir()
    (shard / "cache" / selected_id).mkdir(parents=True)
    (shard / "logs" / f"{selected_id}.out").write_text("pass", encoding="utf-8")
    (shard / "logs" / f"{selected_id}.err").write_text("", encoding="utf-8")
    return shard


def _install_fake_manifest(monkeypatch, rows):
    monkeypatch.setattr(merge_matrix, "load_manifest", lambda: _matrix(rows))
    monkeypatch.setattr(merge_matrix, "select_rows", _selection)


def test_complete_disjoint_shards_merge_all_rows(tmp_path, monkeypatch):
    rows = _rows()
    _install_fake_manifest(monkeypatch, rows)
    first = _shard(tmp_path, "shard-a", rows, "a")
    second = _shard(tmp_path, "shard-b", rows, "b")
    result = merge_matrix.merge_shards([first, second], tmp_path / "merged")
    summary = result["summary"]
    assert summary["complete"] is True
    assert summary["denominator"] == 3
    assert summary["selected"] == 2
    assert summary["not_selected"] == 1
    assert summary["passed"] == 2
    assert summary["numeric_passed"] == 1
    assert len(result["records"]) == 3
    assert all(record["shard_run_id"] in ("shard-a", "shard-b") for record in result["records"])
    assert (tmp_path / "merged" / "results.jsonl").is_file()


def test_duplicate_executed_row_is_rejected(tmp_path, monkeypatch):
    rows = _rows()
    _install_fake_manifest(monkeypatch, rows)
    first = _shard(tmp_path, "shard-a", rows, "a")
    duplicate = _shard(tmp_path, "duplicate-a", rows, "a")
    with pytest.raises(ValueError, match="multiple shards"):
        merge_matrix.merge_shards([first, duplicate], tmp_path / "merged")


def test_missing_eligible_row_is_rejected(tmp_path, monkeypatch):
    rows = _rows()
    _install_fake_manifest(monkeypatch, rows)
    first = _shard(tmp_path, "shard-a", rows, "a")
    with pytest.raises(ValueError, match="eligible rows missing"):
        merge_matrix.merge_shards([first], tmp_path / "merged")
