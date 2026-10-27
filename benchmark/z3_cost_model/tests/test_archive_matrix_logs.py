"""Tests for deterministic, cache-free compile-matrix log archives."""

from __future__ import annotations

import hashlib
import json
import zipfile

import pytest

from benchmark.z3_cost_model import archive_matrix_logs as aml


def _write_files(root, files):
    for relative_path, data in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def _representative_logs():
    return {
        "batches/batch-0/logs/x.stdout.log": b"compile stdout\n",
        "classification-final/batch-0/logs/x.stderr.log": b"compile stderr\n",
        "classification-final/batch-0/logs/x.driver.log": b"driver output\n",
        "cache-results/kept.driver.log": b"not a cache directory\n",
    }


def test_archive_manifest_exclusions_and_cli_summary(tmp_path, capsys):
    root = tmp_path / "matrix"
    expected = _representative_logs()
    _write_files(root, expected)
    _write_files(root, {
        "batches/batch-0/cache/hidden.stdout.log": b"cache",
        "batches/batch-0/compile-cache/hidden.stderr.log": b"cache",
        "classification-final/batch-0/not-cache/hidden.driver.log": b"cache",
        "batches/batch-0/logs/results.json": b"not a log",
        "batches/batch-0/logs/x.stdout.log.tmp": b"not a log",
        "batches/batch-0/logs/stdout.log": b"not a matching suffix",
    })
    output = tmp_path / "new" / "parents" / "matrix-logs.zip"

    assert aml.main(["--root", str(root), "--output", str(output)]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert len(captured.out.strip().splitlines()) == 1
    summary = json.loads(captured.out)

    expected_paths = sorted(expected)
    expected_bytes = sum(map(len, expected.values()))
    assert summary == {
        "output": str(output),
        "files": len(expected),
        "bytes": expected_bytes,
        "zip_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }

    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert archive.namelist() == sorted(expected_paths + [aml.MANIFEST_NAME])
        for info in archive.infolist():
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
            assert info.compress_type == zipfile.ZIP_DEFLATED
            assert (info.external_attr >> 16) & 0o777 == 0o644
        for archive_path, data in expected.items():
            assert archive.read(archive_path) == data

        manifest = json.loads(archive.read(aml.MANIFEST_NAME))

    assert manifest["schema"] == aml.SCHEMA
    assert manifest["root"] == aml.ROOT_NOTE
    assert manifest["file_count"] == len(expected)
    assert manifest["total_uncompressed_bytes"] == expected_bytes
    assert manifest["files"] == [
        {
            "archive_path": archive_path,
            "bytes": len(expected[archive_path]),
            "sha256": hashlib.sha256(expected[archive_path]).hexdigest(),
        }
        for archive_path in expected_paths
    ]


def test_zip_bytes_are_deterministic_across_output_names(tmp_path):
    root = tmp_path / "matrix"
    _write_files(root, _representative_logs())
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    first_summary = aml.archive_logs(root, first)
    second_summary = aml.archive_logs(root, second)

    assert first.read_bytes() == second.read_bytes()
    assert first_summary["zip_sha256"] == second_summary["zip_sha256"]
    assert first_summary["files"] == second_summary["files"]
    assert first_summary["bytes"] == second_summary["bytes"]


def test_existing_output_is_not_overwritten(tmp_path):
    root = tmp_path / "matrix"
    _write_files(root, _representative_logs())
    output = tmp_path / "existing.zip"
    output.write_bytes(b"existing evidence")
    unrelated = tmp_path / "unrelated.tmp"
    unrelated.write_bytes(b"keep me")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        aml.archive_logs(root, output)

    assert output.read_bytes() == b"existing evidence"
    assert unrelated.read_bytes() == b"keep me"


def test_manifest_does_not_leak_absolute_source_path(tmp_path):
    root = tmp_path / "private-source-tree"
    _write_files(root, {"nested/run.driver.log": b"driver\n"})
    output = tmp_path / "logs.zip"
    aml.archive_logs(root, output)

    with zipfile.ZipFile(output) as archive:
        manifest_text = archive.read(aml.MANIFEST_NAME).decode("utf-8")

    assert str(root) not in manifest_text
    assert root.name not in manifest_text
    assert json.loads(manifest_text)["files"][0]["archive_path"] == "nested/run.driver.log"
