"""Archive raw compile-matrix logs without recording machine-specific paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import zipfile


LOG_SUFFIXES = (".stdout.log", ".stderr.log", ".driver.log")
MANIFEST_NAME = "logs-manifest.json"
SCHEMA = "tilelang.cuda.compile-matrix-logs.v1"
ROOT_NOTE = (
    "Archive paths are relative to the directory supplied with --root; "
    "the source absolute path is intentionally omitted."
)
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def _is_cache_component(name):
    return name == "cache" or name.endswith("-cache")


def _raise_walk_error(error):
    raise error


def _collect_logs(root):
    logs = []
    for directory, dirnames, filenames in os.walk(
        root, topdown=True, onerror=_raise_walk_error, followlinks=False
    ):
        dirnames[:] = sorted(
            name for name in dirnames if not _is_cache_component(name)
        )
        for name in sorted(filenames):
            if not name.endswith(LOG_SUFFIXES):
                continue
            source_path = Path(directory) / name
            if not source_path.is_file():
                continue
            archive_path = source_path.relative_to(root).as_posix()
            logs.append((archive_path, source_path))
    logs.sort(key=lambda item: item[0])
    return logs


def _manifest_bytes(logs):
    records = []
    total_bytes = 0
    for archive_path, source_path in logs:
        data = source_path.read_bytes()
        records.append({
            "archive_path": archive_path,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        })
        total_bytes += len(data)

    manifest = {
        "schema": SCHEMA,
        "root": ROOT_NOTE,
        "file_count": len(records),
        "total_uncompressed_bytes": total_bytes,
        "files": records,
    }
    text = json.dumps(
        manifest,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    return (text + "\n").encode("utf-8"), records, total_bytes


def _zip_info(archive_path):
    info = zipfile.ZipInfo(archive_path, date_time=ZIP_TIMESTAMP)
    info.create_system = 3
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    return info


def _write_member(archive, archive_path, data):
    archive.writestr(
        _zip_info(archive_path),
        data,
        compress_type=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    )


def _write_zip(temp_path, logs, manifest_data, records):
    with zipfile.ZipFile(temp_path, mode="w") as archive:
        manifest_written = False
        for (archive_path, source_path), record in zip(logs, records):
            if not manifest_written and MANIFEST_NAME < archive_path:
                _write_member(archive, MANIFEST_NAME, manifest_data)
                manifest_written = True

            data = source_path.read_bytes()
            if (
                len(data) != record["bytes"]
                or hashlib.sha256(data).hexdigest() != record["sha256"]
            ):
                raise RuntimeError(
                    f"source log changed while archiving: {archive_path}"
                )
            _write_member(archive, archive_path, data)

        if not manifest_written:
            _write_member(archive, MANIFEST_NAME, manifest_data)


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_logs(root, output):
    root = Path(root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"log root is not a directory: {root}")
    output = Path(output).expanduser()
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite existing output: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    logs = _collect_logs(root)
    manifest_data, records, total_bytes = _manifest_bytes(logs)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temp_path = Path(temp_name)
    try:
        os.close(fd)
        _write_zip(temp_path, logs, manifest_data, records)
        zip_sha256 = _file_sha256(temp_path)
        if os.path.lexists(output):
            raise FileExistsError(f"refusing to overwrite existing output: {output}")
        os.replace(temp_path, output)
    except BaseException:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise

    return {
        "output": str(output),
        "files": len(records),
        "bytes": total_bytes,
        "zip_sha256": zip_sha256,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    summary = archive_logs(args.root, args.output)
    print(json.dumps(
        summary,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
