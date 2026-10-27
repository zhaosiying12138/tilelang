#!/usr/bin/env python3
"""Frozen, isolated pytest driver for the CUDA compile/correctness matrix.

The manifest is the node inventory.  This program never discovers a different
run inventory with globbing or pytest -k.  It has no third-party imports and
never installs dependencies, builds TileLang, or clears a cache.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any


RUNNER_VERSION = 1
RESULT_SCHEMA = "tilelang.cuda.compile-matrix-result"
SUMMARY_SCHEMA = "tilelang.cuda.compile-matrix-summary"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "manifest" / "compile_matrix.json"
PASS_CONFIG_ENV = "TILELANG_COMPILE_MATRIX_PASS_CONFIGS"
MODE_ENV = "TILELANG_COMPILE_MATRIX_MODE"
ROW_ENV = "TILELANG_COMPILE_MATRIX_ROW_ID"
ARCH_ENV = "TILELANG_MATRIX_ARCH"
Z3_PIPELINE_CONFIG = "tl.enable_z3_pipeline_planning"

COLLECTED = "COLLECTED"
PASS_COMPILE_ONLY = "PASS_COMPILE_ONLY"
PASS_NUMERIC = "PASS_NUMERIC"
G_SMOKE = "G_SMOKE"
SKIP_ARCH = "SKIP_ARCH"
SKIP_DEP = "SKIP_DEP"
SKIP_PERF = "SKIP_PERF"
FAIL_LOWER = "FAIL_LOWER"
FAIL_DEVICE_COMPILE = "FAIL_DEVICE_COMPILE"
FAIL_LAUNCH = "FAIL_LAUNCH"
FAIL_NUMERIC = "FAIL_NUMERIC"
FAIL_ENV = "FAIL_ENV"
FAIL_INFRA = "FAIL_INFRA"
NOT_RUN = "NOT_RUN"
STATUSES = (
    COLLECTED,
    PASS_COMPILE_ONLY,
    PASS_NUMERIC,
    G_SMOKE,
    SKIP_ARCH,
    SKIP_DEP,
    SKIP_PERF,
    FAIL_LOWER,
    FAIL_DEVICE_COMPILE,
    FAIL_LAUNCH,
    FAIL_NUMERIC,
    FAIL_ENV,
    FAIL_INFRA,
    NOT_RUN,
)
PASS_STATUSES = frozenset((PASS_COMPILE_ONLY, PASS_NUMERIC))
SKIP_STATUSES = frozenset((SKIP_ARCH, SKIP_DEP, SKIP_PERF))
FAIL_STATUSES = frozenset(
    (FAIL_LOWER, FAIL_DEVICE_COMPILE, FAIL_LAUNCH, FAIL_NUMERIC, FAIL_ENV, FAIL_INFRA)
)
PYTEST_COUNT_NAMES = (
    "collected",
    "passed",
    "failed",
    "skipped",
    "deselected",
    "xfailed",
    "xpassed",
    "errors",
    "warnings",
)


class MatrixError(ValueError):
    """A manifest or command-line configuration error."""


@dataclass(frozen=True)
class Row:
    id: str
    group_id: str
    path: str
    node: str
    role: str
    tiers: tuple[str, ...]
    layer: str
    arch_lane: str
    dependencies: tuple[str, ...]
    optional_dependencies: tuple[str, ...]
    numeric_gate: bool
    z3_opt_in: Mapping[str, Any] | None = None
    notes: str | None = None

    @property
    def target(self) -> str:
        return f"{self.path}::{self.node}"

    @property
    def pass_configs(self) -> Mapping[str, Any] | None:
        if self.z3_opt_in is None:
            return None
        return self.z3_opt_in["pass_configs"]


@dataclass(frozen=True)
class FrozenManifest:
    data: Mapping[str, Any]
    rows: tuple[Row, ...]
    sha256: str
    path: Path

    def tier_rows(self, tier: str) -> tuple[Row, ...]:
        return tuple(row for row in self.rows if tier in row.tiers)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_relative_test_path(path: str) -> None:
    candidate = Path(path)
    if not path or candidate.is_absolute() or ".." in candidate.parts or candidate.suffix != ".py":
        raise MatrixError(f"unsafe or non-Python manifest target path: {path!r}")


def _expand_group(group: Mapping[str, Any], expected_role: str) -> list[Row]:
    if not isinstance(group, dict):
        raise MatrixError("every matrix group must be an object")
    group_id = group.get("id")
    role = group.get("role")
    path = group.get("path")
    nodes = group.get("nodes")
    if not isinstance(group_id, str) or not group_id:
        raise MatrixError("every group needs a non-empty id")
    if role != expected_role:
        raise MatrixError(f"group {group_id}: expected role {expected_role}, got {role!r}")
    if not isinstance(path, str):
        raise MatrixError(f"group {group_id}: path must be a string")
    _validate_relative_test_path(path)
    if not isinstance(nodes, list) or not nodes:
        raise MatrixError(f"group {group_id}: nodes must be a non-empty frozen list")
    if any(not isinstance(node, str) or not node or "::" in node for node in nodes):
        raise MatrixError(f"group {group_id}: invalid bare pytest node name")
    if len(set(nodes)) != len(nodes):
        raise MatrixError(f"group {group_id}: duplicate nodes")
    quick_nodes = group.get("quick_nodes", [])
    if not isinstance(quick_nodes, list) or any(node not in nodes for node in quick_nodes):
        raise MatrixError(f"group {group_id}: quick_nodes must be a subset of nodes")
    if len(set(quick_nodes)) != len(quick_nodes):
        raise MatrixError(f"group {group_id}: duplicate quick_nodes")
    layer = group.get("layer")
    if layer not in {"H0", "H1", "G", "P"}:
        raise MatrixError(f"group {group_id}: invalid layer {layer!r}")
    numeric_gate = group.get("numeric_gate")
    if not isinstance(numeric_gate, bool):
        raise MatrixError(f"group {group_id}: numeric_gate must be boolean")
    if numeric_gate != (layer == "P"):
        raise MatrixError(f"group {group_id}: only P rows may declare a numeric hard gate")
    dependencies = group.get("dependencies", [])
    optional_dependencies = group.get("optional_dependencies", [])
    if not isinstance(dependencies, list) or not all(isinstance(dep, str) for dep in dependencies):
        raise MatrixError(f"group {group_id}: dependencies must be a string list")
    if not isinstance(optional_dependencies, list) or not all(
        isinstance(dep, str) for dep in optional_dependencies
    ):
        raise MatrixError(f"group {group_id}: optional_dependencies must be a string list")
    if set(dependencies) & set(optional_dependencies):
        raise MatrixError(f"group {group_id}: dependency cannot be required and optional")
    z3_opt_in = group.get("z3_opt_in")
    if role == "z3_fixture":
        if not isinstance(z3_opt_in, dict) or not isinstance(z3_opt_in.get("pass_configs"), dict):
            raise MatrixError(f"group {group_id}: z3 fixture requires explicit pass_configs")
        configs = z3_opt_in["pass_configs"]
        if configs.get(Z3_PIPELINE_CONFIG) is not True:
            raise MatrixError(f"group {group_id}: z3 fixture must explicitly enable {Z3_PIPELINE_CONFIG}")
        if z3_opt_in.get("environment") != PASS_CONFIG_ENV:
            raise MatrixError(f"group {group_id}: unexpected pass-config transport")
        if quick_nodes:
            raise MatrixError(f"group {group_id}: opt-in fixture cannot be in the quick default tier")
    elif z3_opt_in is not None:
        raise MatrixError(f"group {group_id}: ordinary rows cannot masquerade as z3 fixtures")

    rows = []
    for node in nodes:
        tiers = ["full"]
        if role == "host":
            tiers.extend(("quick", "host"))
        elif role == "operator" and node in quick_nodes:
            tiers.append("quick")
        rows.append(
            Row(
                id=f"{path}::{node}",
                group_id=group_id,
                path=path,
                node=node,
                role=role,
                tiers=tuple(tiers),
                layer=layer,
                arch_lane=group.get("arch_lane"),
                dependencies=tuple(dependencies),
                optional_dependencies=tuple(optional_dependencies),
                numeric_gate=numeric_gate,
                z3_opt_in=z3_opt_in,
                notes=group.get("notes"),
            )
        )
    return rows


def expand_manifest(manifest: Mapping[str, Any]) -> tuple[Row, ...]:
    """Validate and deterministically expand the frozen groups into matrix rows."""

    if manifest.get("schema") != "tilelang.cuda.compile-matrix" or manifest.get("version") != 1:
        raise MatrixError("unsupported compile-matrix schema or version")
    rows: list[Row] = []
    group_ids: set[str] = set()
    for key, role in (
        ("operator_groups", "operator"),
        ("host_groups", "host"),
        ("z3_fixture_groups", "z3_fixture"),
    ):
        groups = manifest.get(key)
        if not isinstance(groups, list):
            raise MatrixError(f"{key} must be a list")
        for group in groups:
            expanded = _expand_group(group, role)
            if expanded[0].group_id in group_ids:
                raise MatrixError(f"duplicate group id: {expanded[0].group_id}")
            group_ids.add(expanded[0].group_id)
            rows.extend(expanded)
    row_ids = [row.id for row in rows]
    if len(row_ids) != len(set(row_ids)):
        duplicates = sorted(row_id for row_id, count in Counter(row_ids).items() if count > 1)
        raise MatrixError(f"pytest nodes appear in multiple roles/groups: {duplicates[:3]}")

    arch_lanes = manifest.get("arch_lanes", {})
    dependencies = manifest.get("dependencies", {})
    for row in rows:
        lane = arch_lanes.get(row.arch_lane)
        if not isinstance(lane, dict) or not isinstance(lane.get("requires_device"), bool):
            raise MatrixError(f"row {row.id}: invalid arch lane {row.arch_lane!r}")
        for dep in (*row.dependencies, *row.optional_dependencies):
            if dep not in dependencies:
                raise MatrixError(f"row {row.id}: undeclared dependency {dep!r}")
        if row.layer == "G" and (row.numeric_gate or row.role != "operator"):
            raise MatrixError(f"row {row.id}: G smoke cannot be a numeric gate")

    for tier in ("quick", "full", "host"):
        counts = Counter(row.role for row in rows if tier in row.tiers)
        actual = {
            "operator": counts["operator"],
            "host": counts["host"],
            "z3_fixture": counts["z3_fixture"],
            "total": sum(counts.values()),
        }
        expected = manifest.get("expected_rows", {}).get(tier)
        if expected != actual:
            raise MatrixError(f"{tier} row count mismatch: expected {expected}, expanded {actual}")
    return tuple(rows)


def load_manifest(path: Path | str = DEFAULT_MANIFEST) -> FrozenManifest:
    manifest_path = Path(path).expanduser().resolve()
    raw = manifest_path.read_bytes()
    data = json.loads(raw)
    return FrozenManifest(data=data, rows=expand_manifest(data), sha256=_sha256_bytes(raw), path=manifest_path)


def validate_target_files(rows: Iterable[Row], repo_root: Path) -> None:
    missing = sorted({row.path for row in rows if not (repo_root / row.path).is_file()})
    if missing:
        raise MatrixError(f"manifest target files do not exist: {missing[:5]}")


def select_rows(
    rows: Sequence[Row],
    *,
    action: str,
    mode: str,
    only: Sequence[str] = (),
) -> list[tuple[Row, bool, str | None]]:
    """Apply exact selection and mode gates while preserving the frozen denominator."""

    if action not in {"collect", "run"} or mode not in {"default", "z3-opt-in"}:
        raise MatrixError(f"invalid action/mode: {action}/{mode}")
    if any(any(character in value for character in "*?") for value in only):
        raise MatrixError("--only accepts exact row/group/file IDs, not wildcard patterns")
    requested = set(only)
    aliases = {
        "row": {row.id for row in rows},
        "group": {row.group_id for row in rows},
        "file": {row.path for row in rows},
    }
    known = set().union(*aliases.values())
    unknown = sorted(requested - known)
    if unknown:
        raise MatrixError(f"--only did not match an exact frozen row/group/file ID: {unknown[:5]}")

    selected_by_only = {
        row.id
        for row in rows
        if not requested or row.id in requested or row.group_id in requested or row.path in requested
    }
    if mode == "z3-opt-in" and requested:
        invalid = sorted(row.id for row in rows if row.id in selected_by_only and row.z3_opt_in is None)
        if invalid:
            raise MatrixError(
                "z3-opt-in requires manifest z3_opt_in.pass_configs on a dedicated fixture; "
                f"ordinary nodes are not eligible: {invalid[:3]}"
            )
    if action == "run" and mode == "default" and requested:
        invalid = sorted(row.id for row in rows if row.id in selected_by_only and row.z3_opt_in is not None)
        if invalid:
            raise MatrixError(f"dedicated z3 fixtures require --mode z3-opt-in: {invalid[:3]}")

    selection: list[tuple[Row, bool, str | None]] = []
    eligible_count = 0
    for row in rows:
        selected = row.id in selected_by_only
        reason = None if selected else "not selected by exact --only"
        if selected and mode == "z3-opt-in" and row.z3_opt_in is None:
            if requested:
                raise MatrixError(f"internal selection error for ordinary node {row.id}")
            selected = False
            reason = "NOT_RUN: z3-opt-in selects only dedicated manifest fixtures"
        if selected and action == "run" and mode == "default" and row.z3_opt_in is not None:
            if requested:
                raise MatrixError(f"internal default-mode fixture selection error for {row.id}")
            selected = False
            reason = "NOT_RUN: dedicated fixture is disabled in default run mode"
        if selected:
            eligible_count += 1
        selection.append((row, selected, reason))
    if eligible_count == 0:
        raise MatrixError(f"no rows are eligible for action={action}, mode={mode}, only={list(only)!r}")
    return selection


def build_command(
    row: Row,
    *,
    action: str,
    mode: str,
    python: str = sys.executable,
) -> list[str]:
    """Build one node's argv.  No shell, no -k, and no multi-node subprocess."""

    if action not in {"collect", "run"}:
        raise MatrixError(f"unknown pytest action: {action}")
    if mode == "z3-opt-in" and row.z3_opt_in is None:
        raise MatrixError(f"row {row.id} has no explicit z3_opt_in.pass_configs")
    if action == "run" and mode == "default" and row.z3_opt_in is not None:
        raise MatrixError(f"row {row.id} requires z3-opt-in mode")
    command = [
        python,
        "-m",
        "pytest",
        "-q",
        "-rA",
        "--tb=short",
        "-p",
        "no:cacheprovider",
        "-o",
        "addopts=",
    ]
    if action == "collect":
        # testing/conftest.py turns a successful --collect-only into exit 5 in
        # pytest_terminal_summary.  Collection needs no fixtures or runtime
        # hooks, so bypass conftest while preserving the exact frozen target.
        command.extend(("--noconftest", "--collect-only"))
    command.append(row.target)
    return command


def build_cache_identity(
    row: Row,
    *,
    manifest_sha256: str,
    source_sha256: str,
    mode: str,
    python: str = sys.executable,
) -> str:
    material = {
        "runner_version": RUNNER_VERSION,
        "manifest_sha256": manifest_sha256,
        "source_sha256": source_sha256,
        "row": row.id,
        "layer": row.layer,
        "arch_lane": row.arch_lane,
        "numeric_gate": row.numeric_gate,
        "mode": mode,
        "pass_configs": row.pass_configs if mode == "z3-opt-in" else None,
        "python": str(Path(python).resolve()),
        "tilelang_target": os.environ.get("TILELANG_TARGET", ""),
    }
    return _sha256_bytes(_canonical_json(material).encode("utf-8"))


def prepare_output(output: Path | str, repo_root: Path) -> Path:
    path = Path(output).expanduser().resolve()
    guarded = {Path(path.anchor).resolve(), Path.home().resolve(), repo_root.resolve()}
    if path in guarded:
        raise MatrixError(f"refusing to use a repository/home/filesystem root as output: {path}")
    cache_root = (path / "cache").resolve()
    inherited_cache = os.environ.get("TILELANG_CACHE_DIR")
    if inherited_cache and cache_root == Path(inherited_cache).expanduser().resolve():
        raise MatrixError("output cache would alias the inherited shared TILELANG_CACHE_DIR")
    path.mkdir(parents=True, exist_ok=True)
    (path / "logs").mkdir(exist_ok=True)
    cache_root.mkdir(exist_ok=True)
    if not cache_root.is_relative_to(path):
        raise MatrixError("cache root escaped the output directory")
    return path


def isolated_cache_directory(output: Path, cache_identity: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{64}", cache_identity):
        raise MatrixError("invalid cache identity")
    cache_root = (output / "cache").resolve()
    cache_dir = (cache_root / cache_identity).resolve()
    if cache_dir.parent != cache_root or not cache_dir.is_relative_to(output.resolve()):
        raise MatrixError("per-node cache escaped its isolated root")
    cache_dir.mkdir(exist_ok=True)
    for child in ("torch_extension", "triton", "cuda", "xdg", "tmp"):
        (cache_dir / child).mkdir(exist_ok=True)
    return cache_dir


def build_environment(
    row: Row,
    *,
    mode: str,
    cache_dir: Path,
    repo_root: Path,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Create an isolated child environment without a pretend global Z3 switch."""

    env = dict(os.environ if base_env is None else base_env)
    cache_dir = cache_dir.resolve()
    env.update(
        {
            "TILELANG_CACHE_DIR": str(cache_dir),
            "TORCH_EXTENSIONS_DIR": str(cache_dir / "torch_extension"),
            "TRITON_CACHE_DIR": str(cache_dir / "triton"),
            "CUDA_CACHE_PATH": str(cache_dir / "cuda"),
            "XDG_CACHE_HOME": str(cache_dir / "xdg"),
            "TMPDIR": str(cache_dir / "tmp"),
            "TEMP": str(cache_dir / "tmp"),
            "TMP": str(cache_dir / "tmp"),
            "PYTHONHASHSEED": "0",
            "PYTHONUNBUFFERED": "1",
            "MPLBACKEND": "Agg",
            MODE_ENV: mode,
            ROW_ENV: row.id,
        }
    )
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(repo_root) + (os.pathsep + old_pythonpath if old_pythonpath else "")
    env.pop("PYTEST_ADDOPTS", None)
    env.pop(PASS_CONFIG_ENV, None)
    if mode == "z3-opt-in":
        if row.z3_opt_in is None:
            raise MatrixError(f"refusing z3-opt-in for ordinary pytest node {row.id}")
        env[PASS_CONFIG_ENV] = _canonical_json(row.pass_configs)
    for key in (
        "TILELANG_CACHE_DIR",
        "TORCH_EXTENSIONS_DIR",
        "TRITON_CACHE_DIR",
        "CUDA_CACHE_PATH",
        "XDG_CACHE_HOME",
        "TMPDIR",
    ):
        if not Path(env[key]).resolve().is_relative_to(cache_dir):
            raise MatrixError(f"cache variable {key} escaped {cache_dir}")
    return env


def parse_pytest_summary(text: str) -> dict[str, int]:
    counts = {name: 0 for name in PYTEST_COUNT_NAMES}
    collected_patterns = (
        r"\bcollected\s+(\d+)\s+items?\b",
        r"\b(\d+)\s+tests?\s+collected\b",
        r"\b(\d+)\s+items?\s+collected\b",
    )
    for pattern in collected_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            counts["collected"] = max(int(value) for value in matches)
            break
    for name in PYTEST_COUNT_NAMES[1:]:
        singular = name[:-1] if name.endswith("s") else name
        matches = re.findall(rf"\b(\d+)\s+{name}\b|\b(\d+)\s+{singular}\b", text, flags=re.IGNORECASE)
        counts[name] = max((int(left or right) for left, right in matches), default=0)
    return counts


def _matches_any(text: str, patterns: Sequence[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE) for pattern in patterns)


def classify_failure(text: str, *, numeric_gate: bool, default: str | None = None) -> str:
    """Classify diagnostics in stable priority order; never infer a pass."""

    env_patterns = (
        r"ModuleNotFoundError",
        r"No module named",
        r"ImportError:",
        r"cannot open shared object file",
        r"libcuda\.so.*(not found|cannot)",
        r"CUDA driver.*(not found|unavailable)",
        r"(nvcc|ptxas)(:|\s)+.*(not found|no such file)",
        r"Could not find.*(CUDA|nvcc|ptxas)",
    )
    launch_patterns = (
        r"CUDA_ERROR_[A-Z_]+",
        r"cudaError[A-Za-z]+",
        r"illegal memory access",
        r"device-side assert",
        r"misaligned address",
        r"invalid device function",
        r"invalid launch (configuration|argument)",
        r"launch failure",
        r"an illegal instruction was encountered",
        r"Failed to set the allowed dynamic shared memory size",
    )
    device_compile_patterns = (
        r"ptxas\s+(fatal|error)",
        r"nvcc\s+(fatal|error)",
        r"NVRTC_ERROR",
        r"NVRTC.*(failed|error)",
        r"NVCCCompilationError",
        r"device_kernel\.cu.*error",
        r"compilation terminated",
        r"failed to compile.*(CUDA|device|kernel)",
        r"CUDA compilation.*(failed|error)",
        r"RuntimeError: #include <tl_templates/cuda/nvrtc_std\.h>",
        r"Compilation error:",
    )
    numeric_patterns = (
        r"assert_(all)?close",
        r"assert_allclose",
        r"torch_assert_close",
        r"Tensor-likes are not close",
        r"(values|tensors?).*mismatch",
        r"Mismatch:.*%",
        r"nan.*(mismatch|expected|actual)",
        r"numeric(al)? (mismatch|failure)",
    )
    lower_patterns = (
        r"tilelang[\\/](engine|cuda|transform)[\\/].*lower",
        r"tilelang\.engine\.lower",
        r"Z3PipelinePlanning",
        z3_pipeline_config_pattern(),
        r"LowerTileOp",
        r"LayoutInference",
        r"TVMError|tvm\.error\.InternalError|InternalError:",
        r"src[\\/](transform|cuda)[\\/].*\.cc",
        r"codegen.*(failed|error)",
    )
    if _matches_any(text, env_patterns):
        return FAIL_ENV
    if _matches_any(text, launch_patterns):
        return FAIL_LAUNCH
    if _matches_any(text, device_compile_patterns):
        return FAIL_DEVICE_COMPILE
    if _matches_any(text, numeric_patterns):
        return FAIL_NUMERIC
    if _matches_any(text, lower_patterns):
        return FAIL_LOWER
    if "AssertionError" in text:
        return FAIL_NUMERIC if numeric_gate else FAIL_LOWER
    if default is not None:
        return default
    return FAIL_NUMERIC if numeric_gate else FAIL_LOWER


def z3_pipeline_config_pattern() -> str:
    return re.escape(Z3_PIPELINE_CONFIG)


def classify_skip(text: str) -> tuple[str, str]:
    perf_patterns = (
        r"performance test skipped by default",
        r"pass --run-perf",
        r"Skipped \d+ perf test",
    )
    if _matches_any(text, perf_patterns):
        return SKIP_PERF, "pytest reported an opt-in performance-test skip; not counted as passed"
    dependency_patterns = (
        r"requires package",
        r"importorskip",
        r"No module named",
        r"ModuleNotFoundError",
        r"(flash[-_]attn|flash_attn|\bfla\b|triton|einops).*(not installed|unavailable|missing)",
        r"(not installed|unavailable|missing).*(flash[-_]attn|flash_attn|\bfla\b|triton|einops)",
    )
    arch_patterns = (
        r"Requires CUDA",
        r"CUDA.*(not available|unavailable|required)",
        r"compute (capability|version|ge|eq|le|gt|lt)",
        r"\bSM[_ ]?\d+.*(required|required only|unsupported)",
        r"(Hopper|Blackwell|Ampere|Turing|Volta).*required",
        r"GPU.*(not available|required)",
        r"Run-time support for .*not present",
        r"architecture.*(unsupported|required)",
    )
    has_dep = _matches_any(text, dependency_patterns)
    has_arch = _matches_any(text, arch_patterns)
    if has_dep and not has_arch:
        return SKIP_DEP, "pytest reported a dependency skip; not counted as passed"
    if has_arch:
        return SKIP_ARCH, "pytest reported an architecture/device skip; not counted as passed"
    if has_dep:
        return SKIP_DEP, "pytest reported a dependency skip; not counted as passed"
    return NOT_RUN, "pytest skip/xfail was not recognized as arch/dependency/perf; not counted as passed"


def classify_result(
    row: Row,
    *,
    action: str,
    exit_code: int | None,
    text: str,
    counts: Mapping[str, int],
    timed_out: bool = False,
    spawn_error: str | None = None,
) -> tuple[str, str]:
    if timed_out:
        return FAIL_INFRA, "isolated subprocess exceeded --timeout and its process group was killed"
    if spawn_error is not None:
        return FAIL_INFRA, f"could not start isolated subprocess: {spawn_error}"
    if exit_code is None:
        return FAIL_INFRA, "subprocess has no exit code"
    if action == "collect":
        if exit_code == 0 and counts["collected"] > 0:
            return COLLECTED, "pytest collection only; this is not a compile or numeric pass"
        if exit_code in {2, 3, 4, 5}:
            return FAIL_INFRA, f"pytest collection failed with infrastructure exit {exit_code}"
        return classify_failure(text, numeric_gate=False, default=FAIL_INFRA), "collection did not produce nodes"
    if exit_code in {2, 3, 4, 5}:
        has_explicit_skip = bool(counts["skipped"]) or re.search(r"(?m)^SKIPPED(?:\s|\[)", text) is not None
        if exit_code == 5 and has_explicit_skip:
            status, reason = classify_skip(text)
            if status in SKIP_STATUSES:
                return status, reason + "; pytest exit 5 followed an all-skipped selection"
        return FAIL_INFRA, f"pytest infrastructure/usage exit {exit_code}; no tests are treated as passed"
    if exit_code != 0:
        status = classify_failure(text, numeric_gate=row.numeric_gate)
        return status, f"pytest exited {exit_code}; classified from captured diagnostics"
    if counts["failed"] or counts["errors"]:
        status = classify_failure(text, numeric_gate=row.numeric_gate)
        return status, "pytest summary contains failures/errors despite a zero exit"
    executed = counts["passed"] + counts["xpassed"]
    incomplete = counts["skipped"] + counts["deselected"] + counts["xfailed"]
    if incomplete:
        status, reason = classify_skip(text)
        return status, reason + f"; passed_cases={counts['passed']}, incomplete_cases={incomplete}"
    if executed == 0:
        return FAIL_INFRA, "zero exit without an executed test; no-tests/empty selection is not a pass"
    if row.layer == "G":
        return G_SMOKE, "GDN smoke passed; no numerical correctness claim"
    if row.numeric_gate:
        return PASS_NUMERIC, "all collected cases of the numerical hard gate passed"
    return PASS_COMPILE_ONLY, "compile/structural test passed without a numerical hard gate"


def aggregate_results(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    status_counts = {status: 0 for status in STATUSES}
    pytest_totals = {name: 0 for name in PYTEST_COUNT_NAMES}
    for record in records:
        status = record["status"]
        if status not in status_counts:
            raise MatrixError(f"unknown result status {status!r}")
        status_counts[status] += 1
        for name in PYTEST_COUNT_NAMES:
            pytest_totals[name] += int(record.get("pytest", {}).get(name, 0))
    denominator = len(records)
    if sum(status_counts.values()) != denominator:
        raise AssertionError("status accounting lost a frozen row")
    passed = status_counts[PASS_COMPILE_ONLY] + status_counts[PASS_NUMERIC]
    skipped = sum(status_counts[status] for status in SKIP_STATUSES)
    failed = sum(status_counts[status] for status in FAIL_STATUSES)
    numeric_eligible = sum(bool(record.get("selected")) and bool(record.get("numeric_gate")) for record in records)
    numeric_passed = sum(
        bool(record.get("selected"))
        and bool(record.get("numeric_gate"))
        and record["status"] == PASS_NUMERIC
        for record in records
    )
    return {
        "schema": SUMMARY_SCHEMA,
        "denominator": denominator,
        "selected": sum(bool(record.get("selected")) for record in records),
        "attempted": sum(bool(record.get("attempted")) for record in records),
        "not_selected": sum(not bool(record.get("selected")) for record in records),
        "selected_not_run": sum(
            bool(record.get("selected")) and record["status"] == NOT_RUN for record in records
        ),
        "status_counts": status_counts,
        "passed": passed,
        "compile_only_passed": status_counts[PASS_COMPILE_ONLY],
        "numeric_passed": status_counts[PASS_NUMERIC],
        "g_smoke_not_numeric": status_counts[G_SMOKE],
        "successful_non_numeric": status_counts[PASS_COMPILE_ONLY] + status_counts[G_SMOKE],
        "collected_not_passed": status_counts[COLLECTED],
        "skipped_not_passed": skipped,
        "not_run_not_passed": status_counts[NOT_RUN],
        "failed": failed,
        "numeric_gate": {
            "selected_eligible": numeric_eligible,
            "passed": numeric_passed,
            "not_satisfied": numeric_eligible - numeric_passed,
        },
        "pytest_case_totals": pytest_totals,
        "accounting": "denominator=frozen manifest rows; COLLECTED/SKIP_*/NOT_RUN/G_SMOKE are not PASS_NUMERIC",
    }


def _version_at_least(version: str, major: int, minor: int) -> bool | None:
    match = re.match(r"\s*(\d+)\.(\d+)", version)
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2))) >= (major, minor)


class DependencyChecker:
    """Read-only, cached module/distribution probes.  No imports or installs."""

    def __init__(self, manifest: Mapping[str, Any], repo_root: Path):
        self.specs = manifest["dependencies"]
        self.repo_root = repo_root
        self._cache: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def _probe(self, name: str) -> dict[str, Any]:
        spec = self.specs[name]
        result: dict[str, Any] = {"name": name, "required": True, "available": False}
        if "in_tree" in spec:
            path = self.repo_root / spec["in_tree"]
            result["in_tree"] = spec["in_tree"]
            result["available"] = path.is_file()
        else:
            module = spec["module"]
            result["module"] = module
            try:
                result["available"] = importlib.util.find_spec(module) is not None
            except (ImportError, AttributeError, ValueError) as error:
                result["probe_error"] = str(error)
        distribution = spec.get("distribution")
        if distribution:
            result["distribution"] = distribution
            try:
                result["distribution_version"] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                result["distribution_version"] = None
        if spec.get("torch_requirement") == "<2.14":
            try:
                torch_version = importlib.metadata.version("torch")
            except importlib.metadata.PackageNotFoundError:
                torch_version = ""
            conflict = _version_at_least(torch_version, 2, 14)
            result["torch_version"] = torch_version or None
            result["constraint_conflict"] = bool(conflict)
            if conflict:
                result["constraint_note"] = (
                    "requirements-test-cuda.txt combines flash-attn==2.5.8 with torch<2.14; "
                    "the runner will not install flash-attn or downgrade torch"
                )
        return result

    def _cached_probe(self, name: str) -> dict[str, Any]:
        with self._lock:
            if name not in self._cache:
                self._cache[name] = self._probe(name)
            return dict(self._cache[name])

    def check(self, row: Row) -> tuple[list[str], list[dict[str, Any]]]:
        missing: list[str] = []
        diagnostics = []
        for required, names in ((True, row.dependencies), (False, row.optional_dependencies)):
            for name in names:
                diagnostic = self._cached_probe(name)
                diagnostic["required"] = required
                diagnostics.append(diagnostic)
                if required and not diagnostic["available"]:
                    reason = f"missing required dependency {name}"
                    if diagnostic.get("constraint_conflict"):
                        reason += "; " + diagnostic["constraint_note"]
                    missing.append(reason)
        return missing, diagnostics


def parse_arch(value: str | None) -> int | None:
    if not value or value.strip().lower() in {"none", "unavailable", "no-gpu"}:
        return None
    normalized = value.strip().lower().replace("sm_", "").replace("sm", "")
    if "." in normalized:
        major, minor = normalized.split(".", 1)
        return int(major) * 10 + int(minor)
    if normalized.isdigit():
        number = int(normalized)
        return number if number >= 70 else number * 10
    raise MatrixError(f"cannot parse {ARCH_ENV}={value!r}; use sm_120, 120, or 12.0")


def detect_cuda_arch(python: str = sys.executable) -> int | None:
    """Probe capability without launching a kernel; ARCH_ENV can pin CI explicitly."""

    if ARCH_ENV in os.environ:
        return parse_arch(os.environ[ARCH_ENV])
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run(
                [nvidia_smi, "--query-gpu=compute_cap", "--format=csv,noheader"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                return parse_arch(result.stdout.splitlines()[0])
        except (OSError, subprocess.TimeoutExpired, MatrixError, ValueError):
            pass
    try:
        result = subprocess.run(
            [
                python,
                "-c",
                "import torch; print(torch.cuda.get_device_capability(0) if torch.cuda.is_available() else '')",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        if result.returncode == 0 and result.stdout.strip():
            major, minor = result.stdout.strip().splitlines()[-1].strip("() ").split(",")
            return int(major) * 10 + int(minor.strip())
    except (OSError, subprocess.TimeoutExpired, ValueError):
        pass
    return None


def arch_skip_reason(row: Row, lane: Mapping[str, Any], current_arch: int | None) -> str | None:
    if not lane["requires_device"]:
        return None
    if current_arch is None:
        return f"{row.arch_lane}: CUDA device/capability unavailable; set {ARCH_ENV} explicitly for a remote lane"
    exact = lane.get("exact")
    minimum = lane.get("min")
    maximum = lane.get("max")
    if exact is not None and current_arch != exact:
        return f"{row.arch_lane}: requires sm_{exact}, detected sm_{current_arch}"
    if minimum is not None and current_arch < minimum:
        return f"{row.arch_lane}: requires sm_{minimum}+ , detected sm_{current_arch}"
    if maximum is not None and current_arch > maximum:
        return f"{row.arch_lane}: requires sm_{maximum} or older, detected sm_{current_arch}"
    return None


def _terminate_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass


def execute_subprocess(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout: float,
    stdout_path: Path,
    stderr_path: Path,
) -> dict[str, Any]:
    start = time.monotonic()
    timed_out = False
    spawn_error = None
    exit_code = None
    with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
        try:
            popen_kwargs: dict[str, Any] = {}
            if os.name == "posix":
                popen_kwargs["start_new_session"] = True
            process = subprocess.Popen(
                list(command),
                cwd=str(cwd),
                env=dict(env),
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                shell=False,
                **popen_kwargs,
            )
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                _terminate_process_group(process)
                process.wait()
            exit_code = 124 if timed_out else process.returncode
        except OSError as error:
            spawn_error = str(error)
            stderr_handle.write((spawn_error + "\n").encode("utf-8", errors="replace"))
    return {
        "exit_code": exit_code,
        "duration_seconds": time.monotonic() - start,
        "timed_out": timed_out,
        "spawn_error": spawn_error,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_log(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _relative_output(path: Path, output: Path) -> str:
    return path.resolve().relative_to(output.resolve()).as_posix()


def execute_matrix(
    matrix: FrozenManifest,
    *,
    repo_root: Path,
    action: str,
    tier: str,
    mode: str,
    only: Sequence[str],
    timeout: float,
    output: Path | str,
    collect_workers: int = 1,
    python: str = sys.executable,
) -> dict[str, Any]:
    if tier not in {"quick", "full", "host"}:
        raise MatrixError(f"unknown tier {tier}")
    if timeout <= 0:
        raise MatrixError("--timeout must be positive")
    if collect_workers not in {1, 2, 3, 4}:
        raise MatrixError("--collect-workers must be in [1, 4]")
    if action == "run" and collect_workers != 1:
        raise MatrixError("GPU/run execution is always serial; --collect-workers is collect-only")
    if collect_workers > 1 and tier != "host":
        raise MatrixError("parallel collection is restricted to the CPU-only host tier")
    rows = matrix.tier_rows(tier)
    validate_target_files(rows, repo_root)
    selection = select_rows(rows, action=action, mode=mode, only=only)
    output_path = prepare_output(output, repo_root)
    source_hashes = {row.path: _sha256_file(repo_root / row.path) for row in rows}
    dependency_checker = DependencyChecker(matrix.data, repo_root)
    needs_arch = action == "run" and any(
        selected and matrix.data["arch_lanes"][row.arch_lane]["requires_device"]
        for row, selected, _ in selection
    )
    current_arch = detect_cuda_arch(python) if needs_arch else None
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    metadata = {
        "schema": "tilelang.cuda.compile-matrix-run",
        "run_id": run_id,
        "started_at": _utc_now(),
        "action": action,
        "tier": tier,
        "mode": mode,
        "only": list(only),
        "timeout_seconds": timeout,
        "collect_workers": collect_workers,
        "python": python,
        "repo_root": str(repo_root),
        "manifest": str(matrix.path),
        "manifest_sha256": matrix.sha256,
        "source_sha256": source_hashes,
        "arch_probe": {"required": needs_arch, "current": current_arch, "override_env": ARCH_ENV},
        "frozen_node_ids": [row.id for row in rows],
        "dependency_install": False,
        "shared_cache_clear": False,
        "gpu_run_parallelism": 1,
    }

    def process(item: tuple[Row, bool, str | None]) -> dict[str, Any]:
        row, selected, selection_reason = item
        cache_identity = build_cache_identity(
            row,
            manifest_sha256=matrix.sha256,
            source_sha256=source_hashes[row.path],
            mode=mode,
            python=python,
        )
        stdout_path = output_path / "logs" / f"{cache_identity}.stdout.log"
        stderr_path = output_path / "logs" / f"{cache_identity}.stderr.log"
        record: dict[str, Any] = {
            "schema": RESULT_SCHEMA,
            "run_id": run_id,
            "id": row.id,
            "group_id": row.group_id,
            "target": row.target,
            "role": row.role,
            "tier": tier,
            "row_tiers": list(row.tiers),
            "layer": row.layer,
            "arch_lane": row.arch_lane,
            "required_dependencies": list(row.dependencies),
            "optional_dependencies": list(row.optional_dependencies),
            "numeric_gate": row.numeric_gate,
            "action": action,
            "mode": mode,
            "selected": selected,
            "attempted": False,
            "status": NOT_RUN,
            "reason": selection_reason or NOT_RUN,
            "exit_code": None,
            "duration_seconds": 0.0,
            "cache_identity": cache_identity,
            "cache_dir": f"cache/{cache_identity}",
            "source_sha256": source_hashes[row.path],
            "manifest_sha256": matrix.sha256,
            "pass_configs": row.pass_configs if mode == "z3-opt-in" and selected else None,
            "command": None,
            "stdout": _relative_output(stdout_path, output_path),
            "stderr": _relative_output(stderr_path, output_path),
            "pytest": {name: 0 for name in PYTEST_COUNT_NAMES},
            "dependencies": [],
            "started_at": None,
            "finished_at": _utc_now(),
        }
        if not selected:
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text((selection_reason or NOT_RUN) + "\n", encoding="utf-8")
            return record

        started = time.monotonic()
        record["started_at"] = _utc_now()
        command = build_command(row, action=action, mode=mode, python=python)
        record["command"] = command
        missing, dependency_diagnostics = dependency_checker.check(row)
        record["dependencies"] = dependency_diagnostics
        if missing:
            record["status"] = SKIP_DEP
            record["reason"] = "; ".join(missing)
        else:
            reason = (
                arch_skip_reason(row, matrix.data["arch_lanes"][row.arch_lane], current_arch)
                if action == "run"
                else None
            )
            if reason is not None:
                record["status"] = SKIP_ARCH
                record["reason"] = reason
            else:
                cache_dir = isolated_cache_directory(output_path, cache_identity)
                env = build_environment(row, mode=mode, cache_dir=cache_dir, repo_root=repo_root)
                execution = execute_subprocess(
                    command,
                    cwd=repo_root,
                    env=env,
                    timeout=timeout,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                )
                record.update(execution)
                record["attempted"] = True
                text = _read_log(stdout_path) + "\n" + _read_log(stderr_path)
                counts = parse_pytest_summary(text)
                record["pytest"] = counts
                record["status"], record["reason"] = classify_result(
                    row,
                    action=action,
                    exit_code=execution["exit_code"],
                    text=text,
                    counts=counts,
                    timed_out=execution["timed_out"],
                    spawn_error=execution["spawn_error"],
                )
        if not record["attempted"]:
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text(str(record["reason"]) + "\n", encoding="utf-8")
        record["duration_seconds"] = (
            execution["duration_seconds"] if record["attempted"] else time.monotonic() - started
        )
        record["finished_at"] = _utc_now()
        return record

    if collect_workers > 1:
        with ThreadPoolExecutor(max_workers=collect_workers) as executor:
            records = list(executor.map(process, selection))
    else:
        records = [process(item) for item in selection]
    summary = aggregate_results(records)
    summary.update(
        {
            "run_id": run_id,
            "action": action,
            "tier": tier,
            "mode": mode,
            "manifest_sha256": matrix.sha256,
            "arch_detected": current_arch,
            "results": "results.jsonl",
        }
    )
    with (output_path / "results.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(_canonical_json(record) + "\n")
    metadata["finished_at"] = _utc_now()
    metadata["summary"] = summary
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_path / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {"records": records, "summary": summary, "output": output_path, "metadata": metadata}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--collect", action="store_true", help="pytest --collect-only; never a pass")
    action.add_argument("--run", action="store_true", help="run the same frozen manifest nodes")
    parser.add_argument("--tier", choices=("quick", "full", "host"), default="quick")
    parser.add_argument("--mode", choices=("default", "z3-opt-in"), default="default")
    parser.add_argument("--only", action="append", default=[], help="exact row/group/file ID; repeatable")
    parser.add_argument("--timeout", type=float, default=300.0, help="per-node subprocess timeout in seconds")
    parser.add_argument("--output", required=True, help="result directory (logs, JSONL, summary, isolated caches)")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help=argparse.SUPPRESS)
    parser.add_argument(
        "--collect-workers",
        type=int,
        default=1,
        help="1-4 workers for host-tier collection only; run is always serial",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        matrix = load_manifest(args.manifest)
        result = execute_matrix(
            matrix,
            repo_root=REPO_ROOT,
            action="collect" if args.collect else "run",
            tier=args.tier,
            mode=args.mode,
            only=args.only,
            timeout=args.timeout,
            output=args.output,
            collect_workers=args.collect_workers,
            python=sys.executable,
        )
    except (MatrixError, json.JSONDecodeError, OSError) as error:
        print(f"compile-matrix configuration/error: {error}", file=sys.stderr)
        return 2
    print(_canonical_json(result["summary"]))
    return 1 if result["summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
