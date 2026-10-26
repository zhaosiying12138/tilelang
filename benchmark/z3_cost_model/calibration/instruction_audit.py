#!/usr/bin/env python3
"""CPU-only PTX/SASS instruction audits for calibration artifacts.

The worker uses the same rules as this standalone re-audit entry point.  A
source-level T.gemm/T.fma marker is not accepted as instruction evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Sequence

AUDITED_BENCHMARKS = ("bf16_mma_gemm", "f16_mma_gemm", "tf32_mma_gemm", "cuda_fp32_fma")
ARTIFACT_KINDS = ("ptx", "sass")


class InstructionAuditError(ValueError):
    pass


def _mma_ptx(dtype: str, k: str) -> str:
    return rf"mma\.sync\.aligned\.m16n8k{k}\.row\.col\.f32\.{dtype}\.{dtype}\.f32"


def _mma_sass(dtype: str) -> str:
    if dtype == "F16":
        return r"\bHMMA\.16816\.F32(?=\s)"
    return rf"\bHMMA\.[A-Z0-9_.]*{dtype}[A-Z0-9_.]*"


def _rules() -> dict[str, dict[str, Any]]:
    tensor_forbidden = {
        "ptx": (r"\bwgmma\.", r"\btcgen05\."),
        "sass": (r"\bHGMMA\b", r"\bUTCMMA\b", r"\bWGMMA\b"),
    }
    rules = {}
    for name, dtype, k in (
        ("bf16_mma_gemm", "bf16", "16"),
        ("f16_mma_gemm", "f16", "16"),
        ("tf32_mma_gemm", "tf32", "(?:4|8)"),
    ):
        rules[name] = {
            "ptx": {
                "required": (((_mma_ptx(dtype, k), 1),),),
                "forbidden": tensor_forbidden["ptx"],
            },
            "sass": {
                "required": (((_mma_sass(dtype.upper()), 1),),),
                "forbidden": tensor_forbidden["sass"],
            },
        }
    rules["cuda_fp32_fma"] = {
        "ptx": {
            # Accept four scalar FMAs or two packed f32x2 instructions. The
            # third tuple element is the scalar-FMA-equivalent weight.
            "required": ((
                (r"fma\.rn(?:\.ftz)?\.f32(?!x2)", 4, 1),
                (r"fma\.rn(?:\.ftz)?\.f32x2", 2, 2),
            ),),
            "forbidden": (r"\bmma\.sync\.", r"\bwgmma\.", r"\btcgen05\."),
        },
        "sass": {
            "required": (((r"\bFFMA(?:\.[A-Z0-9_]+)?\b", 1),),),
            "forbidden": (r"\bHMMA\b", r"\bOMMA\b", r"\bQMMA\b", r"\bHGMMA\b", r"\bUTCMMA\b"),
        },
    }
    return rules


RULES = _rules()


def audit_instruction_text(
    benchmark: str,
    text: str,
    artifact_kind: str,
    *,
    artifact_path: str | None = None,
    export_method: str | None = None,
) -> dict[str, Any]:
    """Return a JSON-safe audit; never infer an instruction from a function name."""

    if benchmark not in AUDITED_BENCHMARKS:
        raise InstructionAuditError(f"No instruction rule for {benchmark!r}")
    if artifact_kind not in ARTIFACT_KINDS:
        raise InstructionAuditError(f"artifact_kind must be one of {ARTIFACT_KINDS}")
    if not isinstance(text, str) or not text.strip():
        raise InstructionAuditError("Instruction artifact must be non-empty text")
    rule = RULES[benchmark][artifact_kind]
    missing = []
    matched = []
    match_counts = {}
    weighted_counts = {}
    for group in rule["required"]:
        group_matches = []
        group_missing = []
        for requirement in group:
            pattern, minimum = requirement[:2]
            weight = requirement[2] if len(requirement) == 3 else 1
            count = len(re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE))
            weighted = count * weight
            match_counts[pattern] = count
            weighted_counts[pattern] = weighted
            if weighted >= minimum:
                group_matches.append({"pattern": pattern, "count": count, "scalar_equivalent": weighted, "minimum": minimum})
            else:
                group_missing.append({"pattern": pattern, "minimum": minimum, "scalar_equivalent": weighted})
        if group_matches:
            matched.append(group_matches)
        else:
            missing.append(group_missing)
    forbidden = []
    for pattern in rule["forbidden"]:
        found = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if found:
            forbidden.append({"pattern": pattern, "count": len(found), "examples": sorted(set(found))[:4]})
    encoded = text.encode("utf-8")
    return {
        "benchmark": benchmark,
        "artifact_kind": artifact_kind,
        "artifact_path": artifact_path,
        "export_method": export_method,
        "artifact_sha256": hashlib.sha256(encoded).hexdigest(),
        "artifact_bytes": len(encoded),
        "passed": not missing and not forbidden,
        "matched_required": matched,
        "missing_required": missing,
        "forbidden_matches": forbidden,
        "match_counts": match_counts,
        "scalar_equivalent_counts": weighted_counts,
        "evidence_scope": "compiled instruction artifact, not a TileLang function-name/source-marker assertion",
    }


def require_instruction_audit(*args: Any, **kwargs: Any) -> dict[str, Any]:
    report = audit_instruction_text(*args, **kwargs)
    if not report["passed"]:
        raise InstructionAuditError("Compiled instruction audit failed: " + json.dumps({
            "benchmark": report["benchmark"],
            "missing_required": report["missing_required"],
            "forbidden_matches": report["forbidden_matches"],
        }, sort_keys=True))
    return report


def audit_file(benchmark: str, path: str | Path, artifact_kind: str) -> dict[str, Any]:
    artifact = Path(path)
    text = artifact.read_text(encoding="utf-8")
    return audit_instruction_text(benchmark, text, artifact_kind, artifact_path=str(artifact))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", help="Exported .ptx or .sass text file")
    parser.add_argument("--benchmark", required=True, choices=AUDITED_BENCHMARKS)
    parser.add_argument("--kind", required=True, choices=ARTIFACT_KINDS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = audit_file(args.benchmark, args.artifact, args.kind)
    except (OSError, UnicodeError, InstructionAuditError) as error:
        print(json.dumps({"passed": False, "error": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
