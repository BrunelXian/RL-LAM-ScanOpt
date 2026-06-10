from __future__ import annotations

import csv
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCE_ROOTS = [Path(r"E:\Projects\RL-LAM-ScanOpt"), Path(r"D:\Projects\RL-LAM-ScanOpt")]
TARGET_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
RUN_ID = "run_04_true_variable_n_cae_template_parameterisation_audit"
OUTPUT_DIR = TARGET_ROOT / "outputs" / "stage3_run_04_true_variable_n_cae_template_parameterisation_audit"
REPORT_DIR = TARGET_ROOT / "docs" / "stage3" / "runs" / "run_04_true_variable_n_cae_template_parameterisation_audit"
REPORT_PATH = REPORT_DIR / "RUN_04_TRUE_VARIABLE_N_CAE_TEMPLATE_PARAMETERISATION_AUDIT_REPORT.md"
MANIFEST_PATH = TARGET_ROOT / "artifacts" / "manifests" / "stage3_run_04_manifest.json"

ALLOWED_EXTENSIONS = {".py", ".jnl", ".inp", ".md", ".txt", ".json", ".csv"}
FORBIDDEN_EXTENSIONS = {
    ".odb",
    ".cae",
    ".sim",
    ".dat",
    ".msg",
    ".sta",
    ".lck",
    ".prt",
    ".com",
    ".res",
    ".abq",
    ".pac",
    ".sel",
    ".ipm",
}
MAX_FILE_BYTES = 4 * 1024 * 1024
SCAN_KEYWORDS = [
    "cae",
    "abaqus",
    "inp",
    "jnl",
    "heat",
    "load",
    "track",
    "scan",
    "32track",
    "full",
    "model",
    "generation",
    "generate",
    "builder",
    "lded",
    "thermal",
    "stress",
]
SKIP_DIRS = {".git", "__pycache__", "outputs", "models", "figures", "post-odb"}
SELF_AUDIT_TOKENS = {
    "run_04_true_variable_n_cae_template_parameterisation_audit",
    "stage3_run_04_true_variable_n_cae_template_parameterisation_audit",
}

HARD_CODE_PATTERNS = [
    ("track_count", re.compile(r"\bTRACK_COUNT\s*=\s*32\b|\btrack_count\s*[=:]\s*32\b", re.IGNORECASE), "track_count", "high"),
    ("range32", re.compile(r"\brange\s*\(\s*32\s*\)|\b0\s*\.\.\s*31\b|list\s*\(\s*range\s*\(\s*32\s*\)\s*\)", re.IGNORECASE), "track_count / range(track_count)", "high"),
    ("32track_name", re.compile(r"32track|full32|32-track", re.IGNORECASE), "n-aware naming token", "medium"),
    ("scan_order_len32", re.compile(r"len\s*\(\s*scan_order\s*\)\s*(?:==|!=)\s*32|scan_order.*32", re.IGNORECASE), "len(scan_order)", "high"),
    ("fixed_arrays_32", re.compile(r"\[[^\]\n]*(?:31|30|29)[^\]\n]*\]"), "generated order/position array", "medium"),
    ("job_name_32", re.compile(r"Job_.*32track|J2D_.*32track|job_name.*32", re.IGNORECASE), "N-aware job_name", "medium"),
    ("output_path_32", re.compile(r"32track_[A-Za-z0-9_\\/-]+|[A-Za-z0-9_\\/-]+32track", re.IGNORECASE), "N-aware output_dir", "medium"),
    ("hardcoded_step", re.compile(r"Step-?32|step_?32|scan_step_?32", re.IGNORECASE), "loop over scan_order steps", "medium"),
    ("domain_width_literal", re.compile(r"\b(width|domain_width|coupon_width|part_width)\s*[=:]\s*[0-9.]+", re.IGNORECASE), "domain_width from N and pitch", "medium"),
    ("fixed_track_positions", re.compile(r"track_positions|region_records_by_track|region_by_track|make_track_rows", re.IGNORECASE), "track_positions generated from N", "medium"),
    ("postprocess_32", re.compile(r"32random|64prototype|full48|teacher_labels|scan_order_valid", re.IGNORECASE), "postprocessing parameterized by N", "medium"),
]


@dataclass
class SourceFileAudit:
    source_root: str
    relative_path: str
    extension: str
    size_bytes: int
    relevance_score: int
    matched_keywords: list[str]
    contains_range32: bool
    contains_32track: bool
    contains_scan_order: bool
    contains_heat_load: bool
    contains_abaqus: bool
    contains_mdb: bool
    contains_job: bool
    notes: str


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)


def rel_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def should_skip(path: Path) -> bool:
    lower_parts = {part.lower() for part in path.parts}
    return bool(lower_parts & SKIP_DIRS)


def is_allowed_file(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in FORBIDDEN_EXTENSIONS:
        return False
    if suffix not in ALLOWED_EXTENSIONS:
        return False
    try:
        return path.stat().st_size <= MAX_FILE_BYTES
    except OSError:
        return False


def discover_files() -> list[tuple[Path, Path]]:
    discovered: list[tuple[Path, Path]] = []
    seen: set[str] = set()
    for root in SOURCE_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if should_skip(path) or not path.is_file() or not is_allowed_file(path):
                continue
            path_text = str(path).lower()
            if any(token in path_text for token in SELF_AUDIT_TOKENS):
                continue
            if not any(keyword in path_text for keyword in SCAN_KEYWORDS):
                continue
            key = str(path.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            discovered.append((root, path))
    return sorted(discovered, key=lambda item: (str(item[0]).lower(), str(item[1]).lower()))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def relevance_for(relative_path: str, text: str) -> tuple[int, list[str]]:
    haystack = f"{relative_path}\n{text[:50000]}".lower()
    matched = [keyword for keyword in SCAN_KEYWORDS if keyword in haystack]
    score = len(matched)
    bonus_terms = ["mdb", "scan_order", "create_loads", "heat", "track_count", "build_case", "writeinp", "job_name", "region_by_track"]
    score += sum(3 for term in bonus_terms if term in haystack)
    if relative_path.lower().endswith(".py"):
        score += 3
    if "abaqus_scripts" in relative_path.lower():
        score += 8
    if "build_" in relative_path.lower() or "generate_" in relative_path.lower():
        score += 6
    if "postprocess" in relative_path.lower():
        score += 2
    return score, matched


def inventory_sources(files: list[tuple[Path, Path]]) -> tuple[list[SourceFileAudit], dict[str, str]]:
    inventory: list[SourceFileAudit] = []
    text_by_path: dict[str, str] = {}
    for root, path in files:
        text = read_text(path)
        relative = rel_to_root(path, root)
        score, matched = relevance_for(relative, text)
        lower = text.lower()
        text_by_path[str(path)] = text
        inventory.append(
            SourceFileAudit(
                source_root=str(root),
                relative_path=relative,
                extension=path.suffix.lower(),
                size_bytes=path.stat().st_size,
                relevance_score=score,
                matched_keywords=matched,
                contains_range32=bool(re.search(r"\brange\s*\(\s*32\s*\)", text)),
                contains_32track="32track" in lower or "32-track" in lower or "full32" in lower,
                contains_scan_order="scan_order" in lower,
                contains_heat_load=("heat" in lower and "load" in lower) or "create_loads" in lower,
                contains_abaqus="abaqus" in lower,
                contains_mdb="mdb" in lower,
                contains_job="job" in lower,
                notes="candidate generation/template source" if score >= 12 else "supporting source or documentation",
            )
        )
    return inventory, text_by_path


def source_row(item: SourceFileAudit) -> dict[str, Any]:
    return {
        "source_root": item.source_root,
        "relative_path": item.relative_path,
        "extension": item.extension,
        "size_bytes": item.size_bytes,
        "relevance_score": item.relevance_score,
        "matched_keywords": ";".join(item.matched_keywords),
        "contains_range32": item.contains_range32,
        "contains_32track": item.contains_32track,
        "contains_scan_order": item.contains_scan_order,
        "contains_heat_load": item.contains_heat_load,
        "contains_abaqus": item.contains_abaqus,
        "contains_mdb": item.contains_mdb,
        "contains_job": item.contains_job,
        "notes": item.notes,
    }


def audit_hardcodes(inventory: list[SourceFileAudit], text_by_path: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root_by_relative = {(item.source_root, item.relative_path): item for item in inventory}
    for item in inventory:
        if item.relevance_score < 10:
            continue
        path = Path(item.source_root) / item.relative_path
        text = text_by_path.get(str(path), "")
        for line_number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            for hardcode_type, pattern, suggested, risk in HARD_CODE_PATTERNS:
                if pattern.search(stripped):
                    rows.append(
                        {
                            "relative_path": item.relative_path,
                            "line_number": line_number,
                            "hardcode_type": hardcode_type,
                            "matched_text": stripped[:240],
                            "suggested_parameter": suggested,
                            "risk_level": risk,
                            "notes": "Refactor before true N=16/24/40 model generation.",
                        }
                    )
                    break
            if len(rows) >= 800:
                return rows
    return rows


def candidate_generation_files(inventory: list[SourceFileAudit]) -> list[SourceFileAudit]:
    candidates = [
        item
        for item in inventory
        if item.relevance_score >= 16
        and (
            item.contains_mdb
            or item.contains_abaqus
            or item.contains_heat_load
            or "abaqus_scripts" in item.relative_path.lower()
            or "build_" in item.relative_path.lower()
            or "generate_" in item.relative_path.lower()
        )
    ]
    return sorted(candidates, key=lambda item: item.relevance_score, reverse=True)


def make_parameterisation_plan(relevant_files: list[str]) -> list[dict[str, Any]]:
    files = "; ".join(relevant_files[:8])
    return [
        {
            "parameter_name": "track_count",
            "current_status": "often fixed at 32 or implied by fixed arrays",
            "required_for_true_variable_n": True,
            "proposed_default_or_rule": "N in {16, 24, 32, 40}; validate scan_order is a permutation of range(N)",
            "affected_files": files,
            "risk_level": "high",
            "notes": "Primary adapter entry point.",
        },
        {
            "parameter_name": "track_pitch",
            "current_status": "implicit or local to geometry/load construction",
            "required_for_true_variable_n": True,
            "proposed_default_or_rule": "Keep physical pitch constant across N unless source model documents a different convention.",
            "affected_files": files,
            "risk_level": "medium",
            "notes": "Supports true geometry rather than masked/subset N.",
        },
        {
            "parameter_name": "domain_width",
            "current_status": "may be fixed for 32-track coupon",
            "required_for_true_variable_n": True,
            "proposed_default_or_rule": "derive from margins + heat_source_width + (N - 1) * track_pitch",
            "affected_files": files,
            "risk_level": "high",
            "notes": "N=40 becomes physically wider than N=32 under fixed pitch.",
        },
        {
            "parameter_name": "track_positions",
            "current_status": "some scripts build region records by fixed track index",
            "required_for_true_variable_n": True,
            "proposed_default_or_rule": "generate positions from N, pitch, and margins; store track_index as metadata only",
            "affected_files": files,
            "risk_level": "high",
            "notes": "Must replace hardcoded 0..31 assumptions.",
        },
        {
            "parameter_name": "scan_order",
            "current_status": "validated as 0..31 in fixed-32 scripts",
            "required_for_true_variable_n": True,
            "proposed_default_or_rule": "validate len(scan_order) == N and set(scan_order) == set(range(N))",
            "affected_files": files,
            "risk_level": "high",
            "notes": "Future orders must be N-specific, not 32-track filtered subsets.",
        },
        {
            "parameter_name": "step_count",
            "current_status": "typically coupled to scan_order length but naming may assume 32",
            "required_for_true_variable_n": True,
            "proposed_default_or_rule": "generate heat-load/cooling steps by iterating over scan_order",
            "affected_files": files,
            "risk_level": "medium",
            "notes": "Do not hardcode 32 activation steps.",
        },
        {
            "parameter_name": "batch_name",
            "current_status": "many paths encode 32track",
            "required_for_true_variable_n": True,
            "proposed_default_or_rule": "include N, geometry version, strategy family, and batch id",
            "affected_files": files,
            "risk_level": "medium",
            "notes": "Avoid mixing true N with masked/subset N.",
        },
        {
            "parameter_name": "output_dir",
            "current_status": "existing folders encode 32track_full/masked",
            "required_for_true_variable_n": True,
            "proposed_default_or_rule": "outputs/true_variable_N/N{N}/... outside Git for generated CAE products",
            "affected_files": files,
            "risk_level": "medium",
            "notes": "Stage 3 repo should retain only summaries/manifests.",
        },
    ]


def make_required_changes(relevant_files: list[str]) -> list[dict[str, Any]]:
    files = "; ".join(relevant_files[:8])
    return [
        {
            "change_id": "R04-C01",
            "component": "geometry adapter",
            "current_fixed32_behavior": "32-track coupon geometry and/or region records are assumed.",
            "proposed_true_variable_n_behavior": "Create a geometry parameter object with N, pitch, margins, heat source width, and derived domain width.",
            "affected_files": files,
            "implementation_difficulty": "medium",
            "scientific_risk": "medium",
            "notes": "Must be dry-run audited before any CAE generation.",
        },
        {
            "change_id": "R04-C02",
            "component": "track position generation",
            "current_fixed32_behavior": "track indices and positions are tied to 0..31.",
            "proposed_true_variable_n_behavior": "Generate track positions from range(N) with fixed pitch.",
            "affected_files": files,
            "implementation_difficulty": "medium",
            "scientific_risk": "high",
            "notes": "Central distinction between true variable-N and masked/subset-N.",
        },
        {
            "change_id": "R04-C03",
            "component": "scan order validation",
            "current_fixed32_behavior": "permutation checks target range(32).",
            "proposed_true_variable_n_behavior": "Validate scan_order against range(N).",
            "affected_files": files,
            "implementation_difficulty": "low",
            "scientific_risk": "high",
            "notes": "Prevents accidental 32-track filtered subsets.",
        },
        {
            "change_id": "R04-C04",
            "component": "heat-load and step writer",
            "current_fixed32_behavior": "load/step names and counts may imply 32 scan activations.",
            "proposed_true_variable_n_behavior": "Derive activation/cooling steps from len(scan_order).",
            "affected_files": files,
            "implementation_difficulty": "medium",
            "scientific_risk": "medium",
            "notes": "No INP/JNL/CAE writing in run_04.",
        },
        {
            "change_id": "R04-C05",
            "component": "naming and manifests",
            "current_fixed32_behavior": "folder/job names often include 32track.",
            "proposed_true_variable_n_behavior": "Encode true N and geometry mode in job, batch, manifest, and output names.",
            "affected_files": files,
            "implementation_difficulty": "low",
            "scientific_risk": "medium",
            "notes": "Required for clean evidence separation.",
        },
        {
            "change_id": "R04-C06",
            "component": "postprocessing contract",
            "current_fixed32_behavior": "teacher tables and validators may assume 32 tracks.",
            "proposed_true_variable_n_behavior": "Parameterize postprocessing by N and store N in label tables.",
            "affected_files": files,
            "implementation_difficulty": "medium",
            "scientific_risk": "high",
            "notes": "Must happen before teacher comparisons across N.",
        },
    ]


def make_design_decisions() -> list[dict[str, Any]]:
    return [
        {
            "decision_id": "R04-D01",
            "design_question": "What should N=16/24/40 mean?",
            "chosen_decision": "true variable-N geometry",
            "rejected_alternative": "masked/subset tracks inside existing 32-track geometry",
            "rationale": "Stage 3 asks whether scan-order policy principles transfer across true track counts, not filtered 32-track masks.",
            "scientific_risk": "lower than subset-N for transfer claims",
            "implementation_risk": "requires adapter refactor",
        },
        {
            "decision_id": "R04-D02",
            "design_question": "Fixed pitch or fixed domain width?",
            "chosen_decision": "fixed physical track pitch with N-dependent domain width",
            "rejected_alternative": "fixed domain width with N-dependent track spacing",
            "rationale": "If process track spacing is physically fixed, fixed pitch is cleaner for scan-order principle transfer.",
            "scientific_risk": "N=40 is physically wider and U2 scales cannot share a full-32 guard",
            "implementation_risk": "geometry and mesh dimensions must be parameterized",
        },
        {
            "decision_id": "R04-D03",
            "design_question": "When should baselines be generated?",
            "chosen_decision": "after true variable-N CAE adapter feasibility is understood",
            "rejected_alternative": "generate N-specific baseline orders immediately",
            "rationale": "Order generation for teacher validation should not outrun the true geometry template contract.",
            "scientific_risk": "prevents accidental masked/subset-N interpretation",
            "implementation_risk": "delays baseline generator to a later run",
        },
        {
            "decision_id": "R04-D04",
            "design_question": "What is run_04 allowed to produce?",
            "chosen_decision": "small audit CSV/JSON/MD only",
            "rejected_alternative": "CAE/INP/JNL dry-run outputs",
            "rationale": "The run is a parameterisation audit, not model generation.",
            "scientific_risk": "none",
            "implementation_risk": "adapter validation deferred to run_05",
        },
    ]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def feasibility_category(candidate_count: int, hardcodes: list[dict[str, Any]]) -> str:
    if candidate_count == 0:
        return "D_INSUFFICIENT_SOURCE_EVIDENCE"
    high_risk = sum(1 for row in hardcodes if row["risk_level"] == "high")
    if high_risk == 0:
        return "A_READY_FOR_TRUE_VARIABLE_N_ADAPTER"
    if high_risk < 160:
        return "B_TRUE_VARIABLE_N_ADAPTER_REQUIRED_BUT_FEASIBLE"
    return "C_TRUE_VARIABLE_N_MAJOR_REWRITE_REQUIRED"


def verdict_for(category: str) -> str:
    if category == "A_READY_FOR_TRUE_VARIABLE_N_ADAPTER":
        return "PASS_TRUE_VARIABLE_N_CAE_TEMPLATE_AUDIT_READY"
    if category == "B_TRUE_VARIABLE_N_ADAPTER_REQUIRED_BUT_FEASIBLE":
        return "WARNING_TRUE_VARIABLE_N_CAE_TEMPLATE_ADAPTER_REQUIRED"
    if category == "C_TRUE_VARIABLE_N_MAJOR_REWRITE_REQUIRED":
        return "WARNING_TRUE_VARIABLE_N_CAE_TEMPLATE_MAJOR_REWRITE_REQUIRED"
    return "FAIL_TRUE_VARIABLE_N_CAE_TEMPLATE_SOURCE_NOT_FOUND"


def write_report(
    verdict: str,
    category: str,
    inventory: list[SourceFileAudit],
    candidates: list[SourceFileAudit],
    hardcodes: list[dict[str, Any]],
    relevant_files: list[str],
    outputs_written: list[str],
) -> None:
    hardcode_counts: dict[str, int] = {}
    for row in hardcodes:
        hardcode_counts[row["hardcode_type"]] = hardcode_counts.get(row["hardcode_type"], 0) + 1
    hardcode_text = "\n".join(f"- `{key}`: `{value}`" for key, value in sorted(hardcode_counts.items())) or "- None found"
    relevant_text = "\n".join(f"- `{path}`" for path in relevant_files[:12]) or "- None"
    outputs_text = "\n".join(f"- `{path}`" for path in outputs_written)
    report = f"""# Run 04 True Variable-N CAE Template Parameterisation Audit Report

## Executive Verdict

{verdict}

## Final Feasibility Category

{category}

## What Was Audited

- Source roots scanned: `{[str(root) for root in SOURCE_ROOTS]}`
- Files scanned: `{len(inventory)}`
- Candidate CAE generation files found: `{len(candidates)}`
- Most relevant files:

{relevant_text}

## Guardrails

- No Abaqus jobs.
- No datacheck.
- No ODB opened.
- No CAE/INP/JNL generated.
- No model training.
- No teacher validation.
- D-drive source was not modified.

## Fixed-32 Hardcode Audit

The audit found fixed-32 assumptions in these categories:

{hardcode_text}

The detailed hardcode table is capped at 800 rows to keep run_04 outputs small and GitHub-safe.

Key risk areas are track count, `range(32)` or 0..31 checks, scan-order length assumptions, 32track names, output paths, and postprocessing tables that carry fixed-32 naming or label contracts.

## True Variable-N Geometry Design

Preferred parameterisation for Option A:

- `track_count = N`, with N in `{16, 24, 32, 40}`.
- `track_pitch` remains physically constant unless the existing model documents another convention.
- `heat_source_width` is explicit and independent from N.
- `domain_width = margin_left + margin_right + heat_source_width + (N - 1) * track_pitch`.
- `domain_height` and thickness should remain physically consistent unless a documented coupon-scaling rule is introduced.
- `track_positions` are generated from N, pitch, and margins.
- `scan_order` must satisfy `len(scan_order) == N` and `set(scan_order) == set(range(N))`.
- `step_count` is derived from scan_order length N.
- `batch_name`, `strategy_name`, and `output_dir` encode N and `true_variable_n_geometry`.

## Fixed Pitch vs Fixed Domain Width

Recommendation: use fixed physical track pitch and N-dependent domain width. Fixed pitch means N=40 is physically wider than N=32, while fixed domain width would change track spacing with N. For scan-order principle transfer, fixed pitch is cleaner if the process track spacing is physically fixed.

## Required Template Changes

- Introduce a geometry/config object carrying N, pitch, heat source width, margins, derived domain width, mesh controls, and naming tokens.
- Replace `range(32)` and fixed 0..31 checks with `range(track_count)`.
- Generate track positions and sets/surfaces from N.
- Derive heat-load step loops and activation/cooling names from scan_order length.
- Make job/batch/output naming N-aware and distinguish true-N from masked/subset-N.
- Parameterize postprocessing and teacher label schemas by N.

## Risks

- N=40 may require larger geometry and mesh.
- N=16 may be less directly comparable due to a smaller domain.
- The full-32 U2 absolute guard cannot be shared across N.
- Mesh density must remain physically consistent.
- Heat source magnitude must remain physically consistent.
- Postprocessing must not assume 32 tracks.
- Computational cost may increase for N=40.

## Stop Point

The next step after run_04 is `run_05_true_variable_n_cae_generator_adapter_dryrun`.

## Claim Boundary

This run does not prove variable-N generalisation. It does not generate models. It does not validate physics. It only prepares the true variable-N teacher environment design.

## Outputs

{outputs_text}

## Final Verdict

{verdict}
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> int:
    ensure_dirs()
    files = discover_files()
    inventory, text_by_path = inventory_sources(files)
    candidates = candidate_generation_files(inventory)
    hardcodes = audit_hardcodes(inventory, text_by_path)
    relevant_files = [item.relative_path for item in candidates[:15]]
    category = feasibility_category(len(candidates), hardcodes)
    verdict = verdict_for(category)
    plan_rows = make_parameterisation_plan(relevant_files)
    change_rows = make_required_changes(relevant_files)
    decision_rows = make_design_decisions()

    outputs_written: list[str] = []
    inventory_path = OUTPUT_DIR / "cae_generation_source_inventory.csv"
    write_csv(
        inventory_path,
        [source_row(item) for item in inventory],
        [
            "source_root",
            "relative_path",
            "extension",
            "size_bytes",
            "relevance_score",
            "matched_keywords",
            "contains_range32",
            "contains_32track",
            "contains_scan_order",
            "contains_heat_load",
            "contains_abaqus",
            "contains_mdb",
            "contains_job",
            "notes",
        ],
    )
    outputs_written.append(str(inventory_path))

    hardcode_path = OUTPUT_DIR / "fixed32_hardcode_audit.csv"
    write_csv(hardcode_path, hardcodes, ["relative_path", "line_number", "hardcode_type", "matched_text", "suggested_parameter", "risk_level", "notes"])
    outputs_written.append(str(hardcode_path))

    plan_path = OUTPUT_DIR / "true_variable_n_parameterisation_plan.csv"
    write_csv(plan_path, plan_rows, ["parameter_name", "current_status", "required_for_true_variable_n", "proposed_default_or_rule", "affected_files", "risk_level", "notes"])
    outputs_written.append(str(plan_path))

    changes_path = OUTPUT_DIR / "required_template_changes.csv"
    write_csv(changes_path, change_rows, ["change_id", "component", "current_fixed32_behavior", "proposed_true_variable_n_behavior", "affected_files", "implementation_difficulty", "scientific_risk", "notes"])
    outputs_written.append(str(changes_path))

    decisions_path = OUTPUT_DIR / "true_variable_n_design_decisions.csv"
    write_csv(decisions_path, decision_rows, ["decision_id", "design_question", "chosen_decision", "rejected_alternative", "rationale", "scientific_risk", "implementation_risk"])
    outputs_written.append(str(decisions_path))

    summary_path = OUTPUT_DIR / "variable_n_cae_feasibility_summary.json"
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "feasibility_category": category,
        "recommended_route": "true_variable_n_geometry_with_fixed_physical_track_pitch",
        "rejected_routes": [
            "masked_or_subset_tracks_inside_existing_32track_geometry_as_main_stage3_teacher_design",
            "claiming masked/subset-N as clean variable-N generalisation",
        ],
        "most_relevant_files": relevant_files[:12],
        "required_changes": [row["change_id"] for row in change_rows],
        "next_run": "run_05_true_variable_n_cae_generator_adapter_dryrun",
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    outputs_written.append(str(summary_path))

    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_roots": [str(root) for root in SOURCE_ROOTS],
        "target_root": str(TARGET_ROOT),
        "python_executable": sys.executable,
        "run_id": RUN_ID,
        "verdict": verdict,
        "feasibility_category": category,
        "selected_route": "true_variable_n_geometry",
        "files_scanned_count": len(inventory),
        "candidate_cae_generation_files_count": len(candidates),
        "most_relevant_files": relevant_files[:12],
        "outputs_written": outputs_written,
        "forbidden_actions_confirmed": {
            "no_abaqus_jobs": True,
            "no_datacheck": True,
            "no_odb_opened": True,
            "no_cae_generated": True,
            "no_inp_generated": True,
            "no_jnl_generated": True,
            "no_model_training": True,
            "no_teacher_validation": True,
            "stage2_source_read_only": True,
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    outputs_written.append(str(MANIFEST_PATH))
    write_report(verdict, category, inventory, candidates, hardcodes, relevant_files, outputs_written)
    outputs_written.append(str(REPORT_PATH))

    print(f"Files scanned: {len(inventory)}")
    print(f"Candidate CAE generation files: {len(candidates)}")
    print(f"Fixed-32 hardcode findings: {len(hardcodes)}")
    print(f"Feasibility category: {category}")
    print("Selected route: true_variable_n_geometry")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Report: {REPORT_PATH}")
    print(verdict)
    return 1 if verdict == "FAIL_TRUE_VARIABLE_N_CAE_TEMPLATE_SOURCE_NOT_FOUND" else 0


if __name__ == "__main__":
    raise SystemExit(main())
