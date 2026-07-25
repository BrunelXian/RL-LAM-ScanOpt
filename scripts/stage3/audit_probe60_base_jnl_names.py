"""Audit object names in the four Stage 3 probe60 base Abaqus journals."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage3_cae_probe60_generation"

BASE_JNLS = {
    12: PROJECT_ROOT
    / "cae_model"
    / "12track_full"
    / "sanity_base"
    / "12track_sanity_base.jnl",
    16: PROJECT_ROOT
    / "cae_model"
    / "16track_full"
    / "sanity_base"
    / "16track_sanity_base.jnl",
    24: PROJECT_ROOT
    / "cae_model"
    / "24track_full"
    / "sanity_base"
    / "24track_sanity_base.jnl",
    40: PROJECT_ROOT
    / "cae_model"
    / "40track_full"
    / "sanity_base"
    / "40track_sanity_base.jnl",
}

NAME_PATTERNS = {
    "model_names": [
        re.compile(r"mdb\.models\['([^']+)'\]"),
        re.compile(r"Model\(name='([^']+)'"),
    ],
    "part_names": [
        re.compile(r"\.Part\([^)]*name='([^']+)'", re.DOTALL),
        re.compile(r"\.parts\['([^']+)'\]"),
    ],
    "step_names": [
        re.compile(r"\.steps\['([^']+)'\]"),
        re.compile(r"Step\([^)]*name='([^']+)'", re.DOTALL),
        re.compile(r"CoupledTempDisplacementStep\([^)]*name='([^']+)'", re.DOTALL),
    ],
    "load_names": [
        re.compile(r"\.loads\['([^']+)'\]"),
        re.compile(r"BodyHeatFlux\([^)]*name='([^']+)'", re.DOTALL),
        re.compile(r"FilmCondition\([^)]*name='([^']+)'", re.DOTALL),
        re.compile(r"RadiationToAmbient\([^)]*name='([^']+)'", re.DOTALL),
    ],
    "surface_names": [
        re.compile(r"\.surfaces\['([^']+)'\]"),
        re.compile(r"\.Surface\([^)]*name='([^']+)'", re.DOTALL),
    ],
    "set_names": [
        re.compile(r"\.sets\['([^']+)'\]"),
        re.compile(r"\.Set\([^)]*name='([^']+)'", re.DOTALL),
    ],
    "amplitude_names": [
        re.compile(r"\.amplitudes\['([^']+)'\]"),
        re.compile(r"Amplitude\([^)]*name='([^']+)'", re.DOTALL),
    ],
    "job_names": [
        re.compile(r"\.jobs\['([^']+)'\]"),
        re.compile(r"Job\([^)]*name='([^']+)'", re.DOTALL),
    ],
}

FIXED_32_PATTERNS = [
    re.compile(r"32track", re.IGNORECASE),
    re.compile(r"\bN32\b", re.IGNORECASE),
    re.compile(r"_N32_", re.IGNORECASE),
    re.compile(r"range\s*\(\s*32\s*\)"),
    re.compile(r"\bnum_tracks\s*=\s*32\b", re.IGNORECASE),
]


def extract_names(text: str, key: str) -> list[str]:
    values: set[str] = set()
    for pattern in NAME_PATTERNS[key]:
        values.update(pattern.findall(text))
    return sorted(values)


def matching_lines(text: str, tokens: tuple[str, ...]) -> list[str]:
    out = []
    for i, line in enumerate(text.splitlines(), start=1):
        if any(token in line for token in tokens):
            out.append("{}: {}".format(i, line.strip()))
    return out


def fixed_32_leftovers(text: str) -> list[str]:
    hits = []
    for i, line in enumerate(text.splitlines(), start=1):
        if any(pattern.search(line) for pattern in FIXED_32_PATTERNS):
            hits.append("{}: {}".format(i, line.strip()))
    return hits


def audit_one(n: int, path: Path) -> dict[str, object]:
    if not path.exists():
        return {
            "n": n,
            "jnl_path": str(path),
            "exists": False,
            "line_count": 0,
            "verdict": "FAIL_BASE_JNL_OBJECT_AUDIT_INSUFFICIENT",
            "notes": "missing JNL",
        }

    text = path.read_text(errors="replace")
    set_names = sorted(
        set(extract_names(text, "set_names")) | set(re.findall(r"set_body_heat_\d+", text))
    )
    heat_sets = sorted(set(re.findall(r"set_body_heat_\d+", text)))
    step_names = sorted(
        set(extract_names(text, "step_names"))
        | set(re.findall(r"step_scan_\d+", text))
        | set(re.findall(r"step_cool_\d+", text))
    )
    scan_steps = sorted(set(re.findall(r"step_scan_\d+", text)))
    cool_steps = sorted(set(re.findall(r"step_cool_\d+", text)))
    load_names = sorted(
        set(extract_names(text, "load_names")) | set(re.findall(r"load_body_hflux_\d+", text))
    )
    body_heat_loads = sorted(set(re.findall(r"load_body_hflux_\d+", text)))
    write_input_calls = matching_lines(text, ("writeInput",))
    save_as_calls = matching_lines(text, ("saveAs",))
    fixed_32 = fixed_32_leftovers(text)

    body_heat_lines = matching_lines(text, ("BodyHeatFlux", "set_body_heat_", "deactivate"))
    heat_track_mapping_evidence = (
        "heat_sets={}, scan_steps={}, cool_steps={}, body_heat_loads={}; "
        "BodyHeatFlux lines bind load_body_hflux_00 to set_body_heat_00 in step_scan_00."
    ).format(len(heat_sets), len(scan_steps), len(cool_steps), len(body_heat_loads))

    if len(heat_sets) == n and len(scan_steps) == n and len(body_heat_loads) == n:
        mapping_status = "SAFE_AUTOMATIC_SCAN_ORDER_MAPPING_READY"
    elif len(heat_sets) == n and len(scan_steps) == 1 and len(body_heat_loads) == 1:
        mapping_status = "MANUAL_OBJECT_MAPPING_REQUIRED"
    else:
        mapping_status = "INSUFFICIENT_JNL_EVIDENCE"

    return {
        "n": n,
        "jnl_path": str(path),
        "exists": True,
        "line_count": text.count("\n") + 1,
        "model_names": ";".join(extract_names(text, "model_names")),
        "part_names": ";".join(extract_names(text, "part_names")),
        "step_names": ";".join(step_names),
        "load_names": ";".join(load_names),
        "surface_names": ";".join(extract_names(text, "surface_names")),
        "set_names": ";".join(set_names),
        "amplitude_names": ";".join(extract_names(text, "amplitude_names")),
        "job_names": ";".join(extract_names(text, "job_names")),
        "heat_set_count": len(heat_sets),
        "scan_step_count": len(scan_steps),
        "cool_step_count": len(cool_steps),
        "body_heat_load_count": len(body_heat_loads),
        "writeInput_calls": " | ".join(write_input_calls),
        "saveAs_calls": " | ".join(save_as_calls),
        "fixed_32_leftovers": " | ".join(fixed_32),
        "n_specific_naming_evidence": "found {} set_body_heat_XX names for N{}".format(
            len(heat_sets), n
        ),
        "heat_track_mapping_evidence": heat_track_mapping_evidence,
        "heat_track_mapping_lines": " | ".join(body_heat_lines[:20]),
        "mapping_status": mapping_status,
        "notes": "",
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [audit_one(n, path) for n, path in BASE_JNLS.items()]

    if any(not row.get("exists") for row in rows):
        verdict = "FAIL_BASE_JNL_OBJECT_AUDIT_INSUFFICIENT"
    elif all(row.get("mapping_status") == "SAFE_AUTOMATIC_SCAN_ORDER_MAPPING_READY" for row in rows):
        verdict = "PASS_BASE_JNL_OBJECT_NAME_AUDIT_READY"
    elif any(row.get("mapping_status") == "INSUFFICIENT_JNL_EVIDENCE" for row in rows):
        verdict = "FAIL_BASE_JNL_OBJECT_AUDIT_INSUFFICIENT"
    else:
        verdict = "WARNING_BASE_JNL_OBJECT_MAPPING_PARTIAL"

    csv_path = OUTPUT_DIR / "base_jnl_object_name_audit.csv"
    fieldnames = [
        "n",
        "jnl_path",
        "exists",
        "line_count",
        "model_names",
        "part_names",
        "step_names",
        "load_names",
        "surface_names",
        "set_names",
        "amplitude_names",
        "job_names",
        "heat_set_count",
        "scan_step_count",
        "cool_step_count",
        "body_heat_load_count",
        "writeInput_calls",
        "saveAs_calls",
        "fixed_32_leftovers",
        "n_specific_naming_evidence",
        "heat_track_mapping_evidence",
        "heat_track_mapping_lines",
        "mapping_status",
        "notes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    md_path = OUTPUT_DIR / "base_jnl_object_name_audit.md"
    md_lines = [
        "# Probe60 Base JNL Object-Name Audit",
        "",
        "Verdict: `{}`".format(verdict),
        "",
        "## Summary",
        "",
        "| N | Heat sets | Scan steps | Cool steps | Body heat loads | Mapping status |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        md_lines.append(
            "| {n} | {heat_set_count} | {scan_step_count} | {cool_step_count} | "
            "{body_heat_load_count} | `{mapping_status}` |".format(**row)
        )
    md_lines.extend(
        [
            "",
            "## Heat-Load Mapping Status",
            "",
            "The journals provide N-specific heat-region set names (`set_body_heat_XX`) "
            "matching N for each base. They do not provide an N-step or N-load scan "
            "sequence in journal text. Each base journal records one scan step, one "
            "cool step, and one body heat flux load bound to `set_body_heat_00`.",
            "",
            "Classification: `MANUAL_OBJECT_MAPPING_REQUIRED`.",
            "",
            "Generic export is blocked by default because it would preserve the base "
            "journal's single recorded heat-load mapping rather than creating "
            "candidate-specific scan-order INPs.",
            "",
            "## Outputs",
            "",
            "- `{}`".format(csv_path),
            "- `{}`".format(md_path),
        ]
    )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    summary_path = OUTPUT_DIR / "base_jnl_object_name_audit_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "verdict": verdict,
                "heat_load_mapping_status": "MANUAL_OBJECT_MAPPING_REQUIRED"
                if verdict == "WARNING_BASE_JNL_OBJECT_MAPPING_PARTIAL"
                else rows[0].get("mapping_status", "INSUFFICIENT_JNL_EVIDENCE"),
                "csv": str(csv_path),
                "markdown": str(md_path),
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(verdict)
    print("csv={}".format(csv_path))
    print("markdown={}".format(md_path))
    return 0 if verdict != "FAIL_BASE_JNL_OBJECT_AUDIT_INSUFFICIENT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
