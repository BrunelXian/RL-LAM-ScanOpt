"""Extract heat-load mapping evidence from Stage 3 probe60 base JNL files."""

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

TOKENS = (
    "set_body_heat_",
    "step_scan_",
    "step_cool_",
    "load_body_hflux_",
    "BodyHeatFlux",
    "ConcentratedHeatFlux",
    "SurfaceHeatFlux",
    "mdb.models",
    "HeatTransferStep",
    "CoupledTempDisplacementStep",
    "StaticStep",
    "loads[",
    "createStepName",
    "deactivate",
    "setValuesInStep",
    "region=",
    "Region",
    "Job(",
    "writeInput",
    "saveAs",
)


def snippet_lines(lines: list[str], line_index: int, radius: int = 3) -> str:
    start = max(0, line_index - radius)
    end = min(len(lines), line_index + radius + 1)
    return "\n".join(
        "{:04d}: {}".format(i + 1, lines[i].rstrip()) for i in range(start, end)
    )


def find_snippets(n: int, path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return [
            {
                "n": n,
                "jnl_path": str(path),
                "line": 0,
                "matched_tokens": "MISSING_FILE",
                "snippet": "",
            }
        ]
    lines = path.read_text(errors="replace").splitlines()
    rows: list[dict[str, object]] = []
    for i, line in enumerate(lines):
        matched = [token for token in TOKENS if token in line]
        if matched:
            rows.append(
                {
                    "n": n,
                    "jnl_path": str(path),
                    "line": i + 1,
                    "matched_tokens": ";".join(matched),
                    "snippet": snippet_lines(lines, i),
                }
            )
    return rows


def first_or_none(values: list[str]) -> str | None:
    return values[0] if values else None


def extract_step_period(text: str, step_name: str) -> dict[str, object]:
    idx = text.find("name='{}'".format(step_name))
    if idx < 0:
        return {"step_name": step_name, "procedure": None, "time_period": None, "evidence": ""}
    proc_starts = list(re.finditer(r"([A-Za-z]+Step)\(", text[:idx]))
    start = proc_starts[-1].start() if proc_starts else max(0, idx - 250)
    next_call = text.find("\nmdb.models", idx + len(step_name))
    end = next_call if next_call > idx else idx + 450
    window = text[start:end]
    proc_match = re.search(r"([A-Za-z]+Step)\(", window)
    period_match = re.search(r"timePeriod\s*=\s*([0-9.eE+-]+)", window)
    return {
        "step_name": step_name,
        "procedure": proc_match.group(1) if proc_match else None,
        "time_period": float(period_match.group(1)) if period_match else None,
        "evidence": " ".join(window.split()),
    }


def extract_load_evidence(text: str, load_name: str) -> dict[str, object]:
    idx = text.find("name='{}'".format(load_name))
    if idx < 0:
        return {
            "load_name": load_name,
            "load_type": None,
            "magnitude": None,
            "region_evidence": "",
            "evidence": "",
        }
    window = text[max(0, idx - 250) : idx + 450]
    load_type_match = re.search(
        r"(BodyHeatFlux|ConcentratedHeatFlux|SurfaceHeatFlux)\(", window
    )
    magnitude_match = re.search(r"magnitude\s*=\s*([0-9.eE+-]+)", window)
    region_match = re.search(r"region\s*=\s*(.+?)(?:\)|, distributionType|$)", window)
    return {
        "load_name": load_name,
        "load_type": load_type_match.group(1) if load_type_match else None,
        "magnitude": float(magnitude_match.group(1)) if magnitude_match else None,
        "region_evidence": " ".join(region_match.group(1).split())
        if region_match
        else "",
        "evidence": " ".join(window.split()),
    }


def infer_for_jnl(n: int, path: Path) -> dict[str, object]:
    missing: list[str] = []
    if not path.exists():
        return {
            "n": n,
            "jnl_path": str(path),
            "heat_set_pattern": None,
            "heat_set_count": 0,
            "scan_step_template_name": None,
            "cool_step_template_name": None,
            "load_template_name": None,
            "model_name_candidates": [],
            "part_or_assembly_region_evidence": [],
            "load_type_evidence": None,
            "load_magnitude_evidence": None,
            "step_duration_evidence": None,
            "cool_duration_evidence": None,
            "can_infer_mapping_from_jnl": False,
            "missing_evidence": ["missing JNL file"],
        }

    text = path.read_text(errors="replace")
    heat_sets = sorted(set(re.findall(r"set_body_heat_\d+", text)))
    scan_steps = sorted(set(re.findall(r"step_scan_\d+", text)))
    cool_steps = sorted(set(re.findall(r"step_cool_\d+", text)))
    loads = sorted(set(re.findall(r"load_body_hflux_\d+", text)))
    model_names = sorted(set(re.findall(r"mdb\.models\['([^']+)'\]", text)))

    scan_template = first_or_none(scan_steps)
    cool_template = first_or_none(cool_steps)
    load_template = first_or_none(loads)

    if len(heat_sets) != n:
        missing.append("expected {} set_body_heat_XX names, found {}".format(n, len(heat_sets)))
    if not scan_template:
        missing.append("missing step_scan_XX template")
    if not cool_template:
        missing.append("missing step_cool_XX template")
    if not load_template:
        missing.append("missing load_body_hflux_XX template")
    if not model_names:
        missing.append("missing mdb.models model-name evidence")

    region_evidence = []
    for line in text.splitlines():
        if "set_body_heat_" in line or "region=" in line:
            compact = " ".join(line.strip().split())
            if compact and compact not in region_evidence:
                region_evidence.append(compact)
        if len(region_evidence) >= 20:
            break

    load_info = extract_load_evidence(text, load_template) if load_template else {}
    scan_info = extract_step_period(text, scan_template) if scan_template else {}
    cool_info = extract_step_period(text, cool_template) if cool_template else {}

    if not load_info.get("load_type"):
        missing.append("missing heat flux load type evidence")
    if load_info.get("magnitude") is None:
        missing.append("missing load magnitude evidence")
    if scan_info.get("time_period") is None:
        missing.append("missing scan step duration evidence")
    if cool_info.get("time_period") is None:
        missing.append("missing cool step duration evidence")

    missing.append("CAE model object inspection not run in this text-only extractor")
    missing.append("user confirmation config remains required before generation")

    can_infer = (
        len(heat_sets) == n
        and scan_template is not None
        and cool_template is not None
        and load_template is not None
        and bool(model_names)
        and bool(load_info.get("load_type"))
        and load_info.get("magnitude") is not None
        and scan_info.get("time_period") is not None
        and cool_info.get("time_period") is not None
    )

    return {
        "n": n,
        "jnl_path": str(path),
        "heat_set_pattern": "set_body_heat_{track:02d}" if heat_sets else None,
        "heat_set_count": len(heat_sets),
        "scan_step_template_name": scan_template,
        "cool_step_template_name": cool_template,
        "load_template_name": load_template,
        "model_name_candidates": model_names,
        "part_or_assembly_region_evidence": region_evidence,
        "load_type_evidence": load_info,
        "load_magnitude_evidence": load_info.get("magnitude"),
        "step_duration_evidence": scan_info,
        "cool_duration_evidence": cool_info,
        "can_infer_mapping_from_jnl": can_infer,
        "missing_evidence": missing,
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    snippet_rows: list[dict[str, object]] = []
    inferred: dict[str, object] = {
        "verdict": "",
        "notes": (
            "JNL text can infer naming and template parameters, but Abaqus CAE "
            "object inspection and user confirmation are still required before generation."
        ),
        "by_n": {},
    }
    for n, path in BASE_JNLS.items():
        snippet_rows.extend(find_snippets(n, path))
        inferred["by_n"][str(n)] = infer_for_jnl(n, path)

    can_infer_all = all(
        item["can_infer_mapping_from_jnl"]
        for item in inferred["by_n"].values()  # type: ignore[union-attr]
    )
    files_exist = all(path.exists() for path in BASE_JNLS.values())
    if can_infer_all:
        verdict = "PASS_HEAT_MAPPING_JNL_PATTERNS_READY"
    elif files_exist:
        verdict = "WARNING_HEAT_MAPPING_JNL_PARTIAL"
    else:
        verdict = "FAIL_HEAT_MAPPING_JNL_INSUFFICIENT"
    inferred["verdict"] = verdict

    csv_path = OUTPUT_DIR / "heat_mapping_jnl_snippets.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["n", "jnl_path", "line", "matched_tokens", "snippet"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(snippet_rows)

    json_path = OUTPUT_DIR / "heat_mapping_inferred_patterns.json"
    json_path.write_text(json.dumps(inferred, indent=2), encoding="utf-8")

    md_path = OUTPUT_DIR / "heat_mapping_jnl_snippets.md"
    md_lines = [
        "# Probe60 Heat Mapping JNL Snippets",
        "",
        "Verdict: `{}`".format(verdict),
        "",
        "The snippets below are extracted from journal text only. They do not "
        "replace Abaqus CAE object inspection before generation.",
        "",
        "## Inferred Patterns",
        "",
        "| N | Heat sets | Scan template | Cool template | Load template | Can infer from JNL |",
        "|---:|---:|---|---|---|---|",
    ]
    for n in BASE_JNLS:
        item = inferred["by_n"][str(n)]  # type: ignore[index]
        md_lines.append(
            "| {n} | {heat_set_count} | `{scan}` | `{cool}` | `{load}` | `{can}` |".format(
                n=n,
                heat_set_count=item["heat_set_count"],
                scan=item["scan_step_template_name"],
                cool=item["cool_step_template_name"],
                load=item["load_template_name"],
                can=item["can_infer_mapping_from_jnl"],
            )
        )
    md_lines.extend(["", "## Snippets", ""])
    for row in snippet_rows:
        md_lines.extend(
            [
                "### N{} line {} ({})".format(
                    row["n"], row["line"], row["matched_tokens"]
                ),
                "",
                "```text",
                str(row["snippet"]),
                "```",
                "",
            ]
        )
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(verdict)
    print("markdown={}".format(md_path))
    print("csv={}".format(csv_path))
    print("json={}".format(json_path))
    return 0 if verdict != "FAIL_HEAT_MAPPING_JNL_INSUFFICIENT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
