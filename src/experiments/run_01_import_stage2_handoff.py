from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


SOURCE_ROOT = Path(r"D:\Projects\RL-LAM-ScanOpt")
TARGET_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")

APPROVED_STAGE2_DOCS = (
    "STAGE2_FINAL_SUMMARY.md",
    "STAGE2_CLAIM_BOUNDARY.md",
    "STAGE2_STAGE3_HANDOFF.md",
    "STAGE2_KEY_RESULTS_TABLE.csv",
)

REQUIRED_DIRS = (
    Path("docs/stage3"),
    Path("docs/stage3/runs/run_01_handoff_import"),
    Path("docs/stage2_reference"),
    Path("src/experiments"),
    Path("src/feature_builders"),
    Path("src/policies"),
    Path("src/baselines"),
    Path("src/evaluation"),
    Path("artifacts/manifests"),
    Path("outputs"),
)

DISALLOWED_SUFFIXES = {
    ".odb",
    ".cae",
    ".sim",
    ".dat",
    ".msg",
    ".sta",
    ".lck",
    ".prt",
    ".com",
    ".mdl",
    ".stt",
    ".res",
    ".abq",
    ".pac",
    ".sel",
    ".ipm",
}


def ensure_dirs() -> None:
    for relative_dir in REQUIRED_DIRS:
        (TARGET_ROOT / relative_dir).mkdir(parents=True, exist_ok=True)


def find_stage2_reference(filename: str) -> Path | None:
    direct_path = SOURCE_ROOT / "docs" / "stage2" / filename
    if direct_path.is_file():
        return direct_path

    matches = [path for path in SOURCE_ROOT.rglob(filename) if path.is_file()]
    if not matches:
        return None
    return sorted(matches, key=lambda path: len(path.parts))[0]


def copy_approved_references() -> tuple[list[dict[str, str | int]], list[str]]:
    copied: list[dict[str, str | int]] = []
    missing: list[str] = []
    destination_dir = TARGET_ROOT / "docs" / "stage2_reference"

    for filename in APPROVED_STAGE2_DOCS:
        source_path = find_stage2_reference(filename)
        if source_path is None:
            missing.append(filename)
            continue
        if source_path.suffix.lower() in DISALLOWED_SUFFIXES:
            missing.append(filename)
            continue

        destination_path = destination_dir / filename
        shutil.copy2(source_path, destination_path)
        copied.append(
            {
                "filename": filename,
                "source": str(source_path),
                "destination": str(destination_path),
                "bytes": destination_path.stat().st_size,
            }
        )

    return copied, missing


def verdict_for(copied_count: int, missing_count: int) -> str:
    if copied_count == 0:
        return "FAIL_STAGE3_RUN01_NO_STAGE2_REFERENCES_FOUND"
    if missing_count:
        return "WARNING_STAGE3_RUN01_PARTIAL_IMPORT_MISSING_REFERENCES"
    return "PASS_STAGE3_RUN01_HANDOFF_IMPORT_READY"


def write_manifest(copied: list[dict[str, str | int]], missing: list[str], verdict: str) -> Path:
    manifest_path = TARGET_ROOT / "artifacts" / "manifests" / "stage3_run_01_manifest.json"
    manifest = {
        "run_id": "run_01_handoff_import",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(SOURCE_ROOT),
        "target_root": str(TARGET_ROOT),
        "approved_stage2_docs": list(APPROVED_STAGE2_DOCS),
        "copied_references": copied,
        "missing_references": missing,
        "constraints": {
            "abaqus_jobs": False,
            "datacheck": False,
            "odb_opening": False,
            "cae_generation": False,
            "inp_jnl_generation": False,
            "model_training": False,
            "candidate_generation": False,
        },
        "stage3_boundary": {
            "goal": "Variable-N Graph Pointer RL Policy feasibility test",
            "n_train": [16, 32],
            "n_test": [24, 40],
            "evaluation": "within-N ranking and normalized improvement",
            "not_claimed": [
                "arbitrary-N generalisation solved",
                "GNN/RL is final physical optimiser",
                "same full-32 U2 guard transfers to all N",
                "masked transfer solved",
                "SurfaceT optimisation solved",
            ],
        },
        "verdict": verdict,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def write_report(copied: list[dict[str, str | int]], missing: list[str], verdict: str, manifest_path: Path) -> Path:
    report_path = (
        TARGET_ROOT
        / "docs"
        / "stage3"
        / "runs"
        / "run_01_handoff_import"
        / "RUN_01_HANDOFF_IMPORT_REPORT.md"
    )

    copied_lines = "\n".join(
        f"- {item['filename']} ({item['bytes']} bytes)" for item in copied
    ) or "- None"
    missing_lines = "\n".join(f"- {filename}" for filename in missing) or "- None"

    report = f"""# Run 01 Handoff Import Report

## Verdict

{verdict}

## Scope

Run 01 imports only approved Stage 2 reference documents from `{SOURCE_ROOT}` into `{TARGET_ROOT / 'docs' / 'stage2_reference'}`.

## Copied References

{copied_lines}

## Missing References

{missing_lines}

## Manifest

- `{manifest_path}`

## Constraints Confirmed

- No Abaqus jobs.
- No datacheck.
- No ODB opening.
- No CAE generation.
- No INP/JNL generation.
- No model training.
- No candidate generation.
- No solver outputs copied.

## Stage 3 Boundary

Stage 3 tests Variable-N Graph Pointer RL Policy feasibility for `N_train = {{16, 32}}` and `N_test = {{24, 40}}`. Evidence must use within-N ranking and normalized improvement. It does not claim arbitrary-N generalisation, a final physical optimiser, universal full-32 U2 guard transfer, solved masked transfer, or solved SurfaceT optimisation.
"""
    report_path.write_text(report, encoding="utf-8")
    return report_path


def main() -> int:
    ensure_dirs()
    copied, missing = copy_approved_references()
    verdict = verdict_for(len(copied), len(missing))
    manifest_path = write_manifest(copied, missing, verdict)
    report_path = write_report(copied, missing, verdict, manifest_path)

    print(f"Copied references: {len(copied)}")
    if missing:
        print("Missing references:")
        for filename in missing:
            print(f"  - {filename}")
    else:
        print("Missing references: none")
    print(f"Manifest: {manifest_path}")
    print(f"Report: {report_path}")
    print(verdict)

    return 1 if verdict == "FAIL_STAGE3_RUN01_NO_STAGE2_REFERENCES_FOUND" else 0


if __name__ == "__main__":
    raise SystemExit(main())
