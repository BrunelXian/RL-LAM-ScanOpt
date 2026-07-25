"""Prepare compact ChatGPT Project upload package for final PPO 320 evidence.

Packaging-only. This script does not run Abaqus, open/extract ODB files, run
solver/datacheck/enqueue, generate CAE/INP/JNL files, train models, generate
candidates, mutate scan orders, commit, or push.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd


ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
PACKAGE_NAME = "RL_LAM_ScanOpt_PPO_Final_320_Evidence_Package_v01"
UPLOAD_ROOT = ROOT / "CHATGPT_PROJECT_UPLOAD"
PACKAGE_ROOT = UPLOAD_ROOT / PACKAGE_NAME
ZIP_PATH = UPLOAD_ROOT / f"{PACKAGE_NAME}.zip"

EXCLUDED_EXTENSIONS = {".odb", ".cae", ".inp", ".jnl", ".sim", ".sta", ".dat", ".msg", ".lck", ".zip"}
PLOT_COPY_LIMIT_BYTES = 25 * 1024 * 1024
FINAL_VERDICT = "PASS_CHATGPT_PROJECT_UPLOAD_PACKAGE_READY"

DIRS = [
    "00_README",
    "01_FINAL_STAGE_X_FREEZE",
    "02_STAGE_W_ANALYSIS",
    "03_STAGE_HISTORY_SUMMARIES",
    "04_CORE_REFERENCE_DATA",
    "05_MANUSCRIPT_TABLES",
    "06_CLAIM_BOUNDARIES",
    "07_HASHES_AND_MANIFESTS",
    "08_WRITING_BRIEF",
    "09_OPTIONAL_PLOTS_INDEX_ONLY",
]


STAGE_X_OUTPUT = ROOT / "outputs" / "stage3_ppo_final_pool_320_evidence_freeze"
STAGE_X_DOCS = ROOT / "docs" / "stage3_ppo_final_pool_320_evidence_freeze"
STAGE_W_OUTPUT = ROOT / "outputs" / "stage3_ppo_final_pool_320_analysis"
STAGE_W_DOCS = ROOT / "docs" / "stage3_ppo_final_pool_320_analysis"
RUN78_OUTPUT = ROOT / "outputs" / "stage3_run_78_final_evidence_freeze_package"


def git_branch() -> str:
    try:
        result = subprocess.run(["git", "branch", "--show-current"], cwd=str(ROOT), capture_output=True, text=True, check=False)
        return result.stdout.strip() or "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_package_root() -> None:
    if PACKAGE_ROOT.exists():
        shutil.rmtree(PACKAGE_ROOT)
    for rel in DIRS:
        (PACKAGE_ROOT / rel).mkdir(parents=True, exist_ok=True)
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)


def safe_copy(src: Path, dst_dir: Path, role: str, copied: list[dict], missing: list[dict], dst_name: str | None = None, required: bool = False) -> None:
    if not src.exists():
        missing.append({"source_path": str(src), "intended_role": role, "required": required, "reason": "not_found"})
        if required:
            raise FileNotFoundError(f"Required file missing: {src}")
        return
    if src.suffix.lower() in EXCLUDED_EXTENSIONS:
        missing.append({"source_path": str(src), "intended_role": role, "required": required, "reason": f"excluded_extension_{src.suffix.lower()}"})
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / (dst_name or src.name)
    shutil.copy2(src, dst)
    copied.append(
        {
            "relative_path": str(dst.relative_to(PACKAGE_ROOT)),
            "source_path": str(src),
            "file_role": role,
            "file_size_bytes": dst.stat().st_size,
            "extension": dst.suffix.lower(),
            "copied_timestamp": datetime.now().isoformat(timespec="seconds"),
        }
    )


def copy_named_files(copied: list[dict], missing: list[dict]) -> None:
    # Stage X final freeze.
    stage_x_files = [
        (STAGE_X_OUTPUT / "frozen_tables" / "FROZEN_PPO_final_pool_320_teacher_metrics.csv", "stageX_frozen_final_pool", True),
        (STAGE_X_OUTPUT / "frozen_tables" / "FROZEN_PPO_final_pool_320_teacher_metric_ranking_full.csv", "stageX_frozen_ranking", True),
        (STAGE_X_OUTPUT / "frozen_tables" / "FROZEN_combined552_plus_PPO_final_pool_320_analysis_dataset.csv", "stageX_frozen_combined552_plus_ppo", True),
        (STAGE_X_OUTPUT / "frozen_tables" / "FROZEN_PPO_final_pool_320_claim_decision_table.csv", "stageX_frozen_claim_decision", True),
        (STAGE_X_OUTPUT / "manuscript_tables" / "PPO_final_pool_320_composition_for_manuscript.csv", "stageX_manuscript_composition", True),
        (STAGE_X_OUTPUT / "manuscript_tables" / "PPO_final_pool_320_main_results_for_manuscript.csv", "stageX_manuscript_main_results", True),
        (STAGE_X_OUTPUT / "manuscript_tables" / "PPO_final_pool_320_safe_claims_for_manuscript.csv", "stageX_manuscript_safe_claims", True),
        (STAGE_X_OUTPUT / "hashes" / "FROZEN_PPO_final_pool_320_file_hashes.csv", "stageX_hashes", True),
        (STAGE_X_OUTPUT / "stageX_PPO_final_pool_320_evidence_freeze_manifest.json", "stageX_manifest", True),
        (STAGE_X_DOCS / "PPO_FINAL_POOL_320_EVIDENCE_INDEX.md", "stageX_evidence_index", True),
        (STAGE_X_DOCS / "PPO_FINAL_POOL_320_FINAL_CLAIM_BOUNDARY.md", "stageX_final_claim_boundary", True),
        (STAGE_X_DOCS / "PPO_FINAL_POOL_320_MANUSCRIPT_INTEGRATION_MEMO.md", "stageX_manuscript_memo", True),
        (STAGE_X_DOCS / "PPO_FINAL_POOL_320_EVIDENCE_FREEZE_REPORT.md", "stageX_freeze_report", True),
    ]
    for src, role, required in stage_x_files:
        safe_copy(src, PACKAGE_ROOT / "01_FINAL_STAGE_X_FREEZE", role, copied, missing, required=required)

    # Duplicate manuscript-ready materials into dedicated folder.
    for src, role, required in stage_x_files:
        if "manuscript" in role or "claim_boundary" in role:
            subdir = "05_MANUSCRIPT_TABLES" if src.suffix.lower() == ".csv" else "06_CLAIM_BOUNDARIES"
            safe_copy(src, PACKAGE_ROOT / subdir, f"{role}_copy", copied, missing, required=False)

    # Stage W analysis.
    stage_w_files = [
        (STAGE_W_OUTPUT / "tables" / "ppo_final_pool_320_teacher_metrics.csv", "stageW_final_pool", True),
        (STAGE_W_OUTPUT / "tables" / "combined552_plus_ppo_final_pool_320_analysis_dataset.csv", "stageW_combined552_plus_ppo", True),
        (STAGE_W_OUTPUT / "tables" / "ppo_final_pool_320_teacher_metric_ranking_full.csv", "stageW_ranking", True),
        (STAGE_W_OUTPUT / "tables" / "ppo_final_pool_320_claim_decision_table.csv", "stageW_claim_decision", True),
        (STAGE_W_OUTPUT / "tables" / "ppo_final_pool_320_topk_summary_by_N.csv", "stageW_topk_by_N", True),
        (STAGE_W_OUTPUT / "tables" / "ppo_final_pool_320_topk_summary_by_version.csv", "stageW_topk_by_version", True),
        (STAGE_W_OUTPUT / "tables" / "ppo_final_pool_320_new_record_candidates.csv", "stageW_new_records", True),
        (STAGE_W_OUTPUT / "tables" / "ppo_final_pool_320_vs_bootstrap_random_reference_global.csv", "stageW_bootstrap_global", True),
        (STAGE_W_OUTPUT / "tables" / "ppo_final_pool_320_vs_bootstrap_random_reference_by_N.csv", "stageW_bootstrap_by_N", True),
        (STAGE_W_OUTPUT / "tables" / "ppo_final_pool_320_vs_identified_baseline_families.csv", "stageW_baseline_families", True),
        (STAGE_W_OUTPUT / "tables" / "ppo_final_pool_320_industrial_efficiency_proxy_summary.csv", "stageW_efficiency_proxy_summary", True),
        (STAGE_W_OUTPUT / "tables" / "ppo_final_pool_320_efficiency_proxy_vs_teacher_metrics.csv", "stageW_efficiency_proxy_vs_teacher", True),
        (STAGE_W_OUTPUT / "stageW_final_ppo_pool_320_analysis_manifest.json", "stageW_manifest", True),
        (STAGE_W_DOCS / "PPO_FINAL_POOL_320_STAGEW_RANKING_AND_COMPARISON_REPORT.md", "stageW_report", True),
        (STAGE_W_DOCS / "PPO_FINAL_POOL_320_STAGEW_CLAIM_BOUNDARY.md", "stageW_claim_boundary", True),
    ]
    for src, role, required in stage_w_files:
        safe_copy(src, PACKAGE_ROOT / "02_STAGE_W_ANALYSIS", role, copied, missing, required=required)

    # Stage history summaries.
    history_files = [
        ROOT / "docs" / "stage3_ppo_rl_lam_fea_addendum_v01" / "PPO_FINAL_EVIDENCE_FREEZE_REPORT.md",
        ROOT / "docs" / "stage3_ppo_rl_lam_fea_addendum_v01" / "PPO_FINAL_CLAIM_BOUNDARY.md",
        ROOT / "docs" / "stage3_ppo_rl_lam_fea_addendum_v01" / "PPO_MANUSCRIPT_INTEGRATION_FINAL_MEMO.md",
        ROOT / "docs" / "stage3_ppo_rl_lam_fea_addendum_v02_targeted_N24_N40" / "PPO_V02K2_STAGEM_ODB_TEACHER_METRIC_EXTRACTION_REPORT.md",
        ROOT / "docs" / "stage3_ppo_rl_lam_fea_addendum_v02_targeted_N24_N40" / "PPO_V02K2_STAGEN_TEACHER_METRIC_RANKING_REPORT.md",
        ROOT / "docs" / "stage3_ppo_rl_lam_fea_addendum_v02_targeted_N24_N40" / "PPO_V02K2_STAGEN_CLAIM_BOUNDARY.md",
        ROOT / "docs" / "stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40" / "PPO_V03_LEX_PRIMARY_CANDIDATE_GENERATION_REPORT.md",
        ROOT / "docs" / "stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40" / "PPO_V03_STAGER_ODB_TEACHER_METRIC_EXTRACTION_REPORT.md",
        ROOT / "docs" / "stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40" / "PPO_V03_STAGES_TEACHER_METRIC_RANKING_REPORT.md",
        ROOT / "docs" / "stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40" / "PPO_V03_STAGES_CLAIM_BOUNDARY.md",
        ROOT / "docs" / "stage3_ppo_final_expansion_224_to_320" / "PPO_FINAL_EXPANSION_224_CANDIDATE_GENERATION_REPORT.md",
        ROOT / "docs" / "stage3_ppo_final_expansion_224_to_320" / "PPO_FINAL_EXPANSION_STAGEU_CAE_INP_HANDOFF_REPORT.md",
        ROOT / "docs" / "stage3_ppo_final_expansion_224_to_320" / "PPO_FINAL_EXPANSION_STAGEV_ODB_TEACHER_METRIC_EXTRACTION_REPORT.md",
        ROOT / "docs" / "stage3_ppo_final_expansion_224_to_320" / "PPO_FINAL_EXPANSION_STAGEU_CLAIM_BOUNDARY.md",
        ROOT / "docs" / "stage3_ppo_final_expansion_224_to_320" / "PPO_FINAL_EXPANSION_224_CLAIM_BOUNDARY.md",
    ]
    for src in history_files:
        safe_copy(src, PACKAGE_ROOT / "03_STAGE_HISTORY_SUMMARIES", "stage_history_summary_or_boundary", copied, missing, required=False)

    # Core reference data.
    core_files = [
        RUN78_OUTPUT / "FROZEN_stage3_native_combined552_teacher_dataset.csv",
        RUN78_OUTPUT / "FROZEN_stage3_native_combined552_summary.csv",
        RUN78_OUTPUT / "FROZEN_stage3_native_combined552_file_hashes.csv",
        ROOT / "docs" / "stage3_run_78_final_evidence_freeze_package" / "STAGE3_FINAL_EVIDENCE_FREEZE_REPORT.md",
        ROOT / "docs" / "stage3_run_78_final_evidence_freeze_package" / "STAGE3_CLAIM_BOUNDARY.md",
    ]
    for src in core_files:
        safe_copy(src, PACKAGE_ROOT / "04_CORE_REFERENCE_DATA", "core_reference_data_or_report", copied, missing, required=(src.name == "FROZEN_stage3_native_combined552_teacher_dataset.csv"))

    # Search fallback for optional Run78/core support files.
    wanted_names = {item.name for item in core_files if not item.exists()}
    if wanted_names:
        for base in [ROOT / "outputs", ROOT / "docs"]:
            if not base.exists():
                continue
            for path in base.rglob("*"):
                if path.is_file() and path.name in wanted_names and path.suffix.lower() not in EXCLUDED_EXTENSIONS:
                    safe_copy(path, PACKAGE_ROOT / "04_CORE_REFERENCE_DATA", "core_reference_search_fallback", copied, missing, required=False)


def write_writing_briefs(copied: list[dict]) -> dict[str, str]:
    brief = PACKAGE_ROOT / "08_WRITING_BRIEF"
    readme = brief / "CHATGPT_PROJECT_MASTER_README.md"
    one_page = brief / "PPO_FINAL_320_ONE_PAGE_SUMMARY.md"
    safe_para = brief / "PPO_FINAL_320_SAFE_MANUSCRIPT_PARAGRAPH.md"
    do_not = brief / "PPO_FINAL_320_DO_NOT_CLAIM.md"
    terms = brief / "PPO_FINAL_320_TERMINOLOGY_GUIDE.md"

    readme.write_text(
        """# ChatGPT Project Master README

## What This Package Is

This is a compact evidence package for manuscript writing and claim checking around the final 320-case PPO addendum for RL-LAM-ScanOpt.

## How To Use It

Start with:

1. `08_WRITING_BRIEF/PPO_FINAL_320_ONE_PAGE_SUMMARY.md`
2. `08_WRITING_BRIEF/PPO_FINAL_320_SAFE_MANUSCRIPT_PARAGRAPH.md`
3. `01_FINAL_STAGE_X_FREEZE/PPO_FINAL_POOL_320_EVIDENCE_FREEZE_REPORT.md`
4. `01_FINAL_STAGE_X_FREEZE/PPO_FINAL_POOL_320_FINAL_CLAIM_BOUNDARY.md`
5. `02_STAGE_W_ANALYSIS/PPO_FINAL_POOL_320_STAGEW_RANKING_AND_COMPARISON_REPORT.md`

Use CSV tables only when checking exact numeric claims.

## Final PPO Conclusion

A 320-case PPO-generated scan-order pool was teacher-metric extracted using Abaqus. PPO produced legal, executable and independently teacher-evaluated scan orders with bounded small-N competitiveness and SurfaceT-related signals, but produced 0 new records against native combined552 and did not outperform the mature surrogate-assisted optimiser.

## Safe Claims

- 320 PPO-generated scan orders were teacher-metric extracted.
- PPO generated legal and executable scan-order permutations.
- PPO showed bounded small-N competitiveness.
- PPO showed SurfaceT-related signals but not U2/lex dominance.
- Final physical claims are based on Abaqus teacher metrics, not surrogate scores.

## Unsafe Claims

- PPO beat combined552 best.
- PPO outperformed the mature surrogate-assisted optimiser.
- PPO solved N24/N40 scan-order optimisation.
- PPO demonstrated experimentally validated industrial-efficiency improvement.
- This was online Abaqus-in-the-loop PPO.

## Manuscript Placement

Use PPO as a policy-gradient evidence-chain addendum in Methods/Results/Discussion. Do not present PPO as the best optimiser in the study.

## Distinctions

- Mature surrogate-assisted optimiser: the stronger reference optimisation loop represented by combined552.
- Surrogate-trained PPO policy generation: PPO trained on surrogate rewards to generate scan-order candidates.
- Abaqus teacher metrics: independent finite-element teacher evaluation used for final physical claims.
- Experimental validation: not performed here.
""",
        encoding="utf-8",
    )
    one_page.write_text(
        """# PPO Final 320 One-Page Summary

## Pool Composition

- Total PPO teacher-metric-extracted cases: 320
- v01: 32
- v02K2: 32
- v03: 32
- final expansion: 224
- N12: 40
- N16: 40
- N24: 120
- N40: 120

## Stage Chain

Stage A established PPO environment foundation. Stage B trained a surrogate reward model. Stage C trained PPO v01. Stage D/E/G/H/I/J completed v01 candidate generation, handoff, teacher extraction, ranking, freeze and evidence strengthening. Stage K/K2 built targeted v02 for N24/N40. Stage P/R/S built and ranked v03. Stage T generated final 224 expansion candidates. Stage V extracted final expansion teacher metrics. Stage W ranked the cumulative 320 pool. Stage X froze final evidence.

## Main Results

- New records vs combined552: 0
- Primary top25-any count: 106
- Equal-budget bootstrap interpretation: weak
- Best primary lex N12: rank 6
- Best primary lex N16: rank 2
- Best primary lex N24: rank 114
- Best primary lex N40: rank 147

## Final Interpretation

PPO is evidence for feasible policy-gradient scan-order generation under teacher evaluation. It is not evidence that PPO surpassed the mature surrogate-assisted optimiser.
""",
        encoding="utf-8",
    )
    safe_para.write_text(
        """A final 320-case PPO-generated scan-order pool was independently evaluated using Abaqus teacher simulations across N12, N16, N24 and N40. The PPO pool demonstrated legal and executable policy generation and provided a large-scale teacher-metric-extracted evidence base for surrogate-trained policy-gradient scan-order optimisation. However, no PPO-generated candidate produced a new best record relative to the mature combined552 surrogate-assisted reference, and equal-budget bootstrap comparison indicated weak global enrichment. These results support PPO as a feasible policy-generation mechanism with bounded teacher-validated competitiveness, rather than as a replacement for the mature surrogate-assisted optimiser.
""",
        encoding="utf-8",
    )
    do_not.write_text(
        """# PPO Final 320 Do Not Claim

- PPO beat combined552 best.
- PPO outperformed mature surrogate-assisted optimiser.
- PPO produced the strongest scan orders.
- PPO solved N24/N40 scan-order optimisation.
- PPO demonstrated experimentally validated industrial-efficiency improvement.
- Surrogate score alone predicts physical quality.
- This was online Abaqus-in-the-loop PPO.
- SurfaceT-only enrichment proves U2/lex dominance.
""",
        encoding="utf-8",
    )
    terms.write_text(
        """# PPO Final 320 Terminology Guide

- teacher-metric-extracted: Abaqus-derived teacher metrics were extracted for a case.
- teacher-evaluated: a case has finite-element teacher metrics available.
- teacher-validated: use cautiously; here it means teacher metrics were extracted, not that a candidate is superior.
- surrogate reward model: supervised emulator trained from teacher-labelled data and used for PPO reward.
- mature surrogate-assisted optimiser: the stronger combined552 reference optimisation evidence.
- MaskablePPO policy generation: action-masked PPO generating legal scan-order permutations.
- combined552 reference: native Stage 3 frozen teacher-labelled reference dataset.
- top-k competitiveness: entering top10/top25 regions under defined metrics/ranks.
- equal-budget bootstrap: comparison against same-size draws from existing teacher-labelled reference distribution.
- bounded no-new-records: teacher-evaluated evidence exists, but no new best records were found.
- SurfaceT signal vs U2/lex dominance: SurfaceT enrichment does not imply primary U2->PEEQ->SurfaceT lexicographic dominance.
- industrial-efficiency proxy: sequence descriptor, not physically validated efficiency.
""",
        encoding="utf-8",
    )
    paths = {
        "master_readme": str(readme),
        "one_page_summary": str(one_page),
        "safe_paragraph": str(safe_para),
        "do_not_claim": str(do_not),
        "terminology_guide": str(terms),
    }
    for role, path_str in paths.items():
        path = Path(path_str)
        copied.append(
            {
                "relative_path": str(path.relative_to(PACKAGE_ROOT)),
                "source_path": "generated",
                "file_role": f"generated_{role}",
                "file_size_bytes": path.stat().st_size,
                "extension": path.suffix.lower(),
                "copied_timestamp": datetime.now().isoformat(timespec="seconds"),
            }
        )
    return paths


def write_plots_index(copied: list[dict]) -> dict[str, object]:
    plots_dir = STAGE_W_OUTPUT / "plots"
    target_dir = PACKAGE_ROOT / "09_OPTIONAL_PLOTS_INDEX_ONLY"
    plot_files = sorted([p for p in plots_dir.glob("*") if p.is_file()]) if plots_dir.exists() else []
    total_size = sum(p.stat().st_size for p in plot_files)
    copied_plots = False
    rows = []
    if total_size <= PLOT_COPY_LIMIT_BYTES:
        copied_plots = True
        (target_dir / "plots").mkdir(parents=True, exist_ok=True)
    for p in plot_files:
        purpose = p.stem.replace("_", " ")
        rows.append(f"| {p.name} | `{p}` | {p.stat().st_size} | {purpose} |")
        if copied_plots and p.suffix.lower() not in EXCLUDED_EXTENSIONS:
            dst = target_dir / "plots" / p.name
            shutil.copy2(p, dst)
            copied.append(
                {
                    "relative_path": str(dst.relative_to(PACKAGE_ROOT)),
                    "source_path": str(p),
                    "file_role": "optional_stageW_plot",
                    "file_size_bytes": dst.stat().st_size,
                    "extension": dst.suffix.lower(),
                    "copied_timestamp": datetime.now().isoformat(timespec="seconds"),
                }
            )
    index = target_dir / "PLOTS_AVAILABLE_INDEX.md"
    index.write_text(
        "# Plots Available Index\n\n"
        f"Stage W plots directory: `{plots_dir}`\n\n"
        f"Total plot folder size: {total_size} bytes\n\n"
        f"Plots copied into package: {copied_plots}\n\n"
        "| filename | full path | file size bytes | inferred purpose |\n"
        "|---|---|---:|---|\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )
    copied.append(
        {
            "relative_path": str(index.relative_to(PACKAGE_ROOT)),
            "source_path": "generated",
            "file_role": "plots_index",
            "file_size_bytes": index.stat().st_size,
            "extension": index.suffix.lower(),
            "copied_timestamp": datetime.now().isoformat(timespec="seconds"),
        }
    )
    return {"plots_dir": str(plots_dir), "plot_count": len(plot_files), "total_size": total_size, "copied": copied_plots}


def scan_excluded_extensions() -> list[Path]:
    bad = []
    for path in PACKAGE_ROOT.rglob("*"):
        if path.is_file() and path.suffix.lower() in EXCLUDED_EXTENSIONS:
            bad.append(path)
    return bad


def write_inventory_hashes_manifest(copied: list[dict], missing: list[dict], writing_paths: dict[str, str], plot_info: dict[str, object]) -> dict[str, Path | int]:
    hash_manifest_dir = PACKAGE_ROOT / "07_HASHES_AND_MANIFESTS"
    inventory_path = hash_manifest_dir / "CHATGPT_UPLOAD_FILE_INVENTORY.csv"
    hashes_path = hash_manifest_dir / "CHATGPT_UPLOAD_FILE_HASHES.csv"
    missing_path = hash_manifest_dir / "CHATGPT_UPLOAD_MISSING_FILES.csv"
    manifest_path = hash_manifest_dir / "CHATGPT_UPLOAD_MANIFEST.json"

    # Inventory before adding manifest/report is updated after report generation.
    all_files = sorted([p for p in PACKAGE_ROOT.rglob("*") if p.is_file()])
    inv_rows = []
    for p in all_files:
        rec = next((r for r in copied if r["relative_path"] == str(p.relative_to(PACKAGE_ROOT))), None)
        inv_rows.append(
            {
                "relative_path": str(p.relative_to(PACKAGE_ROOT)),
                "source_path": rec["source_path"] if rec else "generated",
                "file_role": rec["file_role"] if rec else "generated_or_packaging_metadata",
                "file_size_bytes": p.stat().st_size,
                "extension": p.suffix.lower(),
                "copied_timestamp": rec["copied_timestamp"] if rec else datetime.now().isoformat(timespec="seconds"),
            }
        )
    with inventory_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["relative_path", "source_path", "file_role", "file_size_bytes", "extension", "copied_timestamp"])
        writer.writeheader()
        writer.writerows(inv_rows)

    hash_rows = []
    for p in sorted([p for p in PACKAGE_ROOT.rglob("*") if p.is_file()]):
        hash_rows.append({"relative_path": str(p.relative_to(PACKAGE_ROOT)), "sha256": sha256_file(p), "file_size_bytes": p.stat().st_size})
    with hashes_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["relative_path", "sha256", "file_size_bytes"])
        writer.writeheader()
        writer.writerows(hash_rows)

    with missing_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["source_path", "intended_role", "required", "reason"])
        writer.writeheader()
        writer.writerows(missing)

    manifest = {
        "package_name": PACKAGE_NAME,
        "package_root": str(PACKAGE_ROOT),
        "created_timestamp": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(ROOT),
        "branch": git_branch(),
        "final_verdict": FINAL_VERDICT,
        "final_ppo_pool_count": 320,
        "by_N_counts": {"N12": 40, "N16": 40, "N24": 120, "N40": 120},
        "new_records_vs_combined552": 0,
        "bootstrap_interpretation": "weak",
        "writing_paths": writing_paths,
        "plot_info": plot_info,
        "no_large_solver_files_included": True,
        "no_ODB": True,
        "no_CAE": True,
        "no_INP": True,
        "no_solver_outputs": True,
        "no_commit_or_push": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Recompute inventory/hash once to include manifest itself.
    all_files = sorted([p for p in PACKAGE_ROOT.rglob("*") if p.is_file()])
    with inventory_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["relative_path", "source_path", "file_role", "file_size_bytes", "extension", "copied_timestamp"])
        writer.writeheader()
        for p in all_files:
            rec = next((r for r in copied if r["relative_path"] == str(p.relative_to(PACKAGE_ROOT))), None)
            writer.writerow(
                {
                    "relative_path": str(p.relative_to(PACKAGE_ROOT)),
                    "source_path": rec["source_path"] if rec else "generated",
                    "file_role": rec["file_role"] if rec else "generated_or_packaging_metadata",
                    "file_size_bytes": p.stat().st_size,
                    "extension": p.suffix.lower(),
                    "copied_timestamp": rec["copied_timestamp"] if rec else datetime.now().isoformat(timespec="seconds"),
                }
            )
    with hashes_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["relative_path", "sha256", "file_size_bytes"])
        writer.writeheader()
        for p in all_files:
            writer.writerow({"relative_path": str(p.relative_to(PACKAGE_ROOT)), "sha256": sha256_file(p), "file_size_bytes": p.stat().st_size})

    return {
        "inventory_path": inventory_path,
        "hashes_path": hashes_path,
        "missing_path": missing_path,
        "manifest_path": manifest_path,
        "file_count": len(all_files),
        "package_size_bytes": sum(p.stat().st_size for p in all_files),
    }


def write_packaging_report(summary: dict[str, Path | int], missing: list[dict], excluded: list[Path], writing_paths: dict[str, str]) -> Path:
    report = PACKAGE_ROOT / "00_README" / "PACKAGING_REPORT.md"
    safe_para = Path(writing_paths["safe_paragraph"]).read_text(encoding="utf-8").strip()
    report.write_text(
        f"""# Packaging Report

## Purpose

Create a compact ChatGPT Project upload folder for final Stage 3 PPO evidence and manuscript writing.

## Package Root

`{PACKAGE_ROOT}`

## ZIP Archive Path

`{ZIP_PATH}`

## Included File Count

{summary['file_count']}

## Total Package Size

{summary['package_size_bytes']} bytes

## Key Files Copied

- Stage X frozen final PPO pool tables.
- Stage W ranking and comparison tables.
- Stage history reports for v01/v02K2/v03/final expansion.
- Native combined552 reference dataset.
- Manuscript writing briefs and claim boundaries.

## Missing Optional Files

See `{summary['missing_path']}`. Missing optional files do not invalidate the package.

## Excluded File Types

`.odb`, `.cae`, `.inp`, `.jnl`, `.sim`, `.sta`, `.dat`, `.msg`, `.lck`, and inner `.zip` files.

Excluded files found after package assembly: {len(excluded)}

## Final PPO Conclusion

A 320-case PPO-generated scan-order pool was teacher-metric extracted using Abaqus. It provides a policy-generation evidence chain, but it does not beat combined552 best records or replace the mature surrogate-assisted optimiser.

## Safe Claim Paragraph

{safe_para}

## Upload Instruction For ChatGPT Project

Upload the folder or ZIP archive into the ChatGPT Project. Start review from `08_WRITING_BRIEF/CHATGPT_PROJECT_MASTER_README.md`.

## Final Verdict

{FINAL_VERDICT}
""",
        encoding="utf-8",
    )
    return report


def create_zip() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with ZipFile(ZIP_PATH, "w", compression=ZIP_DEFLATED) as zf:
        for path in sorted(PACKAGE_ROOT.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(PACKAGE_ROOT.parent))


def main() -> None:
    copied: list[dict] = []
    missing: list[dict] = []
    clean_package_root()
    copy_named_files(copied, missing)
    writing_paths = write_writing_briefs(copied)
    plot_info = write_plots_index(copied)
    excluded = scan_excluded_extensions()
    if excluded:
        raise RuntimeError("Excluded files found in package before ZIP: " + "; ".join(str(p) for p in excluded[:20]))
    summary = write_inventory_hashes_manifest(copied, missing, writing_paths, plot_info)
    report_path = write_packaging_report(summary, missing, excluded, writing_paths)

    # Rebuild inventory/hash/manifest after adding packaging report.
    copied.append(
        {
            "relative_path": str(report_path.relative_to(PACKAGE_ROOT)),
            "source_path": "generated",
            "file_role": "packaging_report",
            "file_size_bytes": report_path.stat().st_size,
            "extension": report_path.suffix.lower(),
            "copied_timestamp": datetime.now().isoformat(timespec="seconds"),
        }
    )
    summary = write_inventory_hashes_manifest(copied, missing, writing_paths, plot_info)
    excluded = scan_excluded_extensions()
    if excluded:
        raise RuntimeError("Excluded files found in package before ZIP: " + "; ".join(str(p) for p in excluded[:20]))
    create_zip()

    result = {
        "branch": git_branch(),
        "package_root": str(PACKAGE_ROOT),
        "zip_archive": str(ZIP_PATH),
        "file_count": int(summary["file_count"]),
        "total_package_size_bytes": int(summary["package_size_bytes"]),
        "excluded_large_files_included": False,
        "master_readme": writing_paths["master_readme"],
        "one_page_summary": writing_paths["one_page_summary"],
        "safe_paragraph": writing_paths["safe_paragraph"],
        "do_not_claim": writing_paths["do_not_claim"],
        "terminology_guide": writing_paths["terminology_guide"],
        "inventory": str(summary["inventory_path"]),
        "hashes": str(summary["hashes_path"]),
        "manifest": str(summary["manifest_path"]),
        "missing_optional": str(summary["missing_path"]),
        "packaging_report": str(report_path),
        "zip_size_bytes": ZIP_PATH.stat().st_size,
        "final_packaging_verdict": FINAL_VERDICT,
        "no_Abaqus": True,
        "no_ODB": True,
        "no_solver": True,
        "no_training": True,
        "no_candidate_generation": True,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
