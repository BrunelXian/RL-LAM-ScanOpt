"""Stage I final PPO evidence freeze and manuscript-support package.

This script is documentation/evidence-freeze only. It reads existing PPO
addendum artifacts, copies summary tables, computes SHA256 hashes, and writes
manuscript-facing reports. It does not run Abaqus, open ODB files, run solver,
generate CAE/INP/JNL, train models, or generate candidates.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
BRANCH_FALLBACK = "stage3-variable-n-graph-pointer-init-v01"
NAMESPACE = "stage3_ppo_rl_lam_fea_addendum_v01"

OUT_ROOT = PROJECT_ROOT / "outputs" / NAMESPACE / "stageI_final_ppo_evidence_freeze"
FROZEN_TABLES_DIR = OUT_ROOT / "frozen_tables"
HASHES_DIR = OUT_ROOT / "hashes"
MANUSCRIPT_TABLES_DIR = OUT_ROOT / "manuscript_tables"
REPORTS_DIR = OUT_ROOT / "reports"
PLOTS_INDEX_DIR = OUT_ROOT / "plots_index"
DOCS_DIR = PROJECT_ROOT / "docs" / NAMESPACE

SURROGATE_MODEL = PROJECT_ROOT / "outputs" / NAMESPACE / "surrogate_reward_model" / "models" / "ppo_surrogate_reward_model_best.joblib"
SURROGATE_REPORT = DOCS_DIR / "PPO_SURROGATE_REWARD_MODEL_REPORT.md"
PPO_CHECKPOINT = PROJECT_ROOT / "outputs" / NAMESPACE / "ppo_training" / "checkpoints" / "maskable_ppo_lam_scan_order_final.zip"
PPO_CONFIG = PROJECT_ROOT / "outputs" / NAMESPACE / "ppo_training" / "ppo_training_config.json"
PPO_TRAINING_SUMMARY = PROJECT_ROOT / "outputs" / NAMESPACE / "ppo_training" / "reports" / "ppo_training_summary.json"
PPO_PARAMETER_COUNT = PROJECT_ROOT / "outputs" / NAMESPACE / "ppo_training" / "reports" / "ppo_parameter_count.json"
PPO_STAGE_C_REPORT = DOCS_DIR / "PPO_TRAINING_STAGE_C_REPORT.md"
PPO_SELECTED_BATCH = PROJECT_ROOT / "outputs" / NAMESPACE / "ppo_candidate_generation" / "selected_batch32" / "ppo_policy_only_candidate_batch32.csv"
PPO_STAGE_D_REPORT = DOCS_DIR / "PPO_POLICY_ONLY_CANDIDATE_GENERATION_STAGE_D_REPORT.md"
PPO_STAGE_E_REPORT = DOCS_DIR / "PPO_STAGEE_CAE_INP_HANDOFF_REPORT.md"
PPO_STAGE_F_REPORT = DOCS_DIR / "PPO_STAGEF_SOLVER_EXECUTION_REPORT.md"
PPO_STAGE_G_REPORT = DOCS_DIR / "PPO_STAGEG_ODB_TEACHER_METRIC_EXTRACTION_REPORT.md"
PPO_STAGE_H_REPORT = DOCS_DIR / "PPO_STAGEH_TEACHER_METRIC_RANKING_REPORT.md"
PPO_STAGE_H_CLAIM_BOUNDARY = DOCS_DIR / "PPO_STAGEH_CLAIM_BOUNDARY.md"

STAGE_G_TEACHER_METRICS = PROJECT_ROOT / "outputs" / NAMESPACE / "stageG_odb_teacher_metrics" / "tables" / "stageG_ppo_batch32_teacher_metrics.csv"
STAGE_H_ROOT = PROJECT_ROOT / "outputs" / NAMESPACE / "stageH_teacher_metric_ranking"
STAGE_H_FULL_RANKING = STAGE_H_ROOT / "tables" / "ppo_batch32_teacher_metric_ranking_full.csv"
STAGE_H_SUMMARY_BY_N = STAGE_H_ROOT / "tables" / "ppo_batch32_summary_by_N.csv"
STAGE_H_NEW_RECORDS = STAGE_H_ROOT / "tables" / "ppo_batch32_new_record_candidates.csv"
STAGE_H_TOPK = STAGE_H_ROOT / "tables" / "ppo_batch32_topk_candidates.csv"
STAGE_H_ALIGNMENT_JSON = STAGE_H_ROOT / "tables" / "ppo_surrogate_vs_teacher_alignment_summary.json"
STAGE_H_ALIGNMENT_CSV = STAGE_H_ROOT / "tables" / "ppo_surrogate_vs_teacher_alignment.csv"
STAGE_H_RECOVERY = STAGE_H_ROOT / "tables" / "ppo_recovery_anchor_duplicate_audit.csv"
STAGE_H_ANALYSIS_DATASET = STAGE_H_ROOT / "tables" / "combined552_plus_ppo32_analysis_dataset.csv"

HASHES_CSV = HASHES_DIR / "FROZEN_PPO_file_hashes.csv"
HASHES_JSON = HASHES_DIR / "FROZEN_PPO_file_hashes.json"
EVIDENCE_CHAIN_CSV = MANUSCRIPT_TABLES_DIR / "ppo_evidence_chain_table.csv"
EVIDENCE_CHAIN_MD = DOCS_DIR / "PPO_EVIDENCE_CHAIN_TABLE.md"
PERFORMANCE_SUMMARY_CSV = MANUSCRIPT_TABLES_DIR / "ppo_performance_summary_for_manuscript.csv"
PERFORMANCE_SUMMARY_MD = DOCS_DIR / "PPO_PERFORMANCE_SUMMARY_FOR_MANUSCRIPT.md"
CLAIM_SUPPORT_CSV = MANUSCRIPT_TABLES_DIR / "ppo_claim_support_table.csv"
CLAIM_SUPPORT_MD = DOCS_DIR / "PPO_CLAIM_SUPPORT_TABLE.md"
FINAL_CLAIM_BOUNDARY = DOCS_DIR / "PPO_FINAL_CLAIM_BOUNDARY.md"
MANUSCRIPT_MEMO = DOCS_DIR / "PPO_MANUSCRIPT_INTEGRATION_FINAL_MEMO.md"
FINAL_REPORT = DOCS_DIR / "PPO_FINAL_EVIDENCE_FREEZE_REPORT.md"
ARA_INDEX = OUT_ROOT / "ARA_PPO_FINAL_EVIDENCE_INDEX.md"
MANIFEST = OUT_ROOT / "stageI_ppo_final_evidence_freeze_manifest.json"
PLOTS_INDEX_CSV = PLOTS_INDEX_DIR / "FROZEN_PPO_stageH_plots_index.csv"
PLOTS_INDEX_MD = PLOTS_INDEX_DIR / "FROZEN_PPO_stageH_plots_index.md"

VERDICT = "PASS_PPO_FINAL_EVIDENCE_FREEZE_TEACHER_VALIDATED_COMPETITIVE_BOUNDED"


def ensure_dirs() -> None:
    for directory in [OUT_ROOT, FROZEN_TABLES_DIR, HASHES_DIR, MANUSCRIPT_TABLES_DIR, REPORTS_DIR, PLOTS_INDEX_DIR, DOCS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def git_branch() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "branch", "--show-current"],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() or BRANCH_FALLBACK
    except Exception:
        return BRANCH_FALLBACK


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_file(src: Path, dst: Path) -> Path:
    if not src.exists():
        raise FileNotFoundError(str(src))
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    def cell(value: Any) -> str:
        text = "" if value is None else str(value)
        return text.replace("|", "\\|").replace("\n", "<br>")

    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(cell(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines) + "\n"


def freeze_tables() -> dict[str, str]:
    copies = {
        "batch32_selected_candidates": (PPO_SELECTED_BATCH, "FROZEN_PPO_policy_only_candidate_batch32.csv"),
        "batch32_teacher_metrics": (STAGE_G_TEACHER_METRICS, "FROZEN_PPO_batch32_teacher_metrics.csv"),
        "batch32_teacher_metric_ranking_full": (STAGE_H_FULL_RANKING, "FROZEN_PPO_batch32_teacher_metric_ranking_full.csv"),
        "batch32_summary_by_N": (STAGE_H_SUMMARY_BY_N, "FROZEN_PPO_batch32_summary_by_N.csv"),
        "batch32_new_record_candidates": (STAGE_H_NEW_RECORDS, "FROZEN_PPO_batch32_new_record_candidates.csv"),
        "batch32_topk_candidates": (STAGE_H_TOPK, "FROZEN_PPO_batch32_topk_candidates.csv"),
        "surrogate_vs_teacher_alignment": (STAGE_H_ALIGNMENT_CSV, "FROZEN_PPO_surrogate_vs_teacher_alignment.csv"),
        "surrogate_vs_teacher_alignment_summary": (STAGE_H_ALIGNMENT_JSON, "FROZEN_PPO_surrogate_vs_teacher_alignment_summary.json"),
        "recovery_anchor_duplicate_audit": (STAGE_H_RECOVERY, "FROZEN_PPO_recovery_anchor_duplicate_audit.csv"),
        "combined552_plus_ppo32_analysis_dataset": (STAGE_H_ANALYSIS_DATASET, "FROZEN_PPO_combined552_plus_ppo32_analysis_dataset.csv"),
    }
    outputs: dict[str, str] = {}
    for key, (src, name) in copies.items():
        outputs[key] = str(copy_file(src, FROZEN_TABLES_DIR / name))
    return outputs


def read_stage_h_summaries() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    by_n = pd.read_csv(STAGE_H_SUMMARY_BY_N)
    topk = pd.read_csv(STAGE_H_TOPK)
    new_records = pd.read_csv(STAGE_H_NEW_RECORDS)
    alignment = json.loads(STAGE_H_ALIGNMENT_JSON.read_text(encoding="utf-8"))
    recovery = pd.read_csv(STAGE_H_RECOVERY)
    return by_n, topk, new_records, alignment, recovery


def create_evidence_chain_table() -> None:
    rows = [
        {
            "stage": "A",
            "evidence": "PPO scan-order environment foundation and dependency/data preflight",
            "key output": str(PROJECT_ROOT / "outputs" / NAMESPACE / "ppo_addendum_foundation_manifest.json"),
            "verdict": "PASS_PPO_ADDENDUM_DEPENDENCIES_AND_DATA_READY",
            "manuscript-safe claim": "A PPO-compatible masked scan-order environment was specified for native N12/N16/N24/N40.",
            "claim boundary": "No PPO training or FEA validation occurred in Stage A.",
        },
        {
            "stage": "B",
            "evidence": "FEA teacher-labelled combined552 used to train supervised terminal reward emulator",
            "key output": str(SURROGATE_MODEL),
            "verdict": "PASS_PPO_SURROGATE_REWARD_MODEL_READY_FOR_PPO_TRAINING",
            "manuscript-safe claim": "A surrogate terminal-reward model was trained on native combined552 teacher-labelled data.",
            "claim boundary": "The surrogate is not the physical teacher.",
        },
        {
            "stage": "C",
            "evidence": "MaskablePPO + MlpPolicy trained in frozen surrogate reward environment",
            "key output": str(PPO_CHECKPOINT),
            "verdict": "PASS_PPO_TRAINING_READY_FOR_POLICY_ONLY_CANDIDATE_GENERATION",
            "manuscript-safe claim": "A MaskablePPO policy was trained in a surrogate reward environment.",
            "claim boundary": "This was not online Abaqus RL.",
        },
        {
            "stage": "D",
            "evidence": "PPO-only candidates generated from frozen PPO checkpoint",
            "key output": str(PPO_SELECTED_BATCH),
            "verdict": "PASS_PPO_POLICY_ONLY_BATCH32_READY_FOR_STAGE_E_TEACHER_VALIDATION_HANDOFF",
            "manuscript-safe claim": "PPO-only scan-order candidates were generated by checkpoint inference.",
            "claim boundary": "Surrogate ranking was selection support only; no teacher validation yet.",
        },
        {
            "stage": "E",
            "evidence": "PPO batch32 converted to Abaqus CAE/INP cases",
            "key output": str(PPO_STAGE_E_REPORT),
            "verdict": "PASS_STAGEE_PPO_BATCH32_READY_FOR_USER_CONTROLLED_ABAQUS_TEACHER_VALIDATION",
            "manuscript-safe claim": "Thirty-two PPO-generated scan orders were converted into Abaqus teacher-validation cases.",
            "claim boundary": "No solver metrics were produced in Stage E.",
        },
        {
            "stage": "F",
            "evidence": "Abaqus solver completed for 32/32 PPO cases",
            "key output": str(PPO_STAGE_F_REPORT),
            "verdict": "WARNING_STAGEG_PPO_BATCH32_SOLVER_COMPLETION_WITH_NONFATAL_WARNINGS",
            "manuscript-safe claim": "All 32 PPO cases completed with nonzero ODB outputs despite nonfatal warnings.",
            "claim boundary": "Completion alone is not performance evidence.",
        },
        {
            "stage": "G",
            "evidence": "ODB teacher metrics extracted for 32/32 PPO cases",
            "key output": str(STAGE_G_TEACHER_METRICS),
            "verdict": "PASS_STAGEG_PPO_BATCH32_TEACHER_METRICS_EXTRACTED",
            "manuscript-safe claim": "Teacher metrics were extracted for all PPO cases.",
            "claim boundary": "Metric extraction does not imply superiority.",
        },
        {
            "stage": "H",
            "evidence": "PPO ranking against native combined552 completed",
            "key output": str(STAGE_H_FULL_RANKING),
            "verdict": "PASS_STAGEH_PPO_BATCH32_TEACHER_VALIDATED_AND_COMPETITIVE",
            "manuscript-safe claim": "PPO batch32 was teacher-validated and small-N/top-k competitive, with no new combined552 records.",
            "claim boundary": "No global-best or all-N superiority claim is supported.",
        },
    ]
    write_csv(EVIDENCE_CHAIN_CSV, rows)
    EVIDENCE_CHAIN_MD.write_text("# PPO Evidence Chain Table\n\n" + markdown_table(rows, list(rows[0].keys())), encoding="utf-8")


def create_performance_summary_table(by_n: pd.DataFrame) -> None:
    topk_known = {12: 5, 16: 4, 24: 3, 40: 0}
    interpretations = {
        12: "Small-N top-k competitive; best lexicographic reference rank 6.",
        16: "Small-N top-k competitive; best lexicographic reference rank 2.",
        24: "Some metric-level top-k hits, but weak primary lexicographic position.",
        40: "Not competitive under primary metrics; only diagnostic Mises top-k hits were observed.",
    }
    rows = []
    for _, row in by_n.iterrows():
        n = int(row["n"])
        rows.append(
            {
                "N": n,
                "PPO count": int(row["ppo_count"]),
                "best PPO lex candidate": row["best_ppo_lex_strategy_name"],
                "best PPO ref lex rank": int(row["best_ppo_ref_lex_rank"]),
                "top-k count": topk_known[n],
                "new records count": 0,
                "main interpretation": interpretations[n],
            }
        )
    rows.append(
        {
            "N": "All",
            "PPO count": 32,
            "best PPO lex candidate": "N-specific",
            "best PPO ref lex rank": "N-specific",
            "top-k count": 12,
            "new records count": 0,
            "main interpretation": "Teacher-validated and bounded competitive, with competitiveness concentrated in N12/N16.",
        }
    )
    write_csv(PERFORMANCE_SUMMARY_CSV, rows)
    PERFORMANCE_SUMMARY_MD.write_text("# PPO Performance Summary For Manuscript\n\n" + markdown_table(rows, list(rows[0].keys())), encoding="utf-8")


def create_claim_support_table() -> None:
    rows = [
        {
            "claim": "PPO policy was trained in a surrogate reward environment derived from FEA teacher-labelled LDED scan-order data.",
            "support status": "PASS",
            "evidence path": str(PPO_CHECKPOINT),
            "safe wording": "A MaskablePPO policy was trained in a surrogate terminal-reward environment derived from FEA teacher-labelled scan-order data.",
            "unsafe wording": "PPO was trained online in Abaqus.",
        },
        {
            "claim": "PPO-generated scan orders were converted into Abaqus teacher-validation cases.",
            "support status": "PASS",
            "evidence path": str(PPO_STAGE_E_REPORT),
            "safe wording": "Thirty-two PPO-generated scan orders were converted into Abaqus CAE/INP teacher-validation cases.",
            "unsafe wording": "PPO candidates were validated at Stage E.",
        },
        {
            "claim": "PPO-generated scan orders were independently evaluated by Abaqus FEA.",
            "support status": "PASS",
            "evidence path": str(STAGE_G_TEACHER_METRICS),
            "safe wording": "Thirty-two PPO-generated scan orders were independently evaluated by Abaqus FEA.",
            "unsafe wording": "PPO is experimentally validated.",
        },
        {
            "claim": "PPO batch32 was teacher-metric extracted 32/32.",
            "support status": "PASS",
            "evidence path": str(STAGE_G_TEACHER_METRICS),
            "safe wording": "Teacher metrics were extracted for 32/32 PPO candidates.",
            "unsafe wording": "Extraction alone proves PPO superiority.",
        },
        {
            "claim": "PPO showed small-N top-k competitiveness.",
            "support status": "BOUNDED",
            "evidence path": str(STAGE_H_TOPK),
            "safe wording": "PPO achieved bounded small-N top-k competitiveness, especially for N12/N16 lexicographic rankings.",
            "unsafe wording": "PPO dominated all native N.",
        },
        {
            "claim": "PPO did not produce new combined552 records.",
            "support status": "PASS",
            "evidence path": str(STAGE_H_NEW_RECORDS),
            "safe wording": "No PPO candidate beat the prior combined552 best for primary metrics or lexicographic rank.",
            "unsafe wording": "PPO produced a new global best.",
        },
        {
            "claim": "PPO superiority across all native N is not supported.",
            "support status": "PASS",
            "evidence path": str(STAGE_H_SUMMARY_BY_N),
            "safe wording": "The evidence does not support all-N PPO superiority.",
            "unsafe wording": "PPO outperformed all known scan strategies.",
        },
        {
            "claim": "PPO is not online Abaqus RL.",
            "support status": "PASS",
            "evidence path": str(FINAL_CLAIM_BOUNDARY),
            "safe wording": "PPO was trained in a surrogate reward environment and later evaluated by independent Abaqus teacher simulations.",
            "unsafe wording": "Online Abaqus PPO was performed.",
        },
    ]
    write_csv(CLAIM_SUPPORT_CSV, rows)
    CLAIM_SUPPORT_MD.write_text("# PPO Claim Support Table\n\n" + markdown_table(rows, list(rows[0].keys())), encoding="utf-8")


def create_final_claim_boundary() -> None:
    FINAL_CLAIM_BOUNDARY.write_text(
        """# PPO Final Claim Boundary

## Safe Claims

- A MaskablePPO policy was trained in a surrogate terminal-reward environment derived from FEA teacher-labelled LDED scan-order data.
- PPO-only scan-order candidates were generated from the frozen PPO checkpoint.
- Thirty-two PPO-generated scan orders were converted into Abaqus CAE/INP cases.
- All 32 PPO-generated cases completed solver execution and yielded nonzero ODB files.
- Teacher metrics were extracted for 32/32 PPO cases.
- PPO batch32 was teacher-validated and small-N/top-k competitive.
- PPO achieved N12/N16 top-k competitiveness, with best lexicographic reference ranks 6 and 2 respectively.
- PPO produced no new combined552 records.
- N40 PPO performance was not competitive under primary metrics; only diagnostic Mises top-k hits were observed.
- PPO was trained in a surrogate environment, not online Abaqus.

## Unsafe Claims

- PPO outperformed all known scan strategies.
- PPO produced a new global best.
- PPO dominated all native N.
- PPO solved arbitrary-N scan-order optimisation.
- PPO was trained online in Abaqus.
- PPO is experimentally validated.
- PPO is first in the world.

Any "first PPO/RL+LAM+FEA" claim requires a separate literature-priority audit.
""",
        encoding="utf-8",
    )


def create_manuscript_memo() -> None:
    MANUSCRIPT_MEMO.write_text(
        f"""# PPO Manuscript Integration Final Memo

## 1. How This Changes The Current Paper

The PPO addendum upgrades the reinforcement-learning evidence from environment/planning language to a completed, teacher-validated PPO evidence chain: surrogate reward model, MaskablePPO policy training, checkpoint inference, Abaqus case conversion, solver completion, ODB metric extraction, and comparison against native combined552.

## 2. Recommended Title Direction

Use bounded language such as: "Surrogate-trained PPO policy generation with Abaqus teacher validation for LDED scan-order design." Avoid title wording that implies global optimisation or online Abaqus RL.

## 3. Recommended Abstract Wording Boundary

Safe abstract wording: "We trained a MaskablePPO policy in a surrogate terminal-reward environment derived from FEA teacher-labelled scan-order data, generated 32 PPO-only scan-order candidates, and independently evaluated them using Abaqus teacher simulations. The PPO batch achieved bounded small-N top-k competitiveness but produced no new combined552 records."

## 4. Recommended Methods Additions

- PPO surrogate reward environment: describe the terminal sparse reward emulator trained from native combined552 only.
- MaskablePPO policy training: report MaskablePPO + MlpPolicy, 200352 timesteps, and 72937 parameters.
- PPO-only candidate generation: state candidates came from frozen checkpoint inference and were not hand-mutated.
- Abaqus teacher validation: describe CAE/INP generation, solver execution, and ODB metric extraction for 32/32 cases.

## 5. Recommended Results Subsection

Suggested subsection title: "PPO-generated scan orders under Abaqus teacher validation."

Report: 32/32 evaluated, 0 new records, 12 top-k candidates, with N12/N16 competitiveness and limited N40 primary-metric performance.

## 6. Recommended Discussion Wording

The PPO result validates RL policy generation feasibility in this workflow. It does not establish global superiority. The surrogate-to-teacher alignment was weak positive, so independent teacher validation remains necessary. The strongest evidence is small-N top-k competitiveness; N40 remains limited.

## 7. Suggested Figure/Table List

- PPO evidence chain schematic.
- PPO performance summary by N.
- PPO vs combined552 metric distributions.
- Surrogate vs teacher alignment.
- Claim boundary table.

## 8. How To Avoid Overclaiming

Do not claim online Abaqus RL, experimental validation, new global bests, dominance over all native N, arbitrary-N optimisation, or first-in-world status. Use the final claim boundary: `{FINAL_CLAIM_BOUNDARY}`.
""",
        encoding="utf-8",
    )


def create_final_report(by_n: pd.DataFrame, topk: pd.DataFrame, new_records: pd.DataFrame, alignment: dict[str, Any], recovery: pd.DataFrame) -> None:
    best_lines = []
    for _, row in by_n.iterrows():
        best_lines.append(f"- N{int(row['n'])}: `{row['best_ppo_lex_strategy_name']}`, reference lex rank {int(row['best_ppo_ref_lex_rank'])}")
    recovery_note = "Recovery-anchor audit not available."
    if not recovery.empty:
        r = recovery.iloc[0]
        recovery_note = f"`PPOV01_N12_B02_surrogate_top` matched `{r.get('duplicate_order_hash_source_strategy', 'unknown')}` by {r.get('duplicate_source_match_mode', 'unknown')} and is not a novel PPO discovery."
    FINAL_REPORT.write_text(
        f"""# PPO Final Evidence Freeze Report

## 1. Purpose

Freeze the completed PPO + LAM + FEA addendum evidence and provide manuscript-facing claim support.

## 2. Evidence Chain Overview

FEA teacher-labelled native combined552 data -> supervised surrogate terminal reward model -> MaskablePPO training -> PPO checkpoint inference -> PPO-only batch32 -> Abaqus CAE/INP handoff -> solver completion -> ODB teacher metric extraction -> ranking against combined552.

## 3. Frozen Inputs And Outputs

- Frozen table directory: `{FROZEN_TABLES_DIR}`
- Hash table: `{HASHES_CSV}`
- Manifest: `{MANIFEST}`

## 4. Surrogate Reward Model Summary

The surrogate model was trained on native combined552 only, without N32. The selected model was HistGradientBoostingRegressor for `reward_lex_u2_peeq_surfacet`, with validation Spearman 0.8786 and Pearson 0.8863.

## 5. PPO Training Summary

MaskablePPO + MlpPolicy was trained in the surrogate reward environment for 200352 timesteps. Parameter count was 72937. Checkpoint: `{PPO_CHECKPOINT}`.

## 6. PPO Candidate Generation Summary

Stage D generated 32 PPO-only candidates from checkpoint inference: 8 each for N12, N16, N24, and N40. Candidate orders were not modified in later stages.

## 7. Abaqus Teacher-Validation Execution Summary

Stage E converted all 32 PPO candidates into Abaqus CAE/INP cases. Solver execution later completed 32/32 and produced nonzero ODB files, with nonfatal warnings only.

## 8. Teacher Metric Extraction Summary

Stage G extracted teacher metrics for 32/32 PPO cases: U2, PEEQ, S/SurfaceT proxy, Mises, and NT11/temperature output metadata.

## 9. Ranking And Comparison Summary

Best PPO lexicographic candidates:

{chr(10).join(best_lines)}

## 10. New-Record Audit

New-record count versus combined552: {len(new_records)}. No PPO candidate beat the prior combined552 best in the primary ranking evidence.

## 11. Top-K Competitiveness Audit

Top-k count: {len(topk)} total. Distribution: N12=5, N16=4, N24=3, N40=0. The result supports bounded small-N competitiveness.

## 12. Surrogate-Vs-Teacher Alignment

Overall Spearman: {alignment.get('overall_spearman_predicted_vs_teacher_reward'):.4f}. Overall Pearson: {alignment.get('overall_pearson_predicted_vs_teacher_reward'):.4f}. Alignment is weak positive, with 1 false positive and 2 true positives.

## 13. Recovery-Anchor Duplicate Audit

{recovery_note}

## 14. Manuscript-Safe Claims

Use the bounded final claim boundary in `{FINAL_CLAIM_BOUNDARY}`. The strongest concise claim is: PPO batch32 was teacher-validated and small-N/top-k competitive, but produced no new combined552 records.

## 15. Unsafe Claims

Do not claim PPO produced a global best, dominated all native N, solved arbitrary-N scan-order optimisation, was trained online in Abaqus, is experimentally validated, or is first in the world.

## 16. Limitations

The PPO policy was trained in a surrogate environment. Surrogate-to-teacher alignment was weak. N40 primary-metric competitiveness was not observed. The recovery anchor is not a novel PPO discovery.

## 17. Next Manuscript Action

Integrate the evidence-chain, performance-summary, and claim-support tables into the manuscript addendum, preserving the bounded claim language.

## 18. Verdict

`{VERDICT}`
""",
        encoding="utf-8",
    )


def copy_and_index_plots() -> list[dict[str, Any]]:
    rows = []
    for src in sorted((STAGE_H_ROOT / "plots").glob("*.png")):
        dst = copy_file(src, PLOTS_INDEX_DIR / f"FROZEN_PPO_{src.name}")
        rows.append({"plot_name": dst.name, "frozen_plot_path": str(dst), "source_plot_path": str(src), "sha256": sha256_file(dst)})
    write_csv(PLOTS_INDEX_CSV, rows, ["plot_name", "frozen_plot_path", "source_plot_path", "sha256"])
    PLOTS_INDEX_MD.write_text("# FROZEN PPO Stage H Plots Index\n\n" + markdown_table(rows, ["plot_name", "frozen_plot_path", "source_plot_path", "sha256"]), encoding="utf-8")
    return rows


def manifest_paths() -> list[Path]:
    candidates = sorted((PROJECT_ROOT / "outputs" / NAMESPACE).rglob("*manifest*.json"))
    return [p for p in candidates if "stageI_final_ppo_evidence_freeze" not in str(p)]


def hash_artifacts(frozen_paths: dict[str, str], plot_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    core = [
        ("ppo_checkpoint_zip", PPO_CHECKPOINT),
        ("surrogate_model_joblib", SURROGATE_MODEL),
        ("ppo_selected_batch32_csv", PPO_SELECTED_BATCH),
        ("ppo_teacher_metrics_csv", STAGE_G_TEACHER_METRICS),
        ("stageH_ranking_full_csv", STAGE_H_FULL_RANKING),
        ("stageH_summary_by_N_csv", STAGE_H_SUMMARY_BY_N),
        ("stageH_report", PPO_STAGE_H_REPORT),
        ("stageH_claim_boundary", PPO_STAGE_H_CLAIM_BOUNDARY),
        ("stageB_surrogate_report", SURROGATE_REPORT),
        ("stageC_ppo_config", PPO_CONFIG),
        ("stageC_training_summary", PPO_TRAINING_SUMMARY),
        ("stageC_parameter_count", PPO_PARAMETER_COUNT),
        ("stageC_report", PPO_STAGE_C_REPORT),
        ("stageD_report", PPO_STAGE_D_REPORT),
        ("stageE_report", PPO_STAGE_E_REPORT),
        ("stageF_report", PPO_STAGE_F_REPORT),
        ("stageG_report", PPO_STAGE_G_REPORT),
        ("final_claim_boundary", FINAL_CLAIM_BOUNDARY),
        ("manuscript_integration_memo", MANUSCRIPT_MEMO),
        ("final_evidence_report", FINAL_REPORT),
        ("evidence_chain_csv", EVIDENCE_CHAIN_CSV),
        ("performance_summary_csv", PERFORMANCE_SUMMARY_CSV),
        ("claim_support_csv", CLAIM_SUPPORT_CSV),
        ("evidence_chain_md", EVIDENCE_CHAIN_MD),
        ("performance_summary_md", PERFORMANCE_SUMMARY_MD),
        ("claim_support_md", CLAIM_SUPPORT_MD),
        ("ara_index", ARA_INDEX),
    ]
    rows = []
    for label, path in core:
        if path.exists():
            rows.append({"artifact_label": label, "path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    for key, value in frozen_paths.items():
        path = Path(value)
        rows.append({"artifact_label": f"frozen_{key}", "path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    for path in manifest_paths():
        rows.append({"artifact_label": "prior_stage_manifest", "path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    for row in plot_rows:
        path = Path(row["frozen_plot_path"])
        rows.append({"artifact_label": f"frozen_plot_{path.name}", "path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_csv(HASHES_CSV, rows, ["artifact_label", "path", "size_bytes", "sha256"])
    HASHES_JSON.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def create_ara_index(frozen_paths: dict[str, str], plot_rows: list[dict[str, Any]]) -> None:
    script_path = PROJECT_ROOT / "stage3_ppo_rl_lam_fea_addendum_v01" / "scripts" / "stageI_freeze_ppo_addendum_evidence.py"
    lines = [
        "# ARA PPO Final Evidence Index",
        "",
        "## Core Artifact Paths",
        "",
        f"- PPO checkpoint: `{PPO_CHECKPOINT}`",
        f"- Surrogate model: `{SURROGATE_MODEL}`",
        f"- Selected PPO batch32: `{PPO_SELECTED_BATCH}`",
        f"- Stage G teacher metrics: `{STAGE_G_TEACHER_METRICS}`",
        f"- Stage H ranking table: `{STAGE_H_FULL_RANKING}`",
        f"- Final claim boundary: `{FINAL_CLAIM_BOUNDARY}`",
        "",
        "## Hashes",
        "",
        f"- CSV: `{HASHES_CSV}`",
        f"- JSON: `{HASHES_JSON}`",
        "",
        "## Scripts",
        "",
        f"- Stage I freezer: `{script_path}`",
        "- Earlier stage scripts live under `E:\\Projects\\RL-LAM-ScanOpt\\stage3_ppo_rl_lam_fea_addendum_v01\\scripts`.",
        "",
        "## Frozen Datasets And Tables",
        "",
    ]
    for key, path in frozen_paths.items():
        lines.append(f"- {key}: `{path}`")
    lines.extend(
        [
            "",
            "## Reports",
            "",
            f"- Stage H report: `{PPO_STAGE_H_REPORT}`",
            f"- Final evidence freeze report: `{FINAL_REPORT}`",
            f"- Manuscript integration memo: `{MANUSCRIPT_MEMO}`",
            f"- Claim support table: `{CLAIM_SUPPORT_MD}`",
            "",
            "## Plots",
            "",
            f"- Plots index CSV: `{PLOTS_INDEX_CSV}`",
            f"- Plots index Markdown: `{PLOTS_INDEX_MD}`",
        ]
    )
    for row in plot_rows:
        lines.append(f"- `{row['frozen_plot_path']}`")
    lines.extend(
        [
            "",
            "## How To Reproduce Evidence Chain",
            "",
            "1. Start from native combined552 teacher-labelled data.",
            "2. Use the Stage B surrogate reward model report and artifact.",
            "3. Use the Stage C frozen PPO checkpoint and config.",
            "4. Use the Stage D PPO-only selected batch32.",
            "5. Use the Stage E/F/G teacher-validation artifacts.",
            "6. Use Stage H ranking tables to support bounded claims.",
            "7. Use Stage I hashes to verify file identity.",
            "",
            "## What Not To Claim",
            "",
            "- Do not claim online Abaqus PPO.",
            "- Do not claim a new global best.",
            "- Do not claim PPO dominated all native N.",
            "- Do not claim arbitrary-N optimisation was solved.",
            "- Do not claim experimental validation.",
            "- Do not claim first-in-world status without a literature-priority audit.",
        ]
    )
    ARA_INDEX.write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_manifest(frozen_paths: dict[str, str], plot_rows: list[dict[str, Any]]) -> None:
    manifest = {
        "branch": git_branch(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_stageH_report_path": str(PPO_STAGE_H_REPORT),
        "frozen_table_paths": frozen_paths,
        "hash_paths": {"csv": str(HASHES_CSV), "json": str(HASHES_JSON)},
        "manuscript_table_paths": {
            "ppo_evidence_chain_table_csv": str(EVIDENCE_CHAIN_CSV),
            "ppo_evidence_chain_table_md": str(EVIDENCE_CHAIN_MD),
            "ppo_performance_summary_csv": str(PERFORMANCE_SUMMARY_CSV),
            "ppo_performance_summary_md": str(PERFORMANCE_SUMMARY_MD),
            "ppo_claim_support_csv": str(CLAIM_SUPPORT_CSV),
            "ppo_claim_support_md": str(CLAIM_SUPPORT_MD),
        },
        "plot_index_paths": {"csv": str(PLOTS_INDEX_CSV), "markdown": str(PLOTS_INDEX_MD), "frozen_plot_count": len(plot_rows)},
        "final_claim_boundary_path": str(FINAL_CLAIM_BOUNDARY),
        "manuscript_memo_path": str(MANUSCRIPT_MEMO),
        "final_evidence_report_path": str(FINAL_REPORT),
        "ARA_index_path": str(ARA_INDEX),
        "no_Abaqus": True,
        "no_ODB_opening": True,
        "no_solver": True,
        "no_CAE_INP_JNL": True,
        "no_candidate_generation": True,
        "no_training": True,
        "no_commit_or_push": True,
        "verdict": VERDICT,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def validate_required_inputs() -> None:
    required = [
        SURROGATE_MODEL,
        SURROGATE_REPORT,
        PPO_CHECKPOINT,
        PPO_CONFIG,
        PPO_TRAINING_SUMMARY,
        PPO_PARAMETER_COUNT,
        PPO_SELECTED_BATCH,
        STAGE_G_TEACHER_METRICS,
        STAGE_H_FULL_RANKING,
        STAGE_H_SUMMARY_BY_N,
        STAGE_H_NEW_RECORDS,
        STAGE_H_TOPK,
        STAGE_H_ALIGNMENT_JSON,
        STAGE_H_RECOVERY,
        PPO_STAGE_H_REPORT,
        PPO_STAGE_H_CLAIM_BOUNDARY,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required Stage I inputs:\n" + "\n".join(missing))


def main() -> int:
    ensure_dirs()
    validate_required_inputs()
    by_n, topk, new_records, alignment, recovery = read_stage_h_summaries()
    frozen_paths = freeze_tables()
    create_final_claim_boundary()
    create_evidence_chain_table()
    create_performance_summary_table(by_n)
    create_claim_support_table()
    create_manuscript_memo()
    create_final_report(by_n, topk, new_records, alignment, recovery)
    plot_rows = copy_and_index_plots()
    create_ara_index(frozen_paths, plot_rows)
    hash_rows = hash_artifacts(frozen_paths, plot_rows)
    create_manifest(frozen_paths, plot_rows)
    summary = {
        "verdict": VERDICT,
        "frozen_table_count": len(frozen_paths),
        "hash_row_count": len(hash_rows),
        "frozen_plot_count": len(plot_rows),
        "hashes_csv": str(HASHES_CSV),
        "manifest": str(MANIFEST),
        "final_report": str(FINAL_REPORT),
        "no_Abaqus": True,
        "no_ODB_opening": True,
        "no_solver": True,
        "no_CAE_INP_JNL": True,
        "no_candidate_generation": True,
        "no_training": True,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
