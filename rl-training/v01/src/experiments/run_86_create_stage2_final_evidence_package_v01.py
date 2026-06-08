#!/usr/bin/env python
"""Create Stage 2 final documentation/evidence package.

Documentation-only consolidation.  Does not train, generate candidates, create
CAE/INP/JNL, submit Abaqus, run datacheck, or open ODB files.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(r"D:\Projects\RL-LAM-ScanOpt")
RL = ROOT / "rl-training" / "v01"
OUTPUTS = RL / "outputs"
REPORTS = OUTPUTS / "reports"
DOCS = ROOT / "docs"
STAGE2 = DOCS / "stage2"
PACKAGE = DOCS / "stage2_final_evidence_package_v01"
TIMESTAMP = datetime.now().isoformat(timespec="seconds")


def ensure() -> None:
    STAGE2.mkdir(parents=True, exist_ok=True)
    PACKAGE.mkdir(parents=True, exist_ok=True)


def exists(path: Path) -> str:
    return "present" if path.exists() else "missing"


def write(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str] | None = None) -> None:
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_path(run: str, script: str, out: str, report: str) -> Dict[str, str]:
    return {
        "run_id": run,
        "script_path": str(RL / "src" / "experiments" / script),
        "output_path": str(OUTPUTS / out),
        "report_path": str(REPORTS / report),
        "script_status": exists(RL / "src" / "experiments" / script),
        "output_status": exists(OUTPUTS / out),
        "report_status": exists(REPORTS / report),
    }


RUNS = [
    {**run_path("run71c", "run_71c_batch_backfill_surface_gradient_under_abaqus_python_v01.py", "surface_tensile_gradient_full_field_label_backfill_v03", "surface_tensile_gradient_full_field_label_backfill_v03_report.md"), "purpose": "Canonical full-field SurfaceT/Gradient label backfill", "final_verdict": "batch/resume backfill support; canonical labels available", "key_result": "canonical full-field label count 387; SurfaceT/Gradient training-ready rows 336", "generated_CAE": "False", "opened_ODB": "read-only only when run under Abaqus Python; documentation package does not open ODB", "trained_models": "False", "submitted_Abaqus": "False"},
    {**run_path("run72", "run_72_train_surface_and_gradient_single_objective_surrogates_v01.py", "surface_gradient_single_objective_surrogate_training_v01", "surface_gradient_single_objective_surrogate_training_v01_report.md"), "purpose": "Train separate SurfaceT and Gradient diagnostic surrogates", "final_verdict": "SurfaceT diagnostic signal; Gradient weak", "key_result": "SurfaceT ExtraTrees Spearman 0.5301/top10 6.6; Gradient Spearman 0.3820", "generated_CAE": "False", "opened_ODB": "False", "trained_models": "True", "submitted_Abaqus": "False"},
    {**run_path("run76", "run_76_analyze_U2_SurfaceT_tradeoff_laws_v01.py", "U2_SurfaceT_tradeoff_law_analysis_v01", "U2_SurfaceT_tradeoff_law_analysis_v01_report.md"), "purpose": "Analyze U2-SurfaceT relationship", "final_verdict": "weak/nonlinear/family-dependent relation", "key_result": "387 valid rows; global Spearman 0.4939; Pearson 0.0302; 97 U2-pass rows; 57 Pareto candidates", "generated_CAE": "False", "opened_ODB": "False", "trained_models": "False", "submitted_Abaqus": "False"},
    {**run_path("run79", "run_79_UST_probe10_ODB_teacher_validation_v01.py", "UST_probe10_ODB_teacher_validation_v01", "UST_probe10_ODB_teacher_validation_v01_report.md"), "purpose": "Teacher-validate U2-safe SurfaceT probe10", "final_verdict": "no SurfaceT improvement claim", "key_result": "10/10 ODB postprocessed; U2 pass 2/10; PEEQ pass 10/10; SurfaceT Spearman 0.9515; best UST SurfaceT did not beat best reference", "generated_CAE": "False", "opened_ODB": "True in original run; package does not open ODB", "trained_models": "False", "submitted_Abaqus": "False"},
    {**run_path("run80", "run_80_freeze_GNN_RL_policy_learning_evidence_v01.py", "GNN_RL_policy_learning_evidence_freeze_v01", "GNN_RL_policy_learning_evidence_freeze_v01_report.md"), "purpose": "Freeze GNN/RL policy-learning evidence", "final_verdict": "PASS_GNN_RL_POLICY_LEARNING_EVIDENCE_FROZEN; WARNING_GNN_RL_NOT_FINAL_PHYSICAL_OPTIMISER", "key_result": "GNN/RL is policy-learning / agent-feasibility evidence, not final physical optimiser", "generated_CAE": "False", "opened_ODB": "False", "trained_models": "False", "submitted_Abaqus": "False"},
    {**run_path("run81", "run_81_transformer_sequence_surrogate_ablation_v01.py", "transformer_sequence_surrogate_ablation_v01", "transformer_sequence_surrogate_ablation_v01_report.md"), "purpose": "Transformer sequence surrogate ablation", "final_verdict": "WARNING_TRANSFORMER_ABLATION_NO_CLEAR_IMPROVEMENT", "key_result": "Transformer SurfaceT Spearman 0.4479/top10 5.8 vs ExtraTrees 0.5301/top10 6.6; U2 0.3680; PEEQ 0.1220", "generated_CAE": "False", "opened_ODB": "False", "trained_models": "True", "submitted_Abaqus": "False"},
    {**run_path("run84", "run_84_audit_GNN_RL_vs_stage1_full32_baselines_current_teacher_metrics_v01.py", "GNN_RL_vs_stage1_full32_baseline_audit_v01", "GNN_RL_vs_stage1_full32_baseline_audit_v01_report.md"), "purpose": "Audit GNN/RL vs earliest full32 baselines", "final_verdict": "FAIL_BASELINE10_LABELS_MISSING", "key_result": "GNN/RL advantage over 9/10 labelled early baselines; smartscan_proxy_variance missing", "generated_CAE": "False", "opened_ODB": "False", "trained_models": "False", "submitted_Abaqus": "False"},
    {**run_path("run85", "run_85_freeze_masked_UST_probe40_evidence_and_U2_guard_audit_v01.py", "masked_UST_probe40_evidence_freeze_and_U2_guard_audit_v01", "masked_UST_probe40_evidence_freeze_and_U2_guard_audit_v01_report.md"), "purpose": "Freeze masked probe40 and U2 guard boundary", "final_verdict": "PASS_MASKED_PROBE40_EVIDENCE_FROZEN plus guard warnings", "key_result": "absolute U2 guard 7.8362e-05; U2 pass 0/40; PEEQ pass 28/40; best masked U2 4.47x guard; do not scale masked to 400 yet", "generated_CAE": "False", "opened_ODB": "False", "trained_models": "False", "submitted_Abaqus": "False"},
]


KEY_RESULTS = [
    {"topic": "Search space", "metric": "32!", "value": "≈ 2.63 × 10^35", "source_run": "Stage 2 summary", "source_file": "STAGE2_FINAL_SUMMARY.md", "interpretation": "Stage 2 is not brute-force search."},
    {"topic": "Objective hierarchy", "metric": "final hierarchy", "value": "U2 primary; PEEQ safety; SurfaceT secondary; Gradient/Mises/internal diagnostics", "source_run": "Stage 2 synthesis", "source_file": "STAGE2_FINAL_SUMMARY.md", "interpretation": "U2-first + SurfaceT-secondary constrained search is the current best route."},
    {"topic": "Full-field labels", "metric": "canonical labels", "value": "387", "source_run": "run71c", "source_file": "surface_tensile_gradient_full_field_label_backfill_v03", "interpretation": "Sufficient full-field dataset for diagnostic SurfaceT/Gradient surrogate analysis."},
    {"topic": "Full-field labels", "metric": "training-ready SurfaceT/Gradient rows", "value": "336 / 336", "source_run": "run71c", "source_file": "surface_tensile_gradient_full_field_label_backfill_v03", "interpretation": "Run72 justified as diagnostic training."},
    {"topic": "SurfaceT surrogate", "metric": "ExtraTrees leave-family-out Spearman", "value": "0.5301", "source_run": "run72", "source_file": "surface_gradient_single_objective_surrogate_training_v01_report.md", "interpretation": "SurfaceT has diagnostic ranking signal."},
    {"topic": "SurfaceT surrogate", "metric": "top10 overlap", "value": "6.6", "source_run": "run72", "source_file": "surface_gradient_single_objective_surrogate_training_v01_report.md", "interpretation": "Useful but not final teacher replacement."},
    {"topic": "Gradient surrogate", "metric": "ExtraTrees leave-family-out Spearman", "value": "0.3820", "source_run": "run72", "source_file": "surface_gradient_single_objective_surrogate_training_v01_report.md", "interpretation": "Gradient remains weak/diagnostic."},
    {"topic": "U2-SurfaceT relation", "metric": "valid rows", "value": "387", "source_run": "run76", "source_file": "U2_SurfaceT_tradeoff_law_analysis_v01_report.md", "interpretation": "Analysis uses canonical full-field labels."},
    {"topic": "U2-SurfaceT relation", "metric": "global Spearman / Pearson", "value": "0.4939 / 0.0302", "source_run": "run76", "source_file": "U2_SurfaceT_tradeoff_law_analysis_v01_report.md", "interpretation": "Weak/nonlinear/family-dependent relation, not a simple monotonic law."},
    {"topic": "U2 feasible set", "metric": "U2-pass rows / Pareto candidates", "value": "97 / 57", "source_run": "run76", "source_file": "U2_SurfaceT_tradeoff_law_analysis_v01_report.md", "interpretation": "Feasible-region analysis motivates U2-first strategy."},
    {"topic": "UST probe10", "metric": "ODB postprocessed", "value": "10/10", "source_run": "run79", "source_file": "UST_probe10_ODB_teacher_validation_v01_report.md", "interpretation": "Teacher validation completed."},
    {"topic": "UST probe10", "metric": "U2 pass / PEEQ pass / combined pass", "value": "2/10 / 10/10 / 2/10", "source_run": "run79", "source_file": "UST_probe10_ODB_teacher_validation_v01_report.md", "interpretation": "SurfaceT search must remain constrained by U2."},
    {"topic": "UST probe10", "metric": "SurfaceT proxy-vs-teacher Spearman", "value": "0.9515", "source_run": "run79", "source_file": "UST_probe10_ODB_teacher_validation_v01_report.md", "interpretation": "Small-batch proxy ranking signal, but no improvement over best reference."},
    {"topic": "GNN/RL policy", "metric": "run80 verdict", "value": "PASS_GNN_RL_POLICY_LEARNING_EVIDENCE_FROZEN", "source_run": "run80", "source_file": "GNN_RL_policy_learning_evidence_freeze_v01_report.md", "interpretation": "Policy-learning / agent-feasibility evidence frozen."},
    {"topic": "GNN/RL physical boundary", "metric": "run80 warning", "value": "WARNING_GNN_RL_NOT_FINAL_PHYSICAL_OPTIMISER", "source_run": "run80", "source_file": "GNN_RL_policy_learning_evidence_freeze_v01_report.md", "interpretation": "Do not frame as final physical optimiser."},
    {"topic": "Transformer", "metric": "SurfaceT Spearman / top10", "value": "0.4479 / 5.8", "source_run": "run81", "source_file": "transformer_sequence_surrogate_ablation_v01_report.md", "interpretation": "No clear improvement over ExtraTrees."},
    {"topic": "Transformer", "metric": "ExtraTrees SurfaceT Spearman / top10", "value": "0.5301 / 6.6", "source_run": "run81/run72", "source_file": "surface_gradient_single_objective_surrogate_training_v01_report.md", "interpretation": "Feature-based ExtraTrees remains stronger on current teacher data."},
    {"topic": "GNN/RL vs baselines", "metric": "labelled baseline count", "value": "9/10", "source_run": "run84", "source_file": "GNN_RL_vs_stage1_full32_baseline_audit_v01_report.md", "interpretation": "Use 9/10 wording; smartscan_proxy_variance missing."},
    {"topic": "GNN/RL vs baselines", "metric": "best labelled baseline U2 vs best GNN/RL U2", "value": "9.97175e-05 vs 4.7092337e-05", "source_run": "run84", "source_file": "GNN_RL_vs_stage1_full32_baseline_audit_v01_report.md", "interpretation": "Teacher-validated advantage over labelled baseline set."},
    {"topic": "GNN/RL vs baselines", "metric": "median U2 / PEEQ / SurfaceT", "value": "GNN/RL 0.00045039433 / 0.1453374 / 0.0071755; baseline 0.0009858571 / 0.1524441 / 0.0140886", "source_run": "run84", "source_file": "GNN_RL_vs_stage1_full32_baseline_audit_v01_report.md", "interpretation": "Supports advantage over labelled 9/10 early baseline set, not all 10."},
    {"topic": "Masked", "metric": "absolute U2 guard", "value": "7.8362e-05", "source_run": "run85", "source_file": "masked_UST_probe40_evidence_freeze_and_U2_guard_audit_v01_report.md", "interpretation": "Full-32 U2 guard is too strict/uncalibrated for masked regimes."},
    {"topic": "Masked", "metric": "U2 pass / PEEQ pass / combined feasible", "value": "0/40 / 28/40 / 0/40", "source_run": "run85", "source_file": "masked_UST_probe40_evidence_freeze_and_U2_guard_audit_v01_report.md", "interpretation": "Do not scale masked to 400 without per-mask guard calibration."},
]


MANIFEST = [
    {"evidence_id": "E001", "run_id": "run71c", "file_path": str(OUTPUTS / "surface_tensile_gradient_full_field_label_backfill_v03" / "canonical_full_field_surface_gradient_teacher_labels_v03.csv"), "file_type": "csv", "description": "Canonical full-field SurfaceT/Gradient labels", "supports_claim": "SurfaceT/Gradient full-field label backfill complete", "limitations": "Derived from existing ODB postprocessing; not a candidate generation result", "github_include_recommendation": "include if size is acceptable"},
    {"evidence_id": "E002", "run_id": "run72", "file_path": str(REPORTS / "surface_gradient_single_objective_surrogate_training_v01_report.md"), "file_type": "markdown", "description": "SurfaceT/Gradient single-objective surrogate training report", "supports_claim": "SurfaceT diagnostic ranking signal; Gradient weak", "limitations": "Surrogate does not replace teacher validation", "github_include_recommendation": "include"},
    {"evidence_id": "E003", "run_id": "run76", "file_path": str(REPORTS / "U2_SurfaceT_tradeoff_law_analysis_v01_report.md"), "file_type": "markdown", "description": "U2-SurfaceT relationship analysis", "supports_claim": "U2-first + SurfaceT-secondary constrained search", "limitations": "No simple global monotonic law", "github_include_recommendation": "include"},
    {"evidence_id": "E004", "run_id": "run79", "file_path": str(REPORTS / "UST_probe10_ODB_teacher_validation_v01_report.md"), "file_type": "markdown", "description": "UST probe10 teacher validation", "supports_claim": "SurfaceT proxy signal but no SurfaceT improvement over best reference", "limitations": "Small probe10; no broad SurfaceT success claim", "github_include_recommendation": "include"},
    {"evidence_id": "E005", "run_id": "run80", "file_path": str(REPORTS / "GNN_RL_policy_learning_evidence_freeze_v01_report.md"), "file_type": "markdown", "description": "GNN/RL policy-learning evidence freeze", "supports_claim": "Learning-based scan-order policy feasibility", "limitations": "Not final physical optimiser", "github_include_recommendation": "include"},
    {"evidence_id": "E006", "run_id": "run81", "file_path": str(REPORTS / "transformer_sequence_surrogate_ablation_v01_report.md"), "file_type": "markdown", "description": "Transformer ablation", "supports_claim": "Transformer did not beat ExtraTrees on current data", "limitations": "Current teacher data only", "github_include_recommendation": "include"},
    {"evidence_id": "E007", "run_id": "run84", "file_path": str(REPORTS / "GNN_RL_vs_stage1_full32_baseline_audit_v01_report.md"), "file_type": "markdown", "description": "GNN/RL vs early full32 baseline audit", "supports_claim": "Teacher-validated advantage over 9/10 labelled early baselines", "limitations": "smartscan_proxy_variance missing; do not claim all 10", "github_include_recommendation": "include"},
    {"evidence_id": "E008", "run_id": "run85", "file_path": str(REPORTS / "masked_UST_probe40_evidence_freeze_and_U2_guard_audit_v01_report.md"), "file_type": "markdown", "description": "Masked probe40 evidence freeze", "supports_claim": "Masked boundary; do not scale to 400 yet", "limitations": "Masked guard calibration needed", "github_include_recommendation": "include"},
]


def main() -> None:
    ensure()
    missing = []
    for run in RUNS:
        if run["report_status"] == "missing" or run["output_status"] == "missing":
            missing.append(run["run_id"])

    summary = f"""
# Stage 2 Final Summary

## Executive Summary
Stage 2 established a teacher-validated evidence chain for scan-order optimisation without pretending to brute-force the full search space. The full 32-track search space is `32! ≈ 2.63 × 10^35`, so the scientific contribution is not exhaustive enumeration. The final Stage 2 position is a constrained physical hierarchy: U2/warpage first, PEEQ safety required, SurfaceT as secondary performance inside the feasible region, and Gradient/Mises/internal tensile stress as diagnostics.

## Research Problem
Laser additive scan-order optimisation is a sequential combinatorial decision problem. The project asks whether learning-guided policies and teacher-validated surrogate analysis can find useful scan-order families while remaining physically honest.

## Teacher-Validation Workflow
Stage 2 uses Abaqus/ODB teacher validation as the arbiter for physical claims. Surrogates, GNNs, RL policies, and Transformer models are candidate generators or diagnostic rankers unless teacher validation supports a physical claim.

## Final Metric Hierarchy
1. Primary: `U2` / vertical in-plane warpage.
2. Safety: `PEEQ`.
3. Secondary: `SurfaceT` / surface tensile residual stress inside U2/PEEQ feasible or near-feasible candidates.
4. Diagnostics: Gradient, Mises, internal tensile stress.

The old multi-weight residual-stress composite was demoted because it was confounded and because unconstrained S/SurfaceT-first selection caused U2/PEEQ safety problems. Unconstrained SurfaceT-first is not the final route. The current best route is `U2-first + SurfaceT-secondary constrained search`.

## GNN/RL Evidence
Run80 freezes GNN/RL as successful policy-learning / agent-feasibility evidence. GNN/RL demonstrates that scan-ordering can be formulated as a sequential decision problem and that learned policies or graph/node scoring can produce legal and useful candidate families. It is not the final physical optimisation engine by itself.

Run84 adds a current-metric audit: GNN/RL has teacher-validated advantage over `9/10` labelled early full-32 baselines. Use that wording. Do not claim advantage over all 10 because `smartscan_proxy_variance` is missing teacher evidence.

## SurfaceT / Gradient Evidence
Run71c consolidated `387` canonical full-field labels with `336` training-ready SurfaceT rows and `336` training-ready Gradient rows. Run72 found SurfaceT ExtraTrees leave-family-out Spearman `0.5301` and top10 overlap `6.6`, supporting a diagnostic ranking signal. Gradient Spearman `0.3820` remains weak and diagnostic.

## U2-SurfaceT Relationship
Run76 found `387` valid U2+SurfaceT rows, global Spearman `0.4939`, Pearson `0.0302`, `97` U2-pass rows, and `57` Pareto candidates. The relationship is weak/nonlinear/family-dependent, not a simple global monotonic stress-release trade-off.

## UST Probe10
Run79 postprocessed `10/10` ODBs. Teacher U2 pass was `2/10`, PEEQ pass `10/10`, combined pass `2/10`, and SurfaceT proxy-vs-teacher Spearman was `0.9515`. The best UST SurfaceT did not beat the best existing reference, so SurfaceT improvement should not be claimed.

## Transformer Ablation
Run81 showed no clear Transformer improvement over feature-based ExtraTrees: Transformer SurfaceT Spearman `0.4479` and top10 `5.8` versus ExtraTrees SurfaceT Spearman `0.5301` and top10 `6.6`.

## Masked Generalisation Boundary
Run85 froze masked probe40 evidence. The full-32 absolute U2 guard `7.8362e-05` was too restrictive or uncalibrated for masked cases: U2 pass `0/40`, PEEQ pass `28/40`, combined feasible `0/40`; best masked U2 was `4.47x` guard and median masked U2 was `5.96x` guard. Masked should not scale to 400 without per-mask guard calibration.

## What Can Be Claimed
- Stage 2 is not brute force; it is teacher-guided evidence-driven search in a huge combinatorial space.
- U2-first + SurfaceT-secondary is the final Stage 2 physical direction.
- GNN/RL is successful policy-learning / agent-feasibility evidence.
- GNN/RL has teacher-validated advantage over 9/10 labelled early full-32 baselines under current metrics.
- SurfaceT has diagnostic ranking signal; Gradient remains weak.
- Transformer did not beat ExtraTrees under current teacher data.
- Masked transfer needs per-mask guard calibration.

## What Cannot Be Claimed
- Do not claim global optimum.
- Do not claim GNN/RL is the final physical optimiser.
- Do not claim superiority over all earliest 10 baselines unless `smartscan_proxy_variance` is validated or excluded with justification.
- Do not claim SurfaceT optimisation is solved.
- Do not claim masked transfer success or scale masked to 400 now.
- Do not claim arbitrary-N generalisation is already solved.

## Why Stage 2 Should Stop Here
Stage 2 has enough evidence to define the physical objective hierarchy, freeze GNN/RL as policy-learning evidence, demote confounded objectives, and identify masked/fixed-32 limits. Further fixed-32 tuning risks diminishing returns.

## Stage 3 Handoff
Stage 3 should move to Variable-N Graph Pointer RL Policy with per-instance feasibility guards, not more fixed-32 tuning.
"""
    write(STAGE2 / "STAGE2_FINAL_SUMMARY.md", summary)

    claim_boundary = """
# Stage 2 Claim Boundary

## Safe Claims
- The 32-track search space is `32! ≈ 2.63 × 10^35`.
- Stage 2 should not be framed as brute-force search.
- The final objective hierarchy is U2 primary, PEEQ safety, SurfaceT secondary, Gradient/Mises/internal diagnostics.
- GNN/RL is successful policy-learning / agent-feasibility evidence.
- GNN/RL has teacher-validated advantage over 9/10 labelled early full-32 baselines.
- SurfaceT has diagnostic ranking signal.
- Gradient remains weak/diagnostic.
- Transformer did not improve over ExtraTrees under current teacher data.
- Masked probe40 exposes a masked generalisation boundary.

## Conditional Claims
- GNN/RL exceeds all earliest 10 baselines only if `smartscan_proxy_variance` is teacher-validated or excluded with explicit justification.
- SurfaceT-guided generation can be claimed only as diagnostic unless U2/PEEQ feasibility and teacher SurfaceT improvement are confirmed.
- Masked generalisation can be claimed only after per-mask U2/PEEQ guard calibration.

## Unsafe Claims
- Global optimum found.
- SurfaceT optimisation solved.
- GNN/RL is the final physical optimiser.
- Transformer is superior to ExtraTrees.
- Masked transfer success is proven.
- Variable-N generalisation is already solved.

## Missing Evidence
- `smartscan_proxy_variance` teacher label is missing from the earliest 10 baseline comparison.
- SurfaceT improvement over the best existing reference is not demonstrated by UST probe10.
- Masked feasible SurfaceT region is not established under the full-32 absolute guard.

## Suggested Paper Wording
“Stage 2 demonstrates a teacher-guided learning framework for scan-order optimisation in a 32-track search space of approximately `2.63 × 10^35` permutations. The final evidence supports a U2-first, PEEQ-safe, SurfaceT-secondary objective hierarchy. GNN/RL models provide policy-learning and candidate-generation evidence, while physical claims remain bounded by teacher validation.”

## Overclaims To Avoid
- “The optimiser solved scan-ordering.”
- “GNN/RL beats all baselines.”
- “SurfaceT optimisation is solved.”
- “Masked generalisation is proven.”
- “Transformer is the best model.”
"""
    write(STAGE2 / "STAGE2_CLAIM_BOUNDARY.md", claim_boundary)

    run_index_md = "# Stage 2 Run Index\n\n| Run | Purpose | Verdict | Key Result | Script | Output | Report | CAE? | ODB? | Trained? | Abaqus? |\n|---|---|---|---|---|---|---|---|---|---|---|\n"
    for r in RUNS:
        run_index_md += f"| {r['run_id']} | {r['purpose']} | {r['final_verdict']} | {r['key_result']} | `{r['script_path']}` | `{r['output_path']}` | `{r['report_path']}` | {r['generated_CAE']} | {r['opened_ODB']} | {r['trained_models']} | {r['submitted_Abaqus']} |\n"
    write(STAGE2 / "STAGE2_RUN_INDEX.md", run_index_md)

    github = """
# Stage 2 GitHub README Draft

This repository contains the Stage 2 evidence package for RL-LAM-ScanOpt.

## What To Include
- Scripts under `rl-training/v01/src/experiments`.
- Documentation under `docs/stage2`.
- Small CSV summaries and reports.
- Evidence manifest and key result tables.

## What To Exclude
- ODB, CAE, SIM, PRT, STA, MSG, DAT, LCK, and other Abaqus outputs.
- Large raw extraction folders.
- Cache folders and temporary Abaqus files.

## .gitignore Recommendation
Verify entries for:

```gitignore
*.odb
*.cae
*.sim
*.prt
*.sta
*.msg
*.dat
*.lck
__pycache__/
.pytest_cache/
outputs/**/large_raw/
cae_models/**/large_raw/
```

## Scientific Positioning
Stage 2 is a teacher-guided evidence package. It supports U2-first + SurfaceT-secondary constrained search and GNN/RL policy-learning feasibility. It does not claim global optimum or autonomous closed-loop physical optimisation.
"""
    write(STAGE2 / "STAGE2_GITHUB_README_DRAFT.md", github)

    handoff = """
# Stage 2 to Stage 3 Handoff

## Why Not Continue Fixed-32 Tuning
Fixed-32 evidence is mature enough to define the objective hierarchy and model boundaries. Additional fixed-32 tuning risks overfitting to known families.

## Why Masked Should Not Scale To 400 Now
Run85 shows the full-32 absolute U2 guard does not transfer directly to masked regimes: U2 pass `0/40`, combined feasible `0/40`, best masked U2 `4.47x` guard. Per-mask guard calibration is required first.

## Stage 3 Direction
Move to Variable-N Graph Pointer RL Policy.

## Proposed Proof of Concept
1. Variable-N graph representation for available tracks.
2. Pointer-style policy with legality masking.
3. Per-instance U2/PEEQ guard calibration.
4. SurfaceT secondary ranking only inside feasible or near-feasible region.
5. Teacher validation on small calibrated batches before scale-up.

## Claim Boundary
Stage 3 should aim to prove variable-N policy feasibility and calibrated constrained generation. It should not claim arbitrary-N physical superiority before teacher validation.
"""
    write(STAGE2 / "STAGE2_STAGE3_HANDOFF.md", handoff)

    write_csv(STAGE2 / "STAGE2_RUN_INDEX.csv", RUNS)
    write_csv(STAGE2 / "STAGE2_EVIDENCE_MANIFEST.csv", MANIFEST)
    write_csv(STAGE2 / "STAGE2_KEY_RESULTS_TABLE.csv", KEY_RESULTS)

    # Required run index is Markdown; CSV is a helpful companion. Keep required name only as MD.
    report = f"""
# Stage 2 Final Evidence Consolidation Report

Generated: `{TIMESTAMP}`

## Output Locations
- Public docs folder: `{STAGE2}`
- Evidence package folder: `{PACKAGE}`

## Files Created
- `docs/stage2/STAGE2_FINAL_SUMMARY.md`
- `docs/stage2/STAGE2_CLAIM_BOUNDARY.md`
- `docs/stage2/STAGE2_RUN_INDEX.md`
- `docs/stage2/STAGE2_EVIDENCE_MANIFEST.csv`
- `docs/stage2/STAGE2_KEY_RESULTS_TABLE.csv`
- `docs/stage2/STAGE2_GITHUB_README_DRAFT.md`
- `docs/stage2/STAGE2_STAGE3_HANDOFF.md`
- `docs/stage2_final_evidence_package_v01/stage2_final_evidence_consolidation_report.md`

## Missing Expected Reports
`{', '.join(missing) if missing else 'none'}`

## Scientific Consolidation
The final Stage 2 package preserves these conclusions:

1. The full 32-track search space is `32! ≈ 2.63 × 10^35`.
2. Stage 2 is teacher-guided, not brute force.
3. Final objective hierarchy is U2 primary, PEEQ safety, SurfaceT secondary, Gradient/Mises/internal diagnostics.
4. Old multi-weight residual-stress composite was demoted.
5. Unconstrained SurfaceT-first is not the final route.
6. GNN/RL is successful policy-learning / agent-feasibility evidence.
7. GNN/RL has teacher-validated advantage over 9/10 labelled early full-32 baselines.
8. Do not claim all-10 baseline superiority until `smartscan_proxy_variance` is validated or excluded.
9. SurfaceT has diagnostic ranking signal; Gradient remains weak.
10. Transformer did not beat feature-based ExtraTrees under current teacher data.
11. Masked probe40 defines a boundary, not a scale-up success.
12. Stage 3 should move to Variable-N Graph Pointer RL Policy.

## GitHub Upload Recommendation
Include scripts, docs, small CSV summaries, and reports. Exclude ODB/CAE/large INP/Abaqus temporary outputs/cache folders. Existing `.gitignore` already covers most Abaqus files; verify `outputs/**/large_raw/` and `cae_models/**/large_raw/`.

## Guardrails
- Models trained: `False`
- Candidates generated: `False`
- CAE/INP/JNL generated: `False`
- Abaqus jobs submitted: `False`
- Datacheck run: `False`
- ODB files opened: `False`
- Teacher modules modified: `False`

## Final Verdict
`{'WARNING_STAGE2_SOME_REPORTS_MISSING' if missing else 'PASS_STAGE2_EVIDENCE_PACKAGE_CREATED'}`
"""
    write(PACKAGE / "stage2_final_evidence_consolidation_report.md", report)

    # Copy key Markdown summaries into package as lightweight references.
    for src in [
        STAGE2 / "STAGE2_FINAL_SUMMARY.md",
        STAGE2 / "STAGE2_CLAIM_BOUNDARY.md",
        STAGE2 / "STAGE2_RUN_INDEX.md",
        STAGE2 / "STAGE2_GITHUB_README_DRAFT.md",
        STAGE2 / "STAGE2_STAGE3_HANDOFF.md",
    ]:
        write(PACKAGE / src.name, src.read_text(encoding="utf-8"))

    print("created files:")
    for p in [
        STAGE2 / "STAGE2_FINAL_SUMMARY.md",
        STAGE2 / "STAGE2_CLAIM_BOUNDARY.md",
        STAGE2 / "STAGE2_RUN_INDEX.md",
        STAGE2 / "STAGE2_EVIDENCE_MANIFEST.csv",
        STAGE2 / "STAGE2_KEY_RESULTS_TABLE.csv",
        STAGE2 / "STAGE2_GITHUB_README_DRAFT.md",
        STAGE2 / "STAGE2_STAGE3_HANDOFF.md",
        PACKAGE / "stage2_final_evidence_consolidation_report.md",
    ]:
        print(f"- {p}")
    print(f"missing expected reports: {', '.join(missing) if missing else 'none'}")
    print("final verdict:", "WARNING_STAGE2_SOME_REPORTS_MISSING" if missing else "PASS_STAGE2_EVIDENCE_PACKAGE_CREATED")


if __name__ == "__main__":
    main()
