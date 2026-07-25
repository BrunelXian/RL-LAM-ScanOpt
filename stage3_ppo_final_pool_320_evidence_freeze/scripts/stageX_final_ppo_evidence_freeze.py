"""Stage X final evidence freeze for the 320-case PPO pool.

This script freezes existing Stage W evidence into a final manuscript-facing
package. It does not run Abaqus, open ODB files, extract ODB metrics, run
solver/datacheck/enqueue, generate CAE/INP/JNL files, train models, or generate
new candidates.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
NS = "stage3_ppo_final_pool_320_evidence_freeze"
OUT_ROOT = ROOT / "outputs" / NS
DOCS_ROOT = ROOT / "docs" / NS
FROZEN = OUT_ROOT / "frozen_tables"
HASHES = OUT_ROOT / "hashes"
MANUSCRIPT = OUT_ROOT / "manuscript_tables"
REPORTS = OUT_ROOT / "reports"

STAGEW_ROOT = ROOT / "outputs" / "stage3_ppo_final_pool_320_analysis"
STAGEW_TABLES = STAGEW_ROOT / "tables"
STAGEW_DOCS = ROOT / "docs" / "stage3_ppo_final_pool_320_analysis"

INPUTS = {
    "final_ppo_pool": STAGEW_TABLES / "ppo_final_pool_320_teacher_metrics.csv",
    "combined552_plus_ppo": STAGEW_TABLES / "combined552_plus_ppo_final_pool_320_analysis_dataset.csv",
    "ranking_full": STAGEW_TABLES / "ppo_final_pool_320_teacher_metric_ranking_full.csv",
    "claim_decision": STAGEW_TABLES / "ppo_final_pool_320_claim_decision_table.csv",
    "stageW_report": STAGEW_DOCS / "PPO_FINAL_POOL_320_STAGEW_RANKING_AND_COMPARISON_REPORT.md",
    "stageW_claim_boundary": STAGEW_DOCS / "PPO_FINAL_POOL_320_STAGEW_CLAIM_BOUNDARY.md",
    "stageW_manifest": STAGEW_ROOT / "stageW_final_ppo_pool_320_analysis_manifest.json",
    "combined552_reference": ROOT / "outputs" / "stage3_run_78_final_evidence_freeze_package" / "FROZEN_stage3_native_combined552_teacher_dataset.csv",
}

OPTIONAL_STAGEW_TABLES = {
    "topk_candidates": "ppo_final_pool_320_topk_competitive_candidates.csv",
    "topk_by_N": "ppo_final_pool_320_topk_summary_by_N.csv",
    "topk_by_version": "ppo_final_pool_320_topk_summary_by_version.csv",
    "new_records": "ppo_final_pool_320_new_record_candidates.csv",
    "bootstrap_by_N": "ppo_final_pool_320_vs_bootstrap_random_reference_by_N.csv",
    "bootstrap_global": "ppo_final_pool_320_vs_bootstrap_random_reference_global.csv",
    "baseline_comparison": "ppo_final_pool_320_vs_identified_baseline_families.csv",
    "baseline_inventory": "ppo_final_pool_320_identified_baseline_family_inventory.csv",
    "industrial_proxy_summary": "ppo_final_pool_320_industrial_efficiency_proxy_summary.csv",
    "industrial_proxy_vs_teacher": "ppo_final_pool_320_efficiency_proxy_vs_teacher_metrics.csv",
    "version_summary": "ppo_final_pool_320_version_summary.csv",
    "best_candidates": "ppo_final_pool_320_best_candidates_by_N.csv",
    "final_expansion_vs_prior": "final_expansion_vs_prior_ppo_by_N.csv",
}

EXPECTED_COUNTS = {12: 40, 16: 40, 24: 120, 40: 120}
FINAL_VERDICT = "PASS_STAGEX_PPO_FINAL_POOL_320_EVIDENCE_FREEZE_BOUNDED_NO_NEW_RECORDS"


def ensure_dirs() -> None:
    for path in [OUT_ROOT, FROZEN, HASHES, MANUSCRIPT, REPORTS, DOCS_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


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


def preflight() -> tuple[str, dict]:
    checks = []

    def add(name: str, ok: bool, detail: str, value: object = "") -> None:
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail, "value": value})

    for key, path in INPUTS.items():
        add(f"{key}_exists", path.exists(), str(path))

    manifest_ok = False
    manifest_verdict = None
    if INPUTS["stageW_manifest"].exists():
        manifest = json.loads(INPUTS["stageW_manifest"].read_text(encoding="utf-8"))
        manifest_verdict = manifest.get("final_verdict")
        manifest_ok = manifest_verdict == "PASS_STAGEW_PPO_FINAL_POOL_320_BOUNDED_NO_NEW_RECORDS"
    add("stageW_final_verdict", manifest_ok, "expected PASS_STAGEW_PPO_FINAL_POOL_320_BOUNDED_NO_NEW_RECORDS", manifest_verdict)

    counts = {}
    rows = None
    if INPUTS["final_ppo_pool"].exists():
        df = pd.read_csv(INPUTS["final_ppo_pool"])
        rows = len(df)
        counts = {int(k): int(v) for k, v in df["n"].astype(int).value_counts().sort_index().items()}
    add("final_ppo_pool_row_count", rows == 320, "expected 320 rows", rows)
    add("final_ppo_pool_by_N_counts", counts == EXPECTED_COUNTS, json.dumps(EXPECTED_COUNTS), json.dumps(counts))

    for key in ["ranking_full", "claim_decision", "stageW_report", "stageW_claim_boundary", "combined552_reference"]:
        add(f"{key}_ready", INPUTS[key].exists(), str(INPUTS[key]))

    verdict = "PASS_STAGEX_PPO_FINAL_EVIDENCE_PREFLIGHT_READY" if all(row["status"] == "PASS" for row in checks) else "FAIL_STAGEX_PPO_FINAL_EVIDENCE_PREFLIGHT_BLOCKED"
    summary = {
        "branch": git_branch(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "verdict": verdict,
        "stageW_final_verdict": manifest_verdict,
        "final_PPO_pool_count": rows,
        "by_N_counts": {str(k): int(v) for k, v in counts.items()},
        "checks": checks,
        "no_Abaqus": True,
        "no_ODB_opening": True,
        "no_ODB_extraction": True,
        "no_solver": True,
        "no_training": True,
        "no_candidate_generation": True,
    }
    (REPORTS / "stageX_preflight_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return verdict, summary


def copy_frozen_tables() -> dict[str, str]:
    mapping = {
        "final_ppo_pool": (INPUTS["final_ppo_pool"], FROZEN / "FROZEN_PPO_final_pool_320_teacher_metrics.csv"),
        "combined552_plus_ppo": (INPUTS["combined552_plus_ppo"], FROZEN / "FROZEN_combined552_plus_PPO_final_pool_320_analysis_dataset.csv"),
        "ranking_full": (INPUTS["ranking_full"], FROZEN / "FROZEN_PPO_final_pool_320_teacher_metric_ranking_full.csv"),
        "claim_decision": (INPUTS["claim_decision"], FROZEN / "FROZEN_PPO_final_pool_320_claim_decision_table.csv"),
    }
    for role, filename in OPTIONAL_STAGEW_TABLES.items():
        src = STAGEW_TABLES / filename
        if src.exists():
            mapping[role] = (src, FROZEN / f"FROZEN_{filename}")

    out: dict[str, str] = {}
    for role, (src, dst) in mapping.items():
        shutil.copy2(src, dst)
        out[role] = str(dst)
    return out


def write_manuscript_tables() -> dict[str, str]:
    pool = pd.read_csv(INPUTS["final_ppo_pool"])
    ranking = pd.read_csv(INPUTS["ranking_full"])
    claim = pd.read_csv(INPUTS["claim_decision"])
    topk_by_n = pd.read_csv(STAGEW_TABLES / "ppo_final_pool_320_topk_summary_by_N.csv")
    bootstrap = pd.read_csv(STAGEW_TABLES / "ppo_final_pool_320_vs_bootstrap_random_reference_global.csv")
    best = pd.read_csv(STAGEW_TABLES / "ppo_final_pool_320_best_candidates_by_N.csv")
    version_counts = pool["ppo_version"].value_counts().to_dict()
    n_counts = {f"N{int(k)}": int(v) for k, v in pool["n"].astype(int).value_counts().sort_index().items()}

    composition_rows = [
        {"category": "total_PPO_cases", "label": "all", "count": 320, "notes": "teacher metrics extracted 320/320"},
    ]
    for version in ["v01", "v02K2", "v03", "final_expansion"]:
        composition_rows.append({"category": "by_version", "label": version, "count": int(version_counts.get(version, 0)), "notes": ""})
    for label, count in n_counts.items():
        composition_rows.append({"category": "by_N", "label": label, "count": count, "notes": ""})
    composition_rows.append({"category": "teacher_metrics_extracted", "label": "Abaqus teacher metrics", "count": 320, "notes": "320/320"})
    composition_path = MANUSCRIPT / "PPO_final_pool_320_composition_for_manuscript.csv"
    pd.DataFrame(composition_rows).to_csv(composition_path, index=False)

    boot_row = bootstrap[bootstrap["statistic"] == "primary_top25_any_unique"].iloc[0]
    lex_best = best[best["criterion"] == "primary_lex"].sort_values("N")
    result_rows = [
        {"result": "new_records_vs_combined552", "value": "0", "notes": "No PPO candidate beat native combined552 best records."},
        {"result": "primary_top25_any_count", "value": "106", "notes": "Candidates in at least one primary top25 region."},
        {"result": "bootstrap_primary_top25_any_mean", "value": f"{boot_row['bootstrap_mean']:.3f}", "notes": "Equal-budget random-reference bootstrap."},
        {"result": "bootstrap_primary_top25_any_q05_q95", "value": f"{boot_row['bootstrap_q05']:.0f} / {boot_row['bootstrap_q95']:.0f}", "notes": "Equal-budget random-reference bootstrap."},
        {"result": "bootstrap_interpretation", "value": str(boot_row["interpretation"]), "notes": "Overall primary top25 enrichment is weak."},
        {"result": "SurfaceT_top10_enrichment", "value": "yes", "notes": "SurfaceT signal is not evidence for U2/lex superiority."},
        {"result": "final_expansion_best_lex_improvement", "value": "no", "notes": "Final expansion did not improve prior best lex rank for any N."},
        {"result": "combined552_new_records", "value": "0", "notes": "No combined552 new records."},
    ]
    for _, row in lex_best.iterrows():
        result_rows.append(
            {
                "result": f"best_primary_lex_N{int(row['N'])}",
                "value": f"{row['strategy_name']} / ref lex rank {int(row['ref_rank'])}",
                "notes": f"ppo_version={row['ppo_version']}",
            }
        )
    main_results_path = MANUSCRIPT / "PPO_final_pool_320_main_results_for_manuscript.csv"
    pd.DataFrame(result_rows).to_csv(main_results_path, index=False)

    safe_rows = [
        ("320 PPO-generated scan orders were teacher-metric extracted.", "SAFE", "Supported by final PPO pool table."),
        ("PPO generated legal and executable scan-order permutations.", "SAFE", "Legality was audited before handoff and metrics extracted after solver completion."),
        ("PPO showed bounded small-N competitiveness.", "SAFE_BOUNDED", "Best N12/N16 lex ranks were 6 and 2."),
        ("PPO showed SurfaceT-related signals but not U2/lex dominance.", "SAFE_BOUNDED", "SurfaceT top10 enrichment was observed; primary bootstrap was weak."),
        ("PPO did not beat combined552 best records.", "SAFE", "New-record audit has zero rows."),
        ("PPO did not outperform the mature surrogate-assisted optimiser.", "SAFE", "No new records and weak random-reference enrichment."),
        ("Final physical claims are based on Abaqus teacher metrics, not surrogate scores.", "SAFE", "Stage X freezes teacher-metric evidence."),
        ("Industrial-efficiency descriptors remain proxies only.", "SAFE_BOUNDARY", "No physical industrial-efficiency validation."),
    ]
    safe_path = MANUSCRIPT / "PPO_final_pool_320_safe_claims_for_manuscript.csv"
    pd.DataFrame(safe_rows, columns=["claim", "status", "evidence_or_boundary"]).to_csv(safe_path, index=False)

    # Copy claim-decision into manuscript folder as a convenience summary.
    claim.to_csv(MANUSCRIPT / "PPO_final_pool_320_claim_decision_for_manuscript.csv", index=False)
    return {
        "composition": str(composition_path),
        "main_results": str(main_results_path),
        "safe_claims": str(safe_path),
        "claim_decision_copy": str(MANUSCRIPT / "PPO_final_pool_320_claim_decision_for_manuscript.csv"),
    }


def write_docs(frozen_paths: dict[str, str], manuscript_paths: dict[str, str], hash_path: Path, manifest_path: Path) -> dict[str, str]:
    evidence_index = DOCS_ROOT / "PPO_FINAL_POOL_320_EVIDENCE_INDEX.md"
    final_claim = DOCS_ROOT / "PPO_FINAL_POOL_320_FINAL_CLAIM_BOUNDARY.md"
    memo = DOCS_ROOT / "PPO_FINAL_POOL_320_MANUSCRIPT_INTEGRATION_MEMO.md"
    report = DOCS_ROOT / "PPO_FINAL_POOL_320_EVIDENCE_FREEZE_REPORT.md"

    frozen_list = "\n".join(f"- `{path}`" for path in frozen_paths.values())
    manuscript_list = "\n".join(f"- `{path}`" for path in manuscript_paths.values())
    evidence_index.write_text(
        f"""# PPO Final Pool 320 Evidence Index

## Evidence Chain Overview

- PPO v01: 32 teacher-metric-extracted cases.
- PPO v02K2: 32 teacher-metric-extracted targeted N24/N40 cases.
- PPO v03: 32 teacher-metric-extracted lex-primary N24/N40 cases.
- Final expansion: 224 teacher-metric-extracted cases.
- Stage W ranking: final 320-case comparison against native combined552.
- Stage X freeze: final frozen evidence, hashes, manuscript tables, and claim boundary.

## Frozen Tables

{frozen_list}

## Manuscript Tables

{manuscript_list}

## Hash Table

- `{hash_path}`

## Reports and Claim Boundaries

- Stage W report: `{INPUTS['stageW_report']}`
- Stage W claim boundary: `{INPUTS['stageW_claim_boundary']}`
- Stage X report: `{report}`
- Stage X final claim boundary: `{final_claim}`

## Final Verdict

`{FINAL_VERDICT}`

## Exact Safe Claim Paragraph

A 320-case PPO-generated scan-order pool was independently teacher-metric extracted using Abaqus. The pool demonstrates legal, executable and teacher-evaluable policy-generated scan orders with bounded small-N competitiveness and SurfaceT-related signals, but it produced no new combined552 records and did not outperform the mature surrogate-assisted optimisation reference.
""",
        encoding="utf-8",
    )

    final_claim.write_text(
        """# PPO Final Pool 320 Final Claim Boundary

## Safe Claims

- A 320-case PPO-generated scan-order pool was independently teacher-metric extracted using Abaqus.
- The PPO pool contains legal, executable and teacher-evaluated scan-order candidates across N12, N16, N24 and N40.
- PPO achieved bounded competitiveness in selected regimes, especially small-N lex ranks and SurfaceT-related signals.
- PPO did not produce any new combined552 best record.
- PPO did not outperform the mature surrogate-assisted optimiser.
- The final PPO result defines a practical boundary of surrogate-trained policy-gradient generation under the current surrogate reward and finite-element teacher setup.

## Unsafe Claims

- PPO beats combined552 best.
- PPO is superior to the surrogate-assisted optimiser.
- PPO solves N24/N40.
- PPO provides experimentally validated industrial-efficiency improvement.
- PPO performance can be judged from surrogate score alone.
- PPO was online Abaqus-RL.
- PPO generated the strongest scan orders in the full study.
""",
        encoding="utf-8",
    )

    memo.write_text(
        """# PPO Final Pool 320 Manuscript Integration Memo

## Recommended Final Positioning

The manuscript should present PPO as a strict policy-gradient evidence chain and large-scale teacher-validated policy-generation addendum, not as the strongest optimiser in the study.

## Recommended Abstract Sentence

"A 320-case PPO-generated scan-order pool was independently evaluated by Abaqus teacher simulations. The pool demonstrated legal executable policy generation and bounded teacher-validated competitiveness, but did not exceed the best records from the mature surrogate-assisted optimisation reference."

## Recommended Results Paragraph

"The final PPO evidence pool contained 320 teacher-metric-extracted candidates across N12, N16, N24 and N40. No PPO candidate produced a new combined552 record. The strongest PPO lexicographic ranks occurred in N12 and N16, while N24 and N40 remained limited in U2/lex performance. Although 106 PPO candidates entered at least one primary top25 region, equal-budget bootstrap comparison indicated weak global enrichment relative to the existing teacher-labelled reference distribution. These results support PPO-generated policy feasibility and bounded competitiveness, rather than dominance over the mature surrogate-assisted optimiser."

## Recommended Discussion Paragraph

"The PPO results clarify the distinction between policy generation and mature surrogate-assisted optimisation. The surrogate-assisted loop remains the stronger optimiser in the present evidence pool, whereas PPO provides a reusable policy-gradient mechanism that can generate legal, executable and teacher-evaluable scan orders. The lack of new records and the weak bootstrap enrichment indicate a practical boundary of the current surrogate-trained PPO formulation, especially for high-N U2/lex optimisation."

## What To Avoid

- Do not say PPO found the best scan orders.
- Do not say PPO outperformed surrogate-assisted optimisation.
- Do not say PPO solved high-N scan-order optimisation.
- Do not treat SurfaceT-only signals as U2/lex dominance.
- Do not claim industrial efficiency improvement without validation.
""",
        encoding="utf-8",
    )

    report.write_text(
        f"""# PPO Final Pool 320 Evidence Freeze Report

## Purpose

Stage X freezes the final 320-case PPO teacher-metric evidence package and creates manuscript-facing summary materials.

## Final Evidence Chain

PPO v01, PPO v02K2, PPO v03 and the final expansion together form a 320-case teacher-metric-extracted PPO pool.

## Final PPO Pool Composition

- Total: 320
- N12: 40
- N16: 40
- N24: 120
- N40: 120
- v01: 32
- v02K2: 32
- v03: 32
- final expansion: 224

## Input Integrity

Stage X preflight passed and confirmed Stage W verdict `PASS_STAGEW_PPO_FINAL_POOL_320_BOUNDED_NO_NEW_RECORDS`.

## Frozen Table List

{frozen_list}

## Hash Table

`{hash_path}`

## Main Stage W Results

- New records vs combined552: 0
- Primary top25-any count: 106
- Equal-budget bootstrap primary top25-any mean: 163.106
- Bootstrap q05/q95: 154 / 173
- Bootstrap interpretation: weak
- SurfaceT top10 enrichment was observed but is not U2/lex dominance.

## New-Record Audit

No PPO candidate produced a new combined552 record.

## Top-k Competitiveness

Top-k evidence is bounded. The strongest lexicographic PPO ranks are in N12 and N16.

## Bootstrap Interpretation

The 320-case PPO pool is weak relative to equal-budget random-reference draws for overall primary top25 enrichment.

## Baseline-Family Comparison

PPO can be compared to identified conventional baseline labels where available, but label-derived baseline comparisons should not be overclaimed.

## Industrial-Efficiency Proxy Caveat

Industrial-efficiency fields are sequence descriptors only and are not physically validated efficiency measurements.

## Claim Boundary

See `{final_claim}`.

## Manuscript Integration Summary

Use PPO as a large-scale teacher-metric-extracted policy-generation addendum, not as the strongest optimiser.

## Final Verdict

{FINAL_VERDICT}
""",
        encoding="utf-8",
    )
    return {
        "evidence_index": str(evidence_index),
        "final_claim_boundary": str(final_claim),
        "manuscript_integration_memo": str(memo),
        "final_evidence_report": str(report),
    }


def write_manifest(frozen_paths: dict[str, str], manuscript_paths: dict[str, str], doc_paths: dict[str, str], hash_path: Path) -> Path:
    manifest_path = OUT_ROOT / "stageX_PPO_final_pool_320_evidence_freeze_manifest.json"
    manifest = {
        "branch": git_branch(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "final_PPO_pool_count": 320,
        "by_N_counts": {"12": 40, "16": 40, "24": 120, "40": 120},
        "input_paths": {k: str(v) for k, v in INPUTS.items()},
        "frozen_table_paths": frozen_paths,
        "manuscript_table_paths": manuscript_paths,
        "hash_table_path": str(hash_path),
        "evidence_index_path": doc_paths["evidence_index"],
        "final_claim_boundary_path": doc_paths["final_claim_boundary"],
        "manuscript_integration_memo_path": doc_paths["manuscript_integration_memo"],
        "final_report_path": doc_paths["final_evidence_report"],
        "no_Abaqus": True,
        "no_ODB_opening": True,
        "no_ODB_extraction": True,
        "no_solver": True,
        "no_datacheck": True,
        "no_enqueue": True,
        "no_training": True,
        "no_candidate_generation": True,
        "no_commit_or_push": True,
        "final_verdict": FINAL_VERDICT,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def write_hash_table(frozen_paths: dict[str, str], doc_paths: dict[str, str], manifest_path: Path, hash_path: Path) -> None:
    rows = []

    def add(role: str, original: str | Path, frozen: str | Path | None = None) -> None:
        path = Path(frozen or original)
        rows.append(
            {
                "file_role": role,
                "original_path": str(original),
                "frozen_path": str(frozen or original),
                "file_size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
        )

    for role, frozen in frozen_paths.items():
        add(f"frozen_table_{role}", frozen, frozen)
    add("stageW_report", INPUTS["stageW_report"])
    add("stageW_claim_boundary", INPUTS["stageW_claim_boundary"])
    add("stageW_manifest", INPUTS["stageW_manifest"])
    add("stageX_report", doc_paths["final_evidence_report"])
    add("stageX_claim_boundary", doc_paths["final_claim_boundary"])
    add("stageX_manuscript_integration_memo", doc_paths["manuscript_integration_memo"])
    add("stageX_evidence_index", doc_paths["evidence_index"])
    add("stageX_manifest", manifest_path)

    with hash_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ensure_dirs()
    preflight_verdict, preflight_summary = preflight()
    if preflight_verdict.startswith("FAIL"):
        print(json.dumps(preflight_summary, indent=2))
        return

    hash_path = HASHES / "FROZEN_PPO_final_pool_320_file_hashes.csv"
    frozen_paths = copy_frozen_tables()
    manuscript_paths = write_manuscript_tables()
    # Pass known hash/manifest paths into docs before writing manifest and hashes.
    manifest_path = OUT_ROOT / "stageX_PPO_final_pool_320_evidence_freeze_manifest.json"
    doc_paths = write_docs(frozen_paths, manuscript_paths, hash_path, manifest_path)
    manifest_path = write_manifest(frozen_paths, manuscript_paths, doc_paths, hash_path)
    write_hash_table(frozen_paths, doc_paths, manifest_path, hash_path)

    result = {
        "preflight_verdict": preflight_verdict,
        "final_verdict": FINAL_VERDICT,
        "frozen_final_pool": frozen_paths["final_ppo_pool"],
        "frozen_ranking": frozen_paths["ranking_full"],
        "frozen_combined552_plus_ppo": frozen_paths["combined552_plus_ppo"],
        "frozen_claim_decision": frozen_paths["claim_decision"],
        "hash_table": str(hash_path),
        "manuscript_tables": manuscript_paths,
        "docs": doc_paths,
        "manifest": str(manifest_path),
        "no_Abaqus": True,
        "no_ODB_opening": True,
        "no_solver": True,
        "no_training": True,
        "no_candidate_generation": True,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
