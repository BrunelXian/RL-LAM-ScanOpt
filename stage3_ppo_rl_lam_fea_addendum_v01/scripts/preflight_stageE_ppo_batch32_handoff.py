"""Preflight PPO batch32 for Stage E CAE/INP handoff.

Normal Python only. This script does not run Abaqus, solver, datacheck,
abqjobpilot, enqueue, or ODB extraction.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
NAMESPACE = "stage3_ppo_rl_lam_fea_addendum_v01"
BATCH = "stage3_ppo_policy_only_batch32_v01"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / NAMESPACE / "stageE_teacher_validation_handoff"
CHECKS_DIR = OUTPUT_ROOT / "checks"
SELECTED_CSV = PROJECT_ROOT / "outputs" / NAMESPACE / "ppo_candidate_generation" / "selected_batch32" / "ppo_policy_only_candidate_batch32.csv"
SCAN_JSON_DIR = PROJECT_ROOT / "outputs" / NAMESPACE / "ppo_candidate_generation" / "selected_batch32" / "scan_orders"
CASE_ROOT = PROJECT_ROOT / "cae_model" / BATCH
STAGE_C_CHECKPOINT = PROJECT_ROOT / "outputs" / NAMESPACE / "ppo_training" / "checkpoints" / "maskable_ppo_lam_scan_order_final.zip"
COMBINED552 = PROJECT_ROOT / "outputs" / "stage3_run_78_final_evidence_freeze_package" / "FROZEN_stage3_native_combined552_teacher_dataset.csv"
CSV_OUT = CHECKS_DIR / "stageE_preflight_ppo_batch32.csv"
JSON_OUT = CHECKS_DIR / "stageE_preflight_ppo_batch32_summary.json"
BASES = {
    12: PROJECT_ROOT / "cae_model" / "12track_full" / "sanity_base" / "12track_sanity_base.cae",
    16: PROJECT_ROOT / "cae_model" / "16track_full" / "sanity_base" / "16track_sanity_base.cae",
    24: PROJECT_ROOT / "cae_model" / "24track_full" / "sanity_base" / "24track_sanity_base.cae",
    40: PROJECT_ROOT / "cae_model" / "40track_full" / "sanity_base" / "40track_sanity_base.cae",
}
SOLVER_EXTS = (".odb", ".sim", ".sta", ".dat", ".msg", ".lck")
GENERATED_EXTS = (".cae", ".inp", ".jnl")


def parse_order_compact(text: str) -> list[int]:
    return [int(item) for item in str(text).split(",") if item != ""]


def order_legal(n: int, order: list[int]) -> bool:
    return len(order) == n and sorted(order) == list(range(n))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def combined_hashes() -> set[str]:
    if not COMBINED552.exists():
        return set()
    df = pd.read_csv(COMBINED552, usecols=["n", "order_json"])
    import hashlib

    hashes = set()
    for _, row in df.iterrows():
        order = json.loads(row["order_json"])
        payload = "N{}:{}".format(int(row["n"]), ",".join(str(int(x)) for x in order))
        hashes.add(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16])
    return hashes


def main() -> int:
    CHECKS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    blockers = []
    warnings = []
    selected_exists = SELECTED_CSV.exists()
    rows.append({"check": "selected_batch_csv_exists", "status": "PASS" if selected_exists else "FAIL", "details": str(SELECTED_CSV)})
    if not selected_exists:
        blockers.append("selected batch CSV missing")
    else:
        df = pd.read_csv(SELECTED_CSV)
        counts = {str(int(k)): int(v) for k, v in df.groupby("n").size().sort_index().items()}
        expected_counts = {"12": 8, "16": 8, "24": 8, "40": 8}
        checks = {
            "selected_count_32": len(df) == 32,
            "counts_8_each": counts == expected_counts,
            "no_N32": not (df["n"].astype(int) == 32).any(),
            "every_order_legal": all(order_legal(int(r["n"]), parse_order_compact(r["order_compact"])) for _, r in df.iterrows()),
            "no_duplicate_order_hash": not df["order_hash"].duplicated().any(),
            "candidate_source_ppo": (df["candidate_source"] == "PPO_checkpoint_inference").all(),
            "ppo_checkpoint_exists_and_matches": "ppo_checkpoint" in df.columns and (df["ppo_checkpoint"].astype(str) == str(STAGE_C_CHECKPOINT)).all() and STAGE_C_CHECKPOINT.exists(),
            "teacher_validated_false": not df["teacher_validated"].astype(bool).any(),
            "abaqus_validated_false": not df["abaqus_validated"].astype(bool).any(),
        }
        for name, ok in checks.items():
            rows.append({"check": name, "status": "PASS" if bool(ok) else "FAIL", "details": json.dumps(counts) if name == "counts_8_each" else ""})
            if not bool(ok):
                blockers.append(name)

        combo_hashes = combined_hashes()
        duplicate_rows = df[df["order_hash"].isin(combo_hashes)]
        duplicate_count = int(len(duplicate_rows))
        duplicate_ok = duplicate_count == 1 and int(duplicate_rows.iloc[0]["n"]) == 12
        rows.append({"check": "one_N12_duplicate_vs_combined552_flagged_recovery_anchor", "status": "PASS" if duplicate_ok else "WARNING", "details": duplicate_rows[["n", "strategy_name", "order_hash"]].to_json(orient="records")})
        if not duplicate_ok:
            warnings.append("duplicate vs combined552 pattern differs from expected one N12 duplicate")

        for _, r in df.iterrows():
            strategy = str(r["strategy_name"])
            json_path = SCAN_JSON_DIR / "scan_order_{}.json".format(strategy)
            ok = json_path.exists()
            detail = str(json_path)
            if ok:
                payload = load_json(json_path)
                order_from_json = [int(x) for x in payload.get("order", [])]
                order_from_csv = parse_order_compact(r["order_compact"])
                same = order_from_json == order_from_csv
                ok = ok and same
                detail += " same_order={}".format(same)
            rows.append({"check": "scan_json_exists_and_matches:{}".format(strategy), "status": "PASS" if ok else "FAIL", "details": detail})
            if not ok:
                blockers.append("scan json mismatch {}".format(strategy))

    for n, base in BASES.items():
        ok = base.exists()
        rows.append({"check": "base_cae_exists_N{}".format(n), "status": "PASS" if ok else "FAIL", "details": str(base)})
        if not ok:
            blockers.append("missing base CAE N{}".format(n))

    if CASE_ROOT.exists():
        lcks = list(CASE_ROOT.rglob("*.lck"))
        solver_files = [p for ext in SOLVER_EXTS for p in CASE_ROOT.rglob("*{}".format(ext))]
        generated_files = [p for ext in GENERATED_EXTS for p in CASE_ROOT.rglob("*{}".format(ext))]
    else:
        lcks = []
        solver_files = []
        generated_files = []
    rows.append({"check": "case_root_no_active_lck", "status": "PASS" if not lcks else "FAIL", "details": ";".join(str(p) for p in lcks)})
    rows.append({"check": "case_root_no_solver_outputs", "status": "PASS" if not solver_files else "FAIL", "details": ";".join(str(p) for p in solver_files[:20])})
    rows.append({"check": "case_root_no_existing_cae_inp_jnl", "status": "PASS" if not generated_files else "FAIL", "details": ";".join(str(p) for p in generated_files[:20])})
    if lcks:
        blockers.append("active lck under case root")
    if solver_files:
        blockers.append("solver outputs under case root")
    if generated_files:
        blockers.append("existing CAE/INP/JNL under case root")

    with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["check", "status", "details"])
        writer.writeheader()
        writer.writerows(rows)
    verdict = "FAIL_STAGEE_PPO_BATCH32_PREFLIGHT_BLOCKED" if blockers else ("WARNING_STAGEE_PPO_BATCH32_PREFLIGHT_REVIEW" if warnings else "PASS_STAGEE_PPO_BATCH32_PREFLIGHT_READY")
    summary = {
        "verdict": verdict,
        "blockers": blockers,
        "warnings": warnings,
        "csv_path": str(CSV_OUT),
        "selected_batch": str(SELECTED_CSV),
        "case_root": str(CASE_ROOT),
        "no_solver": True,
        "no_datacheck": True,
        "no_ODB": True,
        "no_abqjobpilot_execution": True,
        "no_enqueue_execution": True,
    }
    JSON_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if verdict != "FAIL_STAGEE_PPO_BATCH32_PREFLIGHT_BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
