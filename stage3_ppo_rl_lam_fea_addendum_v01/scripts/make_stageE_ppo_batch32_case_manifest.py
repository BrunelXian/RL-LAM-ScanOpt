"""Create Stage E PPO batch32 case manifest and case directories."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
NAMESPACE = "stage3_ppo_rl_lam_fea_addendum_v01"
BATCH = "stage3_ppo_policy_only_batch32_v01"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / NAMESPACE / "stageE_teacher_validation_handoff"
MANIFEST_DIR = OUTPUT_ROOT / "manifest"
SELECTED_CSV = PROJECT_ROOT / "outputs" / NAMESPACE / "ppo_candidate_generation" / "selected_batch32" / "ppo_policy_only_candidate_batch32.csv"
SCAN_JSON_DIR = PROJECT_ROOT / "outputs" / NAMESPACE / "ppo_candidate_generation" / "selected_batch32" / "scan_orders"
CASE_ROOT = PROJECT_ROOT / "cae_model" / BATCH
CSV_OUT = MANIFEST_DIR / "stageE_ppo_batch32_case_manifest.csv"
JSON_OUT = MANIFEST_DIR / "stageE_ppo_batch32_case_manifest.json"
BASES = {
    12: PROJECT_ROOT / "cae_model" / "12track_full" / "sanity_base" / "12track_sanity_base.cae",
    16: PROJECT_ROOT / "cae_model" / "16track_full" / "sanity_base" / "16track_sanity_base.cae",
    24: PROJECT_ROOT / "cae_model" / "24track_full" / "sanity_base" / "24track_sanity_base.cae",
    40: PROJECT_ROOT / "cae_model" / "40track_full" / "sanity_base" / "40track_sanity_base.cae",
}


def order_list(text: str) -> list[int]:
    return [int(x) for x in str(text).split(",") if x != ""]


def main() -> int:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    CASE_ROOT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SELECTED_CSV)
    rows = []
    for _, r in df.sort_values(["n", "batch_index"]).iterrows():
        n = int(r["n"])
        strategy = str(r["strategy_name"])
        job = "J2D_{}".format(strategy)
        case_dir = CASE_ROOT / "N{}_{}".format(n, strategy)
        case_dir.mkdir(parents=True, exist_ok=True)
        source_json = SCAN_JSON_DIR / "scan_order_{}.json".format(strategy)
        target_json = case_dir / "scan_order_{}.json".format(strategy)
        scan_order = order_list(r["order_compact"])
        duplicate = bool(r["duplicate_order_hash_in_combined552"])
        duplicate_role = "recovery_anchor" if duplicate and n == 12 else ""
        payload = {
            "batch": BATCH,
            "n": n,
            "strategy_name": strategy,
            "job_name": job,
            "scan_order": scan_order,
            "order_json": r["order_json"],
            "order_compact": r["order_compact"],
            "candidate_source": "PPO_checkpoint_inference",
            "ppo_checkpoint": r["ppo_checkpoint"],
            "predicted_surrogate_reward_lex": float(r["predicted_surrogate_reward_lex"]),
            "generation_mode": r["generation_mode"],
            "selection_tag": r["selection_tag"],
            "order_hash": r["order_hash"],
            "duplicate_vs_combined552": duplicate,
            "duplicate_role": duplicate_role,
            "teacher_validated": False,
            "abaqus_validated": False,
            "notes": "PPO-generated Stage D candidate converted for Stage E Abaqus teacher-validation handoff; not physically validated yet.",
        }
        target_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if source_json.exists():
            shutil.copyfile(str(source_json), str(case_dir / "stageD_source_{}".format(source_json.name)))
        row = {
            "batch": BATCH,
            "n": n,
            "strategy_name": strategy,
            "job_name": job,
            "scan_order": json.dumps(scan_order, separators=(",", ":")),
            "order_json": r["order_json"],
            "order_compact": r["order_compact"],
            "order_hash": r["order_hash"],
            "duplicate_vs_combined552": duplicate,
            "duplicate_role": duplicate_role,
            "candidate_source": "PPO_checkpoint_inference",
            "ppo_checkpoint": r["ppo_checkpoint"],
            "teacher_validated": False,
            "abaqus_validated": False,
            "base_cae": str(BASES[n]),
            "case_dir": str(case_dir),
            "scan_order_json": str(target_json),
            "cae_path": str(case_dir / "{}.cae".format(job)),
            "inp_path": str(case_dir / "{}.inp".format(job)),
            "jnl_path": str(case_dir / "{}.jnl".format(job)),
            "generation_log_path": str(case_dir / "{}_stageE_generation_log.json".format(job)),
        }
        rows.append(row)
    with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    JSON_OUT.write_text(json.dumps({"batch": BATCH, "case_root": str(CASE_ROOT), "row_count": len(rows), "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps({"manifest_csv": str(CSV_OUT), "manifest_json": str(JSON_OUT), "row_count": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
