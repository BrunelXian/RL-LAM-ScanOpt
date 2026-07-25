"""Create Stage L PPO v02K2 N24/N40 case manifest and case directories."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


PROJECT_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
NAMESPACE = "stage3_ppo_rl_lam_fea_addendum_v02_targeted_N24_N40"
BATCH_NAME = "stage3_ppo_v02K2_targeted_N24_N40_batch32_v01"
SELECTED_CSV = PROJECT_ROOT / "outputs" / NAMESPACE / "stageK2_n40_completion" / "selected_batch32_K2" / "v02K2_ppo_targeted_N24_N40_candidate_batch32.csv"
SCAN_JSON_DIR = PROJECT_ROOT / "outputs" / NAMESPACE / "stageK2_n40_completion" / "selected_batch32_K2" / "scan_orders"
CASE_ROOT = PROJECT_ROOT / "cae_model" / BATCH_NAME
MANIFEST_DIR = PROJECT_ROOT / "outputs" / NAMESPACE / "stageL_CAE_INP_handoff" / "manifest"
CSV_OUT = MANIFEST_DIR / "stageL_v02K2_case_manifest.csv"
JSON_OUT = MANIFEST_DIR / "stageL_v02K2_case_manifest.json"
BASES = {
    24: PROJECT_ROOT / "cae_model" / "24track_full" / "sanity_base" / "24track_sanity_base.cae",
    40: PROJECT_ROOT / "cae_model" / "40track_full" / "sanity_base" / "40track_sanity_base.cae",
}


def parse_order(value: str) -> list[int]:
    value = str(value).strip()
    if value.startswith("["):
        return [int(x) for x in json.loads(value)]
    return [int(x) for x in value.split(",") if x]


def main() -> int:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    CASE_ROOT.mkdir(parents=True, exist_ok=True)
    with SELECTED_CSV.open(newline="", encoding="utf-8-sig") as f:
        input_rows = list(csv.DictReader(f))
    rows = []
    for row in sorted(input_rows, key=lambda r: (int(r["n"]), r["strategy_name"])):
        n = int(row["n"])
        strategy = row["strategy_name"]
        order = parse_order(row["order_json"])
        job_name = f"J2D_{strategy}"
        case_dir = CASE_ROOT / f"N{n}{strategy}"
        case_dir.mkdir(parents=True, exist_ok=True)
        source_json = SCAN_JSON_DIR / f"scan_order_{strategy}.json"
        target_json = case_dir / f"scan_order_{strategy}.json"
        if not target_json.exists():
            shutil.copyfile(source_json, target_json)
        manifest_row = {
            "strategy_name": strategy,
            "n": n,
            "order_json": str(target_json),
            "source_order_json": str(source_json),
            "scan_order": json.dumps(order, separators=(",", ":")),
            "order_compact": row["order_compact"],
            "order_hash": row["order_hash"],
            "candidate_source": row["candidate_source"],
            "ppo_checkpoint": row["ppo_checkpoint"],
            "ppo_seed": row["ppo_seed"],
            "selected_by": row["selected_by"],
            "case_dir": str(case_dir),
            "job_name": job_name,
            "expected_cae": str(case_dir / f"{job_name}.cae"),
            "expected_inp": str(case_dir / f"{job_name}.inp"),
            "expected_jnl": str(case_dir / f"{job_name}.jnl"),
            "generation_log": str(case_dir / f"{job_name}_stageL_generation_log.json"),
            "base_cae": str(BASES[n]),
            "batch_name": BATCH_NAME,
            "teacher_validated": row["teacher_validated"],
            "abaqus_validated": row["abaqus_validated"],
            "stageK2_role": row["stageK2_role"],
        }
        rows.append(manifest_row)
    with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    JSON_OUT.write_text(json.dumps({"batch_name": BATCH_NAME, "case_root": str(CASE_ROOT), "row_count": len(rows), "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps({"manifest_csv": str(CSV_OUT), "manifest_json": str(JSON_OUT), "row_count": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
