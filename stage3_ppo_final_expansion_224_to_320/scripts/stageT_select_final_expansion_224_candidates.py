"""Select the Stage T 224-case final PPO expansion set.

Selection is from PPO checkpoint-inference rollout pools only. This script does
not train PPO/surrogates, run Abaqus, open ODB files, run solver/datacheck,
enqueue jobs, or generate CAE/INP/JNL files.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
NS = "stage3_ppo_final_expansion_224_to_320"
OUT_ROOT = ROOT / "outputs" / NS
DOCS_ROOT = ROOT / "docs" / NS
POOL_PATH = OUT_ROOT / "rollout_pools" / "ppo_final_expansion_rollout_pool.csv"
SELECTED_DIR = OUT_ROOT / "selected_candidates"
BATCH_DIR = SELECTED_DIR / "batches"
JSON_DIR = SELECTED_DIR / "scan_orders"
AUDIT_DIR = OUT_ROOT / "audits"
HANDOFF_DIR = OUT_ROOT / "handoff_preview"

COMBINED552 = ROOT / "outputs" / "stage3_run_78_final_evidence_freeze_package" / "FROZEN_stage3_native_combined552_teacher_dataset.csv"
V01_METRICS = ROOT / "outputs" / "stage3_ppo_rl_lam_fea_addendum_v01" / "stageI_final_ppo_evidence_freeze" / "frozen_tables" / "FROZEN_PPO_batch32_teacher_metrics.csv"
V02K2_METRICS = ROOT / "outputs" / "stage3_ppo_rl_lam_fea_addendum_v02_targeted_N24_N40" / "stageM_ODB_teacher_metric_extraction" / "stageM_v02K2_teacher_metrics.csv"
V03_METRICS = ROOT / "outputs" / "stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40" / "stageR_ODB_teacher_metric_extraction" / "stageR_v03_teacher_metrics.csv"

TARGET_COUNTS = {12: 32, 16: 32, 24: 80, 40: 80}
BUCKET_COUNTS = {
    32: {"quality": 11, "diversity": 8, "efficiency": 6, "novelty": 3, "baseline_proximity": 4},
    80: {"quality": 28, "diversity": 20, "efficiency": 16, "novelty": 8, "baseline_proximity": 8},
}
SUPPORTED_N = (12, 16, 24, 40)


def git_branch() -> str:
    try:
        result = subprocess.run(["git", "branch", "--show-current"], cwd=str(ROOT), capture_output=True, text=True, check=False)
        return result.stdout.strip() or "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def parse_order(value: object) -> list[int] | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, list):
        return [int(v) for v in value]
    text = str(value).strip()
    if not text:
        return None
    if text.startswith("["):
        try:
            return [int(v) for v in json.loads(text)]
        except Exception:
            pass
    parts = [p for p in text.replace("|", ",").replace(";", ",").replace(" ", ",").split(",") if p]
    try:
        return [int(float(p)) for p in parts]
    except Exception:
        return None


def legal(n: int, order: list[int] | None) -> bool:
    return order is not None and len(order) == int(n) and sorted(order) == list(range(int(n)))


def order_hash(order: list[int]) -> str:
    return hashlib.sha256(",".join(str(int(v)) for v in order).encode("utf-8")).hexdigest()[:16]


def hamming(a: list[int], b: list[int]) -> int:
    return int(sum(int(x) != int(y) for x, y in zip(a, b)))


def reference_hashes(path: Path) -> dict[int, set[str]]:
    refs = {n: set() for n in SUPPORTED_N}
    if not path.exists():
        return refs
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        n_col = "n" if "n" in row.index else ("N" if "N" in row.index else None)
        if n_col is None or pd.isna(row.get(n_col)):
            continue
        n = int(row[n_col])
        if n not in refs:
            continue
        if "order_hash" in row.index and pd.notna(row.get("order_hash")):
            refs[n].add(str(row.get("order_hash")))
            continue
        for col in ("order_json", "order_compact", "scan_order", "order"):
            if col in row.index:
                order = parse_order(row.get(col))
                if legal(n, order):
                    refs[n].add(order_hash(order))
                    break
    return refs


def synthetic_baselines(n: int) -> dict[str, list[int]]:
    raster = list(range(n))
    odd_even = list(range(1, n, 2)) + list(range(0, n, 2))
    center_order = sorted(range(n), key=lambda x: (abs(x - (n - 1) / 2.0), x))
    edge_in = []
    lo, hi = 0, n - 1
    while lo <= hi:
        edge_in.append(lo)
        if hi != lo:
            edge_in.append(hi)
        lo += 1
        hi -= 1
    jump = max(2, n // 5)
    regular_jump = []
    seen = set()
    cur = 0
    for _ in range(n * 2):
        if cur not in seen:
            regular_jump.append(cur)
            seen.add(cur)
        cur = (cur + jump) % n
        if len(regular_jump) == n:
            break
    regular_jump += [x for x in range(n) if x not in seen]
    return {
        "raster": raster,
        "odd_even": odd_even,
        "center_out": center_order,
        "edge_in": edge_in,
        "regular_jump": regular_jump,
    }


def add_baseline_proximity(pool: pd.DataFrame) -> pd.DataFrame:
    values = []
    labels = []
    for _, row in pool.iterrows():
        n = int(row["n"])
        order = parse_order(row["order_compact"])
        if not legal(n, order):
            values.append(np.nan)
            labels.append("NA")
            continue
        distances = {name: hamming(order, base) for name, base in synthetic_baselines(n).items()}
        label, value = min(distances.items(), key=lambda kv: kv[1])
        values.append(float(value))
        labels.append(label)
    pool["min_hamming_to_conventional_baseline"] = values
    pool["nearest_conventional_baseline"] = labels
    return pool


def prepare_pool() -> pd.DataFrame:
    if not POOL_PATH.exists():
        raise FileNotFoundError(f"Missing rollout pool: {POOL_PATH}")
    pool = pd.read_csv(POOL_PATH)
    for col in ("duplicate_vs_combined552", "duplicate_vs_ppo_v01", "duplicate_vs_ppo_v02K2", "duplicate_vs_ppo_v03"):
        if col not in pool.columns:
            pool[col] = False
        pool[col] = pool[col].fillna(False).astype(bool)
    pool["n"] = pool["n"].astype(int)
    pool["selected_already"] = False
    pool["predicted_quality_score"] = pd.to_numeric(pool.get("predicted_quality_score"), errors="coerce").fillna(pool.get("predicted_quality_score", 0)).fillna(0.0)
    pool["path_complexity_score"] = pd.to_numeric(pool.get("path_complexity_score"), errors="coerce").fillna(999.0)
    pool["total_travel_proxy"] = pd.to_numeric(pool.get("total_travel_proxy"), errors="coerce").fillna(9999.0)
    pool["adjacent_fraction"] = pd.to_numeric(pool.get("adjacent_fraction"), errors="coerce").fillna(0.0)
    pool["novelty_distance_score"] = pd.to_numeric(pool.get("novelty_distance_score"), errors="coerce").fillna(0.0)
    pool = add_baseline_proximity(pool)
    pool = pool.sort_values(["n", "predicted_quality_score", "novelty_distance_score"], ascending=[True, False, False])
    pool = pool.drop_duplicates(subset=["n", "order_hash"], keep="first").reset_index(drop=True)
    strict = ~(
        pool["duplicate_vs_combined552"]
        | pool["duplicate_vs_ppo_v01"]
        | pool["duplicate_vs_ppo_v02K2"]
        | pool["duplicate_vs_ppo_v03"]
    )
    pool = pool[strict].copy().reset_index(drop=True)
    return pool


def maximin_select(candidates: pd.DataFrame, selected_orders: list[list[int]], k: int) -> list[int]:
    chosen: list[int] = []
    if k <= 0 or candidates.empty:
        return chosen
    available = set(candidates.index.tolist())
    selected = list(selected_orders)
    while len(chosen) < k and available:
        best_idx = None
        best_score = -1.0
        for idx in list(available):
            order = parse_order(candidates.loc[idx, "order_compact"])
            if not order:
                continue
            if selected:
                min_dist = min(hamming(order, other) for other in selected)
            else:
                min_dist = float(candidates.loc[idx, "novelty_distance_score"])
            score = float(min_dist) + 0.01 * float(candidates.loc[idx, "predicted_quality_score"])
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            break
        chosen.append(best_idx)
        selected.append(parse_order(candidates.loc[best_idx, "order_compact"]))
        available.remove(best_idx)
    return chosen


def select_for_n(pool: pd.DataFrame, n: int, target: int) -> pd.DataFrame:
    sub = pool[pool["n"] == n].copy()
    if sub["order_hash"].nunique() < target:
        raise RuntimeError(f"Insufficient strict-novel PPO candidates for N{n}: have {sub['order_hash'].nunique()}, need {target}")
    counts = BUCKET_COUNTS[target]
    selected_indices: list[int] = []
    selected_hashes: set[str] = set()

    def add_indices(indices: list[int], bucket: str, limit: int) -> None:
        for idx in indices:
            if len([i for i in selected_indices if sub.loc[i, "selected_by_bucket"] == bucket]) >= limit:
                break
            h = str(sub.loc[idx, "order_hash"])
            if h in selected_hashes:
                continue
            sub.loc[idx, "selected_by_bucket"] = bucket
            selected_indices.append(idx)
            selected_hashes.add(h)

    sub["selected_by_bucket"] = ""
    quality_order = sub.sort_values(["predicted_quality_score", "novelty_distance_score"], ascending=[False, False]).index.tolist()
    add_indices(quality_order, "quality", counts["quality"])

    selected_orders = [parse_order(sub.loc[idx, "order_compact"]) for idx in selected_indices]
    selected_orders = [order for order in selected_orders if order is not None]
    upper_half = sub.sort_values("predicted_quality_score", ascending=False).head(max(target, len(sub) // 2))
    diversity_candidates = upper_half[~upper_half["order_hash"].isin(selected_hashes)]
    add_indices(maximin_select(diversity_candidates, selected_orders, counts["diversity"]), "diversity", counts["diversity"])

    efficiency_order = sub[~sub["order_hash"].isin(selected_hashes)].sort_values(
        ["path_complexity_score", "total_travel_proxy", "adjacent_fraction", "predicted_quality_score"],
        ascending=[True, True, False, False],
    ).index.tolist()
    add_indices(efficiency_order, "efficiency", counts["efficiency"])

    novelty_order = sub[~sub["order_hash"].isin(selected_hashes)].sort_values(
        ["novelty_distance_score", "predicted_quality_score"],
        ascending=[False, False],
    ).index.tolist()
    add_indices(novelty_order, "novelty", counts["novelty"])

    baseline_order = sub[~sub["order_hash"].isin(selected_hashes)].sort_values(
        ["min_hamming_to_conventional_baseline", "predicted_quality_score"],
        ascending=[True, False],
    ).index.tolist()
    add_indices(baseline_order, "baseline_proximity", counts["baseline_proximity"])

    if len(selected_indices) < target:
        filler = sub[~sub["order_hash"].isin(selected_hashes)].sort_values(
            ["predicted_quality_score", "novelty_distance_score"],
            ascending=[False, False],
        ).index.tolist()
        for idx in filler:
            if len(selected_indices) >= target:
                break
            sub.loc[idx, "selected_by_bucket"] = "quality"
            selected_indices.append(idx)
            selected_hashes.add(str(sub.loc[idx, "order_hash"]))

    selected = sub.loc[selected_indices].copy()
    if len(selected) != target or selected["order_hash"].nunique() != target:
        raise RuntimeError(f"Selection failed for N{n}: selected={len(selected)} unique={selected['order_hash'].nunique()} target={target}")
    return selected


def assign_batches(selected_by_n: dict[int, pd.DataFrame]) -> pd.DataFrame:
    ordered_rows = []
    batch_plan = [
        ("final_expansion_batch01", [(12, 16), (16, 16)]),
        ("final_expansion_batch02", [(12, 16), (16, 16)]),
        ("final_expansion_batch03", [(24, 32)]),
        ("final_expansion_batch04", [(24, 32)]),
        ("final_expansion_batch05", [(24, 16), (40, 16)]),
        ("final_expansion_batch06", [(40, 32)]),
        ("final_expansion_batch07", [(40, 32)]),
    ]
    cursors = {n: 0 for n in SUPPORTED_N}
    global_idx = 1
    for batch_name, allocations in batch_plan:
        for n, count in allocations:
            frame = selected_by_n[n].reset_index(drop=True)
            chunk = frame.iloc[cursors[n] : cursors[n] + count].copy()
            cursors[n] += count
            for _, row in chunk.iterrows():
                rec = row.to_dict()
                bucket = str(rec["selected_by_bucket"])
                rec["final_expansion_batch"] = batch_name
                rec["strategy_name"] = f"PPOFINAL_N{n}_B{global_idx:03d}_{bucket}"
                rec["global_candidate_index"] = global_idx
                rec["candidate_source"] = "PPO_final_expansion_checkpoint_inference"
                rec["teacher_validated"] = False
                rec["abaqus_validated"] = False
                rec["notes"] = "final expansion PPO candidate; not physically validated yet."
                ordered_rows.append(rec)
                global_idx += 1
    return pd.DataFrame(ordered_rows)


def write_scan_jsons(selected: pd.DataFrame) -> None:
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    for _, row in selected.iterrows():
        order = parse_order(row["order_compact"])
        payload = {
            "strategy_name": row["strategy_name"],
            "final_expansion_batch": row["final_expansion_batch"],
            "n": int(row["n"]),
            "order": order,
            "order_json": json.dumps(order),
            "order_compact": row["order_compact"],
            "order_hash": row["order_hash"],
            "ppo_checkpoint": row["ppo_checkpoint"],
            "ppo_version_source": row["ppo_version_source"],
            "ppo_generation_mode": row["ppo_generation_mode"],
            "selected_by_bucket": row["selected_by_bucket"],
            "candidate_source": "PPO_final_expansion_checkpoint_inference",
            "teacher_validated": False,
            "abaqus_validated": False,
            "notes": "final expansion PPO candidate; not physically validated yet.",
        }
        for key in [
            "predicted_quality_score",
            "predicted_reward_available",
            "mean_abs_jump",
            "max_abs_jump",
            "long_jump_count",
            "adjacent_fraction",
            "total_travel_proxy",
            "jump_variance",
            "local_continuity_score",
            "path_complexity_score",
            "novelty_distance_score",
            "min_hamming_to_combined552_sameN",
            "nearest_conventional_baseline",
            "min_hamming_to_conventional_baseline",
        ]:
            if key in row.index:
                value = row[key]
                if isinstance(value, (np.integer, np.floating)):
                    value = value.item()
                payload[key] = value
        (JSON_DIR / f"scan_order_{row['strategy_name']}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_batches(selected: pd.DataFrame) -> None:
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    master = SELECTED_DIR / "PPO_FINAL_EXPANSION_224_SELECTED_MASTER.csv"
    selected.to_csv(master, index=False)
    for batch_name, sub in selected.groupby("final_expansion_batch", sort=True):
        sub.to_csv(BATCH_DIR / f"PPO_FINAL_EXPANSION_{batch_name.replace('final_expansion_', '')}.csv", index=False)


def audit_outputs(selected: pd.DataFrame) -> dict:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    refs = {
        "combined552": reference_hashes(COMBINED552),
        "ppo_v01": reference_hashes(V01_METRICS),
        "ppo_v02K2": reference_hashes(V02K2_METRICS),
        "ppo_v03": reference_hashes(V03_METRICS),
    }

    legality_rows = []
    novelty_rows = []
    for _, row in selected.iterrows():
        n = int(row["n"])
        order = parse_order(row["order_compact"])
        is_legal = legal(n, order)
        legality_rows.append(
            {
                "strategy_name": row["strategy_name"],
                "n": n,
                "order_hash": row["order_hash"],
                "legal": bool(is_legal),
                "candidate_source_ok": row["candidate_source"] == "PPO_final_expansion_checkpoint_inference",
                "teacher_validated": bool(row["teacher_validated"]),
                "abaqus_validated": bool(row["abaqus_validated"]),
            }
        )
        novelty_rows.append(
            {
                "strategy_name": row["strategy_name"],
                "n": n,
                "order_hash": row["order_hash"],
                "duplicate_within_selected": bool(selected[(selected["n"] == n) & (selected["order_hash"] == row["order_hash"])].shape[0] > 1),
                "duplicate_vs_combined552": bool(row["order_hash"] in refs["combined552"].get(n, set())),
                "duplicate_vs_ppo_v01": bool(row["order_hash"] in refs["ppo_v01"].get(n, set())),
                "duplicate_vs_ppo_v02K2": bool(row["order_hash"] in refs["ppo_v02K2"].get(n, set())),
                "duplicate_vs_ppo_v03": bool(row["order_hash"] in refs["ppo_v03"].get(n, set())),
                "min_hamming_to_combined552_sameN": row.get("min_hamming_to_combined552_sameN", np.nan),
            }
        )

    legality = pd.DataFrame(legality_rows)
    novelty = pd.DataFrame(novelty_rows)
    bucket_actual = selected.groupby(["n", "selected_by_bucket"]).size().reset_index(name="actual_count")
    expected_rows = []
    for n, target in TARGET_COUNTS.items():
        for bucket, count in BUCKET_COUNTS[target].items():
            expected_rows.append({"n": n, "selected_by_bucket": bucket, "expected_count": count})
    bucket = pd.DataFrame(expected_rows).merge(bucket_actual, how="left", on=["n", "selected_by_bucket"]).fillna({"actual_count": 0})
    bucket["within_tolerance"] = (bucket["actual_count"].astype(int) - bucket["expected_count"].astype(int)).abs() <= 1

    expected_batches = {
        "final_expansion_batch01": {12: 16, 16: 16},
        "final_expansion_batch02": {12: 16, 16: 16},
        "final_expansion_batch03": {24: 32},
        "final_expansion_batch04": {24: 32},
        "final_expansion_batch05": {24: 16, 40: 16},
        "final_expansion_batch06": {40: 32},
        "final_expansion_batch07": {40: 32},
    }
    batch_rows = []
    for batch_name, expected in expected_batches.items():
        sub = selected[selected["final_expansion_batch"] == batch_name]
        observed = {int(k): int(v) for k, v in sub["n"].value_counts().sort_index().items()}
        batch_rows.append(
            {
                "final_expansion_batch": batch_name,
                "expected_counts_by_N": json.dumps(expected, sort_keys=True),
                "observed_counts_by_N": json.dumps(observed, sort_keys=True),
                "total_count": int(len(sub)),
                "pass": bool(observed == expected and len(sub) == 32),
            }
        )
    batch_audit = pd.DataFrame(batch_rows)

    legality_path = AUDIT_DIR / "final_expansion_legality_audit.csv"
    novelty_path = AUDIT_DIR / "final_expansion_novelty_audit.csv"
    bucket_path = AUDIT_DIR / "final_expansion_bucket_balance_audit.csv"
    batch_path = AUDIT_DIR / "final_expansion_batch_count_audit.csv"
    legality.to_csv(legality_path, index=False)
    novelty.to_csv(novelty_path, index=False)
    bucket.to_csv(bucket_path, index=False)
    batch_audit.to_csv(batch_path, index=False)

    return {
        "legality_path": str(legality_path),
        "novelty_path": str(novelty_path),
        "bucket_path": str(bucket_path),
        "batch_path": str(batch_path),
        "all_legal": bool(legality["legal"].all()),
        "all_candidate_source_ok": bool(legality["candidate_source_ok"].all()),
        "all_unvalidated": bool((~legality["teacher_validated"]).all() and (~legality["abaqus_validated"]).all()),
        "duplicate_within_selected_count": int(novelty["duplicate_within_selected"].sum()),
        "duplicate_vs_any_reference_count": int(
            (
                novelty["duplicate_vs_combined552"]
                | novelty["duplicate_vs_ppo_v01"]
                | novelty["duplicate_vs_ppo_v02K2"]
                | novelty["duplicate_vs_ppo_v03"]
            ).sum()
        ),
        "bucket_balance_ok": bool(bucket["within_tolerance"].all()),
        "batch_counts_ok": bool(batch_audit["pass"].all()),
    }


def write_handoff(selected: pd.DataFrame) -> Path:
    HANDOFF_DIR.mkdir(parents=True, exist_ok=True)
    base_family = {12: "12track_full/sanity_base", 16: "16track_full/sanity_base", 24: "24track_full/sanity_base", 40: "40track_full/sanity_base"}
    rows = []
    for _, row in selected.iterrows():
        n = int(row["n"])
        job = f"J2D_{row['strategy_name']}"
        case_placeholder = f"<future_case_root>/{row['final_expansion_batch']}/N{n}_{row['strategy_name']}"
        rows.append(
            {
                "batch": row["final_expansion_batch"],
                "strategy_name": row["strategy_name"],
                "N": n,
                "expected_base_CAE_family": base_family[n],
                "intended_case_root_placeholder": case_placeholder,
                "expected_job_name": job,
                "expected_INP_path_placeholder": f"{case_placeholder}/{job}.inp",
                "order_hash": row["order_hash"],
                "selected_by_bucket": row["selected_by_bucket"],
                "candidate_source": row["candidate_source"],
                "teacher_validated": False,
            }
        )
    path = HANDOFF_DIR / "PPO_FINAL_EXPANSION_224_CAE_INP_HANDOFF_PREVIEW.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_report(selected: pd.DataFrame, audit: dict, pool_summary: dict, handoff_path: Path, verdict: str) -> tuple[Path, Path]:
    DOCS_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = DOCS_ROOT / "PPO_FINAL_EXPANSION_224_CANDIDATE_GENERATION_REPORT.md"
    claim_path = DOCS_ROOT / "PPO_FINAL_EXPANSION_224_CLAIM_BOUNDARY.md"
    counts_by_n = {int(k): int(v) for k, v in selected["n"].value_counts().sort_index().items()}
    buckets = selected["selected_by_bucket"].value_counts().to_dict()
    efficiency_summary = selected.groupby("n")[["mean_abs_jump", "max_abs_jump", "total_travel_proxy", "adjacent_fraction", "path_complexity_score"]].agg(["mean", "min", "max"]).round(4)
    report_path.write_text(
        f"""# PPO Final Expansion 224 Candidate Generation Report

## Purpose

Stage T creates a fixed-budget 224-case PPO-generated expansion set so that future teacher validation can bring the cumulative PPO pool from 96 cases to 320 cases.

## Current and Target Pool

- Current PPO teacher-validated pool: 96
- Target PPO teacher-validated pool: 320
- Remaining candidate target: 224

## Why Fixed-Budget Expansion

Stage S showed that v03 did not improve U2/lex physical performance and had weak surrogate-to-teacher alignment. This stage therefore stops open-ended reward redesign and uses existing PPO checkpoint inference to create a broad, auditable evidence pool.

## Rollout Pool Summary

- Rollout pool path: `{POOL_PATH}`
- Total pool rows: {pool_summary.get('total_rows')}
- Total unique order hashes: {pool_summary.get('total_unique_order_hashes')}

## Selected Candidate Summary

- Selected total: {len(selected)}
- Counts by N: {counts_by_n}
- Batch count: {selected['final_expansion_batch'].nunique()}

## Bucket Distribution

{json.dumps({str(k): int(v) for k, v in buckets.items()}, indent=2)}

## Legality Audit

- All legal: {audit['all_legal']}
- Candidate source OK: {audit['all_candidate_source_ok']}
- All unvalidated: {audit['all_unvalidated']}

## Novelty Audit

- Duplicate within selected: {audit['duplicate_within_selected_count']}
- Duplicate vs combined552/v01/v02K2/v03 references: {audit['duplicate_vs_any_reference_count']}

## Industrial-Efficiency Proxy Summary

These are sequence descriptors only, not teacher metrics or physically validated efficiency measures.

```text
{efficiency_summary.to_string()}
```

## Handoff Preview

Handoff preview: `{handoff_path}`

## Claim Boundary

Stage T supports only candidate-generation and handoff-readiness claims. It does not support physical improvement, teacher validation, or industrial efficiency claims.

## Verdict

{verdict}
""",
        encoding="utf-8",
    )
    claim_path.write_text(
        """# PPO Final Expansion 224 Claim Boundary

## Safe After Stage T

- A final expansion set of 224 legal PPO-generated scan-order candidates was created.
- The selected candidates are ready for later CAE/INP generation.
- The final expansion is designed to bring the cumulative PPO teacher-validation pool from 96 to 320 after future validation.
- Industrial-efficiency descriptors are proxy descriptors only.

## Unsafe After Stage T

- The final expansion improves physical metrics.
- The final expansion beats combined552.
- The final expansion is teacher validated.
- Industrial efficiency is physically validated.
- PPO solved scan-order optimisation.
""",
        encoding="utf-8",
    )
    return report_path, claim_path


def write_manifest(selected: pd.DataFrame, audit: dict, handoff_path: Path, report_path: Path, claim_path: Path, verdict: str) -> Path:
    manifest_path = OUT_ROOT / "stageT_final_expansion_224_manifest.json"
    batch_paths = sorted(str(path) for path in BATCH_DIR.glob("PPO_FINAL_EXPANSION_batch*.csv"))
    manifest = {
        "branch": git_branch(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "target_total": 320,
        "current_validated_total": 96,
        "selected_expansion_total": int(len(selected)),
        "selected_master_csv": str(SELECTED_DIR / "PPO_FINAL_EXPANSION_224_SELECTED_MASTER.csv"),
        "per_batch_csv_paths": batch_paths,
        "scan_order_JSON_dir": str(JSON_DIR),
        "audit_paths": {
            "legality": audit["legality_path"],
            "novelty": audit["novelty_path"],
            "bucket_balance": audit["bucket_path"],
            "batch_count": audit["batch_path"],
        },
        "handoff_preview": str(handoff_path),
        "report_path": str(report_path),
        "claim_boundary_path": str(claim_path),
        "selected_counts_by_N": {str(int(k)): int(v) for k, v in selected["n"].value_counts().sort_index().items()},
        "no_Abaqus": True,
        "no_ODB_opening": True,
        "no_ODB_extraction": True,
        "no_solver": True,
        "no_datacheck": True,
        "no_enqueue": True,
        "no_CAE_INP_JNL": True,
        "no_teacher_validation": True,
        "no_training": True,
        "no_commit_or_push": True,
        "final_verdict": verdict,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    pool_summary_path = OUT_ROOT / "rollout_pools" / "ppo_final_expansion_rollout_pool_summary.json"
    pool_summary = json.loads(pool_summary_path.read_text(encoding="utf-8")) if pool_summary_path.exists() else {}
    pool = prepare_pool()

    selected_by_n = {}
    for n, target in TARGET_COUNTS.items():
        selected_by_n[n] = select_for_n(pool, n, target)
    selected = assign_batches(selected_by_n)

    # Stable column order for manuscript/handoff readability.
    preferred = [
        "strategy_name",
        "final_expansion_batch",
        "global_candidate_index",
        "n",
        "order_json",
        "order_compact",
        "order_hash",
        "ppo_checkpoint",
        "ppo_version_source",
        "ppo_generation_mode",
        "seed",
        "selected_by_bucket",
        "predicted_reward_available",
        "predicted_quality_score",
        "mean_abs_jump",
        "max_abs_jump",
        "long_jump_count",
        "adjacent_fraction",
        "total_travel_proxy",
        "jump_variance",
        "local_continuity_score",
        "path_complexity_score",
        "novelty_distance_score",
        "min_hamming_to_combined552_sameN",
        "min_hamming_to_conventional_baseline",
        "nearest_conventional_baseline",
        "candidate_source",
        "teacher_validated",
        "abaqus_validated",
        "notes",
    ]
    selected = selected[[col for col in preferred if col in selected.columns] + [col for col in selected.columns if col not in preferred]]

    write_batches(selected)
    write_scan_jsons(selected)
    audit = audit_outputs(selected)
    handoff_path = write_handoff(selected)

    pass_conditions = [
        len(selected) == 224,
        selected["n"].value_counts().sort_index().to_dict() == TARGET_COUNTS,
        audit["all_legal"],
        audit["all_candidate_source_ok"],
        audit["all_unvalidated"],
        audit["duplicate_within_selected_count"] == 0,
        audit["duplicate_vs_any_reference_count"] == 0,
        audit["bucket_balance_ok"],
        audit["batch_counts_ok"],
    ]
    verdict = (
        "PASS_PPO_FINAL_EXPANSION_224_READY_FOR_BATCHED_CAE_INP_HANDOFF"
        if all(pass_conditions)
        else "WARNING_PPO_FINAL_EXPANSION_224_REVIEW"
    )
    report_path, claim_path = write_report(selected, audit, pool_summary, handoff_path, verdict)
    manifest_path = write_manifest(selected, audit, handoff_path, report_path, claim_path, verdict)

    summary = {
        "verdict": verdict,
        "selected_total": int(len(selected)),
        "selected_counts_by_N": {str(int(k)): int(v) for k, v in selected["n"].value_counts().sort_index().items()},
        "batch_counts": {str(k): int(v) for k, v in selected["final_expansion_batch"].value_counts().sort_index().items()},
        "audit": audit,
        "selected_master_csv": str(SELECTED_DIR / "PPO_FINAL_EXPANSION_224_SELECTED_MASTER.csv"),
        "batch_dir": str(BATCH_DIR),
        "scan_order_json_dir": str(JSON_DIR),
        "handoff_preview": str(handoff_path),
        "report_path": str(report_path),
        "claim_boundary_path": str(claim_path),
        "manifest_path": str(manifest_path),
        "no_Abaqus": True,
        "no_ODB_opening": True,
        "no_solver": True,
        "no_CAE_INP_JNL": True,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
