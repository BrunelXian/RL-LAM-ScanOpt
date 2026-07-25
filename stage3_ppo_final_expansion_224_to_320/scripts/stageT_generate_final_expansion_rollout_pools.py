"""Generate Stage T final-expansion PPO rollout pools.

The pool is built from existing PPO checkpoint-inference rollout pools and, if
needed, fresh inference-only rollouts from the frozen v01 checkpoint for N12/N16.
No PPO training, Abaqus, ODB opening, solver, datacheck, enqueue, or CAE/INP/JNL
generation is performed.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
NS = "stage3_ppo_final_expansion_224_to_320"
OUT_ROOT = ROOT / "outputs" / NS
POOL_DIR = OUT_ROOT / "rollout_pools"

V01_SRC = ROOT / "stage3_ppo_rl_lam_fea_addendum_v01" / "src"
if str(V01_SRC) not in sys.path:
    sys.path.insert(0, str(V01_SRC))


INPUTS = {
    "combined552": ROOT / "outputs" / "stage3_run_78_final_evidence_freeze_package" / "FROZEN_stage3_native_combined552_teacher_dataset.csv",
    "v01_metrics": ROOT / "outputs" / "stage3_ppo_rl_lam_fea_addendum_v01" / "stageI_final_ppo_evidence_freeze" / "frozen_tables" / "FROZEN_PPO_batch32_teacher_metrics.csv",
    "v02K2_metrics": ROOT / "outputs" / "stage3_ppo_rl_lam_fea_addendum_v02_targeted_N24_N40" / "stageM_ODB_teacher_metric_extraction" / "stageM_v02K2_teacher_metrics.csv",
    "v03_metrics": ROOT / "outputs" / "stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40" / "stageR_ODB_teacher_metric_extraction" / "stageR_v03_teacher_metrics.csv",
    "v01_checkpoint": ROOT / "outputs" / "stage3_ppo_rl_lam_fea_addendum_v01" / "ppo_training" / "checkpoints" / "maskable_ppo_lam_scan_order_final.zip",
    "v01_surrogate": ROOT / "outputs" / "stage3_ppo_rl_lam_fea_addendum_v01" / "surrogate_reward_model" / "models" / "ppo_surrogate_reward_model_best.joblib",
}

POOL_SOURCES = [
    {
        "stageT_pool_source": "existing_v01_rollout_pool",
        "ppo_version_source": "v01",
        "path": ROOT / "outputs" / "stage3_ppo_rl_lam_fea_addendum_v01" / "ppo_candidate_generation" / "rollout_pool" / "ppo_generated_rollout_pool.csv",
        "default_checkpoint": INPUTS["v01_checkpoint"],
    },
    {
        "stageT_pool_source": "existing_v02_rollout_pool",
        "ppo_version_source": "v02",
        "path": ROOT / "outputs" / "stage3_ppo_rl_lam_fea_addendum_v02_targeted_N24_N40" / "candidate_generation_v02" / "rollout_pool" / "v02_ppo_rollout_pool.csv",
        "default_checkpoint": ROOT / "outputs" / "stage3_ppo_rl_lam_fea_addendum_v02_targeted_N24_N40" / "ppo_training_v02" / "checkpoints",
    },
    {
        "stageT_pool_source": "existing_v02K2_rollout_pool",
        "ppo_version_source": "v02K2",
        "path": ROOT / "outputs" / "stage3_ppo_rl_lam_fea_addendum_v02_targeted_N24_N40" / "stageK2_n40_completion" / "rollout_pool" / "stageK2_n40_ppo_rollout_pool.csv",
        "default_checkpoint": ROOT / "outputs" / "stage3_ppo_rl_lam_fea_addendum_v02_targeted_N24_N40" / "stageK2_n40_completion" / "checkpoints" / "N40_seed20260624_maskable_ppo_v02_K2.zip",
    },
    {
        "stageT_pool_source": "existing_v03_rollout_pool",
        "ppo_version_source": "v03",
        "path": ROOT / "outputs" / "stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40" / "candidate_generation_v03" / "rollout_pool" / "v03_ppo_rollout_pool.csv",
        "default_checkpoint": "",
    },
]

TARGET_COUNTS = {12: 32, 16: 32, 24: 80, 40: 80}
MIN_POOL_UNIQUE = {12: 80, 16: 120, 24: 320, 40: 320}
SUPPORTED_N = (12, 16, 24, 40)


def parse_order_value(value: object) -> list[int] | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, list):
        return [int(v) for v in value]
    text = str(value).strip()
    if not text:
        return None
    if text.startswith("["):
        try:
            loaded = json.loads(text)
            return [int(v) for v in loaded]
        except Exception:
            pass
    cleaned = text.replace(";", ",").replace("|", ",").replace(" ", ",")
    parts = [p for p in cleaned.split(",") if p != ""]
    try:
        return [int(float(p)) for p in parts]
    except Exception:
        return None


def parse_order_from_row(row: pd.Series) -> list[int] | None:
    for col in ("order_json", "order", "scan_order", "order_compact", "terminal_order"):
        if col in row.index:
            order = parse_order_value(row.get(col))
            if order is not None:
                return order
    return None


def order_hash(order: Iterable[int]) -> str:
    compact = ",".join(str(int(v)) for v in order)
    return hashlib.sha256(compact.encode("utf-8")).hexdigest()[:16]


def legal_order(n: int, order: list[int] | None) -> bool:
    return order is not None and len(order) == int(n) and sorted(int(v) for v in order) == list(range(int(n)))


def descriptors(n: int, order: list[int]) -> dict[str, float]:
    arr = np.asarray(order, dtype=float)
    jumps = np.abs(np.diff(arr)) if len(arr) > 1 else np.asarray([], dtype=float)
    signed = np.diff(arr) if len(arr) > 1 else np.asarray([], dtype=float)
    center = (n - 1) / 2.0
    early = arr[: max(1, n // 4)]
    edge_dist = np.minimum(early, n - 1 - early) if len(early) else np.asarray([0.0])
    center_dist = np.abs(early - center) if len(early) else np.asarray([0.0])
    long_threshold = max(2.0, n / 3.0)
    total_travel = float(jumps.sum()) if jumps.size else 0.0
    jump_var = float(np.var(jumps)) if jumps.size else 0.0
    adjacent = float(np.mean(jumps == 1)) if jumps.size else 0.0
    mean_jump = float(np.mean(jumps)) if jumps.size else 0.0
    path_complexity = (total_travel / max(1.0, n * (n - 1))) + (jump_var / max(1.0, (n - 1) ** 2))
    return {
        "mean_abs_jump": mean_jump,
        "max_abs_jump": float(np.max(jumps)) if jumps.size else 0.0,
        "long_jump_count": int(np.sum(jumps >= long_threshold)) if jumps.size else 0,
        "adjacent_fraction": adjacent,
        "total_travel_proxy": total_travel,
        "jump_variance": jump_var,
        "local_continuity_score": float(adjacent - mean_jump / max(1.0, n - 1)),
        "path_complexity_score": float(path_complexity),
        "parity_switch_fraction": float(np.mean(np.abs(np.diff(arr % 2)) > 0)) if len(arr) > 1 else 0.0,
        "early_center_bias": float(1.0 - np.mean(center_dist) / max(1.0, center)) if len(early) else 0.0,
        "early_edge_bias": float(1.0 - np.mean(edge_dist) / max(1.0, center)) if len(early) else 0.0,
    }


def first_present(row: pd.Series, cols: list[str], default: object = np.nan) -> object:
    for col in cols:
        if col in row.index and pd.notna(row.get(col)):
            return row.get(col)
    return default


def normalize_pool(source: dict) -> pd.DataFrame:
    path = Path(source["path"])
    if not path.exists():
        return pd.DataFrame()
    raw = pd.read_csv(path)
    rows = []
    for idx, row in raw.iterrows():
        n = int(first_present(row, ["n", "N"], -1))
        if n not in SUPPORTED_N:
            continue
        order = parse_order_from_row(row)
        if not legal_order(n, order):
            continue
        h = str(first_present(row, ["order_hash"], order_hash(order)))
        quality = first_present(
            row,
            [
                "final_v03_score",
                "terminal_reward",
                "conservative_reward",
                "predicted_surrogate_reward_lex",
                "predicted_reward",
                "mean_pred_reward",
            ],
            np.nan,
        )
        try:
            quality = float(quality)
        except Exception:
            quality = np.nan
        chk = first_present(row, ["ppo_checkpoint", "ppo_v02_checkpoint", "ppo_v03_checkpoint"], "")
        if pd.isna(chk) or str(chk).strip() == "":
            chk = str(source.get("default_checkpoint", ""))
        rec = {
            "stageT_pool_source": source["stageT_pool_source"],
            "ppo_version_source": source["ppo_version_source"],
            "source_row_index": int(idx),
            "n": n,
            "order_json": json.dumps(order),
            "order_compact": ",".join(str(v) for v in order),
            "order_hash": h,
            "ppo_checkpoint": str(chk),
            "ppo_generation_mode": str(first_present(row, ["ppo_generation_mode", "generation_mode"], "stochastic")),
            "seed": first_present(row, ["rollout_seed", "seed", "ppo_seed"], np.nan),
            "predicted_quality_score": quality,
            "predicted_reward_available": bool(not pd.isna(quality)),
            "candidate_source": "PPO_final_expansion_checkpoint_inference_pool",
            "teacher_validated": False,
            "abaqus_validated": False,
        }
        for key, value in descriptors(n, order).items():
            if key in row.index and pd.notna(row.get(key)):
                rec[key] = row.get(key)
            elif key == "adjacent_fraction" and "adjacent_jump_fraction" in row.index and pd.notna(row.get("adjacent_jump_fraction")):
                rec[key] = row.get("adjacent_jump_fraction")
            else:
                rec[key] = value
        rows.append(rec)
    return pd.DataFrame(rows)


def reference_hashes_from(path: Path, n_filter: int | None = None) -> dict[int, set[str]]:
    out = {n: set() for n in SUPPORTED_N}
    if not path.exists():
        return out
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        n_val = first_present(row, ["n", "N"], np.nan)
        if pd.isna(n_val):
            continue
        n = int(n_val)
        if n_filter is not None and n != n_filter:
            continue
        if n not in out:
            continue
        if "order_hash" in row.index and pd.notna(row.get("order_hash")):
            out[n].add(str(row.get("order_hash")))
            continue
        order = parse_order_from_row(row)
        if legal_order(n, order):
            out[n].add(order_hash(order))
    return out


def hamming(a: list[int], b: list[int]) -> int:
    return int(sum(int(x) != int(y) for x, y in zip(a, b)))


def reference_orders(path: Path) -> dict[int, list[list[int]]]:
    out = {n: [] for n in SUPPORTED_N}
    if not path.exists():
        return out
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        n_val = first_present(row, ["n", "N"], np.nan)
        if pd.isna(n_val):
            continue
        n = int(n_val)
        if n not in out:
            continue
        order = parse_order_from_row(row)
        if legal_order(n, order):
            out[n].append(order)
    return out


def add_novelty_columns(pool: pd.DataFrame) -> pd.DataFrame:
    ref_sources = {
        "combined552": INPUTS["combined552"],
        "ppo_v01": INPUTS["v01_metrics"],
        "ppo_v02K2": INPUTS["v02K2_metrics"],
        "ppo_v03": INPUTS["v03_metrics"],
    }
    ref_hashes = {name: reference_hashes_from(path) for name, path in ref_sources.items()}
    combined_orders = reference_orders(INPUTS["combined552"])
    min_hamming = []
    for idx, row in pool.iterrows():
        n = int(row["n"])
        order = parse_order_value(row["order_compact"])
        for name in ref_sources:
            pool.loc[idx, f"duplicate_vs_{name}"] = bool(row["order_hash"] in ref_hashes[name].get(n, set()))
        if order and combined_orders.get(n):
            min_hamming.append(min(hamming(order, ref_order) for ref_order in combined_orders[n]))
        else:
            min_hamming.append(np.nan)
    pool["min_hamming_to_combined552_sameN"] = min_hamming
    pool["novelty_distance_score"] = pool["min_hamming_to_combined552_sameN"].fillna(0).astype(float)
    return pool


def run_extra_v01_rollouts(existing: pd.DataFrame) -> pd.DataFrame:
    """Fresh inference-only v01 rollouts for low-N diversity if needed."""
    need_ns = []
    for n, minimum in MIN_POOL_UNIQUE.items():
        if n not in (12, 16):
            continue
        have = existing[existing["n"] == n]["order_hash"].nunique() if not existing.empty else 0
        if have < minimum:
            need_ns.append(n)
    if not need_ns:
        return pd.DataFrame()

    try:
        from sb3_contrib import MaskablePPO
        from ppo_surrogate_env_wrapper import build_surrogate_env_from_paths
    except Exception as exc:
        print(f"WARNING: unable to import MaskablePPO/v01 env for extra low-N rollouts: {exc}")
        return pd.DataFrame()

    rows = []
    model = MaskablePPO.load(str(INPUTS["v01_checkpoint"]))
    for n in need_ns:
        existing_hashes = set(existing[existing["n"] == n]["order_hash"]) if not existing.empty else set()
        target_unique = MIN_POOL_UNIQUE[n]
        max_attempts = 12000 if n == 12 else 8000
        deterministic_done = False
        attempts = 0
        env = build_surrogate_env_from_paths(
            INPUTS["v01_surrogate"],
            fixed_n=n,
            random_n=False,
            supported_n=(12, 16, 24, 40),
            seed=20260629 + n,
        )
        while len(existing_hashes) < target_unique and attempts < max_attempts:
            attempts += 1
            mode = "deterministic" if not deterministic_done else "stochastic"
            deterministic_done = True
            obs, _ = env.reset(seed=202606290 + n * 100000 + attempts, options={"n": n})
            terminated = False
            truncated = False
            info = {}
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=(mode == "deterministic"), action_masks=env.action_masks())
                obs, reward, terminated, truncated, info = env.step(int(action))
            order = info.get("terminal_order") or env.terminal_order()
            if not legal_order(n, order):
                continue
            h = order_hash(order)
            if h in existing_hashes:
                continue
            existing_hashes.add(h)
            rec = {
                "stageT_pool_source": "fresh_v01_checkpoint_inference_lowN_topup",
                "ppo_version_source": "v01",
                "source_row_index": attempts,
                "n": n,
                "order_json": json.dumps(order),
                "order_compact": ",".join(str(v) for v in order),
                "order_hash": h,
                "ppo_checkpoint": str(INPUTS["v01_checkpoint"]),
                "ppo_generation_mode": mode,
                "seed": 202606290 + n * 100000 + attempts,
                "predicted_quality_score": float(reward),
                "predicted_reward_available": True,
                "candidate_source": "PPO_final_expansion_checkpoint_inference_pool",
                "teacher_validated": False,
                "abaqus_validated": False,
            }
            rec.update(descriptors(n, order))
            rows.append(rec)
        print(f"N{n} extra v01 inference attempts={attempts} unique_after={len(existing_hashes)}")
    return pd.DataFrame(rows)


def main() -> None:
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    pool_parts = [normalize_pool(source) for source in POOL_SOURCES]
    pool = pd.concat([part for part in pool_parts if not part.empty], ignore_index=True) if any(not p.empty for p in pool_parts) else pd.DataFrame()

    extra = run_extra_v01_rollouts(pool)
    if not extra.empty:
        pool = pd.concat([pool, extra], ignore_index=True)

    if pool.empty:
        raise RuntimeError("No PPO rollout pool rows available.")

    pool = pool.drop_duplicates(subset=["n", "order_hash", "ppo_version_source", "stageT_pool_source"], keep="first").reset_index(drop=True)
    pool = add_novelty_columns(pool)

    # Keep a provenance-rich pool. Selection script will enforce final duplicate policy.
    out_path = POOL_DIR / "ppo_final_expansion_rollout_pool.csv"
    pool.to_csv(out_path, index=False)

    summary_by_n = {}
    for n in SUPPORTED_N:
        sub = pool[pool["n"] == n]
        summary_by_n[str(n)] = {
            "rows": int(len(sub)),
            "unique_order_hashes": int(sub["order_hash"].nunique()),
            "sources": {str(k): int(v) for k, v in sub["stageT_pool_source"].value_counts().items()},
            "unique_after_default_duplicate_reject": int(
                sub[
                    ~(sub["duplicate_vs_combined552"].astype(bool)
                      | sub["duplicate_vs_ppo_v01"].astype(bool)
                      | sub["duplicate_vs_ppo_v02K2"].astype(bool)
                      | sub["duplicate_vs_ppo_v03"].astype(bool))
                ]["order_hash"].nunique()
            ),
        }

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "rollout_pool_path": str(out_path),
        "total_rows": int(len(pool)),
        "total_unique_order_hashes": int(pool["order_hash"].nunique()),
        "summary_by_N": summary_by_n,
        "target_counts": {str(k): int(v) for k, v in TARGET_COUNTS.items()},
        "no_Abaqus": True,
        "no_ODB_opening": True,
        "no_ODB_extraction": True,
        "no_solver": True,
        "no_datacheck": True,
        "no_enqueue": True,
        "no_CAE_INP_JNL": True,
        "no_training": True,
    }
    summary_path = POOL_DIR / "ppo_final_expansion_rollout_pool_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
