"""Generate refreshed Stage K2 N40 candidates from K2 PPO checkpoints."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO

PROJECT_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
NS = "stage3_ppo_rl_lam_fea_addendum_v02_targeted_N24_N40"
SRC_DIR = PROJECT_ROOT / NS / "src"
sys.path.insert(0, str(SRC_DIR))

from v02_fixedN_ppo_env import FixedNLamScanOrderPPOEnv, V02EnvConfig, validate_order  # noqa: E402

OUT_ROOT = PROJECT_ROOT / "outputs" / NS
K2_ROOT = OUT_ROOT / "stageK2_n40_completion"
CHECKPOINT_DIR = K2_ROOT / "checkpoints"
POOL_DIR = K2_ROOT / "rollout_pool"
SEL_DIR = K2_ROOT / "selected_batch32_K2"
SCAN_DIR = SEL_DIR / "scan_orders"
TABLES_DIR = K2_ROOT / "tables"
SURR_MODEL = OUT_ROOT / "surrogate_v02" / "models" / "N40_surrogate_reward_v02.joblib"
DATASET = OUT_ROOT / "data" / "v02_targeted_N24_N40_teacher_dataset.csv"
STAGEK_SELECTED = OUT_ROOT / "candidate_generation_v02" / "selected_batch32" / "v02_ppo_targeted_N24_N40_candidate_batch32.csv"

POOL_CSV = POOL_DIR / "stageK2_n40_ppo_rollout_pool.csv"
REFRESHED_CSV = SEL_DIR / "stageK2_refreshed_N40_candidates.csv"
REFRESHED_SUMMARY = TABLES_DIR / "stageK2_refreshed_N40_generation_summary.json"
ROLLOUT_ATTEMPTS_TOTAL = 3000


def git_branch() -> str:
    try:
        result = subprocess.run(["git", "-C", str(PROJECT_ROOT), "branch", "--show-current"], capture_output=True, text=True, timeout=10)
        return result.stdout.strip()
    except Exception:
        return ""


def ensure_dirs() -> None:
    for p in [POOL_DIR, SEL_DIR, SCAN_DIR, TABLES_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def order_compact(order: list[int]) -> str:
    return ",".join(str(x) for x in order)


def order_json(order: list[int]) -> str:
    return json.dumps(order, separators=(",", ":"))


def order_hash(order: list[int]) -> str:
    return hashlib.sha256(order_compact(order).encode("utf-8")).hexdigest()[:16]


def parse_order(text: Any) -> list[int]:
    return [int(x) for x in re.findall(r"-?\d+", "" if pd.isna(text) else str(text))]


def hamming(a: list[int], b: list[int]) -> int:
    return sum(int(x != y) for x, y in zip(a, b)) if len(a) == len(b) else max(len(a), len(b))


def descriptors(order: list[int]) -> dict[str, float]:
    arr = np.asarray(order, dtype=float)
    jumps = np.abs(np.diff(arr))
    parity = arr.astype(int) % 2
    early = arr[:10]
    center = 19.5
    return {
        "mean_abs_jump": float(jumps.mean()),
        "max_abs_jump": float(jumps.max()),
        "adjacent_fraction": float(np.mean(jumps == 1)),
        "parity_switch_fraction": float(np.mean(parity[1:] != parity[:-1])),
        "early_center_bias": float(np.mean(np.abs(early - center) / center)),
        "early_edge_bias": float(np.mean(np.minimum(early, 39 - early) / center)),
        "early_spread_proxy": float(np.ptp(early) / 39.0),
    }


def make_env(seed: int) -> FixedNLamScanOrderPPOEnv:
    return FixedNLamScanOrderPPOEnv(V02EnvConfig(n=40, surrogate_model_path=str(SURR_MODEL), seed=seed, conservative_reward=True))


def reference_orders() -> dict[str, list[tuple[str, list[int]]]]:
    refs = {"training_reference": [], "stageK_N40_selected": []}
    df = pd.read_csv(DATASET)
    for _, row in df[df["n"].astype(int) == 40].iterrows():
        refs["training_reference"].append((str(row["strategy_name"]), parse_order(row["order_json"])))
    stagek = pd.read_csv(STAGEK_SELECTED)
    for _, row in stagek[stagek["n"].astype(int) == 40].iterrows():
        refs["stageK_N40_selected"].append((str(row["strategy_name"]), parse_order(row["order_json"])))
    return refs


def rollout(model: MaskablePPO, env: FixedNLamScanOrderPPOEnv, seed: int, deterministic: bool) -> tuple[list[int], dict[str, float]]:
    obs, _ = env.reset(seed=seed)
    done = False
    info = {}
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic, action_masks=env.action_masks())
        obs, _, done, _, info = env.step(int(action))
    order = [int(x) for x in info["terminal_order"]]
    return order, env.reward_model.predict_components(40, order)


def generate_pool() -> pd.DataFrame:
    refs = reference_orders()
    ckpts = sorted(CHECKPOINT_DIR.glob("N40_seed*_maskable_ppo_v02_K2.zip"))
    if not ckpts:
        raise FileNotFoundError(f"No K2 N40 checkpoints in {CHECKPOINT_DIR}")
    per_ckpt = max(1, ROLLOUT_ATTEMPTS_TOTAL // len(ckpts))
    rows = []
    for ckpt in ckpts:
        seed_match = re.search(r"seed(\d+)", ckpt.name)
        ppo_seed = int(seed_match.group(1)) if seed_match else 20260624
        model = MaskablePPO.load(ckpt)
        env = make_env(ppo_seed)
        attempts = [("deterministic", ppo_seed, True)] + [("stochastic", 20260628 + i + ppo_seed, False) for i in range(per_ckpt)]
        for mode, seed, deterministic in attempts:
            order, comps = rollout(model, env, seed, deterministic)
            if not validate_order(40, order):
                continue
            train_dist = [hamming(order, r) for _, r in refs["training_reference"]]
            stagek_dist = [hamming(order, r) for _, r in refs["stageK_N40_selected"]]
            rows.append({
                "n": 40,
                "order_json": order_json(order),
                "order_compact": order_compact(order),
                "order_hash": order_hash(order),
                "ppo_checkpoint": str(ckpt),
                "ppo_seed": ppo_seed,
                "rollout_seed": seed,
                "ppo_generation_mode": mode,
                "predicted_reward": comps["best_model_reward"],
                "mean_pred_reward": comps["mean_pred_reward"],
                "std_pred_reward": comps["std_pred_reward"],
                "conservative_reward": comps["conservative_reward"],
                "duplicate_vs_combined552_plus_v01_training": any(d == 0 for d in train_dist),
                "duplicate_vs_stageK_N40_selected": any(d == 0 for d in stagek_dist),
                "minimum_hamming_to_training_reference": int(min(train_dist)),
                "minimum_hamming_to_stageK_N40": int(min(stagek_dist)),
                "candidate_source": "PPO_v02K2_checkpoint_inference",
                "teacher_validated": False,
                "abaqus_validated": False,
                **descriptors(order),
            })
    pool = pd.DataFrame(rows)
    pool.to_csv(POOL_CSV, index=False)
    return pool


def maximin(candidates: pd.DataFrame, selected_orders: list[list[int]], count: int) -> pd.DataFrame:
    chosen = []
    remaining = candidates.copy()
    while len(chosen) < count and not remaining.empty:
        best_idx, best_score = None, -1
        for idx, row in remaining.iterrows():
            order = parse_order(row["order_json"])
            score = min(hamming(order, existing) for existing in selected_orders) if selected_orders else 40
            if score > best_score:
                best_idx, best_score = idx, score
        chosen.append(best_idx)
        selected_orders.append(parse_order(remaining.loc[best_idx, "order_json"]))
        remaining = remaining.drop(index=best_idx)
    return candidates.loc[chosen] if chosen else candidates.iloc[0:0]


def select(pool: pd.DataFrame) -> pd.DataFrame:
    unique = pool.sort_values(["conservative_reward", "minimum_hamming_to_training_reference"], ascending=[False, False]).drop_duplicates("order_hash").reset_index(drop=True)
    if len(unique) < 16:
        raise RuntimeError(f"Insufficient unique N40 K2 orders: {len(unique)}")
    rows = []
    selected_orders: list[list[int]] = []
    det = unique[unique["ppo_generation_mode"] == "deterministic"].head(1)
    if len(det):
        row = det.iloc[0].to_dict()
        row["selected_by"] = "deterministic"
        rows.append(row)
        selected_orders.append(parse_order(row["order_json"]))
    for _, r in unique[~unique["order_hash"].isin([x["order_hash"] for x in rows])].head(7).iterrows():
        row = r.to_dict()
        row["selected_by"] = "top_reward"
        rows.append(row)
        selected_orders.append(parse_order(row["order_json"]))
    upper = unique.head(max(64, len(unique) // 2))
    upper = upper[~upper["order_hash"].isin([x["order_hash"] for x in rows])]
    diverse = maximin(upper, selected_orders, 5)
    for _, r in diverse.iterrows():
        row = r.to_dict()
        row["selected_by"] = "diverse_tophalf"
        rows.append(row)
    chosen = {x["order_hash"] for x in rows}
    novelty = upper[~upper["order_hash"].isin(chosen)].sort_values(["minimum_hamming_to_training_reference", "conservative_reward"], ascending=[False, False]).head(16 - len(rows))
    for _, r in novelty.iterrows():
        row = r.to_dict()
        row["selected_by"] = "novelty_tophalf"
        rows.append(row)
    if len(rows) < 16:
        filler = unique[~unique["order_hash"].isin([x["order_hash"] for x in rows])].head(16 - len(rows))
        for _, r in filler.iterrows():
            row = r.to_dict()
            row["selected_by"] = "top_reward"
            rows.append(row)
    rows = rows[:16]
    for i, row in enumerate(rows, start=1):
        row["strategy_name"] = f"PPOV02K2_N40_B{i:02d}_{row['selected_by']}"
        row["notes"] = "N40 candidate refreshed after Stage K2 N40 PPO completion; not physically validated yet."
    selected = pd.DataFrame(rows)
    selected.to_csv(REFRESHED_CSV, index=False)
    for _, row in selected.iterrows():
        payload = row.to_dict()
        payload["order"] = parse_order(row["order_json"])
        (SCAN_DIR / f"scan_order_{row['strategy_name']}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return selected


def main() -> int:
    ensure_dirs()
    pool = generate_pool()
    selected = select(pool)
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rollout_pool_size": int(len(pool)),
        "unique_orders": int(pool["order_hash"].nunique()),
        "selected_count": int(len(selected)),
        "selected_path": str(REFRESHED_CSV),
        "rollout_pool_path": str(POOL_CSV),
        "all_selected_legal": all(validate_order(40, parse_order(x)) for x in selected["order_json"]),
        "duplicate_vs_training_count": int(selected["duplicate_vs_combined552_plus_v01_training"].sum()),
        "duplicate_vs_stageK_N40_selected_count": int(selected["duplicate_vs_stageK_N40_selected"].sum()),
        "branch": git_branch(),
        "no_Abaqus": True,
        "no_ODB_opening": True,
        "no_solver": True,
    }
    REFRESHED_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
