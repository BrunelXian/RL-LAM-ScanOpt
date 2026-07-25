"""Evaluate Stage K2 N40 PPO checkpoints in the surrogate environment."""

from __future__ import annotations

import json
import re
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
TABLES_DIR = K2_ROOT / "tables"
CHECKPOINT_DIR = K2_ROOT / "checkpoints"
SURR_MODEL = OUT_ROOT / "surrogate_v02" / "models" / "N40_surrogate_reward_v02.joblib"
DATASET = OUT_ROOT / "data" / "v02_targeted_N24_N40_teacher_dataset.csv"
V01_RANKING = PROJECT_ROOT / "outputs" / "stage3_ppo_rl_lam_fea_addendum_v01" / "stageI_final_ppo_evidence_freeze" / "frozen_tables" / "FROZEN_PPO_batch32_teacher_metric_ranking_full.csv"
K_SELECTED = OUT_ROOT / "candidate_generation_v02" / "selected_batch32" / "v02_ppo_targeted_N24_N40_candidate_batch32.csv"

EVAL_CSV = TABLES_DIR / "stageK2_n40_internal_eval_by_seed.csv"
SUMMARY_JSON = TABLES_DIR / "stageK2_n40_internal_eval_summary.json"


def make_env(seed: int) -> FixedNLamScanOrderPPOEnv:
    return FixedNLamScanOrderPPOEnv(V02EnvConfig(n=40, surrogate_model_path=str(SURR_MODEL), seed=seed, conservative_reward=True))


def order_compact(order: list[int]) -> str:
    return ",".join(str(x) for x in order)


def parse_order(text: Any) -> list[int]:
    return [int(x) for x in re.findall(r"-?\d+", "" if pd.isna(text) else str(text))]


def descriptors(order: list[int]) -> dict[str, float]:
    arr = np.asarray(order, dtype=float)
    jumps = np.abs(np.diff(arr))
    parity = arr.astype(int) % 2
    return {
        "mean_abs_jump": float(jumps.mean()) if len(jumps) else 0.0,
        "max_abs_jump": float(jumps.max()) if len(jumps) else 0.0,
        "adjacent_fraction": float(np.mean(jumps == 1)) if len(jumps) else 0.0,
        "parity_switch_fraction": float(np.mean(parity[1:] != parity[:-1])) if len(parity) > 1 else 0.0,
    }


def rollout(model: MaskablePPO, env: FixedNLamScanOrderPPOEnv, seed: int, deterministic: bool) -> dict[str, Any]:
    obs, _ = env.reset(seed=seed)
    done = False
    reward = 0.0
    info = {}
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic, action_masks=env.action_masks())
        obs, reward, done, _, info = env.step(int(action))
    order = [int(x) for x in info["terminal_order"]]
    comps = env.reward_model.predict_components(40, order)
    return {
        "rollout_seed": seed,
        "mode": "deterministic" if deterministic else "stochastic",
        "legal": validate_order(40, order),
        "order_compact": order_compact(order),
        "reward": float(reward),
        **comps,
        **descriptors(order),
    }


def simple_baselines(env: FixedNLamScanOrderPPOEnv) -> list[dict[str, Any]]:
    baselines = {
        "raster": list(range(40)),
        "odd_even": list(range(1, 40, 2)) + list(range(0, 40, 2)),
        "center_out": [19, 20] + [x for pair in zip(range(18, -1, -1), range(21, 40)) for x in pair],
        "edge_in": [x for pair in zip(range(40), range(39, -1, -1)) for x in pair if x not in []],
    }
    rows = []
    seen = set()
    for name, order in baselines.items():
        compact = order_compact(order)
        if compact in seen or not validate_order(40, order):
            continue
        seen.add(compact)
        comps = env.reward_model.predict_components(40, order)
        rows.append({"comparison_group": "simple_baseline", "name": name, "reward": comps["conservative_reward"], **comps})
    return rows


def reference_scores(env: FixedNLamScanOrderPPOEnv) -> dict[str, float]:
    df = pd.read_csv(DATASET)
    sub = df[df["n"].astype(int) == 40]
    rewards = []
    for _, row in sub.iterrows():
        order = parse_order(row["order_json"])
        rewards.append(env.reward_model.predict_components(40, order)["conservative_reward"])
    return {"combined_plus_v01_N40_count": len(rewards), "reference_mean_reward": float(np.mean(rewards)), "reference_median_reward": float(np.median(rewards)), "reference_max_reward": float(np.max(rewards))}


def main() -> int:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for ckpt in sorted(CHECKPOINT_DIR.glob("N40_seed*_maskable_ppo_v02_K2.zip")):
        seed_match = re.search(r"seed(\d+)", ckpt.name)
        seed = int(seed_match.group(1)) if seed_match else 20260624
        model = MaskablePPO.load(ckpt)
        env = make_env(seed)
        deterministic_row = rollout(model, env, seed, True)
        deterministic_row.update({"checkpoint": str(ckpt), "ppo_seed": seed})
        rows.append(deterministic_row)
        for i in range(500):
            row = rollout(model, env, seed + 1000 + i, False)
            row.update({"checkpoint": str(ckpt), "ppo_seed": seed})
            rows.append(row)
    eval_df = pd.DataFrame(rows)
    eval_df.to_csv(EVAL_CSV, index=False)
    env = make_env(20260624)
    baselines = simple_baselines(env)
    ref = reference_scores(env)
    original_k = pd.read_csv(K_SELECTED)
    original_n40 = original_k[original_k["n"].astype(int) == 40]
    original_rewards = pd.to_numeric(original_n40["conservative_reward"], errors="coerce")
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint_count": int(eval_df["checkpoint"].nunique()),
        "rollout_count": int(len(eval_df)),
        "all_legal": bool(eval_df["legal"].all()),
        "unique_orders": int(eval_df["order_compact"].nunique()),
        "mean_reward": float(eval_df["reward"].mean()),
        "max_reward": float(eval_df["reward"].max()),
        "by_seed": eval_df.groupby("ppo_seed").agg(count=("reward", "size"), mean_reward=("reward", "mean"), max_reward=("reward", "max"), unique_orders=("order_compact", "nunique"), legal_all=("legal", "all")).reset_index().to_dict(orient="records"),
        "simple_baselines": baselines,
        "reference_surrogate_space": ref,
        "original_stageK_N40_selected_mean_conservative_reward": float(original_rewards.mean()),
        "original_stageK_N40_selected_max_conservative_reward": float(original_rewards.max()),
        "no_Abaqus": True,
        "no_ODB_opening": True,
        "no_solver": True,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
