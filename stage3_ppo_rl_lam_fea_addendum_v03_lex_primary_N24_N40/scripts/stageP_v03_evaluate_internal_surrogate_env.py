"""Internal surrogate-environment evaluation for PPO v03 checkpoints."""

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
NS = "stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40"
SRC_DIR = PROJECT_ROOT / NS / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from v03_fixedN_ppo_env import V03EnvConfig, V03FixedNLamScanOrderPPOEnv, validate_order  # noqa: E402

OUT_ROOT = PROJECT_ROOT / "outputs" / NS
SURR_MODELS = OUT_ROOT / "surrogate_v03" / "models"
TRAIN_DIR = OUT_ROOT / "ppo_training_v03"
CHECKPOINT_DIR = TRAIN_DIR / "checkpoints"
TABLES_DIR = TRAIN_DIR / "tables"
DOCS_DIR = PROJECT_ROOT / "docs" / NS
EVAL_CSV = TABLES_DIR / "v03_internal_eval_by_N_seed.csv"
EVAL_JSON = TABLES_DIR / "v03_internal_eval_summary.json"
REPORT = DOCS_DIR / "PPO_V03_TRAINING_AND_INTERNAL_EVAL_REPORT.md"
ROLLOUTS_PER_POLICY = 20


def parse_seed(path: Path) -> int:
    m = re.search(r"seed(\d+)", path.name)
    return int(m.group(1)) if m else 20260627


def make_env(n: int, seed: int) -> V03FixedNLamScanOrderPPOEnv:
    return V03FixedNLamScanOrderPPOEnv(V03EnvConfig(n=n, surrogate_model_path=str(SURR_MODELS / f"N{n}_v03_lex_primary_surrogate.joblib"), seed=seed))


def order_compact(order: list[int]) -> str:
    return ",".join(str(x) for x in order)


def descriptors(order: list[int]) -> dict[str, float]:
    arr = np.asarray(order, dtype=float)
    jumps = np.abs(np.diff(arr))
    parity = arr.astype(int) % 2
    center = (len(order) - 1) / 2.0
    early = arr[: max(1, len(order) // 4)]
    return {
        "mean_abs_jump": float(jumps.mean()),
        "max_abs_jump": float(jumps.max()),
        "adjacent_fraction": float(np.mean(jumps == 1)),
        "parity_switch_fraction": float(np.mean(parity[1:] != parity[:-1])),
        "early_center_bias": float(np.mean(np.abs(early - center) / max(1.0, center))),
        "early_edge_bias": float(np.mean(np.minimum(early, len(order) - 1 - early) / max(1.0, center))),
    }


def rollout(model: MaskablePPO, n: int, seed: int, deterministic: bool) -> dict[str, Any]:
    env = make_env(n, seed)
    obs, _ = env.reset(seed=seed)
    done = False
    info = {}
    reward = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic, action_masks=env.action_masks())
        obs, reward, done, _, info = env.step(int(action))
    order = [int(x) for x in info["terminal_order"]]
    comps = env.reward_model.predict_components(n, order)
    return {
        "n": n,
        "rollout_seed": seed,
        "mode": "deterministic" if deterministic else "stochastic",
        "order_compact": order_compact(order),
        "legal": validate_order(n, order),
        **comps,
        **descriptors(order),
    }


def main() -> int:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    ckpts = sorted(CHECKPOINT_DIR.glob("N*_seed*_maskable_ppo_v03.zip"))
    if not ckpts:
        raise FileNotFoundError(f"No v03 checkpoints in {CHECKPOINT_DIR}")
    for ckpt in ckpts:
        n = int(re.search(r"N(\d+)_", ckpt.name).group(1))
        seed = parse_seed(ckpt)
        model = MaskablePPO.load(ckpt)
        base = rollout(model, n, seed, True)
        base.update({"checkpoint": str(ckpt), "ppo_seed": seed})
        rows.append(base)
        for i in range(ROLLOUTS_PER_POLICY):
            r = rollout(model, n, seed + 10000 + i, False)
            r.update({"checkpoint": str(ckpt), "ppo_seed": seed})
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(EVAL_CSV, index=False)
    summary_rows = []
    for (n, seed), sub in df.groupby(["n", "ppo_seed"]):
        summary_rows.append({
            "n": int(n),
            "seed": int(seed),
            "rollout_count": int(len(sub)),
            "all_legal": bool(sub["legal"].all()),
            "unique_order_count": int(sub["order_compact"].nunique()),
            "mean_terminal_reward": float(sub["terminal_reward"].mean()),
            "max_terminal_reward": float(sub["terminal_reward"].max()),
            "mean_lex_primary_score": float(sub["predicted_lex_primary_score"].mean()),
            "max_lex_primary_score": float(sub["predicted_lex_primary_score"].max()),
            "mean_u2_guarded_score": float(sub["predicted_u2_guarded_score"].mean()),
            "max_u2_guarded_score": float(sub["predicted_u2_guarded_score"].max()),
            "surfaceT_false_positive_penalty_count": int((sub["surfaceT_only_false_positive_penalty"] > 0).sum()),
        })
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "verdict": "PASS_V03_PPO_READY_FOR_CANDIDATE_GENERATION" if all(r["all_legal"] for r in summary_rows) else "WARNING_V03_PPO_PARTIAL_REVIEW",
        "rollout_csv": str(EVAL_CSV),
        "summary_by_N_seed": summary_rows,
        "no_Abaqus": True,
        "no_ODB_opening": True,
        "no_solver": True,
    }
    EVAL_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    REPORT.write_text(
        "# PPO v03 Training And Internal Surrogate Evaluation Report\n\n"
        f"Evaluation table: `{EVAL_CSV}`\n\n"
        f"Summary JSON: `{EVAL_JSON}`\n\n"
        f"Verdict: `{summary['verdict']}`\n\n"
        "This is surrogate-environment evaluation only. No physical validation is claimed.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
