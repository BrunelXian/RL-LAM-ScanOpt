"""Train fixed-N MaskablePPO v02 policies for N24 and N40."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

PROJECT_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
V02_NAMESPACE = "stage3_ppo_rl_lam_fea_addendum_v02_targeted_N24_N40"
SRC_DIR = PROJECT_ROOT / V02_NAMESPACE / "src"
sys.path.insert(0, str(SRC_DIR))

from v02_fixedN_ppo_env import FixedNLamScanOrderPPOEnv, V02EnvConfig, validate_order  # noqa: E402

OUT_ROOT = PROJECT_ROOT / "outputs" / V02_NAMESPACE
SURR_MODELS = OUT_ROOT / "surrogate_v02" / "models"
TRAIN_DIR = OUT_ROOT / "ppo_training_v02"
CHECKPOINT_DIR = TRAIN_DIR / "checkpoints"
LOG_DIR = TRAIN_DIR / "logs"
TABLES_DIR = TRAIN_DIR / "tables"
REPORT = PROJECT_ROOT / "docs" / V02_NAMESPACE / "PPO_V02_TRAINING_REPORT.md"
CONFIG_JSON = TRAIN_DIR / "v02_ppo_training_config.json"
SUMMARY_JSON = TRAIN_DIR / "v02_ppo_training_summary.json"
PARAM_JSON = TRAIN_DIR / "v02_ppo_parameter_count.json"
MONITOR_CSV = LOG_DIR / "v02_ppo_training_monitor.csv"
EVAL_CSV = TABLES_DIR / "v02_internal_eval_by_N_seed.csv"
MASK_CSV = TABLES_DIR / "v02_action_mask_audit.csv"

SEEDS_BY_N = {24: [20260624], 40: [20260624]}
TIMESTEPS_BY_N = {24: 300000, 40: 100000}
VERDICT = "WARNING_V02_FIXED_N_PPO_PARTIAL_REVIEW"


class EpisodeLogger(BaseCallback):
    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        self.rows: list[dict[str, Any]] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if info.get("terminal_order") is not None:
                self.rows.append(
                    {
                        "timesteps": self.num_timesteps,
                        "n": info.get("n"),
                        "episode_length": len(info.get("terminal_order") or []),
                        "terminal_reward": self.locals.get("rewards", [None])[0],
                        "legality": info.get("legality"),
                    }
                )
        if len(self.rows) and len(self.rows) % 100 == 0:
            self.flush()
        return True

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["timesteps", "n", "episode_length", "terminal_reward", "legality"])
            writer.writeheader()
            writer.writerows(self.rows)

    def _on_training_end(self) -> None:
        self.flush()


def ensure_dirs() -> None:
    for p in [CHECKPOINT_DIR, LOG_DIR, TABLES_DIR, REPORT.parent]:
        p.mkdir(parents=True, exist_ok=True)


def parameter_count(model: MaskablePPO) -> dict[str, int]:
    total = sum(p.numel() for p in model.policy.parameters())
    trainable = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    return {"total_parameters": int(total), "trainable_parameters": int(trainable)}


def simple_markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def make_env(n: int, seed: int) -> FixedNLamScanOrderPPOEnv:
    return FixedNLamScanOrderPPOEnv(
        V02EnvConfig(
            n=n,
            surrogate_model_path=str(SURR_MODELS / f"N{n}_surrogate_reward_v02.joblib"),
            reward_clip_min=-1.0,
            reward_clip_max=1.25,
            reward_scale=1.0,
            seed=seed,
            conservative_reward=True,
        )
    )


def latest_intermediate_checkpoint(n: int, seed: int) -> Path | None:
    checkpoints = sorted(
        CHECKPOINT_DIR.glob(f"N{n}_seed{seed}_checkpoint_*_steps.zip"),
        key=lambda path: int(path.stem.split("_checkpoint_")[-1].split("_steps")[0]),
    )
    return checkpoints[-1] if checkpoints else None


def verify_masks() -> list[dict[str, Any]]:
    rows = []
    for n in [24, 40]:
        env = make_env(n, 123)
        obs, _ = env.reset()
        initial = env.action_masks()
        env.step(0)
        after = env.action_masks()
        rows.append({"n": n, "initial_valid": int(initial.sum()), "after_one_valid": int(after.sum()), "mask_length": len(initial), "pass": bool(len(initial) == 40 and int(initial.sum()) == n and int(after.sum()) == n - 1)})
    with MASK_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def rollout(model: MaskablePPO, n: int, seed: int, deterministic: bool) -> dict[str, Any]:
    env = make_env(n, seed)
    obs, _ = env.reset(seed=seed)
    done = False
    reward = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic, action_masks=env.action_masks())
        obs, reward, done, _, info = env.step(int(action))
    order = info["terminal_order"]
    return {"n": n, "seed": seed, "mode": "deterministic" if deterministic else "stochastic", "reward": float(reward), "legal": validate_order(n, order), "order_compact": ",".join(map(str, order))}


def evaluate_model(model: MaskablePPO, n: int, seed: int) -> list[dict[str, Any]]:
    rows = [rollout(model, n, seed, True)]
    for i in range(200):
        rows.append(rollout(model, n, seed + 1000 + i, False))
    return rows


def train_one(n: int, seed: int, config: dict[str, Any]) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    set_random_seed(seed)
    torch.manual_seed(seed)
    env = make_env(n, seed)
    ckpt = CHECKPOINT_DIR / f"N{n}_seed{seed}_maskable_ppo_v02.zip"
    if ckpt.exists():
        model = MaskablePPO.load(ckpt, env=env)
        learned = 0
        mode = "existing_checkpoint_reused"
    elif latest_intermediate_checkpoint(n, seed) is not None:
        latest = latest_intermediate_checkpoint(n, seed)
        assert latest is not None
        model = MaskablePPO.load(latest, env=env)
        learned = int(model.num_timesteps)
        mode = f"partial_checkpoint_frozen_from_{latest.name}"
        model.save(ckpt)
    else:
        model = MaskablePPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=float(config["learning_rate"]),
            n_steps=int(config["n_steps"]),
            batch_size=int(config["batch_size"]),
            gamma=float(config["gamma"]),
            gae_lambda=float(config["gae_lambda"]),
            clip_range=float(config["clip_range"]),
            ent_coef=float(config["ent_coef"]),
            vf_coef=float(config["vf_coef"]),
            max_grad_norm=float(config["max_grad_norm"]),
            seed=seed,
            verbose=int(config["verbose"]),
            tensorboard_log=str(LOG_DIR / "tensorboard"),
        )
        logger = EpisodeLogger(MONITOR_CSV)
        checkpoint_cb = CheckpointCallback(save_freq=100000, save_path=str(CHECKPOINT_DIR), name_prefix=f"N{n}_seed{seed}_checkpoint")
        model.learn(total_timesteps=int(config["total_timesteps_by_N"][str(n)]), callback=[logger, checkpoint_cb], progress_bar=False)
        model.save(ckpt)
        learned = int(config["total_timesteps_by_N"][str(n)])
        mode = "fresh_one_seed_training"
    params = parameter_count(model)
    eval_rows = evaluate_model(model, n, seed)
    summary = {"n": n, "seed": seed, "checkpoint": str(ckpt), "learn_timesteps": learned, "mode": mode, "num_timesteps": int(model.num_timesteps), "parameter_count": params, "eval_mean_reward": float(np.mean([r["reward"] for r in eval_rows])), "eval_legal_all": all(r["legal"] for r in eval_rows)}
    return ckpt, summary, eval_rows


def main() -> int:
    ensure_dirs()
    config = {
        "algorithm": "MaskablePPO",
        "policy": "MlpPolicy",
        "seed_base": 20260624,
        "seeds_by_N": {str(k): v for k, v in SEEDS_BY_N.items()},
        "training_seed_policy": "one_seed_per_N_v02_initial_pass; N40 frozen at 100k partial checkpoint after long-run timeout",
        "total_timesteps_by_N": {"24": TIMESTEPS_BY_N[24], "40": TIMESTEPS_BY_N[40]},
        "learning_rate": 3e-4,
        "n_steps": 1024,
        "batch_size": 256,
        "gamma": 1.0,
        "gae_lambda": 1.0,
        "clip_range": 0.2,
        "ent_coef": 0.02,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "verbose": 1,
    }
    CONFIG_JSON.write_text(json.dumps(config, indent=2), encoding="utf-8")
    masks = verify_masks()
    summaries = []
    eval_rows = []
    checkpoints = []
    for n, seeds in SEEDS_BY_N.items():
        for seed in seeds:
            ckpt, summary, rows = train_one(n, seed, config)
            checkpoints.append(str(ckpt))
            summaries.append(summary)
            eval_rows.extend(rows)
    pd.DataFrame(eval_rows).to_csv(EVAL_CSV, index=False)
    PARAM_JSON.write_text(json.dumps({f"N{s['n']}_seed{s['seed']}": s["parameter_count"] for s in summaries}, indent=2), encoding="utf-8")
    summary_json = {"timestamp": datetime.now(timezone.utc).isoformat(), "verdict": VERDICT, "checkpoints": checkpoints, "summaries": summaries, "action_masks_verified": all(r["pass"] for r in masks), "eval_csv": str(EVAL_CSV), "no_Abaqus": True, "no_ODB_opening": True, "no_solver": True}
    SUMMARY_JSON.write_text(json.dumps(summary_json, indent=2), encoding="utf-8")
    by = pd.DataFrame(eval_rows).groupby(["n", "mode"]).agg(count=("reward", "size"), mean_reward=("reward", "mean"), max_reward=("reward", "max"), unique_orders=("order_compact", "nunique"), legal_all=("legal", "all")).reset_index()
    REPORT.write_text(
        "# PPO v02 Fixed-N Training Report\n\n"
        "## Training Scope\n\nOne seed per N was trained for the initial targeted v02 pass. The optional three-seed ensemble was deferred and is not claimed.\n\n"
        f"## Checkpoints\n\n" + "\n".join(f"- `{p}`" for p in checkpoints) + "\n\n"
        "## Internal Surrogate Evaluation\n\n" + simple_markdown_table(by) + "\n\n"
        f"## Verdict\n\n`{VERDICT}`\n",
        encoding="utf-8",
    )
    print(json.dumps(summary_json, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
