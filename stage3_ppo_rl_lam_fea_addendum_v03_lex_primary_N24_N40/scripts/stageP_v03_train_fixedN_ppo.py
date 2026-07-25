"""Train fixed-N MaskablePPO v03 policies for N24 and N40."""

from __future__ import annotations

import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed


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
LOG_DIR = TRAIN_DIR / "logs"
TABLES_DIR = TRAIN_DIR / "tables"
DOCS_DIR = PROJECT_ROOT / "docs" / NS
CONFIG_JSON = TRAIN_DIR / "v03_ppo_training_config.json"
SUMMARY_JSON = TRAIN_DIR / "v03_ppo_training_summary.json"
SUMMARY_CSV = TABLES_DIR / "v03_ppo_training_summary_by_N_seed.csv"
PARAM_JSON = TRAIN_DIR / "v03_ppo_parameter_count.json"
MONITOR_CSV = LOG_DIR / "v03_ppo_training_monitor.csv"
MASK_CSV = TABLES_DIR / "v03_action_mask_audit.csv"

SEEDS_BY_N = {24: [20260627], 40: [20260627]}
# Runtime-limited Stage P pass: full requested budgets are recorded in the
# config, while the executable pass freezes one seed/N. N24 can reuse the
# saved 100k intermediate checkpoint from an interrupted long run.
TIMESTEPS_BY_N = {24: 100000, 40: 60000}
REQUESTED_TIMESTEPS_BY_N = {24: 400000, 40: 700000}


class EpisodeLogger(BaseCallback):
    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        self.rows: list[dict[str, Any]] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [None])
        for info in infos:
            if info.get("terminal_order") is not None:
                self.rows.append(
                    {
                        "timesteps": self.num_timesteps,
                        "n": info.get("n"),
                        "episode_length": len(info.get("terminal_order") or []),
                        "terminal_reward": float(rewards[0]) if rewards is not None else "",
                        "legality": info.get("legality"),
                    }
                )
        if self.rows and len(self.rows) % 100 == 0:
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
    for path in [CHECKPOINT_DIR, LOG_DIR, TABLES_DIR, DOCS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def make_env(n: int, seed: int) -> V03FixedNLamScanOrderPPOEnv:
    return V03FixedNLamScanOrderPPOEnv(
        V03EnvConfig(
            n=n,
            surrogate_model_path=str(SURR_MODELS / f"N{n}_v03_lex_primary_surrogate.joblib"),
            reward_clip_min=-1.0,
            reward_clip_max=1.5,
            reward_scale=1.0,
            seed=seed,
        )
    )


def parameter_count(model: MaskablePPO) -> dict[str, int]:
    total = sum(p.numel() for p in model.policy.parameters())
    trainable = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    return {"total_parameters": int(total), "trainable_parameters": int(trainable)}


def verify_masks() -> list[dict[str, Any]]:
    rows = []
    for n in [24, 40]:
        env = make_env(n, 123)
        obs, _ = env.reset()
        initial = env.action_masks()
        env.step(0)
        after = env.action_masks()
        rows.append({"n": n, "mask_length": len(initial), "initial_valid": int(initial.sum()), "after_one_valid": int(after.sum()), "pass": bool(len(initial) == 40 and int(initial.sum()) == n and int(after.sum()) == n - 1)})
    with MASK_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def latest_intermediate_checkpoint(n: int, seed: int) -> Path | None:
    checkpoints = sorted(
        CHECKPOINT_DIR.glob(f"N{n}_seed{seed}_checkpoint_*_steps.zip"),
        key=lambda p: int(re.search(r"_checkpoint_(\d+)_steps", p.name).group(1)) if re.search(r"_checkpoint_(\d+)_steps", p.name) else -1,
    )
    return checkpoints[-1] if checkpoints else None


def train_one(n: int, seed: int, config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    set_random_seed(seed)
    torch.manual_seed(seed)
    env = make_env(n, seed)
    ckpt = CHECKPOINT_DIR / f"N{n}_seed{seed}_maskable_ppo_v03.zip"
    if ckpt.exists():
        model = MaskablePPO.load(ckpt, env=env)
        mode = "existing_checkpoint_reused"
        learned = int(model.num_timesteps)
    elif latest_intermediate_checkpoint(n, seed) is not None:
        latest = latest_intermediate_checkpoint(n, seed)
        assert latest is not None
        model = MaskablePPO.load(latest, env=env)
        model.save(ckpt)
        mode = f"interrupted_long_run_intermediate_frozen_from_{latest.name}"
        learned = int(model.num_timesteps)
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
            target_kl=float(config["target_kl"]),
            seed=seed,
            verbose=int(config["verbose"]),
            tensorboard_log=str(LOG_DIR / "tensorboard"),
        )
        logger = EpisodeLogger(MONITOR_CSV)
        checkpoint_cb = CheckpointCallback(save_freq=25000, save_path=str(CHECKPOINT_DIR), name_prefix=f"N{n}_seed{seed}_checkpoint")
        total = int(config["total_timesteps_by_N"][str(n)])
        model.learn(total_timesteps=total, callback=[logger, checkpoint_cb], progress_bar=False)
        model.save(ckpt)
        mode = "fresh_one_seed_training"
        learned = total
    params = parameter_count(model)
    return ckpt, {"n": n, "seed": seed, "checkpoint": str(ckpt), "mode": mode, "target_timesteps": int(config["total_timesteps_by_N"][str(n)]), "timesteps_completed": int(model.num_timesteps if model.num_timesteps else learned), "parameter_count": params}


def main() -> int:
    ensure_dirs()
    config = {
        "algorithm": "MaskablePPO",
        "policy": "MlpPolicy",
        "seeds_by_N": {str(k): v for k, v in SEEDS_BY_N.items()},
        "deferred_optional_seeds": {"24": [20260628], "40": [20260628]},
        "total_timesteps_by_N": {"24": TIMESTEPS_BY_N[24], "40": TIMESTEPS_BY_N[40]},
        "requested_full_timesteps_by_N": {"24": REQUESTED_TIMESTEPS_BY_N[24], "40": REQUESTED_TIMESTEPS_BY_N[40]},
        "gamma": 1.0,
        "gae_lambda": 1.0,
        "learning_rate": 2e-4,
        "n_steps": 2048,
        "batch_size": 256,
        "ent_coef": 0.04,
        "clip_range": 0.2,
        "target_kl": 0.03,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "verbose": 1,
        "runtime_policy": "train one seed per N first; optional second seeds deferred unless explicitly requested",
    }
    CONFIG_JSON.write_text(json.dumps(config, indent=2), encoding="utf-8")
    mask_rows = verify_masks()
    summaries = []
    checkpoints = []
    params_by_ckpt = {}
    for n in [24, 40]:
        for seed in SEEDS_BY_N[n]:
            ckpt, summary = train_one(n, seed, config)
            summaries.append(summary)
            checkpoints.append(str(ckpt))
            params_by_ckpt[str(ckpt)] = summary["parameter_count"]
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "seed", "checkpoint", "mode", "target_timesteps", "timesteps_completed", "parameter_count"])
        writer.writeheader()
        for row in summaries:
            r = dict(row)
            r["parameter_count"] = json.dumps(r["parameter_count"])
            writer.writerow(r)
    PARAM_JSON.write_text(json.dumps(params_by_ckpt, indent=2), encoding="utf-8")
    all_complete = all(row["timesteps_completed"] >= row["target_timesteps"] for row in summaries)
    masks_ok = all(row["pass"] for row in mask_rows)
    full_budget_complete = all(row["timesteps_completed"] >= REQUESTED_TIMESTEPS_BY_N[int(row["n"])] for row in summaries)
    verdict = "PASS_V03_PPO_TRAINING_COMPLETE_ONE_SEED_PER_N" if all_complete and masks_ok and full_budget_complete else "WARNING_V03_PPO_TRAINING_PARTIAL_REVIEW"
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "config_path": str(CONFIG_JSON),
        "checkpoints": checkpoints,
        "summaries": summaries,
        "mask_audit": str(MASK_CSV),
        "monitor_csv": str(MONITOR_CSV),
        "parameter_count_json": str(PARAM_JSON),
        "no_Abaqus": True,
        "no_ODB_opening": True,
        "no_solver": True,
        "no_CAE_INP_JNL": True,
    }
    SUMMARY_JSON.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0 if not verdict.startswith("FAIL") else 1


if __name__ == "__main__":
    raise SystemExit(main())
