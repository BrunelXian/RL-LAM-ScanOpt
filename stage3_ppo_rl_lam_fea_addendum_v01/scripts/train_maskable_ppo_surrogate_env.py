"""Train MaskablePPO in the frozen surrogate reward environment.

This is PPO training, but not online Abaqus training. The policy interacts only
with the surrogate terminal reward environment derived from teacher-labelled
data. No final Abaqus candidate orders are generated here.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from ppo_surrogate_env_wrapper import (  # noqa: E402
    PPOSurrogateEnvConfig,
    LamScanOrderSurrogateRewardEnv,
    verify_action_masks_for_supported_n,
)


PROJECT_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
BRANCH = "stage3-variable-n-graph-pointer-init-v01"
NAMESPACE = "stage3_ppo_rl_lam_fea_addendum_v01"
TRAIN_DIR = PROJECT_ROOT / "outputs" / NAMESPACE / "ppo_training"
CHECKPOINT_DIR = TRAIN_DIR / "checkpoints"
LOG_DIR = TRAIN_DIR / "logs"
TABLES_DIR = TRAIN_DIR / "tables"
REPORTS_DIR = TRAIN_DIR / "reports"
PLOTS_DIR = TRAIN_DIR / "plots"
CONFIG_PATH = TRAIN_DIR / "ppo_training_config.json"
FINAL_MODEL_PATH = CHECKPOINT_DIR / "maskable_ppo_lam_scan_order_final.zip"
MONITOR_CSV = LOG_DIR / "ppo_training_monitor.csv"
SUMMARY_JSON = REPORTS_DIR / "ppo_training_summary.json"
PARAMETER_COUNT_JSON = REPORTS_DIR / "ppo_parameter_count.json"
ACTION_MASK_AUDIT_CSV = TABLES_DIR / "ppo_action_mask_verification.csv"


class EpisodeCSVCallback(BaseCallback):
    def __init__(self, output_path: Path) -> None:
        super().__init__()
        self.output_path = output_path
        self.rows: list[dict[str, Any]] = []
        self.fieldnames = [
            "time_elapsed_sec",
            "num_timesteps",
            "episode_index",
            "n",
            "episode_length",
            "terminal_reward",
            "terminated",
            "truncated",
            "legal",
            "legality",
            "illegal_action",
            "terminal_order",
        ]

    def _on_training_start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writeheader()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            record = info.get("episode_record") if isinstance(info, dict) else None
            if not record:
                continue
            row = {
                "time_elapsed_sec": float(time.time() - self.training_env.get_attr("_training_start_time")[0])
                if hasattr(self.training_env, "get_attr")
                else None,
                "num_timesteps": int(self.num_timesteps),
                **record,
            }
            self.rows.append(row)
            with self.output_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
                writer.writerow(row)
        return True


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def make_env_config(config: dict[str, Any]) -> PPOSurrogateEnvConfig:
    return PPOSurrogateEnvConfig(
        model_path=config["surrogate_model_path"],
        feature_schema_path=config["feature_schema_path"],
        target_schema_path=config["target_schema_path"],
        fixed_n=config.get("fixed_n"),
        random_n=bool(config.get("random_n", True)),
        supported_n=tuple(int(value) for value in config["supported_n"]),
        n_sampling_mode=config.get("n_sampling_mode", "balanced"),
        reward_clip_min=config.get("reward_clip_min"),
        reward_clip_max=config.get("reward_clip_max"),
        reward_scale=float(config.get("reward_scale", 1.0)),
        illegal_action_penalty=float(config.get("illegal_action_penalty", -100.0)),
        seed=int(config["seed"]),
    )


def parameter_count(model: MaskablePPO) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.policy.parameters())
    trainable = sum(parameter.numel() for parameter in model.policy.parameters() if parameter.requires_grad)
    return {
        "policy_parameter_count_total": int(total),
        "policy_parameter_count_trainable": int(trainable),
    }


def latest_checkpoint() -> tuple[Path | None, int]:
    checkpoints = []
    for path in CHECKPOINT_DIR.glob("maskable_ppo_lam_scan_order_checkpoint_*_steps.zip"):
        stem = path.stem
        try:
            steps = int(stem.split("_")[-2])
        except (IndexError, ValueError):
            continue
        checkpoints.append((steps, path))
    if not checkpoints:
        return None, 0
    steps, path = sorted(checkpoints)[-1]
    return path, steps


def main() -> int:
    for directory in [CHECKPOINT_DIR, LOG_DIR, TABLES_DIR, REPORTS_DIR, PLOTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    config = load_config()
    set_random_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))

    env_config = make_env_config(config)
    mask_rows = verify_action_masks_for_supported_n(env_config)
    with ACTION_MASK_AUDIT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mask_rows[0].keys()))
        writer.writeheader()
        writer.writerows(mask_rows)
    if not all(row["pass"] for row in mask_rows):
        raise RuntimeError(f"Action mask verification failed: {mask_rows}")

    env = LamScanOrderSurrogateRewardEnv.from_config(env_config)
    env._training_start_time = time.time()

    tensorboard_log = config.get("tensorboard_log")
    resume_checkpoint, resume_steps = (None, 0) if FINAL_MODEL_PATH.exists() else latest_checkpoint()
    if FINAL_MODEL_PATH.exists():
        model = MaskablePPO.load(FINAL_MODEL_PATH, env=env)
        resume_mode = "final_checkpoint_already_exists_reopened_for_summary"
    elif resume_checkpoint is not None and resume_steps < int(config["total_timesteps"]):
        model = MaskablePPO.load(resume_checkpoint, env=env, tensorboard_log=tensorboard_log)
        resume_mode = f"resumed_from_{resume_checkpoint.name}"
    else:
        model = MaskablePPO(
            policy=config["policy"],
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
            seed=int(config["seed"]),
            verbose=int(config["verbose"]),
            tensorboard_log=tensorboard_log,
        )
        resume_mode = "fresh_training_run"

    callbacks = CallbackList(
        [
            EpisodeCSVCallback(MONITOR_CSV),
            CheckpointCallback(
                save_freq=50000,
                save_path=str(CHECKPOINT_DIR),
                name_prefix="maskable_ppo_lam_scan_order_checkpoint",
            ),
        ]
    )

    start = time.time()
    target_total = int(config["total_timesteps"])
    if model.num_timesteps >= target_total:
        learn_timesteps = 0
    elif resume_mode.startswith("resumed_from"):
        learn_timesteps = target_total - int(model.num_timesteps)
    else:
        learn_timesteps = target_total
    if learn_timesteps > 0:
        model.learn(
            total_timesteps=learn_timesteps,
            callback=callbacks,
            progress_bar=False,
            reset_num_timesteps=not resume_mode.startswith("resumed_from"),
        )
    elapsed = time.time() - start
    model.save(FINAL_MODEL_PATH)

    params = {
        **parameter_count(model),
        "algorithm": config["algorithm"],
        "policy": config["policy"],
    }
    PARAMETER_COUNT_JSON.write_text(json.dumps(params, indent=2), encoding="utf-8")

    episode_rows = []
    if MONITOR_CSV.exists():
        with MONITOR_CSV.open("r", newline="", encoding="utf-8") as handle:
            episode_rows = list(csv.DictReader(handle))
    rewards = [float(row["terminal_reward"]) for row in episode_rows if row.get("terminal_reward") not in (None, "")]
    lengths = [int(row["episode_length"]) for row in episode_rows if row.get("episode_length") not in (None, "")]
    summary = {
        "branch": BRANCH,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_path": str(CONFIG_PATH),
        "surrogate_model_path": config["surrogate_model_path"],
        "algorithm": config["algorithm"],
        "policy": config["policy"],
        "requested_total_timesteps": int(config["total_timesteps"]),
        "total_timesteps_completed": int(model.num_timesteps),
        "learn_timesteps_this_invocation": int(learn_timesteps),
        "resume_mode": resume_mode,
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
        "resume_checkpoint_steps": int(resume_steps),
        "training_elapsed_sec": elapsed,
        "n_envs": int(config["n_envs"]),
        "vectorization_status": config.get("vectorization_status"),
        "checkpoint_path": str(FINAL_MODEL_PATH),
        "monitor_csv": str(MONITOR_CSV),
        "action_mask_audit_csv": str(ACTION_MASK_AUDIT_CSV),
        "episode_count_logged": len(episode_rows),
        "mean_terminal_reward": float(np.mean(rewards)) if rewards else None,
        "min_terminal_reward": float(np.min(rewards)) if rewards else None,
        "max_terminal_reward": float(np.max(rewards)) if rewards else None,
        "mean_episode_length": float(np.mean(lengths)) if lengths else None,
        "parameter_count_path": str(PARAMETER_COUNT_JSON),
        "parameter_count": params,
        "action_masks_verified": all(row["pass"] for row in mask_rows),
        "no_Abaqus": True,
        "no_ODB": True,
        "no_solver": True,
        "no_CAE_INP_JNL": True,
        "no_final_candidate_generation": True,
        "no_commit_or_push": True,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
