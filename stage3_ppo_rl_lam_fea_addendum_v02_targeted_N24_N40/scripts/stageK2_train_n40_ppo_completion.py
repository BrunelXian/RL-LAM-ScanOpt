"""Continue N40 PPO v02 training for Stage K2.

Only normal Python and MaskablePPO training are used. No Abaqus, ODB, solver,
datacheck, enqueue, or CAE/INP/JNL generation is performed.
"""

from __future__ import annotations

import csv
import json
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
NS = "stage3_ppo_rl_lam_fea_addendum_v02_targeted_N24_N40"
SRC_DIR = PROJECT_ROOT / NS / "src"
sys.path.insert(0, str(SRC_DIR))

from v02_fixedN_ppo_env import FixedNLamScanOrderPPOEnv, V02EnvConfig  # noqa: E402

OUT_ROOT = PROJECT_ROOT / "outputs" / NS
K2_ROOT = OUT_ROOT / "stageK2_n40_completion"
CHECKPOINT_DIR = K2_ROOT / "checkpoints"
LOG_DIR = K2_ROOT / "logs"
TABLES_DIR = K2_ROOT / "tables"
SURR_MODEL = OUT_ROOT / "surrogate_v02" / "models" / "N40_surrogate_reward_v02.joblib"
PARTIAL_K_CKPT = OUT_ROOT / "ppo_training_v02" / "checkpoints" / "N40_seed20260624_maskable_ppo_v02.zip"

SUMMARY_JSON = TABLES_DIR / "stageK2_n40_training_summary.json"
PARAM_JSON = TABLES_DIR / "stageK2_n40_parameter_count.json"
MONITOR_CSV = LOG_DIR / "stageK2_n40_training_monitor.csv"
CONFIG_JSON = TABLES_DIR / "stageK2_n40_training_config.json"

SEED_TARGETS = {
    20260624: {"target_total_timesteps": 500000, "source": str(PARTIAL_K_CKPT), "train": True},
    20260625: {"target_total_timesteps": 300000, "source": None, "train": False},
    20260626: {"target_total_timesteps": 300000, "source": None, "train": False},
}
VERDICT = "WARNING_STAGEK2_N40_COMPLETION_SINGLE_SEED_ONLY"


class EpisodeLogger(BaseCallback):
    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        self.rows: list[dict[str, Any]] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if info.get("terminal_order") is not None:
                reward = self.locals.get("rewards", [None])[0]
                self.rows.append({
                    "timesteps": self.num_timesteps,
                    "n": info.get("n"),
                    "episode_length": len(info.get("terminal_order") or []),
                    "terminal_reward": float(reward) if reward is not None else "",
                    "legality": info.get("legality"),
                })
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
    for p in [CHECKPOINT_DIR, LOG_DIR, TABLES_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def make_env(seed: int) -> FixedNLamScanOrderPPOEnv:
    return FixedNLamScanOrderPPOEnv(
        V02EnvConfig(
            n=40,
            surrogate_model_path=str(SURR_MODEL),
            seed=seed,
            reward_clip_min=-1.0,
            reward_clip_max=1.25,
            conservative_reward=True,
        )
    )


def parameter_count(model: MaskablePPO) -> dict[str, int]:
    total = sum(p.numel() for p in model.policy.parameters())
    trainable = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    return {"total_parameters": int(total), "trainable_parameters": int(trainable)}


def train_seed(seed: int, cfg: dict[str, Any]) -> dict[str, Any]:
    out_ckpt = CHECKPOINT_DIR / f"N40_seed{seed}_maskable_ppo_v02_K2.zip"
    target = int(cfg["target_total_timesteps"])
    if not cfg.get("train", False):
        return {"seed": seed, "status": "DEFERRED_RUNTIME_BUDGET", "target_total_timesteps": target, "checkpoint": "", "completed_target": False}

    env = make_env(seed)
    set_random_seed(seed)
    torch.manual_seed(seed)
    if out_ckpt.exists():
        model = MaskablePPO.load(out_ckpt, env=env)
        source = str(out_ckpt)
        source_mode = "existing_K2_checkpoint"
    elif cfg.get("source"):
        model = MaskablePPO.load(cfg["source"], env=env)
        source = cfg["source"]
        source_mode = "continued_from_stageK_partial_checkpoint"
    else:
        model = MaskablePPO(
            "MlpPolicy",
            env,
            learning_rate=2e-4,
            n_steps=2048,
            batch_size=256,
            gamma=1.0,
            gae_lambda=1.0,
            clip_range=0.2,
            ent_coef=0.03,
            vf_coef=0.5,
            max_grad_norm=0.5,
            seed=seed,
            verbose=1,
            tensorboard_log=str(LOG_DIR / "tensorboard"),
        )
        source = ""
        source_mode = "fresh_training"

    before = int(model.num_timesteps)
    remaining = max(0, target - before)
    if remaining > 0:
        logger = EpisodeLogger(MONITOR_CSV)
        ckpt_cb = CheckpointCallback(save_freq=100000, save_path=str(CHECKPOINT_DIR), name_prefix=f"N40_seed{seed}_K2_checkpoint")
        model.learn(total_timesteps=remaining, callback=[logger, ckpt_cb], progress_bar=False, reset_num_timesteps=False)
    after = int(model.num_timesteps)
    model.save(out_ckpt)
    return {
        "seed": seed,
        "status": "COMPLETED_TARGET" if after >= target else "PARTIAL",
        "source_checkpoint": source,
        "source_mode": source_mode,
        "target_total_timesteps": target,
        "timesteps_before": before,
        "timesteps_after": after,
        "learn_timesteps_this_stage": max(0, after - before),
        "checkpoint": str(out_ckpt),
        "completed_target": after >= target,
        "parameter_count": parameter_count(model),
    }


def main() -> int:
    ensure_dirs()
    config = {
        "algorithm": "MaskablePPO",
        "policy": "MlpPolicy",
        "fixed_n": 40,
        "learning_rate": 2e-4,
        "n_steps": 2048,
        "batch_size": 256,
        "gamma": 1.0,
        "gae_lambda": 1.0,
        "ent_coef": 0.03,
        "clip_range": 0.2,
        "seed_targets": SEED_TARGETS,
        "note": "Stage K2 trains seed 20260624 to 500k total; extra seeds are deferred unless runtime budget is extended.",
    }
    CONFIG_JSON.write_text(json.dumps(config, indent=2), encoding="utf-8")
    rows = [train_seed(seed, cfg) for seed, cfg in SEED_TARGETS.items()]
    params = {str(r["seed"]): r.get("parameter_count") for r in rows if r.get("parameter_count")}
    PARAM_JSON.write_text(json.dumps(params, indent=2), encoding="utf-8")
    final_verdict = "PASS_STAGEK2_N40_COMPLETION_TRAINING_READY" if rows[0].get("completed_target") else "WARNING_STAGEK2_N40_COMPLETION_TRAINING_PARTIAL"
    if any(r["status"].startswith("DEFERRED") for r in rows[1:]):
        final_verdict = VERDICT if rows[0].get("completed_target") else final_verdict
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "verdict": final_verdict,
        "rows": rows,
        "config_path": str(CONFIG_JSON),
        "parameter_count_path": str(PARAM_JSON),
        "monitor_csv": str(MONITOR_CSV),
        "no_Abaqus": True,
        "no_ODB_opening": True,
        "no_solver": True,
        "no_CAE_INP_JNL": True,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if rows[0].get("completed_target") else 1


if __name__ == "__main__":
    raise SystemExit(main())
