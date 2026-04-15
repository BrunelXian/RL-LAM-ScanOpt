"""Train a Maskable PPO policy for the scan-planning environment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.viz import save_training_curves_figure
from rl.env_scan import ScanPlanningEnv, build_twi_mask


TEXT = "TWI"
GRID_SIZE = 64
CANVAS_SIZE = 1024
TOTAL_TIMESTEPS = 50_000
SEED = 42
N_STEPS = 512
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
GAMMA = 0.99
MODEL_PATH = PROJECT_ROOT / "assets" / "models" / "maskable_ppo_twi.zip"
TRAINING_CURVES_PATH = PROJECT_ROOT / "assets" / "figures" / "training_curves_maskable_ppo.png"
TRAINING_HISTORY_PATH = PROJECT_ROOT / "assets" / "models" / "training_history_maskable_ppo_twi.json"


def _mask_fn(env: object) -> object:
    return env.unwrapped.action_masks()


def make_env() -> object:
    """Create the wrapped single-environment training instance."""
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.monitor import Monitor

    planning_mask = build_twi_mask(grid_size=GRID_SIZE, canvas_size=CANVAS_SIZE, text=TEXT)
    max_steps = max(int(planning_mask.sum()) * 2, 1)
    env = ScanPlanningEnv(
        planning_mask=planning_mask,
        grid_size=GRID_SIZE,
        text=TEXT,
        canvas_size=CANVAS_SIZE,
        max_steps=max_steps,
    )
    env = Monitor(env)
    return ActionMasker(env, _mask_fn)


class TrainingMetricsCallback:
    """Collect episode-level terminal metrics during training."""

    def __init__(self) -> None:
        self.history: dict[str, list[float]] = {
            "episode_reward": [],
            "coverage_ratio": [],
            "thermal_mean": [],
            "thermal_peak": [],
            "thermal_variance": [],
        }

    def __call__(self, locals_: dict, globals_: dict) -> bool:
        infos = locals_.get("infos", [])
        for info in infos:
            if "episode" not in info or "metrics" not in info:
                continue
            metrics = info["metrics"]
            self.history["episode_reward"].append(float(info["episode"]["r"]))
            self.history["coverage_ratio"].append(float(metrics["coverage_ratio"]))
            self.history["thermal_mean"].append(float(metrics["thermal_mean"]))
            self.history["thermal_peak"].append(float(metrics["thermal_peak"]))
            self.history["thermal_variance"].append(float(metrics["thermal_variance"]))
        return True


def _save_training_history(history: dict[str, list[float]]) -> None:
    TRAINING_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRAINING_HISTORY_PATH.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)


def main() -> None:
    """Train and save a Maskable PPO policy."""
    try:
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.callbacks import ConvertCallback
    except ImportError as exc:
        raise ImportError(
            "Maskable PPO dependencies are missing. Install torch and sb3-contrib first."
        ) from exc

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRAINING_CURVES_PATH.parent.mkdir(parents=True, exist_ok=True)
    env = make_env()
    metrics_callback = TrainingMetricsCallback()

    model = MaskablePPO(
        policy="CnnPolicy",
        env=env,
        verbose=1,
        seed=SEED,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        device="auto",
    )

    print(f"Training Maskable PPO for {TOTAL_TIMESTEPS} timesteps...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        progress_bar=False,
        callback=ConvertCallback(metrics_callback),
    )
    model.save(str(MODEL_PATH))
    _save_training_history(metrics_callback.history)
    save_training_curves_figure(metrics_callback.history, TRAINING_CURVES_PATH)
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved training history to: {TRAINING_HISTORY_PATH}")
    print(f"Saved training curves to: {TRAINING_CURVES_PATH}")

    env.close()


if __name__ == "__main__":
    main()
