"""Train a Maskable PPO policy for the scan-planning environment."""

from __future__ import annotations

import argparse
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
DEFAULT_TIMESTEPS = 200_000
SEED = 42
N_STEPS = 2048
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
GAMMA = 0.99
ENT_COEF = 0.0
DEFAULT_MODEL_PATH = PROJECT_ROOT / "assets" / "models" / "maskable_ppo_twi_stripe.zip"
DEFAULT_TRAINING_CURVES_PATH = PROJECT_ROOT / "assets" / "figures" / "training_curves_maskable_ppo.png"
DEFAULT_TRAINING_HISTORY_PATH = PROJECT_ROOT / "assets" / "models" / "training_history_maskable_ppo_twi.json"


def _mask_fn(env: object) -> object:
    return env.unwrapped.action_masks()


def _make_single_env() -> object:
    """Create one wrapped environment instance for DummyVecEnv."""
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.monitor import Monitor

    planning_mask = build_twi_mask(grid_size=GRID_SIZE, canvas_size=CANVAS_SIZE, text=TEXT)
    env = ScanPlanningEnv(
        planning_mask=planning_mask,
        grid_size=GRID_SIZE,
        text=TEXT,
        canvas_size=CANVAS_SIZE,
    )
    env = Monitor(env)
    return ActionMasker(env, _mask_fn)


def make_env() -> object:
    """Create a single-environment DummyVecEnv for Maskable PPO training."""
    from stable_baselines3.common.vec_env import DummyVecEnv

    return DummyVecEnv([_make_single_env])


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
    DEFAULT_TRAINING_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DEFAULT_TRAINING_HISTORY_PATH.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for stripe-based PPO training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--log-interval", type=int, default=1000)
    return parser.parse_args()


def _resolve_output_paths(save_path: Path) -> tuple[Path, Path]:
    """Build history and curve output paths from the requested model path."""
    history_path = save_path.with_name(f"{save_path.stem}_history.json")
    curves_path = PROJECT_ROOT / "assets" / "figures" / f"{save_path.stem}_training_curves.png"
    return history_path, curves_path


def main() -> None:
    """Train and save a Maskable PPO policy."""
    args = parse_args()
    if args.timesteps <= 0:
        raise ValueError("--timesteps must be positive")
    if args.log_interval <= 0:
        raise ValueError("--log-interval must be positive")

    try:
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.callbacks import ConvertCallback
    except ImportError as exc:
        raise ImportError(
            "Maskable PPO dependencies are missing. Install torch and sb3-contrib first."
        ) from exc

    model_path = Path(args.save_path)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    training_history_path, training_curves_path = _resolve_output_paths(model_path)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    training_curves_path.parent.mkdir(parents=True, exist_ok=True)
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
        ent_coef=ENT_COEF,
        device=args.device,
    )

    print(f"Training Maskable PPO for {args.timesteps} timesteps...")
    model.learn(
        total_timesteps=args.timesteps,
        progress_bar=False,
        log_interval=args.log_interval,
        callback=ConvertCallback(metrics_callback),
    )
    model.save(str(model_path))
    with training_history_path.open("w", encoding="utf-8") as file:
        json.dump(metrics_callback.history, file, indent=2)
    save_training_curves_figure(metrics_callback.history, training_curves_path)
    print(f"Saved model to: {model_path}")
    print(f"Saved training history to: {training_history_path}")
    print(f"Saved training curves to: {training_curves_path}")
    print(f"Training complete: timesteps={args.timesteps}, model saved to {model_path}")

    env.close()


if __name__ == "__main__":
    main()
