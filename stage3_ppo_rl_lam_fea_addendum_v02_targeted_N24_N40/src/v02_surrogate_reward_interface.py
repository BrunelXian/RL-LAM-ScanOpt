"""PPO v02 N-specific surrogate reward interface."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np

V01_SRC = Path(r"E:\Projects\RL-LAM-ScanOpt\stage3_ppo_rl_lam_fea_addendum_v01\src")
if str(V01_SRC) not in sys.path:
    sys.path.insert(0, str(V01_SRC))

from surrogate_reward_features import order_to_features, validate_order  # noqa: E402


class V02SurrogateReward:
    """Load an N-specific v02 reward payload and predict terminal rewards."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.n = int(payload["n"])
        self.primary_target = payload.get("primary_target", "reward_lex_u2_peeq_surfacet")
        self.best_model = payload["best_model"]
        self.ensemble_models = payload.get("ensemble_models", [])
        self.use_conservative_default = bool(payload.get("use_conservative_default", True))

    @classmethod
    def load(cls, model_path: str | Path) -> "V02SurrogateReward":
        return cls(joblib.load(model_path))

    def validate_input_order(self, n: int, order: Sequence[int]) -> bool:
        return int(n) == self.n and validate_order(int(n), [int(x) for x in order])

    def predict_components(self, n: int, order: Sequence[int]) -> dict[str, float]:
        if not self.validate_input_order(n, order):
            raise ValueError(f"Illegal order for N{n}: {order}")
        x = order_to_features(int(n), [int(v) for v in order]).reshape(1, -1)
        best = float(np.asarray(self.best_model.predict(x)).reshape(-1)[0])
        if self.ensemble_models:
            preds = np.asarray([float(np.asarray(model.predict(x)).reshape(-1)[0]) for model in self.ensemble_models], dtype=float)
            mean = float(np.mean(preds))
            std = float(np.std(preds))
            conservative = mean - 0.5 * std
        else:
            mean = best
            std = 0.0
            conservative = best
        return {
            "best_model_reward": best,
            "mean_pred_reward": mean,
            "std_pred_reward": std,
            "conservative_reward": conservative,
        }

    def predict_reward(self, n: int, order: Sequence[int], conservative: bool | None = None) -> float:
        components = self.predict_components(n, order)
        use_conservative = self.use_conservative_default if conservative is None else bool(conservative)
        return float(components["conservative_reward"] if use_conservative else components["best_model_reward"])
