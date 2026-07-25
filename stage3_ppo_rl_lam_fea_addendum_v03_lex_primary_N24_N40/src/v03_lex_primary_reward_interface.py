"""PPO v03 lex-primary surrogate reward interface."""

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


class V03LexPrimaryReward:
    """Load an N-specific v03 surrogate payload and score full scan orders."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.n = int(payload["n"])
        self.target_models = payload["target_models"]
        self.classifier = payload.get("top25_classifier")
        self.final_formula = payload.get("final_formula", "0.6*lex + 0.3*u2_guarded + 0.1*record - penalty")

    @classmethod
    def load(cls, model_path: str | Path) -> "V03LexPrimaryReward":
        return cls(joblib.load(model_path))

    def validate_input_order(self, n: int, order: Sequence[int]) -> bool:
        return int(n) == self.n and validate_order(int(n), [int(x) for x in order])

    def _features(self, n: int, order: Sequence[int]) -> np.ndarray:
        if not self.validate_input_order(n, order):
            raise ValueError(f"Illegal order for N{n}: {order}")
        return order_to_features(int(n), [int(v) for v in order]).reshape(1, -1)

    def _target_pred(self, x: np.ndarray, target: str) -> dict[str, float]:
        entry = self.target_models[target]
        best = float(np.asarray(entry["best_model"].predict(x)).reshape(-1)[0])
        ensemble = entry.get("ensemble_models", [])
        if ensemble:
            preds = np.asarray([float(np.asarray(model.predict(x)).reshape(-1)[0]) for model in ensemble], dtype=float)
            mean = float(np.mean(preds))
            std = float(np.std(preds))
            conservative = mean - 0.5 * std
        else:
            mean = best
            std = 0.0
            conservative = best
        return {
            f"{target}_best": best,
            f"{target}_mean": mean,
            f"{target}_std": std,
            f"{target}_conservative": conservative,
        }

    def predict_components(self, n: int, order: Sequence[int]) -> dict[str, float]:
        x = self._features(n, order)
        lex = self._target_pred(x, "reward_lex_primary_v03")
        u2 = self._target_pred(x, "reward_u2_guarded_v03")
        record = self._target_pred(x, "reward_record_seeking_v03")
        lex_score = lex["reward_lex_primary_v03_conservative"]
        u2_score = u2["reward_u2_guarded_v03_conservative"]
        record_score = record["reward_record_seeking_v03_conservative"]
        top25_prob = np.nan
        if self.classifier is not None:
            if hasattr(self.classifier, "predict_proba"):
                proba = np.asarray(self.classifier.predict_proba(x))
                top25_prob = float(proba.reshape(proba.shape[0], -1)[0, -1])
            else:
                top25_prob = float(np.asarray(self.classifier.predict(x)).reshape(-1)[0])
        surface_only_penalty = 0.0
        # Penalize cases where the U2-guarded model is much less confident than lex.
        if lex_score > 0.70 and u2_score < 0.45:
            surface_only_penalty = 0.20
        final_v03_score = 0.6 * lex_score + 0.3 * u2_score + 0.1 * record_score - surface_only_penalty
        if not np.isnan(top25_prob):
            final_reward = 0.7 * final_v03_score + 0.3 * top25_prob
        else:
            final_reward = final_v03_score
        out: dict[str, float] = {}
        out.update(lex)
        out.update(u2)
        out.update(record)
        out.update(
            {
                "predicted_lex_primary_score": float(lex_score),
                "predicted_u2_guarded_score": float(u2_score),
                "predicted_record_seeking_score": float(record_score),
                "top25_probability": float(top25_prob) if not np.isnan(top25_prob) else np.nan,
                "surfaceT_only_false_positive_penalty": float(surface_only_penalty),
                "final_v03_score": float(final_v03_score),
                "terminal_reward": float(final_reward),
            }
        )
        return out

    def predict_reward(self, n: int, order: Sequence[int]) -> float:
        return float(self.predict_components(n, order)["terminal_reward"])
