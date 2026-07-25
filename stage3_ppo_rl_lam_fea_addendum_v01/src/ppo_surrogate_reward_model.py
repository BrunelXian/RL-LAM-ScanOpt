"""Loadable PPO terminal reward surrogate interface."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import joblib
import numpy as np

from surrogate_reward_features import order_to_features, validate_order


class PPOSurrogateRewardModel:
    """Fail-closed terminal reward emulator for completed scan orders."""

    def __init__(self, payload: dict, feature_schema: dict, target_schema: dict) -> None:
        self.payload = payload
        self.feature_schema = feature_schema
        self.target_schema = target_schema
        self.primary_model = payload["primary_model"]
        self.diagnostic_models = payload.get("diagnostic_models", {})
        self.primary_target = payload.get("primary_target", target_schema.get("primary_target"))
        self.diagnostic_targets = payload.get("diagnostic_targets", [])

    @classmethod
    def load(
        cls,
        model_path: str | Path,
        feature_schema_path: str | Path | None = None,
        target_schema_path: str | Path | None = None,
    ) -> "PPOSurrogateRewardModel":
        payload = joblib.load(model_path)
        feature_schema = payload.get("feature_schema")
        target_schema = payload.get("target_schema")
        if feature_schema_path is not None:
            import json

            feature_schema = json.loads(Path(feature_schema_path).read_text(encoding="utf-8"))
        if target_schema_path is not None:
            import json

            target_schema = json.loads(Path(target_schema_path).read_text(encoding="utf-8"))
        if feature_schema is None or target_schema is None:
            raise ValueError("Feature and target schemas are required.")
        return cls(payload=payload, feature_schema=feature_schema, target_schema=target_schema)

    def validate_input_order(self, n: int, order: Sequence[int]) -> bool:
        return validate_order(int(n), [int(item) for item in order])

    def predict_reward(self, n: int, order: Sequence[int]) -> float:
        if not self.validate_input_order(n, order):
            raise ValueError(f"Illegal scan order for n={n}: {order}")
        features = order_to_features(int(n), [int(item) for item in order]).reshape(1, -1)
        prediction = self.primary_model.predict(features)
        return float(np.asarray(prediction).reshape(-1)[0])

    def predict_diagnostics(self, n: int, order: Sequence[int]) -> dict[str, float]:
        if not self.validate_input_order(n, order):
            raise ValueError(f"Illegal scan order for n={n}: {order}")
        features = order_to_features(int(n), [int(item) for item in order]).reshape(1, -1)
        out: dict[str, float] = {}
        for target, model in self.diagnostic_models.items():
            out[target] = float(np.asarray(model.predict(features)).reshape(-1)[0])
        return out
