"""Smoke-test the frozen PPO surrogate reward model interface."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from ppo_surrogate_reward_model import PPOSurrogateRewardModel  # noqa: E402
from surrogate_reward_features import parse_order  # noqa: E402


PROJECT_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
NAMESPACE = "stage3_ppo_rl_lam_fea_addendum_v01"
DATASET_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "stage3_run_78_final_evidence_freeze_package"
    / "FROZEN_stage3_native_combined552_teacher_dataset.csv"
)
MODEL_DIR = PROJECT_ROOT / "outputs" / NAMESPACE / "surrogate_reward_model" / "models"
TABLES_DIR = PROJECT_ROOT / "outputs" / NAMESPACE / "surrogate_reward_model" / "tables"
MODEL_PATH = MODEL_DIR / "ppo_surrogate_reward_model_best.joblib"
FEATURE_SCHEMA_PATH = MODEL_DIR / "ppo_surrogate_feature_schema.json"
TARGET_SCHEMA_PATH = MODEL_DIR / "ppo_surrogate_target_schema.json"
OUT_CSV = TABLES_DIR / "surrogate_reward_smoke_predictions.csv"
MANIFEST_PATH = PROJECT_ROOT / "outputs" / NAMESPACE / "surrogate_reward_model" / "ppo_surrogate_reward_model_manifest.json"


def raster_left_to_right(n: int) -> list[int]:
    return list(range(n))


def odd_even(n: int) -> list[int]:
    return list(range(0, n, 2)) + list(range(1, n, 2))


def center_out(n: int) -> list[int]:
    left = (n - 1) // 2
    right = left + 1
    order: list[int] = []
    while left >= 0 or right < n:
        if left >= 0:
            order.append(left)
            left -= 1
        if right < n:
            order.append(right)
            right += 1
    return order


def main() -> int:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    model = PPOSurrogateRewardModel.load(MODEL_PATH, FEATURE_SCHEMA_PATH, TARGET_SCHEMA_PATH)
    df = pd.read_csv(DATASET_PATH)
    rows: list[dict[str, object]] = []

    for n in [12, 16, 24, 40]:
        known = df[df["n"] == n].head(3)
        for _, row in known.iterrows():
            order = parse_order(row)
            record = {
                "source": "known_combined552",
                "n": n,
                "label": row.get("strategy_name", ""),
                "order": ",".join(str(item) for item in order),
                "predicted_reward": model.predict_reward(n, order),
            }
            record.update({f"predicted_{k}": v for k, v in model.predict_diagnostics(n, order).items()})
            rows.append(record)

        for label, fn in [
            ("raster_left_to_right", raster_left_to_right),
            ("odd_even", odd_even),
            ("center_out", center_out),
        ]:
            order = fn(n)
            record = {
                "source": "deterministic_baseline_smoke",
                "n": n,
                "label": label,
                "order": ",".join(str(item) for item in order),
                "predicted_reward": model.predict_reward(n, order),
            }
            record.update({f"predicted_{k}": v for k, v in model.predict_diagnostics(n, order).items()})
            rows.append(record)

    fieldnames = sorted({key for row in rows for key in row})
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        manifest["smoke_test"] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "PASS_SURROGATE_REWARD_MODEL_SMOKE",
            "script": str(Path(__file__).resolve()),
            "prediction_table": str(OUT_CSV),
            "rows": len(rows),
        }
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print({"smoke_verdict": "PASS_SURROGATE_REWARD_MODEL_SMOKE", "rows": len(rows), "output": str(OUT_CSV)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
