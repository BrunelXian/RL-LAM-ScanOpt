"""Reward target builder for PPO surrogate terminal reward models."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


METRIC_COLUMNS = {
    "u2": "u2_range",
    "peeq": "peeq_max",
    "surfacet": "surface_t_proxy",
    "mises": "mises_max",
}

TARGET_COLUMNS = [
    "reward_lex_u2_peeq_surfacet",
    "reward_u2_primary",
    "reward_constrained",
    "cost_u2_norm",
    "cost_peeq_norm",
    "cost_surfacet_norm",
    "cost_mises_norm",
    "reward_strict_penalty_guard_like",
]


def _within_n_minmax_cost(group: pd.Series) -> pd.Series:
    values = pd.to_numeric(group, errors="coerce")
    min_value = values.min()
    max_value = values.max()
    if not np.isfinite(min_value) or not np.isfinite(max_value) or max_value == min_value:
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    return (values - min_value) / (max_value - min_value)


def _within_n_rank_reward(group: pd.Series) -> pd.Series:
    values = pd.to_numeric(group, errors="coerce")
    count = int(values.notna().sum())
    if count <= 1:
        return pd.Series(np.ones(len(values), dtype=float), index=values.index)
    rank_smaller_better = values.rank(method="average", ascending=True)
    cost_rank = (rank_smaller_better - 1.0) / max(1.0, count - 1.0)
    return 1.0 - cost_rank


def add_reward_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Add leakage-safe within-N reward targets to a dataframe."""

    required = ["n", *METRIC_COLUMNS.values()]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for reward targets: {missing}")

    out = df.copy()
    out["n"] = pd.to_numeric(out["n"], errors="raise").astype(int)
    for metric_column in METRIC_COLUMNS.values():
        out[metric_column] = pd.to_numeric(out[metric_column], errors="coerce")

    for metric_name, metric_column in METRIC_COLUMNS.items():
        out[f"cost_{metric_name}_norm"] = out.groupby("n", group_keys=False)[metric_column].apply(_within_n_minmax_cost)
        out[f"reward_{metric_name}_rank"] = out.groupby("n", group_keys=False)[metric_column].apply(_within_n_rank_reward)

    out["reward_u2_primary"] = out["reward_u2_rank"]
    out["reward_lex_u2_peeq_surfacet"] = (
        1.0 * out["reward_u2_rank"]
        + 0.1 * out["reward_peeq_rank"]
        + 0.01 * out["reward_surfacet_rank"]
    )

    out["reward_constrained"] = (
        out["reward_u2_rank"]
        - 0.25 * out["cost_peeq_norm"]
        - 0.10 * out["cost_surfacet_norm"]
    )

    threshold_fields = _detect_threshold_fields(out.columns)
    if threshold_fields:
        out["reward_strict_penalty_guard_like"] = out["reward_constrained"]
        threshold_status = "FOUND_BUT_NOT_USED_FOR_PHYSICAL_THRESHOLDS"
    else:
        out["reward_strict_penalty_guard_like"] = out["reward_constrained"]
        threshold_status = "NOT_FOUND"

    schema = target_schema(threshold_status=threshold_status, threshold_fields=threshold_fields)
    return out, schema


def _detect_threshold_fields(columns: Iterable[str]) -> list[str]:
    threshold_tokens = ("threshold", "guard_limit", "allowable", "feasible_limit")
    found = []
    for column in columns:
        lowered = column.lower()
        if any(token in lowered for token in threshold_tokens):
            found.append(column)
    return sorted(found)


def target_schema(threshold_status: str = "NOT_EVALUATED", threshold_fields: list[str] | None = None) -> dict[str, object]:
    return {
        "schema_name": "ppo_surrogate_reward_targets_v01",
        "target_columns": TARGET_COLUMNS,
        "metric_columns": METRIC_COLUMNS,
        "normalization": "within-N only; no cross-N leakage",
        "smaller_is_better_metrics": list(METRIC_COLUMNS.values()),
        "primary_target": "reward_lex_u2_peeq_surfacet",
        "primary_reward_formula": "1.0*reward_u2_rank + 0.1*reward_peeq_rank + 0.01*reward_surfacet_rank",
        "reward_direction": "larger_is_better",
        "diagnostic_target": "cost_mises_norm",
        "mises_role": "diagnostic_only",
        "strict_threshold_status": threshold_status,
        "strict_threshold_fields": threshold_fields or [],
        "strict_penalty_note": (
            "No physical threshold is invented. If threshold fields are unavailable, "
            "reward_strict_penalty_guard_like mirrors a rank/minmax guard-like diagnostic target."
        ),
    }


def schema_markdown(schema: dict[str, object]) -> str:
    lines = [
        "# PPO Surrogate Target Schema",
        "",
        f"- Schema: `{schema['schema_name']}`",
        f"- Primary target: `{schema['primary_target']}`",
        f"- Formula: `{schema['primary_reward_formula']}`",
        f"- Normalization: `{schema['normalization']}`",
        f"- Reward direction: `{schema['reward_direction']}`",
        f"- Mises role: `{schema['mises_role']}`",
        f"- Strict threshold status: `{schema['strict_threshold_status']}`",
        "",
        "## Target Columns",
        "",
    ]
    for column in schema["target_columns"]:
        lines.append(f"- `{column}`")
    lines.extend(["", "## Metric Columns", ""])
    metric_columns = schema["metric_columns"]
    assert isinstance(metric_columns, dict)
    for name, column in metric_columns.items():
        lines.append(f"- `{name}`: `{column}`")
    return "\n".join(lines)
