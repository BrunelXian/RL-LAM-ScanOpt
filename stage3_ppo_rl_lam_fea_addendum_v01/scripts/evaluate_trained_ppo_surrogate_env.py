"""Internal surrogate-environment evaluation for Stage C PPO policy.

This script does not create final Abaqus validation candidates. The generated
orders are internal surrogate-environment evaluation telemetry only.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from ppo_scan_order_env import validate_scan_order  # noqa: E402
from ppo_surrogate_env_wrapper import PPOSurrogateEnvConfig, LamScanOrderSurrogateRewardEnv  # noqa: E402


PROJECT_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
BRANCH = "stage3-variable-n-graph-pointer-init-v01"
NAMESPACE = "stage3_ppo_rl_lam_fea_addendum_v01"
TRAIN_DIR = PROJECT_ROOT / "outputs" / NAMESPACE / "ppo_training"
CHECKPOINT_DIR = TRAIN_DIR / "checkpoints"
LOG_DIR = TRAIN_DIR / "logs"
TABLES_DIR = TRAIN_DIR / "tables"
REPORTS_DIR = TRAIN_DIR / "reports"
PLOTS_DIR = TRAIN_DIR / "plots"
DOCS_DIR = PROJECT_ROOT / "docs" / NAMESPACE
CONFIG_PATH = TRAIN_DIR / "ppo_training_config.json"
TRAINING_SUMMARY_PATH = REPORTS_DIR / "ppo_training_summary.json"
PARAMETER_COUNT_PATH = REPORTS_DIR / "ppo_parameter_count.json"
CHECKPOINT_PATH = CHECKPOINT_DIR / "maskable_ppo_lam_scan_order_final.zip"
ORDERS_CSV = TABLES_DIR / "ppo_internal_eval_orders.csv"
SUMMARY_BY_N_CSV = TABLES_DIR / "ppo_internal_eval_summary_by_N.csv"
SUMMARY_JSON = REPORTS_DIR / "ppo_internal_eval_summary.json"
BASELINE_CSV = TABLES_DIR / "ppo_internal_eval_baseline_comparison.csv"
LEGALITY_CSV = TABLES_DIR / "ppo_internal_eval_legality_audit.csv"
REPORT_PATH = DOCS_DIR / "PPO_TRAINING_STAGE_C_REPORT.md"
CLAIM_BOUNDARY_PATH = DOCS_DIR / "PPO_TRAINING_CLAIM_BOUNDARY.md"
MANIFEST_PATH = TRAIN_DIR / "ppo_training_stage_c_manifest.json"
SUPPORTED_N = [12, 16, 24, 40]


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def env_config_for_n(config: dict[str, Any], n: int, seed: int) -> PPOSurrogateEnvConfig:
    return PPOSurrogateEnvConfig(
        model_path=config["surrogate_model_path"],
        feature_schema_path=config["feature_schema_path"],
        target_schema_path=config["target_schema_path"],
        fixed_n=int(n),
        random_n=False,
        supported_n=tuple(int(value) for value in config["supported_n"]),
        n_sampling_mode=config.get("n_sampling_mode", "balanced"),
        reward_clip_min=config.get("reward_clip_min"),
        reward_clip_max=config.get("reward_clip_max"),
        reward_scale=float(config.get("reward_scale", 1.0)),
        illegal_action_penalty=float(config.get("illegal_action_penalty", -100.0)),
        seed=int(seed),
    )


def order_descriptors(order: list[int], n: int) -> dict[str, float]:
    arr = np.asarray(order, dtype=float)
    jumps = np.abs(np.diff(arr))
    parity = arr.astype(int) % 2
    center = (n - 1) / 2.0
    center_denom = max(1.0, center)
    early = arr[: max(1, n // 4)]
    early_center = np.mean(np.abs(early - center) / center_denom)
    early_edge = np.mean(np.minimum(early, n - 1 - early) / center_denom)
    return {
        "mean_jump": float(np.mean(jumps)) if len(jumps) else 0.0,
        "max_jump": float(np.max(jumps)) if len(jumps) else 0.0,
        "adjacent_fraction": float(np.mean(jumps == 1)) if len(jumps) else 0.0,
        "parity_switch_fraction": float(np.mean(parity[1:] != parity[:-1])) if len(parity) > 1 else 0.0,
        "center_edge_early_bias": float(early_center - early_edge),
    }


def rollout_policy(model: MaskablePPO, config: dict[str, Any], n: int, deterministic: bool, seed: int) -> dict[str, Any]:
    env = LamScanOrderSurrogateRewardEnv.from_config(env_config_for_n(config, n, seed))
    obs, _ = env.reset(seed=seed, options={"n": n})
    total_reward = 0.0
    while True:
        mask = env.action_masks()
        action, _ = model.predict(obs, deterministic=deterministic, action_masks=mask)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += float(reward)
        if terminated or truncated:
            order = info.get("terminal_order") or env.terminal_order() or []
            validation = validate_scan_order(order, n)
            row = {
                "evaluation_scope": "internal_surrogate_environment_only_not_abq_candidate",
                "mode": "deterministic" if deterministic else "stochastic",
                "seed": int(seed),
                "n": int(n),
                "order": ",".join(str(item) for item in order),
                "episode_length": int(len(order)),
                "surrogate_reward": float(total_reward),
                "legal": bool(validation.legal),
                "legality": validation.reason,
            }
            row.update(order_descriptors(order, n) if order else {})
            return row


def raster(n: int) -> list[int]:
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


def edge_in(n: int) -> list[int]:
    order: list[int] = []
    left = 0
    right = n - 1
    while left <= right:
        order.append(left)
        if right != left:
            order.append(right)
        left += 1
        right -= 1
    return order


def regular_jump(n: int) -> list[int]:
    step = max(1, n // 3)
    order: list[int] = []
    seen = set()
    current = 0
    while len(order) < n:
        if current not in seen:
            order.append(current)
            seen.add(current)
        current = (current + step) % n
        if current in seen and len(order) < n:
            for candidate in range(n):
                if candidate not in seen:
                    current = candidate
                    break
    return order


def baseline_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    baselines: dict[str, Callable[[int], list[int]]] = {
        "raster": raster,
        "odd_even": odd_even,
        "center_out": center_out,
        "edge_in": edge_in,
        "regular_jump": regular_jump,
    }
    rows = []
    for n in SUPPORTED_N:
        env = LamScanOrderSurrogateRewardEnv.from_config(env_config_for_n(config, n, int(config["seed"])))
        env.reset(options={"n": n})
        for name, fn in baselines.items():
            order = fn(n)
            reward = env.surrogate_model.predict_reward(n, order)
            if env.config.reward_clip_min is not None:
                reward = max(float(env.config.reward_clip_min), reward)
            if env.config.reward_clip_max is not None:
                reward = min(float(env.config.reward_clip_max), reward)
            validation = validate_scan_order(order, n)
            row = {
                "evaluation_scope": "surrogate_baseline_only_not_teacher_validation",
                "baseline": name,
                "n": int(n),
                "order": ",".join(str(item) for item in order),
                "surrogate_reward": float(reward),
                "legal": bool(validation.legal),
                "legality": validation.reason,
            }
            row.update(order_descriptors(order, n))
            rows.append(row)
    return rows


def summarize_eval(orders: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n, group in orders.groupby("n"):
        baseline_group = baselines[baselines["n"] == n]
        baseline_best = float(baseline_group["surrogate_reward"].max())
        baseline_mean = float(baseline_group["surrogate_reward"].mean())
        unique_orders = int(group["order"].nunique())
        rows.append(
            {
                "n": int(n),
                "eval_episodes": int(len(group)),
                "legal_orders": int(group["legal"].sum()),
                "illegal_orders": int((~group["legal"]).sum()),
                "mean_surrogate_reward": float(group["surrogate_reward"].mean()),
                "max_surrogate_reward": float(group["surrogate_reward"].max()),
                "min_surrogate_reward": float(group["surrogate_reward"].min()),
                "unique_orders": unique_orders,
                "duplicate_rate": float(1.0 - unique_orders / max(1, len(group))),
                "mean_jump": float(group["mean_jump"].mean()),
                "max_jump_mean": float(group["max_jump"].mean()),
                "adjacent_fraction_mean": float(group["adjacent_fraction"].mean()),
                "parity_switch_fraction_mean": float(group["parity_switch_fraction"].mean()),
                "center_edge_early_bias_mean": float(group["center_edge_early_bias"].mean()),
                "simple_baseline_best_surrogate_reward": baseline_best,
                "simple_baseline_mean_surrogate_reward": baseline_mean,
                "ppo_minus_baseline_best": float(group["surrogate_reward"].mean() - baseline_best),
                "ppo_minus_baseline_mean": float(group["surrogate_reward"].mean() - baseline_mean),
            }
        )
    return pd.DataFrame(rows).sort_values("n")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.6g}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def plot_training_curve() -> None:
    monitor_path = LOG_DIR / "ppo_training_monitor.csv"
    if not monitor_path.exists():
        return
    df = pd.read_csv(monitor_path)
    if df.empty:
        return
    plt.figure(figsize=(7, 4))
    rolling = df["terminal_reward"].rolling(window=50, min_periods=1).mean()
    plt.plot(df["num_timesteps"], df["terminal_reward"], alpha=0.25, label="episode")
    plt.plot(df["num_timesteps"], rolling, label="rolling mean 50")
    plt.xlabel("Timesteps")
    plt.ylabel("Terminal surrogate reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "training_reward_curve.png", dpi=180)
    plt.close()


def plot_eval(summary_by_n: pd.DataFrame, baselines: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 4))
    plt.bar(summary_by_n["n"].astype(str), summary_by_n["mean_surrogate_reward"])
    plt.xlabel("N")
    plt.ylabel("Mean PPO surrogate reward")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "surrogate_eval_reward_by_N.png", dpi=180)
    plt.close()

    baseline_best = baselines.groupby("n")["surrogate_reward"].max().reset_index()
    plt.figure(figsize=(6, 4))
    x = np.arange(len(summary_by_n))
    plt.bar(x - 0.18, summary_by_n["mean_surrogate_reward"], width=0.36, label="PPO eval mean")
    plt.bar(x + 0.18, baseline_best["surrogate_reward"], width=0.36, label="baseline best")
    plt.xticks(x, summary_by_n["n"].astype(str))
    plt.xlabel("N")
    plt.ylabel("Surrogate reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ppo_vs_baseline_surrogate_reward_by_N.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(summary_by_n["n"].astype(str), summary_by_n["unique_orders"])
    plt.xlabel("N")
    plt.ylabel("Unique internal eval orders")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ppo_order_uniqueness_by_N.png", dpi=180)
    plt.close()


def write_claim_boundary() -> None:
    CLAIM_BOUNDARY_PATH.write_text(
        "\n".join(
            [
                "# PPO Training Claim Boundary",
                "",
                "## Safe After Stage C",
                "",
                "- A MaskablePPO policy was trained in a surrogate terminal-reward environment derived from FEA teacher-labelled scan orders.",
                "- The trained PPO policy can generate legal scan-order permutations in internal surrogate-environment evaluation.",
                "- The model checkpoint, config, logs, and parameter count are frozen.",
                "",
                "## Not Safe After Stage C",
                "",
                "- PPO-generated candidates are Abaqus validated.",
                "- PPO improves physical U2/PEEQ/SurfaceT.",
                "- PPO outperforms teacher-validated baselines.",
                "- PPO is final physical optimiser.",
                "",
                "Those claims require Stage D candidate generation and Stage E Abaqus teacher validation.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_report(
    training_summary: dict[str, Any],
    parameter_count: dict[str, Any],
    eval_summary: dict[str, Any],
    summary_by_n: pd.DataFrame,
    baselines: pd.DataFrame,
    legality: pd.DataFrame,
    verdict: str,
) -> None:
    lines = [
        "# PPO Training Stage C Report",
        "",
        "## 1. Purpose",
        "",
        "Train a MaskablePPO policy in `LamScanOrderPPOEnv` using the frozen FEA-teacher-trained surrogate reward model as sparse terminal reward. This is not online Abaqus PPO.",
        "",
        "## 2. Inputs",
        "",
        f"- PPO config: `{CONFIG_PATH}`",
        f"- Surrogate reward model: `{training_summary['surrogate_model_path']}`",
        "",
        "## 3. Surrogate Reward Model Used",
        "",
        "The Stage B frozen `HistGradientBoostingRegressor` reward emulator was loaded through `PPOSurrogateRewardModel` and used only for terminal rewards.",
        "",
        "## 4. PPO Environment Definition",
        "",
        "- Environment: `LamScanOrderSurrogateRewardEnv` wrapping `LamScanOrderPPOEnv`",
        "- Supported N: `[12, 16, 24, 40]`",
        "- N sampling: balanced random N during training",
        "- Reward: sparse terminal surrogate reward",
        "- Intermediate valid rewards: 0",
        "",
        "## 5. Action Mask Verification",
        "",
        f"- Action mask audit: `{training_summary['action_mask_audit_csv']}`",
        f"- Action masks verified: `{training_summary['action_masks_verified']}`",
        "",
        "## 6. PPO Config",
        "",
        f"- Algorithm: `{training_summary['algorithm']}`",
        f"- Policy: `{training_summary['policy']}`",
        f"- Timesteps completed: `{training_summary['total_timesteps_completed']}`",
        f"- n_envs: `{training_summary['n_envs']}`",
        f"- Vectorization status: `{training_summary['vectorization_status']}`",
        "",
        "## 7. Training Status",
        "",
        f"- Episode count logged: `{training_summary['episode_count_logged']}`",
        f"- Mean terminal surrogate reward during training: `{training_summary['mean_terminal_reward']}`",
        f"- Training elapsed seconds: `{training_summary['training_elapsed_sec']}`",
        "",
        "## 8. Checkpoint Path",
        "",
        f"`{CHECKPOINT_PATH}`",
        "",
        "## 9. Parameter Count",
        "",
        f"- Total policy parameters: `{parameter_count['policy_parameter_count_total']}`",
        f"- Trainable policy parameters: `{parameter_count['policy_parameter_count_trainable']}`",
        "",
        "## 10. Internal Surrogate Evaluation Results",
        "",
        dataframe_to_markdown(summary_by_n),
        "",
        "## 11. Legality Audit",
        "",
        dataframe_to_markdown(legality),
        "",
        "## 12. Baseline Comparison In Surrogate Environment",
        "",
        "These comparisons are surrogate-environment only, not Abaqus teacher validation.",
        "",
        dataframe_to_markdown(baselines.groupby('n', as_index=False)['surrogate_reward'].agg(['mean', 'max']).reset_index()),
        "",
        "## 13. Limitations",
        "",
        "- Internal evaluation orders are not final Stage D candidate orders.",
        "- Rewards are surrogate predictions, not Abaqus teacher metrics.",
        "- No physical superiority claim can be made until Stage E teacher validation.",
        "- Vectorized action-mask training was deferred for the first audit run.",
        "",
        "## 14. Claim Boundary",
        "",
        "Safe claim: a MaskablePPO policy was trained in a surrogate reward environment derived from FEA teacher-labelled scan-order data.",
        "",
        "Not safe: teacher validation, physical improvement, global optimisation, or final optimiser claims.",
        "",
        "## 15. Ready For Stage D PPO-Only Candidate Generation",
        "",
        f"`{eval_summary['ready_for_stage_d']}`",
        "",
        "## 16. Verdict",
        "",
        f"`{verdict}`",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    for directory in [TABLES_DIR, REPORTS_DIR, PLOTS_DIR, DOCS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    config = load_config()
    model = MaskablePPO.load(CHECKPOINT_PATH)

    rows: list[dict[str, Any]] = []
    for n in SUPPORTED_N:
        for i in range(20):
            rows.append(rollout_policy(model, config, n, deterministic=True, seed=int(config["seed"]) + i))
        for seed_offset in range(5):
            for i in range(4):
                rows.append(rollout_policy(model, config, n, deterministic=False, seed=int(config["seed"]) + 1000 * seed_offset + i))

    orders = pd.DataFrame(rows)
    orders.to_csv(ORDERS_CSV, index=False)
    baselines = pd.DataFrame(baseline_rows(config))
    baselines.to_csv(BASELINE_CSV, index=False)
    summary_by_n = summarize_eval(orders, baselines)
    summary_by_n.to_csv(SUMMARY_BY_N_CSV, index=False)

    legality = orders.groupby("n").agg(
        eval_rows=("legal", "size"),
        legal_rows=("legal", "sum"),
        unique_orders=("order", "nunique"),
    ).reset_index()
    legality["illegal_rows"] = legality["eval_rows"] - legality["legal_rows"]
    legality["all_legal"] = legality["illegal_rows"] == 0
    legality.to_csv(LEGALITY_CSV, index=False)

    plot_training_curve()
    plot_eval(summary_by_n, baselines)

    training_summary = json.loads(TRAINING_SUMMARY_PATH.read_text(encoding="utf-8"))
    parameter_count = json.loads(PARAMETER_COUNT_PATH.read_text(encoding="utf-8"))
    checkpoint_exists = CHECKPOINT_PATH.exists()
    masks_verified = bool(training_summary.get("action_masks_verified"))
    training_completed = int(training_summary.get("total_timesteps_completed", 0)) >= int(config["total_timesteps"])
    all_legal = bool(legality["all_legal"].all())
    not_catastrophic = bool((summary_by_n["ppo_minus_baseline_mean"] > -0.25).all())

    if checkpoint_exists and masks_verified and training_completed and all_legal and not_catastrophic:
        verdict = "PASS_PPO_TRAINING_READY_FOR_POLICY_ONLY_CANDIDATE_GENERATION"
    elif checkpoint_exists and masks_verified and all_legal:
        verdict = "WARNING_PPO_TRAINING_PARTIAL_REVIEW_BEFORE_CANDIDATE_GENERATION"
    else:
        verdict = "FAIL_PPO_TRAINING_NOT_READY"

    eval_summary = {
        "branch": BRANCH,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "evaluation_scope": "internal_surrogate_environment_only_not_abq_candidate_generation",
        "orders_csv": str(ORDERS_CSV),
        "summary_by_N_csv": str(SUMMARY_BY_N_CSV),
        "baseline_comparison_csv": str(BASELINE_CSV),
        "legality_audit_csv": str(LEGALITY_CSV),
        "evaluation_rows": int(len(orders)),
        "all_orders_legal": all_legal,
        "ready_for_stage_d": verdict.startswith("PASS"),
        "surrogate_reward_summary_by_N": summary_by_n.to_dict(orient="records"),
        "verdict": verdict,
        "no_Abaqus": True,
        "no_ODB": True,
        "no_solver": True,
        "no_CAE_INP_JNL": True,
        "no_final_candidate_generation": True,
        "no_commit_or_push": True,
    }
    SUMMARY_JSON.write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")
    write_claim_boundary()
    write_report(training_summary, parameter_count, eval_summary, summary_by_n, baselines, legality, verdict)

    manifest = {
        "branch": BRANCH,
        "timestamp": eval_summary["timestamp"],
        "surrogate_model_path": config["surrogate_model_path"],
        "ppo_config_path": str(CONFIG_PATH),
        "ppo_checkpoint_path": str(CHECKPOINT_PATH),
        "training_log_paths": {
            "monitor_csv": str(LOG_DIR / "ppo_training_monitor.csv"),
            "training_summary_json": str(TRAINING_SUMMARY_PATH),
        },
        "evaluation_output_paths": {
            "orders_csv": str(ORDERS_CSV),
            "summary_by_N_csv": str(SUMMARY_BY_N_CSV),
            "summary_json": str(SUMMARY_JSON),
            "baseline_comparison_csv": str(BASELINE_CSV),
            "legality_audit_csv": str(LEGALITY_CSV),
        },
        "parameter_count_path": str(PARAMETER_COUNT_PATH),
        "report_path": str(REPORT_PATH),
        "claim_boundary_path": str(CLAIM_BOUNDARY_PATH),
        "no_Abaqus": True,
        "no_ODB": True,
        "no_solver": True,
        "no_CAE_INP_JNL": True,
        "no_final_candidate_generation": True,
        "no_commit_or_push": True,
        "verdict": verdict,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "summary": eval_summary}, indent=2))
    return 0 if not verdict.startswith("FAIL") else 1


if __name__ == "__main__":
    raise SystemExit(main())
