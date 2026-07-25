"""Generate Stage D PPO-only candidate batch32 for later teacher validation.

This script performs PPO checkpoint inference only. It does not run Abaqus,
open ODB files, run solver/datacheck, enqueue jobs, generate CAE/INP/JNL, train
PPO, retrain a surrogate, mutate PPO orders, or claim teacher validation.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
from surrogate_reward_features import parse_order  # noqa: E402


PROJECT_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
BRANCH = "stage3-variable-n-graph-pointer-init-v01"
NAMESPACE = "stage3_ppo_rl_lam_fea_addendum_v01"
SUPPORTED_N = [12, 16, 24, 40]
SEED = 20260623

PPO_CHECKPOINT = PROJECT_ROOT / "outputs" / NAMESPACE / "ppo_training" / "checkpoints" / "maskable_ppo_lam_scan_order_final.zip"
SURROGATE_MODEL = PROJECT_ROOT / "outputs" / NAMESPACE / "surrogate_reward_model" / "models" / "ppo_surrogate_reward_model_best.joblib"
FEATURE_SCHEMA = PROJECT_ROOT / "outputs" / NAMESPACE / "surrogate_reward_model" / "models" / "ppo_surrogate_feature_schema.json"
TARGET_SCHEMA = PROJECT_ROOT / "outputs" / NAMESPACE / "surrogate_reward_model" / "models" / "ppo_surrogate_target_schema.json"
COMBINED552 = PROJECT_ROOT / "outputs" / "stage3_run_78_final_evidence_freeze_package" / "FROZEN_stage3_native_combined552_teacher_dataset.csv"

OUT_DIR = PROJECT_ROOT / "outputs" / NAMESPACE / "ppo_candidate_generation"
ROLLOUT_DIR = OUT_DIR / "rollout_pool"
SELECTED_DIR = OUT_DIR / "selected_batch32"
SCAN_ORDER_DIR = SELECTED_DIR / "scan_orders"
TABLES_DIR = OUT_DIR / "tables"
REPORTS_DIR = OUT_DIR / "reports"
PLOTS_DIR = OUT_DIR / "plots"
DOCS_DIR = PROJECT_ROOT / "docs" / NAMESPACE

ROLLOUT_CSV = ROLLOUT_DIR / "ppo_generated_rollout_pool.csv"
ROLLOUT_JSON = ROLLOUT_DIR / "ppo_generated_rollout_pool.json"
SELECTED_CSV = SELECTED_DIR / "ppo_policy_only_candidate_batch32.csv"
SELECTED_JSON = SELECTED_DIR / "ppo_policy_only_candidate_batch32.json"
HANDOFF_PREVIEW_CSV = SELECTED_DIR / "ppo_stageE_candidate_handoff_preview.csv"
LEGALITY_AUDIT_CSV = TABLES_DIR / "ppo_candidate_legality_audit.csv"
NOVELTY_AUDIT_CSV = TABLES_DIR / "ppo_candidate_novelty_audit.csv"
SCORE_SUMMARY_CSV = TABLES_DIR / "ppo_candidate_surrogate_score_summary_by_N.csv"
DESCRIPTOR_SUMMARY_CSV = TABLES_DIR / "ppo_candidate_order_descriptor_summary.csv"
REPORT_PATH = DOCS_DIR / "PPO_POLICY_ONLY_CANDIDATE_GENERATION_STAGE_D_REPORT.md"
CLAIM_BOUNDARY_PATH = DOCS_DIR / "PPO_POLICY_ONLY_CANDIDATE_CLAIM_BOUNDARY.md"
MANIFEST_PATH = OUT_DIR / "ppo_policy_only_candidate_generation_stage_d_manifest.json"


def ensure_dirs() -> None:
    for directory in [ROLLOUT_DIR, SELECTED_DIR, SCAN_ORDER_DIR, TABLES_DIR, REPORTS_DIR, PLOTS_DIR, DOCS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def order_json(order: list[int]) -> str:
    return json.dumps([int(x) for x in order], separators=(",", ":"))


def order_compact(order: list[int]) -> str:
    return ",".join(str(int(x)) for x in order)


def order_hash(n: int, order: list[int]) -> str:
    payload = f"N{int(n)}:{order_compact(order)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def env_config_for_n(n: int, seed: int) -> PPOSurrogateEnvConfig:
    return PPOSurrogateEnvConfig(
        model_path=str(SURROGATE_MODEL),
        feature_schema_path=str(FEATURE_SCHEMA),
        target_schema_path=str(TARGET_SCHEMA),
        fixed_n=int(n),
        random_n=False,
        supported_n=tuple(SUPPORTED_N),
        n_sampling_mode="balanced",
        reward_clip_min=-1.0,
        reward_clip_max=1.2,
        reward_scale=1.0,
        illegal_action_penalty=-100.0,
        seed=int(seed),
    )


def order_descriptors(order: list[int], n: int) -> dict[str, float]:
    arr = np.asarray(order, dtype=float)
    jumps = np.abs(np.diff(arr))
    parity = arr.astype(int) % 2
    center = (n - 1) / 2.0
    center_denom = max(1.0, center)
    early = arr[: max(1, n // 4)]
    early_center_bias = float(np.mean(np.abs(early - center) / center_denom))
    early_edge_bias = float(np.mean(np.minimum(early, n - 1 - early) / center_denom))
    return {
        "mean_abs_jump": float(np.mean(jumps)) if len(jumps) else 0.0,
        "max_abs_jump": float(np.max(jumps)) if len(jumps) else 0.0,
        "adjacent_jump_fraction": float(np.mean(jumps == 1)) if len(jumps) else 0.0,
        "parity_switch_fraction": float(np.mean(parity[1:] != parity[:-1])) if len(parity) > 1 else 0.0,
        "early_center_bias": early_center_bias,
        "early_edge_bias": early_edge_bias,
    }


def load_combined552_orders() -> dict[int, dict[str, Any]]:
    df = pd.read_csv(COMBINED552)
    out: dict[int, dict[str, Any]] = {}
    for n, group in df.groupby("n"):
        orders = []
        hashes = set()
        for _, row in group.iterrows():
            order = parse_order(row)
            h = order_hash(int(n), order)
            hashes.add(h)
            orders.append(np.asarray(order, dtype=int))
        out[int(n)] = {"hashes": hashes, "orders": orders}
    return out


def novelty_for_order(n: int, order: list[int], combined: dict[int, dict[str, Any]]) -> dict[str, Any]:
    h = order_hash(n, order)
    same_n = combined[int(n)]
    arr = np.asarray(order, dtype=int)
    distances = [int(np.sum(arr != known)) for known in same_n["orders"]]
    return {
        "duplicate_order_hash_in_combined552": bool(h in same_n["hashes"]),
        "minimum_hamming_distance_to_sameN_combined552": int(min(distances)) if distances else None,
        "mean_hamming_distance_to_sameN_combined552": float(np.mean(distances)) if distances else None,
        "maximum_hamming_distance_to_sameN_combined552": int(max(distances)) if distances else None,
    }


def rollout_one(model: MaskablePPO, n: int, generation_mode: str, seed: int) -> dict[str, Any] | None:
    deterministic = generation_mode == "deterministic"
    env = LamScanOrderSurrogateRewardEnv.from_config(env_config_for_n(n, seed))
    obs, _ = env.reset(seed=seed, options={"n": n})
    while True:
        mask = env.action_masks()
        action, _ = model.predict(obs, deterministic=deterministic, action_masks=mask)
        obs, reward, terminated, truncated, info = env.step(int(action))
        if terminated or truncated:
            order = info.get("terminal_order") or env.terminal_order() or []
            validation = validate_scan_order(order, n)
            if not validation.legal:
                return None
            diagnostics = env.surrogate_model.predict_diagnostics(n, order)
            row = {
                "n": int(n),
                "order_json": order_json(order),
                "order_compact": order_compact(order),
                "order_hash": order_hash(n, order),
                "generation_mode": generation_mode,
                "seed": int(seed),
                "attempt_kind": generation_mode,
                "predicted_surrogate_reward_lex": float(env.surrogate_model.predict_reward(n, order)),
                "candidate_source": "PPO_checkpoint_inference",
                "ppo_checkpoint": str(PPO_CHECKPOINT),
                "teacher_validated": False,
                "abaqus_validated": False,
                "legal_order": True,
                "legality": validation.reason,
            }
            row.update({f"predicted_{key}": float(value) for key, value in diagnostics.items()})
            row.update(order_descriptors(order, n))
            return row


def generate_rollout_pool() -> pd.DataFrame:
    model = MaskablePPO.load(PPO_CHECKPOINT)
    rows: list[dict[str, Any]] = []
    for n in SUPPORTED_N:
        for attempt in range(16):
            row = rollout_one(model, n, "deterministic", SEED + n * 10000 + attempt)
            if row is not None:
                row["attempt_index"] = attempt
                rows.append(row)
        for attempt in range(512):
            row = rollout_one(model, n, "stochastic", SEED + n * 100000 + attempt)
            if row is not None:
                row["attempt_index"] = attempt
                rows.append(row)
    pool = pd.DataFrame(rows)
    if pool.empty:
        raise RuntimeError("PPO rollout pool is empty.")
    return pool


def add_novelty(pool: pd.DataFrame, combined: dict[int, dict[str, Any]]) -> pd.DataFrame:
    records = []
    for row in pool.to_dict(orient="records"):
        order = [int(x) for x in str(row["order_compact"]).split(",") if str(x) != ""]
        row.update(novelty_for_order(int(row["n"]), order, combined))
        records.append(row)
    return pd.DataFrame(records)


def hamming(order_a: str, order_b: str) -> int:
    a = [int(x) for x in order_a.split(",")]
    b = [int(x) for x in order_b.split(",")]
    return int(sum(x != y for x, y in zip(a, b)))


def unique_pool_for_n(pool: pd.DataFrame, n: int) -> pd.DataFrame:
    subset = pool[pool["n"] == n].copy()
    subset = subset.sort_values(
        ["predicted_surrogate_reward_lex", "generation_mode", "minimum_hamming_distance_to_sameN_combined552", "seed"],
        ascending=[False, True, False, True],
    )
    unique = subset.drop_duplicates("order_hash", keep="first").reset_index(drop=True)
    return unique


def add_selected(selected: list[dict[str, Any]], row: pd.Series, tag: str) -> None:
    record = row.to_dict()
    record["selection_tag"] = tag
    selected.append(record)


def choose_diverse(candidates: pd.DataFrame, selected: list[dict[str, Any]]) -> pd.Series | None:
    used_hashes = {item["order_hash"] for item in selected}
    remaining = candidates[~candidates["order_hash"].isin(used_hashes)]
    if remaining.empty:
        return None
    if not selected:
        return remaining.iloc[0]
    scored = []
    selected_orders = [item["order_compact"] for item in selected]
    for _, row in remaining.iterrows():
        min_dist = min(hamming(str(row["order_compact"]), selected_order) for selected_order in selected_orders)
        scored.append((min_dist, float(row["predicted_surrogate_reward_lex"]), int(row["seed"]), row))
    scored.sort(key=lambda item: (item[0], item[1], -item[2]), reverse=True)
    return scored[0][3]


def select_batch(pool: pd.DataFrame) -> pd.DataFrame:
    selected_all: list[dict[str, Any]] = []
    for n in SUPPORTED_N:
        unique = unique_pool_for_n(pool, n)
        if len(unique) < 8:
            raise RuntimeError(f"Insufficient unique PPO-generated orders for N{n}: {len(unique)} < 8")
        selected: list[dict[str, Any]] = []

        deterministic = unique[unique["generation_mode"] == "deterministic"]
        if not deterministic.empty:
            add_selected(selected, deterministic.iloc[0], "deterministic")
        else:
            add_selected(selected, unique.iloc[0], "stochastic_highreward")

        top_candidates = unique[~unique["order_hash"].isin({item["order_hash"] for item in selected})]
        for _, row in top_candidates.head(3).iterrows():
            if len(selected) >= 4:
                break
            add_selected(selected, row, "surrogate_top")

        upper_half = unique.head(max(8, len(unique) // 2)).copy()
        for _ in range(2):
            row = choose_diverse(upper_half, selected)
            if row is not None:
                add_selected(selected, row, "diverse_tophalf")

        novelty_candidates = upper_half[
            (~upper_half["order_hash"].isin({item["order_hash"] for item in selected}))
            & (~upper_half["duplicate_order_hash_in_combined552"])
        ].sort_values(
            ["minimum_hamming_distance_to_sameN_combined552", "predicted_surrogate_reward_lex"],
            ascending=[False, False],
        )
        if not novelty_candidates.empty:
            add_selected(selected, novelty_candidates.iloc[0], "novelty_tophalf")

        remaining = unique[~unique["order_hash"].isin({item["order_hash"] for item in selected})]
        for _, row in remaining.iterrows():
            if len(selected) >= 8:
                break
            add_selected(selected, row, "stochastic_highreward")

        if len(selected) != 8:
            raise RuntimeError(f"Selection failed for N{n}: selected {len(selected)}")

        for batch_index, record in enumerate(selected, start=1):
            record["batch_index"] = batch_index
            record["strategy_name"] = f"PPOV01_N{n}_B{batch_index:02d}_{record['selection_tag']}"
            selected_all.append(record)
    selected_df = pd.DataFrame(selected_all)
    return selected_df.sort_values(["n", "batch_index"]).reset_index(drop=True)


def write_selected_jsons(selected: pd.DataFrame) -> None:
    for row in selected.to_dict(orient="records"):
        order = [int(x) for x in row["order_compact"].split(",")]
        payload = {
            "strategy_name": row["strategy_name"],
            "n": int(row["n"]),
            "order": order,
            "order_json": order_json(order),
            "order_compact": row["order_compact"],
            "candidate_source": "PPO_checkpoint_inference",
            "ppo_checkpoint": str(PPO_CHECKPOINT),
            "predicted_surrogate_reward_lex": float(row["predicted_surrogate_reward_lex"]),
            "generation_mode": row["generation_mode"],
            "selection_tag": row["selection_tag"],
            "teacher_validated": False,
            "abaqus_validated": False,
            "notes": "PPO-generated candidate for later Abaqus teacher validation; not physically validated yet.",
        }
        path = SCAN_ORDER_DIR / f"scan_order_{row['strategy_name']}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_handoff_preview(selected: pd.DataFrame) -> None:
    handoff = selected[
        [
            "n",
            "strategy_name",
            "order_json",
            "order_compact",
            "candidate_source",
            "predicted_surrogate_reward_lex",
        ]
    ].copy()
    handoff["expected_future_case_dir"] = handoff["strategy_name"].map(lambda name: f"STAGE_E_PLACEHOLDER_CASE_DIR/{name}")
    handoff["expected_future_inp"] = handoff["strategy_name"].map(lambda name: f"STAGE_E_PLACEHOLDER_INP/{name}.inp")
    handoff["teacher_validation_status"] = "NOT_YET_VALIDATED"
    handoff["stageE_required"] = True
    handoff.to_csv(HANDOFF_PREVIEW_CSV, index=False)


def write_audits(pool: pd.DataFrame, selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    legality = selected[["n", "strategy_name", "order_hash", "legal_order", "legality", "candidate_source", "teacher_validated", "abaqus_validated"]].copy()
    legality["duplicate_order_hash_within_selected"] = legality["order_hash"].duplicated(keep=False)
    legality["no_N32"] = legality["n"] != 32
    legality["no_cae_inp_odb_path_created"] = True
    legality["order_mutated_after_ppo_inference"] = False
    legality.to_csv(LEGALITY_AUDIT_CSV, index=False)

    novelty = selected[
        [
            "n",
            "strategy_name",
            "order_hash",
            "duplicate_order_hash_in_combined552",
            "minimum_hamming_distance_to_sameN_combined552",
            "mean_hamming_distance_to_sameN_combined552",
            "maximum_hamming_distance_to_sameN_combined552",
        ]
    ].copy()
    novelty.to_csv(NOVELTY_AUDIT_CSV, index=False)

    score_summary = selected.groupby("n").agg(
        selected_count=("strategy_name", "size"),
        reward_mean=("predicted_surrogate_reward_lex", "mean"),
        reward_min=("predicted_surrogate_reward_lex", "min"),
        reward_max=("predicted_surrogate_reward_lex", "max"),
        rollout_pool_rows=("n", lambda s: int(len(pool[pool["n"] == int(s.iloc[0])]))),
        rollout_pool_unique_orders=("n", lambda s: int(pool[pool["n"] == int(s.iloc[0])]["order_hash"].nunique())),
    ).reset_index()
    score_summary.to_csv(SCORE_SUMMARY_CSV, index=False)

    descriptor_summary = selected.groupby("n").agg(
        selected_count=("strategy_name", "size"),
        mean_abs_jump_mean=("mean_abs_jump", "mean"),
        max_abs_jump_mean=("max_abs_jump", "mean"),
        adjacent_jump_fraction_mean=("adjacent_jump_fraction", "mean"),
        parity_switch_fraction_mean=("parity_switch_fraction", "mean"),
        early_center_bias_mean=("early_center_bias", "mean"),
        early_edge_bias_mean=("early_edge_bias", "mean"),
    ).reset_index()
    descriptor_summary.to_csv(DESCRIPTOR_SUMMARY_CSV, index=False)
    return legality, novelty, score_summary, descriptor_summary


def plot_outputs(pool: pd.DataFrame, selected: pd.DataFrame, novelty: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 4))
    selected.boxplot(column="predicted_surrogate_reward_lex", by="n")
    plt.suptitle("")
    plt.title("Selected PPO Candidate Surrogate Reward")
    plt.xlabel("N")
    plt.ylabel("Predicted surrogate reward")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "predicted_surrogate_reward_by_N_selected.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    for n in SUPPORTED_N:
        vals = pool[pool["n"] == n]["predicted_surrogate_reward_lex"]
        plt.hist(vals, bins=30, alpha=0.45, label=f"N{n}")
    plt.xlabel("Predicted surrogate reward")
    plt.ylabel("Rollout count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "rollout_pool_reward_distribution_by_N.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    x = np.arange(len(SUPPORTED_N))
    grouped = selected.groupby("n")
    plt.bar(x - 0.2, grouped["mean_abs_jump"].mean().reindex(SUPPORTED_N), width=0.4, label="mean abs jump")
    plt.bar(x + 0.2, grouped["max_abs_jump"].mean().reindex(SUPPORTED_N), width=0.4, label="max abs jump mean")
    plt.xticks(x, [str(n) for n in SUPPORTED_N])
    plt.xlabel("N")
    plt.ylabel("Jump descriptor")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "selected_candidate_jump_descriptors_by_N.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 4))
    novelty.groupby("n")["minimum_hamming_distance_to_sameN_combined552"].mean().reindex(SUPPORTED_N).plot(kind="bar")
    plt.xlabel("N")
    plt.ylabel("Mean minimum Hamming distance")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "selected_candidate_novelty_by_N.png", dpi=180)
    plt.close()


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            vals.append(f"{val:.6g}" if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_claim_boundary() -> None:
    CLAIM_BOUNDARY_PATH.write_text(
        "\n".join(
            [
                "# PPO Policy-Only Candidate Claim Boundary",
                "",
                "## Safe After Stage D",
                "",
                "- A trained MaskablePPO policy generated legal scan-order candidates.",
                "- Candidate orders were selected from PPO checkpoint inference only.",
                "- Candidate orders are ready for future Abaqus teacher validation handoff.",
                "- Surrogate scores are predictions only.",
                "",
                "## Not Safe After Stage D",
                "",
                "- PPO candidates are physically validated.",
                "- PPO improves U2/PEEQ/SurfaceT under Abaqus.",
                "- PPO outperforms teacher-validated baselines.",
                "- PPO is final physical optimiser.",
                "",
                "## Stage E Required",
                "",
                "- Generate CAE/INP from these 32 candidates.",
                "- Run Abaqus.",
                "- Extract U/PEEQ/S/NT11 teacher metrics.",
                "- Compare against combined552 and baselines.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_report(
    pool: pd.DataFrame,
    selected: pd.DataFrame,
    legality: pd.DataFrame,
    novelty: pd.DataFrame,
    score_summary: pd.DataFrame,
    descriptor_summary: pd.DataFrame,
    verdict: str,
) -> None:
    pool_summary = pool.groupby("n").agg(
        rollout_rows=("order_hash", "size"),
        unique_orders=("order_hash", "nunique"),
        deterministic_rows=("generation_mode", lambda s: int((s == "deterministic").sum())),
        stochastic_rows=("generation_mode", lambda s: int((s == "stochastic").sum())),
    ).reset_index()
    selected_counts = selected.groupby("n").size().reset_index(name="selected_count")
    lines = [
        "# PPO Policy-Only Candidate Generation Stage D Report",
        "",
        "## 1. Purpose",
        "",
        "Generate a clean 32-case PPO-only candidate batch for later Abaqus teacher validation. This stage generates candidate orders only.",
        "",
        "## 2. PPO Checkpoint Used",
        "",
        f"`{PPO_CHECKPOINT}`",
        "",
        "## 3. Surrogate Reward Model Used For Ranking Only",
        "",
        f"`{SURROGATE_MODEL}`",
        "",
        "## 4. Candidate-Source Boundary",
        "",
        "Every selected candidate has `candidate_source = PPO_checkpoint_inference`, `teacher_validated = false`, and `abaqus_validated = false`.",
        "",
        "## 5. Rollout Pool Generation Method",
        "",
        "For each N, the script ran at least 16 deterministic and 512 stochastic PPO checkpoint rollouts with action masks. Illegal orders were not admitted to the pool.",
        "",
        md_table(pool_summary),
        "",
        "## 6. Selection Rule",
        "",
        "For each N: include one deterministic candidate, add top surrogate-reward PPO-generated candidates, add diversity candidates from the upper half of the PPO-predicted reward pool by maximin Hamming distance, add a novelty-favored upper-half candidate when available, then fill remaining slots from high-reward stochastic PPO-generated orders. No order is repaired, mutated, or hand-designed.",
        "",
        "## 7. Selected Batch32 Summary",
        "",
        md_table(selected_counts),
        "",
        "## 8. Legality Audit",
        "",
        f"- All legal: `{bool(legality['legal_order'].all())}`",
        f"- Duplicate selected hashes: `{int(legality['duplicate_order_hash_within_selected'].sum())}`",
        "",
        "## 9. Novelty Audit Against Combined552",
        "",
        md_table(novelty.groupby("n").agg(
            selected_count=("strategy_name", "size"),
            duplicate_combined552_count=("duplicate_order_hash_in_combined552", "sum"),
            min_hamming_mean=("minimum_hamming_distance_to_sameN_combined552", "mean"),
            min_hamming_min=("minimum_hamming_distance_to_sameN_combined552", "min"),
        ).reset_index()),
        "",
        "## 10. Predicted Surrogate Reward Summary",
        "",
        md_table(score_summary),
        "",
        "## 11. Order Descriptor Summary",
        "",
        md_table(descriptor_summary),
        "",
        "## 12. Stage E Abaqus Validation Handoff Preview",
        "",
        f"`{HANDOFF_PREVIEW_CSV}`",
        "",
        "## 13. Limitations",
        "",
        "- Surrogate scores are predictions only.",
        "- Candidate orders are not physically validated yet.",
        "- Stage D does not generate CAE/INP/JNL files.",
        "- Stage D does not run Abaqus, solver, datacheck, enqueue, or ODB extraction.",
        "",
        "## 14. Claim Boundary",
        "",
        "Safe: a trained PPO policy generated legal scan-order candidates selected from PPO checkpoint inference only. Not safe: physical improvement or teacher-validation claims.",
        "",
        "## 15. Ready For Stage E CAE/INP Generation And Abaqus Teacher Validation",
        "",
        f"`{verdict.startswith('PASS')}`",
        "",
        "## 16. Verdict",
        "",
        f"`{verdict}`",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ensure_dirs()
    timestamp = datetime.now(timezone.utc).isoformat()
    combined = load_combined552_orders()
    pool = add_novelty(generate_rollout_pool(), combined)
    pool.to_csv(ROLLOUT_CSV, index=False)
    ROLLOUT_JSON.write_text(json.dumps(pool.to_dict(orient="records"), indent=2), encoding="utf-8")

    selected = select_batch(pool)
    selected.to_csv(SELECTED_CSV, index=False)
    SELECTED_JSON.write_text(json.dumps(selected.to_dict(orient="records"), indent=2), encoding="utf-8")
    write_selected_jsons(selected)
    write_handoff_preview(selected)
    legality, novelty, score_summary, descriptor_summary = write_audits(pool, selected)
    plot_outputs(pool, selected, novelty)

    total_selected = len(selected)
    selected_counts_by_n = {str(int(k)): int(v) for k, v in selected.groupby("n").size().sort_index().items()}
    pass_checks = {
        "total_selected_32": total_selected == 32,
        "exactly_8_per_n": all(selected_counts_by_n.get(str(n), 0) == 8 for n in SUPPORTED_N),
        "all_legal": bool(legality["legal_order"].all()),
        "no_duplicate_order_hash_within_selected": not bool(legality["duplicate_order_hash_within_selected"].any()),
        "all_from_ppo_rollout_pool": set(selected["order_hash"]).issubset(set(pool["order_hash"])),
        "candidate_source_ok": bool((selected["candidate_source"] == "PPO_checkpoint_inference").all()),
        "teacher_validated_false": not bool(selected["teacher_validated"].astype(bool).any()),
        "abaqus_validated_false": not bool(selected["abaqus_validated"].astype(bool).any()),
        "no_n32": not bool((selected["n"] == 32).any()),
    }
    verdict = (
        "PASS_PPO_POLICY_ONLY_BATCH32_READY_FOR_STAGE_E_TEACHER_VALIDATION_HANDOFF"
        if all(pass_checks.values())
        else "FAIL_PPO_POLICY_ONLY_CANDIDATE_GENERATION_NOT_READY"
    )

    write_claim_boundary()
    write_report(pool, selected, legality, novelty, score_summary, descriptor_summary, verdict)

    manifest = {
        "branch": BRANCH,
        "timestamp": timestamp,
        "ppo_checkpoint_path": str(PPO_CHECKPOINT),
        "surrogate_reward_model_path": str(SURROGATE_MODEL),
        "rollout_pool_paths": {"csv": str(ROLLOUT_CSV), "json": str(ROLLOUT_JSON)},
        "selected_batch_paths": {"csv": str(SELECTED_CSV), "json": str(SELECTED_JSON), "handoff_preview_csv": str(HANDOFF_PREVIEW_CSV)},
        "scan_order_json_directory": str(SCAN_ORDER_DIR),
        "audit_table_paths": {
            "legality": str(LEGALITY_AUDIT_CSV),
            "novelty": str(NOVELTY_AUDIT_CSV),
            "surrogate_score_summary_by_N": str(SCORE_SUMMARY_CSV),
            "order_descriptor_summary": str(DESCRIPTOR_SUMMARY_CSV),
        },
        "report_path": str(REPORT_PATH),
        "claim_boundary_path": str(CLAIM_BOUNDARY_PATH),
        "selected_counts_by_N": selected_counts_by_n,
        "rollout_pool_size_by_N": {str(int(k)): int(v) for k, v in pool.groupby("n").size().sort_index().items()},
        "unique_ppo_generated_orders_by_N": {str(int(k)): int(v) for k, v in pool.groupby("n")["order_hash"].nunique().sort_index().items()},
        "pass_checks": pass_checks,
        "no_Abaqus": True,
        "no_ODB": True,
        "no_solver": True,
        "no_CAE_INP_JNL": True,
        "no_teacher_validation": True,
        "no_commit_or_push": True,
        "verdict": verdict,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "selected_counts_by_N": selected_counts_by_n, "pass_checks": pass_checks}, indent=2))
    return 0 if verdict.startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
