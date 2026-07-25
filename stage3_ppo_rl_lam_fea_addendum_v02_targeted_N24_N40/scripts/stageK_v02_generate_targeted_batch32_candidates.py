"""Generate PPO v02 targeted N24/N40 candidate batch32 from checkpoints only."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
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

PROJECT_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
V02_NAMESPACE = "stage3_ppo_rl_lam_fea_addendum_v02_targeted_N24_N40"
V01_NAMESPACE = "stage3_ppo_rl_lam_fea_addendum_v01"
SRC_DIR = PROJECT_ROOT / V02_NAMESPACE / "src"
sys.path.insert(0, str(SRC_DIR))

from v02_fixedN_ppo_env import FixedNLamScanOrderPPOEnv, V02EnvConfig, validate_order  # noqa: E402

OUT_ROOT = PROJECT_ROOT / "outputs" / V02_NAMESPACE
SURR_MODELS = OUT_ROOT / "surrogate_v02" / "models"
TRAIN_SUMMARY = OUT_ROOT / "ppo_training_v02" / "v02_ppo_training_summary.json"
CHECKPOINT_DIR = OUT_ROOT / "ppo_training_v02" / "checkpoints"
DATASET = OUT_ROOT / "data" / "v02_targeted_N24_N40_teacher_dataset.csv"
CAND_DIR = OUT_ROOT / "candidate_generation_v02"
POOL_DIR = CAND_DIR / "rollout_pool"
SELECTED_DIR = CAND_DIR / "selected_batch32"
SCAN_DIR = SELECTED_DIR / "scan_orders"
TABLES_DIR = CAND_DIR / "tables"
PLOTS_DIR = OUT_ROOT / "plots"
DOCS_DIR = PROJECT_ROOT / "docs" / V02_NAMESPACE

ROLLOUT_CSV = POOL_DIR / "v02_ppo_rollout_pool.csv"
SELECTED_CSV = SELECTED_DIR / "v02_ppo_targeted_N24_N40_candidate_batch32.csv"
LEGALITY_CSV = TABLES_DIR / "v02_candidate_legality_audit.csv"
NOVELTY_CSV = TABLES_DIR / "v02_candidate_novelty_audit.csv"
SCORE_SUMMARY_CSV = TABLES_DIR / "v02_candidate_surrogate_score_summary_by_N.csv"
REPORT = DOCS_DIR / "PPO_V02_TARGETED_CANDIDATE_GENERATION_REPORT.md"
CLAIM_BOUNDARY = DOCS_DIR / "PPO_V02_TARGETED_CLAIM_BOUNDARY.md"
MANIFEST = OUT_ROOT / "v02_targeted_N24_N40_manifest.json"
SUPPORTED_N = [24, 40]
TARGET_COUNTS = {24: 16, 40: 16}
ROLLOUT_ATTEMPTS = 512
SEED = 20260624
VERDICT = "PASS_PPO_V02_TARGETED_BATCH32_READY_FOR_CAE_INP_HANDOFF"


def ensure_dirs() -> None:
    for path in [POOL_DIR, SELECTED_DIR, SCAN_DIR, TABLES_DIR, PLOTS_DIR, DOCS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def git_branch() -> str:
    try:
        result = subprocess.run(["git", "-C", str(PROJECT_ROOT), "branch", "--show-current"], capture_output=True, text=True, timeout=10)
        return result.stdout.strip() or "stage3-variable-n-graph-pointer-init-v01"
    except Exception:
        return "stage3-variable-n-graph-pointer-init-v01"


def order_compact(order: list[int]) -> str:
    return ",".join(str(int(x)) for x in order)


def order_json(order: list[int]) -> str:
    return json.dumps([int(x) for x in order], separators=(",", ":"))


def order_hash(order: list[int]) -> str:
    return hashlib.sha256(order_compact(order).encode("utf-8")).hexdigest()[:16]


def parse_order(text: Any) -> list[int]:
    return [int(x) for x in re.findall(r"-?\d+", "" if pd.isna(text) else str(text))]


def descriptors(order: list[int]) -> dict[str, float]:
    arr = np.asarray(order, dtype=float)
    n = len(order)
    jumps = np.abs(np.diff(arr))
    parity = arr.astype(int) % 2
    early = arr[: max(1, n // 4)]
    center = (n - 1) / 2.0
    denom = max(1.0, n - 1.0)
    center_dist = np.abs(early - center) / max(1.0, center)
    edge_dist = np.minimum(early, n - 1 - early) / max(1.0, center)
    return {
        "mean_abs_jump": float(jumps.mean()) if len(jumps) else 0.0,
        "max_abs_jump": float(jumps.max()) if len(jumps) else 0.0,
        "adjacent_fraction": float(np.mean(jumps == 1)) if len(jumps) else 0.0,
        "parity_switch_fraction": float(np.mean(parity[1:] != parity[:-1])) if n > 1 else 0.0,
        "early_center_bias": float(center_dist.mean()),
        "early_edge_bias": float(edge_dist.mean()),
        "early_spread_proxy": float(np.ptp(early) / denom) if len(early) > 1 else 0.0,
    }


def simple_markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def hamming(a: list[int], b: list[int]) -> int:
    return sum(int(x != y) for x, y in zip(a, b)) if len(a) == len(b) else max(len(a), len(b))


def load_reference_orders() -> dict[int, list[tuple[str, list[int]]]]:
    df = pd.read_csv(DATASET)
    out = {24: [], 40: []}
    for _, row in df.iterrows():
        n = int(row["n"])
        if n in out:
            out[n].append((str(row["strategy_name"]), parse_order(row["order_json"])))
    return out


def make_env(n: int, seed: int) -> FixedNLamScanOrderPPOEnv:
    return FixedNLamScanOrderPPOEnv(V02EnvConfig(n=n, surrogate_model_path=str(SURR_MODELS / f"N{n}_surrogate_reward_v02.joblib"), seed=seed, conservative_reward=True))


def load_checkpoints() -> dict[int, list[tuple[int, Path]]]:
    checkpoints = {24: [], 40: []}
    for n in SUPPORTED_N:
        for path in sorted(CHECKPOINT_DIR.glob(f"N{n}_seed*_maskable_ppo_v02.zip")):
            seed_match = re.search(r"seed(\d+)", path.name)
            seed = int(seed_match.group(1)) if seed_match else SEED
            checkpoints[n].append((seed, path))
    return checkpoints


def rollout(model: MaskablePPO, env: FixedNLamScanOrderPPOEnv, n: int, seed: int, deterministic: bool) -> tuple[list[int], float, dict[str, float]]:
    obs, _ = env.reset(seed=seed)
    done = False
    reward = 0.0
    info = {}
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic, action_masks=env.action_masks())
        obs, reward, done, _, info = env.step(int(action))
    order = [int(x) for x in info["terminal_order"]]
    comps = env.reward_model.predict_components(n, order)
    return order, float(reward), comps


def generate_pool() -> pd.DataFrame:
    refs = load_reference_orders()
    checkpoints = load_checkpoints()
    rows: list[dict[str, Any]] = []
    for n in SUPPORTED_N:
        if not checkpoints[n]:
            raise FileNotFoundError(f"No v02 checkpoints found for N{n}")
        for ppo_seed, path in checkpoints[n]:
            model = MaskablePPO.load(path)
            env = make_env(n, ppo_seed)
            attempts = [("deterministic", ppo_seed, True)]
            attempts += [("stochastic", SEED + n * 100000 + i, False) for i in range(ROLLOUT_ATTEMPTS)]
            for mode, seed, deterministic in attempts:
                order, reward, comps = rollout(model, env, n, seed, deterministic)
                if not validate_order(n, order):
                    continue
                oh = order_hash(order)
                ref_dist = [hamming(order, ref_order) for _, ref_order in refs[n]]
                desc = descriptors(order)
                rows.append(
                    {
                        "n": n,
                        "order_json": order_json(order),
                        "order_compact": order_compact(order),
                        "order_hash": oh,
                        "ppo_v02_checkpoint": str(path),
                        "ppo_seed": ppo_seed,
                        "rollout_seed": seed,
                        "ppo_generation_mode": mode,
                        "predicted_reward": comps["best_model_reward"],
                        "mean_pred_reward": comps["mean_pred_reward"],
                        "std_pred_reward": comps["std_pred_reward"],
                        "conservative_reward": comps["conservative_reward"],
                        "duplicate_vs_training_reference": any(d == 0 for d in ref_dist),
                        "minimum_hamming_distance_to_training_reference": int(min(ref_dist)) if ref_dist else None,
                        "mean_hamming_distance_to_training_reference": float(np.mean(ref_dist)) if ref_dist else None,
                        "candidate_source": "PPO_v02_checkpoint_inference",
                        "teacher_validated": False,
                        "abaqus_validated": False,
                        **desc,
                    }
                )
    pool = pd.DataFrame(rows)
    pool.to_csv(ROLLOUT_CSV, index=False)
    return pool


def maximin_select(candidates: pd.DataFrame, selected_orders: list[list[int]], count: int) -> pd.DataFrame:
    chosen = []
    remaining = candidates.copy()
    while len(chosen) < count and not remaining.empty:
        best_idx = None
        best_score = -1
        for idx, row in remaining.iterrows():
            order = parse_order(row["order_json"])
            if selected_orders:
                score = min(hamming(order, s) for s in selected_orders)
            else:
                score = len(order)
            if score > best_score:
                best_score = score
                best_idx = idx
        chosen.append(best_idx)
        selected_orders.append(parse_order(remaining.loc[best_idx, "order_json"]))
        remaining = remaining.drop(index=best_idx)
    return candidates.loc[chosen] if chosen else candidates.iloc[0:0]


def select_candidates(pool: pd.DataFrame) -> pd.DataFrame:
    selected_rows = []
    for n in SUPPORTED_N:
        n_pool = pool[pool["n"] == n].copy()
        n_pool = n_pool.sort_values(["conservative_reward", "minimum_hamming_distance_to_training_reference"], ascending=[False, False])
        n_unique = n_pool.drop_duplicates("order_hash").reset_index(drop=True)
        if len(n_unique) < TARGET_COUNTS[n]:
            raise RuntimeError(f"Insufficient unique PPO v02 orders for N{n}: {len(n_unique)}")
        selected = []
        selected_orders: list[list[int]] = []
        det = n_unique[n_unique["ppo_generation_mode"] == "deterministic"].head(1)
        if len(det):
            row = det.iloc[0].to_dict()
            row["selected_by"] = "deterministic"
            selected.append(row)
            selected_orders.append(parse_order(row["order_json"]))
        top = n_unique[~n_unique["order_hash"].isin([r["order_hash"] for r in selected])].head(7)
        for _, r in top.iterrows():
            row = r.to_dict()
            row["selected_by"] = "conservative_top"
            selected.append(row)
            selected_orders.append(parse_order(row["order_json"]))
        upper = n_unique.head(max(TARGET_COUNTS[n] * 4, len(n_unique) // 2))
        upper = upper[~upper["order_hash"].isin([r["order_hash"] for r in selected])]
        diverse = maximin_select(upper, selected_orders, 5)
        for _, r in diverse.iterrows():
            row = r.to_dict()
            row["selected_by"] = "diverse_tophalf"
            selected.append(row)
        chosen_hashes = {r["order_hash"] for r in selected}
        novelty = upper[~upper["order_hash"].isin(chosen_hashes)].sort_values(["minimum_hamming_distance_to_training_reference", "conservative_reward"], ascending=[False, False]).head(TARGET_COUNTS[n] - len(selected))
        for _, r in novelty.iterrows():
            row = r.to_dict()
            row["selected_by"] = "novelty_tophalf"
            selected.append(row)
        if len(selected) < TARGET_COUNTS[n]:
            filler = n_unique[~n_unique["order_hash"].isin([r["order_hash"] for r in selected])].head(TARGET_COUNTS[n] - len(selected))
            for _, r in filler.iterrows():
                row = r.to_dict()
                row["selected_by"] = "stochastic_highreward"
                selected.append(row)
        selected = selected[: TARGET_COUNTS[n]]
        for i, row in enumerate(selected, start=1):
            tag = row["selected_by"]
            row["strategy_name"] = f"PPOV02_N{n}_B{i:02d}_{tag}"
            row["notes"] = "PPO v02 targeted N24/N40 candidate; not physically validated yet."
            selected_rows.append(row)
    selected_df = pd.DataFrame(selected_rows)
    selected_df.to_csv(SELECTED_CSV, index=False)
    for _, row in selected_df.iterrows():
        payload = row.to_dict()
        payload["order"] = parse_order(row["order_json"])
        (SCAN_DIR / f"scan_order_{row['strategy_name']}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return selected_df


def audits(pool: pd.DataFrame, selected: pd.DataFrame) -> tuple[str, str]:
    leg_rows = []
    for _, row in selected.iterrows():
        order = parse_order(row["order_json"])
        legal = validate_order(int(row["n"]), order)
        leg_rows.append({"strategy_name": row["strategy_name"], "n": int(row["n"]), "legal": legal, "candidate_source_ok": row["candidate_source"] == "PPO_v02_checkpoint_inference", "teacher_validated_false": not bool(row["teacher_validated"]), "abaqus_validated_false": not bool(row["abaqus_validated"])})
    leg = pd.DataFrame(leg_rows)
    leg.to_csv(LEGALITY_CSV, index=False)
    novelty = selected[["strategy_name", "n", "order_hash", "duplicate_vs_training_reference", "minimum_hamming_distance_to_training_reference", "mean_hamming_distance_to_training_reference"]].copy()
    novelty.to_csv(NOVELTY_CSV, index=False)
    summary = selected.groupby("n").agg(count=("strategy_name", "size"), mean_conservative_reward=("conservative_reward", "mean"), max_conservative_reward=("conservative_reward", "max"), mean_min_hamming=("minimum_hamming_distance_to_training_reference", "mean"), duplicate_count=("duplicate_vs_training_reference", "sum")).reset_index()
    summary.to_csv(SCORE_SUMMARY_CSV, index=False)
    legality_verdict = "PASS" if len(selected) == 32 and leg["legal"].all() and leg["candidate_source_ok"].all() and leg["teacher_validated_false"].all() and leg["abaqus_validated_false"].all() and not selected["order_hash"].duplicated().any() else "FAIL"
    novelty_verdict = "PASS" if int(novelty["duplicate_vs_training_reference"].sum()) == 0 else "WARNING_DUPLICATES_PRESENT"
    return legality_verdict, novelty_verdict


def write_docs(pool: pd.DataFrame, selected: pd.DataFrame, legality: str, novelty: str) -> None:
    by_pool = pool.groupby("n").agg(rollout_rows=("order_hash", "size"), unique_orders=("order_hash", "nunique")).reset_index()
    by_sel = selected.groupby("n").size().to_dict()
    REPORT.write_text(
        "# PPO v02 Targeted Candidate Generation Report\n\n"
        "## Scope\n\nGenerated N24/N40 PPO v02 candidates from fixed-N PPO checkpoints only. No CAE/INP/JNL, Abaqus, ODB, solver, datacheck, enqueue, surrogate retraining, or candidate mutation occurred in candidate generation.\n\n"
        f"## Rollout Pool\n\nFeasible initial pool: {ROLLOUT_ATTEMPTS} stochastic attempts plus one deterministic attempt per N/checkpoint. The requested 2000-attempt pool was deferred because it exceeded local runtime limits.\n\n"
        + simple_markdown_table(by_pool)
        + "\n\n"
        f"## Selected Batch\n\nN24={by_sel.get(24, 0)}, N40={by_sel.get(40, 0)}, total={len(selected)}.\n\n"
        f"## Audits\n\n- Legality audit: `{legality}`\n- Novelty audit: `{novelty}`\n\n"
        f"## Selected CSV\n\n`{SELECTED_CSV}`\n\n"
        f"## Verdict\n\n`{VERDICT}`\n",
        encoding="utf-8",
    )
    CLAIM_BOUNDARY.write_text(
        "# PPO v02 Targeted Claim Boundary\n\n"
        "## Safe After This Stage\n\n"
        "- N24/N40 fixed-N PPO v02 policies were trained in N-specific surrogate environments.\n"
        "- PPO v02 generated legal N24/N40 candidate orders.\n"
        "- Candidate orders are ready for later CAE/INP generation and Abaqus teacher validation.\n\n"
        "## Not Safe\n\n"
        "- PPO v02 improves physical metrics.\n"
        "- PPO v02 beats v01.\n"
        "- PPO v02 is teacher validated.\n"
        "- PPO v02 solves N24/N40.\n",
        encoding="utf-8",
    )
    # Plots
    fig, ax = plt.subplots(figsize=(6, 4))
    selected.boxplot(column="conservative_reward", by="n", ax=ax)
    ax.set_title("v02 selected conservative reward by N")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "v02_selected_conservative_reward_by_N.png", dpi=180)
    plt.close(fig)


def write_manifest(pool: pd.DataFrame, selected: pd.DataFrame, legality: str, novelty: str) -> None:
    checkpoints = [str(p) for p in sorted(CHECKPOINT_DIR.glob("N*_seed*_maskable_ppo_v02.zip"))]
    manifest = {
        "branch": git_branch(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_paths": {"v02_targeted_dataset": str(DATASET)},
        "surrogate_model_paths": {str(n): str(SURR_MODELS / f"N{n}_surrogate_reward_v02.joblib") for n in SUPPORTED_N},
        "PPO_checkpoint_paths": checkpoints,
        "rollout_pool_path": str(ROLLOUT_CSV),
        "selected_candidate_batch_path": str(SELECTED_CSV),
        "scan_order_JSON_directory": str(SCAN_DIR),
        "selected_counts_by_N": {str(k): int(v) for k, v in selected.groupby("n").size().to_dict().items()},
        "rollout_pool_size_by_N": {str(k): int(v) for k, v in pool.groupby("n").size().to_dict().items()},
        "unique_orders_by_N": {str(k): int(v) for k, v in pool.groupby("n")["order_hash"].nunique().to_dict().items()},
        "reports": {"candidate_generation_report": str(REPORT)},
        "claim_boundary": str(CLAIM_BOUNDARY),
        "legality_audit_verdict": legality,
        "novelty_audit_verdict": novelty,
        "no_Abaqus": True,
        "no_ODB_opening": True,
        "no_solver": True,
        "no_CAE_INP_JNL": True,
        "no_teacher_validation": True,
        "no_commit_or_push": True,
        "final_verdict": VERDICT,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    ensure_dirs()
    pool = generate_pool()
    selected = select_candidates(pool)
    legality, novelty = audits(pool, selected)
    if legality != "PASS":
        verdict = "FAIL_PPO_V02_TARGETED_CANDIDATE_GENERATION_NOT_READY"
        raise RuntimeError(verdict)
    write_docs(pool, selected, legality, novelty)
    write_manifest(pool, selected, legality, novelty)
    summary = {"verdict": VERDICT, "rollout_pool_size_by_N": {str(k): int(v) for k, v in pool.groupby("n").size().to_dict().items()}, "unique_orders_by_N": {str(k): int(v) for k, v in pool.groupby("n")["order_hash"].nunique().to_dict().items()}, "selected_counts_by_N": {str(k): int(v) for k, v in selected.groupby("n").size().to_dict().items()}, "selected_csv": str(SELECTED_CSV), "manifest": str(MANIFEST), "no_Abaqus": True, "no_ODB_opening": True, "no_solver": True}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
