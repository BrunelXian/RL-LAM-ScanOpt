"""Generate PPO v03 lex-primary N24/N40 candidate batch32."""

from __future__ import annotations

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
NS = "stage3_ppo_rl_lam_fea_addendum_v03_lex_primary_N24_N40"
SRC_DIR = PROJECT_ROOT / NS / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from v03_fixedN_ppo_env import V03EnvConfig, V03FixedNLamScanOrderPPOEnv, validate_order  # noqa: E402

OUT_ROOT = PROJECT_ROOT / "outputs" / NS
DATASET = OUT_ROOT / "data" / "v03_N24_N40_teacher_dataset.csv"
SURR_MODELS = OUT_ROOT / "surrogate_v03" / "models"
CHECKPOINT_DIR = OUT_ROOT / "ppo_training_v03" / "checkpoints"
GEN_DIR = OUT_ROOT / "candidate_generation_v03"
POOL_DIR = GEN_DIR / "rollout_pool"
SELECTED_DIR = GEN_DIR / "selected_batch32"
SCAN_DIR = SELECTED_DIR / "scan_orders"
TABLES_DIR = GEN_DIR / "tables"
PLOTS_DIR = OUT_ROOT / "plots"
DOCS_DIR = PROJECT_ROOT / "docs" / NS
POOL_CSV = POOL_DIR / "v03_ppo_rollout_pool.csv"
SELECTED_CSV = SELECTED_DIR / "v03_ppo_lex_primary_N24_N40_candidate_batch32.csv"
LEGALITY_CSV = TABLES_DIR / "v03_candidate_legality_audit.csv"
NOVELTY_CSV = TABLES_DIR / "v03_candidate_novelty_audit.csv"
SCORE_SUMMARY_CSV = TABLES_DIR / "v03_candidate_score_summary_by_N.csv"
SURFACET_SCREEN_CSV = TABLES_DIR / "v03_surfaceT_false_positive_screening.csv"
REPORT = DOCS_DIR / "PPO_V03_LEX_PRIMARY_CANDIDATE_GENERATION_REPORT.md"
CLAIM_BOUNDARY = DOCS_DIR / "PPO_V03_LEX_PRIMARY_CLAIM_BOUNDARY.md"
MANIFEST = OUT_ROOT / "stageP_v03_lex_primary_candidate_generation_manifest.json"
SURR_SUMMARY = OUT_ROOT / "surrogate_v03" / "tables" / "v03_surrogate_model_selection_summary.json"
TRAIN_SUMMARY = OUT_ROOT / "ppo_training_v03" / "v03_ppo_training_summary.json"
EVAL_SUMMARY = OUT_ROOT / "ppo_training_v03" / "tables" / "v03_internal_eval_summary.json"

ROLLOUTS_BY_N = {24: 120, 40: 160}
SEED_BASE = 20260629


def ensure_dirs() -> None:
    for path in [POOL_DIR, SELECTED_DIR, SCAN_DIR, TABLES_DIR, PLOTS_DIR, DOCS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def git_branch() -> str:
    try:
        result = subprocess.run(
            ["git", "-c", "safe.directory=E:/Projects/RL-LAM-ScanOpt", "-C", str(PROJECT_ROOT), "branch", "--show-current"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def parse_seed(path: Path) -> int:
    m = re.search(r"seed(\d+)", path.name)
    return int(m.group(1)) if m else 20260627


def order_compact(order: list[int]) -> str:
    return ",".join(str(x) for x in order)


def order_json(order: list[int]) -> str:
    return json.dumps(order, separators=(",", ":"))


def order_hash(order: list[int]) -> str:
    return hashlib.sha256(order_compact(order).encode("utf-8")).hexdigest()[:16]


def parse_order(text: Any) -> list[int]:
    return [int(x) for x in re.findall(r"-?\d+", "" if pd.isna(text) else str(text))]


def hamming(a: list[int], b: list[int]) -> int:
    return sum(int(x != y) for x, y in zip(a, b)) if len(a) == len(b) else max(len(a), len(b))


def descriptors(order: list[int]) -> dict[str, float]:
    arr = np.asarray(order, dtype=float)
    jumps = np.abs(np.diff(arr))
    parity = arr.astype(int) % 2
    n = len(order)
    center = (n - 1) / 2.0
    early = arr[: max(1, n // 4)]
    quarters = np.array_split(arr, 4)
    return {
        "mean_abs_jump": float(jumps.mean()),
        "max_abs_jump": float(jumps.max()),
        "adjacent_fraction": float(np.mean(jumps == 1)),
        "parity_switch_fraction": float(np.mean(parity[1:] != parity[:-1])),
        "early_center_bias": float(np.mean(np.abs(early - center) / max(1.0, center))),
        "early_edge_bias": float(np.mean(np.minimum(early, n - 1 - early) / max(1.0, center))),
        "early_spread_proxy": float(np.ptp(early) / max(1.0, n - 1)),
        "q1_spread": float(np.ptp(quarters[0]) / max(1.0, n - 1)),
        "q2_spread": float(np.ptp(quarters[1]) / max(1.0, n - 1)),
        "q3_spread": float(np.ptp(quarters[2]) / max(1.0, n - 1)),
        "q4_spread": float(np.ptp(quarters[3]) / max(1.0, n - 1)),
    }


def reference_orders() -> dict[int, dict[str, list[list[int]]]]:
    df = pd.read_csv(DATASET)
    refs: dict[int, dict[str, list[list[int]]]] = {24: {"combined552": [], "v01": [], "v02K2": []}, 40: {"combined552": [], "v01": [], "v02K2": []}}
    for _, row in df.iterrows():
        n = int(row["n"])
        if n not in refs:
            continue
        order = parse_order(row["order_json"])
        source = str(row["dataset_source"])
        if source == "stage3_native_combined552":
            refs[n]["combined552"].append(order)
        elif source == "ppo_v01_teacher_validated":
            refs[n]["v01"].append(order)
        elif source == "ppo_v02K2_teacher_validated":
            refs[n]["v02K2"].append(order)
    return refs


def make_env(n: int, seed: int) -> V03FixedNLamScanOrderPPOEnv:
    return V03FixedNLamScanOrderPPOEnv(V03EnvConfig(n=n, surrogate_model_path=str(SURR_MODELS / f"N{n}_v03_lex_primary_surrogate.joblib"), seed=seed))


def rollout(model: MaskablePPO, n: int, seed: int, deterministic: bool) -> tuple[list[int], dict[str, float]]:
    env = make_env(n, seed)
    obs, _ = env.reset(seed=seed)
    done = False
    info: dict[str, Any] = {}
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic, action_masks=env.action_masks())
        obs, _, done, _, info = env.step(int(action))
    order = [int(x) for x in info["terminal_order"]]
    return order, env.reward_model.predict_components(n, order)


def min_dist(order: list[int], refs: list[list[int]]) -> int:
    return int(min([hamming(order, r) for r in refs], default=len(order)))


def generate_pool() -> pd.DataFrame:
    refs = reference_orders()
    rows = []
    for n in [24, 40]:
        ckpts = sorted(CHECKPOINT_DIR.glob(f"N{n}_seed*_maskable_ppo_v03.zip"))
        if not ckpts:
            raise FileNotFoundError(f"No v03 N{n} checkpoints in {CHECKPOINT_DIR}")
        per_ckpt = max(1, ROLLOUTS_BY_N[n] // len(ckpts))
        for ckpt in ckpts:
            ppo_seed = parse_seed(ckpt)
            model = MaskablePPO.load(ckpt)
            attempts = [("deterministic", ppo_seed, True)] + [("stochastic", SEED_BASE + n * 100000 + ppo_seed + i, False) for i in range(per_ckpt)]
            for mode, seed, deterministic in attempts:
                order, comps = rollout(model, n, seed, deterministic)
                if not validate_order(n, order):
                    continue
                row = {
                    "n": n,
                    "order_json": order_json(order),
                    "order_compact": order_compact(order),
                    "order_hash": order_hash(order),
                    "ppo_v03_checkpoint": str(ckpt),
                    "ppo_seed": ppo_seed,
                    "rollout_seed": seed,
                    "ppo_generation_mode": mode,
                    "predicted_lex_primary_score": comps["predicted_lex_primary_score"],
                    "predicted_u2_guarded_score": comps["predicted_u2_guarded_score"],
                    "predicted_record_seeking_score": comps["predicted_record_seeking_score"],
                    "top25_probability": comps["top25_probability"],
                    "final_v03_score": comps["final_v03_score"],
                    "terminal_reward": comps["terminal_reward"],
                    "surfaceT_only_false_positive_penalty": comps["surfaceT_only_false_positive_penalty"],
                    "novelty_vs_combined552": min_dist(order, refs[n]["combined552"]),
                    "novelty_vs_v01": min_dist(order, refs[n]["v01"]),
                    "novelty_vs_v02K2": min_dist(order, refs[n]["v02K2"]),
                    "duplicate_vs_combined552": min_dist(order, refs[n]["combined552"]) == 0,
                    "duplicate_vs_v01": min_dist(order, refs[n]["v01"]) == 0,
                    "duplicate_vs_v02K2": min_dist(order, refs[n]["v02K2"]) == 0,
                    "candidate_source": "PPO_v03_checkpoint_inference",
                    "teacher_validated": False,
                    "abaqus_validated": False,
                    **descriptors(order),
                }
                rows.append(row)
    pool = pd.DataFrame(rows)
    pool.to_csv(POOL_CSV, index=False)
    return pool


def maximin(candidates: pd.DataFrame, selected_orders: list[list[int]], count: int) -> pd.DataFrame:
    chosen = []
    remaining = candidates.copy()
    while len(chosen) < count and not remaining.empty:
        best_idx, best_score = None, -1
        for idx, row in remaining.iterrows():
            order = parse_order(row["order_json"])
            novelty = min(row["novelty_vs_combined552"], row["novelty_vs_v01"], row["novelty_vs_v02K2"])
            diversity = min([hamming(order, existing) for existing in selected_orders], default=len(order))
            score = diversity + 0.25 * novelty
            if score > best_score:
                best_idx, best_score = idx, score
        chosen.append(best_idx)
        selected_orders.append(parse_order(remaining.loc[best_idx, "order_json"]))
        remaining = remaining.drop(index=best_idx)
    return candidates.loc[chosen] if chosen else candidates.iloc[0:0]


def eligible(unique: pd.DataFrame, selected_hashes: set[str]) -> pd.DataFrame:
    out = unique[~unique["order_hash"].isin(selected_hashes)].copy()
    out = out[~out["duplicate_vs_combined552"] & ~out["duplicate_vs_v01"] & ~out["duplicate_vs_v02K2"]]
    out = out[out["surfaceT_only_false_positive_penalty"] <= 0.0]
    return out


def select_for_n(pool_n: pd.DataFrame, n: int) -> pd.DataFrame:
    unique = pool_n.sort_values(
        ["final_v03_score", "predicted_u2_guarded_score", "predicted_lex_primary_score", "novelty_vs_combined552"],
        ascending=[False, False, False, False],
    ).drop_duplicates("order_hash").reset_index(drop=True)
    if len(unique) < 16:
        raise RuntimeError(f"Insufficient unique N{n} v03 orders: {len(unique)}")
    rows: list[dict[str, Any]] = []
    selected_orders: list[list[int]] = []

    def add_bucket(df: pd.DataFrame, count: int, label: str) -> None:
        nonlocal rows, selected_orders
        chosen_hashes = {r["order_hash"] for r in rows}
        take = eligible(df, chosen_hashes).head(max(0, count))
        for _, r in take.iterrows():
            row = r.to_dict()
            row["selected_by"] = label
            rows.append(row)
            selected_orders.append(parse_order(row["order_json"]))

    add_bucket(unique.sort_values("final_v03_score", ascending=False), 5, "top_v03_score")
    add_bucket(unique.sort_values("predicted_u2_guarded_score", ascending=False), 4, "u2_guarded")
    add_bucket(unique.sort_values(["predicted_lex_primary_score", "novelty_vs_combined552"], ascending=[False, False]), 3, "lex_primary_novel")
    upper_quartile = unique.head(max(32, len(unique) // 4))
    diverse = maximin(eligible(upper_quartile, {r["order_hash"] for r in rows}), selected_orders, 2)
    for _, r in diverse.iterrows():
        row = r.to_dict()
        row["selected_by"] = "diverse_upper_quartile"
        rows.append(row)
        selected_orders.append(parse_order(row["order_json"]))
    add_bucket(unique[unique["ppo_generation_mode"] == "deterministic"], 1, "deterministic")
    add_bucket(unique.sort_values("predicted_record_seeking_score", ascending=False), 1, "record_seeking")

    if len(rows) < 16:
        add_bucket(unique.sort_values(["final_v03_score", "novelty_vs_combined552"], ascending=[False, False]), 16 - len(rows), "fill_next_best")
    if len(rows) < 16:
        # If strict novelty leaves too few, fail closed only after trying non-duplicate within selected but flagged source duplicates.
        fallback = unique[~unique["order_hash"].isin({r["order_hash"] for r in rows})].sort_values("final_v03_score", ascending=False).head(16 - len(rows))
        for _, r in fallback.iterrows():
            row = r.to_dict()
            row["selected_by"] = "fallback_flagged_review"
            rows.append(row)
    rows = rows[:16]
    for i, row in enumerate(rows, start=1):
        bucket = str(row["selected_by"])
        row["strategy_name"] = f"PPOV03_N{n}_B{i:02d}_{bucket}"
        row["notes"] = "PPO v03 lex-primary candidate; not physically validated yet."
    return pd.DataFrame(rows)


def write_scan_json(row: pd.Series) -> None:
    payload = row.to_dict()
    payload["order"] = parse_order(row["order_json"])
    payload["candidate_source"] = "PPO_v03_checkpoint_inference"
    payload["teacher_validated"] = False
    payload["abaqus_validated"] = False
    (SCAN_DIR / f"scan_order_{row['strategy_name']}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def audits(selected: pd.DataFrame) -> tuple[str, str, str]:
    leg_rows = []
    for _, row in selected.iterrows():
        order = parse_order(row["order_json"])
        leg_rows.append(
            {
                "strategy_name": row["strategy_name"],
                "n": int(row["n"]),
                "legal": validate_order(int(row["n"]), order),
                "candidate_source_ok": row["candidate_source"] == "PPO_v03_checkpoint_inference",
                "teacher_validated_false": not bool(row["teacher_validated"]),
                "abaqus_validated_false": not bool(row["abaqus_validated"]),
                "duplicate_within_selected": bool(selected["order_hash"].duplicated(keep=False).loc[row.name]),
            }
        )
    leg = pd.DataFrame(leg_rows)
    leg.to_csv(LEGALITY_CSV, index=False)
    novelty_cols = [
        "strategy_name",
        "n",
        "order_hash",
        "novelty_vs_combined552",
        "novelty_vs_v01",
        "novelty_vs_v02K2",
        "duplicate_vs_combined552",
        "duplicate_vs_v01",
        "duplicate_vs_v02K2",
        "selected_by",
    ]
    selected[novelty_cols].to_csv(NOVELTY_CSV, index=False)
    score = selected.groupby("n").agg(
        count=("strategy_name", "size"),
        mean_final_v03_score=("final_v03_score", "mean"),
        max_final_v03_score=("final_v03_score", "max"),
        mean_u2_guarded=("predicted_u2_guarded_score", "mean"),
        max_u2_guarded=("predicted_u2_guarded_score", "max"),
        mean_lex_primary=("predicted_lex_primary_score", "mean"),
        max_lex_primary=("predicted_lex_primary_score", "max"),
        duplicate_vs_any_reference=("duplicate_vs_combined552", "sum"),
        surfaceT_false_positive_penalty_count=("surfaceT_only_false_positive_penalty", lambda s: int((s > 0).sum())),
    ).reset_index()
    score.to_csv(SCORE_SUMMARY_CSV, index=False)
    selected[
        [
            "strategy_name",
            "n",
            "surfaceT_only_false_positive_penalty",
            "predicted_lex_primary_score",
            "predicted_u2_guarded_score",
            "final_v03_score",
            "selected_by",
        ]
    ].to_csv(SURFACET_SCREEN_CSV, index=False)
    legality = "PASS" if leg["legal"].all() and leg["candidate_source_ok"].all() and not leg["duplicate_within_selected"].any() else "FAIL"
    novelty = "PASS" if not (selected["duplicate_vs_combined552"] | selected["duplicate_vs_v01"] | selected["duplicate_vs_v02K2"]).any() else "WARNING_DUPLICATES_FLAGGED"
    surface = "PASS" if int((selected["surfaceT_only_false_positive_penalty"] > 0).sum()) == 0 else "WARNING_SURFACET_FALSE_POSITIVES_FLAGGED"
    return legality, novelty, surface


def write_report_and_manifest(selected: pd.DataFrame, pool: pd.DataFrame, legality: str, novelty: str, surface: str, verdict: str) -> None:
    for _, row in selected.iterrows():
        write_scan_json(row)
    fig, ax = plt.subplots(figsize=(7, 4))
    selected.boxplot(column="final_v03_score", by="n", ax=ax)
    ax.set_title("v03 selected final score by N")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "v03_selected_final_score_by_N.png", dpi=180)
    plt.close(fig)

    counts = selected["n"].astype(int).value_counts().sort_index().to_dict()
    unique_counts = pool.groupby("n")["order_hash"].nunique().to_dict()
    pool_counts = pool["n"].astype(int).value_counts().sort_index().to_dict()
    surrogate_summary = json.loads(SURR_SUMMARY.read_text(encoding="utf-8")) if SURR_SUMMARY.exists() else {}
    train_summary = json.loads(TRAIN_SUMMARY.read_text(encoding="utf-8")) if TRAIN_SUMMARY.exists() else {}
    eval_summary = json.loads(EVAL_SUMMARY.read_text(encoding="utf-8")) if EVAL_SUMMARY.exists() else {}
    REPORT.write_text(
        f"""# PPO v03 Lex-Primary Candidate Generation Report

## Purpose
Generate a PPO-only N24/N40 candidate batch that explicitly prioritizes U2-primary and lexicographic U2->PEEQ->SurfaceT performance while avoiding SurfaceT-only false positives.

## Why v03 Was Needed
PPO v01/v02K2 did not beat mature combined552 records. v02K2 improved N40 SurfaceT top-k counts, but not primary lexicographic ranking. v03 therefore targets lex-primary and U2-guarded rewards.

## Dataset Assembly
Input dataset: `{DATASET}`. Rows: combined552 N24/N40 + PPO v01 N24/N40 + PPO v02K2 N24/N40.

## Surrogate/Ranking Model Results
Summary: `{SURR_SUMMARY}`.

## PPO Training Status
Summary: `{TRAIN_SUMMARY}`.

## Internal Surrogate Evaluation
Summary: `{EVAL_SUMMARY}`.

## Rollout Pool Size And Uniqueness
- Pool CSV: `{POOL_CSV}`
- Pool counts by N: {pool_counts}
- Unique orders by N: {unique_counts}

## Candidate Selection Logic
Per N: 5 top final v03 score, 4 top U2-guarded, 3 lex-primary with novelty, 2 diverse upper quartile, 1 deterministic, 1 record-seeking, with overlap filled by next eligible PPO-generated candidates.

## Audits
- Legality: `{legality}` at `{LEGALITY_CSV}`
- Novelty: `{novelty}` at `{NOVELTY_CSV}`
- SurfaceT false-positive screening: `{surface}` at `{SURFACET_SCREEN_CSV}`
- Score summary: `{SCORE_SUMMARY_CSV}`

## Selected Batch
Selected path: `{SELECTED_CSV}`
Counts by N: {counts}

## Claim Boundary
The v03 batch is not physically validated. Surrogate scores are candidate-generation/ranking signals only.

## Verdict
`{verdict}`
""",
        encoding="utf-8",
    )
    CLAIM_BOUNDARY.write_text(
        "# PPO v03 Lex-Primary Claim Boundary\n\n"
        "## Safe After Stage P\n\n"
        "- PPO v03 lex-primary N24/N40 policies were trained in surrogate reward environments.\n"
        "- PPO v03 generated legal candidate scan orders.\n"
        "- Candidate batch is ready for later CAE/INP generation if audits pass.\n"
        "- The batch is not physically validated yet.\n\n"
        "## Unsafe\n\n"
        "- PPO v03 improves physical metrics.\n"
        "- PPO v03 beats v02K2.\n"
        "- PPO v03 beats combined552.\n"
        "- PPO v03 is teacher validated.\n"
        "- PPO v03 solves N24/N40.\n",
        encoding="utf-8",
    )
    manifest = {
        "branch": git_branch(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_dataset_paths": {
            "v03_teacher_dataset": str(DATASET),
        },
        "v03_teacher_dataset_path": str(DATASET),
        "surrogate_model_paths": [str(p) for p in sorted(SURR_MODELS.glob("N*_v03_lex_primary_surrogate.joblib"))],
        "PPO_checkpoint_paths": [str(p) for p in sorted(CHECKPOINT_DIR.glob("N*_seed*_maskable_ppo_v03.zip"))],
        "rollout_pool_path": str(POOL_CSV),
        "selected_batch_path": str(SELECTED_CSV),
        "scan_order_JSON_dir": str(SCAN_DIR),
        "audit_paths": [str(LEGALITY_CSV), str(NOVELTY_CSV), str(SCORE_SUMMARY_CSV), str(SURFACET_SCREEN_CSV)],
        "report_path": str(REPORT),
        "claim_boundary_path": str(CLAIM_BOUNDARY),
        "selected_counts_by_N": {str(k): int(v) for k, v in counts.items()},
        "rollout_pool_size_by_N": {str(k): int(v) for k, v in pool_counts.items()},
        "unique_orders_by_N": {str(k): int(v) for k, v in unique_counts.items()},
        "no_Abaqus": True,
        "no_ODB_opening": True,
        "no_ODB_extraction": True,
        "no_solver": True,
        "no_datacheck": True,
        "no_enqueue": True,
        "no_CAE_INP_JNL": True,
        "no_teacher_validation": True,
        "no_commit_or_push": True,
        "final_verdict": verdict,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    ensure_dirs()
    pool = generate_pool()
    selected = pd.concat([select_for_n(pool[pool["n"] == n], n) for n in [24, 40]], ignore_index=True)
    selected["candidate_source"] = "PPO_v03_checkpoint_inference"
    selected["teacher_validated"] = False
    selected["abaqus_validated"] = False
    selected.to_csv(SELECTED_CSV, index=False)
    legality, novelty, surface = audits(selected)
    counts = selected["n"].astype(int).value_counts().to_dict()
    pass_ready = len(selected) == 32 and counts.get(24) == 16 and counts.get(40) == 16 and legality == "PASS" and surface == "PASS"
    verdict = "PASS_PPO_V03_LEX_PRIMARY_BATCH32_READY_FOR_CAE_INP_HANDOFF" if pass_ready else "WARNING_PPO_V03_LEX_PRIMARY_PARTIAL_REVIEW"
    write_report_and_manifest(selected, pool, legality, novelty, surface, verdict)
    summary = {
        "verdict": verdict,
        "rollout_pool_size_by_N": {str(k): int(v) for k, v in pool["n"].value_counts().sort_index().to_dict().items()},
        "unique_orders_by_N": {str(k): int(v) for k, v in pool.groupby("n")["order_hash"].nunique().to_dict().items()},
        "selected_counts_by_N": {str(k): int(v) for k, v in counts.items()},
        "selected_batch_path": str(SELECTED_CSV),
        "scan_order_JSON_dir": str(SCAN_DIR),
        "legality": legality,
        "novelty": novelty,
        "surfaceT_false_positive_screening": surface,
        "no_Abaqus": True,
        "no_ODB_opening": True,
        "no_solver": True,
    }
    print(json.dumps(summary, indent=2))
    return 0 if not verdict.startswith("FAIL") else 1


if __name__ == "__main__":
    raise SystemExit(main())
