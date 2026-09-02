"""Stage Y2 exploratory N64 candidates lifted from N32 PPO-inferred motifs.

This script is transformation-only.  It never performs direct N64 PPO action
inference, Abaqus execution, ODB access, training, or scan-order repair.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
NS = "stage3_policy_lift_N64_from_N32_PPO_motifs"
OUT = ROOT / "outputs" / NS
DOCS = ROOT / "docs" / NS
STAGE_Y = ROOT / "outputs" / "stage3_policy_extrapolation_N32_N64_predicted_optima"
N32_POOL = STAGE_Y / "rollouts" / "N32_policy_inference_rollout_pool.csv"
N32_RANKED = STAGE_Y / "tables" / "N32_predicted_optima_ranked.csv"
STAGE_Y_MANIFEST = STAGE_Y / "stageY_N32_N64_policy_extrapolation_manifest.json"


def ensure_dirs() -> None:
    for path in [OUT, OUT / "tables", OUT / "scan_orders", OUT / "reports", DOCS]:
        path.mkdir(parents=True, exist_ok=True)


def branch() -> str:
    result = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, capture_output=True, text=True, check=False)
    return result.stdout.strip() or "UNKNOWN"


def parse_order(value: object) -> list[int]:
    text = str(value).strip()
    if text.startswith("["):
        return [int(item) for item in json.loads(text)]
    return [int(item) for item in text.replace("|", ",").split(",") if item.strip()]


def compact(order: list[int]) -> str:
    return ",".join(str(item) for item in order)


def order_hash(order: list[int]) -> str:
    return hashlib.sha256(compact(order).encode("utf-8")).hexdigest()[:16]


def legal(order: list[int], n: int = 64) -> bool:
    return len(order) == n and set(order) == set(range(n))


def descriptors(order: list[int]) -> dict[str, float | int]:
    arr = np.asarray(order, dtype=float)
    jumps = np.abs(np.diff(arr))
    quarters = (arr // 16).astype(int)
    halves = (arr >= 32).astype(int)
    center = 31.5
    early = {}
    for length in (8, 16, 32):
        prefix = arr[:length]
        early[f"early_coverage_span_{length}"] = float(prefix.max() - prefix.min())
    unique_quartiles = int(len(set(quarters[:16].tolist())))
    edge_distance = np.minimum(arr, 63 - arr) / center
    edge_center_balance = float(np.mean(np.abs(arr - center) / center) - np.mean(edge_distance))
    adjacent = float(np.mean(jumps == 1))
    mean_jump = float(jumps.mean())
    same_quarter = float(np.mean(quarters[1:] == quarters[:-1]))
    alternating_halves = float(np.mean(halves[1:] != halves[:-1]))
    coarse_dispersion = float(
        0.32 * (np.mean(jumps) / 63.0)
        + 0.22 * (np.median(jumps) / 63.0)
        + 0.18 * (unique_quartiles / 4.0)
        + 0.18 * alternating_halves
        + 0.10 * (1.0 - same_quarter)
    )
    return {
        "total_travel_proxy": float(jumps.sum()),
        "mean_abs_jump": mean_jump,
        "median_abs_jump": float(np.median(jumps)),
        "min_abs_jump": float(jumps.min()),
        "max_abs_jump": float(jumps.max()),
        "long_jump_count_8": int(np.sum(jumps >= 8)),
        "long_jump_count_16": int(np.sum(jumps >= 16)),
        "adjacent_fraction": adjacent,
        "jump_variance": float(jumps.var()),
        "local_continuity_score": float(adjacent - mean_jump / 63.0),
        "path_complexity_score": float(jumps.sum() / (64.0 * 63.0) + jumps.var() / (63.0**2)),
        **early,
        "early_unique_quartiles_16": unique_quartiles,
        "same_quartile_transition_fraction": same_quarter,
        "alternating_half_fraction": alternating_halves,
        "edge_center_balance_score": edge_center_balance,
        "coarse_dispersion_score": coarse_dispersion,
    }


def preflight() -> dict[str, object]:
    checks = {
        "N32_rollout_pool_exists": N32_POOL.exists(),
        "N32_ranked_table_exists": N32_RANKED.exists(),
        "StageY_manifest_exists": STAGE_Y_MANIFEST.exists(),
        "transformation_only": True,
        "direct_N64_PPO": False,
        "no_Abaqus": True,
    }
    stage_y = {}
    if STAGE_Y_MANIFEST.exists():
        stage_y = json.loads(STAGE_Y_MANIFEST.read_text(encoding="utf-8"))
        checks["StageY_N64_blocked"] = "BLOCKED" in str(stage_y.get("N64_status", ""))
    else:
        checks["StageY_N64_blocked"] = False
    legal_seed_count = 0
    if N32_POOL.exists():
        pool = pd.read_csv(N32_POOL)
        for item in pool.get("order_compact", pd.Series(dtype=str)):
            order = parse_order(item)
            legal_seed_count += int(legal(order, 32))
    checks["legal_N32_orders"] = legal_seed_count
    verdict = "PASS_STAGEY2_N64_POLICY_LIFT_PREFLIGHT_READY" if all(
        bool(checks[key]) for key in ["N32_rollout_pool_exists", "N32_ranked_table_exists", "StageY_manifest_exists", "StageY_N64_blocked"]
    ) and legal_seed_count > 0 else "FAIL_STAGEY2_N64_POLICY_LIFT_PREFLIGHT_BLOCKED"
    payload = {"timestamp": datetime.now().isoformat(timespec="seconds"), "checks": checks, "stageY_status": stage_y, "verdict": verdict}
    (OUT / "reports" / "stageY2_preflight_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def select_seeds(pool: pd.DataFrame, ranked: pd.DataFrame) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    ranked = ranked.copy()
    ranked["predicted_reward"] = pd.to_numeric(ranked["predicted_reward"], errors="coerce")
    selected.append(ranked.nlargest(50, "predicted_reward"))
    selected.append(ranked[ranked["inference_mode"].eq("deterministic")].head(1))
    selected.append(pool.nsmallest(25, "total_travel_proxy"))
    efficiency = pool.sort_values(["adjacent_fraction", "long_jump_count", "total_travel_proxy"], ascending=[False, True, True]).head(25)
    selected.append(efficiency)
    top_half = ranked.nlargest(max(1, len(ranked) // 2), "predicted_reward")
    # Deterministic hash-stratified selection: retains broad policy-rollout coverage.
    selected.append(top_half.assign(_hash_value=top_half["order_hash"].map(lambda value: int(str(value), 16))).sort_values("_hash_value").head(100))
    seeds = pd.concat(selected, ignore_index=True, sort=False).drop_duplicates("order_hash", keep="first")
    seeds = seeds[seeds["order_compact"].map(lambda value: legal(parse_order(value), 32))].copy()
    seeds["source_N32_predicted_reward"] = pd.to_numeric(seeds.get("predicted_reward"), errors="coerce")
    seeds["source_N32_policy_logprob"] = pd.to_numeric(seeds.get("policy_score"), errors="coerce")
    seeds["inherited_policy_rank"] = seeds["source_N32_predicted_reward"].rank(method="first", ascending=False).astype(int)
    seeds["inherited_predicted_reward_rank"] = seeds["inherited_policy_rank"]
    keep = [column for column in ["strategy_name", "order_compact", "order_hash", "source_N32_predicted_reward", "source_N32_policy_logprob", "inherited_policy_rank", "inherited_predicted_reward_rank"] if column in seeds]
    return seeds[keep].sort_values("inherited_policy_rank").reset_index(drop=True)


def pair_forward(seed: list[int]) -> list[int]:
    return [track for macro in seed for track in (2 * macro, 2 * macro + 1)]


def pair_reverse(seed: list[int]) -> list[int]:
    return [track for macro in seed for track in (2 * macro + 1, 2 * macro)]


def rank_parity(seed: list[int]) -> list[int]:
    return [track for rank, macro in enumerate(seed) for track in ((2 * macro, 2 * macro + 1) if rank % 2 == 0 else (2 * macro + 1, 2 * macro))]


def macro_parity(seed: list[int]) -> list[int]:
    return [track for macro in seed for track in ((2 * macro, 2 * macro + 1) if macro % 2 == 0 else (2 * macro + 1, 2 * macro))]


def split(seed: list[int], second_first: bool = False) -> list[int]:
    first = [2 * macro + (1 if second_first else 0) for macro in seed]
    second = [2 * macro + (0 if second_first else 1) for macro in seed]
    return first + second


def mirror_forward(seed: list[int]) -> list[int]:
    return [track for macro in seed for track in (2 * macro, 63 - 2 * macro)]


def macro_mirror_half(seed: list[int]) -> list[int]:
    return [track for macro in seed for track in (macro, macro + 32)]


def block_halves(seed: list[int], upper_first: bool = False, alternating: bool = False) -> list[int]:
    lower = list(seed)
    upper = [macro + 32 for macro in seed]
    if alternating:
        return [track for index in range(32) for track in ((upper[index], lower[index]) if upper_first else (lower[index], upper[index]))]
    return (upper + lower) if upper_first else (lower + upper)


def dispersed_pair(seed: list[int]) -> list[int]:
    rank = {macro: position for position, macro in enumerate(seed)}
    remaining = set(seed)
    ordered = [seed[0]]
    remaining.remove(seed[0])
    while remaining:
        last = ordered[-1]
        # maximize coarse separation; original PPO order rank resolves ties reproducibly.
        next_macro = sorted(remaining, key=lambda macro: (-abs(macro - last), rank[macro], macro))[0]
        ordered.append(next_macro)
        remaining.remove(next_macro)
    return pair_forward(ordered)


def lifts(seed: list[int]) -> dict[str, list[int]]:
    return {
        "adjacent_pair_forward": pair_forward(seed),
        "adjacent_pair_reverse": pair_reverse(seed),
        "rank_parity_alternating": rank_parity(seed),
        "macro_even_odd_local": macro_parity(seed),
        "pair_split_first_pass_then_second_pass": split(seed),
        "pair_split_second_pass_then_first_pass": split(seed, second_first=True),
        "mirror_pair_forward": mirror_forward(seed),
        "macro_order_plus_mirror_half": macro_mirror_half(seed),
        "blockwise_lower_then_upper": block_halves(seed),
        "blockwise_upper_then_lower": block_halves(seed, upper_first=True),
        "blockwise_alternating_lower_upper": block_halves(seed, alternating=True),
        "blockwise_alternating_upper_lower": block_halves(seed, upper_first=True, alternating=True),
        "physically_dispersed_pair_lift": dispersed_pair(seed),
    }


def generate_pool(seeds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    seen: set[str] = set()
    for _, source in seeds.iterrows():
        seed = parse_order(source["order_compact"])
        for family, order in lifts(seed).items():
            if not legal(order):
                continue
            digest = order_hash(order)
            if digest in seen:
                continue
            seen.add(digest)
            row = {
                "N": 64,
                "strategy_name": f"N32LIFT_N64_{len(rows)+1:05d}_{family}",
                "order_compact": compact(order),
                "order_hash": digest,
                "source": "N32_policy_lifted_N64_candidate",
                "candidate_label": "PPO_motif_extrapolated_N64_candidate",
                "evidence_tier": "policy_motif_extrapolated_prediction_only",
                "teacher_metrics_available": False,
                "direct_N64_policy": False,
                "physical_claim": "none",
                "source_N32_strategy": source["strategy_name"],
                "source_N32_predicted_reward": source["source_N32_predicted_reward"],
                "source_N32_policy_logprob": source["source_N32_policy_logprob"],
                "inherited_policy_rank": source["inherited_policy_rank"],
                "inherited_predicted_reward_rank": source["inherited_predicted_reward_rank"],
                "lift_family": family,
                "caveat": "N32-policy-lifted N64 extrapolation, not direct N64 PPO inference and not Abaqus teacher-validated.",
            }
            row.update(descriptors(order))
            rows.append(row)
    return pd.DataFrame(rows)


def add_rankings(pool: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inherited = pool.sort_values(
        ["source_N32_predicted_reward", "inherited_policy_rank", "coarse_dispersion_score", "total_travel_proxy"],
        ascending=[False, True, False, True],
    ).copy()
    dispersion = pool.sort_values(
        ["coarse_dispersion_score", "early_unique_quartiles_16", "same_quartile_transition_fraction", "total_travel_proxy"],
        ascending=[False, False, True, True],
    ).copy()
    efficiency = pool.sort_values(
        ["total_travel_proxy", "max_abs_jump", "long_jump_count_16", "adjacent_fraction"],
        ascending=[True, True, True, False],
    ).copy()
    for frame in (inherited, dispersion, efficiency):
        frame["ranking_position"] = np.arange(1, len(frame) + 1)
    rank_map = pd.DataFrame({
        "order_hash": pool["order_hash"],
        "rank_inherited_reward": inherited.set_index("order_hash").loc[pool["order_hash"], "ranking_position"].to_numpy(),
        "rank_coarse_dispersion": dispersion.set_index("order_hash").loc[pool["order_hash"], "ranking_position"].to_numpy(),
        "rank_efficiency": efficiency.set_index("order_hash").loc[pool["order_hash"], "ranking_position"].to_numpy(),
    })
    balanced = pool.merge(rank_map, on="order_hash", how="left")
    # A transparent equal-weight rank aggregation.  No physical score is implied.
    balanced["balanced_rank_score"] = balanced[["rank_inherited_reward", "rank_coarse_dispersion", "rank_efficiency"]].sum(axis=1)
    balanced = balanced.sort_values(["balanced_rank_score", "source_N32_predicted_reward", "total_travel_proxy"], ascending=[True, False, True]).copy()
    balanced["balanced_rank"] = np.arange(1, len(balanced) + 1)
    return inherited, dispersion, efficiency, balanced.head(20)


def export_jsons(frame: pd.DataFrame, directory: Path, limit: int) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for _, row in frame.head(limit).iterrows():
        path = directory / f"scan_order_{row['strategy_name']}.json"
        payload = {
            "strategy_name": row["strategy_name"],
            "N": 64,
            "order": parse_order(row["order_compact"]),
            "order_hash": row["order_hash"],
            "source": "N32_policy_lifted_N64_candidate",
            "source_N32_strategy": row["source_N32_strategy"],
            "source_N32_predicted_reward": float(row["source_N32_predicted_reward"]),
            "lift_family": row["lift_family"],
            "evidence_tier": "policy_motif_extrapolated_prediction_only",
            "teacher_metrics_available": False,
            "direct_N64_policy": False,
            "physical_claim": "none",
            "caveat": "This is N32-policy-lifted N64 extrapolation, not direct N64 PPO inference and not Abaqus teacher-validated.",
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        paths[str(row["order_hash"])] = str(path)
    return paths


def summary_row(category: str, row: pd.Series, json_path: str) -> dict[str, object]:
    return {
        "category": category,
        "strategy_name": row["strategy_name"],
        "source_N32_strategy": row["source_N32_strategy"],
        "lift_family": row["lift_family"],
        "source_N32_predicted_reward": row["source_N32_predicted_reward"],
        "balanced_rank": row.get("balanced_rank", np.nan),
        "total_travel_proxy": row["total_travel_proxy"],
        "mean_abs_jump": row["mean_abs_jump"],
        "max_abs_jump": row["max_abs_jump"],
        "adjacent_fraction": row["adjacent_fraction"],
        "coarse_dispersion_score": row["coarse_dispersion_score"],
        "order_compact": row["order_compact"],
        "order_json_path": json_path,
        "safe_claim": "N64 candidate generated by N32-policy-lifted PPO-motif extrapolation; prediction-only; not Abaqus teacher-validated.",
        "unsafe_claim": "Direct N64 PPO inference, physical optimum, teacher-validated optimum, or proof of N64 performance.",
    }


def write_docs(preflight_data: dict[str, object], summary: pd.DataFrame, verdict: str) -> tuple[Path, Path]:
    claim = DOCS / "N64_POLICY_LIFT_CLAIM_BOUNDARY.md"
    claim.write_text(
        """# N64 Policy-Lift Claim Boundary

## Safe

- N64 candidates were generated by lifting N32 PPO-inferred scan-order motifs.
- This is exploratory prediction-only extrapolation.
- The final Discrete(40) MaskablePPO was not used as a direct N64 action policy.
- No Abaqus teacher metrics are available for these N64 candidates.
- These candidates may be proposed for future N64 validation.

## Unsafe

- Direct N64 PPO inference.
- N64 teacher validation, physical optimum, global optimum, or superiority over combined552.
- Any assertion that an inherited N32 score proves N64 physical quality.
""",
        encoding="utf-8",
    )
    records = summary.to_dict(orient="records")
    report = DOCS / "N64_POLICY_LIFT_EXTRAPOLATION_REPORT.md"
    report.write_text(
        f"""# N64 Policy-Lift Extrapolation Report

## Purpose

Generate exploratory N64 prediction-only scan orders by deterministically lifting motifs from N32 PPO-inferred orders.

## Direct N64 Boundary

Stage Y confirmed that direct N64 PPO inference is blocked: the final policy has `Discrete(40)` actions and no trained variable-N or pointer-policy checkpoint with `max_N >= 64` was recovered. Stage Y2 does not load that PPO policy for N64 actions.

## Preflight

`{preflight_data['verdict']}`. The source N32 rollout pool and ranking table were present, legal seed orders were found, and the Stage Y N64 block was confirmed.

## Lift Transformations

Adjacent-pair, parity, split-pass, mirror/half, blockwise-half, and deterministic coarse-dispersal lifts were applied. Every retained N64 result is a legal `0..63` permutation.

## Best Candidates

```json
{json.dumps(records, indent=2, default=str)}
```

## Descriptor Interpretation

Travel, continuity and thermal-dispersion-style quantities are path descriptors only, not Abaqus or physical metrics.

## Recommended Presentation Wording

"The proposed N64 scan orders were generated by deterministic lifting of N32 PPO-inferred scan-order motifs. They are exploratory prediction-only candidates, not direct N64 PPO outputs and not Abaqus teacher-validated optima."

## Final Verdict

{verdict}
""",
        encoding="utf-8",
    )
    return claim, report


def main() -> None:
    ensure_dirs()
    check = preflight()
    if check["verdict"] != "PASS_STAGEY2_N64_POLICY_LIFT_PREFLIGHT_READY":
        raise SystemExit(check["verdict"])
    pool = pd.read_csv(N32_POOL)
    ranked = pd.read_csv(N32_RANKED)
    seeds = select_seeds(pool, ranked)
    seeds.to_csv(OUT / "tables" / "N32_seed_orders_for_N64_lift.csv", index=False)
    generated = generate_pool(seeds)
    if generated.empty or not generated["order_compact"].map(lambda value: legal(parse_order(value))).all():
        raise RuntimeError("No valid N64 policy-lift candidates were generated")
    generated_path = OUT / "tables" / "N64_policy_lift_generated_pool.csv"
    generated.to_csv(generated_path, index=False)
    inherited, dispersion, efficiency, balanced = add_rankings(generated)
    table_dir = OUT / "tables"
    inherited_path = table_dir / "N64_predicted_by_inherited_N32_reward.csv"
    dispersion_path = table_dir / "N64_predicted_by_physical_dispersion_proxy.csv"
    efficiency_path = table_dir / "N64_predicted_by_industrial_efficiency_proxy.csv"
    balanced_path = table_dir / "N64_predicted_balanced_top_candidates.csv"
    inherited.to_csv(inherited_path, index=False)
    dispersion.to_csv(dispersion_path, index=False)
    efficiency.to_csv(efficiency_path, index=False)
    balanced.to_csv(balanced_path, index=False)
    balanced_json = export_jsons(balanced, OUT / "scan_orders" / "N64_balanced_top20", 20)
    inherited_json = export_jsons(inherited, OUT / "scan_orders" / "N64_by_inherited_reward_top10", 10)
    dispersion_json = export_jsons(dispersion, OUT / "scan_orders" / "N64_by_dispersion_top10", 10)
    efficiency_json = export_jsons(efficiency, OUT / "scan_orders" / "N64_by_efficiency_top10", 10)
    best_rows = [
        summary_row("best_by_inherited_N32_reward", inherited.iloc[0], inherited_json.get(str(inherited.iloc[0]["order_hash"]), "")),
        summary_row("best_by_dispersion_proxy", dispersion.iloc[0], dispersion_json.get(str(dispersion.iloc[0]["order_hash"]), "")),
        summary_row("best_by_efficiency_proxy", efficiency.iloc[0], efficiency_json.get(str(efficiency.iloc[0]["order_hash"]), "")),
        summary_row("best_balanced_candidate", balanced.iloc[0], balanced_json.get(str(balanced.iloc[0]["order_hash"]), "")),
    ]
    summary = pd.DataFrame(best_rows)
    summary_path = table_dir / "N64_policy_lift_best_candidate_summary.csv"
    summary.to_csv(summary_path, index=False)
    verdict = "PASS_STAGEY2_N64_POLICY_LIFT_EXTRAPOLATION_COMPLETED"
    claim_path, report_path = write_docs(check, summary, verdict)
    manifest = {
        "branch": branch(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_N32_rollout_pool": str(N32_POOL),
        "input_N32_ranked_table": str(N32_RANKED),
        "N32_seed_order_count": int(len(seeds)),
        "generated_pool_path": str(generated_path),
        "N64_generated_count": int(len(generated)),
        "N64_unique_count": int(generated["order_hash"].nunique()),
        "ranking_table_paths": [str(inherited_path), str(dispersion_path), str(efficiency_path), str(balanced_path)],
        "top_JSON_directories": [str(OUT / "scan_orders" / name) for name in ["N64_balanced_top20", "N64_by_inherited_reward_top10", "N64_by_dispersion_top10", "N64_by_efficiency_top10"]],
        "best_summary_table": str(summary_path),
        "report_path": str(report_path),
        "claim_boundary_path": str(claim_path),
        "direct_N64_PPO": False,
        "N32_policy_lift": True,
        "teacher_metrics_available": False,
        "no_Abaqus": True,
        "no_ODB_opening": True,
        "no_solver": True,
        "no_datacheck": True,
        "no_enqueue": True,
        "no_training": True,
        "no_commit_or_push": True,
        "final_verdict": verdict,
    }
    (OUT / "stageY2_N64_policy_lift_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
