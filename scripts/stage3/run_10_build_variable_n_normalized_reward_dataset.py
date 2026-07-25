from __future__ import annotations

import csv
import itertools
import json
import math
import random
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
RUN_ID = "run_10_variable_n_normalized_reward_surrogate_dataset"
RUN_NAME = "variable-N normalized reward and surrogate-pretraining dataset"

RUN09_CANONICAL = ROOT / "outputs" / "stage3_run_09_variable_n_probe60_teacher_ranking_analysis" / "probe60_teacher_ranked_canonical.csv"
RUN09_FAMILY_SUMMARY = ROOT / "outputs" / "stage3_run_09_variable_n_probe60_teacher_ranking_analysis" / "strategy_family_summary.csv"
RUN09_GROUP_COMPARISON = ROOT / "outputs" / "stage3_run_09_variable_n_probe60_teacher_ranking_analysis" / "candidate_group_comparison.csv"
RUN09_PARETO = ROOT / "outputs" / "stage3_run_09_variable_n_probe60_teacher_ranking_analysis" / "pareto_front_cases.csv"
RUN09_CLAIM_BOUNDARY = ROOT / "outputs" / "stage3_run_09_variable_n_probe60_teacher_ranking_analysis" / "run09_claim_boundary.md"
RUN09_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_09_variable_n_probe60_teacher_ranking_analysis" / "RUN_09_VARIABLE_N_PROBE60_TEACHER_RANKING_ANALYSIS_REPORT.md"
RUN08_TEACHER_LABELS = ROOT / "outputs" / "stage3_run_08_probe60_odb_teacher_validation" / "probe60_odb_teacher_labels.csv"
CANDIDATE_ORDERS = ROOT / "outputs" / "stage3_run_06_variable_n_probe60_candidate_order_generation" / "variable_N_probe60_candidate_orders.csv"

OUTPUT_DIR = ROOT / "outputs" / f"stage3_{RUN_ID}"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = ROOT / "docs" / "stage3" / "runs" / RUN_ID
REPORT_PATH = REPORT_DIR / "RUN_10_VARIABLE_N_NORMALIZED_REWARD_SURROGATE_DATASET_REPORT.md"
MANIFEST_PATH = ROOT / "artifacts" / "manifests" / "stage3_run_10_manifest.json"
RUN_INDEX_PATH = ROOT / "docs" / "stage3" / "STAGE3_RUN_INDEX.md"

EXPECTED_N = [12, 16, 24, 40]
EXPECTED_ROWS_PER_N = 15
EXPECTED_TOTAL_ROWS = 60

REWARD_COLUMNS = [
    "reward_v01_u2_primary",
    "reward_v02_safety_weighted",
    "reward_v03_surfaceT_aware",
    "reward_v04_penalized",
    "reward_v05_lexicographic",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        if not fieldnames:
            fieldnames = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_int(value: Any) -> int:
    text = str(value).strip()
    if text.upper().startswith("N"):
        text = text[1:]
    return int(text)


def parse_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    return numerator / denominator if denominator else default


def clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def mean(values: list[float], default: float = 0.0) -> float:
    return statistics.fmean(values) if values else default


def std(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def median(values: list[float], default: float = 0.0) -> float:
    return statistics.median(values) if values else default


def rank_values(values: list[float], reverse: bool = False) -> list[float]:
    indexed = list(enumerate(values))
    indexed.sort(key=lambda item: item[1], reverse=reverse)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return None
    xvals = [p[0] for p in pairs]
    yvals = [p[1] for p in pairs]
    mx = mean(xvals)
    my = mean(yvals)
    numerator = sum((x - mx) * (y - my) for x, y in pairs)
    denominator = math.sqrt(sum((x - mx) ** 2 for x in xvals) * sum((y - my) ** 2 for y in yvals))
    return safe_divide(numerator, denominator, default=0.0)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return None
    xr = rank_values([p[0] for p in pairs])
    yr = rank_values([p[1] for p in pairs])
    return pearson(xr, yr)


def load_metadata() -> tuple[dict[str, dict[str, str]], list[str]]:
    warnings: list[str] = []
    if not CANDIDATE_ORDERS.exists():
        return {}, [f"Candidate order metadata missing: {CANDIDATE_ORDERS}"]
    rows = read_csv(CANDIDATE_ORDERS)
    metadata = {}
    for row in rows:
        name = row.get("strategy_name", "").strip()
        if name:
            metadata[name] = row
    return metadata, warnings


def parse_order(text: str) -> list[int] | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, list):
        return None
    order: list[int] = []
    for item in value:
        if isinstance(item, bool):
            return None
        try:
            integer = int(item)
        except (TypeError, ValueError):
            return None
        if integer != item and not (isinstance(item, str) and str(integer) == item.strip()):
            return None
        order.append(integer)
    return order


def validate_order(order: list[int] | None, n: int) -> tuple[bool, str]:
    if order is None:
        return False, "scan order could not be parsed"
    if len(order) != n:
        return False, f"length {len(order)} != N {n}"
    expected = set(range(n))
    actual = set(order)
    if len(actual) != len(order):
        return False, "duplicate tracks"
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        return False, f"missing={missing}; extra={extra}"
    return True, "legal permutation"


def load_pareto_flags() -> dict[str, dict[str, bool]]:
    flags: dict[str, dict[str, bool]] = defaultdict(lambda: {
        "pareto_overall_u2_peeq": False,
        "pareto_overall_u2_peeq_surfaceT": False,
        "pareto_within_n_u2_peeq": False,
        "pareto_within_n_u2_peeq_surfaceT": False,
    })
    if not RUN09_PARETO.exists():
        return flags
    for row in read_csv(RUN09_PARETO):
        name = row.get("strategy_name", "")
        scope = row.get("scope", "")
        if not name:
            continue
        if scope == "overall":
            flags[name]["pareto_overall_u2_peeq"] |= parse_bool(row.get("pareto_u2_peeq"))
            flags[name]["pareto_overall_u2_peeq_surfaceT"] |= parse_bool(row.get("pareto_u2_peeq_surfaceT"))
        elif scope.startswith("N"):
            flags[name]["pareto_within_n_u2_peeq"] |= parse_bool(row.get("pareto_u2_peeq"))
            flags[name]["pareto_within_n_u2_peeq_surfaceT"] |= parse_bool(row.get("pareto_u2_peeq_surfaceT"))
    return flags


def validate_inputs(canonical_rows: list[dict[str, str]], metadata: dict[str, dict[str, str]]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    counts = defaultdict(int)
    missing_orders: list[str] = []
    invalid_orders: list[dict[str, Any]] = []
    required_columns = [
        "n",
        "strategy_name",
        "u2_range_canonical",
        "peeq_max_canonical",
        "surfaceT_proxy_canonical",
        "u2_rank_within_n",
        "peeq_rank_within_n",
        "surfaceT_rank_within_n",
    ]
    headers = set(canonical_rows[0].keys()) if canonical_rows else set()
    for column in required_columns:
        if column not in headers:
            errors.append(f"Missing required canonical column: {column}")
    names_seen = set()
    for row in canonical_rows:
        try:
            n = parse_int(row.get("n"))
        except (TypeError, ValueError):
            errors.append(f"Invalid N value: {row.get('n')}")
            continue
        counts[n] += 1
        name = row.get("strategy_name", "").strip()
        if not name:
            errors.append("Missing strategy_name")
            continue
        names_seen.add(name)
        order_text = metadata.get(name, {}).get("order_json") or row.get("order_json") or row.get("raw_scan_order", "")
        order = parse_order(order_text)
        legal, reason = validate_order(order, n)
        if not order_text:
            missing_orders.append(name)
        if not legal:
            invalid_orders.append({"strategy_name": name, "n": n, "reason": reason})
    if len(canonical_rows) != EXPECTED_TOTAL_ROWS:
        errors.append(f"Expected {EXPECTED_TOTAL_ROWS} rows, found {len(canonical_rows)}")
    if sorted(counts) != EXPECTED_N:
        errors.append(f"Expected N values {EXPECTED_N}, found {sorted(counts)}")
    for n in EXPECTED_N:
        if counts[n] != EXPECTED_ROWS_PER_N:
            errors.append(f"Expected {EXPECTED_ROWS_PER_N} rows for N={n}, found {counts[n]}")
    if "N24_A07_regular_jump_coprime" not in names_seen:
        errors.append("N24_A07_regular_jump_coprime is missing")
    if missing_orders:
        warnings.append(f"Missing order metadata rows: {missing_orders}")
    if invalid_orders:
        errors.append(f"Invalid scan orders: {invalid_orders[:5]}")
    missing_metadata = sorted(name for name in names_seen if name not in metadata)
    if missing_metadata:
        warnings.append(f"Candidate metadata incomplete for {len(missing_metadata)} rows")
    verdict = "PASS_RUN10_INPUTS_READY_60_TEACHER_LABELS_WITH_SCAN_ORDERS" if not errors else "FAIL_RUN10_INPUTS_INVALID"
    return {
        "verdict": verdict,
        "errors": errors,
        "warnings": warnings,
        "total_rows": len(canonical_rows),
        "per_n_counts": dict(sorted(counts.items())),
        "metadata_rows": len(metadata),
        "missing_metadata_count": len(missing_metadata),
        "missing_order_count": len(missing_orders),
        "invalid_order_count": len(invalid_orders),
        "n24_a07_regular_jump_coprime_valid": "N24_A07_regular_jump_coprime" in names_seen,
    }


def build_reward_dataset(canonical_rows: list[dict[str, str]], metadata: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    pareto_flags = load_pareto_flags()
    rows: list[dict[str, Any]] = []
    for row in canonical_rows:
        n = parse_int(row["n"])
        name = row["strategy_name"]
        meta = metadata.get(name, {})
        order_text = meta.get("order_json") or row.get("order_json") or row.get("raw_scan_order", "")
        order = parse_order(order_text)
        legal, reason = validate_order(order, n)
        if not legal:
            raise ValueError(f"Illegal scan order for {name}: {reason}")
        flags = pareto_flags.get(name, {})
        output = {
            "n": n,
            "strategy_name": name,
            "job_name": row.get("job_name_canonical") or row.get("raw_job_name", ""),
            "strategy_id": meta.get("strategy_id") or row.get("raw_strategy_id", ""),
            "strategy_family": meta.get("family") or row.get("strategy_family", ""),
            "candidate_group": row.get("candidate_group", ""),
            "policy_source": meta.get("policy_source") or row.get("policy_source", ""),
            "trained_policy_used": meta.get("trained_policy_used") or row.get("trained_policy_used", "False"),
            "teacher_validated": "True",
            "u2_range": parse_float(row["u2_range_canonical"]),
            "peeq_max": parse_float(row["peeq_max_canonical"]),
            "surfaceT_proxy": parse_float(row["surfaceT_proxy_canonical"]),
            "u2_rank_within_n": parse_float(row["u2_rank_within_n"]),
            "peeq_rank_within_n": parse_float(row["peeq_rank_within_n"]),
            "surfaceT_rank_within_n": parse_float(row["surfaceT_rank_within_n"]),
            "u2_percentile_within_n": parse_float(row.get("u2_percentile_within_n")),
            "peeq_percentile_within_n": parse_float(row.get("peeq_percentile_within_n")),
            "surfaceT_percentile_within_n": parse_float(row.get("surfaceT_percentile_within_n")),
            "u2_z_within_n": parse_float(row.get("u2_z_within_n")),
            "peeq_z_within_n": parse_float(row.get("peeq_z_within_n")),
            "surfaceT_z_within_n": parse_float(row.get("surfaceT_z_within_n")),
            "simple_mean_rank": parse_float(row.get("simple_mean_rank")),
            "constrained_rank_within_n": parse_float(row.get("constrained_rank_within_n")),
            "scan_order_compact": ",".join(str(x) for x in order),
            "scan_order_json": json.dumps(order, separators=(",", ":")),
            "order_length": len(order),
            "scan_order_valid": True,
            "pareto_overall_u2_peeq": flags.get("pareto_overall_u2_peeq", False),
            "pareto_overall_u2_peeq_surfaceT": flags.get("pareto_overall_u2_peeq_surfaceT", False),
            "pareto_within_n_u2_peeq": flags.get("pareto_within_n_u2_peeq", False),
            "pareto_within_n_u2_peeq_surfaceT": flags.get("pareto_within_n_u2_peeq_surfaceT", False),
        }
        rows.append(output)
    for n in EXPECTED_N:
        group = [row for row in rows if row["n"] == n]
        count = len(group)
        denom = max(1, count - 1)
        for metric, score_col in [
            ("u2_range", "u2_score_minmax"),
            ("peeq_max", "peeq_score_minmax"),
            ("surfaceT_proxy", "surfaceT_score_minmax"),
        ]:
            values = [row[metric] for row in group]
            mn = min(values)
            mx = max(values)
            span = mx - mn
            for row in group:
                row[score_col] = clip01(1.0 - safe_divide(row[metric] - mn, span, default=0.0))
        for row in group:
            row["u2_score_rank"] = 1.0 - safe_divide(row["u2_rank_within_n"] - 1.0, denom)
            row["peeq_score_rank"] = 1.0 - safe_divide(row["peeq_rank_within_n"] - 1.0, denom)
            row["surfaceT_score_rank"] = 1.0 - safe_divide(row["surfaceT_rank_within_n"] - 1.0, denom)
            row["u2_score_z"] = -row["u2_z_within_n"] if math.isfinite(row["u2_z_within_n"]) else 0.0
            row["peeq_score_z"] = -row["peeq_z_within_n"] if math.isfinite(row["peeq_z_within_n"]) else 0.0
            row["surfaceT_score_z"] = -row["surfaceT_z_within_n"] if math.isfinite(row["surfaceT_z_within_n"]) else 0.0
            row["u2_score_percentile"] = 1.0 - row["u2_percentile_within_n"]
            row["peeq_score_percentile"] = 1.0 - row["peeq_percentile_within_n"]
            row["surfaceT_score_percentile"] = 1.0 - row["surfaceT_percentile_within_n"]
            row["reward_v01_u2_primary"] = 0.70 * row["u2_score_rank"] + 0.20 * row["peeq_score_rank"] + 0.10 * row["surfaceT_score_rank"]
            row["reward_v02_safety_weighted"] = 0.60 * row["u2_score_rank"] + 0.30 * row["peeq_score_rank"] + 0.10 * row["surfaceT_score_rank"]
            row["reward_v03_surfaceT_aware"] = 0.55 * row["u2_score_rank"] + 0.25 * row["peeq_score_rank"] + 0.20 * row["surfaceT_score_rank"]
            base = 0.65 * row["u2_score_rank"] + 0.25 * row["peeq_score_rank"] + 0.10 * row["surfaceT_score_rank"]
            penalty = 0.0
            if row["peeq_rank_within_n"] > 10:
                penalty += 0.15
            if row["u2_rank_within_n"] > 10:
                penalty += 0.20
            if row["surfaceT_rank_within_n"] > 10:
                penalty += 0.05
            row["reward_v04_penalized"] = clip01(base - penalty)
        lex_sorted = sorted(group, key=lambda row: (row["u2_rank_within_n"], row["peeq_rank_within_n"], row["surfaceT_rank_within_n"]))
        for rank, row in enumerate(lex_sorted, start=1):
            row["lexicographic_rank_within_n"] = rank
            row["reward_v05_lexicographic"] = 1.0 - safe_divide(rank - 1.0, denom)
        reward_std_values: list[float] = []
        for row in group:
            rewards = [row[column] for column in REWARD_COLUMNS]
            row["reward_mean_all"] = mean(rewards)
            row["reward_std_all"] = std(rewards)
            reward_std_values.append(row["reward_std_all"])
        q3 = sorted(reward_std_values)[int(math.ceil(0.75 * len(reward_std_values))) - 1]
        consensus_sorted = sorted(group, key=lambda row: row["reward_mean_all"], reverse=True)
        for rank, row in enumerate(consensus_sorted, start=1):
            row["reward_consensus_rank_within_n"] = rank
            row["reward_uncertainty_flag"] = row["reward_std_all"] >= q3
    return rows


def entropy(values: list[int]) -> float:
    if not values:
        return 0.0
    counts = defaultdict(int)
    for value in values:
        counts[value] += 1
    total = len(values)
    return -sum((count / total) * math.log(count / total, 2) for count in counts.values())


def max_unvisited_gap_stats(order: list[int], n: int) -> tuple[float, float]:
    visited: set[int] = set()
    gaps: list[int] = []
    for track in order:
        visited.add(track)
        unvisited = sorted(set(range(n)) - visited)
        if not unvisited:
            gaps.append(0)
            continue
        current_gap = 1
        max_gap = 1
        for prev, curr in zip(unvisited, unvisited[1:]):
            if curr == prev + 1:
                current_gap += 1
                max_gap = max(max_gap, current_gap)
            else:
                current_gap = 1
        gaps.append(max_gap)
    return mean([g / n for g in gaps]), max([g / n for g in gaps], default=0.0)


def build_order_features(reward_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    for row in reward_rows:
        n = int(row["n"])
        order = json.loads(row["scan_order_json"])
        center = (n - 1) / 2.0
        jumps = [abs(b - a) for a, b in zip(order, order[1:])]
        signed_diffs = [b - a for a, b in zip(order, order[1:])]
        first_quarter_len = max(1, math.ceil(0.25 * n))
        early = order[:first_quarter_len]
        late = order[-first_quarter_len:]
        outer = set(range(0, max(1, n // 4))) | set(range(n - max(1, n // 4), n))
        center_start = n // 4
        center_end = n - n // 4
        center_tracks = set(range(center_start, center_end))
        parity_switches = sum(1 for a, b in zip(order, order[1:]) if (a % 2) != (b % 2))
        same_sign = sum(1 for a, b in zip(signed_diffs, signed_diffs[1:]) if a and b and (a > 0) == (b > 0))
        reversals = sum(1 for a, b in zip(signed_diffs, signed_diffs[1:]) if a and b and (a > 0) != (b > 0))
        max_gap_mean, max_gap_max = max_unvisited_gap_stats(order, n)
        family = row.get("strategy_family", "")
        name = row["strategy_name"].lower()
        feature_row = {
            "n": n,
            "strategy_name": row["strategy_name"],
            "strategy_family": family,
            "candidate_group": row.get("candidate_group", ""),
            "first_track": order[0],
            "last_track": order[-1],
            "center_track_index": center,
            "first_track_norm": safe_divide(order[0], n - 1),
            "last_track_norm": safe_divide(order[-1], n - 1),
            "mean_jump": mean(jumps),
            "median_jump": median(jumps),
            "max_jump": max(jumps, default=0),
            "min_jump": min(jumps, default=0),
            "std_jump": std([float(j) for j in jumps]),
            "total_jump": sum(jumps),
            "normalized_mean_jump": safe_divide(mean(jumps), n - 1),
            "normalized_max_jump": safe_divide(max(jumps, default=0), n - 1),
            "adjacent_jump_count": sum(1 for jump in jumps if jump == 1),
            "long_jump_count": sum(1 for jump in jumps if jump >= n / 2),
            "jump_entropy": entropy(jumps),
            "running_center_distance_mean": mean([abs(track - center) / max(1.0, center) for track in order]),
            "early_center_bias": mean([abs(track - center) / max(1.0, center) for track in early]),
            "late_center_bias": mean([abs(track - center) / max(1.0, center) for track in late]),
            "edge_early_count": sum(1 for track in early if track in outer),
            "center_early_count": sum(1 for track in early if track in center_tracks),
            "odd_even_transition_count": parity_switches,
            "parity_switch_rate": safe_divide(parity_switches, max(1, n - 1)),
            "monotonicity_fraction": safe_divide(same_sign, max(1, len(signed_diffs) - 1)),
            "direction_reversal_count": reversals,
            "max_unvisited_gap_proxy_mean_norm": max_gap_mean,
            "max_unvisited_gap_proxy_max_norm": max_gap_max,
            "is_raster_like": family == "raster" or "raster" in name,
            "is_center_out_like": family == "center_out" or "center_out" in name,
            "is_edge_in_like": family == "edge_in" or "edge_in" in name,
            "is_odd_even_like": family == "odd_even" or "odd_even" in name,
            "is_regular_jump_like": family == "regular_jump" or "regular_jump" in name,
            "is_block_interleaved_like": family == "block_interleaved" or "block_interleaved" in name,
        }
        features.append(feature_row)
    return features


def build_surrogate_table(reward_rows: list[dict[str, Any]], feature_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name = {row["strategy_name"]: row for row in feature_rows}
    merged: list[dict[str, Any]] = []
    for row in reward_rows:
        feature = by_name[row["strategy_name"]]
        output = {**row}
        for key, value in feature.items():
            if key not in output:
                output[key] = value
            else:
                output[f"feature_{key}"] = value
        output.update(
            {
                "target_u2_score_rank": row["u2_score_rank"],
                "target_peeq_score_rank": row["peeq_score_rank"],
                "target_surfaceT_score_rank": row["surfaceT_score_rank"],
                "target_reward_v01_u2_primary": row["reward_v01_u2_primary"],
                "target_reward_v02_safety_weighted": row["reward_v02_safety_weighted"],
                "target_reward_v04_penalized": row["reward_v04_penalized"],
                "target_reward_v05_lexicographic": row["reward_v05_lexicographic"],
                "target_reward_mean_all": row["reward_mean_all"],
            }
        )
        merged.append(output)
    return merged


def build_pairwise(reward_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for n in EXPECTED_N:
        group = sorted([row for row in reward_rows if row["n"] == n], key=lambda row: row["strategy_name"])
        for i, j in itertools.combinations(group, 2):
            if i["reward_mean_all"] > j["reward_mean_all"]:
                preferred_reward = "i"
            elif j["reward_mean_all"] > i["reward_mean_all"]:
                preferred_reward = "j"
            else:
                preferred_reward = "tie"
            if i["u2_rank_within_n"] < j["u2_rank_within_n"]:
                preferred_u2 = "i"
            elif j["u2_rank_within_n"] < i["u2_rank_within_n"]:
                preferred_u2 = "j"
            else:
                preferred_u2 = "tie"
            pairs.append(
                {
                    "case_i": i["strategy_name"],
                    "case_j": j["strategy_name"],
                    "n": n,
                    "strategy_i": i["strategy_name"],
                    "strategy_j": j["strategy_name"],
                    "order_i": i["scan_order_json"],
                    "order_j": j["scan_order_json"],
                    "reward_mean_all_i": i["reward_mean_all"],
                    "reward_mean_all_j": j["reward_mean_all"],
                    "u2_rank_i": i["u2_rank_within_n"],
                    "u2_rank_j": j["u2_rank_within_n"],
                    "peeq_rank_i": i["peeq_rank_within_n"],
                    "peeq_rank_j": j["peeq_rank_within_n"],
                    "surfaceT_rank_i": i["surfaceT_rank_within_n"],
                    "surfaceT_rank_j": j["surfaceT_rank_within_n"],
                    "preferred_by_reward_mean": preferred_reward,
                    "preferred_by_u2": preferred_u2,
                    "reward_margin": abs(i["reward_mean_all"] - j["reward_mean_all"]),
                    "u2_rank_margin": abs(i["u2_rank_within_n"] - j["u2_rank_within_n"]),
                    "peeq_rank_margin": abs(i["peeq_rank_within_n"] - j["peeq_rank_within_n"]),
                    "surfaceT_rank_margin": abs(i["surfaceT_rank_within_n"] - j["surfaceT_rank_within_n"]),
                }
            )
    return pairs


def build_splits(reward_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    names_by_n = {n: sorted(row["strategy_name"] for row in reward_rows if row["n"] == n) for n in EXPECTED_N}
    splits: dict[str, Any] = {"leave_N_out": {}, "core_generalization": {}, "larger_N_generalization": {}, "random_stratified_5fold": {}}
    for test_n in EXPECTED_N:
        train_n = [n for n in EXPECTED_N if n != test_n]
        splits["leave_N_out"][f"test_N{test_n}"] = {
            "train_n": train_n,
            "test_n": [test_n],
            "train_cases": [case for n in train_n for case in names_by_n[n]],
            "test_cases": names_by_n[test_n],
        }
    splits["core_generalization"]["train_N12_N16_N24_test_N40"] = {
        "train_n": [12, 16, 24],
        "test_n": [40],
        "train_cases": [case for n in [12, 16, 24] for case in names_by_n[n]],
        "test_cases": names_by_n[40],
    }
    splits["larger_N_generalization"]["train_N12_N16_test_N24_N40"] = {
        "train_n": [12, 16],
        "test_n": [24, 40],
        "train_cases": [case for n in [12, 16] for case in names_by_n[n]],
        "test_cases": [case for n in [24, 40] for case in names_by_n[n]],
    }
    rng = random.Random(310)
    fold_cases = {fold: [] for fold in range(5)}
    for n in EXPECTED_N:
        names = names_by_n[n][:]
        rng.shuffle(names)
        for idx, name in enumerate(names):
            fold_cases[idx % 5].append(name)
    for fold in range(5):
        test_cases = sorted(fold_cases[fold])
        train_cases = sorted(set(row["strategy_name"] for row in reward_rows) - set(test_cases))
        splits["random_stratified_5fold"][f"fold_{fold + 1}"] = {
            "train_cases": train_cases,
            "test_cases": test_cases,
            "test_count_by_n": {n: sum(1 for case in test_cases if case.startswith(f"N{n}_")) for n in EXPECTED_N},
        }
    summary_rows: list[dict[str, Any]] = []
    for split_family, definitions in splits.items():
        for split_name, spec in definitions.items():
            summary_rows.append(
                {
                    "split_family": split_family,
                    "split_name": split_name,
                    "train_case_count": len(spec["train_cases"]),
                    "test_case_count": len(spec["test_cases"]),
                    "train_n": json.dumps(spec.get("train_n", [])),
                    "test_n": json.dumps(spec.get("test_n", [])),
                }
            )
    return splits, summary_rows


def build_diagnostics(reward_rows: list[dict[str, Any]], feature_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    reward_matrix_cols = REWARD_COLUMNS + ["reward_mean_all"]
    for a, b in itertools.combinations(reward_matrix_cols, 2):
        diagnostics.append({"diagnostic_type": "reward_variant_spearman", "item_a": a, "item_b": b, "value": spearman([r[a] for r in reward_rows], [r[b] for r in reward_rows])})
    for metric in ["u2_range", "peeq_max", "surfaceT_proxy", "u2_rank_within_n", "peeq_rank_within_n", "surfaceT_rank_within_n"]:
        diagnostics.append({"diagnostic_type": "raw_metric_vs_reward_spearman", "item_a": metric, "item_b": "reward_mean_all", "value": spearman([r[metric] for r in reward_rows], [r["reward_mean_all"] for r in reward_rows])})
    for n in EXPECTED_N:
        group = sorted([row for row in reward_rows if row["n"] == n], key=lambda row: row["reward_mean_all"], reverse=True)
        for rank, row in enumerate(group[:5], start=1):
            diagnostics.append({"diagnostic_type": "top_reward_mean_all_per_n", "n": n, "rank": rank, "strategy_name": row["strategy_name"], "value": row["reward_mean_all"]})
        for reward_col in REWARD_COLUMNS:
            best = max(group, key=lambda row, col=reward_col: row[col])
            diagnostics.append({"diagnostic_type": "top_reward_variant_per_n", "n": n, "item_a": reward_col, "strategy_name": best["strategy_name"], "value": best[reward_col]})
    for row in sorted(reward_rows, key=lambda r: r["reward_std_all"], reverse=True)[:10]:
        diagnostics.append({"diagnostic_type": "high_reward_disagreement", "n": row["n"], "strategy_name": row["strategy_name"], "value": row["reward_std_all"]})
    for row in reward_rows:
        is_pareto = row["pareto_within_n_u2_peeq_surfaceT"] or row["pareto_overall_u2_peeq_surfaceT"]
        high_reward = row["reward_consensus_rank_within_n"] <= 5
        if is_pareto and not high_reward:
            diagnostics.append({"diagnostic_type": "pareto_not_high_reward", "n": row["n"], "strategy_name": row["strategy_name"], "value": row["reward_consensus_rank_within_n"]})
        if high_reward and not is_pareto:
            diagnostics.append({"diagnostic_type": "high_reward_not_pareto", "n": row["n"], "strategy_name": row["strategy_name"], "value": row["reward_consensus_rank_within_n"]})

    feature_by_name = {row["strategy_name"]: row for row in feature_rows}
    feature_candidates = [
        "first_track_norm",
        "last_track_norm",
        "normalized_mean_jump",
        "normalized_max_jump",
        "adjacent_jump_count",
        "long_jump_count",
        "jump_entropy",
        "running_center_distance_mean",
        "early_center_bias",
        "late_center_bias",
        "edge_early_count",
        "center_early_count",
        "parity_switch_rate",
        "monotonicity_fraction",
        "direction_reversal_count",
        "max_unvisited_gap_proxy_mean_norm",
        "max_unvisited_gap_proxy_max_norm",
    ]
    feature_corrs: list[dict[str, Any]] = []
    for feature in feature_candidates:
        xs = [parse_float(feature_by_name[row["strategy_name"]].get(feature)) for row in reward_rows]
        ys = [row["reward_mean_all"] for row in reward_rows]
        corr = spearman(xs, ys)
        feature_corrs.append({"feature_name": feature, "target": "reward_mean_all", "spearman": corr, "abs_spearman": abs(corr or 0.0)})
    feature_corrs.sort(key=lambda row: row["abs_spearman"], reverse=True)
    summary = {
        "strongest_reward_variant_correlation": max(
            [row for row in diagnostics if row["diagnostic_type"] == "reward_variant_spearman"],
            key=lambda row: abs(row["value"] or 0.0),
        ),
        "strongest_feature_reward_correlation": feature_corrs[0] if feature_corrs else None,
        "high_reward_disagreement_count": sum(1 for row in diagnostics if row["diagnostic_type"] == "high_reward_disagreement"),
        "pareto_not_high_reward_count": sum(1 for row in diagnostics if row["diagnostic_type"] == "pareto_not_high_reward"),
        "high_reward_not_pareto_count": sum(1 for row in diagnostics if row["diagnostic_type"] == "high_reward_not_pareto"),
    }
    return diagnostics, feature_corrs, summary


def maybe_plot(reward_rows: list[dict[str, Any]], diagnostics: list[dict[str, Any]], feature_corrs: list[dict[str, Any]]) -> list[str]:
    written: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        return [f"PLOTTING_SKIPPED: {exc}"]
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    by_n = {n: [row for row in reward_rows if row["n"] == n] for n in EXPECTED_N}
    plt.figure(figsize=(8, 4))
    plt.boxplot([[row["reward_mean_all"] for row in by_n[n]] for n in EXPECTED_N], tick_labels=[f"N{n}" for n in EXPECTED_N])
    plt.ylabel("reward_mean_all")
    plt.title("Reward Mean Distribution by N")
    path = FIGURE_DIR / "reward_mean_all_distribution_per_N.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    written.append(str(path))

    columns = REWARD_COLUMNS + ["reward_mean_all"]
    matrix = []
    for a in columns:
        matrix.append([spearman([row[a] for row in reward_rows], [row[b] for row in reward_rows]) or 0.0 for b in columns])
    plt.figure(figsize=(7, 6))
    plt.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(label="Spearman")
    plt.xticks(range(len(columns)), columns, rotation=70, ha="right", fontsize=7)
    plt.yticks(range(len(columns)), columns, fontsize=7)
    plt.title("Reward Variant Correlation")
    path = FIGURE_DIR / "reward_variant_correlation_heatmap.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    written.append(str(path))

    for rank_col, file_name in [
        ("u2_rank_within_n", "u2_rank_vs_reward_mean_all_per_N.png"),
        ("peeq_rank_within_n", "peeq_rank_vs_reward_mean_all_per_N.png"),
        ("surfaceT_rank_within_n", "surfaceT_rank_vs_reward_mean_all_per_N.png"),
    ]:
        plt.figure(figsize=(7, 4))
        for n in EXPECTED_N:
            rows = by_n[n]
            plt.scatter([row[rank_col] for row in rows], [row["reward_mean_all"] for row in rows], label=f"N{n}", s=28)
        plt.xlabel(rank_col)
        plt.ylabel("reward_mean_all")
        plt.legend()
        plt.title(f"{rank_col} vs reward_mean_all")
        path = FIGURE_DIR / file_name
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        written.append(str(path))

    top_features = feature_corrs[:10]
    plt.figure(figsize=(8, 4))
    plt.bar([row["feature_name"] for row in top_features], [row["spearman"] for row in top_features])
    plt.xticks(rotation=65, ha="right", fontsize=8)
    plt.ylabel("Spearman with reward_mean_all")
    plt.title("Top Feature-Reward Correlations")
    path = FIGURE_DIR / "top_feature_reward_correlations.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    written.append(str(path))
    return written


def write_claim_boundary(path_md: Path, path_json: Path) -> None:
    safe_claims = [
        "Run10 creates a within-N normalized reward and surrogate-pretraining dataset from 60/60 teacher-labelled variable-N cases.",
        "The dataset supports future variable-N surrogate, reward-model, and preference-model experiments.",
        "The reward is explicitly U2-primary, PEEQ-safety-aware, and SurfaceT-secondary.",
        "Pairwise preference data are generated within each N only.",
        "Cross-N generalization splits are defined but not yet evaluated.",
    ]
    unsafe_claims = [
        "Do not claim trained variable-N RL policy superiority.",
        "Do not claim surrogate accuracy.",
        "Do not claim arbitrary-N generalization.",
        "Do not claim a physical optimum.",
        "Do not claim fixed-32 absolute guard transfer.",
        "Do not claim reward V01-V05 are physically final; they are candidate reward formulations for later ablation.",
    ]
    md = ["# Run 10 Claim Boundary", "", "## Safe Claims", ""]
    md += [f"- {claim}" for claim in safe_claims]
    md += ["", "## Unsafe Claims", ""]
    md += [f"- {claim}" for claim in unsafe_claims]
    path_md.parent.mkdir(parents=True, exist_ok=True)
    path_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    write_json(path_json, {"safe_claims": safe_claims, "unsafe_claims": unsafe_claims, "verdict": "RUN10_DATASET_CONSTRUCTION_ONLY_NO_RL_POLICY_TRAINING"})


def write_report(
    validation: dict[str, Any],
    reward_rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    surrogate_rows: list[dict[str, Any]],
    pairwise_rows: list[dict[str, Any]],
    split_summary_rows: list[dict[str, Any]],
    diagnostics_summary: dict[str, Any],
    output_files: list[str],
) -> None:
    top_by_n = {
        f"N{n}": max([row for row in reward_rows if row["n"] == n], key=lambda row: row["reward_mean_all"])["strategy_name"]
        for n in EXPECTED_N
    }
    lines = [
        "# Stage 3 Run 10 - Variable-N Normalized Reward and Surrogate Dataset",
        "",
        "## Purpose",
        "Build a clean within-N normalized reward dataset, scan-order feature table, surrogate-pretraining table, and pairwise preference dataset from the 60 teacher-labelled variable-N probe cases.",
        "",
        "## Inputs",
        f"- `{RUN09_CANONICAL}`",
        f"- `{CANDIDATE_ORDERS}`",
        f"- `{RUN09_PARETO}`",
        f"- `{RUN09_CLAIM_BOUNDARY}`",
        f"- `{RUN08_TEACHER_LABELS}`",
        "",
        "## Validation Verdict",
        f"- `{validation['verdict']}`",
        f"- Total rows: {len(reward_rows)}",
        f"- Per-N counts: {validation['per_n_counts']}",
        "",
        "## Objective Hierarchy",
        "- Primary: U2 / warpage.",
        "- Safety: PEEQ.",
        "- Secondary diagnostic / tie-breaker: SurfaceT proxy.",
        "- All ranking and reward normalization are within N.",
        "",
        "## Normalization Strategy",
        "- Rank scores map best within-N rank to 1.0 and worst within-N rank to 0.0.",
        "- Min-max scores are computed within each N.",
        "- Z-score desirability negates within-N z scores because lower raw metrics are better.",
        "- Raw objective magnitudes are preserved but are not used as direct cross-N reward scales.",
        "",
        "## Reward Variants V01-V05",
        "- V01: U2-primary rank reward, 70/20/10 for U2/PEEQ/SurfaceT.",
        "- V02: safety-weighted rank reward, 60/30/10.",
        "- V03: SurfaceT-aware diagnostic reward, 55/25/20.",
        "- V04: constrained penalty reward with penalties for weak U2, PEEQ, or SurfaceT ranks.",
        "- V05: lexicographic constrained score sorted by U2, then PEEQ, then SurfaceT.",
        "",
        "## Canonical Reward Dataset Summary",
        f"- Rows: {len(reward_rows)}",
        f"- Top reward_mean_all per N: {top_by_n}",
        "",
        "## Scan-Order Feature Table Summary",
        f"- Rows: {len(feature_rows)}",
        "- Features include normalized jump, edge/center timing, parity, monotonicity, direction reversal, and unvisited-gap proxy summaries.",
        "",
        "## Surrogate-Pretraining Table Summary",
        f"- Rows: {len(surrogate_rows)}",
        "- Includes raw metrics, within-N normalized scores, reward targets, scan-order features, candidate labels, and Pareto flags.",
        "",
        "## Pairwise Preference Dataset Summary",
        f"- Rows: {len(pairwise_rows)}",
        "- Pairwise preferences are generated within each N only: 15 choose 2 per N, 420 total.",
        "",
        "## Dataset Split Definitions",
        f"- Split definitions: {len(split_summary_rows)} total split rows.",
        "- Includes leave-N-out, core N40 generalization, larger-N generalization, and 5 random stratified folds.",
        "",
        "## Reward Diagnostics",
        f"- Strongest reward variant correlation: {diagnostics_summary.get('strongest_reward_variant_correlation')}",
        f"- Strongest feature-reward correlation: {diagnostics_summary.get('strongest_feature_reward_correlation')}",
        f"- Pareto but not high-reward cases: {diagnostics_summary.get('pareto_not_high_reward_count')}",
        f"- High-reward but not Pareto cases: {diagnostics_summary.get('high_reward_not_pareto_count')}",
        "",
        "## Claim Boundary",
        "- Run10 constructs a dataset and candidate reward formulations only.",
        "- It does not train the final RL policy.",
        "- It does not prove surrogate accuracy or variable-N RL policy superiority.",
        "- It does not transfer fixed-32 absolute U2 guards to variable-N.",
        "",
        "## Output Files",
    ]
    lines += [f"- `{path}`" for path in output_files]
    lines += [
        "",
        "## Recommended Run11",
        "Build the first lightweight variable-N surrogate / reward model using the run10 table. Use leave-N-out validation and report whether simple feature-based models can predict within-N normalized reward/ranks. Do not train the final RL policy yet.",
    ]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_run_index(verdict: str) -> None:
    entry = (
        "| run_10 | Variable-N normalized reward surrogate dataset | Build within-N normalized reward targets, scan-order features, pairwise preferences, and split definitions from 60 teacher-labelled probe cases. | "
        "`scripts/stage3/run_10_build_variable_n_normalized_reward_dataset.py` | "
        "`docs/stage3/runs/run_10_variable_n_normalized_reward_surrogate_dataset/RUN_10_VARIABLE_N_NORMALIZED_REWARD_SURROGATE_DATASET_REPORT.md` | "
        "`outputs/stage3_run_10_variable_n_normalized_reward_surrogate_dataset/` | "
        f"`{verdict}` | No Abaqus, no datacheck, no ODB opening, no abqjobpilot, no RL policy training, no commit/push. Next: `run_11_variable_n_lightweight_surrogate_reward_model`. |"
    )
    if not RUN_INDEX_PATH.exists():
        return
    text = RUN_INDEX_PATH.read_text(encoding="utf-8")
    lines = text.splitlines()
    updated = False
    for idx, line in enumerate(lines):
        if line.startswith("| run_10 | Variable-N normalized reward surrogate dataset |"):
            lines[idx] = entry
            updated = True
            break
    if not updated:
        lines.append(entry)
    RUN_INDEX_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def git_branch() -> str:
    try:
        result = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception:
        return "UNKNOWN"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    canonical_rows = read_csv(RUN09_CANONICAL)
    metadata, metadata_warnings = load_metadata()
    validation = validate_inputs(canonical_rows, metadata)
    validation["warnings"].extend(metadata_warnings)
    validation_path = OUTPUT_DIR / "run10_input_validation_summary.json"
    write_json(validation_path, validation)
    if validation["verdict"].startswith("FAIL"):
        print(validation["verdict"])
        print(json.dumps(validation, indent=2))
        return 2

    reward_rows = build_reward_dataset(canonical_rows, metadata)
    feature_rows = build_order_features(reward_rows)
    surrogate_rows = build_surrogate_table(reward_rows, feature_rows)
    pairwise_rows = build_pairwise(reward_rows)
    splits, split_summary_rows = build_splits(reward_rows)
    diagnostics, feature_corrs, diagnostics_summary = build_diagnostics(reward_rows, feature_rows)
    plot_outputs = maybe_plot(reward_rows, diagnostics, feature_corrs)

    reward_csv = OUTPUT_DIR / "probe60_variable_n_reward_dataset.csv"
    reward_json = OUTPUT_DIR / "probe60_variable_n_reward_dataset.json"
    feature_csv = OUTPUT_DIR / "probe60_scan_order_features.csv"
    feature_json = OUTPUT_DIR / "probe60_scan_order_features.json"
    surrogate_csv = OUTPUT_DIR / "probe60_surrogate_pretraining_table.csv"
    surrogate_json = OUTPUT_DIR / "probe60_surrogate_pretraining_table.json"
    pairwise_csv = OUTPUT_DIR / "probe60_pairwise_preference_dataset.csv"
    pairwise_json = OUTPUT_DIR / "probe60_pairwise_preference_dataset.json"
    splits_json = OUTPUT_DIR / "probe60_dataset_splits.json"
    splits_summary_csv = OUTPUT_DIR / "probe60_dataset_splits_summary.csv"
    diagnostics_csv = OUTPUT_DIR / "reward_diagnostics_summary.csv"
    diagnostics_json = OUTPUT_DIR / "reward_diagnostics_summary.json"
    feature_corr_csv = OUTPUT_DIR / "feature_reward_correlation_summary.csv"
    claim_md = OUTPUT_DIR / "run10_claim_boundary.md"
    claim_json = OUTPUT_DIR / "run10_claim_boundary.json"

    write_csv(reward_csv, reward_rows)
    write_json(reward_json, reward_rows)
    write_csv(feature_csv, feature_rows)
    write_json(feature_json, feature_rows)
    write_csv(surrogate_csv, surrogate_rows)
    write_json(surrogate_json, surrogate_rows)
    write_csv(pairwise_csv, pairwise_rows)
    write_json(pairwise_json, pairwise_rows)
    write_json(splits_json, splits)
    write_csv(splits_summary_csv, split_summary_rows)
    write_csv(diagnostics_csv, diagnostics)
    write_json(diagnostics_json, {"diagnostics": diagnostics, "summary": diagnostics_summary})
    write_csv(feature_corr_csv, feature_corrs)
    write_claim_boundary(claim_md, claim_json)

    output_files = [
        str(validation_path),
        str(reward_csv),
        str(reward_json),
        str(feature_csv),
        str(feature_json),
        str(surrogate_csv),
        str(surrogate_json),
        str(pairwise_csv),
        str(pairwise_json),
        str(splits_json),
        str(splits_summary_csv),
        str(diagnostics_csv),
        str(diagnostics_json),
        str(feature_corr_csv),
        str(claim_md),
        str(claim_json),
        *[path for path in plot_outputs if not path.startswith("PLOTTING_SKIPPED")],
        str(REPORT_PATH),
    ]
    write_report(validation, reward_rows, feature_rows, surrogate_rows, pairwise_rows, split_summary_rows, diagnostics_summary, output_files)
    update_run_index(validation["verdict"])

    manifest = {
        "run_id": RUN_ID,
        "run_name": RUN_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "branch": git_branch(),
        "script_path": str(Path(__file__).resolve()),
        "input_files": [
            str(RUN09_CANONICAL),
            str(RUN09_FAMILY_SUMMARY),
            str(RUN09_GROUP_COMPARISON),
            str(RUN09_PARETO),
            str(RUN09_CLAIM_BOUNDARY),
            str(RUN09_REPORT),
            str(RUN08_TEACHER_LABELS),
            str(CANDIDATE_ORDERS),
        ],
        "output_files": output_files,
        "validation_verdict": validation["verdict"],
        "total_rows": len(reward_rows),
        "per_n_counts": validation["per_n_counts"],
        "pairwise_row_count": len(pairwise_rows),
        "split_definitions_path": str(splits_json),
        "report_path": str(REPORT_PATH),
        "claim_boundary_path": str(claim_md),
        "reward_variants": REWARD_COLUMNS,
        "plot_outputs": plot_outputs,
        "no_solver_run": True,
        "no_odb_opened": True,
        "no_abqjobpilot_run": True,
        "no_rl_policy_training": True,
        "no_commit_or_push": True,
    }
    write_json(MANIFEST_PATH, manifest)

    top_reward_by_n = {
        f"N{n}": max([row for row in reward_rows if row["n"] == n], key=lambda row: row["reward_mean_all"])["strategy_name"]
        for n in EXPECTED_N
    }
    print(validation["verdict"])
    print(f"rows={len(reward_rows)}")
    print(f"per_n_counts={validation['per_n_counts']}")
    print(f"pairwise_rows={len(pairwise_rows)}")
    print(f"top_reward_mean_all={top_reward_by_n}")
    print(f"surrogate_table={surrogate_csv}")
    print(f"report={REPORT_PATH}")
    print(f"manifest={MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
