from __future__ import annotations

import csv
import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
RUN_ID = "run_09_variable_n_probe60_teacher_ranking_analysis"
RUN_NAME = "variable-N probe60 teacher-label ranking analysis"
INPUT_CSV = ROOT / "outputs" / "stage3_run_08_probe60_odb_teacher_validation" / "probe60_odb_teacher_labels.csv"
INPUT_JSON = ROOT / "outputs" / "stage3_run_08_probe60_odb_teacher_validation" / "probe60_odb_teacher_labels.json"
RUN08_SUMMARY = ROOT / "outputs" / "stage3_run_08_probe60_odb_teacher_validation" / "probe60_odb_teacher_validation_summary.json"
RUN08_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_08_variable_n_probe60_odb_teacher_validation" / "RUN_08_VARIABLE_N_PROBE60_ODB_TEACHER_VALIDATION_REPORT.md"
RUN08_MANIFEST = ROOT / "artifacts" / "manifests" / "stage3_run_08_manifest.json"
CANDIDATE_CSV = ROOT / "outputs" / "stage3_run_06_variable_n_probe60_candidate_order_generation" / "variable_N_probe60_candidate_orders.csv"
OUTPUT_DIR = ROOT / "outputs" / "stage3_run_09_variable_n_probe60_teacher_ranking_analysis"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = ROOT / "docs" / "stage3" / "runs" / RUN_ID
REPORT_PATH = REPORT_DIR / "RUN_09_VARIABLE_N_PROBE60_TEACHER_RANKING_ANALYSIS_REPORT.md"
MANIFEST_PATH = ROOT / "artifacts" / "manifests" / "stage3_run_09_manifest.json"

EXPECTED_N = [12, 16, 24, 40]
EXPECTED_ROWS_PER_N = 15
EXPECTED_TOTAL = 60


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


def parse_n(value: str) -> int:
    text = str(value).strip()
    if text.upper().startswith("N"):
        text = text[1:]
    return int(text)


def parse_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def find_column(headers: list[str], required_terms: list[str], fallback_terms: list[str] | None = None) -> str | None:
    lowered = {h.lower(): h for h in headers}
    for header in headers:
        h = header.lower()
        if all(term.lower() in h for term in required_terms):
            return header
    if fallback_terms:
        for header in headers:
            h = header.lower()
            if any(term.lower() in h for term in fallback_terms):
                return header
    return None


def canonical_columns(headers: list[str]) -> dict[str, str]:
    mapping = {
        "n": "N" if "N" in headers else ("n" if "n" in headers else ""),
        "strategy_name": "case_id" if "case_id" in headers else ("strategy_name" if "strategy_name" in headers else ""),
        "job_name": "job_name" if "job_name" in headers else "",
        "teacher_status": "odb_extraction_status" if "odb_extraction_status" in headers else ("extraction_status" if "extraction_status" in headers else ""),
        "u2_range": "u2_range" if "u2_range" in headers else "",
        "peeq_max": "peeq_max" if "peeq_max" in headers else "",
        "surfaceT_proxy": "",
    }
    surface_candidates = [
        h for h in headers if any(t in h.lower() for t in ["surfacet", "surface_t", "surface", "residual", "stress", "proxy"])
    ]
    preferred = [
        h for h in surface_candidates
        if "surface" in h.lower() and "proxy" in h.lower() and ("mpa" in h.lower() or "pa" in h.lower())
    ]
    mapping["surfaceT_proxy"] = (preferred or surface_candidates or [""])[0]
    missing = [key for key in ["n", "strategy_name", "u2_range", "peeq_max", "surfaceT_proxy"] if not mapping[key]]
    if missing:
        raise RuntimeError(f"Missing required canonical columns: {missing}; headers={headers}")
    return mapping


def rank_ascending(values: list[float]) -> list[int]:
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    ranks = [0] * len(values)
    for rank, idx in enumerate(order, start=1):
        ranks[idx] = rank
    return ranks


def z_scores(values: list[float]) -> list[float]:
    mean = statistics.fmean(values)
    std = statistics.pstdev(values)
    if std == 0:
        return [0.0 for _ in values]
    return [(v - mean) / std for v in values]


def percentile(rank: int, count: int) -> float:
    return (rank - 1) / max(1, count - 1)


def parse_strategy(strategy_name: str, fallback_family: str = "") -> dict[str, str]:
    parts = strategy_name.split("_")
    n_prefix = parts[0] if parts else ""
    action_id = parts[1] if len(parts) > 1 and parts[1].startswith("A") else ""
    core = "_".join(parts[2:]) if action_id else "_".join(parts[1:])
    family = fallback_family or infer_family(core)
    return {"n_prefix": n_prefix, "action_id": action_id, "strategy_core_name": core, "strategy_family": family}


def infer_family(core: str) -> str:
    text = core.lower()
    if "raster" in text:
        return "raster"
    if "odd_even" in text:
        return "odd_even"
    if "maximin" in text:
        return "maximin"
    if "method_c" in text:
        return "method_c"
    if "center_edge" in text:
        return "center_edge"
    if "center_out" in text:
        return "center_out"
    if "edge_in" in text:
        return "edge_in"
    if "regular_jump" in text:
        return "regular_jump"
    if "block_interleaved" in text:
        return "block_interleaved"
    if "graph_pointer" in text:
        return "graph_pointer_proxy"
    return "unknown"


def load_candidate_metadata() -> tuple[dict[str, dict[str, str]], str]:
    if not CANDIDATE_CSV.exists():
        return {}, "WARNING_RUN09_CANDIDATE_GROUP_METADATA_INCOMPLETE"
    rows = read_csv(CANDIDATE_CSV)
    by_strategy = {row.get("strategy_name", ""): row for row in rows}
    return by_strategy, "PASS_RUN09_CANDIDATE_GROUP_METADATA_READY"


def validate_inputs(rows: list[dict[str, Any]], mapping: dict[str, str]) -> dict[str, Any]:
    counts = defaultdict(int)
    statuses = []
    names_by_n = defaultdict(list)
    missing = {"u2_range": 0, "peeq_max": 0, "surfaceT_proxy": 0}
    for row in rows:
        n = parse_n(row[mapping["n"]])
        counts[n] += 1
        names_by_n[n].append(row[mapping["strategy_name"]])
        status_col = mapping.get("teacher_status")
        statuses.append(row.get(status_col, "") if status_col else "")
        for key in missing:
            if parse_float(row[mapping[key]]) is None:
                missing[key] += 1
    duplicates = {
        n: sorted(name for name in set(names) if names.count(name) > 1)
        for n, names in names_by_n.items()
        if any(names.count(name) > 1 for name in names)
    }
    valid_statuses = [s for s in statuses if "PASS" in str(s).upper() or "EXTRACTED" in str(s).upper()]
    n24_a07 = [
        row for row in rows
        if parse_n(row[mapping["n"]]) == 24 and "N24_A07_regular_jump_coprime" == row[mapping["strategy_name"]]
    ]
    verdict = "PASS_RUN09_INPUT_TEACHER_LABELS_60_OF_60_READY"
    errors: list[str] = []
    if len(rows) != EXPECTED_TOTAL:
        errors.append("total_rows_not_60")
    if sorted(counts) != EXPECTED_N:
        errors.append("n_values_not_exact")
    if any(counts[n] != EXPECTED_ROWS_PER_N for n in EXPECTED_N):
        errors.append("per_n_count_not_15")
    if len(valid_statuses) != EXPECTED_TOTAL:
        errors.append("not_all_teacher_valid")
    if any(missing.values()):
        errors.append("missing_required_metric")
    if duplicates:
        errors.append("duplicate_strategy_within_n")
    if not n24_a07 or "PASS" not in str(n24_a07[0].get(mapping.get("teacher_status", ""), "")).upper():
        errors.append("n24_a07_missing_or_invalid")
    if errors:
        verdict = "FAIL_RUN09_INPUT_TEACHER_LABELS_NOT_READY"
    return {
        "verdict": verdict,
        "errors": errors,
        "total_rows": len(rows),
        "per_n_counts": dict(sorted(counts.items())),
        "all_teacher_valid": len(valid_statuses) == EXPECTED_TOTAL,
        "missing_metric_counts": missing,
        "duplicates_within_n": duplicates,
        "n24_a07_regular_jump_coprime_valid": bool(n24_a07 and "PASS" in str(n24_a07[0].get(mapping.get("teacher_status", ""), "")).upper()),
        "canonical_column_mapping": mapping,
        "surfaceT_direction_assumption": "lower is better",
    }


def build_canonical(rows: list[dict[str, Any]], mapping: dict[str, str], metadata: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    canonical: list[dict[str, Any]] = []
    for row in rows:
        strategy_name = row[mapping["strategy_name"]]
        meta = metadata.get(strategy_name, {})
        parsed = parse_strategy(strategy_name, meta.get("family", ""))
        raw_columns = {f"raw_{key}": value for key, value in row.items()}
        group = "unknown"
        if meta:
            if meta.get("is_engineering_baseline", "").lower() == "true":
                group = "engineering_baseline"
            elif meta.get("is_learned_or_proxy", "").lower() == "true" or "proxy" in meta.get("policy_source", ""):
                group = "proxy_fallback_policy"
        canonical.append(
            {
                **raw_columns,
                "n": parse_n(row[mapping["n"]]),
                "strategy_name": strategy_name,
                "job_name_canonical": row.get(mapping.get("job_name", ""), ""),
                "teacher_status_canonical": row.get(mapping.get("teacher_status", ""), ""),
                "u2_range_canonical": parse_float(row[mapping["u2_range"]]),
                "peeq_max_canonical": parse_float(row[mapping["peeq_max"]]),
                "surfaceT_proxy_canonical": parse_float(row[mapping["surfaceT_proxy"]]),
                "surfaceT_proxy_source_column": mapping["surfaceT_proxy"],
                **parsed,
                "candidate_group": group,
                "policy_source": meta.get("policy_source", "metadata_missing"),
                "trained_policy_used": meta.get("trained_policy_used", "False"),
                "teacher_validated_from_run06": meta.get("teacher_validated", "False"),
                "order_json": meta.get("order_json", ""),
            }
        )
    for n in EXPECTED_N:
        group_rows = [row for row in canonical if row["n"] == n]
        for metric, rank_col, pct_col, z_col in [
            ("u2_range_canonical", "u2_rank_within_n", "u2_percentile_within_n", "u2_z_within_n"),
            ("peeq_max_canonical", "peeq_rank_within_n", "peeq_percentile_within_n", "peeq_z_within_n"),
            ("surfaceT_proxy_canonical", "surfaceT_rank_within_n", "surfaceT_percentile_within_n", "surfaceT_z_within_n"),
        ]:
            values = [float(row[metric]) for row in group_rows]
            ranks = rank_ascending(values)
            zs = z_scores(values)
            for row, rank, z in zip(group_rows, ranks, zs):
                row[rank_col] = rank
                row[pct_col] = percentile(rank, len(group_rows))
                row[z_col] = z
        for row in group_rows:
            row["simple_mean_rank"] = statistics.fmean([row["u2_rank_within_n"], row["peeq_rank_within_n"], row["surfaceT_rank_within_n"]])
            row["constrained_rank_key"] = f"{row['u2_rank_within_n']:02d}-{row['peeq_rank_within_n']:02d}-{row['surfaceT_rank_within_n']:02d}"
            row["u2_peeq_feasible_relative"] = row["u2_rank_within_n"] <= 8 and row["peeq_rank_within_n"] <= 8
            row["u2_surfaceT_tradeoff_flag"] = tradeoff_flag(row)
        sorted_rows = sorted(group_rows, key=lambda r: (r["u2_rank_within_n"], r["peeq_rank_within_n"], r["surfaceT_rank_within_n"]))
        for idx, row in enumerate(sorted_rows, start=1):
            row["constrained_rank_within_n"] = idx
    return sorted(canonical, key=lambda r: (r["n"], r["strategy_name"]))


def tradeoff_flag(row: dict[str, Any]) -> str:
    if row["u2_rank_within_n"] <= 5 and row["surfaceT_rank_within_n"] > 8:
        return "good_u2_weak_surfaceT"
    if row["surfaceT_rank_within_n"] <= 5 and row["u2_rank_within_n"] > 8:
        return "good_surfaceT_weak_u2"
    return "none"


def leaderboard_rows(canonical: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs = [
        ("top5_u2", lambda r: (r["u2_rank_within_n"], r["peeq_rank_within_n"], r["surfaceT_rank_within_n"]), 5, False),
        ("top5_peeq", lambda r: (r["peeq_rank_within_n"], r["u2_rank_within_n"], r["surfaceT_rank_within_n"]), 5, False),
        ("top5_surfaceT", lambda r: (r["surfaceT_rank_within_n"], r["u2_rank_within_n"], r["peeq_rank_within_n"]), 5, False),
        ("top5_simple_mean_rank", lambda r: (r["simple_mean_rank"], r["u2_rank_within_n"]), 5, False),
        ("top5_constrained_rank_key", lambda r: (r["constrained_rank_within_n"],), 5, False),
        ("worst3_u2", lambda r: (r["u2_rank_within_n"],), 3, True),
        ("worst3_peeq", lambda r: (r["peeq_rank_within_n"],), 3, True),
        ("worst3_surfaceT", lambda r: (r["surfaceT_rank_within_n"],), 3, True),
    ]
    for n in EXPECTED_N:
        subset = [r for r in canonical if r["n"] == n]
        for board, key_fn, limit, reverse in specs:
            selected = sorted(subset, key=key_fn, reverse=reverse)[:limit]
            for pos, row in enumerate(selected, start=1):
                rows.append(
                    {
                        "n": n,
                        "leaderboard": board,
                        "position": pos,
                        "strategy_name": row["strategy_name"],
                        "strategy_family": row["strategy_family"],
                        "candidate_group": row["candidate_group"],
                        "u2_range": row["u2_range_canonical"],
                        "peeq_max": row["peeq_max_canonical"],
                        "surfaceT_proxy": row["surfaceT_proxy_canonical"],
                        "u2_rank_within_n": row["u2_rank_within_n"],
                        "peeq_rank_within_n": row["peeq_rank_within_n"],
                        "surfaceT_rank_within_n": row["surfaceT_rank_within_n"],
                        "simple_mean_rank": row["simple_mean_rank"],
                        "constrained_rank_within_n": row["constrained_rank_within_n"],
                    }
                )
    return rows


def summarize_leaderboards(rows: list[dict[str, Any]], path: Path) -> None:
    lines = ["# Per-N Leaderboards Summary", ""]
    for n in EXPECTED_N:
        lines.extend([f"## N={n}", ""])
        for board in ["top5_u2", "top5_peeq", "top5_surfaceT", "top5_constrained_rank_key"]:
            lines.append(f"### {board}")
            for row in [r for r in rows if r["n"] == n and r["leaderboard"] == board]:
                lines.append(f"- {row['position']}. `{row['strategy_name']}` ({row['strategy_family']})")
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def family_summary(canonical: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_family = defaultdict(list)
    for row in canonical:
        by_family[row["strategy_family"]].append(row)
    for family, items in sorted(by_family.items()):
        core_to_ns = defaultdict(set)
        for row in items:
            core_to_ns[row["strategy_core_name"]].add(row["n"])
        rows.append(
            {
                "strategy_family": family,
                "count": len(items),
                "mean_u2_rank_within_n": statistics.fmean(r["u2_rank_within_n"] for r in items),
                "median_u2_rank_within_n": statistics.median(r["u2_rank_within_n"] for r in items),
                "mean_peeq_rank_within_n": statistics.fmean(r["peeq_rank_within_n"] for r in items),
                "median_peeq_rank_within_n": statistics.median(r["peeq_rank_within_n"] for r in items),
                "mean_surfaceT_rank_within_n": statistics.fmean(r["surfaceT_rank_within_n"] for r in items),
                "median_surfaceT_rank_within_n": statistics.median(r["surfaceT_rank_within_n"] for r in items),
                "mean_simple_mean_rank": statistics.fmean(r["simple_mean_rank"] for r in items),
                "top3_u2_count": sum(1 for r in items if r["u2_rank_within_n"] <= 3),
                "top5_u2_count": sum(1 for r in items if r["u2_rank_within_n"] <= 5),
                "top3_peeq_count": sum(1 for r in items if r["peeq_rank_within_n"] <= 3),
                "top5_peeq_count": sum(1 for r in items if r["peeq_rank_within_n"] <= 5),
                "top3_surfaceT_count": sum(1 for r in items if r["surfaceT_rank_within_n"] <= 3),
                "top5_surfaceT_count": sum(1 for r in items if r["surfaceT_rank_within_n"] <= 5),
                "stability_across_n_core_count": sum(1 for ns in core_to_ns.values() if len(ns) >= 3),
            }
        )
    return sorted(rows, key=lambda r: r["mean_simple_mean_rank"])


def group_comparison(canonical: list[dict[str, Any]], metadata_status: str) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for row in canonical:
        groups[row["candidate_group"]].append(row)
    rows: list[dict[str, Any]] = []
    for group, items in sorted(groups.items()):
        best_per_n = {}
        worst_per_n = {}
        for n in EXPECTED_N:
            subset = [r for r in items if r["n"] == n]
            if subset:
                best_per_n[str(n)] = min(subset, key=lambda r: r["u2_rank_within_n"])["strategy_name"]
                worst_per_n[str(n)] = max(subset, key=lambda r: r["u2_rank_within_n"])["strategy_name"]
        rows.append(
            {
                "candidate_group": group,
                "metadata_status": metadata_status,
                "count": len(items),
                "mean_u2_rank_within_n": statistics.fmean(r["u2_rank_within_n"] for r in items),
                "median_u2_rank_within_n": statistics.median(r["u2_rank_within_n"] for r in items),
                "mean_peeq_rank_within_n": statistics.fmean(r["peeq_rank_within_n"] for r in items),
                "median_peeq_rank_within_n": statistics.median(r["peeq_rank_within_n"] for r in items),
                "mean_surfaceT_rank_within_n": statistics.fmean(r["surfaceT_rank_within_n"] for r in items),
                "median_surfaceT_rank_within_n": statistics.median(r["surfaceT_rank_within_n"] for r in items),
                "top5_u2_hit_count": sum(1 for r in items if r["u2_rank_within_n"] <= 5),
                "top5_peeq_hit_count": sum(1 for r in items if r["peeq_rank_within_n"] <= 5),
                "top5_surfaceT_hit_count": sum(1 for r in items if r["surfaceT_rank_within_n"] <= 5),
                "best_case_per_n_json": json.dumps(best_per_n, sort_keys=True),
                "worst_case_per_n_json": json.dumps(worst_per_n, sort_keys=True),
            }
        )
    return sorted(rows, key=lambda r: r["mean_u2_rank_within_n"])


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(rank_ascending(xs), rank_ascending(ys))


def pareto_flags(rows: list[dict[str, Any]], metrics: list[str]) -> set[str]:
    front = set()
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            no_worse = all(other[m] <= row[m] for m in metrics)
            better = any(other[m] < row[m] for m in metrics)
            if no_worse and better:
                dominated = True
                break
        if not dominated:
            front.add(row["strategy_name"])
    return front


def metric_interactions(canonical: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary: list[dict[str, Any]] = []
    pareto_rows: list[dict[str, Any]] = []
    scopes = [("overall", canonical)] + [(f"N{n}", [r for r in canonical if r["n"] == n]) for n in EXPECTED_N]
    for scope, rows in scopes:
        top_u2 = {r["strategy_name"] for r in rows if r["u2_rank_within_n"] <= 5}
        top_peeq = {r["strategy_name"] for r in rows if r["peeq_rank_within_n"] <= 5}
        top_surface = {r["strategy_name"] for r in rows if r["surfaceT_rank_within_n"] <= 5}
        p2 = pareto_flags(rows, ["u2_range_canonical", "peeq_max_canonical"])
        p3 = pareto_flags(rows, ["u2_range_canonical", "peeq_max_canonical", "surfaceT_proxy_canonical"])
        summary.append(
            {
                "scope": scope,
                "spearman_u2_peeq": spearman([r["u2_range_canonical"] for r in rows], [r["peeq_max_canonical"] for r in rows]),
                "spearman_u2_surfaceT": spearman([r["u2_range_canonical"] for r in rows], [r["surfaceT_proxy_canonical"] for r in rows]),
                "spearman_peeq_surfaceT": spearman([r["peeq_max_canonical"] for r in rows], [r["surfaceT_proxy_canonical"] for r in rows]),
                "top5_overlap_u2_peeq": len(top_u2 & top_peeq),
                "top5_overlap_u2_surfaceT": len(top_u2 & top_surface),
                "top5_overlap_peeq_surfaceT": len(top_peeq & top_surface),
                "top5_u2_bottom_half_surfaceT": ";".join(sorted(r["strategy_name"] for r in rows if r["u2_rank_within_n"] <= 5 and r["surfaceT_rank_within_n"] > 8)),
                "top5_surfaceT_bottom_half_u2": ";".join(sorted(r["strategy_name"] for r in rows if r["surfaceT_rank_within_n"] <= 5 and r["u2_rank_within_n"] > 8)),
                "pareto_2obj_count": len(p2),
                "pareto_3obj_count": len(p3),
            }
        )
        for row in rows:
            if row["strategy_name"] in p2 or row["strategy_name"] in p3:
                pareto_rows.append(
                    {
                        "scope": scope,
                        "strategy_name": row["strategy_name"],
                        "n": row["n"],
                        "strategy_family": row["strategy_family"],
                        "pareto_u2_peeq": row["strategy_name"] in p2,
                        "pareto_u2_peeq_surfaceT": row["strategy_name"] in p3,
                        "u2_rank_within_n": row["u2_rank_within_n"],
                        "peeq_rank_within_n": row["peeq_rank_within_n"],
                        "surfaceT_rank_within_n": row["surfaceT_rank_within_n"],
                    }
                )
    return summary, pareto_rows


def claim_boundary(canonical: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    safe_claims = [
        "True variable-N teacher validation completed for N=12/16/24/40 with 60/60 ODB-extracted labels.",
        "Within-N ranking is now possible for U2, PEEQ, and SurfaceT proxy.",
        "Cross-N diagnostic comparison is now possible using ranks, percentiles, and normalized scores.",
        "N-specific ranking is required because raw objective magnitudes are N-dependent.",
        "The run provides a teacher-labelled variable-N benchmark dataset for later variable-N policy training/evaluation.",
    ]
    unsafe_claims = [
        "Do not claim trained variable-N RL policy superiority.",
        "Do not claim arbitrary-N generalization.",
        "Do not claim a physical optimum.",
        "Do not claim fixed-32 U2 guard transfer to variable-N.",
        "Do not claim SurfaceT optimization outside U2/PEEQ feasible or near-feasible regions unless supported.",
        "Do not claim proxy/fallback policy is equivalent to trained RL.",
    ]
    payload = {
        "verdict": "PASS_RUN09_CLAIM_BOUNDARY_READY",
        "safe_claims": safe_claims,
        "unsafe_claims": unsafe_claims,
    }
    md = "# Run 09 Claim Boundary\n\n## Safe Claims\n\n" + "\n".join(f"- {x}" for x in safe_claims)
    md += "\n\n## Unsafe Claims\n\n" + "\n".join(f"- {x}" for x in unsafe_claims) + "\n"
    return md, payload


def make_plots(canonical: list[dict[str, Any]], family_rows: list[dict[str, Any]]) -> tuple[list[str], str]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        return [], f"plotting skipped: {exc}"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    for y_metric, y_label, filename in [
        ("peeq_max_canonical", "PEEQ max", "per_N_u2_vs_peeq.png"),
        ("surfaceT_proxy_canonical", "SurfaceT proxy", "per_N_u2_vs_surfaceT.png"),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(9, 7))
        for ax, n in zip(axes.ravel(), EXPECTED_N):
            rows = [r for r in canonical if r["n"] == n]
            ax.scatter([r["u2_range_canonical"] for r in rows], [r[y_metric] for r in rows], s=24)
            ax.set_title(f"N={n}")
            ax.set_xlabel("U2 range")
            ax.set_ylabel(y_label)
        fig.tight_layout()
        path = FIGURE_DIR / filename
        fig.savefig(path, dpi=140)
        plt.close(fig)
        outputs.append(str(path))
    for n in EXPECTED_N:
        rows = sorted([r for r in canonical if r["n"] == n], key=lambda r: r["u2_rank_within_n"])[:5]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar([r["action_id"] for r in rows], [r["u2_range_canonical"] for r in rows])
        ax.set_title(f"N={n} Top 5 U2")
        ax.set_ylabel("U2 range")
        path = FIGURE_DIR / f"N{n}_top5_u2.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        outputs.append(str(path))
    fig, ax = plt.subplots(figsize=(9, 4))
    rows = sorted(family_rows, key=lambda r: r["mean_simple_mean_rank"])
    ax.bar([r["strategy_family"] for r in rows], [r["mean_simple_mean_rank"] for r in rows])
    ax.set_ylabel("Mean rank")
    ax.tick_params(axis="x", rotation=45)
    path = FIGURE_DIR / "family_mean_rank_comparison.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    outputs.append(str(path))
    return outputs, "plots written"


def best_by(canonical: list[dict[str, Any]], rank_col: str) -> dict[str, str]:
    return {
        f"N{n}": min([r for r in canonical if r["n"] == n], key=lambda r: r[rank_col])["strategy_name"]
        for n in EXPECTED_N
    }


def write_report(
    validation: dict[str, Any],
    canonical: list[dict[str, Any]],
    leaderboards: list[dict[str, Any]],
    family_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    pareto_rows: list[dict[str, Any]],
    claim_payload: dict[str, Any],
    outputs: list[str],
    plotting_status: str,
) -> None:
    best_u2 = best_by(canonical, "u2_rank_within_n")
    best_peeq = best_by(canonical, "peeq_rank_within_n")
    best_surface = best_by(canonical, "surfaceT_rank_within_n")
    best_constrained = best_by(canonical, "constrained_rank_within_n")
    top_family = family_rows[0] if family_rows else {}
    group_headline = "; ".join(
        f"{r['candidate_group']}: mean U2 rank {float(r['mean_u2_rank_within_n']):.2f}"
        for r in group_rows
    )
    metric_overall = next((r for r in metric_rows if r["scope"] == "overall"), {})
    lines = [
        "# Stage 3 Run 09 — Variable-N Probe60 Teacher Ranking Analysis",
        "",
        "## Purpose",
        "Analyze the completed run_08 teacher labels using within-N rankings for U2, PEEQ, and SurfaceT proxy.",
        "",
        "## Inputs",
        f"- `{INPUT_CSV}`",
        f"- `{RUN08_SUMMARY}`",
        f"- `{CANDIDATE_CSV}`",
        "",
        "## Validation Status",
        f"- `{validation['verdict']}`",
        f"- Total rows: `{validation['total_rows']}`",
        f"- Per-N counts: `{validation['per_n_counts']}`",
        "",
        "## Objective Hierarchy",
        "U2 / warpage is primary. PEEQ is safety/plasticity. SurfaceT proxy is a secondary residual-stress diagnostic. Lower values are treated as better for all three metrics in this run.",
        "",
        "## Per-N Ranking Results",
        f"- Best U2 per N: `{best_u2}`",
        f"- Best PEEQ per N: `{best_peeq}`",
        f"- Best SurfaceT proxy per N: `{best_surface}`",
        "",
        "## Best Candidates Per N",
        f"- Best constrained-rank per N: `{best_constrained}`",
        "",
        "## Strategy Family Analysis",
        f"- Best mean simple-rank family: `{top_family.get('strategy_family', 'n/a')}` with mean rank `{float(top_family.get('mean_simple_mean_rank', 0)):.3f}`" if top_family else "- No family rows.",
        "",
        "## Candidate Group Comparison",
        group_headline or "Candidate group metadata unavailable.",
        "",
        "## Metric Interaction and Tradeoff Analysis",
        f"- Overall Spearman U2/PEEQ: `{float(metric_overall.get('spearman_u2_peeq', 0)):.3f}`",
        f"- Overall Spearman U2/SurfaceT: `{float(metric_overall.get('spearman_u2_surfaceT', 0)):.3f}`",
        f"- Overall top-5 U2/PEEQ overlap: `{metric_overall.get('top5_overlap_u2_peeq', 'n/a')}`",
        "",
        "## Pareto Front Summary",
        f"- Pareto rows written: `{len(pareto_rows)}`",
        "",
        "## N24_A07 Resolution Note",
        "N24_A07_regular_jump_coprime was previously incomplete during solver completion audit but was rerun/reprocessed successfully and is included as a valid teacher-labelled case in run08/run09.",
        "",
        "## Safe Claims",
        *[f"- {x}" for x in claim_payload["safe_claims"]],
        "",
        "## Claim Boundaries",
        *[f"- {x}" for x in claim_payload["unsafe_claims"]],
        "",
        "## Outputs",
        *[f"- `{x}`" for x in outputs],
        "",
        f"Plotting status: `{plotting_status}`",
        "",
        "## Recommended Next Step",
        "Stage 3 run_10 should use the 60 teacher-labelled variable-N cases to build the first variable-N surrogate / normalized reward dataset, using within-N normalized ranks rather than raw cross-N objective magnitudes.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def git_branch() -> str:
    try:
        return subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = read_csv(INPUT_CSV)
    headers = list(rows[0].keys()) if rows else []
    mapping = canonical_columns(headers)
    validation = validate_inputs(rows, mapping)
    validation_path = OUTPUT_DIR / "run09_input_validation_summary.json"
    write_json(validation_path, validation)
    if not validation["verdict"].startswith("PASS"):
        raise SystemExit(f"Input validation failed: {validation}")
    metadata, metadata_status = load_candidate_metadata()
    canonical = build_canonical(rows, mapping, metadata)
    canonical_csv = OUTPUT_DIR / "probe60_teacher_ranked_canonical.csv"
    canonical_json = OUTPUT_DIR / "probe60_teacher_ranked_canonical.json"
    write_csv(canonical_csv, canonical)
    write_json(canonical_json, canonical)
    leaderboards = leaderboard_rows(canonical)
    leader_csv = OUTPUT_DIR / "per_N_top_bottom_leaderboards.csv"
    leader_json = OUTPUT_DIR / "per_N_top_bottom_leaderboards.json"
    leader_md = OUTPUT_DIR / "per_N_leaderboards_summary.md"
    write_csv(leader_csv, leaderboards)
    write_json(leader_json, leaderboards)
    summarize_leaderboards(leaderboards, leader_md)
    family_rows = family_summary(canonical)
    family_csv = OUTPUT_DIR / "strategy_family_summary.csv"
    family_json = OUTPUT_DIR / "strategy_family_summary.json"
    write_csv(family_csv, family_rows)
    write_json(family_json, family_rows)
    group_rows = group_comparison(canonical, metadata_status)
    group_csv = OUTPUT_DIR / "candidate_group_comparison.csv"
    group_json = OUTPUT_DIR / "candidate_group_comparison.json"
    write_csv(group_csv, group_rows)
    write_json(group_json, group_rows)
    metric_rows, pareto_rows = metric_interactions(canonical)
    metric_csv = OUTPUT_DIR / "metric_interaction_summary.csv"
    metric_json = OUTPUT_DIR / "metric_interaction_summary.json"
    pareto_csv = OUTPUT_DIR / "pareto_front_cases.csv"
    write_csv(metric_csv, metric_rows)
    write_json(metric_json, metric_rows)
    write_csv(pareto_csv, pareto_rows)
    claim_md, claim_payload = claim_boundary(canonical)
    claim_md_path = OUTPUT_DIR / "run09_claim_boundary.md"
    claim_json_path = OUTPUT_DIR / "run09_claim_boundary.json"
    claim_md_path.write_text(claim_md, encoding="utf-8")
    write_json(claim_json_path, claim_payload)
    plot_outputs, plotting_status = make_plots(canonical, family_rows)
    outputs = [
        str(validation_path),
        str(canonical_csv),
        str(canonical_json),
        str(leader_csv),
        str(leader_json),
        str(leader_md),
        str(family_csv),
        str(family_json),
        str(group_csv),
        str(group_json),
        str(metric_csv),
        str(metric_json),
        str(pareto_csv),
        str(claim_md_path),
        str(claim_json_path),
        *plot_outputs,
    ]
    write_report(validation, canonical, leaderboards, family_rows, group_rows, metric_rows, pareto_rows, claim_payload, outputs, plotting_status)
    outputs.append(str(REPORT_PATH))
    manifest = {
        "run_id": RUN_ID,
        "run_name": RUN_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "branch": git_branch(),
        "input_files": [str(INPUT_CSV), str(INPUT_JSON), str(RUN08_SUMMARY), str(RUN08_REPORT), str(RUN08_MANIFEST), str(CANDIDATE_CSV)],
        "output_files": outputs,
        "script_path": str(Path(__file__)),
        "validation_verdict": validation["verdict"],
        "claim_boundary_file_path": str(claim_md_path),
        "report_path": str(REPORT_PATH),
        "no_solver_run": True,
        "no_odb_opened": True,
        "no_abqjobpilot_run": True,
        "no_commit_or_push": True,
    }
    write_json(MANIFEST_PATH, manifest)
    print(validation["verdict"])
    print(f"rows={len(canonical)}")
    print(f"per_n_counts={validation['per_n_counts']}")
    print(f"best_u2={best_by(canonical, 'u2_rank_within_n')}")
    print(f"best_peeq={best_by(canonical, 'peeq_rank_within_n')}")
    print(f"best_surfaceT={best_by(canonical, 'surfaceT_rank_within_n')}")
    print(f"best_constrained={best_by(canonical, 'constrained_rank_within_n')}")
    print(f"report={REPORT_PATH}")
    print(f"manifest={MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
