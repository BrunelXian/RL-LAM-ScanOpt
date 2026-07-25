from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
RUN_ID = "run_32a_stage2_n32_legacy_teacher_label_ingestion_for_stage3"
RUN_NAME = "stage2 n32 legacy teacher label ingestion for stage3"

N32_PREVIEW = ROOT / "outputs" / "stage3_n32_legacy_teacher_data_audit" / "n32_stage3_ingestion_preview.csv"
N32_AUDIT_REPORT = ROOT / "outputs" / "stage3_n32_legacy_teacher_data_audit" / "N32_LEGACY_TEACHER_DATA_AUDIT_REPORT.md"
COMBINED172_TEACHER = ROOT / "outputs" / "stage3_run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking" / "combined172_teacher_dataset.csv"
COMBINED172_READY = ROOT / "outputs" / "stage3_run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking" / "combined172_RL_ready_dataset.csv"
RUN29_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_29_combined172_surrogate_gnn_hybrid_policy_update_and_candidate_generation" / "RUN_29_COMBINED172_SURROGATE_GNN_HYBRID_POLICY_UPDATE_AND_CANDIDATE_GENERATION_REPORT.md"

OUTPUT_DIR = ROOT / "outputs" / "stage3_run_32a_stage2_n32_legacy_teacher_label_ingestion_for_stage3"
REPORT_DIR = ROOT / "docs" / "stage3" / "runs" / RUN_ID
REPORT_PATH = REPORT_DIR / "RUN_32A_STAGE2_N32_LEGACY_TEACHER_LABEL_INGESTION_FOR_STAGE3_REPORT.md"
MANIFEST_PATH = ROOT / "artifacts" / "manifests" / "stage3_run_32a_manifest.json"
RUN_INDEX_PATH = ROOT / "docs" / "stage3" / "STAGE3_RUN_INDEX.md"

EXPECTED_N32_ROWS = 336
EXPECTED_N32_DEDUP = 332
EXPECTED_COMBINED172_COUNTS = {12: 32, 16: 32, 24: 54, 40: 54}
EXPECTED_PLUS_COUNTS = {12: 32, 16: 32, 24: 54, 32: 332, 40: 54}
METRICS = [
    ("u2", "u2_range"),
    ("peeq", "peeq_max"),
    ("surfaceT", "surface_t_proxy"),
    ("mises", "mises_max"),
]
N32_WARNING = "PEEQ uses Stage 2 peeq_guard proxy; Mises uses Stage 2 mises_P95_top_band proxy, not literal native Stage 3 peeq_max/global mises_max."
MISSING_MISES_WARNING = "Some N32 legacy rows have missing mapped Mises proxy values; these rows keep a missing-source flag and use a conservative worst-observed proxy fill for rank/reward compatibility only."


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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_table_json(path: Path, rows: list[dict[str, Any]]) -> None:
    cols: list[str] = []
    for row in rows:
        for key in row:
            if key not in cols:
                cols.append(key)
    write_json(path, {"schema": "columns_and_rows", "columns": cols, "rows": [[row.get(col) for col in cols] for row in rows]})


def parse_int(value: Any) -> int:
    text = str(value).strip()
    if text.upper().startswith("N"):
        text = text[1:]
    return int(float(text))


def parse_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y", "pass", "ok"}


def parse_order(text: Any) -> list[int] | None:
    try:
        value = json.loads(str(text))
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(value, list):
        return None
    try:
        return [int(x) for x in value]
    except (TypeError, ValueError):
        return None


def order_hash(order: list[int]) -> str:
    return hashlib.sha1(",".join(str(x) for x in order).encode("ascii")).hexdigest()[:16]


def validate_order(order: list[int] | None, n: int) -> tuple[bool, str]:
    if order is None:
        return False, "missing/unparsable"
    if len(order) != n:
        return False, f"length {len(order)} != {n}"
    if len(set(order)) != n:
        return False, "duplicates"
    if set(order) != set(range(n)):
        return False, f"not permutation 0..{n-1}"
    return True, "valid"


def safe_divide(num: float, den: float, default: float = 0.0) -> float:
    return num / den if den else default


def mean(values: list[float], default: float = math.nan) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return statistics.fmean(vals) if vals else default


def rank_ascending(rows: list[dict[str, Any]], metric: str, rank_col: str) -> None:
    ordered = sorted(rows, key=lambda row: (row[metric], row.get("strategy_name", "")))
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][metric] == ordered[i][metric]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ordered[k][rank_col] = avg
        i = j


def rank_descending(rows: list[dict[str, Any]], metric: str, rank_col: str) -> None:
    ordered = sorted(rows, key=lambda row: (-row[metric], row.get("strategy_name", "")))
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][metric] == ordered[i][metric]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ordered[k][rank_col] = avg
        i = j


def add_rank_score(rows: list[dict[str, Any]], rank_col: str, score_col: str) -> None:
    denom = max(1, len(rows) - 1)
    for row in rows:
        row[score_col] = 1.0 - safe_divide(row[rank_col] - 1.0, denom)


def add_minmax_cost(rows: list[dict[str, Any]], metric: str, col: str) -> None:
    vals = [row[metric] for row in rows if math.isfinite(row[metric])]
    mn, mx = min(vals), max(vals)
    for row in rows:
        row[col] = safe_divide(row[metric] - mn, mx - mn, 0.0)


def legacy_source_category(source: str) -> str:
    if source == "stage2_n32_full32_legacy":
        return "stage2_n32_full32_baseline_reference_legacy"
    return source or "stage2_n32_gnn_rl_legacy"


def n32_mises_fill_value(rows: list[dict[str, str]]) -> float:
    vals = [parse_float(row.get("mises_max")) for row in rows]
    finite = [v for v in vals if math.isfinite(v)]
    if not finite:
        return math.nan
    mx = max(finite)
    return mx + max(1e-9, abs(mx) * 1e-9)


def mapped_mises_value(row: dict[str, Any], fill_value: float) -> float:
    value = parse_float(row.get("mises_max"))
    return value if math.isfinite(value) else fill_value


def validate_inputs(rows: list[dict[str, str]], combined172_rows: list[dict[str, str]]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = [N32_WARNING]
    counts = Counter()
    invalid_orders: list[str] = []
    missing_mises_rows: list[str] = []
    for row in rows:
        n = parse_int(row.get("n"))
        counts[n] += 1
        order = parse_order(row.get("order_json"))
        ok, reason = validate_order(order, 32)
        if not ok:
            invalid_orders.append(f"{row.get('strategy_name')}: {reason}")
        for col in ["u2_range", "surface_t_proxy"]:
            if not math.isfinite(parse_float(row.get(col))):
                errors.append(f"{row.get('strategy_name')}: missing {col}")
        if not math.isfinite(parse_float(row.get("peeq_max"))):
            errors.append(f"{row.get('strategy_name')}: missing mapped peeq_guard/peeq_max")
        if not math.isfinite(parse_float(row.get("mises_max"))):
            missing_mises_rows.append(row.get("strategy_name", ""))
    if missing_mises_rows:
        warnings.append(MISSING_MISES_WARNING)
    if len(rows) != EXPECTED_N32_ROWS:
        errors.append(f"expected {EXPECTED_N32_ROWS} N32 preview rows, found {len(rows)}")
    if counts != Counter({32: EXPECTED_N32_ROWS}):
        errors.append(f"expected all rows N32, found {dict(counts)}")
    if invalid_orders:
        errors.extend(invalid_orders[:10])
    if not all("source_file" in row and "source_row_index" in row for row in rows):
        warnings.append("source_file/source_row_index missing in some rows")
    if not all("teacher_validation_status" in row or "compatibility_notes" in row for row in rows):
        warnings.append("teacher validation/provenance field missing in some rows")
    h = defaultdict(list)
    for row in rows:
        h[row.get("order_hash") or order_hash(parse_order(row.get("order_json")) or [])].append(row.get("strategy_name", ""))
    duplicate_groups = {digest: names for digest, names in h.items() if len(names) > 1}
    combined_counts = Counter(parse_int(row.get("n")) for row in combined172_rows)
    if len(combined172_rows) != 172 or dict(combined_counts) != EXPECTED_COMBINED172_COUNTS:
        errors.append(f"combined172 mismatch rows={len(combined172_rows)} counts={dict(combined_counts)}")
    verdict = "WARNING_RUN32A_N32_LEGACY_INPUT_READY_WITH_SEMANTIC_MAPPING_WARNINGS" if not errors else "FAIL_RUN32A_INPUT_VALIDATION"
    payload = {
        "verdict": verdict,
        "errors": errors,
        "warnings": warnings,
        "preview_rows": len(rows),
        "n_counts": dict(counts),
        "dataset_source_counts": dict(Counter(row.get("dataset_source", "") for row in rows)),
        "duplicate_order_hash_group_count": len(duplicate_groups),
        "duplicate_order_hash_row_count": sum(len(v) for v in duplicate_groups.values()),
        "missing_mapped_mises_rows": len(missing_mises_rows),
        "missing_mapped_mises_strategies": missing_mises_rows,
        "mises_rank_fill_policy": "conservative_worst_observed_plus_epsilon_for_rank_reward_compatibility_only" if missing_mises_rows else "not_needed",
        "combined172_rows": len(combined172_rows),
        "combined172_per_n_counts": dict(sorted(combined_counts.items())),
        "metric_semantic_warning": True,
    }
    write_json(OUTPUT_DIR / "run32a_input_validation_summary.json", payload)
    return payload


def duplicate_audit(rows: list[dict[str, str]], mises_fill_value: float) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        digest = row.get("order_hash") or order_hash(parse_order(row.get("order_json")) or [])
        groups[digest].append(row)
    audit_rows: list[dict[str, Any]] = []
    group_meta: dict[str, dict[str, Any]] = {}
    group_id = 0
    for digest, members in sorted(groups.items()):
        if len(members) == 1:
            continue
        group_id += 1
        aliases = [row["strategy_name"] for row in members]
        def metric_value(row: dict[str, str], col: str) -> float:
            return mapped_mises_value(row, mises_fill_value) if col == "mises_max" else parse_float(row.get(col))
        def spread(col: str) -> float:
            vals = [metric_value(row, col) for row in members]
            return max(vals) - min(vals)
        representative = min(members, key=lambda row: parse_int(row.get("source_row_index", 10**9)))
        near_identical = all(spread(col) <= max(1e-12, abs(parse_float(representative.get(col))) * 1e-9) for col in ["u2_range", "peeq_max", "surface_t_proxy", "mises_max"])
        reason = "earliest_source_row_representative_metrics_near_identical" if near_identical else "earliest_source_row_representative_metric_spread_recorded"
        gid = f"N32_DUP_{group_id:02d}"
        group_meta[digest] = {"duplicate_group_id": gid, "aliases": aliases, "representative": representative["strategy_name"], "reason": reason}
        for row in members:
            removed = row["strategy_name"] != representative["strategy_name"]
            audit_rows.append(
                {
                    "duplicate_group_id": gid,
                    "order_hash": digest,
                    "strategy_name": row["strategy_name"],
                    "duplicate_aliases": ";".join(aliases),
                    "representative_strategy": representative["strategy_name"],
                    "duplicate_removed_for_training": removed,
                    "representative_selection_reason": reason,
                    "u2_range": row.get("u2_range"),
                    "peeq_guard": row.get("peeq_max"),
                    "surface_t_proxy": row.get("surface_t_proxy"),
                    "mises_P95_top_band": row.get("mises_max"),
                    "mises_source_missing": not math.isfinite(parse_float(row.get("mises_max"))),
                    "mises_rank_fill_value": mapped_mises_value(row, mises_fill_value) if not math.isfinite(parse_float(row.get("mises_max"))) else "",
                    "u2_spread": spread("u2_range"),
                    "peeq_spread": spread("peeq_max"),
                    "surfaceT_spread": spread("surface_t_proxy"),
                    "mises_spread": spread("mises_max"),
                }
            )
    return audit_rows, group_meta


def canonical_full(rows: list[dict[str, str]], group_meta: dict[str, dict[str, Any]], mises_fill_value: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        digest = row.get("order_hash") or order_hash(parse_order(row.get("order_json")) or [])
        meta = group_meta.get(digest, {})
        aliases = meta.get("aliases", [row.get("strategy_name", "")])
        removed = bool(meta and row.get("strategy_name") != meta.get("representative"))
        src_cat = legacy_source_category(row.get("dataset_source", ""))
        raw_mises = parse_float(row.get("mises_max"))
        missing_mises = not math.isfinite(raw_mises)
        compatible_mises = mapped_mises_value(row, mises_fill_value)
        compatibility_notes = row.get("compatibility_notes", "")
        if missing_mises:
            compatibility_notes = (compatibility_notes + " | " if compatibility_notes else "") + "mises_P95_top_band missing in preview; conservative worst-observed proxy fill used for rank/reward compatibility only"
        out.append(
            {
                "n": 32,
                "strategy_name": row.get("strategy_name", ""),
                "dataset_source": "stage2_n32_legacy",
                "legacy_source_category": src_cat,
                "order_json": row.get("order_json", ""),
                "order_hash": digest,
                "u2_range": parse_float(row.get("u2_range")),
                "peeq_guard": parse_float(row.get("peeq_max")),
                "peeq_max_mapped_from_peeq_guard": parse_float(row.get("peeq_max")),
                "surface_t_proxy": parse_float(row.get("surface_t_proxy")),
                "surface_t_proxy_source": "surface_tensile_primary",
                "mises_P95_top_band": "" if missing_mises else raw_mises,
                "mises_max_mapped_from_mises_P95_top_band": compatible_mises,
                "mises_source_missing": missing_mises,
                "mises_rank_fill_policy": "conservative_worst_observed_plus_epsilon_for_rank_reward_compatibility_only" if missing_mises else "source_value_used",
                "mises_rank_fill_value": compatible_mises if missing_mises else "",
                "teacher_validation_status": "LEGACY_TEACHER_LABEL_COMPATIBLE" if boolish(row.get("teacher_validation_status")) else row.get("teacher_validation_status", ""),
                "final_cooling_success": boolish(row.get("teacher_validation_status")),
                "extraction_provenance": "canonical full-field Stage 2 source; see source_file and audit report",
                "source_file": row.get("source_file", ""),
                "source_row_index": row.get("source_row_index", ""),
                "duplicate_group_id": meta.get("duplicate_group_id", ""),
                "duplicate_aliases": ";".join(aliases),
                "duplicate_removed_for_training": removed,
                "representative_selection_reason": meta.get("reason", "unique_order_hash"),
                "metric_semantic_warning": True,
                "compatibility_notes": (compatibility_notes + " | " + N32_WARNING).strip(" |"),
            }
        )
    return out


def dedup_training(full_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [row for row in full_rows if not row["duplicate_removed_for_training"]]
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "n": 32,
                "strategy_name": row["strategy_name"],
                "dataset_source": "stage2_n32_legacy",
                "legacy_source_category": row["legacy_source_category"],
                "order_json": row["order_json"],
                "order_hash": row["order_hash"],
                "u2_range": row["u2_range"],
                "peeq_max": row["peeq_max_mapped_from_peeq_guard"],
                "surface_t_proxy": row["surface_t_proxy"],
                "mises_max": row["mises_max_mapped_from_mises_P95_top_band"],
                "peeq_metric_source": "peeq_guard",
                "mises_metric_source": "mises_P95_top_band" if not row.get("mises_source_missing") else "mises_P95_top_band_missing_conservative_rank_fill",
                "peeq_metric_semantics": "guard_proxy_not_literal_stage3_peeq_max",
                "mises_metric_semantics": "top_band_p95_not_literal_global_mises_max" if not row.get("mises_source_missing") else "missing_top_band_p95_conservative_worst_observed_fill_not_literal_global_mises_max",
                "teacher_validation_status": row["teacher_validation_status"],
                "final_cooling_success": row["final_cooling_success"],
                "source_file": row["source_file"],
                "source_row_index": row["source_row_index"],
                "duplicate_group_id": row["duplicate_group_id"],
                "duplicate_aliases": row["duplicate_aliases"],
                "compatibility_status": "LEGACY_COMPATIBLE_WITH_WARNINGS",
                "metric_semantic_warning": True,
                "mises_source_missing": row.get("mises_source_missing", False),
                "mises_rank_fill_policy": row.get("mises_rank_fill_policy", ""),
                "mises_rank_fill_value": row.get("mises_rank_fill_value", ""),
            }
        )
    return out


def add_n32_ranks(rows: list[dict[str, Any]]) -> None:
    for label, metric in METRICS:
        rank_col = f"{label}_rank_within_n32" if label != "surfaceT" else "surfaceT_rank_within_n32"
        score_col = f"target_{label}_score_n32_rank" if label != "surfaceT" else "target_surfaceT_score_n32_rank"
        rank_ascending(rows, metric, rank_col)
        add_rank_score(rows, rank_col, score_col)
    for row in rows:
        row["target_reward_n32_legacy_mapped_u2_primary"] = (
            0.65 * row["target_u2_score_n32_rank"]
            + 0.20 * row["target_peeq_score_n32_rank"]
            + 0.10 * row["target_surfaceT_score_n32_rank"]
            + 0.05 * row["target_mises_score_n32_rank"]
        )
        row["target_reward_n32_strict_u2_surfaceT"] = 0.80 * row["target_u2_score_n32_rank"] + 0.20 * row["target_surfaceT_score_n32_rank"]
    rank_descending(rows, "target_reward_n32_legacy_mapped_u2_primary", "reward_n32_legacy_mapped_rank")
    rank_descending(rows, "target_reward_n32_strict_u2_surfaceT", "reward_n32_strict_rank")


def n32_leaderboard(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for label, metric in METRICS:
        for idx, row in enumerate(sorted(rows, key=lambda r: r[metric])[:10], start=1):
            out.append({"n": 32, "category": f"top10_{label}", "position": idx, "strategy_name": row["strategy_name"], "value": row[metric], "legacy_source_category": row["legacy_source_category"]})
    for metric, category in [("target_reward_n32_legacy_mapped_u2_primary", "top10_mapped_reward"), ("target_reward_n32_strict_u2_surfaceT", "top10_strict_u2_surfaceT_reward")]:
        for idx, row in enumerate(sorted(rows, key=lambda r: r[metric], reverse=True)[:10], start=1):
            out.append({"n": 32, "category": category, "position": idx, "strategy_name": row["strategy_name"], "value": row[metric], "legacy_source_category": row["legacy_source_category"]})
    return out


def normalize_stage3_row(row: dict[str, str]) -> dict[str, Any]:
    n = parse_int(row.get("n"))
    return {
        **row,
        "n": n,
        "u2_range": parse_float(row.get("u2_range")),
        "peeq_max": parse_float(row.get("peeq_max")),
        "surface_t_proxy": parse_float(row.get("surface_t_proxy")),
        "mises_max": parse_float(row.get("mises_max")),
        "metric_semantic_warning": False,
        "legacy_compatibility_status": "NATIVE_STAGE3",
        "peeq_metric_source": "native_stage3_peeq_max",
        "mises_metric_source": "native_stage3_mises_max",
    }


def build_combined_plus(combined172_rows: list[dict[str, str]], n32_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    combined = [normalize_stage3_row(row) for row in combined172_rows]
    for row in n32_rows:
        combined.append(
            {
                **row,
                "metric_semantic_warning": True,
                "legacy_compatibility_status": "LEGACY_COMPATIBLE_WITH_WARNINGS",
                "is_n32_legacy": True,
            }
        )
    counts = Counter(row["n"] for row in combined)
    if len(combined) != 504 or dict(sorted(counts.items())) != EXPECTED_PLUS_COUNTS:
        raise RuntimeError(f"combined172_plus_N32 mismatch rows={len(combined)} counts={dict(sorted(counts.items()))}")
    for n in sorted(counts):
        group = [row for row in combined if row["n"] == n]
        for label, metric in METRICS:
            rank_col = f"{label}_rank_combined172_plus_N32_within_n" if label != "surfaceT" else "surfaceT_rank_combined172_plus_N32_within_n"
            score_col = f"target_{label}_score_combined172_plus_N32_rank" if label != "surfaceT" else "target_surfaceT_score_combined172_plus_N32_rank"
            cost_col = f"{label}_cost_minmax_combined172_plus_N32_within_n" if label != "surfaceT" else "surfaceT_cost_minmax_combined172_plus_N32_within_n"
            rank_ascending(group, metric, rank_col)
            add_rank_score(group, rank_col, score_col)
            add_minmax_cost(group, metric, cost_col)
        for row in group:
            row["target_reward_combined172_plus_N32_mapped_u2_primary"] = (
                0.65 * row["target_u2_score_combined172_plus_N32_rank"]
                + 0.20 * row["target_peeq_score_combined172_plus_N32_rank"]
                + 0.10 * row["target_surfaceT_score_combined172_plus_N32_rank"]
                + 0.05 * row["target_mises_score_combined172_plus_N32_rank"]
            )
        rank_descending(group, "target_reward_combined172_plus_N32_mapped_u2_primary", "reward_rank_combined172_plus_N32_within_n")
    return combined


def rl_ready(combined: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = [
        "n", "strategy_name", "dataset_source", "legacy_source_category", "order_json", "order_hash",
        "u2_range", "peeq_max", "surface_t_proxy", "mises_max",
        "peeq_metric_source", "mises_metric_source", "metric_semantic_warning", "legacy_compatibility_status",
        "target_u2_score_combined172_plus_N32_rank", "target_peeq_score_combined172_plus_N32_rank",
        "target_surfaceT_score_combined172_plus_N32_rank", "target_mises_score_combined172_plus_N32_rank",
        "target_reward_combined172_plus_N32_mapped_u2_primary", "reward_rank_combined172_plus_N32_within_n",
        "teacher_validation_status", "source_file", "source_row_index",
    ]
    return [{key: row.get(key, "") for key in keep} for row in combined]


def write_notes() -> Path:
    path = OUTPUT_DIR / "n32_training_use_notes.md"
    path.write_text(
        "\n".join(
            [
                "# N32 Training Use Notes",
                "",
                "- N32 can be used as a legacy-compatible additional N group with warnings.",
                "- U2 and SurfaceT mapping are stronger.",
                "- PEEQ and Mises are proxy-compatible, not literal identical metric names.",
                "- Some rows have missing mapped Mises proxy values; those rows are flagged and use conservative worst-observed proxy fill values only so rank/reward tables remain mechanically complete.",
                "- Training should use per-N balancing or sample weighting because N32 has 332 rows versus N12/N16=32 and N24/N40=54.",
                "- Claims should describe N32 as legacy-compatible Stage 2 32-track teacher data, not native Stage 3 newly generated teacher validation.",
                "- N32 is valuable as an intermediate N between N24 and N40 for graph-policy generalization analysis.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def write_claim_boundary() -> tuple[Path, Path]:
    md = OUTPUT_DIR / "run32a_claim_boundary.md"
    js = OUTPUT_DIR / "run32a_claim_boundary.json"
    safe = [
        "Run32A ingests audited Stage 2 N32 legacy teacher labels into a Stage 3-compatible schema.",
        "Run32A creates a deduplicated N32 training table with one row per unique order_hash.",
        "Run32A creates combined172_plus_N32 datasets with N32 as an additional N group.",
        "N32 provides a large intermediate-N teacher-labelled group for future surrogate/GNN/graph-pointer training.",
        "N32 compatibility has metric-semantics warnings.",
        "Rows with missing mapped Mises proxy values are explicitly flagged and use a conservative rank/reward compatibility fill.",
    ]
    unsafe = [
        "N32 data is native Stage 3 teacher validation.",
        "peeq_guard is exactly Stage 3 peeq_max.",
        "mises_P95_top_band is exactly global Stage 3 mises_max.",
        "conservative Mises fill values are literal teacher measurements.",
        "no compatibility limitations.",
        "model improvement.",
        "GNN-RL superiority.",
        "arbitrary-N generalization.",
        "online RL.",
        "any new Abaqus/ODB extraction was performed.",
    ]
    md.write_text("# Run32A Claim Boundary\n\n## Safe Claims\n" + "\n".join(f"- {x}" for x in safe) + "\n\n## Unsafe Claims\n" + "\n".join(f"- Do not claim {x}" for x in unsafe) + "\n", encoding="utf-8")
    write_json(js, {"verdict": "RUN32A_N32_LEGACY_INGESTION_ONLY_WITH_METRIC_SEMANTIC_WARNINGS", "safe_claims": safe, "unsafe_claims": unsafe})
    return md, js


def write_report(validation: dict[str, Any], duplicate_rows: list[dict[str, Any]], n32_rows: list[dict[str, Any]], combined_plus: list[dict[str, Any]], outputs: list[str]) -> None:
    best_u2 = min(n32_rows, key=lambda r: r["u2_range"])
    best_reward = max(n32_rows, key=lambda r: r["target_reward_n32_legacy_mapped_u2_primary"])
    counts = Counter(row["n"] for row in combined_plus)
    lines = [
        "# Stage 3 Run 32A - Stage 2 N32 Legacy Teacher Label Ingestion for Stage 3",
        "",
        "## Purpose",
        "Ingest audited Stage 2 N32 legacy teacher labels into a Stage 3-compatible schema, deduplicate by order hash, compute N32 ranks, and build combined172_plus_N32 datasets without model training or candidate generation.",
        "",
        "## Source Audit Summary",
        "- Audit verdict: `WARNING_N32_LEGACY_TEACHER_DATA_PARTIAL`.",
        f"- Audit report: `{N32_AUDIT_REPORT}`",
        "",
        "## Source Table and Row Counts",
        f"- Preview rows: `{validation['preview_rows']}`",
        "- Complete legal N32 rows: `336`",
        "- Deduplicated training rows: `332`",
        "",
        "## Compatibility Decision",
        "The N32 table is ingested as legacy-compatible with warnings. U2 and SurfaceT are the strongest mappings; PEEQ and Mises are proxy-compatible fields.",
        f"- Missing mapped Mises proxy rows: `{validation.get('missing_mapped_mises_rows', 0)}`",
        f"- Mises fill policy: `{validation.get('mises_rank_fill_policy', 'not_needed')}`",
        "",
        "## Duplicate Handling",
        f"- Duplicate groups: `{len({r['duplicate_group_id'] for r in duplicate_rows})}`",
        "- Representatives are selected by earliest source row, with aliases and metric spread recorded.",
        "",
        "## Full Provenance N32 Dataset",
        "- All 336 compatible rows are preserved with source file, source row, duplicate metadata, and semantic notes.",
        "",
        "## Deduplicated N32 Training Dataset",
        "- One row per unique order hash; duplicate aliases preserved.",
        "",
        "## N32 Within-N Ranking and Leaderboard",
        f"- Best U2: `{best_u2['strategy_name']}`",
        f"- Best mapped reward: `{best_reward['strategy_name']}`",
        "",
        "## combined172_plus_N32 Construction",
        f"- Total rows: `{len(combined_plus)}`",
        f"- Per-N counts: `{dict(sorted(counts.items()))}`",
        "",
        "## Metric Semantic Boundary",
        "- `peeq_guard` is mapped to `peeq_max` only as a proxy-compatible legacy field.",
        "- `mises_P95_top_band` is mapped to `mises_max` only as a proxy-compatible legacy field.",
        "- Rows with missing `mises_P95_top_band` are flagged and receive a conservative worst-observed proxy fill for rank/reward compatibility only.",
        "",
        "## Per-N Imbalance Warning",
        "N32 has 332 rows, much larger than N12/N16=32 and N24/N40=54. Future training should use per-N balancing or sample weighting.",
        "",
        "## Training-Use Recommendations",
        "Use N32 as a legacy-compatible intermediate N group and compare model updates with and without per-N balancing.",
        "",
        "## Claim Boundary",
        "`RUN32A_N32_LEGACY_INGESTION_ONLY_WITH_METRIC_SEMANTIC_WARNINGS`.",
        "",
        "## Output Files",
    ]
    lines.extend(f"- `{p}`" for p in outputs)
    lines.extend(
        [
            "",
            "## Recommended Next Action",
            "- If current hybrid batch32 has not been run yet, the user may still manually run it; this N32 ingestion does not conflict.",
            "- After hybrid batch32 teacher validation, build combined204_plus_N32.",
            "- Future model-update run should compare combined172 only, combined172_plus_N32 with per-N balancing, and combined204_plus_N32 after the 32 new jobs are teacher-validated.",
            "",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def update_run_index(verdict: str) -> None:
    if not RUN_INDEX_PATH.exists():
        return
    text = RUN_INDEX_PATH.read_text(encoding="utf-8")
    if "| run_32a |" in text:
        return
    row = (
        "| run_32a | Stage 2 N32 legacy teacher label ingestion for Stage 3 | "
        "Ingest audited N32 legacy teacher labels, deduplicate order hashes, compute N32 ranks, and build combined172_plus_N32 datasets. | "
        "`scripts/stage3/run_32a_ingest_stage2_n32_legacy_teacher_labels_for_stage3.py` | "
        "`docs/stage3/runs/run_32a_stage2_n32_legacy_teacher_label_ingestion_for_stage3/RUN_32A_STAGE2_N32_LEGACY_TEACHER_LABEL_INGESTION_FOR_STAGE3_REPORT.md` | "
        "`outputs/stage3_run_32a_stage2_n32_legacy_teacher_label_ingestion_for_stage3/` | "
        f"`{verdict}` | No Abaqus, no ODB opening, no abqjobpilot, no CAE/INP/JNL generation, no training, no candidate generation, no commit/push. |"
    )
    RUN_INDEX_PATH.write_text(text.rstrip() + "\n" + row + "\n", encoding="utf-8")


def git_branch() -> str:
    try:
        return subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    preview = read_csv(N32_PREVIEW)
    combined172_raw = read_csv(COMBINED172_TEACHER)
    validation = validate_inputs(preview, combined172_raw)
    if validation["verdict"].startswith("FAIL"):
        print(validation["verdict"])
        return 2
    mises_fill_value = n32_mises_fill_value(preview)
    duplicate_rows, group_meta = duplicate_audit(preview, mises_fill_value)
    full = canonical_full(preview, group_meta, mises_fill_value)
    dedup = dedup_training(full)
    if len(dedup) != EXPECTED_N32_DEDUP:
        raise RuntimeError(f"N32 dedup expected {EXPECTED_N32_DEDUP}, found {len(dedup)}")
    add_n32_ranks(dedup)
    board = n32_leaderboard(dedup)
    combined_plus = build_combined_plus(combined172_raw, dedup)
    ready = rl_ready(combined_plus)
    notes = write_notes()
    claim_md, claim_json = write_claim_boundary()
    summary = {
        "run_id": RUN_ID,
        "n32_full_rows": len(full),
        "n32_dedup_training_rows": len(dedup),
        "duplicate_groups_count": len({r["duplicate_group_id"] for r in duplicate_rows}),
        "combined172_rows": len(combined172_raw),
        "combined172_plus_N32_rows": len(combined_plus),
        "per_n_counts": dict(sorted(Counter(row["n"] for row in combined_plus).items())),
        "dataset_source_counts_n32_full": dict(Counter(row["legacy_source_category"] for row in full)),
        "metric_semantic_warning": True,
        "missing_mapped_mises_rows": validation.get("missing_mapped_mises_rows", 0),
        "mises_rank_fill_policy": validation.get("mises_rank_fill_policy", "not_needed"),
        "mises_rank_fill_value": mises_fill_value if validation.get("missing_mapped_mises_rows", 0) else "",
        "n32_best_u2_strategy": min(dedup, key=lambda r: r["u2_range"])["strategy_name"],
        "n32_best_mapped_reward_strategy": max(dedup, key=lambda r: r["target_reward_n32_legacy_mapped_u2_primary"])["strategy_name"],
    }

    outputs: list[str] = []
    for path, rows in [
        (OUTPUT_DIR / "n32_duplicate_order_hash_audit.csv", duplicate_rows),
        (OUTPUT_DIR / "n32_legacy_teacher_dataset_full_provenance_336.csv", full),
        (OUTPUT_DIR / "n32_legacy_teacher_dataset_dedup_training_332.csv", dedup),
        (OUTPUT_DIR / "n32_legacy_teacher_dataset_dedup_ranked_332.csv", dedup),
        (OUTPUT_DIR / "n32_legacy_per_metric_leaderboard.csv", board),
        (OUTPUT_DIR / "combined172_plus_N32_teacher_dataset.csv", combined_plus),
        (OUTPUT_DIR / "combined172_plus_N32_RL_ready_dataset.csv", ready),
    ]:
        write_csv(path, rows)
        outputs.append(str(path))
    for path, rows in [
        (OUTPUT_DIR / "n32_duplicate_order_hash_audit.json", duplicate_rows),
        (OUTPUT_DIR / "n32_legacy_teacher_dataset_full_provenance_336.json", full),
        (OUTPUT_DIR / "n32_legacy_teacher_dataset_dedup_training_332.json", dedup),
    ]:
        write_table_json(path, rows)
        outputs.append(str(path))
    write_json(OUTPUT_DIR / "n32_legacy_summary.json", summary)
    write_json(OUTPUT_DIR / "combined172_plus_N32_summary.json", summary)
    outputs.extend([str(OUTPUT_DIR / "run32a_input_validation_summary.json"), str(OUTPUT_DIR / "n32_legacy_summary.json"), str(OUTPUT_DIR / "combined172_plus_N32_summary.json"), str(notes), str(claim_md), str(claim_json)])
    write_report(validation, duplicate_rows, dedup, combined_plus, outputs)
    outputs.append(str(REPORT_PATH))
    update_run_index(validation["verdict"])

    manifest = {
        "run_id": RUN_ID,
        "run_name": RUN_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "branch": git_branch(),
        "source_audit_report_path": str(N32_AUDIT_REPORT),
        "source_preview_path": str(N32_PREVIEW),
        "combined172_input_path": str(COMBINED172_TEACHER),
        "output_files": outputs,
        "n32_full_rows": len(full),
        "n32_dedup_training_rows": len(dedup),
        "combined172_rows": len(combined172_raw),
        "combined172_plus_N32_rows": len(combined_plus),
        "per_N_counts": dict(sorted(Counter(row["n"] for row in combined_plus).items())),
        "duplicate_groups_count": len({r["duplicate_group_id"] for r in duplicate_rows}),
        "metric_semantic_warnings": True,
        "report_path": str(REPORT_PATH),
        "claim_boundary_path": str(claim_md),
        "no_solver_run": True,
        "no_odb_opened": True,
        "no_abqjobpilot_run": True,
        "no_cae_inp_generated": True,
        "no_teacher_validation_performed": True,
        "no_training": True,
        "no_candidate_generation": True,
        "no_commit_or_push": True,
    }
    write_json(MANIFEST_PATH, manifest)
    print(validation["verdict"])
    print(f"n32_full={len(full)} n32_dedup={len(dedup)}")
    print(f"combined172_plus_N32={len(combined_plus)} per_n={dict(sorted(Counter(row['n'] for row in combined_plus).items()))}")
    print(f"best_u2={summary['n32_best_u2_strategy']}")
    print(f"best_reward={summary['n32_best_mapped_reward_strategy']}")
    print(f"report={REPORT_PATH}")
    print(f"manifest={MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
