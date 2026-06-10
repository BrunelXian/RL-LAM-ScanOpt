from __future__ import annotations

import ast
import csv
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCE_ROOT = Path(r"D:\Projects\RL-LAM-ScanOpt")
TARGET_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")

RUN_ID = "run_02_fixed32_policy_novelty_audit"
REPORT_DIR = TARGET_ROOT / "docs" / "stage3" / "runs" / "run_02_fixed32_policy_novelty_audit"
OUTPUT_DIR = TARGET_ROOT / "outputs" / "stage3_run_02_policy_novelty_audit"
MANIFEST_PATH = TARGET_ROOT / "artifacts" / "manifests" / "stage3_run_02_manifest.json"
REPORT_PATH = REPORT_DIR / "RUN_02_POLICY_NOVELTY_AUDIT_REPORT.md"

ALLOWED_EXTENSIONS = {".csv", ".json", ".md", ".txt", ".py"}
FORBIDDEN_EXTENSIONS = {
    ".odb",
    ".cae",
    ".sim",
    ".dat",
    ".msg",
    ".sta",
    ".lck",
    ".prt",
    ".com",
    ".res",
    ".abq",
    ".pac",
    ".sel",
    ".ipm",
}
MAX_TEXT_FILE_BYTES = 8 * 1024 * 1024
MAX_REJECTION_ROWS = 1000
MAX_BASELINE_RECORDS = 60
MAX_LEARNED_RECORDS = 120
TRACK_SET = set(range(32))
EDGE_TRACKS = {0, 1, 2, 3, 28, 29, 30, 31}
CENTER_TRACKS = {14, 15, 16, 17}

BASELINE_KEYWORDS = (
    "formal_raster_left_to_right",
    "raster",
    "odd_even_interlaced",
    "odd_even",
    "center_out",
    "edge_in_alternating",
    "edge_in",
    "lhi",
    "greedy_maximin_distance",
    "maximin",
    "smartscan_proxy_variance",
    "smartscan",
    "multi_lag_regular_jump",
    "windowed_max_dispersion",
    "block_interleaved_quarters",
    "center_edge_alternating",
    "method-c",
    "method_c",
    "mc_cand",
    "mc_extra",
)
LEARNED_KEYWORDS = (
    "rlu2m",
    "rlu2",
    "rl20",
    "rlv2",
    "rls",
    "gnnu",
    "gnn",
    "rl_component",
    "rl/policy",
    "learned_hybrid",
    "lhyb",
)
SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    "models",
    "model_ready_torch",
    "figures",
    "post-odb",
    "temp",
}
RELEVANT_PATH_KEYWORDS = tuple(
    sorted(
        set(BASELINE_KEYWORDS + LEARNED_KEYWORDS)
        | {
            "stage2",
            "scan_order",
            "baseline_orders",
            "rule_policy_orders",
            "method_c_policy_orders",
            "teacher_labels",
            "teacher_metrics",
            "full48_teacher_labels_loaded_validated",
            "approved_rl",
            "approved_rlu2",
            "gnn_rl",
            "run_84",
            "run_80",
            "build_full20",
            "build_32track_strategy",
            "strategy_metadata",
            "selected_next10_candidates",
        }
    )
)
METRIC_ALIASES = {
    "U2": ("u2", "warpage", "u_mag", "u_score"),
    "PEEQ": ("peeq",),
    "SurfaceT": ("surfacet", "surface_t", "surface tensile", "surface_tensile"),
    "Gradient": ("gradient",),
    "Mises": ("mises",),
    "internal_tensile_stress": ("internal tensile", "s11", "s22", "s33", "tensile"),
}
REJECTED_TOTAL = 0


@dataclass
class OrderRecord:
    strategy_name: str
    category: str
    family: str
    source_file: str
    scan_order: list[int]
    source_kind: str
    metrics: dict[str, float | str] = field(default_factory=dict)

    @property
    def short_id(self) -> str:
        return sanitize_name(self.strategy_name)


@dataclass
class RejectedOrder:
    source_file: str
    candidate_name: str
    reason: str
    length: int | str
    raw_preview: str


def sanitize_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return safe[:96] or "unnamed"


def ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)


def is_allowed_input(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in FORBIDDEN_EXTENSIONS:
        return False
    if suffix not in ALLOWED_EXTENSIONS:
        return False
    try:
        return path.stat().st_size <= MAX_TEXT_FILE_BYTES
    except OSError:
        return False


def discover_input_files() -> list[Path]:
    candidates: list[Path] = []
    preferred = TARGET_ROOT / "docs" / "stage2_reference"
    seen: set[str] = set()

    def add_path(path: Path) -> None:
        if not path.is_file() or not is_allowed_input(path):
            return
        key = str(path.resolve()).lower()
        if key in seen:
            return
        seen.add(key)
        candidates.append(path)

    if preferred.exists():
        for path in preferred.rglob("*"):
            add_path(path)

    high_value_files = [
        SOURCE_ROOT / "LDED_2D_CAE_Framework" / "abaqus_scripts" / "build_full20_2d_strategy_caes_and_inputs.py",
        SOURCE_ROOT / "LDED_2D_CAE_Framework" / "abaqus_scripts" / "build_32track_strategy_caes_from_base.py",
        SOURCE_ROOT / "LDED_2D_CAE_Framework" / "rl_training_v01" / "outputs" / "sanity" / "full48_teacher_labels_loaded_validated.csv",
        SOURCE_ROOT / "rl-training" / "v01" / "outputs" / "GNN_RL_policy_learning_evidence_freeze_v01" / "GNN_RL_metric_summary_v01.csv",
        SOURCE_ROOT / "rl-training" / "v01" / "outputs" / "GNN_RL_policy_learning_evidence_freeze_v01" / "GNN_RL_teacher_validation_boundary_v01.csv",
        SOURCE_ROOT / "rl-training" / "v01" / "outputs" / "GNN_RL_policy_learning_evidence_freeze_v01" / "GNN_RL_policy_learning_evidence_freeze_v01_report.md",
        SOURCE_ROOT / "rl-training" / "v01" / "outputs" / "GNN_RL_vs_stage1_full32_baseline_audit_v01" / "GNN_RL_vs_baseline10_rank_table_v01.csv",
        SOURCE_ROOT / "rl-training" / "v01" / "outputs" / "GNN_RL_vs_stage1_full32_baseline_audit_v01" / "GNN_RL_vs_stage1_full32_baseline_audit_v01_report.md",
        SOURCE_ROOT / "rl_training" / "outputs" / "baseline_eval" / "baseline_orders.csv",
        SOURCE_ROOT / "rl_training" / "outputs" / "baseline_eval" / "rule_policy_orders.csv",
        SOURCE_ROOT / "rl-training" / "v01" / "outputs" / "method_c_policy" / "method_c_policy_orders.csv",
        SOURCE_ROOT / "rl-training" / "v01" / "outputs" / "RLU2_medium40_focused_ODB_teacher_validation_v01" / "RLU2_medium40_focused_teacher_labels_v01.csv",
        SOURCE_ROOT / "rl-training" / "v01" / "outputs" / "RL_component_v2_conservative30_ODB_teacher_validation_v01" / "RL_component_v2_conservative30_teacher_labels_v01.csv",
        SOURCE_ROOT / "rl-training" / "v01" / "outputs" / "human_review_rl_candidate_shortlist_v01" / "human_review_shortlist_candidates.csv",
        SOURCE_ROOT / "rl-training" / "v01" / "outputs" / "RL_component_v2_generate_60_candidates_v01" / "approved_RL_component_v2_60_candidates_v01.csv",
        SOURCE_ROOT / "rl-training" / "v01" / "outputs" / "RLU2_medium40_focused_FEA_handoff_v01" / "approved_RLU2_medium40_focused_v01.csv",
    ]
    for path in high_value_files:
        add_path(path)

    if SOURCE_ROOT.exists():
        for path in SOURCE_ROOT.rglob("*"):
            parts = {part.lower() for part in path.parts}
            if parts & SKIP_DIR_NAMES:
                continue
            if not path.is_file():
                continue
            path_text = str(path).lower()
            if not any(keyword in path_text for keyword in RELEVANT_PATH_KEYWORDS):
                continue
            add_path(path)
    return sorted(candidates, key=lambda p: (0 if str(p).startswith(str(preferred)) else 1, str(p).lower()))


def parse_possible_order(value: Any) -> tuple[list[int] | None, str]:
    if isinstance(value, list):
        seq = value
    elif isinstance(value, tuple):
        seq = list(value)
    elif isinstance(value, str):
        text = value.strip()
        if not (text.startswith("[") and text.endswith("]")):
            return None, "not-list"
        try:
            seq = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return None, "non-integer"
    else:
        return None, "not-list"

    if not isinstance(seq, list):
        return None, "not-list"
    if len(seq) != 32:
        return None, "wrong length"
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in seq):
        return None, "non-integer"
    order = [int(x) for x in seq]
    if any(x < 0 or x > 31 for x in order):
        return None, "out-of-range tracks"
    if len(set(order)) != len(order):
        return None, "duplicate tracks"
    if set(order) != TRACK_SET:
        return None, "missing tracks"
    return order, "ok"


def infer_category_family(name: str, source_file: str, row: dict[str, Any] | None = None) -> tuple[str, str]:
    name_text = name.lower()
    text_parts = [name, source_file]
    if row:
        text_parts.extend(str(v) for v in row.values() if v is not None)
    text = " ".join(text_parts).lower()

    if any(k in name_text for k in LEARNED_KEYWORDS):
        category = "learned_gnn_rl"
    elif any(k in name_text for k in BASELINE_KEYWORDS):
        category = "engineering_baseline"
    elif any(k in text for k in LEARNED_KEYWORDS):
        category = "learned_gnn_rl"
    elif any(k in text for k in BASELINE_KEYWORDS):
        category = "engineering_baseline"
    else:
        category = "unclassified"

    if "formal_raster_left_to_right" in text or re.search(r"\braster\b", text):
        family = "raster"
    elif "odd_even" in text:
        family = "odd_even"
    elif "center_out" in text:
        family = "center_out"
    elif "edge_in" in text or "lhi" in text:
        family = "edge_in"
    elif "maximin" in text:
        family = "maximin"
    elif "smartscan" in text:
        family = "smartscan"
    elif "multi_lag" in text:
        family = "multi_lag"
    elif "windowed" in text or "max_dispersion" in text:
        family = "windowed_max_dispersion"
    elif "block_interleaved" in text:
        family = "block_interleaved"
    elif "center_edge" in text:
        family = "center_edge"
    elif "method-c" in text or "method_c" in text or "mc_cand" in text or "mc_extra" in text:
        family = "method_c"
    elif "rlu2m" in text:
        family = "RLU2M"
    elif "rlu2" in text:
        family = "RLU2"
    elif "rl20" in text:
        family = "RL20"
    elif "rlv2" in text:
        family = "RLV2"
    elif re.search(r"\brls", text):
        family = "RLS"
    elif "gnnu" in text or "gnn" in text:
        family = "GNN"
    elif "learned_hybrid" in text or "lhyb" in text:
        family = "learned_hybrid"
    else:
        family = "unknown"
    return category, family


def row_strategy_name(row: dict[str, Any], fallback: str) -> str:
    keys = (
        "strategy_name",
        "strategy_id",
        "candidate_id",
        "case_id",
        "name",
        "order_name",
        "run_id",
        "job_name",
        "canonical_order",
    )
    for key in keys:
        if key in row and str(row[key]).strip():
            return str(row[key]).strip()
    return fallback


def extract_metrics(row: dict[str, Any]) -> dict[str, float | str]:
    metrics: dict[str, float | str] = {}
    for out_name, aliases in METRIC_ALIASES.items():
        for key, value in row.items():
            key_norm = key.strip().lower().replace("-", "_")
            if not any(alias in key_norm for alias in aliases):
                continue
            parsed = parse_float(value)
            if parsed is not None:
                metrics[out_name] = parsed
                break
    return metrics


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(float(value)):
            return float(value)
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isfinite(number):
        return number
    return None


def add_record(
    records: list[OrderRecord],
    rejected: list[RejectedOrder],
    source_file: Path,
    candidate_name: str,
    raw_order: Any,
    source_kind: str,
    row: dict[str, Any] | None = None,
) -> None:
    order, reason = parse_possible_order(raw_order)
    if order is None:
        if reason != "not-list":
            record_rejection(
                rejected,
                RejectedOrder(
                    source_file=str(source_file),
                    candidate_name=candidate_name,
                    reason=reason,
                    length=len(raw_order) if isinstance(raw_order, (list, tuple, str)) else "n/a",
                    raw_preview=str(raw_order)[:160],
                ),
            )
        return
    category, family = infer_category_family(candidate_name, str(source_file), row)
    if category == "unclassified":
        return
    records.append(
        OrderRecord(
            strategy_name=candidate_name,
            category=category,
            family=family,
            source_file=str(source_file),
            scan_order=order,
            source_kind=source_kind,
            metrics=extract_metrics(row or {}),
        )
    )


def record_rejection(rejected: list[RejectedOrder], item: RejectedOrder) -> None:
    global REJECTED_TOTAL
    REJECTED_TOTAL += 1
    if len(rejected) < MAX_REJECTION_ROWS:
        rejected.append(item)


def extract_from_csv(path: Path, records: list[OrderRecord], rejected: list[RejectedOrder]) -> None:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(handle, dialect=dialect)
        for row_index, row in enumerate(reader, start=2):
            if not row:
                continue
            fallback = f"{path.stem}_row_{row_index}"
            name = row_strategy_name(row, fallback)
            candidate_cells: list[tuple[str, str]] = []
            for key, value in row.items():
                if value is None:
                    continue
                key_norm = str(key).lower()
                value_text = str(value).strip()
                if "scan_order" in key_norm or (value_text.startswith("[") and value_text.endswith("]")):
                    candidate_cells.append((str(key), value_text))
            for key, value in candidate_cells:
                add_record(records, rejected, path, name, value, f"csv:{key}", row)


def extract_from_json(path: Path, records: list[OrderRecord], rejected: list[RejectedOrder]) -> None:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        data = json.load(handle)

    def walk(obj: Any, parent: dict[str, Any] | None, key_path: str) -> None:
        if isinstance(obj, dict):
            current_parent = obj
            for key, value in obj.items():
                if key == "scan_order" or isinstance(value, list):
                    name = row_strategy_name(obj, f"{path.stem}:{key_path}.{key}")
                    add_record(records, rejected, path, name, value, f"json:{key_path}.{key}", obj)
                walk(value, current_parent, f"{key_path}.{key}" if key_path else str(key))
        elif isinstance(obj, list):
            if parent is not None:
                name = row_strategy_name(parent, f"{path.stem}:{key_path}")
                add_record(records, rejected, path, name, obj, f"json:{key_path}", parent)
            for index, value in enumerate(obj):
                walk(value, parent, f"{key_path}[{index}]")

    walk(data, None, "")


ARRAY_RE = re.compile(r"\[(?:\s*-?\d+\s*,){31,}\s*-?\d+\s*\]", re.MULTILINE)
NAME_RE = re.compile(
    r"([A-Za-z][A-Za-z0-9_.-]*(?:_[A-Za-z0-9_.-]+)*)\s*(?:=|:|\||,|\))?[^\n\[]*$"
)
QUOTED_NAME_RE = re.compile(r"['\"]([A-Za-z0-9_.-]{3,96})['\"]\s*:\s*$")


def infer_text_name(path: Path, text: str, start: int, index: int) -> str:
    context = text[max(0, start - 500) : start]
    quoted = QUOTED_NAME_RE.search(context)
    if quoted:
        return quoted.group(1)
    lines = [line.strip() for line in context.splitlines() if line.strip()]
    for line in reversed(lines[-8:]):
        for key in BASELINE_KEYWORDS + LEARNED_KEYWORDS:
            if key.lower() in line.lower():
                tokens = re.findall(r"[A-Za-z0-9_.-]+", line)
                for token in tokens:
                    if key.lower() in token.lower():
                        return token
                return key
        match = NAME_RE.search(line)
        if match:
            return match.group(1)
    return f"{path.stem}_array_{index}"


def extract_from_text(path: Path, records: list[OrderRecord], rejected: list[RejectedOrder]) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    for index, match in enumerate(ARRAY_RE.finditer(text), start=1):
        name = infer_text_name(path, text, match.start(), index)
        add_record(records, rejected, path, name, match.group(0), "text-array", None)


def dedupe_records(records: list[OrderRecord]) -> list[OrderRecord]:
    chosen: dict[tuple[str, str], OrderRecord] = {}
    for record in records:
        key = (record.strategy_name.lower(), record.category)
        existing = chosen.get(key)
        if existing is None:
            chosen[key] = record
            continue
        if record_priority(record) < record_priority(existing):
            chosen[key] = record
    return trim_records(sorted(chosen.values(), key=lambda r: (r.category, r.family, r.strategy_name, r.source_file)))


def record_priority(record: OrderRecord) -> tuple[int, int, int, str]:
    source = record.source_file.lower()
    source_rank = 5
    if "stage2_reference" in source:
        source_rank = 0
    elif "teacher_labels" in source or "full48_teacher_labels_loaded_validated" in source:
        source_rank = 1
    elif "strategy_metadata" in source:
        source_rank = 2
    elif "outputs" in source:
        source_rank = 3
    metric_rank = -len(record.metrics)
    name_rank = 0 if re.search(r"(rlu2m|rlu2|rl20|rlv2|gnnu|formal_raster|odd_even|center_out|edge_in|maximin|smartscan|multi_lag|windowed|block_interleaved|center_edge|method_c|mc_)", record.strategy_name.lower()) else 1
    return (source_rank, metric_rank, name_rank, record.source_file)


def trim_records(records: list[OrderRecord]) -> list[OrderRecord]:
    grouped: dict[str, list[OrderRecord]] = defaultdict(list)
    for record in records:
        grouped[record.category].append(record)
    selected: list[OrderRecord] = []
    for category, limit in [
        ("engineering_baseline", MAX_BASELINE_RECORDS),
        ("learned_gnn_rl", MAX_LEARNED_RECORDS),
    ]:
        category_records = sorted(grouped.get(category, []), key=record_priority)
        selected.extend(category_records[:limit])
    return sorted(selected, key=lambda r: (r.category, r.family, r.strategy_name, r.source_file))


def directed_pair_overlap(a: list[int], b: list[int]) -> float:
    pairs_a = {(a[i], a[i + 1]) for i in range(31)}
    pairs_b = {(b[i], b[i + 1]) for i in range(31)}
    return len(pairs_a & pairs_b) / 31.0


def undirected_pair_overlap(a: list[int], b: list[int]) -> float:
    pairs_a = {tuple(sorted((a[i], a[i + 1]))) for i in range(31)}
    pairs_b = {tuple(sorted((b[i], b[i + 1]))) for i in range(31)}
    return len(pairs_a & pairs_b) / 31.0


def kendall_distance(a: list[int], b: list[int]) -> float:
    pos_b = {track: idx for idx, track in enumerate(b)}
    seq = [pos_b[track] for track in a]
    inversions = 0
    for i in range(len(seq)):
        ai = seq[i]
        for j in range(i + 1, len(seq)):
            if ai > seq[j]:
                inversions += 1
    return inversions / (len(seq) * (len(seq) - 1) / 2)


def spearman_rank_distance(a: list[int], b: list[int]) -> float:
    pos_a = {track: idx for idx, track in enumerate(a)}
    pos_b = {track: idx for idx, track in enumerate(b)}
    sum_sq = sum((pos_a[track] - pos_b[track]) ** 2 for track in range(32))
    max_sum_sq = 32 * (32**2 - 1) / 3
    return sum_sq / max_sum_sq


def jump_summary(order: list[int]) -> dict[str, float | int]:
    jumps = [abs(order[i + 1] - order[i]) for i in range(31)]
    return {
        "mean_jump": statistics.fmean(jumps),
        "median_jump": statistics.median(jumps),
        "max_jump": max(jumps),
        "min_jump": min(jumps),
        "std_jump": statistics.pstdev(jumps),
        "count_jump_1": sum(1 for jump in jumps if jump == 1),
        "count_jump_le_2": sum(1 for jump in jumps if jump <= 2),
        "count_jump_ge_8": sum(1 for jump in jumps if jump >= 8),
        "count_jump_ge_16": sum(1 for jump in jumps if jump >= 16),
    }


def edge_center_profile(order: list[int]) -> dict[str, float | int]:
    pos = {track: idx for idx, track in enumerate(order)}
    edge_steps = [pos[track] for track in EDGE_TRACKS]
    center_steps = [pos[track] for track in CENTER_TRACKS]
    mean_edge = statistics.fmean(edge_steps) / 31.0
    mean_center = statistics.fmean(center_steps) / 31.0
    return {
        "mean_edge_step_norm": mean_edge,
        "mean_center_step_norm": mean_center,
        "first_edge_step": min(edge_steps),
        "first_center_step": min(center_steps),
        "last_edge_step": max(edge_steps),
        "last_center_step": max(center_steps),
        "edge_before_center_score": mean_center - mean_edge,
    }


def left_right_balance(order: list[int]) -> dict[str, float | int]:
    left = 0
    right = 0
    imbalances: list[int] = []
    for track in order:
        if track <= 15:
            left += 1
        else:
            right += 1
        imbalances.append(abs(left - right))
    return {
        "mean_abs_lr_imbalance": statistics.fmean(imbalances),
        "max_abs_lr_imbalance": max(imbalances),
        "final_abs_lr_imbalance": imbalances[-1],
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        seen: list[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        fieldnames = seen or ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def matrix_rows(records: list[OrderRecord], metric_fn) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels = unique_labels(records)
    for record_a, label_a in zip(records, labels):
        row: dict[str, Any] = {"strategy_name": label_a}
        for record_b, label_b in zip(records, labels):
            row[label_b] = round(metric_fn(record_a.scan_order, record_b.scan_order), 6)
        rows.append(row)
    return rows


def unique_labels(records: list[OrderRecord]) -> list[str]:
    counts: dict[str, int] = defaultdict(int)
    labels: list[str] = []
    for record in records:
        base = sanitize_name(record.strategy_name)
        counts[base] += 1
        labels.append(base if counts[base] == 1 else f"{base}_{counts[base]}")
    return labels


def summary_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "max": None}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def format_stat(stats: dict[str, float | None]) -> str:
    if stats["min"] is None:
        return "n/a"
    return f"min={stats['min']:.4f}, median={stats['median']:.4f}, max={stats['max']:.4f}"


def compute_outputs(records: list[OrderRecord]) -> tuple[list[str], dict[str, Any]]:
    outputs: list[str] = []
    labels = unique_labels(records)
    label_by_record = {id(record): label for record, label in zip(records, labels)}

    order_rows: list[dict[str, Any]] = []
    for record in records:
        row = {
            "strategy_name": record.strategy_name,
            "label": label_by_record[id(record)],
            "category": record.category,
            "family": record.family,
            "source_file": record.source_file,
            "source_kind": record.source_kind,
            "scan_order": json.dumps(record.scan_order),
            "has_teacher_metrics": bool(record.metrics),
        }
        row.update(record.metrics)
        order_rows.append(row)
    outputs.append(str(OUTPUT_DIR / "fixed32_orders_extracted.csv"))
    write_csv(Path(outputs[-1]), order_rows)

    matrix_specs = [
        ("adjacent_pair_overlap_matrix.csv", directed_pair_overlap),
        ("undirected_adjacent_pair_overlap_matrix.csv", undirected_pair_overlap),
        ("kendall_distance_matrix.csv", kendall_distance),
        ("spearman_rank_distance_matrix.csv", spearman_rank_distance),
    ]
    for filename, metric_fn in matrix_specs:
        path = OUTPUT_DIR / filename
        write_csv(path, matrix_rows(records, metric_fn))
        outputs.append(str(path))

    jump_rows = []
    edge_rows = []
    lr_rows = []
    for record in records:
        base = {
            "strategy_name": record.strategy_name,
            "label": label_by_record[id(record)],
            "category": record.category,
            "family": record.family,
        }
        jump_rows.append({**base, **jump_summary(record.scan_order)})
        edge_rows.append({**base, **edge_center_profile(record.scan_order)})
        lr_rows.append({**base, **left_right_balance(record.scan_order)})

    for filename, rows in [
        ("jump_length_summary.csv", jump_rows),
        ("edge_center_timing_profile.csv", edge_rows),
        ("left_right_balance_summary.csv", lr_rows),
    ]:
        path = OUTPUT_DIR / filename
        write_csv(path, rows)
        outputs.append(str(path))

    baselines = [record for record in records if record.category == "engineering_baseline"]
    learned = [record for record in records if record.category == "learned_gnn_rl"]
    rl_vs_rows = []
    directed_values: list[float] = []
    kendall_values: list[float] = []
    closest_by_learned: dict[str, dict[str, Any]] = {}
    for rl in learned:
        best: dict[str, Any] | None = None
        for baseline in baselines:
            directed = directed_pair_overlap(rl.scan_order, baseline.scan_order)
            undirected = undirected_pair_overlap(rl.scan_order, baseline.scan_order)
            kendall = kendall_distance(rl.scan_order, baseline.scan_order)
            spearman = spearman_rank_distance(rl.scan_order, baseline.scan_order)
            directed_values.append(directed)
            kendall_values.append(kendall)
            row = {
                "learned_strategy": rl.strategy_name,
                "learned_family": rl.family,
                "baseline_strategy": baseline.strategy_name,
                "baseline_family": baseline.family,
                "directed_adjacent_overlap": round(directed, 6),
                "undirected_adjacent_overlap": round(undirected, 6),
                "kendall_distance": round(kendall, 6),
                "spearman_rank_distance": round(spearman, 6),
            }
            rl_vs_rows.append(row)
            if best is None or (kendall, -directed) < (best["kendall_distance"], -best["directed_adjacent_overlap"]):
                best = row
        if best is not None:
            closest_by_learned[rl.strategy_name] = best

    path = OUTPUT_DIR / "rl_vs_baseline_novelty_summary.csv"
    write_csv(path, rl_vs_rows)
    outputs.append(str(path))

    directed_stats = summary_stats(directed_values)
    kendall_stats = summary_stats(kendall_values)
    closest_families = defaultdict(int)
    for row in closest_by_learned.values():
        closest_families[str(row["baseline_family"])] += 1
    closest_family = max(closest_families.items(), key=lambda item: item[1])[0] if closest_families else "n/a"

    distinct_supported = (
        directed_stats["median"] is not None
        and kendall_stats["median"] is not None
        and directed_stats["median"] <= 0.25
        and kendall_stats["median"] >= 0.35
    )
    if distinct_supported:
        safe_claim = "The learned fixed-32 candidates are structurally distinct from the audited engineering baselines under adjacent-transition and permutation-distance metrics."
        if directed_stats["max"] is not None and directed_stats["max"] >= 0.999:
            safe_claim += " This aggregate finding coexists with isolated learned-baseline exact or near-exact structural matches."
    else:
        safe_claim = "The learned fixed-32 candidates show substantial structural similarity to existing engineering heuristics; Stage 3 should frame variable-N work as graph-policy formalisation and improvement of known heuristics rather than discovery of a fully novel scan-order family."

    policy_summary = [
        {
            "metric": "legal_fixed32_orders",
            "value": len(records),
            "notes": "Accepted exact permutations of 0..31 only.",
        },
        {
            "metric": "engineering_baseline_orders",
            "value": len(baselines),
            "notes": "Baseline-labelled or heuristic-labelled legal orders.",
        },
        {
            "metric": "learned_gnn_rl_orders",
            "value": len(learned),
            "notes": "GNN/RL-labelled legal orders.",
        },
        {
            "metric": "rl_vs_baseline_directed_overlap",
            "value": format_stat(directed_stats),
            "notes": "Directed adjacent-pair overlap over all learned-baseline pairs.",
        },
        {
            "metric": "rl_vs_baseline_kendall_distance",
            "value": format_stat(kendall_stats),
            "notes": "Normalized Kendall inversion distance over all learned-baseline pairs.",
        },
        {
            "metric": "closest_baseline_family_mode",
            "value": closest_family,
            "notes": "Most frequent closest baseline family by minimum Kendall distance per learned candidate.",
        },
        {
            "metric": "safe_claim",
            "value": safe_claim,
            "notes": "Bounded structural audit wording only.",
        },
    ]
    path = OUTPUT_DIR / "policy_novelty_summary.csv"
    write_csv(path, policy_summary)
    outputs.append(str(path))

    plot_outputs, plotting_status = maybe_write_plots(records, label_by_record, rl_vs_rows, jump_rows, edge_rows)
    outputs.extend(plot_outputs)

    analysis = {
        "baseline_count": len(baselines),
        "learned_count": len(learned),
        "directed_stats": directed_stats,
        "kendall_stats": kendall_stats,
        "closest_family": closest_family,
        "safe_claim": safe_claim,
        "distinct_supported": distinct_supported,
        "teacher_metrics_linked": any("U2" in r.metrics for r in baselines) and any("U2" in r.metrics for r in learned),
        "best_baseline_u2": best_by_u2(baselines),
        "best_learned_u2": best_by_u2(learned),
        "plotting_status": plotting_status,
    }
    return outputs, analysis


def best_by_u2(records: list[OrderRecord]) -> dict[str, Any] | None:
    labelled = [record for record in records if isinstance(record.metrics.get("U2"), float)]
    if not labelled:
        return None
    best = min(labelled, key=lambda record: float(record.metrics["U2"]))
    return {"strategy_name": best.strategy_name, "family": best.family, "U2": best.metrics["U2"]}


def maybe_write_plots(
    records: list[OrderRecord],
    label_by_record: dict[int, str],
    rl_vs_rows: list[dict[str, Any]],
    jump_rows: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
) -> tuple[list[str], str]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local environment
        return [], f"plotting skipped: matplotlib unavailable ({exc})"

    outputs: list[str] = []
    if records:
        labels = [label_by_record[id(record)] for record in records]
        for filename, metric_fn, title in [
            ("adjacent_pair_overlap_heatmap.png", directed_pair_overlap, "Directed Adjacent-Pair Overlap"),
            ("kendall_distance_heatmap.png", kendall_distance, "Kendall Distance"),
        ]:
            data = [[metric_fn(a.scan_order, b.scan_order) for b in records] for a in records]
            fig_size = max(6, min(18, len(records) * 0.18))
            fig, ax = plt.subplots(figsize=(fig_size, fig_size))
            image = ax.imshow(data, aspect="auto", cmap="viridis")
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            path = OUTPUT_DIR / filename
            fig.tight_layout()
            fig.savefig(path, dpi=140)
            plt.close(fig)
            outputs.append(str(path))

    jumps_by_category: dict[str, list[float]] = defaultdict(list)
    for row in jump_rows:
        jumps_by_category[str(row["category"])].append(float(row["mean_jump"]))
    fig, ax = plt.subplots(figsize=(7, 4))
    for category, values in sorted(jumps_by_category.items()):
        ax.hist(values, alpha=0.55, label=category, bins=min(15, max(3, len(values))))
    ax.set_title("Mean Jump-Length Distribution")
    ax.set_xlabel("mean jump")
    ax.set_ylabel("count")
    ax.legend()
    path = OUTPUT_DIR / "jump_length_distribution.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    outputs.append(str(path))

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {"engineering_baseline": "tab:blue", "learned_gnn_rl": "tab:orange"}
    for row in edge_rows:
        ax.scatter(
            float(row["mean_edge_step_norm"]),
            float(row["mean_center_step_norm"]),
            color=colors.get(str(row["category"]), "tab:gray"),
            alpha=0.75,
        )
    ax.set_title("Edge/Center Timing Profile")
    ax.set_xlabel("mean edge step / 31")
    ax.set_ylabel("mean center step / 31")
    path = OUTPUT_DIR / "edge_center_timing_profile.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    outputs.append(str(path))

    return outputs, "plots written"


def extract_all() -> tuple[list[OrderRecord], list[RejectedOrder], int, int]:
    global REJECTED_TOTAL
    REJECTED_TOTAL = 0
    records: list[OrderRecord] = []
    rejected: list[RejectedOrder] = []
    scanned = 0
    for path in discover_input_files():
        suffix = path.suffix.lower()
        if suffix in FORBIDDEN_EXTENSIONS:
            continue
        try:
            if suffix == ".csv":
                extract_from_csv(path, records, rejected)
            elif suffix == ".json":
                extract_from_json(path, records, rejected)
            elif suffix in {".md", ".txt", ".py"}:
                extract_from_text(path, records, rejected)
            else:
                continue
            scanned += 1
        except (OSError, UnicodeError, csv.Error, json.JSONDecodeError) as exc:
            record_rejection(
                rejected,
                RejectedOrder(
                    source_file=str(path),
                    candidate_name=path.stem,
                    reason=f"file read/parse skipped: {exc}",
                    length="n/a",
                    raw_preview="",
                )
            )
    return dedupe_records(records), rejected, scanned, REJECTED_TOTAL


def write_rejection_table(rejected: list[RejectedOrder]) -> str:
    path = OUTPUT_DIR / "extraction_rejection_audit.csv"
    rows = [
        {
            "source_file": item.source_file,
            "candidate_name": item.candidate_name,
            "reason": item.reason,
            "length": item.length,
            "raw_preview": item.raw_preview,
        }
        for item in rejected
    ]
    write_csv(path, rows, ["source_file", "candidate_name", "reason", "length", "raw_preview"])
    return str(path)


def verdict_for(records: list[OrderRecord], analysis: dict[str, Any] | None) -> str:
    if not records:
        return "FAIL_FIXED32_POLICY_NOVELTY_NO_LEGAL_ORDERS"
    if analysis is None:
        return "WARNING_FIXED32_POLICY_NOVELTY_PARTIAL_INPUTS"
    if analysis["baseline_count"] == 0 or analysis["learned_count"] == 0:
        return "WARNING_FIXED32_POLICY_NOVELTY_PARTIAL_INPUTS"
    return "PASS_FIXED32_POLICY_NOVELTY_AUDIT_READY"


def write_manifest(
    files_scanned_count: int,
    records: list[OrderRecord],
    rejected: list[RejectedOrder],
    rejected_count: int,
    outputs_written: list[str],
    verdict: str,
) -> None:
    baseline_count = sum(1 for record in records if record.category == "engineering_baseline")
    learned_count = sum(1 for record in records if record.category == "learned_gnn_rl")
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_root": str(SOURCE_ROOT),
        "target_root": str(TARGET_ROOT),
        "python_executable": sys.executable,
        "run_id": RUN_ID,
        "verdict": verdict,
        "files_scanned_count": files_scanned_count,
        "legal_orders_count": len(records),
        "baseline_orders_count": baseline_count,
        "learned_orders_count": learned_count,
        "rejected_orders_count": rejected_count,
        "rejection_audit_rows_written": len(rejected),
        "outputs_written": outputs_written,
        "forbidden_actions_confirmed": {
            "no_abaqus_jobs": True,
            "no_datacheck": True,
            "no_odb_opened": True,
            "no_cae_generated": True,
            "no_inp_generated": True,
            "no_jnl_generated": True,
            "no_model_training": True,
            "stage2_source_read_only": True,
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def write_report(
    files_scanned_count: int,
    records: list[OrderRecord],
    rejected: list[RejectedOrder],
    rejected_count: int,
    analysis: dict[str, Any] | None,
    verdict: str,
    outputs_written: list[str],
) -> None:
    baseline_count = sum(1 for record in records if record.category == "engineering_baseline")
    learned_count = sum(1 for record in records if record.category == "learned_gnn_rl")
    teacher_linked = bool(analysis and analysis["teacher_metrics_linked"])
    directed_text = format_stat(analysis["directed_stats"]) if analysis else "n/a"
    kendall_text = format_stat(analysis["kendall_stats"]) if analysis else "n/a"
    closest_family = analysis["closest_family"] if analysis else "n/a"
    safe_claim = analysis["safe_claim"] if analysis else "No safe structural claim can be made because legal inputs were incomplete."
    plotting_status = analysis["plotting_status"] if analysis else "plotting skipped"

    teacher_section = "Teacher metric linkage was not complete enough for a safe U2 structural-performance comparison."
    if analysis and teacher_linked:
        teacher_section = (
            "Teacher U2 metrics were linked for at least one labelled baseline and one labelled learned candidate.\n\n"
            f"- Best labelled engineering baseline by U2: `{analysis['best_baseline_u2']}`\n"
            f"- Best labelled GNN/RL candidate by U2: `{analysis['best_learned_u2']}`\n\n"
            "Run_02 remains a structural audit only; it is not new teacher validation."
        )

    outputs_text = "\n".join(f"- `{path}`" for path in outputs_written)

    report = f"""# Run 02 Fixed-32 Policy Novelty Audit Report

## Executive Verdict

{verdict}

## What Was Audited

- Legal fixed-32 engineering baseline orders: `{baseline_count}`
- Legal fixed-32 GNN/RL orders: `{learned_count}`
- Legal fixed-32 orders total: `{len(records)}`
- Source files scanned: `{files_scanned_count}`
- Rejected/non-legal order candidates: `{rejected_count}`
- Rejection audit rows written: `{len(rejected)}` of max `{MAX_REJECTION_ROWS}`
- Teacher metrics linked: `{teacher_linked}`

## Guardrails

- No Abaqus jobs.
- No datacheck.
- No ODB opened.
- No CAE/INP/JNL generated.
- No model training.
- Stage 2 source was read-only.
- No new physical candidates were generated for CAE.

## Core Findings

- RL/GNN vs engineering baseline directed adjacent-pair overlap: `{directed_text}`
- RL/GNN vs engineering baseline Kendall distance: `{kendall_text}`
- Closest baseline family mode across learned candidates: `{closest_family}`
- Structural interpretation: {safe_claim}

## Teacher U2 Linkage

{teacher_section}

## Claim Boundary

This audit does not prove variable-N generalisation. It does not prove arbitrary-N optimisation. It does not prove masked transfer. It does not prove SurfaceT optimisation. It does not replace Abaqus teacher validation.

## Outputs

{outputs_text}

Plotting status: `{plotting_status}`

## Recommended Next Run

`run_03_variable_n_graph_feature_builder`

## Final Verdict

{verdict}
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> int:
    ensure_dirs()
    records, rejected, files_scanned_count, rejected_count = extract_all()
    rejection_path = write_rejection_table(rejected)
    outputs_written = [rejection_path]
    analysis: dict[str, Any] | None = None
    if records:
        generated, analysis = compute_outputs(records)
        outputs_written.extend(generated)
    verdict = verdict_for(records, analysis)
    write_manifest(files_scanned_count, records, rejected, rejected_count, outputs_written, verdict)
    outputs_written.append(str(MANIFEST_PATH))
    write_report(files_scanned_count, records, rejected, rejected_count, analysis, verdict, outputs_written)
    outputs_written.append(str(REPORT_PATH))

    baseline_count = sum(1 for record in records if record.category == "engineering_baseline")
    learned_count = sum(1 for record in records if record.category == "learned_gnn_rl")
    print(f"Files scanned: {files_scanned_count}")
    print(f"Legal fixed-32 orders: {len(records)}")
    print(f"Engineering baseline orders: {baseline_count}")
    print(f"Learned GNN/RL orders: {learned_count}")
    print(f"Rejected/non-legal candidates: {rejected_count}")
    print(f"Rejection audit rows written: {len(rejected)}")
    if analysis:
        print(f"Teacher metrics linked: {analysis['teacher_metrics_linked']}")
        print(f"RL/GNN vs baseline directed overlap: {format_stat(analysis['directed_stats'])}")
        print(f"RL/GNN vs baseline Kendall distance: {format_stat(analysis['kendall_stats'])}")
        print(f"Closest baseline family mode: {analysis['closest_family']}")
        print(f"Plotting status: {analysis['plotting_status']}")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Report: {REPORT_PATH}")
    print(verdict)
    return 1 if verdict == "FAIL_FIXED32_POLICY_NOVELTY_NO_LEGAL_ORDERS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
