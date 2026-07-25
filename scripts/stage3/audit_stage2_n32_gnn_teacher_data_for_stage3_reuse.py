"""Audit Stage 2 N32 / 32-track teacher-labelled data for Stage 3 reuse.

Read-only audit:
- searches D:/ and E:/ RL-LAM-ScanOpt trees for small text/CSV/JSON artifacts;
- inspects likely N32 teacher-label schemas;
- compares the best candidate table(s) against the Stage 3 combined172 schema;
- writes an ingestion preview only, without merging or ranking.

Forbidden operations intentionally absent: Abaqus, ODB access, solver/job submission,
candidate generation, training, CAE/INP/JNL generation, git operations.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"pandas is required for this audit: {exc}")


SCRIPT_PATH = Path(__file__).resolve()
OUTPUT_DIR = Path(r"E:\Projects\RL-LAM-ScanOpt\outputs\stage3_n32_legacy_teacher_data_audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

D_ROOT = Path(r"D:\Projects\RL-LAM-ScanOpt")
E_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
COMBINED172_PATH = (
    E_ROOT
    / "outputs"
    / "stage3_run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking"
    / "combined172_RL_ready_dataset.csv"
)

SEARCH_ROOTS = [
    D_ROOT / "outputs",
    D_ROOT / "rl-training",
    D_ROOT / "docs" / "stage2",
    D_ROOT / "artifacts",
    D_ROOT / "LDED_2D_CAE_Framework" / "cae_models",
    D_ROOT / "cae_model",
    E_ROOT / "outputs",
    E_ROOT / "scripts",
    E_ROOT / "docs",
]

ALLOWED_SUFFIXES = {".csv", ".json", ".md", ".txt", ".parquet"}
SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ipynb_checkpoints",
    "node_modules",
}
KEYWORDS = [
    "n32",
    "32track",
    "32-track",
    "full32",
    "full_32",
    "teacher",
    "label",
    "labels",
    "metrics",
    "canonical",
    "gnn",
    "graph",
    "rl",
    "rlu2",
    "rlu2m",
    "rl20",
    "odb",
    "postprocess",
    "postprocessed",
    "surface",
    "surfacet",
    "gradient",
]

METRIC_ALIASES = {
    "u2_range": [
        "u2_range",
        "pure_u2",
        "pure_U2",
        "masked_xge4_final_U2_range",
        "teacher_U2",
        "teacher_pure_U2",
        "teacher_U2_primary_score",
        "teacher_U_primary_score",
    ],
    "peeq_max": [
        "peeq_max",
        "PEEQ_max",
        "teacher_PEEQ_primary_score",
        "teacher_PEEQ",
        "peeq_guard",
        "PEEQ",
        "masked_xge4_final_PEEQ_max",
    ],
    "surface_t_proxy": [
        "surface_t_proxy",
        "SurfaceT",
        "surfaceT",
        "surface_tensile_primary",
        "teacher_SurfaceT",
        "teacher_surface_tensile_primary",
    ],
    "mises_max": [
        "mises_max",
        "Mises_max",
        "mises_P95_top_band",
        "mises_top5_scan_region",
        "teacher_mises",
        "Mises",
    ],
}

SCAN_ORDER_COLUMNS = [
    "scan_order",
    "order_json",
    "order_compact",
    "order",
    "scanOrder",
    "track_order",
]
ID_COLUMNS = [
    "strategy_id",
    "strategy_name",
    "candidate_id",
    "job_name",
    "model_id",
    "reference_id",
]
STATUS_COLUMNS = [
    "teacher_validation_status",
    "teacher_validation_status_if_available",
    "teacher_validation",
    "final_cooling_success",
    "extraction_success",
    "status",
    "verdict",
]
SOURCE_COLUMNS = [
    "source_run",
    "source_batch",
    "source_file",
    "source_label_file",
    "dataset_source",
    "source_folder",
]
FINAL_STEP_COLUMNS = [
    "final_step",
    "step_name",
    "final_cooling_step",
    "final_cool",
    "final_cooling_success",
    "extraction_mode",
    "fields_extracted",
    "extracted_fields",
]


def safe_rel(path: Path) -> str:
    for root in [D_ROOT, E_ROOT]:
        try:
            return str(path.relative_to(root))
        except Exception:
            pass
    return str(path)


def lower_cols(cols: Iterable[str]) -> Dict[str, str]:
    return {str(c).lower(): str(c) for c in cols}


def find_col(cols: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    cmap = lower_cols(cols)
    for alias in aliases:
        if alias.lower() in cmap:
            return cmap[alias.lower()]
    for alias in aliases:
        a = alias.lower()
        for c in cols:
            if a == str(c).lower():
                return str(c)
    for alias in aliases:
        a = alias.lower()
        for c in cols:
            cl = str(c).lower()
            if a in cl or cl in a:
                return str(c)
    return None


def parse_order(value: Any) -> Optional[List[int]]:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, list):
        try:
            vals = [int(x) for x in value]
            return vals
        except Exception:
            return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    # JSON/list literals.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [int(x) for x in parsed]
    except Exception:
        pass
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return [int(x) for x in parsed]
    except Exception:
        pass
    # Compact comma/space separated sequence.
    nums = re.findall(r"-?\d+", text)
    if len(nums) >= 2:
        try:
            return [int(x) for x in nums]
        except Exception:
            return None
    return None


def is_legal_n32(order: Optional[List[int]]) -> bool:
    return bool(order) and len(order) == 32 and sorted(order) == list(range(32))


def order_json(order: Sequence[int]) -> str:
    return json.dumps([int(x) for x in order], separators=(",", ":"))


def order_hash(order: Sequence[int]) -> str:
    return hashlib.sha1(order_json(order).encode("utf-8")).hexdigest()[:16]


def sample_text_mentions(path: Path, max_bytes: int = 8192) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return f.read(max_bytes)
    except Exception:
        return ""


def file_keyword_score(path: Path) -> Tuple[int, List[str]]:
    hay = (str(path.name) + " " + str(path.parent)).lower()
    hits = [kw for kw in KEYWORDS if kw.lower() in hay]
    return len(hits), hits


def discover_files() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
            for fn in filenames:
                path = Path(dirpath) / fn
                if path.suffix.lower() not in ALLOWED_SUFFIXES:
                    continue
                key = str(path).lower()
                if key in seen:
                    continue
                seen.add(key)
                size = path.stat().st_size if path.exists() else 0
                # Avoid loading large text dumps; keep inventory if filename is promising.
                score, filename_hits = file_keyword_score(path)
                content_hits: List[str] = []
                if score < 2 and path.suffix.lower() in {".md", ".txt", ".json", ".csv"} and size <= 2_000_000:
                    txt = sample_text_mentions(path).lower()
                    content_hits = [kw for kw in KEYWORDS if kw.lower() in txt]
                    score += min(len(content_hits), 5)
                if score > 0:
                    rows.append(
                        {
                            "file_path": str(path),
                            "relative_path": safe_rel(path),
                            "suffix": path.suffix.lower(),
                            "size_bytes": size,
                            "keyword_score": score,
                            "filename_keyword_hits": ";".join(filename_hits),
                            "content_keyword_hits_sample": ";".join(content_hits[:20]),
                        }
                    )
    rows.sort(key=lambda r: (-int(r["keyword_score"]), r["file_path"]))
    return rows


def read_table(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path, low_memory=False)
        if path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
            if isinstance(data, list):
                return pd.DataFrame(data)
            if isinstance(data, dict):
                for key in ["rows", "data", "records", "items"]:
                    if isinstance(data.get(key), list):
                        return pd.DataFrame(data[key])
                return pd.json_normalize(data)
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
    except Exception:
        return None
    return None


def has_numeric(df: pd.DataFrame, col: Optional[str]) -> int:
    if not col or col not in df.columns:
        return 0
    vals = pd.to_numeric(df[col], errors="coerce")
    return int(vals.notna().sum())


def detect_n32_rows(df: pd.DataFrame, scan_col: Optional[str]) -> Tuple[int, int]:
    if "n" in lower_cols(df.columns):
        ncol = lower_cols(df.columns)["n"]
        return int((pd.to_numeric(df[ncol], errors="coerce") == 32).sum()), 0
    if "N" in df.columns:
        return int((pd.to_numeric(df["N"], errors="coerce") == 32).sum()), 0
    n32 = 0
    valid_orders = 0
    if scan_col and scan_col in df.columns:
        sample = df[scan_col].dropna()
        # Full scan across plausible tables, capped only for huge archives.
        if len(sample) > 20000:
            sample = sample.head(20000)
        for val in sample:
            order = parse_order(val)
            if is_legal_n32(order):
                valid_orders += 1
        n32 = valid_orders
    return n32, valid_orders


def status_evidence(df: pd.DataFrame, cols: Sequence[str]) -> Tuple[str, int]:
    status_col = find_col(cols, STATUS_COLUMNS)
    if not status_col:
        return "", 0
    s = df[status_col].astype(str).str.lower()
    ok = int(
        s.str.contains("pass|success|true|extracted|complete|completed|full_field", regex=True, na=False).sum()
    )
    return status_col, ok


def inspect_schema(file_row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    path = Path(file_row["file_path"])
    if path.suffix.lower() not in {".csv", ".json", ".parquet"}:
        return None
    df = read_table(path)
    if df is None:
        return None
    cols = [str(c) for c in df.columns]
    if not cols:
        return None
    scan_col = find_col(cols, SCAN_ORDER_COLUMNS)
    id_col = find_col(cols, ID_COLUMNS)
    metric_cols = {k: find_col(cols, aliases) for k, aliases in METRIC_ALIASES.items()}
    status_col, ok_status_count = status_evidence(df, cols)
    source_col = find_col(cols, SOURCE_COLUMNS)
    final_cols = [c for c in cols if c.lower() in {x.lower() for x in FINAL_STEP_COLUMNS}]
    n32_rows, legal_n32_order_rows = detect_n32_rows(df, scan_col)
    metric_nonnull = {f"{k}_nonnull": has_numeric(df, col) for k, col in metric_cols.items()}
    dup_strategy = 0
    if id_col:
        dup_strategy = int(df[id_col].astype(str).duplicated().sum())
    dup_order = 0
    order_hashes: List[str] = []
    if scan_col:
        for val in df[scan_col].dropna().head(50000):
            order = parse_order(val)
            if is_legal_n32(order):
                order_hashes.append(order_hash(order))
        dup_order = len(order_hashes) - len(set(order_hashes))

    required_metric_count = sum(1 for v in metric_cols.values() if v)
    score = (
        min(int(len(df)), 500)
        + 150 * int(n32_rows >= 100)
        + 60 * required_metric_count
        + 80 * int(bool(scan_col))
        + 60 * int(ok_status_count > 0 or "canonical" in path.name.lower())
        + 40 * int("full_field" in " ".join(cols).lower() or "full_field" in str(path).lower())
    )
    gnn_hint = bool(re.search(r"gnn|graph|rlu2|rlu2m|rl20|rl_|rl-", str(path), re.I))
    return {
        "file_path": str(path),
        "relative_path": safe_rel(path),
        "row_count": int(len(df)),
        "column_count": int(len(cols)),
        "columns": "|".join(cols[:120]),
        "strategy_id_column": id_col or "",
        "scan_order_column": scan_col or "",
        "n32_rows_detected": int(n32_rows),
        "legal_n32_order_rows_detected": int(legal_n32_order_rows),
        "u2_column": metric_cols["u2_range"] or "",
        "peeq_column": metric_cols["peeq_max"] or "",
        "surface_t_column": metric_cols["surface_t_proxy"] or "",
        "mises_column": metric_cols["mises_max"] or "",
        **metric_nonnull,
        "teacher_status_column": status_col,
        "teacher_status_ok_like_rows": int(ok_status_count),
        "source_run_column": source_col or "",
        "final_step_or_extraction_columns": "|".join(final_cols),
        "duplicate_strategy_rows": int(dup_strategy),
        "duplicate_order_hash_rows_sample_or_all": int(dup_order),
        "sample_rows_json": df.head(3).to_json(orient="records", force_ascii=False),
        "gnn_rl_path_hint": gnn_hint,
        "candidate_score": int(score),
        "usable_for_stage3_n32": bool(
            n32_rows >= 1
            and bool(scan_col)
            and bool(id_col)
            and bool(metric_cols["u2_range"])
            and bool(metric_cols["peeq_max"])
            and bool(metric_cols["surface_t_proxy"])
        ),
    }


def save_csv_json(rows: List[Dict[str, Any]], stem: str) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / f"{stem}.csv", index=False)
    with (OUTPUT_DIR / f"{stem}.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def load_combined172_schema() -> Dict[str, Any]:
    if not COMBINED172_PATH.exists():
        return {"exists": False, "columns": []}
    try:
        df = pd.read_csv(COMBINED172_PATH, nrows=5)
        return {"exists": True, "columns": [str(c) for c in df.columns], "sample": df.head(2).to_dict("records")}
    except Exception as exc:
        return {"exists": True, "columns": [], "error": str(exc)}


def infer_family(strategy: str, source_path: str) -> str:
    s = f"{strategy} {source_path}".lower()
    if any(x in s for x in ["gnn", "gnnu", "graph"]):
        return "gnn_graph_policy"
    if any(x in s for x in ["rlu2m", "rlu2_", "rlv2", "rl20", "rls_", "rl_"]):
        return "rl_policy_or_rl_generated"
    if any(x in s for x in ["baseline", "32track_full", "odd_even", "maximin", "greedy", "center"]):
        return "stage2_full32_baseline_or_reference"
    return "stage2_n32_other"


def build_preview(best: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    path = Path(best["file_path"])
    df = read_table(path)
    if df is None:
        return [], {"error": "could_not_read_best_table"}
    cols = [str(c) for c in df.columns]
    scan_col = best.get("scan_order_column") or find_col(cols, SCAN_ORDER_COLUMNS)
    id_col = best.get("strategy_id_column") or find_col(cols, ID_COLUMNS)
    u2_col = best.get("u2_column") or find_col(cols, METRIC_ALIASES["u2_range"])
    peeq_col = best.get("peeq_column") or find_col(cols, METRIC_ALIASES["peeq_max"])
    surface_col = best.get("surface_t_column") or find_col(cols, METRIC_ALIASES["surface_t_proxy"])
    mises_col = best.get("mises_column") or find_col(cols, METRIC_ALIASES["mises_max"])
    status_col = best.get("teacher_status_column") or find_col(cols, STATUS_COLUMNS)

    rows: List[Dict[str, Any]] = []
    missing_order = 0
    duplicate_orders = 0
    seen_hash: set[str] = set()
    for idx, row in df.iterrows():
        order = parse_order(row.get(scan_col)) if scan_col else None
        if not is_legal_n32(order):
            missing_order += 1
            continue
        oh = order_hash(order)
        if oh in seen_hash:
            duplicate_orders += 1
        seen_hash.add(oh)
        strategy = str(row.get(id_col, "")).strip() if id_col else ""
        if not strategy or strategy.lower() in {"nan", "none"}:
            strategy = f"N32_source_row_{idx:05d}"
        source_family = infer_family(strategy, str(path))
        status = str(row.get(status_col, "")).strip() if status_col else ""
        if not status or status.lower() in {"nan", "none"}:
            status = "PASS_ODB_EXTRACTED_INFERRED_FROM_CANONICAL_FULL_FIELD_LABEL_TABLE"
        notes = []
        if peeq_col == "peeq_guard":
            notes.append("peeq_max mapped from Stage 2 peeq_guard diagnostic; verify threshold semantics before final merge")
        if surface_col == "surface_tensile_primary":
            notes.append("surface_t_proxy mapped from full-field surface_tensile_primary")
        if mises_col == "mises_P95_top_band":
            notes.append("mises_max mapped from mises_P95_top_band, not literal global max")
        rows.append(
            {
                "n": 32,
                "strategy_name": strategy,
                "dataset_source": (
                    "stage2_n32_gnn_rl_legacy"
                    if source_family in {"gnn_graph_policy", "rl_policy_or_rl_generated"}
                    else "stage2_n32_full32_legacy"
                ),
                "order_json": order_json(order),
                "order_hash": oh,
                "u2_range": row.get(u2_col) if u2_col else "",
                "peeq_max": row.get(peeq_col) if peeq_col else "",
                "surface_t_proxy": row.get(surface_col) if surface_col else "",
                "mises_max": row.get(mises_col) if mises_col else "",
                "teacher_validation_status": status,
                "source_file": str(path),
                "source_row_index": int(idx),
                "compatibility_notes": "; ".join(notes),
            }
        )
    meta = {
        "source_file": str(path),
        "input_rows": int(len(df)),
        "preview_rows": int(len(rows)),
        "rows_skipped_missing_or_invalid_n32_scan_order": int(missing_order),
        "duplicate_order_hash_rows_in_preview": int(duplicate_orders),
    }
    return rows, meta


def compatibility_rows(best: Dict[str, Any], preview_meta: Dict[str, Any], combined_schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    mapping = {
        "n": ("constant 32", True, "N=32 inferred from legal full 0..31 scan_order and Stage 2 32track source"),
        "strategy_name": (best.get("strategy_id_column", ""), bool(best.get("strategy_id_column")), ""),
        "order_json": (best.get("scan_order_column", ""), bool(best.get("scan_order_column")), ""),
        "u2_range": (best.get("u2_column", ""), bool(best.get("u2_column")), ""),
        "peeq_max": (best.get("peeq_column", ""), bool(best.get("peeq_column")), "May be peeq_guard if canonical table is used"),
        "surface_t_proxy": (best.get("surface_t_column", ""), bool(best.get("surface_t_column")), ""),
        "mises_max": (best.get("mises_column", ""), bool(best.get("mises_column")), "May be P95/top-band Mises diagnostic rather than global max"),
        "dataset_source": ("constant stage2_n32_gnn_rl_legacy/stage2_n32_full32_legacy", True, ""),
        "teacher_validation_status": (
            best.get("teacher_status_column", "") or "inferred from full-field canonical extraction",
            True,
            "",
        ),
    }
    for target, (source, ok, note) in mapping.items():
        rows.append(
            {
                "stage3_target_field": target,
                "source_field_or_rule": source,
                "mappable": bool(ok),
                "combined172_has_field": target in combined_schema.get("columns", []),
                "notes": note,
            }
        )
    rows.append(
        {
            "stage3_target_field": "row_count",
            "source_field_or_rule": str(preview_meta.get("preview_rows", 0)),
            "mappable": int(preview_meta.get("preview_rows", 0)) >= 100,
            "combined172_has_field": False,
            "notes": "Requires at least 100 reusable N32 rows for PASS",
        }
    )
    rows.append(
        {
            "stage3_target_field": "duplicate_order_hash_status",
            "source_field_or_rule": str(preview_meta.get("duplicate_order_hash_rows_in_preview", 0)),
            "mappable": int(preview_meta.get("duplicate_order_hash_rows_in_preview", 0)) == 0,
            "combined172_has_field": False,
            "notes": "Duplicates should be removed or documented before final Stage 3 merge",
        }
    )
    return rows


def report_md(
    inventory: List[Dict[str, Any]],
    schema_rows: List[Dict[str, Any]],
    best_rows: List[Dict[str, Any]],
    preview_meta: Dict[str, Any],
    verdict: str,
    compat_notes: str,
) -> str:
    best = best_rows[0] if best_rows else {}
    top_files = "\n".join(
        f"- `{r['file_path']}` rows={r.get('row_count','')} score={r.get('candidate_score','')}"
        for r in best_rows[:8]
    )
    return f"""# N32 Legacy Teacher Data Audit

Generated: {datetime.now().isoformat(timespec='seconds')}

## Purpose

Audit whether existing Stage 2 N32 / 32-track teacher-labelled CSV/JSON data can be reused as an N=32 group in the Stage 3 combined dataset.

This audit did not run Abaqus, did not open ODB files, did not train models, and did not generate candidates or CAE/INP/JNL files.

## Search Roots

- `{D_ROOT}`
- `{E_ROOT}`

## Candidate Files Found

- Candidate text/table files inventoried: `{len(inventory)}`
- CSV/JSON/Parquet schemas inspected: `{len(schema_rows)}`
- Tables with usable Stage 3 N32 mapping signal: `{sum(1 for r in schema_rows if r.get('usable_for_stage3_n32'))}`

## Best Teacher-Label Table Candidates

{top_files if top_files else '- No reusable table candidates found.'}

## Selected Best Source

- Source file: `{best.get('file_path', 'not found')}`
- Rows in source: `{best.get('row_count', 0)}`
- N32/legal order rows detected: `{best.get('n32_rows_detected', 0)}`
- Strategy column: `{best.get('strategy_id_column', '')}`
- Scan-order column: `{best.get('scan_order_column', '')}`
- U2 column: `{best.get('u2_column', '')}`
- PEEQ column: `{best.get('peeq_column', '')}`
- SurfaceT column: `{best.get('surface_t_column', '')}`
- Mises column: `{best.get('mises_column', '')}`

## Ingestion Preview

- Preview rows with legal N32 scan orders: `{preview_meta.get('preview_rows', 0)}`
- Rows skipped due missing/invalid N32 scan order: `{preview_meta.get('rows_skipped_missing_or_invalid_n32_scan_order', 0)}`
- Duplicate order-hash rows in preview: `{preview_meta.get('duplicate_order_hash_rows_in_preview', 0)}`

## Compatibility With Stage 3 combined172

{compat_notes}

## Teacher-Validation Evidence

The selected source is treated as teacher-labelled only when it contains explicit extraction/final-cooling/status columns or comes from canonical full-field ODB extraction outputs. The audit records inferred teacher status in the preview, but final ingestion should keep the source file and row index for traceability.

## Whether N32 Can Be Introduced As An Additional N Group

If the preview row count is sufficient and the mapped metrics are accepted, the discovered Stage 2 N32 table can be introduced as an additional N=32 group. The preview table is intentionally not merged into the Stage 3 combined dataset in this run.

## Risks And Missing Evidence

- Some Stage 2 fields are semantically close but not name-identical to Stage 3 fields, especially `peeq_guard` versus `peeq_max` and `mises_P95_top_band` versus `mises_max`.
- Final cooling / extracted-field metadata may be inferred from source provenance when explicit columns are absent.
- If a strict Stage 3 merge requires literal global `mises_max` or literal `peeq_max`, an additional mapping decision is needed.
- Duplicates, if any, must be removed or documented before the final combined dataset merge.

## Verdict

`{verdict}`

## Recommended Next Action

{"Create a Stage 3 ingestion run that imports this N32 preview as an additional N=32 group, after confirming metric-name semantics for PEEQ and Mises." if verdict.startswith("PASS") else "Repair missing mappings or locate a stronger source table before merging N32 into Stage 3."}
"""


def main() -> int:
    inventory = discover_files()
    save_csv_json(inventory, "n32_candidate_file_inventory")

    schema_rows: List[Dict[str, Any]] = []
    for row in inventory:
        inspected = inspect_schema(row)
        if inspected is not None:
            schema_rows.append(inspected)
    schema_rows.sort(key=lambda r: (-int(r["candidate_score"]), -int(r["row_count"]), r["file_path"]))
    save_csv_json(schema_rows, "n32_csv_json_schema_audit")

    best_rows = [
        r
        for r in schema_rows
        if r.get("usable_for_stage3_n32")
        and int(r.get("n32_rows_detected", 0)) >= 1
        and bool(r.get("u2_column"))
        and bool(r.get("peeq_column"))
        and bool(r.get("surface_t_column"))
    ]
    # Prefer canonical full-field labels when available, then large GNN/RL teacher metric tables.
    def best_key(r: Dict[str, Any]) -> Tuple[int, int, int, int]:
        p = str(r["file_path"]).lower()
        canonical = int("canonical_full_field_surface_gradient_teacher_labels_v03" in p)
        run84_gnn = int("gnn_rl_teacher_metric_table" in p)
        return (canonical, run84_gnn, int(r["n32_rows_detected"]), int(r["candidate_score"]))

    best_rows.sort(key=best_key, reverse=True)
    save_csv_json(best_rows, "n32_best_teacher_label_table_candidates")

    preview_rows: List[Dict[str, Any]] = []
    preview_meta: Dict[str, Any] = {}
    if best_rows:
        preview_rows, preview_meta = build_preview(best_rows[0])
    save_csv_json(preview_rows, "n32_stage3_ingestion_preview")

    combined_schema = load_combined172_schema()
    compat = compatibility_rows(best_rows[0], preview_meta, combined_schema) if best_rows else []
    save_csv_json(compat, "n32_stage3_schema_compatibility_check")

    preview_count = int(preview_meta.get("preview_rows", 0))
    duplicate_orders = int(preview_meta.get("duplicate_order_hash_rows_in_preview", 0))
    has_core_metrics = False
    if best_rows:
        b = best_rows[0]
        has_core_metrics = all(bool(b.get(k)) for k in ["u2_column", "peeq_column", "surface_t_column", "mises_column"])
    if preview_count >= 100 and has_core_metrics and duplicate_orders == 0:
        verdict = "PASS_N32_LEGACY_TEACHER_DATA_COMPATIBLE"
    elif preview_count >= 100 and has_core_metrics:
        verdict = "WARNING_N32_LEGACY_TEACHER_DATA_PARTIAL"
    elif preview_count > 0:
        verdict = "WARNING_N32_LEGACY_TEACHER_DATA_PARTIAL"
    else:
        verdict = "FAIL_N32_LEGACY_TEACHER_DATA_NOT_READY"

    compat_notes = "\n".join(
        f"- `{r['stage3_target_field']}` <- `{r['source_field_or_rule']}`: {'OK' if r['mappable'] else 'NOT READY'}"
        + (f" ({r['notes']})" if r.get("notes") else "")
        for r in compat
    )
    report = report_md(inventory, schema_rows, best_rows, preview_meta, verdict, compat_notes)
    (OUTPUT_DIR / "N32_LEGACY_TEACHER_DATA_AUDIT_REPORT.md").write_text(report, encoding="utf-8")

    summary = {
        "current_branch": get_git_branch(D_ROOT),
        "candidate_files_found": len(inventory),
        "schemas_inspected": len(schema_rows),
        "best_source": best_rows[0]["file_path"] if best_rows else "",
        "preview_rows": preview_count,
        "duplicate_order_hash_rows": duplicate_orders,
        "verdict": verdict,
        "output_dir": str(OUTPUT_DIR),
        "forbidden_actions": {
            "abaqus_run": False,
            "odb_opened": False,
            "solver_run": False,
            "training_run": False,
            "candidates_generated": False,
        },
    }
    (OUTPUT_DIR / "n32_audit_summary_v01.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def get_git_branch(repo: Path) -> str:
    head = repo / ".git" / "HEAD"
    try:
        text = head.read_text(encoding="utf-8", errors="ignore").strip()
        if text.startswith("ref:"):
            return text.split("/")[-1]
        return text[:12]
    except Exception:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
