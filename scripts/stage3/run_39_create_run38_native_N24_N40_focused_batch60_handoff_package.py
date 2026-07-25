from __future__ import annotations

import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
RUN_ID = "run_39_run38_native_N24_N40_focused_batch60_handoff_package"
RUN_NAME = "run_38 native N24/N40 focused batch60 handoff package"
SCRIPT_PATH = ROOT / "scripts" / "stage3" / "run_39_create_run38_native_N24_N40_focused_batch60_handoff_package.py"

RUN38_POOL = ROOT / "outputs" / "stage3_run_38_combined204_and_combined204_plus_N32_model_update_candidate_generation" / "run38_candidate_pool_scored.csv"
RUN38_OPTION_B = ROOT / "outputs" / "stage3_run_38_combined204_and_combined204_plus_N32_model_update_candidate_generation" / "run38_native_N24_N40_focused_batch32_candidate_orders.csv"
RUN38_REPORT = ROOT / "docs" / "stage3" / "runs" / "run_38_combined204_and_combined204_plus_N32_model_update_candidate_generation" / "RUN_38_COMBINED204_AND_COMBINED204_PLUS_N32_MODEL_UPDATE_CANDIDATE_GENERATION_REPORT.md"
COMBINED204_READY = ROOT / "outputs" / "stage3_run_37_N32_informed_native_batch32_teacher_metrics_ingestion_and_combined204_ranking" / "combined204_RL_ready_dataset.csv"
COMBINED204_TEACHER = ROOT / "outputs" / "stage3_run_37_N32_informed_native_batch32_teacher_metrics_ingestion_and_combined204_ranking" / "combined204_teacher_dataset.csv"
RUN36_ENRICHED = ROOT / "outputs" / "stage3_run_37_N32_informed_native_batch32_teacher_metrics_ingestion_and_combined204_ranking" / "run36_N32_informed_native_batch32_teacher_dataset_enriched.csv"
RUN27_ENRICHED = ROOT / "outputs" / "stage3_run_28_shortlist64_teacher_metrics_ingestion_and_combined172_ranking" / "run27_shortlist64_teacher_dataset_enriched.csv"
SUPERSEDED_RUN31_BATCH = ROOT / "outputs" / "stage3_run_30_run29_hybrid_policy_batch32_handoff_package" / "stage3_run30_hybrid_policy_batch32_candidate_orders.csv"

OUTPUT_DIR = ROOT / "outputs" / "stage3_run_39_run38_native_N24_N40_focused_batch60_handoff_package"
SCAN_ORDER_DIR = OUTPUT_DIR / "scan_orders"
REPORT_DIR = ROOT / "docs" / "stage3" / "runs" / RUN_ID
REPORT_PATH = REPORT_DIR / "RUN_39_RUN38_NATIVE_N24_N40_FOCUSED_BATCH60_HANDOFF_PACKAGE_REPORT.md"
MANIFEST_PATH = ROOT / "artifacts" / "manifests" / "stage3_run_39_manifest.json"
RUN_INDEX_PATH = ROOT / "docs" / "stage3" / "STAGE3_RUN_INDEX.md"

BATCH_NAME = "stage3_run39_native_N24_N40_focused_batch60_v01"
BATCH_OPTION = "native_N24_N40_focused_batch60"
EXPECTED_COUNTS = {24: 30, 40: 30}
EXPECTED_POOL_MIN = {24: 3000, 40: 3000}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def current_branch() -> str:
    try:
        return subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(payload), indent=2, sort_keys=False) + "\n", encoding="utf-8")


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, tuple):
        return [clean_json(v) for v in value]
    if hasattr(value, "item"):
        try:
            return clean_json(value.item())
        except Exception:
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if pd.isna(value):
        return None
    return value


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def parse_order(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(x) for x in value]
    text = str(value).strip()
    if not text:
        raise ValueError("empty order")
    if text.startswith("["):
        return [int(x) for x in json.loads(text)]
    cleaned = text.replace(",", "-").replace(";", "-").replace(" ", "")
    return [int(x) for x in cleaned.split("-") if x != ""]


def is_valid_order(value: Any, n: int) -> bool:
    try:
        order = parse_order(value)
    except Exception:
        return False
    return len(order) == n and sorted(order) == list(range(n))


def compact_order(value: Any) -> str:
    return "-".join(str(x) for x in parse_order(value))


def order_json(value: Any) -> str:
    return json.dumps(parse_order(value), separators=(",", ":"))


def load_hashes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    df = read_csv(path)
    if "order_hash" not in df.columns:
        return set()
    return {str(v) for v in df["order_hash"].dropna().astype(str) if str(v)}


def num(row: pd.Series, col: str, default: float = math.nan) -> float:
    try:
        val = float(row.get(col, default))
    except Exception:
        return default
    return val if math.isfinite(val) else default


def safe_name(text: Any, max_len: int = 28) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", str(text).strip().lower()).strip("_")
    return (cleaned or "candidate")[:max_len]


def short_bucket(row: pd.Series) -> str:
    bucket = str(row.get("selection_bucket", "") or row.get("candidate_source", "candidate"))
    mapping = {
        "N24_calibration_neighborhood": "n24_u2_near",
        "N40_new_best_neighborhood": "n40_u2_near",
        "surrogate_top_predicted": "surrogate_top",
        "surrogate_known_best_local_search": "surrogate_local",
        "graph_pointer_top": "graph_pointer",
        "diversity_coverage": "diversity",
        "hybrid_gnn_surrogate_agreement": "hybrid_agree",
        "hybrid_gnn_surrogate_disagreement": "hybrid_disagree",
        "uncertainty_calibration": "uncertainty",
        "sentinel_control": "sentinel",
    }
    source = str(row.get("candidate_source", ""))
    if "graph_pointer" in source:
        return "graph_pointer"
    return safe_name(mapping.get(bucket, bucket), max_len=24)


def score_sort(df: pd.DataFrame, mode: str = "default") -> pd.DataFrame:
    work = df.copy()
    for col in ["acquisition_score_rank_within_n", "surrogate_reward_pred_rank_within_n", "hybrid_score", "surrogate_reward_pred", "gnn_reward_pred", "uncertainty_score", "gnn_surrogate_disagreement", "novelty_distance_to_combined204_plus_N32"]:
        if col not in work.columns:
            work[col] = math.nan
        work[col] = pd.to_numeric(work[col], errors="coerce")
    if mode == "sentinel":
        return work.sort_values(["surrogate_reward_pred", "novelty_distance_to_combined204_plus_N32"], ascending=[True, False])
    if mode == "uncertainty":
        return work.sort_values(["uncertainty_score", "hybrid_score", "novelty_distance_to_combined204_plus_N32"], ascending=[False, False, False])
    if mode == "diversity":
        return work.sort_values(["novelty_distance_to_combined204_plus_N32", "hybrid_score"], ascending=[False, False])
    if mode == "disagreement":
        return work.sort_values(["gnn_surrogate_disagreement", "hybrid_score"], ascending=[False, False])
    return work.sort_values(["acquisition_score_rank_within_n", "hybrid_score", "surrogate_reward_pred"], ascending=[True, False, False])


def validate_pool(pool: pd.DataFrame) -> dict[str, Any]:
    counts = {int(k): int(v) for k, v in pool["n"].astype(int).value_counts().sort_index().to_dict().items()}
    required = ["n", "order_json", "candidate_id", "strategy_name", "order_hash"]
    missing_required = [col for col in required if col not in pool.columns]
    enough = {n: counts.get(n, 0) >= EXPECTED_POOL_MIN[n] for n in EXPECTED_POOL_MIN}
    validation = {
        "run_id": RUN_ID,
        "timestamp": now_iso(),
        "pool_path": str(RUN38_POOL),
        "pool_rows": int(len(pool)),
        "pool_counts": counts,
        "required_columns_missing": missing_required,
        "enough_N24_N40_candidates": enough,
        "reference_option_b_path": str(RUN38_OPTION_B),
    }
    validation["verdict"] = "PASS_RUN39_NATIVE_N24_N40_FOCUSED_BATCH60_POOL_READY" if not missing_required and all(enough.values()) else "FAIL_RUN39_POOL_VALIDATION"
    return validation


def select_batch(pool: pd.DataFrame, forbidden_hashes: set[str], option_b_hashes: set[str]) -> pd.DataFrame:
    selected_rows: list[pd.Series] = []
    used: set[str] = set()
    seed_counts: dict[tuple[int, str], int] = defaultdict(int)

    quota_by_n = {
        24: [
            ("u2_neighborhood", 8, "default"),
            ("surrogate", 6, "default"),
            ("graph_pointer", 5, "default"),
            ("hybrid_agree", 2, "default"),
            ("hybrid_disagree", 2, "disagreement"),
            ("uncertainty", 3, "uncertainty"),
            ("diversity", 3, "diversity"),
            ("sentinel", 1, "sentinel"),
        ],
        40: [
            ("u2_neighborhood", 8, "default"),
            ("surrogate", 6, "default"),
            ("graph_pointer", 5, "default"),
            ("hybrid_agree", 2, "default"),
            ("hybrid_disagree", 2, "disagreement"),
            ("uncertainty", 3, "uncertainty"),
            ("diversity", 3, "diversity"),
            ("sentinel", 1, "sentinel"),
        ],
    }

    def candidate_mask(df: pd.DataFrame, n: int, token: str) -> pd.Series:
        source = df["candidate_source"].astype(str)
        bucket = df["selection_bucket"].astype(str)
        if token == "u2_neighborhood":
            target = "N24_calibration_neighborhood" if n == 24 else "N40_new_best_neighborhood"
            return bucket.eq(target) | source.eq(target)
        if token == "surrogate":
            return bucket.eq("surrogate_top_predicted") | source.eq("surrogate_top_predicted") | source.eq("native_surrogate_top") | source.eq("native_surrogate_local_search")
        if token == "graph_pointer":
            return source.str.contains("graph_pointer", case=False, na=False)
        if token == "hybrid_agree":
            return bucket.eq("hybrid_gnn_surrogate_agreement") | source.eq("hybrid_gnn_surrogate_agreement")
        if token == "hybrid_disagree":
            return bucket.eq("hybrid_gnn_surrogate_disagreement") | source.eq("hybrid_gnn_surrogate_disagreement")
        if token == "uncertainty":
            return bucket.eq("uncertainty_calibration") | source.eq("uncertainty_calibration")
        if token == "diversity":
            return (bucket.eq("diversity_coverage") | source.eq("diversity_coverage")) & ~source.str.contains("graph_pointer", case=False, na=False)
        if token == "sentinel":
            return bucket.eq("sentinel_control") | source.eq("sentinel_control")
        return pd.Series([False] * len(df), index=df.index)

    def take_from(n: int, candidates: pd.DataFrame, count: int, mode: str) -> None:
        nonlocal selected_rows
        for _, row in score_sort(candidates, mode=mode).iterrows():
            if sum(1 for r in selected_rows if int(r["n"]) == n) >= EXPECTED_COUNTS[n]:
                break
            if count <= 0:
                break
            digest = str(row["order_hash"])
            if digest in used or digest in forbidden_hashes or digest in option_b_hashes:
                continue
            raw_seed = row.get("seed_strategy", "")
            seed = "" if pd.isna(raw_seed) else str(raw_seed)
            if seed and seed_counts[(n, seed)] >= 4:
                continue
            try:
                if not is_valid_order(row["order_json"], n):
                    continue
            except Exception:
                continue
            selected_rows.append(row)
            used.add(digest)
            if seed:
                seed_counts[(n, seed)] += 1
            count -= 1

    for n, quotas in quota_by_n.items():
        n_pool = pool[pool["n"].astype(int).eq(n)].copy()
        n_pool = n_pool[~n_pool["order_hash"].astype(str).isin(forbidden_hashes | option_b_hashes)].copy()
        for token, count, mode in quotas:
            before = sum(1 for r in selected_rows if int(r["n"]) == n)
            take_from(n, n_pool[candidate_mask(n_pool, n, token)], count, mode)
            after = sum(1 for r in selected_rows if int(r["n"]) == n)
            if after - before < count:
                raise RuntimeError(f"Underfilled Run39 quota for N{n} {token}: got {after - before}, expected {count}")
        remaining = n_pool[~n_pool["order_hash"].astype(str).isin(used | forbidden_hashes | option_b_hashes)]
        remaining_need = EXPECTED_COUNTS[n] - sum(1 for r in selected_rows if int(r["n"]) == n)
        if remaining_need:
            take_from(n, remaining, remaining_need, "default")

    selected = pd.DataFrame(selected_rows).copy()
    selected["n"] = selected["n"].astype(int)
    selected = selected.sort_values(["n", "acquisition_score_rank_within_n", "hybrid_score"], ascending=[True, True, False]).reset_index(drop=True)
    if len(selected) != 60 or {int(k): int(v) for k, v in selected["n"].value_counts().sort_index().to_dict().items()} != EXPECTED_COUNTS:
        raise RuntimeError(f"Selection failed: rows={len(selected)} counts={selected['n'].value_counts().sort_index().to_dict()}")
    return selected


def create_handoff(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    counters = defaultdict(int)
    for _, row in selected.iterrows():
        n = int(row["n"])
        counters[n] += 1
        handoff_name = f"S3R39N2440B60_N{n}_B{counters[n]:02d}_{short_bucket(row)}"
        order = order_json(row["order_json"])
        rows.append(
            {
                "run_id": RUN_ID,
                "batch_option": BATCH_OPTION,
                "batch_name": BATCH_NAME,
                "n": n,
                "handoff_strategy_name": handoff_name,
                "original_run38_candidate_id": row.get("candidate_id", ""),
                "original_run38_strategy_name": row.get("strategy_name", ""),
                "candidate_source": row.get("candidate_source", ""),
                "generation_method": row.get("generation_method", ""),
                "selection_bucket": row.get("selection_bucket", ""),
                "priority_role": row.get("selection_bucket", ""),
                "surrogate_prediction": row.get("surrogate_reward_pred", ""),
                "gnn_reward_prediction": row.get("gnn_reward_pred", ""),
                "graph_pointer_policy_score": row.get("graph_pointer_mean_logprob", ""),
                "hybrid_score": row.get("hybrid_score", ""),
                "uncertainty_score": row.get("uncertainty_score", ""),
                "gnn_vs_surrogate_disagreement": row.get("gnn_surrogate_disagreement", ""),
                "novelty_distance": row.get("novelty_distance_to_combined204_plus_N32", ""),
                "nearest_existing_teacher_strategy": row.get("nearest_existing_teacher_strategy", ""),
                "native_validation_N": True,
                "N24_N40_focused": True,
                "overnight_batch60": True,
                "order_json": order,
                "order_compact": compact_order(order),
                "order_hash": row.get("order_hash", ""),
                "teacher_validated": False,
                "teacher_validation_status": "NOT_RUN",
                "notes": "Run39 handoff only. Native N24/N40 focused batch60 candidate is not teacher-validated.",
            }
        )
    return pd.DataFrame(rows)


def write_scan_order_jsons(handoff: pd.DataFrame) -> None:
    SCAN_ORDER_DIR.mkdir(parents=True, exist_ok=True)
    for stale in SCAN_ORDER_DIR.glob("scan_order_*.json"):
        stale.unlink()
    for _, row in handoff.iterrows():
        payload = {
            "run_id": RUN_ID,
            "batch_option": BATCH_OPTION,
            "batch_name": BATCH_NAME,
            "n": int(row["n"]),
            "handoff_strategy_name": row["handoff_strategy_name"],
            "original_run38_candidate_id": row["original_run38_candidate_id"],
            "candidate_source": row["candidate_source"],
            "generation_method": row["generation_method"],
            "selection_bucket": row["selection_bucket"],
            "priority_role": row["priority_role"],
            "surrogate_prediction": row["surrogate_prediction"],
            "gnn_reward_prediction": row["gnn_reward_prediction"],
            "graph_pointer_policy_score": row["graph_pointer_policy_score"],
            "hybrid_score": row["hybrid_score"],
            "uncertainty_score": row["uncertainty_score"],
            "gnn_vs_surrogate_disagreement": row["gnn_vs_surrogate_disagreement"],
            "novelty_distance": row["novelty_distance"],
            "nearest_existing_teacher_strategy": row["nearest_existing_teacher_strategy"],
            "native_validation_N": True,
            "N24_N40_focused": True,
            "overnight_batch60": True,
            "scan_order": parse_order(row["order_json"]),
            "order_hash": row["order_hash"],
            "teacher_validated": False,
            "teacher_validation_status": "NOT_RUN",
            "notes": "Run39 handoff only. Native N24/N40 focused batch60 candidate is not teacher-validated.",
        }
        write_json(SCAN_ORDER_DIR / f"scan_order_{row['handoff_strategy_name']}.json", payload)


def write_future_templates(handoff: pd.DataFrame) -> tuple[Path, Path]:
    manifest_path = OUTPUT_DIR / "stage3_run39_native_N24_N40_focused_batch60_future_cae_handoff_manifest_TEMPLATE.csv"
    command_path = OUTPUT_DIR / "stage3_run39_native_N24_N40_focused_batch60_abqjobpilot_commands_TEMPLATE.txt"
    case_root = ROOT / "cae_model" / BATCH_NAME
    rows = []
    commands = [
        "# TEMPLATE ONLY - INP files do not exist yet.",
        "# Do not execute until CAE/INP generation has completed and passed checks.",
        "# Run39 did not create CAE/INP/JNL files and did not run solver.",
        "",
    ]
    for _, row in handoff.iterrows():
        case_dir = case_root / f"N{int(row['n'])}{row['handoff_strategy_name']}"
        job_name = f"J2D_{row['handoff_strategy_name']}"
        scan_json = SCAN_ORDER_DIR / f"scan_order_{row['handoff_strategy_name']}.json"
        rows.append(
            {
                "n": int(row["n"]),
                "handoff_strategy_name": row["handoff_strategy_name"],
                "expected_case_dir": str(case_dir),
                "expected_job_name": job_name,
                "scan_order_json": str(scan_json),
                "expected_cae": str(case_dir / f"{job_name}.cae"),
                "expected_inp": str(case_dir / f"{job_name}.inp"),
                "expected_jnl": str(case_dir / f"{job_name}.jnl"),
                "expected_odb": str(case_dir / f"{job_name}.odb"),
                "teacher_validated": False,
                "generation_status": "NOT_GENERATED",
                "solver_status": "NOT_SUBMITTED",
                "notes": "Template only. Do not execute abqjobpilot/enqueue until INP exists and passes checks.",
            }
        )
        commands.append(f'enqueue --inp "{case_dir}\\{job_name}.inp" --cpus 14 --batch {BATCH_NAME} --strategy {row["handoff_strategy_name"]}')
    write_csv(manifest_path, pd.DataFrame(rows))
    command_path.write_text("\n".join(commands) + "\n", encoding="utf-8")
    return manifest_path, command_path


def review_summary(handoff: pd.DataFrame, overlaps: dict[str, int]) -> tuple[pd.DataFrame, dict[str, Any], str]:
    numeric_cols = ["surrogate_prediction", "gnn_reward_prediction", "hybrid_score", "uncertainty_score", "gnn_vs_surrogate_disagreement", "novelty_distance"]
    for col in numeric_cols:
        handoff[col] = pd.to_numeric(handoff[col], errors="coerce")
    rows = []
    for n, block in handoff.groupby("n"):
        rows.append(
            {
                "scope": f"N{int(n)}",
                "count": int(len(block)),
                "mean_surrogate_prediction": float(block["surrogate_prediction"].mean()),
                "mean_gnn_reward_prediction": float(block["gnn_reward_prediction"].mean()),
                "mean_hybrid_score": float(block["hybrid_score"].mean()),
                "mean_disagreement": float(block["gnn_vs_surrogate_disagreement"].mean()),
                "mean_novelty_distance": float(block["novelty_distance"].mean()),
            }
        )
    source_comp = handoff["candidate_source"].fillna("").value_counts().to_dict()
    bucket_comp = handoff["selection_bucket"].fillna("").value_counts().to_dict()
    summary = {
        "total_count": int(len(handoff)),
        "per_N_counts": {int(k): int(v) for k, v in handoff["n"].value_counts().sort_index().to_dict().items()},
        "only_N24_N40": sorted(handoff["n"].astype(int).unique().tolist()) == [24, 40],
        "includes_N12": False,
        "includes_N16": False,
        "includes_N32": False,
        "candidate_source_composition": source_comp,
        "selection_bucket_composition": bucket_comp,
        "expected_abaqus_cost": "60 jobs total, with 30 N24 and 30 N40",
        "overlap_status": overlaps,
        "headline": "Selected batch60 focuses directly on N24/N40 U2 gains from Run36 and has no exact overlap with the checked prior/reference sets.",
    }
    md = (
        "# Run39 Native N24/N40 Focused Batch60 Review Summary\n\n"
        "- Total: 60 candidates\n"
        "- Per-N: N24=30, N40=30\n"
        "- No N12, N16, or N32 candidates are included.\n"
        "- Expected Abaqus cost: 60 jobs total, all N24/N40.\n"
        f"- Candidate-source composition: `{source_comp}`\n"
        f"- Selection-bucket composition: `{bucket_comp}`\n"
        f"- Exact-overlap status: `{overlaps}`\n\n"
        "This batch focuses directly on N24/N40 U2 gains from Run36. It is not teacher-validated until future Abaqus validation. Run39 did not create CAE/INP files.\n"
    )
    return pd.DataFrame(rows), summary, md


def write_claim_boundary() -> tuple[Path, Path]:
    md_path = OUTPUT_DIR / "run39_claim_boundary.md"
    json_path = OUTPUT_DIR / "run39_claim_boundary.json"
    safe = [
        "Run39 packages selected Run38-derived native N24/N40 focused batch60 candidates for human review and future CAE generation.",
        "The selected batch contains N24=30 and N40=30.",
        "No N12, N16, or N32 candidates are included.",
        "The batch is designed to test whether Run36 N24/N40 U2 improvements can be further exploited.",
        "Handoff files include scan orders, metadata, future CAE manifest template, and abqjobpilot command template.",
        "No CAE/INP files were generated.",
    ]
    unsafe = [
        "candidates are teacher-validated.",
        "physical superiority.",
        "N32 caused improvement.",
        "GNN-RL has beaten baselines.",
        "online RL with Abaqus.",
        "arbitrary-N generalization.",
        "surrogate/GNN/hybrid predictions are ground truth.",
        "abqjobpilot commands are ready to execute.",
        "CAE/INP files exist.",
    ]
    md_path.write_text("# Run39 Claim Boundary\n\n## Safe Claims\n" + "\n".join(f"- {x}" for x in safe) + "\n\n## Unsafe Claims\n" + "\n".join(f"- Do not claim {x}" for x in unsafe) + "\n", encoding="utf-8")
    write_json(json_path, {"verdict": "RUN39_HANDOFF_ONLY_NATIVE_N24_N40_BATCH60_NO_TEACHER_VALIDATION", "safe_claims": safe, "unsafe_claims": unsafe})
    return md_path, json_path


def write_report(output_files: list[Path], validation: dict[str, Any], review: dict[str, Any]) -> None:
    lines = [
        "# Stage 3 Run 39 - Run38 Native N24/N40 Focused Batch60 Handoff Package",
        "",
        "## 1. Purpose",
        "Package a Run38-derived native N24/N40 focused batch60 for human review and future CAE generation.",
        "",
        "## 2. Inputs",
        f"- Run38 scored pool: `{RUN38_POOL}`",
        f"- Run38 Option B reference: `{RUN38_OPTION_B}`",
        f"- Native combined204 teacher dataset: `{COMBINED204_TEACHER}`",
        "",
        "## 3. User-Selected Batch60",
        f"- Batch name: `{BATCH_NAME}`",
        "- Selected counts: N24=30, N40=30.",
        "",
        "## 4. Why Batch60 Was Selected",
        "The user is running overnight and wants more coverage than the smaller Run38 batch32. Run36 refreshed N24/N40 U2 bests, so Run39 presses those two N values directly.",
        "",
        "## 5. Why Only N24/N40 Are Included",
        "Native combined204 remains the strongest Run38 surrogate regime, and the current scientific pressure point is N24/N40 U2 exploitation and calibration. N12, N16, and N32 are excluded from this selected batch.",
        "",
        "## 6. Validation Status",
        f"- Verdict: `{validation['verdict']}`",
        f"- Pool counts: `{validation['pool_counts']}`",
        "",
        "## 7. Stable Naming Convention",
        "`S3R39N2440B60_N{N}_B{index:02d}_{short_bucket}`",
        "",
        "## 8. Candidate-Order Handoff Package",
        f"- Candidate CSV: `{OUTPUT_DIR / 'stage3_run39_native_N24_N40_focused_batch60_candidate_orders.csv'}`",
        "",
        "## 9. Per-Candidate Scan-Order JSON Outputs",
        f"- Directory: `{SCAN_ORDER_DIR}`",
        "",
        "## 10. Future CAE Handoff Template",
        f"- Template: `{OUTPUT_DIR / 'stage3_run39_native_N24_N40_focused_batch60_future_cae_handoff_manifest_TEMPLATE.csv'}`",
        "",
        "## 11. Future abqjobpilot Command Template",
        f"- Template: `{OUTPUT_DIR / 'stage3_run39_native_N24_N40_focused_batch60_abqjobpilot_commands_TEMPLATE.txt'}`",
        "",
        "## 12. Review Summary",
        f"- {review['headline']}",
        "",
        "## 13. Claim Boundary",
        "- Run39 is handoff packaging only. No CAE/INP, no solver activity, and no teacher validation were performed.",
        "",
        "## 14. Output Files",
        *[f"- `{path}`" for path in output_files],
        "",
        "## 15. Recommended Run40",
        "CAE module should generate CAE/INP/JNL for selected Run39 native N24/N40 focused batch60 only. Do not run solver. Do not execute abqjobpilot. Do not generate smaller batch32 unless explicitly selected later.",
    ]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_run_index(verdict: str) -> None:
    if not RUN_INDEX_PATH.exists():
        return
    text = RUN_INDEX_PATH.read_text(encoding="utf-8")
    if RUN_ID in text:
        return
    row = (
        "| run_39 | Run38 native N24/N40 focused batch60 handoff package | "
        "Packages selected Run38-derived N24/N40 overnight batch60 with scan orders and future CAE templates. | "
        "`scripts/stage3/run_39_create_run38_native_N24_N40_focused_batch60_handoff_package.py` | "
        "`docs/stage3/runs/run_39_run38_native_N24_N40_focused_batch60_handoff_package/RUN_39_RUN38_NATIVE_N24_N40_FOCUSED_BATCH60_HANDOFF_PACKAGE_REPORT.md` | "
        "`outputs/stage3_run_39_run38_native_N24_N40_focused_batch60_handoff_package/` | "
        f"`{verdict}` | No Abaqus, no ODB opening, no abqjobpilot, no CAE/INP/JNL generation, no teacher validation, no training, no commit/push. |"
    )
    RUN_INDEX_PATH.write_text(text.rstrip() + "\n" + row + "\n", encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SCAN_ORDER_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    for path in [RUN38_POOL, RUN38_OPTION_B, RUN38_REPORT, COMBINED204_READY, COMBINED204_TEACHER, RUN36_ENRICHED, SUPERSEDED_RUN31_BATCH]:
        if not path.exists():
            raise FileNotFoundError(path)

    pool = read_csv(RUN38_POOL)
    pool["n"] = pool["n"].astype(int)
    validation = validate_pool(pool)
    if validation["verdict"].startswith("FAIL"):
        write_json(OUTPUT_DIR / "run39_input_validation_summary.json", validation)
        raise RuntimeError(validation["verdict"])

    forbidden = load_hashes(COMBINED204_TEACHER) | load_hashes(RUN36_ENRICHED) | load_hashes(RUN27_ENRICHED) | load_hashes(SUPERSEDED_RUN31_BATCH)
    option_b_hashes = load_hashes(RUN38_OPTION_B)
    selected = select_batch(pool[pool["n"].isin([24, 40])].copy(), forbidden, option_b_hashes)
    handoff = create_handoff(selected)

    selected_hashes = set(handoff["order_hash"].astype(str))
    overlaps = {
        "combined204_teacher": len(selected_hashes & load_hashes(COMBINED204_TEACHER)),
        "run36_teacher": len(selected_hashes & load_hashes(RUN36_ENRICHED)),
        "run27_shortlist64": len(selected_hashes & load_hashes(RUN27_ENRICHED)),
        "superseded_old_run31": len(selected_hashes & load_hashes(SUPERSEDED_RUN31_BATCH)),
        "run38_option_b_batch32_reference": len(selected_hashes & option_b_hashes),
    }
    validation.update(
        {
            "selected_batch_rows": int(len(handoff)),
            "selected_per_N_counts": {int(k): int(v) for k, v in handoff["n"].value_counts().sort_index().to_dict().items()},
            "selected_unique_order_hashes": int(handoff["order_hash"].nunique()),
            "selected_only_N24_N40": sorted(handoff["n"].astype(int).unique().tolist()) == [24, 40],
            "overlap_counts": overlaps,
        }
    )
    write_json(OUTPUT_DIR / "run39_input_validation_summary.json", validation)

    candidate_path = OUTPUT_DIR / "stage3_run39_native_N24_N40_focused_batch60_candidate_orders.csv"
    write_csv(candidate_path, handoff)
    write_scan_order_jsons(handoff)
    future_manifest, command_template = write_future_templates(handoff)

    review_df, review_json, review_md = review_summary(handoff.copy(), overlaps)
    review_csv = OUTPUT_DIR / "native_N24_N40_focused_batch60_review_summary.csv"
    review_json_path = OUTPUT_DIR / "native_N24_N40_focused_batch60_review_summary.json"
    review_md_path = OUTPUT_DIR / "native_N24_N40_focused_batch60_review_summary.md"
    write_csv(review_csv, review_df)
    write_json(review_json_path, review_json)
    review_md_path.write_text(review_md, encoding="utf-8")

    claim_md, claim_json = write_claim_boundary()
    output_files = [
        OUTPUT_DIR / "run39_input_validation_summary.json",
        candidate_path,
        future_manifest,
        command_template,
        review_csv,
        review_json_path,
        review_md_path,
        claim_md,
        claim_json,
        REPORT_PATH,
        MANIFEST_PATH,
    ]
    write_report(output_files, validation, review_json)

    manifest = {
        "run_id": RUN_ID,
        "run_name": RUN_NAME,
        "timestamp": now_iso(),
        "branch": current_branch(),
        "script_path": str(SCRIPT_PATH),
        "input_files": [str(RUN38_POOL), str(RUN38_OPTION_B), str(RUN38_REPORT), str(COMBINED204_READY), str(COMBINED204_TEACHER), str(RUN36_ENRICHED), str(RUN27_ENRICHED), str(SUPERSEDED_RUN31_BATCH)],
        "output_files": [str(path) for path in output_files],
        "selected_batch": "run38_native_N24_N40_focused_batch60",
        "batch_name": BATCH_NAME,
        "batch60_count": 60,
        "per_N_counts": {int(k): int(v) for k, v in handoff["n"].value_counts().sort_index().to_dict().items()},
        "includes_N12": False,
        "includes_N16": False,
        "includes_N24": True,
        "includes_N32": False,
        "includes_N40": True,
        "N24_N40_focused": True,
        "overnight_batch60": True,
        "report_path": str(REPORT_PATH),
        "claim_boundary_path": str(claim_md),
        "no_solver_run": True,
        "no_odb_opened": True,
        "no_abqjobpilot_run": True,
        "no_cae_inp_generated": True,
        "no_teacher_validation": True,
        "no_training": True,
        "no_candidate_generation_model_training": True,
        "no_commit_or_push": True,
    }
    write_json(MANIFEST_PATH, manifest)
    update_run_index("RUN39_HANDOFF_ONLY_NATIVE_N24_N40_BATCH60_NO_TEACHER_VALIDATION")
    print(json.dumps({"verdict": validation["verdict"], "counts": manifest["per_N_counts"], "overlaps": overlaps}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
