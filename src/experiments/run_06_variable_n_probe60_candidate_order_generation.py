from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Callable


TARGET_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
if str(TARGET_ROOT) not in sys.path:
    sys.path.insert(0, str(TARGET_ROOT))

from src.baselines.baseline_utils import (  # noqa: E402
    directed_overlap,
    jump_profile_distance,
    kendall_distance,
    order_json,
    spearman_distance,
    structural_summary,
    undirected_overlap,
    validate_order,
)
from src.baselines.block_interleaved import block_interleaved_quarters  # noqa: E402
from src.baselines.center_out import center_edge_alternating, center_out  # noqa: E402
from src.baselines.edge_in import edge_in_alternating  # noqa: E402
from src.baselines.maximin import greedy_maximin_distance  # noqa: E402
from src.baselines.method_c import method_c_u2_first_engineering  # noqa: E402
from src.baselines.odd_even import odd_even_interlaced  # noqa: E402
from src.baselines.raster import raster_left_to_right  # noqa: E402
from src.baselines.regular_jump import regular_jump_coprime  # noqa: E402
from src.policies.graph_pointer_policy import GraphPointerPolicyPrototype  # noqa: E402


RUN_ID = "run_06_variable_n_probe60_candidate_order_generation"
N_VALUES = [12, 16, 24, 40]
OUTPUT_DIR = TARGET_ROOT / "outputs" / "stage3_run_06_variable_n_probe60_candidate_order_generation"
REPORT_DIR = TARGET_ROOT / "docs" / "stage3" / "runs" / RUN_ID
REPORT_PATH = REPORT_DIR / "RUN_06_VARIABLE_N_PROBE60_CANDIDATE_ORDER_GENERATION_REPORT.md"
MANIFEST_PATH = TARGET_ROOT / "artifacts" / "manifests" / "stage3_run_06_manifest.json"


def strategy_specs() -> list[tuple[str, str, str, Callable[[int], list[int]], bool]]:
    return [
        ("A01", "raster_left_to_right", "raster", raster_left_to_right, True),
        ("A02", "odd_even_interlaced", "odd_even", odd_even_interlaced, True),
        ("A03", "greedy_maximin_distance", "maximin", greedy_maximin_distance, True),
        ("A04", "method_c_u2_first_engineering", "method_c", method_c_u2_first_engineering, True),
        ("A05", "center_out", "center_out", center_out, True),
        ("A06", "edge_in_alternating", "edge_in", edge_in_alternating, True),
        ("A07", "regular_jump_coprime", "regular_jump", regular_jump_coprime, True),
        ("A08", "block_interleaved_quarters", "block_interleaved", block_interleaved_quarters, True),
        ("A09", "center_edge_alternating", "center_edge", center_edge_alternating, True),
        ("A10", "graph_pointer_policy_zero_shot_or_proxy_best", "graph_pointer_proxy", lambda n: GraphPointerPolicyPrototype("proxy_best").decode(n), False),
        ("A11", "graph_pointer_policy_diverse_01", "graph_pointer_proxy", lambda n: GraphPointerPolicyPrototype("diverse_01").decode(n), False),
        ("A12", "graph_pointer_policy_diverse_02", "graph_pointer_proxy", lambda n: GraphPointerPolicyPrototype("diverse_02").decode(n), False),
        ("A13", "graph_pointer_policy_anti_odd_even_novelty", "graph_pointer_proxy", lambda n: GraphPointerPolicyPrototype("anti_odd_even_novelty").decode(n), False),
        ("A14", "graph_pointer_policy_u2first_proxy", "graph_pointer_proxy", lambda n: GraphPointerPolicyPrototype("u2first_proxy").decode(n), False),
        ("A15", "graph_pointer_policy_balanced_dispersion_proxy", "graph_pointer_proxy", lambda n: GraphPointerPolicyPrototype("balanced_dispersion_proxy").decode(n), False),
    ]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    candidates: list[dict] = []
    legality: list[dict] = []
    structural: list[dict] = []
    for n in N_VALUES:
        for code, base_name, family, fn, is_baseline in strategy_specs():
            strategy_name = f"N{n}_{code}_{base_name}"
            order = fn(n)
            ok, reasons = validate_order(order, n)
            source = "engineering_baseline" if is_baseline else "proxy_policy"
            candidates.append(
                {
                    "n": n,
                    "strategy_id": f"N{n}_{code}",
                    "strategy_name": strategy_name,
                    "family": family,
                    "order_json": order_json(order),
                    "order_length": len(order),
                    "is_engineering_baseline": is_baseline,
                    "is_learned_or_proxy": not is_baseline,
                    "policy_source": source,
                    "trained_policy_used": False,
                    "teacher_validated": False,
                    "notes": "true variable-N order design only; no CAE/INP/JNL generated",
                }
            )
            legality.append(
                {
                    "n": n,
                    "strategy_name": strategy_name,
                    "length_equals_n": len(order) == n,
                    "all_integers": all(isinstance(x, int) and not isinstance(x, bool) for x in order),
                    "set_equals_0_to_n_minus_1": set(order) == set(range(n)),
                    "no_duplicates": len(set(order)) == len(order),
                    "no_missing_tracks": not (set(range(n)) - set(order)),
                    "no_out_of_range_tracks": all(isinstance(x, int) and 0 <= x < n for x in order),
                    "pass": ok,
                    "reasons": ";".join(reasons) if reasons else "none",
                }
            )
            structural.append({"n": n, "strategy_name": strategy_name, **structural_summary(order, n)})
    pairwise: list[dict] = []
    by_n = {n: [row for row in candidates if row["n"] == n] for n in N_VALUES}
    for n, rows in by_n.items():
        parsed = [(row, json.loads(row["order_json"])) for row in rows]
        for (a_row, a_order), (b_row, b_order) in combinations(parsed, 2):
            pairwise.append(
                {
                    "n": n,
                    "strategy_a": a_row["strategy_name"],
                    "strategy_b": b_row["strategy_name"],
                    "directed_adjacent_pair_overlap": directed_overlap(a_order, b_order),
                    "undirected_adjacent_pair_overlap": undirected_overlap(a_order, b_order),
                    "kendall_tau_normalized_distance": kendall_distance(a_order, b_order),
                    "spearman_rank_distance": spearman_distance(a_order, b_order),
                    "jump_profile_distance": jump_profile_distance(a_order, b_order, n),
                }
            )
    verdict = "WARNING_VARIABLE_N_PROBE60_POLICY_PROXY_FALLBACK_USED"
    if not all(row["pass"] for row in legality) or len(candidates) != 60:
        verdict = "FAIL_VARIABLE_N_PROBE60_CANDIDATE_ORDER_GENERATION_INVALID"
    candidate_path = OUTPUT_DIR / "variable_N_probe60_candidate_orders.csv"
    write_csv(candidate_path, candidates, list(candidates[0].keys()))
    legality_path = OUTPUT_DIR / "variable_N_probe60_legality_audit.csv"
    write_csv(legality_path, legality, list(legality[0].keys()))
    structural_path = OUTPUT_DIR / "variable_N_probe60_structural_summary.csv"
    write_csv(structural_path, structural, list(structural[0].keys()))
    pairwise_path = OUTPUT_DIR / "variable_N_probe60_pairwise_diversity.csv"
    write_csv(pairwise_path, pairwise, list(pairwise[0].keys()))
    design_manifest_path = OUTPUT_DIR / "variable_N_probe60_candidate_design_manifest.json"
    design_manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": RUN_ID,
        "n_values": N_VALUES,
        "cases_per_n": 15,
        "total_cases": len(candidates),
        "teacher_validated": False,
        "policy_candidate_source": "proxy_policy",
        "trained_policy_used": False,
        "no_abaqus_jobs": True,
        "no_cae_inp_jnl_generated": True,
    }
    design_manifest_path.write_text(json.dumps(design_manifest, indent=2) + "\n", encoding="utf-8")
    outputs = [str(candidate_path), str(legality_path), str(structural_path), str(pairwise_path), str(design_manifest_path)]
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "run_id": RUN_ID,
                "python_executable": sys.executable,
                "verdict": verdict,
                "n_values": N_VALUES,
                "cases_per_n": 15,
                "total_cases": len(candidates),
                "policy_candidate_source": "proxy_policy",
                "outputs_written": outputs,
                "forbidden_actions_confirmed": {
                    "no_abaqus_jobs": True,
                    "no_datacheck": True,
                    "no_odb_opened": True,
                    "no_cae_generated": True,
                    "no_inp_generated": True,
                    "no_jnl_generated": True,
                    "no_abqjobpilot_execution": True,
                    "no_teacher_validation": True,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        f"""# Run 06 Variable-N Probe60 Candidate Order Generation Report

## Executive Verdict

{verdict}

Generated 15 legal scan-order designs for each N in `{N_VALUES}`, total `{len(candidates)}`. A01-A09 are deterministic engineering baselines. A10-A15 are deterministic graph pointer proxy candidates, not trained or teacher validated.

## Guardrails

- No Abaqus jobs.
- No datacheck.
- No ODB opened.
- No CAE/INP/JNL generated.
- No abqjobpilot execution.
- No teacher validation.

## Claim Boundary

These are candidate-generation outputs only. Variable-N is not yet a validated RL result, and no shared full-32 U2 guard is used.

## Outputs

- `{candidate_path}`
- `{legality_path}`
- `{structural_path}`
- `{pairwise_path}`
- `{design_manifest_path}`
- `{MANIFEST_PATH}`
""",
        encoding="utf-8",
    )
    print(verdict)
    print(f"total_cases={len(candidates)}")
    return 1 if verdict.startswith("FAIL") else 0


if __name__ == "__main__":
    raise SystemExit(main())
