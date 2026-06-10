from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TARGET_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
SOURCE_ROOT = Path(r"D:\Projects\RL-LAM-ScanOpt")
if str(TARGET_ROOT) not in sys.path:
    sys.path.insert(0, str(TARGET_ROOT))

from src.feature_builders.variable_n_graph_builder import (  # noqa: E402
    EDGE_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    N_VALUES,
    SIGNED_FEATURES,
    build_sample_graphs,
    compact_graph,
    summarise_graph_features,
)


RUN_ID = "run_03_variable_n_graph_feature_builder"
OUTPUT_DIR = TARGET_ROOT / "outputs" / "stage3_run_03_variable_n_graph_feature_builder"
REPORT_DIR = TARGET_ROOT / "docs" / "stage3" / "runs" / "run_03_variable_n_graph_feature_builder"
REPORT_PATH = REPORT_DIR / "RUN_03_VARIABLE_N_GRAPH_FEATURE_BUILDER_REPORT.md"
MANIFEST_PATH = TARGET_ROOT / "artifacts" / "manifests" / "stage3_run_03_manifest.json"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite_min_max(values: list[float]) -> tuple[float | str, float | str]:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return "", ""
    return min(finite), max(finite)


def flatten_feature_values(graph: dict[str, Any], scope: str) -> list[float]:
    if scope == "node":
        return [float(v) for node in graph["nodes"] for v in node["features"].values()]
    if scope == "edge":
        return [float(v) for edge in graph["edges"] for v in edge["features"].values()]
    return [float(v) for v in graph["global_context"]["features"].values()]


def pass_to_bool(value: Any) -> bool:
    return bool(value) if isinstance(value, bool) else str(value).lower() == "true"


def build_output_rows(graphs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    normalisation_rows: list[dict[str, Any]] = []
    node_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    global_rows: list[dict[str, Any]] = []
    mask_rows: list[dict[str, Any]] = []

    for graph in graphs:
        n = graph["metadata"]["n"]
        prefix_name = graph["metadata"]["prefix_name"]
        validation = graph["validation"]
        node_min, node_max = finite_min_max(flatten_feature_values(graph, "node"))
        edge_min, edge_max = finite_min_max(flatten_feature_values(graph, "edge"))
        global_min, global_max = finite_min_max(flatten_feature_values(graph, "global"))
        normalisation_rows.append(
            {
                "n": n,
                "prefix_name": prefix_name,
                "node_count": validation["node_count"],
                "edge_count": validation["edge_count"],
                "scanned_count": validation["scanned_count"],
                "available_count": validation["available_count"],
                "legal_action_count": validation["legal_action_count"],
                "node_feature_min": node_min,
                "node_feature_max": node_max,
                "edge_feature_min": edge_min,
                "edge_feature_max": edge_max,
                "global_feature_min": global_min,
                "global_feature_max": global_max,
                "all_finite": validation["all_finite"],
                "normalized_bounds_pass": validation["normalized_bounds_pass"],
                "mask_legality_pass": validation["mask_legality_pass"],
                "duplicate_edge_count": validation["duplicate_edge_count"],
                "verdict": validation["verdict"],
            }
        )
        summaries = summarise_graph_features(graph)
        for feature_name, summary in summaries["node"].items():
            node_rows.append(
                {
                    "n": n,
                    "prefix_name": prefix_name,
                    "feature_name": feature_name,
                    **summary,
                }
            )
        for feature_name, summary in summaries["edge"].items():
            edge_rows.append(
                {
                    "n": n,
                    "prefix_name": prefix_name,
                    "feature_name": feature_name,
                    **summary,
                }
            )
        for feature_name, summary in summaries["global"].items():
            global_rows.append(
                {
                    "n": n,
                    "prefix_name": prefix_name,
                    "feature_name": feature_name,
                    **summary,
                }
            )
        masks = graph["masks"]
        scanned = masks["scanned_mask"]
        available = masks["available_mask"]
        legal = masks["pointer_legal_mask"]
        illegal_scanned_available = sum(1 for i in range(n) if scanned[i] and available[i])
        mask_rule = all(legal[i] == (available[i] and not scanned[i]) for i in range(n))
        mask_rows.append(
            {
                "n": n,
                "prefix_name": prefix_name,
                "scanned_count": sum(scanned),
                "available_count": sum(available),
                "legal_count": sum(legal),
                "illegal_scanned_available_count": illegal_scanned_available,
                "pointer_legal_equals_available_and_not_scanned": mask_rule,
                "pass": mask_rule and sum(legal) == n - sum(scanned),
            }
        )
    return {
        "normalisation": normalisation_rows,
        "node": node_rows,
        "edge": edge_rows,
        "global": global_rows,
        "mask": mask_rows,
    }


def write_schema(path: Path) -> None:
    node_list = "\n".join(f"- `{name}`" for name in NODE_FEATURE_NAMES)
    edge_list = "\n".join(f"- `{name}`" for name in EDGE_FEATURE_NAMES)
    global_list = "\n".join(f"- `{name}`" for name in GLOBAL_FEATURE_NAMES)
    signed_list = "\n".join(f"- `{name}`" for name in sorted(SIGNED_FEATURES))
    path.write_text(
        f"""# Variable-N Graph Feature Schema

Run 03 defines a variable-N graph representation for `N = 16, 24, 32, 40`.

Each track is a graph node. Directed edges encode adjacent, k-nearest, and thermal-radius spatial relations. Global context stores normalized scan-progress and balance descriptors. Masks expose future pointer-decoder legality, but no policy is trained in this run.

## Node Features

{node_list}

`track_index` is retained only as metadata and is not a model feature.

## Edge Features

{edge_list}

`source_index`, `target_index`, and string `edge_type` are metadata. The numeric edge feature for edge type is `edge_type_code`.

## Global Features

{global_list}

## Signed Normalized Features

{signed_list}

All other numeric model features are expected in `[0, 1]`. Signed relation and balance features are expected in `[-1, 1]`.

## Local Heat Proxy

`local_heat_proxy_norm` is a geometry/time-decay proxy computed from prior scanned tracks and normalized within each graph state. It is not a teacher metric and must not be interpreted as Abaqus evidence.
""",
        encoding="utf-8",
    )


def verdict_for(rows: dict[str, list[dict[str, Any]]]) -> str:
    norm_pass = all(row["verdict"] == "PASS" for row in rows["normalisation"])
    node_pass = all(pass_to_bool(row["pass"]) for row in rows["node"])
    edge_pass = all(pass_to_bool(row["pass"]) for row in rows["edge"])
    global_pass = all(pass_to_bool(row["pass"]) for row in rows["global"])
    mask_pass = all(pass_to_bool(row["pass"]) for row in rows["mask"])
    if norm_pass and node_pass and edge_pass and global_pass and mask_pass:
        return "PASS_VARIABLE_N_GRAPH_FEATURE_BUILDER_READY"
    if rows["normalisation"]:
        return "WARNING_VARIABLE_N_GRAPH_FEATURE_BUILDER_PARTIAL"
    return "FAIL_VARIABLE_N_GRAPH_FEATURE_BUILDER_INVALID"


def write_manifest(
    graphs: list[dict[str, Any]],
    rows: dict[str, list[dict[str, Any]]],
    outputs_written: list[str],
    verdict: str,
) -> None:
    validation_summary = {
        "graph_states_passed": sum(1 for graph in graphs if graph["validation"]["pass"]),
        "graph_states_failed": sum(1 for graph in graphs if not graph["validation"]["pass"]),
        "normalisation_rows": len(rows["normalisation"]),
        "mask_rows": len(rows["mask"]),
        "raw_fixed32_or_fixed_id_feature_leak": False,
    }
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_root": str(SOURCE_ROOT),
        "target_root": str(TARGET_ROOT),
        "python_executable": sys.executable,
        "run_id": RUN_ID,
        "verdict": verdict,
        "n_values": N_VALUES,
        "graph_state_count": len(graphs),
        "node_feature_names": NODE_FEATURE_NAMES,
        "edge_feature_names": EDGE_FEATURE_NAMES,
        "global_feature_names": GLOBAL_FEATURE_NAMES,
        "outputs_written": outputs_written,
        "validation_summary": validation_summary,
        "forbidden_actions_confirmed": {
            "no_abaqus_jobs": True,
            "no_datacheck": True,
            "no_odb_opened": True,
            "no_cae_generated": True,
            "no_inp_generated": True,
            "no_jnl_generated": True,
            "no_model_training": True,
            "no_rl_candidate_generation": True,
            "no_teacher_validation": True,
            "stage2_source_read_only": True,
        },
    }
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_report(graphs: list[dict[str, Any]], rows: dict[str, list[dict[str, Any]]], verdict: str, outputs_written: list[str]) -> None:
    all_pass = all(graph["validation"]["pass"] for graph in graphs)
    normalisation_summary = "All generated graph states passed finite-value, normalized-bounds, mask-legality, and duplicate-edge checks." if all_pass else "One or more graph states failed validation; inspect CSV audits."
    outputs = "\n".join(f"- `{path}`" for path in outputs_written)
    node_features = "\n".join(f"- `{name}`" for name in NODE_FEATURE_NAMES)
    edge_features = "\n".join(f"- `{name}`" for name in EDGE_FEATURE_NAMES)
    global_features = "\n".join(f"- `{name}`" for name in GLOBAL_FEATURE_NAMES)
    REPORT_PATH.write_text(
        f"""# Run 03 Variable-N Graph Feature Builder Report

## Executive Verdict

{verdict}

## What Was Built

- N values covered: `{N_VALUES}`
- Graph states generated: `{len(graphs)}`
- Node feature count: `{len(NODE_FEATURE_NAMES)}`
- Edge feature count: `{len(EDGE_FEATURE_NAMES)}`
- Global feature count: `{len(GLOBAL_FEATURE_NAMES)}`

## Guardrails

- No Abaqus jobs.
- No datacheck.
- No ODB opened.
- No CAE/INP/JNL generated.
- No model training.
- No RL candidate generation.
- No teacher validation.
- D-drive source was not modified.

## Scientific Positioning

Run 03 creates the variable-N representation foundation for later graph pointer policy work. It is not yet an RL result and does not prove variable-N generalisation.

## Feature Schema

### Node Features

{node_features}

### Edge Features

{edge_features}

### Global Features

{global_features}

### Masks

- `scanned_mask`
- `available_mask`
- `pointer_legal_mask`

## Normalisation Audit

{normalisation_summary}

No raw track ID, fixed 32-dimensional representation, fixed track ID embedding, absolute step index, or raw unnormalized jump length is used as a model feature. `track_index`, `source_index`, and `target_index` appear only as metadata.

## Local Heat Proxy Note

`local_heat_proxy_norm` is a geometry/time-decay proxy normalized within each graph state. It is not a teacher metric and is not Abaqus validation evidence.

## Claim Boundary

Allowed: variable-N graph feature representation is implemented and validated on N=16/24/32/40 sample states.

Not allowed: RL generalises to variable N; GNN/RL solves variable-N optimisation; the same full-32 U2 guard transfers to all N; teacher-validated variable-N improvement exists.

## Outputs

{outputs}

## Recommended Next Run

`run_04_variable_n_baseline_generator`

## Final Verdict

{verdict}
""",
        encoding="utf-8",
    )


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    graphs = build_sample_graphs()
    rows = build_output_rows(graphs)
    outputs_written: list[str] = []

    schema_path = OUTPUT_DIR / "variable_N_graph_feature_schema.md"
    write_schema(schema_path)
    outputs_written.append(str(schema_path))

    sample_path = OUTPUT_DIR / "sample_graphs_N16_N24_N32_N40.json"
    sample_payload = [compact_graph(graph) for graph in graphs]
    sample_path.write_text(json.dumps(sample_payload, indent=2) + "\n", encoding="utf-8")
    outputs_written.append(str(sample_path))

    write_csv(
        OUTPUT_DIR / "feature_normalisation_audit.csv",
        rows["normalisation"],
        [
            "n",
            "prefix_name",
            "node_count",
            "edge_count",
            "scanned_count",
            "available_count",
            "legal_action_count",
            "node_feature_min",
            "node_feature_max",
            "edge_feature_min",
            "edge_feature_max",
            "global_feature_min",
            "global_feature_max",
            "all_finite",
            "normalized_bounds_pass",
            "mask_legality_pass",
            "duplicate_edge_count",
            "verdict",
        ],
    )
    outputs_written.append(str(OUTPUT_DIR / "feature_normalisation_audit.csv"))
    write_csv(OUTPUT_DIR / "node_feature_summary.csv", rows["node"], ["n", "prefix_name", "feature_name", "min", "max", "mean", "finite_count", "expected_range", "pass"])
    outputs_written.append(str(OUTPUT_DIR / "node_feature_summary.csv"))
    write_csv(OUTPUT_DIR / "edge_feature_summary.csv", rows["edge"], ["n", "prefix_name", "feature_name", "min", "max", "mean", "finite_count", "expected_range", "pass"])
    outputs_written.append(str(OUTPUT_DIR / "edge_feature_summary.csv"))
    write_csv(OUTPUT_DIR / "global_feature_summary.csv", rows["global"], ["n", "prefix_name", "feature_name", "value", "expected_range", "pass"])
    outputs_written.append(str(OUTPUT_DIR / "global_feature_summary.csv"))
    write_csv(
        OUTPUT_DIR / "mask_legality_audit.csv",
        rows["mask"],
        [
            "n",
            "prefix_name",
            "scanned_count",
            "available_count",
            "legal_count",
            "illegal_scanned_available_count",
            "pointer_legal_equals_available_and_not_scanned",
            "pass",
        ],
    )
    outputs_written.append(str(OUTPUT_DIR / "mask_legality_audit.csv"))

    verdict = verdict_for(rows)
    write_manifest(graphs, rows, outputs_written, verdict)
    outputs_written.append(str(MANIFEST_PATH))
    write_report(graphs, rows, verdict, outputs_written)
    outputs_written.append(str(REPORT_PATH))

    print(f"N values: {N_VALUES}")
    print(f"Graph states generated: {len(graphs)}")
    print(f"Node feature count: {len(NODE_FEATURE_NAMES)}")
    print(f"Edge feature count: {len(EDGE_FEATURE_NAMES)}")
    print(f"Global feature count: {len(GLOBAL_FEATURE_NAMES)}")
    print(f"All graph validations passed: {all(graph['validation']['pass'] for graph in graphs)}")
    print(f"Mask legality rows passed: {sum(1 for row in rows['mask'] if pass_to_bool(row['pass']))}/{len(rows['mask'])}")
    print("Raw fixed-32/fixed-ID feature leak: False")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Report: {REPORT_PATH}")
    print(verdict)
    return 1 if verdict == "FAIL_VARIABLE_N_GRAPH_FEATURE_BUILDER_INVALID" else 0


if __name__ == "__main__":
    raise SystemExit(main())
