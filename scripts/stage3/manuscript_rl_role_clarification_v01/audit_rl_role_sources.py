from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
OUTPUT_DIR = ROOT / "outputs" / "stage3_manuscript_rl_role_clarification_v01"
TABLE_DIR = OUTPUT_DIR / "tables"

SEARCH_ROOTS = [
    ROOT / "scripts" / "stage3",
    ROOT / "src" / "policies",
    ROOT / "outputs",
    ROOT / "docs" / "stage3",
    ROOT / "rl",
]

TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".json",
    ".csv",
    ".txt",
    ".yaml",
    ".yml",
}

SKIP_SUFFIXES = {
    ".odb",
    ".cae",
    ".inp",
    ".jnl",
    ".zip",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".pdf",
    ".svg",
    ".pkl",
    ".pt",
    ".pth",
    ".joblib",
}

MAX_TEXT_BYTES = 256_000

CATEGORIES = {
    "run26_gnn_reward_model": ["run26", "gnn_reward", "gnn reward", "GNNRewardModel"],
    "run26_pointer_behavior_cloning": ["run26", "graph_pointer_policy", "behavior cloning", "weighted imitation", "PointerPolicy"],
    "run06_graph_pointer_proxy": ["run_06", "graph pointer proxy", "GraphPointerPolicyPrototype", "deterministic proxy"],
    "run29_hybrid_surrogate_gnn": ["run29", "hybrid", "surrogate_gnn", "combined172"],
    "run43_48_53_58_63_68_73_model_updates": [
        "run43",
        "run48",
        "run53",
        "run58",
        "run63",
        "run68",
        "run73",
        "model update",
        "candidate generation",
    ],
    "run78_final_evidence_freeze": ["run78", "final evidence freeze", "stage3_final_claim", "FROZEN"],
    "old_maskable_ppo": ["ppo", "Maskable PPO", "train_maskable_ppo", "sb3-contrib"],
    "final_claim_map_boundary": ["claim boundary", "claim evidence map", "STAGE3_CLAIM_BOUNDARY"],
    "policy_diagnostics": ["OrderGraphMLP", "transition-frequency", "transition frequency", "graph-pointer diagnostics"],
    "active_learning_surrogate_selection": ["active-learning", "active learning", "surrogate-guided", "surrogate screened"],
}


@dataclass
class InventoryRow:
    category: str
    path: str
    source_root: str
    suffix: str
    size_bytes: int
    modified_utc: str
    matched_terms: str
    match_location: str
    notes: str


def safe_rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_text_sample(path: Path) -> str:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return ""
    if path.suffix.lower() in SKIP_SUFFIXES:
        return ""
    try:
        with path.open("rb") as fh:
            data = fh.read(MAX_TEXT_BYTES)
        return data.decode("utf-8", errors="ignore")
    except OSError:
        return ""


def match_file(path: Path, source_root: Path) -> list[InventoryRow]:
    stat = path.stat()
    rel = safe_rel(path)
    hay_name = rel.lower()
    text = read_text_sample(path)
    hay_text = text.lower()
    rows: list[InventoryRow] = []

    for category, terms in CATEGORIES.items():
        matched: list[str] = []
        locations: set[str] = set()
        for term in terms:
            term_l = term.lower()
            if term_l in hay_name:
                matched.append(term)
                locations.add("path")
            if hay_text and term_l in hay_text:
                matched.append(term)
                locations.add("text_sample")
        if matched:
            rows.append(
                InventoryRow(
                    category=category,
                    path=rel,
                    source_root=safe_rel(source_root),
                    suffix=path.suffix.lower(),
                    size_bytes=stat.st_size,
                    modified_utc=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                    matched_terms="; ".join(sorted(set(matched), key=str.lower)),
                    match_location="+".join(sorted(locations)),
                    notes="text sample capped; no ODB/CAE/INP/JNL content opened",
                )
            )
    return rows


def collect_inventory() -> list[InventoryRow]:
    rows: list[InventoryRow] = []
    seen: set[tuple[str, str]] = set()
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() in SKIP_SUFFIXES:
                continue
            for row in match_file(path, root):
                key = (row.category, row.path)
                if key not in seen:
                    rows.append(row)
                    seen.add(key)
    rows.sort(key=lambda r: (r.category, r.path.lower()))
    return rows


def write_outputs(rows: list[InventoryRow]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TABLE_DIR / "rl_role_source_inventory.csv"
    json_path = TABLE_DIR / "rl_role_source_inventory.json"
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(InventoryRow.__dataclass_fields__.keys())
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(ROOT),
        "search_roots": [str(p) for p in SEARCH_ROOTS],
        "safety": {
            "no_abaqus": True,
            "no_odb_opened": True,
            "no_solver": True,
            "no_training": True,
            "no_candidate_generation": True,
            "text_sample_cap_bytes": MAX_TEXT_BYTES,
        },
        "row_count": len(rows),
        "rows": [asdict(row) for row in rows],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    rows = collect_inventory()
    write_outputs(rows)
    print(f"RL_ROLE_SOURCE_INVENTORY_WRITTEN rows={len(rows)}")
    print(TABLE_DIR / "rl_role_source_inventory.csv")
    print(TABLE_DIR / "rl_role_source_inventory.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
