from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


TARGET_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
OUTPUT_DIR = TARGET_ROOT / "outputs" / "stage3_run_06_variable_n_probe60_candidate_order_generation"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch_available = False
    try:
        import torch  # type: ignore

        torch_available = True
        torch_version = str(torch.__version__)
    except Exception as exc:
        torch_version = f"unavailable: {exc}"
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": "run_06b_variable_n_graph_pointer_policy_probe_training",
        "python_executable": sys.executable,
        "torch_available": torch_available,
        "torch_version": torch_version,
        "training_performed": False,
        "policy_status": "proxy_policy",
        "reason": "No new teacher labels for true variable-N; run_06 uses deterministic graph pointer proxy candidates only.",
        "no_checkpoint_written": True,
        "no_odb_opened": True,
        "no_abaqus_jobs": True,
    }
    (OUTPUT_DIR / "policy_probe_training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (OUTPUT_DIR / "policy_probe_training_log.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "event", "status", "notes"])
        writer.writeheader()
        writer.writerow({"step": 0, "event": "probe_start", "status": "ok", "notes": "Small text-only policy probe."})
        writer.writerow({"step": 1, "event": "training_decision", "status": "skipped", "notes": "Proxy-only: no true variable-N teacher labels available."})
        writer.writerow({"step": 2, "event": "candidate_policy_source", "status": "proxy_policy", "notes": "GraphPointerPolicyPrototype deterministic decoders used."})
    print("PROXY_POLICY_FALLBACK_USED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
