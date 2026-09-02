"""Stage Y N32/N64 policy extrapolation for RL-LAM-ScanOpt.

This is inference/recovery/reporting only. It does not run Abaqus, open ODBs,
extract ODB metrics, run solver/datacheck/enqueue, generate CAE/INP/JNL, train
PPO/surrogates, mutate scan orders, commit, or push.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PRIMARY_ROOT = Path(r"E:\Projects\RL-LAM-ScanOpt")
LEGACY_ROOT = Path(r"D:\Projects\RL-LAM-ScanOpt")
NS = "stage3_policy_extrapolation_N32_N64_predicted_optima"
OUT_ROOT = PRIMARY_ROOT / "outputs" / NS
DOCS_ROOT = PRIMARY_ROOT / "docs" / NS
SCRIPT_ROOT = PRIMARY_ROOT / NS / "scripts"

DIRS = [
    OUT_ROOT,
    OUT_ROOT / "inventory",
    OUT_ROOT / "rollouts",
    OUT_ROOT / "tables",
    OUT_ROOT / "scan_orders",
    OUT_ROOT / "scan_orders" / "N32",
    OUT_ROOT / "scan_orders" / "N64",
    OUT_ROOT / "reports",
    DOCS_ROOT,
    SCRIPT_ROOT,
]

KNOWN_FINAL_PPO_CHECKPOINT = PRIMARY_ROOT / "outputs" / "stage3_ppo_rl_lam_fea_addendum_v01" / "ppo_training" / "checkpoints" / "maskable_ppo_lam_scan_order_final.zip"
KNOWN_FINAL_PPO_EVIDENCE = PRIMARY_ROOT / "outputs" / "stage3_ppo_final_pool_320_evidence_freeze"
COMBINED552 = PRIMARY_ROOT / "outputs" / "stage3_run_78_final_evidence_freeze_package" / "FROZEN_stage3_native_combined552_teacher_dataset.csv"
PLUS_N32 = PRIMARY_ROOT / "outputs" / "stage3_run_78_final_evidence_freeze_package" / "FROZEN_stage3_combined552_plus_N32_RL_ready_dataset.csv"
V01_SURROGATE = PRIMARY_ROOT / "outputs" / "stage3_ppo_rl_lam_fea_addendum_v01" / "surrogate_reward_model" / "models" / "ppo_surrogate_reward_model_best.joblib"
V01_SOURCE_ROOT = PRIMARY_ROOT / "stage3_ppo_rl_lam_fea_addendum_v01" / "src"

MAX_N_PPO = 40
OBS_SIZE = 3 + 2 * MAX_N_PPO + 40 * 10
N32_ROLLOUT_ATTEMPTS = 2048

FORBIDDEN_EXTENSIONS = {".odb", ".cae", ".inp", ".jnl", ".sim", ".sta", ".dat", ".msg", ".lck"}
POLICY_EXTENSIONS = {".zip", ".pt", ".pth", ".joblib", ".pkl"}
MODEL_EXTENSIONS = {".joblib", ".pkl", ".pt", ".pth", ".zip"}
ARTIFACT_KEYWORDS = [
    "N32",
    "n32",
    "fixed32",
    "fixed_32",
    "aux_N32",
    "auxiliary_N32",
    "probe32",
    "N32_teacher",
    "N32_surrogate",
    "N32_ranking",
    "N64",
    "n64",
    "variable_N",
    "variable-n",
    "graph_pointer",
    "graph-pointer",
    "pointer",
    "pointer_policy",
    "max_N",
    "max_n",
    "infer_N64",
    "rollout_N64",
    "policy_inference",
    "ppo_inference",
    "scan_order_N64",
]


def ensure_dirs() -> None:
    for path in DIRS:
        path.mkdir(parents=True, exist_ok=True)


def git_branch() -> str:
    try:
        result = subprocess.run(["git", "branch", "--show-current"], cwd=str(PRIMARY_ROOT), capture_output=True, text=True, check=False)
        return result.stdout.strip() or "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def sha_order(order: Iterable[int]) -> str:
    compact = ",".join(str(int(v)) for v in order)
    return hashlib.sha256(compact.encode("utf-8")).hexdigest()[:16]


def parse_order(value: object) -> list[int] | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, list):
        try:
            return [int(v) for v in value]
        except Exception:
            return None
    text = str(value).strip()
    if not text:
        return None
    if text.startswith("["):
        try:
            return [int(v) for v in json.loads(text)]
        except Exception:
            pass
    parts = [p for p in text.replace("|", ",").replace(";", ",").replace("-", ",").replace(" ", ",").split(",") if p != ""]
    try:
        return [int(float(p)) for p in parts]
    except Exception:
        return None


def legal_order(n: int, order: list[int] | None) -> bool:
    return order is not None and len(order) == n and sorted(order) == list(range(n))


def descriptors(n: int, order: list[int]) -> dict[str, float]:
    arr = np.asarray(order, dtype=float)
    jumps = np.abs(np.diff(arr)) if len(arr) > 1 else np.asarray([], dtype=float)
    adjacent = float(np.mean(jumps == 1)) if jumps.size else 0.0
    mean_jump = float(jumps.mean()) if jumps.size else 0.0
    total = float(jumps.sum()) if jumps.size else 0.0
    var = float(jumps.var()) if jumps.size else 0.0
    long_count = int(np.sum(jumps >= max(2.0, n / 3.0))) if jumps.size else 0
    return {
        "total_travel_proxy": total,
        "mean_abs_jump": mean_jump,
        "max_abs_jump": float(jumps.max()) if jumps.size else 0.0,
        "long_jump_count": long_count,
        "adjacent_fraction": adjacent,
        "jump_variance": var,
        "local_continuity_score": float(adjacent - mean_jump / max(1.0, n - 1)),
        "path_complexity_score": float(total / max(1.0, n * (n - 1)) + var / max(1.0, (n - 1) ** 2)),
    }


def walk_relevant_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    results: list[Path] = []
    skip_dirs = {".git", ".pytest_cache", "__pycache__"}
    for current, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        cur = Path(current)
        for name in files:
            path = cur / name
            if path.suffix.lower() in FORBIDDEN_EXTENSIONS:
                continue
            text = str(path).lower()
            if any(k.lower() in text for k in ARTIFACT_KEYWORDS) or ("ppo" in text and path.suffix.lower() in {".csv", ".json", ".md", ".zip", ".pt", ".pth", ".joblib", ".pkl", ".py"}):
                results.append(path)
    return results


def classify_checkpoint(path: Path) -> str:
    text = str(path).lower()
    if path == KNOWN_FINAL_PPO_CHECKPOINT or "maskable_ppo_lam_scan_order_final.zip" in text:
        return "Policy type A: Final MaskablePPO Discrete(40)"
    if any(token in text for token in ["variable_n", "variable-n", "graph_pointer", "graph-pointer", "pointer_policy", "graph policy"]):
        return "Policy type B candidate: variable-N / graph-pointer / pointer-policy"
    return "Policy type C: Unknown checkpoint/model"


def infer_max_n_from_text(path: Path) -> int | None:
    try:
        if path.suffix.lower() not in {".json", ".md", ".py", ".txt", ".csv"} or path.stat().st_size > 5_000_000:
            return None
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    import re

    vals = []
    for pat in [r"max[_-]?[Nn]\D{0,8}(\d+)", r"MAX[_-]?[Nn]\D{0,8}(\d+)", r"training[_-]?[Nn].{0,80}64", r"N64"]:
        for m in re.finditer(pat, text):
            if m.groups():
                vals.append(int(m.group(1)))
            elif "64" in m.group(0):
                vals.append(64)
    return max(vals) if vals else None


def inventory(search_roots: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    files: list[Path] = []
    for root in search_roots:
        files.extend(walk_relevant_files(root))
    files = sorted(set(files), key=lambda p: str(p).lower())

    policy_rows = []
    surrogate_rows = []
    artifact_rows = []
    capability_rows = []
    for path in files:
        ext = path.suffix.lower()
        lower = str(path).lower()
        if ext in POLICY_EXTENSIONS and any(k in lower for k in ["ppo", "policy", "pointer", "checkpoint", "model"]):
            ptype = classify_checkpoint(path)
            max_n = infer_max_n_from_text(path) if ext in {".json", ".md", ".py", ".txt", ".csv"} else None
            policy_rows.append(
                {
                    "path": str(path),
                    "extension": ext,
                    "file_size_bytes": path.stat().st_size,
                    "policy_classification": ptype,
                    "safe_for_N32": bool("Discrete(40)" in ptype or "variable-N" in ptype),
                    "safe_for_N64": bool("variable-N" in ptype and max_n is not None and max_n >= 64),
                    "max_n_evidence": max_n,
                }
            )
        if ext in MODEL_EXTENSIONS and any(k in lower for k in ["surrogate", "reward", "model"]):
            surrogate_rows.append({"path": str(path), "extension": ext, "file_size_bytes": path.stat().st_size, "notes": "existing model artifact; scoring only if compatible"})
        if any(k.lower() in lower for k in ["n32", "n64", "fixed32", "probe32", "rollout_n64", "infer_n64", "scan_order_n64"]):
            artifact_rows.append({"path": str(path), "extension": ext, "file_size_bytes": path.stat().st_size, "artifact_hint": "N32/N64 candidate/metric/scan-order/config artifact"})

    final_ppo_exists = KNOWN_FINAL_PPO_CHECKPOINT.exists()
    capability_rows.append(
        {
            "N": 32,
            "policy_type": "Final MaskablePPO Discrete(40)",
            "status": "READY" if final_ppo_exists else "BLOCKED",
            "reason": "Discrete(40) can infer N32 with custom N32 action mask" if final_ppo_exists else "final PPO checkpoint missing",
            "checkpoint": str(KNOWN_FINAL_PPO_CHECKPOINT),
        }
    )
    variable_safe = [r for r in policy_rows if r["safe_for_N64"]]
    capability_rows.append(
        {
            "N": 64,
            "policy_type": "Variable-N / graph-pointer / pointer-policy",
            "status": "READY" if variable_safe else "BLOCKED",
            "reason": "safe variable-N policy with max_N>=64 found" if variable_safe else "no explicit variable-N policy checkpoint/inference evidence with max_N>=64 found; final Discrete(40) PPO is blocked for N64",
            "checkpoint": variable_safe[0]["path"] if variable_safe else "",
        }
    )

    policy_df = pd.DataFrame(policy_rows)
    surrogate_df = pd.DataFrame(surrogate_rows)
    artifact_df = pd.DataFrame(artifact_rows)
    capability_df = pd.DataFrame(capability_rows)
    verdict = "PASS_STAGEY_POLICY_CAPABILITY_READY" if capability_df["status"].eq("READY").all() else "WARNING_STAGEY_POLICY_CAPABILITY_PARTIAL"
    summary = {
        "branch": git_branch(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "search_roots": [str(r) for r in search_roots],
        "policy_checkpoint_count": int(len(policy_df)),
        "surrogate_model_count": int(len(surrogate_df)),
        "N32_N64_artifact_count": int(len(artifact_df)),
        "N32_capability": capability_df[capability_df["N"] == 32]["status"].iloc[0],
        "N64_capability": capability_df[capability_df["N"] == 64]["status"].iloc[0],
        "verdict": verdict,
    }
    return policy_df, surrogate_df, artifact_df, capability_df, summary


def recover_existing_candidates(search_roots: list[Path], artifact_df: pd.DataFrame, n: int) -> pd.DataFrame:
    rows = []
    candidates = []
    for p in artifact_df["path"].tolist() if not artifact_df.empty else []:
        path = Path(p)
        if path.suffix.lower() not in {".csv", ".json"} or path.stat().st_size > 20_000_000:
            continue
        if f"n{n}" not in str(path).lower() and str(n) not in path.name:
            continue
        candidates.append(path)
    if n == 32 and PLUS_N32.exists():
        candidates.append(PLUS_N32)
    seen = set()
    for path in sorted(set(candidates), key=str):
        try:
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path)
                ncol = "n" if "n" in df.columns else ("N" if "N" in df.columns else None)
                if ncol is None:
                    continue
                sub = df[pd.to_numeric(df[ncol], errors="coerce").fillna(-1).astype(int) == n].head(500)
                for i, row in sub.iterrows():
                    order = None
                    for col in ["order_compact", "order_json", "scan_order", "scan_order_json", "order"]:
                        if col in row.index:
                            order = parse_order(row.get(col))
                            if legal_order(n, order):
                                break
                    if not legal_order(n, order):
                        continue
                    h = str(row.get("order_hash")) if "order_hash" in row.index and pd.notna(row.get("order_hash")) else sha_order(order)
                    key = (n, h, str(path))
                    if key in seen:
                        continue
                    seen.add(key)
                    metrics = {m: row.get(m, np.nan) if m in row.index else np.nan for m in ["u2_range", "peeq_max", "mises_max"]}
                    surface = np.nan
                    for col in ["surface_t_proxy", "surface_t_proxy_mpa", "surface_t_proxy_max_tensile_mpa"]:
                        if col in row.index:
                            surface = row.get(col)
                            break
                    has_teacher = all(pd.notna(metrics[m]) for m in metrics) and pd.notna(surface)
                    rows.append(
                        {
                            "N": n,
                            "strategy_name": str(row.get("strategy_name", row.get("handoff_strategy_name", f"recovered_N{n}_{len(rows)+1:04d}"))),
                            "source_file": str(path),
                            "source_stage": path.parent.name,
                            "candidate_source": str(row.get("candidate_source", "recovered_existing_artifact")),
                            "generated_by": str(row.get("generation_method", "existing_artifact")),
                            "evidence_tier": "teacher_metric_extracted" if has_teacher else "unclear",
                            "u2_range": metrics["u2_range"],
                            "peeq_max": metrics["peeq_max"],
                            "surface_t_proxy": surface,
                            "mises_max": metrics["mises_max"],
                            "predicted_reward": row.get("predicted_reward", row.get("predicted_quality_score", np.nan)),
                            "predicted_lex_score": row.get("predicted_lex_primary_score", np.nan),
                            "policy_score": np.nan,
                            "order_compact": ",".join(str(v) for v in order),
                            "order_hash": h,
                            "order_json_source": str(path),
                            "safe_claim": "existing teacher metrics found" if has_teacher else "existing recovered candidate only",
                            "caveat": "not a new Stage Y policy inference result",
                        }
                    )
            elif path.suffix.lower() == ".json":
                data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
                order = parse_order(data.get("order") or data.get("scan_order") or data.get("order_compact"))
                nn = int(data.get("n", data.get("N", n)))
                if nn != n or not legal_order(n, order):
                    continue
                h = str(data.get("order_hash") or sha_order(order))
                rows.append(
                    {
                        "N": n,
                        "strategy_name": str(data.get("strategy_name", path.stem)),
                        "source_file": str(path),
                        "source_stage": path.parent.name,
                        "candidate_source": str(data.get("candidate_source", "recovered_json")),
                        "generated_by": str(data.get("generated_by", "existing_json")),
                        "evidence_tier": "policy_inferred_prediction_only" if "ppo" in str(data).lower() or "policy" in str(data).lower() else "unclear",
                        "u2_range": np.nan,
                        "peeq_max": np.nan,
                        "surface_t_proxy": np.nan,
                        "mises_max": np.nan,
                        "predicted_reward": data.get("predicted_reward", data.get("predicted_quality_score", np.nan)),
                        "predicted_lex_score": data.get("predicted_lex_primary_score", np.nan),
                        "policy_score": np.nan,
                        "order_compact": ",".join(str(v) for v in order),
                        "order_hash": h,
                        "order_json_source": str(path),
                        "safe_claim": "existing recovered candidate only",
                        "caveat": "not a new Stage Y policy inference result",
                    }
                )
        except Exception:
            continue
    cols = [
        "N",
        "strategy_name",
        "source_file",
        "source_stage",
        "candidate_source",
        "generated_by",
        "evidence_tier",
        "u2_range",
        "peeq_max",
        "surface_t_proxy",
        "mises_max",
        "predicted_reward",
        "predicted_lex_score",
        "policy_score",
        "order_compact",
        "order_hash",
        "order_json_source",
        "safe_claim",
        "caveat",
    ]
    return pd.DataFrame(rows, columns=cols).drop_duplicates(subset=["N", "order_hash", "source_file"], keep="first")


@dataclass
class N32PPOEnv:
    n: int = 32
    max_n: int = 40
    seed: int = 20260830

    def __post_init__(self) -> None:
        import gymnasium as gym
        from gymnasium import spaces

        self.gym = gym
        self.action_space = spaces.Discrete(self.max_n)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)
        self.rng = np.random.default_rng(self.seed)
        self.order: list[int] = []
        self.visited = np.zeros(self.max_n, dtype=bool)
        self.terminated = False

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.order = []
        self.visited = np.zeros(self.max_n, dtype=bool)
        self.terminated = False
        return self._obs(), {"n": self.n}

    def step(self, action: int):
        action = int(action)
        if action < 0 or action >= self.n or self.visited[action]:
            self.terminated = True
            return self._obs(), -100.0, True, False, {"terminal_order": list(self.order), "legality": "illegal_action", "illegal_action": action}
        self.order.append(action)
        self.visited[action] = True
        done = len(self.order) == self.n
        self.terminated = done
        return self._obs(), 0.0, done, False, {"terminal_order": list(self.order) if done else None, "legality": "legal_permutation" if done else "partial_order"}

    def action_masks(self):
        mask = np.zeros(self.max_n, dtype=bool)
        if not self.terminated:
            mask[: self.n] = ~self.visited[: self.n]
        return mask

    def _obs(self):
        step = len(self.order)
        prev = self.order[-1] if self.order else -1
        n_norm = self.n / self.max_n
        step_norm = step / max(1, self.n)
        prev_norm = prev / max(1, self.n - 1) if prev >= 0 else -1.0
        visited_mask = self.visited.astype(np.float32)
        valid_mask = self.action_masks().astype(np.float32)
        per_track = np.zeros((self.max_n, 10), dtype=np.float32)
        center = (self.n - 1) / 2.0
        denom = max(1.0, self.n - 1.0)
        for track in range(self.max_n):
            in_n = track < self.n
            dist = abs(track - prev) / denom if prev >= 0 and in_n else 0.0
            signed = (track - prev) / denom if prev >= 0 and in_n else 0.0
            edge = min(track, self.n - 1 - track) / max(1.0, center) if in_n else 0.0
            center_dist = abs(track - center) / max(1.0, center) if in_n else 0.0
            per_track[track] = np.asarray(
                [
                    track / denom if in_n else 0.0,
                    float(track % 2) if in_n else 0.0,
                    center_dist,
                    edge,
                    float(self.visited[track]),
                    float(valid_mask[track]),
                    float(track == prev),
                    dist,
                    signed,
                    float(in_n and not self.visited[track]),
                ],
                dtype=np.float32,
            )
        return np.concatenate([[n_norm, step_norm, prev_norm], visited_mask, valid_mask, per_track.reshape(-1)]).astype(np.float32)


def run_n32_inference() -> tuple[pd.DataFrame, str]:
    if not KNOWN_FINAL_PPO_CHECKPOINT.exists():
        return pd.DataFrame(), "BLOCKED: final MaskablePPO checkpoint missing"
    try:
        from sb3_contrib import MaskablePPO
    except Exception as exc:
        return pd.DataFrame(), f"BLOCKED: sb3_contrib MaskablePPO import failed: {exc}"

    model = MaskablePPO.load(str(KNOWN_FINAL_PPO_CHECKPOINT))
    env = N32PPOEnv()
    rows = []
    seen = set()
    for attempt in range(N32_ROLLOUT_ATTEMPTS):
        mode = "deterministic" if attempt == 0 else "stochastic"
        obs, _ = env.reset(seed=2026083000 + attempt)
        done = False
        logprob_sum = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=(mode == "deterministic"), action_masks=env.action_masks())
            # Best-effort policy logprob for the selected action.
            try:
                import torch

                obs_tensor = torch.as_tensor(obs.reshape(1, -1), dtype=torch.float32, device=model.device)
                mask_tensor = torch.as_tensor(env.action_masks().reshape(1, -1), dtype=torch.bool, device=model.device)
                dist = model.policy.get_distribution(obs_tensor, action_masks=mask_tensor)
                lp = dist.distribution.log_prob(torch.as_tensor([int(action)], device=model.device))
                logprob_sum += float(lp.detach().cpu().numpy().reshape(-1)[0])
            except Exception:
                pass
            obs, _, terminated, truncated, info = env.step(int(action))
            done = bool(terminated or truncated)
        order = info.get("terminal_order")
        if not legal_order(32, order):
            continue
        h = sha_order(order)
        if h in seen:
            continue
        seen.add(h)
        rec = {
            "rank": np.nan,
            "N": 32,
            "strategy_name": f"POLICYX_N32_{len(rows)+1:04d}_{mode}",
            "evidence_tier": "policy_inferred_prediction_only",
            "teacher_metrics_available": False,
            "policy_source": "final MaskablePPO Discrete(40) with N32 action mask",
            "policy_checkpoint": str(KNOWN_FINAL_PPO_CHECKPOINT),
            "inference_mode": mode,
            "generated_by": "StageY_N32_policy_inference",
            "u2_range": np.nan,
            "peeq_max": np.nan,
            "surface_t_proxy": np.nan,
            "mises_max": np.nan,
            "predicted_reward": np.nan,
            "predicted_u2": np.nan,
            "predicted_peeq": np.nan,
            "predicted_surface_t": np.nan,
            "policy_score": logprob_sum,
            "order_compact": ",".join(str(v) for v in order),
            "order_hash": h,
            "order_json_path": "",
            "safe_claim": "N32 policy-inferred prediction only from final Discrete(40) MaskablePPO",
            "caveat": "not Abaqus teacher-metric-extracted; no physical optimum claim",
        }
        rec.update(descriptors(32, order))
        rows.append(rec)
    out = pd.DataFrame(rows)
    if out.empty:
        return out, "RAN: no unique legal N32 policy rollouts recovered"

    # The frozen v01 feature schema explicitly supports any legal N <= 40 order.
    # These scores rank policy-inferred candidates only; they are not teacher metrics.
    if V01_SURROGATE.exists():
        try:
            sys.path.insert(0, str(V01_SOURCE_ROOT))
            from ppo_surrogate_reward_model import PPOSurrogateRewardModel

            surrogate = PPOSurrogateRewardModel.load(V01_SURROGATE)
            rewards: list[float] = []
            diagnostics: list[dict[str, float]] = []
            for compact in out["order_compact"]:
                order = parse_order(compact)
                assert order is not None
                rewards.append(surrogate.predict_reward(32, order))
                diagnostics.append(surrogate.predict_diagnostics(32, order))
            out["predicted_reward"] = rewards
            if diagnostics:
                diagnostic_names = sorted({name for item in diagnostics for name in item})
                for name in diagnostic_names:
                    out[f"surrogate_{name}"] = [item.get(name, np.nan) for item in diagnostics]
            out["safe_claim"] = "N32 policy-inferred prediction ranked by frozen v01 surrogate; not Abaqus teacher-metric-extracted"
            out["caveat"] = "surrogate score is prediction only; no physical optimum claim"
        except Exception as exc:
            out["surrogate_scoring_status"] = f"unavailable: {type(exc).__name__}: {exc}"
    else:
        out["surrogate_scoring_status"] = "unavailable: frozen v01 surrogate model missing"
    return out, "RAN"


def rank_candidates(df: pd.DataFrame, n: int) -> pd.DataFrame:
    cols = [
        "rank",
        "N",
        "strategy_name",
        "evidence_tier",
        "teacher_metrics_available",
        "policy_source",
        "policy_checkpoint",
        "inference_mode",
        "generated_by",
        "u2_range",
        "peeq_max",
        "surface_t_proxy",
        "mises_max",
        "predicted_reward",
        "predicted_u2",
        "predicted_peeq",
        "predicted_surface_t",
        "policy_score",
        "total_travel_proxy",
        "mean_abs_jump",
        "max_abs_jump",
        "long_jump_count",
        "adjacent_fraction",
        "order_compact",
        "order_hash",
        "order_json_path",
        "safe_claim",
        "caveat",
    ]
    if df.empty:
        return pd.DataFrame(columns=cols)
    out = df.copy()
    has_teacher = out[["u2_range", "peeq_max", "surface_t_proxy"]].notna().all(axis=1).any()
    if has_teacher:
        out = out.sort_values(["u2_range", "peeq_max", "surface_t_proxy", "mises_max"], ascending=[True, True, True, True])
    elif out["predicted_u2"].notna().any():
        out = out.sort_values(["predicted_u2", "predicted_peeq", "predicted_surface_t", "predicted_reward"], ascending=[True, True, True, False])
    elif out["predicted_reward"].notna().any():
        out = out.sort_values(["predicted_reward", "total_travel_proxy", "adjacent_fraction"], ascending=[False, True, False])
    else:
        out = out.sort_values(["inference_mode", "policy_score", "total_travel_proxy", "adjacent_fraction", "long_jump_count"], ascending=[True, False, True, False, True])
    out["rank"] = np.arange(1, len(out) + 1)
    return out[[c for c in cols if c in out.columns]]


def export_top_jsons(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    target_dir = OUT_ROOT / "scan_orders" / f"N{n}"
    target_dir.mkdir(parents=True, exist_ok=True)
    for idx, row in out.head(20).iterrows():
        order = parse_order(row["order_compact"])
        path = target_dir / f"scan_order_{row['strategy_name']}.json"
        payload = {
            "strategy_name": row["strategy_name"],
            "N": n,
            "order": order,
            "order_hash": row["order_hash"],
            "policy_source": row.get("policy_source", ""),
            "checkpoint": row.get("policy_checkpoint", ""),
            "evidence_tier": row.get("evidence_tier", ""),
            "teacher_metrics_available": bool(row.get("teacher_metrics_available", False)),
            "physical_claim": "none",
            "caveat": "policy-inferred prediction only unless teacher metrics are present",
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        out.loc[idx, "order_json_path"] = str(path)
    return out


def write_claim_boundary(n32_status: str, n64_status: str) -> Path:
    path = DOCS_ROOT / "N32_N64_POLICY_EXTRAPOLATION_CLAIM_BOUNDARY.md"
    path.write_text(
        f"""# N32/N64 Policy Extrapolation Claim Boundary

## Safe Claims

- A trained RL/PPO policy was used for N32 policy inference if supported by checkpoint architecture.
- N32/N64 candidates are predicted / policy-inferred scan orders unless Abaqus teacher metrics are found.
- N64 inference requires variable-N policy support and cannot be attributed to the final Discrete(40) MaskablePPO.
- Surrogate-ranked candidates are not physical optima without teacher metrics.

## Unsafe Claims

- N64 is Abaqus teacher-validated unless teacher metrics exist.
- N64 was generated by the final Discrete(40) MaskablePPO.
- N32/N64 predicted candidate is globally optimal.
- Surrogate score alone proves physical quality.
- Policy inference equals physical optimisation proof.

## Stage Y Status

- N32 status: {n32_status}
- N64 status: {n64_status}
""",
        encoding="utf-8",
    )
    return path


def write_report(summary: dict, n32_ranked: pd.DataFrame, n64_ranked: pd.DataFrame, n32_status: str, n64_status: str, final_verdict: str, claim_path: Path) -> Path:
    path = DOCS_ROOT / "N32_N64_POLICY_EXTRAPOLATION_REPORT.md"
    n32_best = n32_ranked.iloc[0].to_dict() if not n32_ranked.empty else {}
    n64_best = n64_ranked.iloc[0].to_dict() if not n64_ranked.empty else {}
    path.write_text(
        f"""# N32/N64 Policy Extrapolation Report

## Purpose

Infer predicted N32 and N64 scan-order candidates from existing trained RL/PPO policy evidence, subject to architecture constraints.

## Architecture Boundary

The final Stage X MaskablePPO policy uses a `Discrete(40)` action space. It may be used for N32 with action masking, but it must not be used directly for N64. N64 requires a variable-N / graph-pointer / pointer-policy checkpoint with explicit N64 or max_N >= 64 support.

## Policy/Checkpoint Inventory

See `policy_checkpoint_inventory.csv`. Inventory verdict: {summary['verdict']}.

## Surrogate/Model Inventory

See `surrogate_model_inventory.csv`. Existing models are used for inventory only unless compatibility is explicit.

## Existing N32/N64 Artifact Recovery

Recovered candidate tables were written for N32 and N64. Recovered artifacts are not new Stage Y policy inference unless stated.

## N32 Inference Result

Status: {n32_status}. N32 used final MaskablePPO Discrete(40) with a custom N32 action mask. Output is prediction only.

## N64 Inference Result

Status: {n64_status}. The final Discrete(40) MaskablePPO was blocked for N64 by architecture.

## Best N32 Predicted Candidate

```json
{json.dumps(n32_best, indent=2, default=str)}
```

## Best N64 Predicted Candidate

```json
{json.dumps(n64_best, indent=2, default=str)}
```

## Industrial-Efficiency Proxy Interpretation

Travel and jump descriptors are path proxies only. They are not Abaqus teacher metrics and cannot establish physical or industrial optimality.

## Whether Teacher Metrics Exist

Any recovered teacher metrics are reported as recovered existing evidence. Stage Y did not run Abaqus, open ODBs, or extract metrics.

## Claim Boundary

See `{claim_path}`.

## Recommended Wording

"N32 candidates are policy-inferred predictions from the trained Discrete(40) MaskablePPO with N32 masking. N64 policy inference was blocked unless an explicit variable-N policy supporting N64 is identified. These outputs are predicted scan orders only and are not teacher-validated optima."

## Final Verdict

{final_verdict}
""",
        encoding="utf-8",
    )
    return path


def write_summary_table(n32_ranked: pd.DataFrame, n64_ranked: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n, df in [(32, n32_ranked), (64, n64_ranked)]:
        if not df.empty:
            teacher = df[df["teacher_metrics_available"].astype(bool)] if "teacher_metrics_available" in df.columns else pd.DataFrame()
            if not teacher.empty:
                r = teacher.sort_values(["u2_range", "peeq_max", "surface_t_proxy"]).iloc[0]
                rows.append(summary_row(f"best N{n} by teacher metrics if available", r, "teacher metrics"))
            r = df.iloc[0]
            rows.append(summary_row(f"best N{n} by surrogate/policy prediction", r, "policy/logprob and path proxy ranking"))
            r2 = df.sort_values("total_travel_proxy").iloc[0] if "total_travel_proxy" in df.columns else df.iloc[0]
            rows.append(summary_row(f"best N{n} by industrial-efficiency proxy", r2, "lowest total_travel_proxy"))
        else:
            rows.append(
                {
                    "category": f"best N{n} by surrogate/policy prediction",
                    "N": n,
                    "strategy_name": "",
                    "evidence_tier": "blocked",
                    "ranking_basis": "not available",
                    "teacher_metrics_available": False,
                    "u2_range": np.nan,
                    "peeq_max": np.nan,
                    "surface_t_proxy": np.nan,
                    "predicted_reward": np.nan,
                    "policy_score": np.nan,
                    "total_travel_proxy": np.nan,
                    "order_json_path": "",
                    "safe_claim": f"N{n} policy inference not available",
                    "unsafe_claim": f"N{n} is optimized or teacher-validated",
                }
            )
    return pd.DataFrame(rows)


def summary_row(category: str, row: pd.Series, basis: str) -> dict:
    return {
        "category": category,
        "N": int(row.get("N", np.nan)),
        "strategy_name": row.get("strategy_name", ""),
        "evidence_tier": row.get("evidence_tier", ""),
        "ranking_basis": basis,
        "teacher_metrics_available": bool(row.get("teacher_metrics_available", False)),
        "u2_range": row.get("u2_range", np.nan),
        "peeq_max": row.get("peeq_max", np.nan),
        "surface_t_proxy": row.get("surface_t_proxy", np.nan),
        "predicted_reward": row.get("predicted_reward", np.nan),
        "policy_score": row.get("policy_score", np.nan),
        "total_travel_proxy": row.get("total_travel_proxy", np.nan),
        "order_json_path": row.get("order_json_path", ""),
        "safe_claim": row.get("safe_claim", ""),
        "unsafe_claim": "global optimum or teacher-validated physical optimum",
    }


def main() -> None:
    ensure_dirs()
    search_roots = [PRIMARY_ROOT] + ([LEGACY_ROOT] if LEGACY_ROOT.exists() and LEGACY_ROOT != PRIMARY_ROOT else [])

    policy_df, surrogate_df, artifact_df, capability_df, inv_summary = inventory(search_roots)
    policy_inventory_path = OUT_ROOT / "inventory" / "policy_checkpoint_inventory.csv"
    surrogate_inventory_path = OUT_ROOT / "inventory" / "surrogate_model_inventory.csv"
    artifact_inventory_path = OUT_ROOT / "inventory" / "N32_N64_existing_artifact_inventory.csv"
    capability_path = OUT_ROOT / "tables" / "N32_N64_policy_capability_audit.csv"
    inv_summary_path = OUT_ROOT / "inventory" / "stageY_inventory_summary.json"
    policy_df.to_csv(policy_inventory_path, index=False)
    surrogate_df.to_csv(surrogate_inventory_path, index=False)
    artifact_df.to_csv(artifact_inventory_path, index=False)
    capability_df.to_csv(capability_path, index=False)
    inv_summary_path.write_text(json.dumps(inv_summary, indent=2), encoding="utf-8")

    n32_recovered = recover_existing_candidates(search_roots, artifact_df, 32)
    n64_recovered = recover_existing_candidates(search_roots, artifact_df, 64)
    n32_recovered_path = OUT_ROOT / "tables" / "N32_existing_candidates_recovered.csv"
    n64_recovered_path = OUT_ROOT / "tables" / "N64_existing_candidates_recovered.csv"
    n32_recovered.to_csv(n32_recovered_path, index=False)
    n64_recovered.to_csv(n64_recovered_path, index=False)

    n32_pool, n32_status = run_n32_inference()
    n32_pool_path = OUT_ROOT / "rollouts" / "N32_policy_inference_rollout_pool.csv"
    n32_pool.to_csv(n32_pool_path, index=False)

    n64_status = "BLOCKED: no safe variable-N / graph-pointer / pointer-policy checkpoint with explicit N64 or max_N>=64 support found"
    n64_pool_path = OUT_ROOT / "rollouts" / "N64_policy_inference_BLOCKED.csv"
    blocked_cols = [
        "rank",
        "N",
        "strategy_name",
        "evidence_tier",
        "teacher_metrics_available",
        "policy_source",
        "policy_checkpoint",
        "inference_mode",
        "generated_by",
        "u2_range",
        "peeq_max",
        "surface_t_proxy",
        "mises_max",
        "predicted_reward",
        "predicted_u2",
        "predicted_peeq",
        "predicted_surface_t",
        "policy_score",
        "total_travel_proxy",
        "mean_abs_jump",
        "max_abs_jump",
        "long_jump_count",
        "adjacent_fraction",
        "order_compact",
        "order_hash",
        "order_json_path",
        "safe_claim",
        "caveat",
        "blocked_reason",
    ]
    pd.DataFrame(columns=blocked_cols).assign(blocked_reason=[]).to_csv(n64_pool_path, index=False)

    n32_ranked = rank_candidates(n32_pool, 32)
    n32_ranked = export_top_jsons(n32_ranked, 32)
    n32_ranked_path = OUT_ROOT / "tables" / "N32_predicted_optima_ranked.csv"
    n32_ranked.to_csv(n32_ranked_path, index=False)

    n64_ranked = rank_candidates(n64_recovered[n64_recovered["evidence_tier"].isin(["teacher_metric_extracted", "policy_inferred_prediction_only"])].copy(), 64) if not n64_recovered.empty else pd.DataFrame(columns=n32_ranked.columns)
    n64_ranked = export_top_jsons(n64_ranked, 64)
    n64_ranked_path = OUT_ROOT / "tables" / "N64_predicted_optima_ranked.csv"
    n64_ranked.to_csv(n64_ranked_path, index=False)

    summary_table = write_summary_table(n32_ranked, n64_ranked)
    summary_table_path = OUT_ROOT / "tables" / "N32_N64_predicted_best_summary.csv"
    summary_table.to_csv(summary_table_path, index=False)

    n64_ran = False
    final_verdict = "PASS_N32_COMPLETED_N64_BLOCKED_BY_ARCHITECTURE" if n32_status == "RAN" and not n64_ran else "WARNING_N32_N64_POLICY_EXTRAPOLATION_PARTIAL"
    claim_path = write_claim_boundary(n32_status, n64_status)
    report_path = write_report(inv_summary, n32_ranked, n64_ranked, n32_status, n64_status, final_verdict, claim_path)

    manifest_path = OUT_ROOT / "stageY_N32_N64_policy_extrapolation_manifest.json"
    manifest = {
        "branch": git_branch(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "search_roots": [str(r) for r in search_roots],
        "policy_inventory_path": str(policy_inventory_path),
        "surrogate_inventory_path": str(surrogate_inventory_path),
        "artifact_inventory_path": str(artifact_inventory_path),
        "capability_audit_path": str(capability_path),
        "N32_rollout_pool_path": str(n32_pool_path),
        "N64_rollout_pool_or_blocked_path": str(n64_pool_path),
        "N32_ranked_table_path": str(n32_ranked_path),
        "N64_ranked_table_path": str(n64_ranked_path),
        "top_JSON_directories": {"N32": str(OUT_ROOT / "scan_orders" / "N32"), "N64": str(OUT_ROOT / "scan_orders" / "N64")},
        "summary_table_path": str(summary_table_path),
        "report_path": str(report_path),
        "claim_boundary_path": str(claim_path),
        "N32_status": n32_status,
        "N64_status": n64_status,
        "N32_rollout_pool_size": int(len(n32_pool)),
        "N32_unique_count": int(n32_pool["order_hash"].nunique()) if not n32_pool.empty else 0,
        "N64_rollout_pool_size": 0,
        "N64_unique_count": int(n64_ranked["order_hash"].nunique()) if not n64_ranked.empty and "order_hash" in n64_ranked.columns else 0,
        "no_Abaqus": True,
        "no_ODB_opening": True,
        "no_solver": True,
        "no_datacheck": True,
        "no_enqueue": True,
        "no_training": True,
        "no_scan_order_mutation": True,
        "no_commit_or_push": True,
        "final_verdict": final_verdict,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
