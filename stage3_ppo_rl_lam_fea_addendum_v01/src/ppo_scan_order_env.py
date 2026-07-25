"""PPO-compatible scan-order environment skeleton for Stage 3 addendum v01.

This module has no Abaqus, ODB, solver, CAE, INP, JNL, or training dependency.
It constructs legal scan-order episodes and exposes a MaskablePPO-style
``action_masks`` method for later PPO training in a surrogate reward environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces

    _GYMNASIUM_AVAILABLE = True
except Exception:  # noqa: BLE001 - allows skeleton import when gymnasium is absent.
    gym = None
    spaces = None
    _GYMNASIUM_AVAILABLE = False


MAX_N = 40
SUPPORTED_N = (12, 16, 24, 40)
PER_TRACK_FEATURES = 10
SCALAR_FEATURES = 3
OBSERVATION_SIZE = SCALAR_FEATURES + (2 * MAX_N) + (MAX_N * PER_TRACK_FEATURES)
RewardFn = Callable[[list[int], int], float]


class _FallbackDiscrete:
    def __init__(self, n: int) -> None:
        self.n = n

    def sample(self) -> int:
        return int(np.random.randint(0, self.n))


class _FallbackBox:
    def __init__(self, low: float, high: float, shape: tuple[int, ...], dtype: object) -> None:
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype


@dataclass(frozen=True)
class ScanOrderValidation:
    legal: bool
    reason: str


def deterministic_smoke_reward(order: list[int], n: int) -> float:
    """Terminal-only test reward with no physical claim.

    The reward is deterministic and bounded. It exists only so smoke tests can
    confirm terminal reward plumbing before a supervised surrogate is added.
    """

    validation = validate_scan_order(order, n)
    if not validation.legal:
        return -100.0
    if n <= 1:
        return 0.0
    jumps = np.abs(np.diff(np.asarray(order, dtype=np.float32)))
    mean_jump = float(jumps.mean()) if jumps.size else 0.0
    return float(1.0 - (mean_jump / max(1, n - 1)))


def validate_scan_order(order: Iterable[int], n: int) -> ScanOrderValidation:
    order_list = [int(item) for item in order]
    if n not in SUPPORTED_N:
        return ScanOrderValidation(False, f"unsupported_n={n}")
    if len(order_list) != n:
        return ScanOrderValidation(False, f"length={len(order_list)} expected={n}")
    if any(item < 0 or item >= n for item in order_list):
        return ScanOrderValidation(False, "track_outside_current_n")
    if len(set(order_list)) != n:
        return ScanOrderValidation(False, "duplicate_track")
    return ScanOrderValidation(True, "legal_permutation")


BaseEnv = gym.Env if _GYMNASIUM_AVAILABLE else object


class LamScanOrderPPOEnv(BaseEnv):
    """Action-masked scan-order construction environment.

    Observation is a fixed-length float vector:
    ``[n_norm, step_norm, prev_norm, visited_mask_40, valid_mask_40,
    per_track_features_40x10]``.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n: int | None = None,
        supported_n: Iterable[int] = SUPPORTED_N,
        max_n: int = MAX_N,
        reward_fn: RewardFn | None = None,
        illegal_action_reward: float = -100.0,
        seed: int | None = None,
    ) -> None:
        if max_n != MAX_N:
            raise ValueError(f"This foundation skeleton uses fixed max_n={MAX_N}.")
        self.supported_n = tuple(int(value) for value in supported_n)
        unsupported = [value for value in self.supported_n if value not in SUPPORTED_N]
        if unsupported:
            raise ValueError(f"Unsupported N values: {unsupported}. Supported: {SUPPORTED_N}")
        if n is not None and int(n) not in self.supported_n:
            raise ValueError(f"n={n} is not in supported_n={self.supported_n}")

        self.fixed_n = int(n) if n is not None else None
        self.max_n = max_n
        self.reward_fn = reward_fn or deterministic_smoke_reward
        self.illegal_action_reward = float(illegal_action_reward)
        self.rng = np.random.default_rng(seed)
        self.n = self.fixed_n or int(self.rng.choice(self.supported_n))
        self.order: list[int] = []
        self.visited = np.zeros(self.max_n, dtype=bool)
        self.terminated = False
        self.truncated = False

        if _GYMNASIUM_AVAILABLE:
            self.action_space = spaces.Discrete(self.max_n)
            self.observation_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(OBSERVATION_SIZE,),
                dtype=np.float32,
            )
        else:
            self.action_space = _FallbackDiscrete(self.max_n)
            self.observation_space = _FallbackBox(
                low=-1.0,
                high=1.0,
                shape=(OBSERVATION_SIZE,),
                dtype=np.float32,
            )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        requested_n = options.get("n", self.fixed_n)
        if requested_n is None:
            self.n = int(self.rng.choice(self.supported_n))
        else:
            requested_n = int(requested_n)
            if requested_n not in self.supported_n:
                raise ValueError(f"Requested n={requested_n} is not in supported_n={self.supported_n}")
            self.n = requested_n

        self.order = []
        self.visited = np.zeros(self.max_n, dtype=bool)
        self.terminated = False
        self.truncated = False
        info = {
            "n": self.n,
            "step_index": 0,
            "gymnasium_available": _GYMNASIUM_AVAILABLE,
            "terminal_order": None,
        }
        return self._build_observation(), info

    def step(self, action: int):
        action = int(action)
        if self.terminated or self.truncated:
            raise RuntimeError("Episode is already terminated. Call reset() before step().")

        if not self._is_valid_action(action):
            self.terminated = True
            info = {
                "n": self.n,
                "step_index": len(self.order),
                "illegal_action": action,
                "terminal_order": list(self.order),
                "legality": "illegal_action",
            }
            return self._build_observation(), self.illegal_action_reward, True, False, info

        self.order.append(action)
        self.visited[action] = True
        terminated = len(self.order) == self.n
        self.terminated = terminated
        reward = float(self.reward_fn(list(self.order), self.n)) if terminated else 0.0
        validation = validate_scan_order(self.order, self.n) if terminated else None
        info = {
            "n": self.n,
            "step_index": len(self.order),
            "illegal_action": None,
            "terminal_order": list(self.order) if terminated else None,
            "legality": validation.reason if validation else "partial_order",
        }
        return self._build_observation(), reward, terminated, False, info

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(self.max_n, dtype=bool)
        if self.terminated or self.truncated:
            return mask
        mask[: self.n] = ~self.visited[: self.n]
        return mask

    def current_order(self) -> list[int]:
        return list(self.order)

    def terminal_order(self) -> list[int] | None:
        return list(self.order) if self.terminated and len(self.order) == self.n else None

    def validate_current_order(self) -> ScanOrderValidation:
        return validate_scan_order(self.order, self.n)

    def _is_valid_action(self, action: int) -> bool:
        return 0 <= action < self.n and not bool(self.visited[action])

    def _build_observation(self) -> np.ndarray:
        step = len(self.order)
        previous = self.order[-1] if self.order else -1
        n_norm = self.n / self.max_n
        step_norm = step / max(1, self.n)
        prev_norm = (previous / max(1, self.n - 1)) if previous >= 0 else -1.0

        visited_mask = self.visited.astype(np.float32)
        valid_mask = self.action_masks().astype(np.float32)
        per_track = np.zeros((self.max_n, PER_TRACK_FEATURES), dtype=np.float32)
        center = (self.n - 1) / 2.0
        denom = max(1.0, self.n - 1.0)

        for track in range(self.max_n):
            in_current_n = track < self.n
            normalized_track = (track / denom) if in_current_n else 0.0
            parity = float(track % 2) if in_current_n else 0.0
            center_distance = abs(track - center) / max(1.0, center) if in_current_n and center else 0.0
            edge_distance = min(track, self.n - 1 - track) / max(1.0, center) if in_current_n else 0.0
            visited_flag = float(self.visited[track])
            valid_flag = float(valid_mask[track])
            previous_flag = float(track == previous)
            if previous >= 0 and in_current_n:
                distance = abs(track - previous) / denom
                signed_distance = (track - previous) / denom
            else:
                distance = 0.0
                signed_distance = 0.0
            remaining_flag = float(in_current_n and not self.visited[track])
            per_track[track] = np.asarray(
                [
                    normalized_track,
                    parity,
                    center_distance,
                    edge_distance,
                    visited_flag,
                    valid_flag,
                    previous_flag,
                    distance,
                    signed_distance,
                    remaining_flag,
                ],
                dtype=np.float32,
            )

        obs = np.concatenate(
            [
                np.asarray([n_norm, step_norm, prev_norm], dtype=np.float32),
                visited_mask,
                valid_mask,
                per_track.reshape(-1),
            ]
        ).astype(np.float32)
        if obs.shape != (OBSERVATION_SIZE,):
            raise RuntimeError(f"Observation shape {obs.shape} != {(OBSERVATION_SIZE,)}")
        return obs
