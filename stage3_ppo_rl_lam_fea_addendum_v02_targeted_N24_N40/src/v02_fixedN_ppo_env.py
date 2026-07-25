"""Fixed-N action-masked PPO environment for targeted N24/N40 v02."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as exc:  # pragma: no cover
    raise RuntimeError("gymnasium is required for PPO v02 training") from exc

from v02_surrogate_reward_interface import V02SurrogateReward


MAX_N = 40
PER_TRACK_FEATURES = 10
OBSERVATION_SIZE = 3 + (2 * MAX_N) + (MAX_N * PER_TRACK_FEATURES)


@dataclass(frozen=True)
class V02EnvConfig:
    n: int
    surrogate_model_path: str
    reward_clip_min: float | None = None
    reward_clip_max: float | None = None
    reward_scale: float = 1.0
    illegal_action_penalty: float = -100.0
    seed: int = 20260624
    conservative_reward: bool = True


def validate_order(n: int, order: Sequence[int]) -> bool:
    order_list = [int(x) for x in order]
    return int(n) in (24, 40) and len(order_list) == int(n) and sorted(order_list) == list(range(int(n)))


class FixedNLamScanOrderPPOEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: V02EnvConfig) -> None:
        super().__init__()
        if int(config.n) not in (24, 40):
            raise ValueError(f"v02 supports fixed N24/N40 only, got {config.n}")
        self.config = config
        self.n = int(config.n)
        self.reward_model = V02SurrogateReward.load(config.surrogate_model_path)
        self.rng = np.random.default_rng(config.seed)
        self.order: list[int] = []
        self.visited = np.zeros(MAX_N, dtype=bool)
        self.terminated = False
        self.action_space = spaces.Discrete(MAX_N)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBSERVATION_SIZE,), dtype=np.float32)
        self.completed_episodes: list[dict] = []

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.order = []
        self.visited = np.zeros(MAX_N, dtype=bool)
        self.terminated = False
        return self._obs(), {"n": self.n, "step_index": 0}

    def step(self, action: int):
        action = int(action)
        if self.terminated:
            raise RuntimeError("Episode already terminated")
        if action < 0 or action >= self.n or self.visited[action]:
            self.terminated = True
            return self._obs(), float(self.config.illegal_action_penalty), True, False, {"illegal_action": action, "terminal_order": list(self.order), "legality": "illegal_action"}
        self.order.append(action)
        self.visited[action] = True
        done = len(self.order) == self.n
        self.terminated = done
        reward = 0.0
        info = {"n": self.n, "step_index": len(self.order), "illegal_action": None, "terminal_order": None, "legality": "partial_order"}
        if done:
            if not validate_order(self.n, self.order):
                reward = float(self.config.illegal_action_penalty)
                info["legality"] = "illegal_terminal_order"
            else:
                reward = self.reward_model.predict_reward(self.n, self.order, conservative=self.config.conservative_reward)
                reward *= float(self.config.reward_scale)
                if self.config.reward_clip_min is not None or self.config.reward_clip_max is not None:
                    lo = -np.inf if self.config.reward_clip_min is None else self.config.reward_clip_min
                    hi = np.inf if self.config.reward_clip_max is None else self.config.reward_clip_max
                    reward = float(np.clip(reward, lo, hi))
                info["legality"] = "legal_permutation"
            info["terminal_order"] = list(self.order)
            self.completed_episodes.append({"n": self.n, "order": list(self.order), "terminal_reward": float(reward), "episode_length": len(self.order)})
        return self._obs(), float(reward), done, False, info

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(MAX_N, dtype=bool)
        if not self.terminated:
            mask[: self.n] = ~self.visited[: self.n]
        return mask

    def _obs(self) -> np.ndarray:
        step = len(self.order)
        prev = self.order[-1] if self.order else -1
        n_norm = self.n / MAX_N
        step_norm = step / max(1, self.n)
        prev_norm = prev / max(1, self.n - 1) if prev >= 0 else -1.0
        visited_mask = self.visited.astype(np.float32)
        valid_mask = self.action_masks().astype(np.float32)
        per_track = np.zeros((MAX_N, PER_TRACK_FEATURES), dtype=np.float32)
        center = (self.n - 1) / 2.0
        denom = max(1.0, self.n - 1.0)
        for track in range(MAX_N):
            in_n = track < self.n
            dist = 0.0 if prev < 0 or not in_n else abs(track - prev) / denom
            signed = 0.0 if prev < 0 or not in_n else (track - prev) / denom
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
