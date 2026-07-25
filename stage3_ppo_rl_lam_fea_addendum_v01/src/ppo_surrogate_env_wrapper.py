"""Surrogate-reward PPO environment wrapper for Stage C training.

This wrapper uses the frozen supervised FEA-teacher reward emulator as the
terminal reward function for ``LamScanOrderPPOEnv``. It has no Abaqus, ODB,
solver, CAE, INP, JNL, enqueue, or candidate-generation dependency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ppo_scan_order_env import LamScanOrderPPOEnv, SUPPORTED_N, validate_scan_order
from ppo_surrogate_reward_model import PPOSurrogateRewardModel


@dataclass
class PPOSurrogateEnvConfig:
    model_path: str
    feature_schema_path: str | None = None
    target_schema_path: str | None = None
    fixed_n: int | None = None
    random_n: bool = True
    supported_n: tuple[int, ...] = SUPPORTED_N
    n_sampling_mode: str = "balanced"
    reward_clip_min: float | None = -1.0
    reward_clip_max: float | None = 1.2
    reward_scale: float = 1.0
    illegal_action_penalty: float = -100.0
    seed: int = 20260623

    def to_dict(self) -> dict:
        data = asdict(self)
        data["supported_n"] = list(self.supported_n)
        return data


class LamScanOrderSurrogateRewardEnv(LamScanOrderPPOEnv):
    """Lam scan-order environment with frozen surrogate terminal reward."""

    def __init__(
        self,
        surrogate_model: PPOSurrogateRewardModel,
        config: PPOSurrogateEnvConfig,
    ) -> None:
        self.surrogate_model = surrogate_model
        self.config = config
        self.episode_log: list[dict] = []
        self._episode_counter = 0
        fixed_n = config.fixed_n if not config.random_n else None
        super().__init__(
            n=fixed_n,
            supported_n=config.supported_n,
            reward_fn=self._surrogate_terminal_reward,
            illegal_action_reward=config.illegal_action_penalty,
            seed=config.seed,
        )

    @classmethod
    def from_config(cls, config: PPOSurrogateEnvConfig) -> "LamScanOrderSurrogateRewardEnv":
        model = PPOSurrogateRewardModel.load(
            config.model_path,
            config.feature_schema_path,
            config.target_schema_path,
        )
        return cls(surrogate_model=model, config=config)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        options = dict(options or {})
        if self.config.random_n and "n" not in options and self.config.n_sampling_mode == "balanced":
            options["n"] = int(self.rng.choice(self.config.supported_n))
        obs, info = super().reset(seed=seed, options=options)
        info["n_sampling_mode"] = self.config.n_sampling_mode
        info["surrogate_reward_model"] = str(self.config.model_path)
        return obs, info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = super().step(action)
        if terminated or truncated:
            terminal_order = info.get("terminal_order")
            validation = validate_scan_order(terminal_order or [], self.n)
            episode_record = {
                "episode_index": self._episode_counter,
                "n": int(self.n),
                "episode_length": int(len(self.order)),
                "terminal_reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "legal": bool(validation.legal),
                "legality": validation.reason if validation else info.get("legality"),
                "illegal_action": info.get("illegal_action"),
                "terminal_order": ",".join(str(item) for item in (terminal_order or [])),
            }
            self.episode_log.append(episode_record)
            self._episode_counter += 1
            info["episode_record"] = episode_record
        return obs, reward, terminated, truncated, info

    def _surrogate_terminal_reward(self, order: Sequence[int], n: int) -> float:
        validation = validate_scan_order(order, n)
        if not validation.legal:
            return float(self.config.illegal_action_penalty)
        reward = float(self.surrogate_model.predict_reward(int(n), [int(item) for item in order]))
        reward *= float(self.config.reward_scale)
        if self.config.reward_clip_min is not None:
            reward = max(float(self.config.reward_clip_min), reward)
        if self.config.reward_clip_max is not None:
            reward = min(float(self.config.reward_clip_max), reward)
        return float(reward)


def build_surrogate_env_from_paths(
    model_path: str | Path,
    feature_schema_path: str | Path | None = None,
    target_schema_path: str | Path | None = None,
    fixed_n: int | None = None,
    random_n: bool = True,
    supported_n: Iterable[int] = SUPPORTED_N,
    seed: int = 20260623,
) -> LamScanOrderSurrogateRewardEnv:
    config = PPOSurrogateEnvConfig(
        model_path=str(model_path),
        feature_schema_path=str(feature_schema_path) if feature_schema_path is not None else None,
        target_schema_path=str(target_schema_path) if target_schema_path is not None else None,
        fixed_n=fixed_n,
        random_n=random_n,
        supported_n=tuple(int(value) for value in supported_n),
        seed=seed,
    )
    return LamScanOrderSurrogateRewardEnv.from_config(config)


def verify_action_masks_for_supported_n(env_config: PPOSurrogateEnvConfig) -> list[dict]:
    rows: list[dict] = []
    for n in env_config.supported_n:
        config = PPOSurrogateEnvConfig(**{**env_config.to_dict(), "fixed_n": int(n), "random_n": False})
        config.supported_n = tuple(env_config.supported_n)
        env = LamScanOrderSurrogateRewardEnv.from_config(config)
        obs, info = env.reset(options={"n": int(n)})
        initial_mask = env.action_masks()
        first_action = int(np.flatnonzero(initial_mask)[0])
        _, reward, terminated, truncated, step_info = env.step(first_action)
        after_mask = env.action_masks()
        rows.append(
            {
                "n": int(n),
                "observation_shape": "x".join(str(dim) for dim in obs.shape),
                "initial_mask_length": int(len(initial_mask)),
                "initial_valid_actions": int(initial_mask.sum()),
                "first_action": first_action,
                "valid_actions_after_one_step": int(after_mask.sum()),
                "intermediate_reward": float(reward),
                "terminated_after_one_step": bool(terminated),
                "truncated_after_one_step": bool(truncated),
                "legality_after_one_step": step_info.get("legality"),
                "pass": bool(
                    len(initial_mask) == 40
                    and int(initial_mask.sum()) == int(n)
                    and int(after_mask.sum()) == int(n) - 1
                    and reward == 0.0
                    and not terminated
                    and not truncated
                ),
            }
        )
    return rows
