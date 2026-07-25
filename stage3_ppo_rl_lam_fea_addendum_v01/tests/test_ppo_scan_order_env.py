from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from ppo_scan_order_env import (  # noqa: E402
    MAX_N,
    OBSERVATION_SIZE,
    SUPPORTED_N,
    LamScanOrderPPOEnv,
    validate_scan_order,
)


def test_reset_for_supported_n_values() -> None:
    for n in SUPPORTED_N:
        env = LamScanOrderPPOEnv(n=n, seed=123)
        obs, info = env.reset()
        assert info["n"] == n
        assert obs.shape == (OBSERVATION_SIZE,)


def test_initial_action_mask_length_and_valid_count() -> None:
    for n in SUPPORTED_N:
        env = LamScanOrderPPOEnv(n=n)
        env.reset()
        mask = env.action_masks()
        assert len(mask) == MAX_N
        assert int(mask.sum()) == n
        assert mask[:n].all()
        assert not mask[n:].any()


def test_valid_action_decreases_mask_by_one() -> None:
    env = LamScanOrderPPOEnv(n=12)
    env.reset()
    _, reward, terminated, truncated, _ = env.step(0)
    assert reward == 0.0
    assert not terminated
    assert not truncated
    assert int(env.action_masks().sum()) == 11


def test_repeated_action_is_invalid() -> None:
    env = LamScanOrderPPOEnv(n=12)
    env.reset()
    env.step(3)
    _, reward, terminated, _, info = env.step(3)
    assert terminated
    assert reward < 0.0
    assert info["illegal_action"] == 3


def test_action_outside_current_n_is_invalid() -> None:
    env = LamScanOrderPPOEnv(n=12)
    env.reset()
    _, reward, terminated, _, info = env.step(12)
    assert terminated
    assert reward < 0.0
    assert info["illegal_action"] == 12


def test_full_episode_produces_legal_permutation() -> None:
    env = LamScanOrderPPOEnv(n=16)
    env.reset()
    rewards = []
    terminal_info = None
    for action in range(16):
        _, reward, terminated, _, info = env.step(action)
        rewards.append(reward)
        terminal_info = info
    assert terminated
    assert rewards[:-1] == [0.0] * 15
    assert rewards[-1] != 0.0
    assert terminal_info is not None
    assert terminal_info["terminal_order"] == list(range(16))
    assert validate_scan_order(terminal_info["terminal_order"], 16).legal


def test_terminal_reward_only_at_end_for_valid_rollout() -> None:
    env = LamScanOrderPPOEnv(n=12)
    env.reset()
    rewards = []
    for action in [0, 2, 4, 6, 8, 10, 11, 9, 7, 5, 3, 1]:
        _, reward, terminated, _, _ = env.step(action)
        rewards.append(reward)
    assert terminated
    assert all(reward == 0.0 for reward in rewards[:-1])
    assert isinstance(rewards[-1], float)


def test_no_invalid_duplicate_tracks_in_generated_order() -> None:
    env = LamScanOrderPPOEnv(n=24)
    env.reset()
    while True:
        valid = np.flatnonzero(env.action_masks())
        _, _, terminated, _, info = env.step(int(valid[0]))
        if terminated:
            order = info["terminal_order"]
            break
    assert order is not None
    assert len(order) == 24
    assert len(set(order)) == 24
    assert validate_scan_order(order, 24).legal


def test_random_valid_rollout_works_for_all_n() -> None:
    rng = np.random.default_rng(321)
    for n in SUPPORTED_N:
        env = LamScanOrderPPOEnv(n=n, seed=321)
        env.reset()
        while True:
            valid = np.flatnonzero(env.action_masks())
            action = int(rng.choice(valid))
            _, _, terminated, _, info = env.step(action)
            if terminated:
                assert validate_scan_order(info["terminal_order"], n).legal
                break


def test_observation_shape_fixed_across_n() -> None:
    shapes = set()
    for n in SUPPORTED_N:
        env = LamScanOrderPPOEnv(n=n)
        obs, _ = env.reset()
        shapes.add(obs.shape)
        env.step(0)
        shapes.add(env._build_observation().shape)
    assert shapes == {(OBSERVATION_SIZE,)}
