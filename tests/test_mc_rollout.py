"""
文件: tests/test_mc_rollout.py
用途: 验证 Monte Carlo Rollout 和折扣回报计算
运行: python -m tests.test_mc_rollout
"""

import numpy as np

from envs.maze import CoinMaze
from scripts.estimate_q_bias import capture_snapshot
from scripts.estimate_mc_return import (
    rollout_once,
    estimate_mc_return,
)


class FixedActionAgent:
    """
    测试用智能体：始终返回同一个动作。
    """

    def __init__(self, action):
        self.action = action

    def act(self, state, epsilon=0.0):
        return self.action


def test_discounted_return():
    """
    从 (0,0) 连续向左撞墙两次。
    每次奖励 -0.03。

    已执行 198 步，剩余 2 步。

    G = -0.03 + 0.95 * (-0.03)
      = -0.0585
    """
    env = CoinMaze(seed=42)
    env.reset()
    env.pos = np.array([0, 0])
    env.t = 198

    snapshot = capture_snapshot(env)

    agent = FixedActionAgent(action=2)

    result = rollout_once(
        agent,
        snapshot,
        gamma=0.95,
    )

    expected = -0.03 + 0.95 * (-0.03)

    assert np.isclose(
        result["discounted_return"],
        expected,
        atol=1e-8,
    )
    assert result["steps"] == 2
    assert result["outcome"] == "timeout"

    print("test_discounted_return PASSED")


def test_coin_terminal():
    """
    在 (0,6) 向右一步，到达 (0,7) 金币。
    折扣回报应为 +1.0。
    """
    env = CoinMaze(seed=42)
    env.reset()

    env.pos = np.array([0, 6])
    env.coin = (0, 7)
    env.t = 50

    snapshot = capture_snapshot(env)

    agent = FixedActionAgent(action=3)

    result = rollout_once(agent, snapshot)

    assert np.isclose(result["discounted_return"], 1.0)
    assert result["steps"] == 1
    assert result["outcome"] == "coin"

    print("test_coin_terminal PASSED")


def test_trap_terminal():
    """
    从 (2,1) 向右进入 (2,2) 陷阱。
    折扣回报应为 -1.0。
    """
    env = CoinMaze(seed=42)
    env.reset()

    env.pos = np.array([2, 1])
    env.coin = (0, 7)
    env.traps = {(2, 2), (2, 5), (5, 6)}
    env.t = 50

    snapshot = capture_snapshot(env)

    agent = FixedActionAgent(action=3)

    result = rollout_once(agent, snapshot)

    assert np.isclose(result["discounted_return"], -1.0)
    assert result["steps"] == 1
    assert result["outcome"] == "trap"

    print("test_trap_terminal PASSED")


def test_mc_repeatability():
    """
    30 次 Rollout 从相同状态出发。

    当前环境和固定动作策略均为确定性的，
    因此应得到相同回报。
    """
    env = CoinMaze(seed=42)
    env.reset()
    env.pos = np.array([0, 0])
    env.t = 198

    snapshot = capture_snapshot(env)
    original_pos = snapshot["pos"].copy()
    original_t = snapshot["t"]

    agent = FixedActionAgent(action=2)

    result = estimate_mc_return(
        agent,
        snapshot,
        n_rollouts=30,
        gamma=0.95,
    )

    assert len(result["returns"]) == 30

    assert np.isclose(result["mean_return"], -0.0585)

    assert np.isclose(
        result["std_return"],
        0.0,
        atol=1e-10,
    )

    assert result["outcomes"]["timeout"] == 30

    # Rollout 不能修改原始快照
    assert np.array_equal(snapshot["pos"], original_pos)
    assert snapshot["t"] == original_t

    print("test_mc_repeatability PASSED")


if __name__ == "__main__":
    test_discounted_return()
    test_coin_terminal()
    test_trap_terminal()
    test_mc_repeatability()

    print("\nAll MC Rollout tests PASSED!")
