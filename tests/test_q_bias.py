"""
文件: tests/test_q_bias.py
用途: 验证 Q 值估计偏差的计算与汇总
运行: python -m tests.test_q_bias
"""

import numpy as np
import torch

from envs.maze import CoinMaze
from scripts.estimate_q_bias import capture_snapshot
from scripts.compare_q_bias import (
    measure_agent_bias,
    summarize_bias,
)


class ConstantQNet(torch.nn.Module):
    def forward(self, x):
        return torch.tensor(
            [[0.0, 1.0, 2.0, 3.0]],
            dtype=torch.float32,
        ).repeat(x.shape[0], 1)


class FixedAgent:
    def __init__(self):
        self.online = ConstantQNet()

    def act(self, state, epsilon=0.0):
        # 永远向左，测试撞墙行为
        return 2


def test_bias_formula():
    env = CoinMaze(seed=42)
    env.reset()
    env.pos = np.array([0, 0])
    env.t = 198

    snapshot = capture_snapshot(env)
    agent = FixedAgent()

    rows = measure_agent_bias(agent, [snapshot], full_30=False)

    assert len(rows) == 1

    # max Q = 3.0
    # G = -0.03 + 0.95*(-0.03) = -0.0585
    # Delta = 3.0 - (-0.0585) = 3.0585
    assert np.isclose(rows[0]["pred_max_q"], 3.0)

    assert np.isclose(rows[0]["mc_return"], -0.0585)

    assert np.isclose(rows[0]["delta"], 3.0585)

    assert rows[0]["outcome"] == "timeout"

    print("test_bias_formula PASSED")


def test_summary():
    rows = [
        {
            "pred_max_q": 0.5,
            "mc_return": 0.2,
            "delta": 0.3,
        },
        {
            "pred_max_q": 0.2,
            "mc_return": 0.4,
            "delta": -0.2,
        },
    ]

    stats = summarize_bias(rows)

    assert stats["n_states"] == 2

    assert np.isclose(stats["mean_delta"], 0.05)

    assert np.isclose(stats["mean_abs_delta"], 0.25)

    assert np.isclose(stats["positive_fraction"], 0.5)

    print("test_summary PASSED")


def test_full_30_consistency():
    env = CoinMaze(seed=42)
    env.reset()
    env.pos = np.array([0, 0])
    env.t = 198

    snapshot = capture_snapshot(env)
    agent = FixedAgent()

    fast = measure_agent_bias(agent, [snapshot], full_30=False)

    full = measure_agent_bias(agent, [snapshot], full_30=True)

    assert np.isclose(
        fast[0]["mc_return"],
        full[0]["mc_return"],
    )

    assert np.isclose(
        fast[0]["delta"],
        full[0]["delta"],
    )

    print("test_full_30_consistency PASSED")


if __name__ == "__main__":
    test_bias_formula()
    test_summary()
    test_full_30_consistency()

    print("\nAll Q-bias tests PASSED!")
