"""
文件: tests/test_target_ablation.py
用途: 验证 Target Network 消融同步频率

运行:
python -m unittest tests.test_target_ablation -v
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import train as train_module


class FakeBuffer:
    def push(self, *args):
        pass


class FakeAgent:
    last_instance = None

    def __init__(self):
        self.buf = FakeBuffer()
        self.sync_calls = 0
        FakeAgent.last_instance = self

    def act(self, state, epsilon):
        return 0

    def update(self):
        return 0.1

    def sync_target(self):
        self.sync_calls += 1


class FakeMaze:
    def __init__(self, seed=0):
        self.pos = (0, 0)
        self.coin = (7, 7)
        self.t = 0

    def reset(self):
        self.t = 0
        return np.zeros(75, dtype=np.float32)

    def step(self, action):
        self.t += 1
        done = self.t >= 3
        return (
            np.zeros(75, dtype=np.float32),
            -0.01,
            done,
            {},
        )


class TargetAblationTest(unittest.TestCase):

    def run_fake_training(self, interval):
        # 隔离输出文件，避免污染真实项目
        with tempfile.TemporaryDirectory() as tmp:
            with patch("os.getcwd", return_value=tmp):
                pass

            import os

            original_cwd = os.getcwd()
            try:
                os.chdir(tmp)

                with (
                    patch.object(train_module, "CoinMaze", FakeMaze),
                    patch.object(train_module, "DDQNAgent", FakeAgent),
                ):
                    train_module.train(
                        n_episodes=1,
                        seed=0,
                        algorithm="ddqn",
                        run_tag=f"test_sync_{interval}",
                        target_sync_interval=interval,
                    )

                    calls = FakeAgent.last_instance.sync_calls
            finally:
                os.chdir(original_cwd)

        return calls

    def test_sync_every_step(self):
        # 一回合恰好3个环境步
        self.assertEqual(
            self.run_fake_training(interval=1),
            3,
        )

    def test_baseline_does_not_sync_early(self):
        # 3步尚未达到500步的同步周期
        self.assertEqual(
            self.run_fake_training(interval=500),
            0,
        )

    def test_invalid_interval(self):
        with self.assertRaises(ValueError):
            train_module.train(
                n_episodes=1,
                seed=0,
                target_sync_interval=0,
            )


if __name__ == "__main__":
    unittest.main()
