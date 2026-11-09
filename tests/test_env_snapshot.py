
"""
文件: tests/test_env_snapshot.py
用途: 验证迷宫环境快照的保存与恢复
运行: python -m tests.test_env_snapshot
"""

import numpy as np

from envs.maze import CoinMaze
from scripts.estimate_q_bias import (
    capture_snapshot,
    restore_snapshot,
)


def test_snapshot_state():
    env = CoinMaze(seed=42)
    env.reset()
    env.step(3)

    snapshot = capture_snapshot(env)
    restored = restore_snapshot(snapshot)

    assert np.array_equal(env.pos, restored.pos)
    assert env.coin == restored.coin
    assert env.traps == restored.traps
    assert env.t == restored.t
    assert env.size == restored.size
    assert env.max_steps == restored.max_steps
    assert np.array_equal(env.obs(), restored.obs())

    print("test_snapshot_state PASSED")


def test_snapshot_independence():
    env = CoinMaze(seed=42)
    env.reset()

    snapshot = capture_snapshot(env)

    original_pos = snapshot["pos"].copy()
    original_traps = snapshot["traps"].copy()

    # 修改原环境
    env.pos[0] = 4
    env.traps.add((0, 0))

    # 快照不应该变化
    assert np.array_equal(snapshot["pos"], original_pos)
    assert snapshot["traps"] == original_traps

    # 修改恢复后的环境
    restored = restore_snapshot(snapshot)
    restored.pos[0] = 5
    restored.traps.add((0, 1))

    # 快照仍然不应该变化
    assert np.array_equal(snapshot["pos"], original_pos)
    assert snapshot["traps"] == original_traps

    print("test_snapshot_independence PASSED")


def test_snapshot_step():
    env = CoinMaze(seed=123)
    env.reset()
    env.step(3)

    snapshot = capture_snapshot(env)
    restored = restore_snapshot(snapshot)

    # 从相同状态执行相同动作
    obs1, reward1, done1, info1 = env.step(3)
    obs2, reward2, done2, info2 = restored.step(3)

    assert np.array_equal(obs1, obs2)
    assert reward1 == reward2
    assert done1 == done2
    assert info1 == info2
    assert np.array_equal(env.pos, restored.pos)
    assert env.t == restored.t

    print("test_snapshot_step PASSED")


def test_snapshot_timeout():
    env = CoinMaze(seed=42)
    env.reset()

    # 模拟已经执行 199 步
    env.t = 199

    snapshot = capture_snapshot(env)
    restored = restore_snapshot(snapshot)

    _, _, done, _ = restored.step(3)

    assert done is True
    assert restored.t == 200

    print("test_snapshot_timeout PASSED")


if __name__ == "__main__":
    test_snapshot_state()
    test_snapshot_independence()
    test_snapshot_step()
    test_snapshot_timeout()

    print("\nAll environment snapshot tests PASSED!")
