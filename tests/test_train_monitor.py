"""
文件: tests/test_train_monitor.py
用途: 测试 train() 的 monitor hook 是否正常工作
运行: python -m tests.test_train_monitor
"""

import csv
from pathlib import Path

from train import train

CALLS = []


def dummy_monitor(agent, global_steps, episode, algorithm, seed):
    """
    一个最简单的监控函数：
    只记录它被调用时的关键信息
    """
    CALLS.append(global_steps)

    return {
        "global_steps": global_steps,
        "episode": episode,
        "algorithm": algorithm,
        "seed": seed,
    }


def test_monitor_callback():
    output_path = Path("results/test_train_monitor.csv")

    if output_path.exists():
        output_path.unlink()

    CALLS.clear()

    # 训练一个很小的例子，只是为了测试 monitor 是否被触发
    train(
        n_episodes=20,
        seed=123,
        algorithm="ddqn",
        monitor_fn=dummy_monitor,
        monitor_interval_steps=100,
        monitor_output_path=output_path,
    )

    # 至少应触发一次
    assert len(CALLS) >= 1

    # 每次触发的位置都应是 100 的倍数
    assert all(step % 100 == 0 for step in CALLS)

    # 输出文件应存在
    assert output_path.exists()

    with output_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == len(CALLS)

    logged_steps = [int(row["global_steps"]) for row in rows]
    assert logged_steps == CALLS

    print("test_monitor_callback PASSED!")


if __name__ == "__main__":
    test_monitor_callback()
    print("\nAll train-monitor tests PASSED!")
