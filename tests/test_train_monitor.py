"""
文件: tests/test_train_monitor.py
用途: 测试 train() 的 monitor hook 是否正常工作
运行: python -m pytest tests/test_train_monitor.py -v
"""

import csv
import os
import tempfile
from pathlib import Path

from train import train

CALLS = []


def dummy_monitor(agent, global_steps, episode, algorithm, seed):
    """
    测试用监控函数：
    记录每次被调用时的全局步数。
    """
    CALLS.append(global_steps)

    return {
        "global_steps": global_steps,
        "episode": episode,
        "algorithm": algorithm,
        "seed": seed,
    }


def test_monitor_callback():

    # 保存原来的工作目录
    original_cwd = os.getcwd()

    # 创建临时目录
    with tempfile.TemporaryDirectory() as tmp:

        try:
            # 切换到临时目录
            os.chdir(tmp)

            # 监控日志输出到临时目录
            output_path = Path("results/test_train_monitor.csv")

            CALLS.clear()

            # 运行一个小规模训练
            train(
                n_episodes=20,
                seed=123,
                algorithm="ddqn",
                monitor_fn=dummy_monitor,
                monitor_interval_steps=100,
                monitor_output_path=output_path,
            )

            # 验收 1：监控函数至少被调用一次
            assert len(CALLS) >= 1

            # 验收 2：每次调用都发生在100步的整数倍
            assert all(step % 100 == 0 for step in CALLS)

            # 验收 3：CSV文件成功生成
            assert output_path.exists()

            # 验收 4：CSV记录数量正确
            with output_path.open(
                "r",
                newline="",
                encoding="utf-8",
            ) as f:
                rows = list(csv.DictReader(f))

            assert len(rows) == len(CALLS)

            # 验收 5：CSV中的步数与回调记录一致
            logged_steps = [int(row["global_steps"]) for row in rows]

            assert logged_steps == CALLS

            print("test_monitor_callback PASSED!")

        finally:
            # 无论测试成功还是失败，都恢复原来的工作目录
            os.chdir(original_cwd)


if __name__ == "__main__":
    test_monitor_callback()
    print("\nAll train-monitor tests PASSED!")
