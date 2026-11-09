"""
文件: scripts/compare_algorithms.py
用途: DQN vs DDQN 独立评估对照图
运行: python -m scripts.compare_algorithms
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_success_rates(algorithm):
    path = Path(f"results/{algorithm}_final_eval.csv")

    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # TODO 1:
    # 确认 CSV 包含 Seed 0～4，且顺序正确
    seeds = [int(row["seed"]) for row in rows]
    assert seeds == list(range(5))

    # TODO 2:
    # 读取 success_rate，转换为 NumPy 数组
    success_rates = np.array([float(row["success_rate"]) for row in rows], dtype=float)

    return success_rates


def compare_algorithms():
    dqn = load_success_rates("dqn")
    ddqn = load_success_rates("ddqn")

    # TODO 3:
    # 使用两组 plt.bar() 绘制分组柱状图
    # x = np.arange(5)
    # width = 0.35
    x = np.arange(5)
    width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, dqn, width=width, label="DQN")
    plt.bar(x + width / 2, ddqn, width=width, label="DDQN")

    # TODO 4:
    # 添加标题、坐标轴、图例
    # 保存图片
    plt.xticks(
        x, [f"Seed {i}" for i in range(5)]
    )  # xticks = X-axis Ticks，意思是 X 轴刻度。
    plt.ylim(0, 1)  # ylim = Y-axis Limits，意思是 Y 轴显示范围。
    plt.legend()
    plt.grid(
        axis="y", alpha=0.3
    )  # 这里的 axis="y" 表示只显示 Y 轴刻度对应的网格线，也就是水平网格线。
    Path("figures").mkdir(exist_ok=True)
    plt.savefig("figures/dqn_vs_ddqn_final_eval.png", dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    compare_algorithms()
