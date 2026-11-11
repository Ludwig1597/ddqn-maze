"""
文件: scripts/plot_delta_training.py
用途: 绘制五种子 DQN/DDQN 训练过程 Delta 曲线
运行: python -m scripts.plot_delta_training
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_one(algorithm, seed):
    path = Path("results") / (f"{algorithm}_seed{seed}_delta_curve.csv")

    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    return {int(r["global_steps"]): float(r["mean_delta"]) for r in rows}


def main():
    datasets = {}

    for algorithm in ["dqn", "ddqn"]:
        datasets[algorithm] = [load_one(algorithm, seed) for seed in range(5)]

    # 只在所有10条曲线都有数据的步数上比较
    common_steps = set.intersection(
        *(set(d.keys()) for group in datasets.values() for d in group)
    )
    steps = sorted(common_steps)

    if len(steps) < 2:
        raise ValueError("Not enough common training checkpoints")

    plt.figure(figsize=(10, 6))

    colors = {"dqn": "tab:orange", "ddqn": "tab:blue"}

    for algorithm in ["dqn", "ddqn"]:
        matrix = np.array([[d[step] for step in steps] for d in datasets[algorithm]])

        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0, ddof=1)

        # n=5时使用 t 分布的95%置信区间
        ci95 = 2.776 * std / np.sqrt(5)

        plt.plot(
            steps,
            mean,
            label=algorithm.upper(),
            color=colors[algorithm],
        )

        plt.fill_between(
            steps,
            mean - ci95,
            mean + ci95,
            color=colors[algorithm],
            alpha=0.2,
        )

    plt.axhline(0, color="gray", linestyle="--")
    plt.xlabel("Global Training Steps")
    plt.ylabel("Q-value Bias (Delta)")
    plt.title("DQN vs DDQN: Q-value Bias During Training")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output = Path("figures/delta_training.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close()

    print(f"Saved: {output}")
    print(f"Common checkpoints: {len(steps)}")


if __name__ == "__main__":
    main()
