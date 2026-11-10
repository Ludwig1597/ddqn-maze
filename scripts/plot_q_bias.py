"""
文件: scripts/plot_q_bias.py
用途: 绘制 DQN vs DDQN 最佳模型的 Q 值偏差
运行: python -m scripts.plot_q_bias
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

INPUT_PATH = Path("results/q_bias_final_summary.csv")
OUTPUT_PATH = Path("figures/dqn_vs_ddqn_q_bias_final.png")

ALGORITHMS = ("dqn", "ddqn")
SEEDS = list(range(5))


def load_bias_summary(path=INPUT_PATH):
    """
    读取 10 个模型的最终 Q 值偏差。

    返回：
    {
        "dqn": np.ndarray(shape=(5,)),
        "ddqn": np.ndarray(shape=(5,))
    }
    """
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 10:
        raise ValueError(f"Expected 10 model records, got {len(rows)}")

    data = {}

    for algorithm in ALGORITHMS:
        selected = [row for row in rows if row["algorithm"] == algorithm]

        selected.sort(key=lambda r: int(r["seed"]))

        seeds = [int(row["seed"]) for row in selected]

        if seeds != SEEDS:
            raise ValueError(f"Invalid seeds for {algorithm}: {seeds}")

        for row in selected:
            if int(row["n_states"]) != 200:
                raise ValueError("Each model must use 200 states")

        deltas = np.array(
            [float(row["mean_delta"]) for row in selected],
            dtype=np.float64,
        )

        if not np.isfinite(deltas).all():
            raise ValueError("Non-finite bias values")

        data[algorithm] = deltas

    return data


def plot_final_q_bias(data, output_path=OUTPUT_PATH):
    """
    绘制五个 Seed 的 DQN / DDQN 平均有符号 Δ。
    """
    dqn = data["dqn"]
    ddqn = data["ddqn"]

    if dqn.shape != (5,) or ddqn.shape != (5,):
        raise ValueError("Expected five seeds per algorithm")

    x = np.arange(5)
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))

    dqn_mean = np.mean(dqn)
    ddqn_mean = np.mean(ddqn)

    dqn_std = np.std(dqn, ddof=1)
    ddqn_std = np.std(ddqn, ddof=1)

    bars_dqn = ax.bar(
        x - width / 2,
        dqn,
        width=width,
        label=f"DQN (mean={dqn_mean:+.3f})",
    )

    bars_ddqn = ax.bar(
        x + width / 2,
        ddqn,
        width=width,
        label=f"DDQN (mean={ddqn_mean:+.3f})",
    )

    # Δ=0 是预测与 MC 回报相等的参考线
    ax.axhline(0.0, color="black", linewidth=1)

    # 在每根柱子上标注正负号和数值
    ax.bar_label(bars_dqn, fmt="%+.3f", padding=3, fontsize=9)
    ax.bar_label(bars_ddqn, fmt="%+.3f", padding=3, fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Seed {seed}" for seed in SEEDS])

    ax.set_ylabel("Q-value Bias (Max Q - MC Return)")
    ax.set_xlabel("Training Seed")

    ax.set_title("DQN vs DDQN - Final Checkpoint Q-value Bias")

    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.margins(y=0.2)
    ax.legend()

    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    return {
        "dqn_mean": float(dqn_mean),
        "dqn_std": float(dqn_std),
        "ddqn_mean": float(ddqn_mean),
        "ddqn_std": float(ddqn_std),
    }


def main():
    data = load_bias_summary()
    stats = plot_final_q_bias(data)

    print("===== Final Q Bias Comparison =====")
    print(f"DQN:  {stats['dqn_mean']:+.4f} " f"± {stats['dqn_std']:.4f}")
    print(f"DDQN: {stats['ddqn_mean']:+.4f} " f"± {stats['ddqn_std']:.4f}")

    difference = stats["dqn_mean"] - stats["ddqn_mean"]

    print(f"DQN - DDQN: {difference:+.4f}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
