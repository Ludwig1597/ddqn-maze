"""
文件: scripts/compare_algorithms.py

用途:
    DQN vs DDQN 五种子独立评估对照。

左图:
    五个 Seed 的独立成功率。

右图:
    五种子平均成功率 + 95% CI。

输入:
    results/dqn_final_eval.csv
    results/ddqn_final_eval.csv

输出:
    figures/dqn_vs_ddqn_final_eval.png

运行:
    python -m scripts.compare_algorithms
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SEEDS = range(5)
T_CRITICAL = 2.776


# ==================================================
# 1. 加载最终评估数据
# ==================================================


def load_success_rates(algorithm):

    path = Path(f"results/{algorithm}_final_eval.csv")

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        encoding="utf-8",
        newline="",
    ) as f:

        rows = list(csv.DictReader(f))

    required = {"seed", "success_rate"}

    if not required.issubset(rows[0].keys()):
        raise ValueError(f"Invalid columns in {path}: {rows[0].keys()}")

    values = {}

    for row in rows:

        seed = int(row["seed"])

        if seed in values:
            raise ValueError(f"Duplicate seed {seed}: {path}")

        values[seed] = float(row["success_rate"])

    if set(values) != set(SEEDS):
        raise ValueError(f"Expected seeds {list(SEEDS)}, " f"got {sorted(values)}")

    rates = np.array(
        [values[seed] for seed in SEEDS],
        dtype=np.float64,
    )

    if not np.all(np.isfinite(rates)):
        raise ValueError(f"Invalid values in {path}")

    if not np.all((rates >= 0) & (rates <= 1)):
        raise ValueError(f"Success rate outside [0, 1]: {path}")

    return rates


# ==================================================
# 2. 计算五种子统计
# ==================================================


def calculate_statistics(values):

    mean = np.mean(values)

    std = np.std(values, ddof=1)

    standard_error = std / np.sqrt(5)

    margin = T_CRITICAL * standard_error

    lower = max(0.0, mean - margin)
    upper = min(1.0, mean + margin)

    return mean, std, lower, upper


# ==================================================
# 3. 绘制 A/B 对照图
# ==================================================


def compare_algorithms():

    dqn = load_success_rates("dqn")
    ddqn = load_success_rates("ddqn")

    dqn_stats = calculate_statistics(dqn)
    ddqn_stats = calculate_statistics(ddqn)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5.5),
    )

    ax1, ax2 = axes

    # ----------------------------------------------
    # 左图：五个 Seed 的独立结果
    # ----------------------------------------------

    x = np.arange(5)

    width = 0.35

    ax1.bar(
        x - width / 2,
        dqn,
        width=width,
        label="DQN",
    )

    ax1.bar(
        x + width / 2,
        ddqn,
        width=width,
        label="DDQN",
    )

    ax1.set_xticks(
        x,
        [f"Seed {i}" for i in SEEDS],
    )

    ax1.set_ylim(0, 1)

    ax1.set_ylabel("Success Rate")

    ax1.set_title("Per-Seed Evaluation")

    ax1.legend()

    ax1.grid(
        axis="y",
        alpha=0.3,
    )

    # ----------------------------------------------
    # 右图：五种子均值和95% CI
    # ----------------------------------------------

    means = [
        dqn_stats[0],
        ddqn_stats[0],
    ]

    lower = [
        dqn_stats[2],
        ddqn_stats[2],
    ]

    upper = [
        dqn_stats[3],
        ddqn_stats[3],
    ]

    yerr = np.array(
        [
            np.array(means) - np.array(lower),
            np.array(upper) - np.array(means),
        ]
    )

    x_summary = np.arange(2)

    bars = ax2.bar(
        x_summary,
        means,
        width=0.5,
        yerr=yerr,
        capsize=7,
        label="Mean with 95% CI",
    )

    ax2.set_xticks(
        x_summary,
        ["DQN", "DDQN"],
    )

    ax2.set_ylim(0, 1)

    ax2.set_ylabel("Success Rate")

    ax2.set_title("Five-Seed Mean and 95% CI")

    ax2.grid(
        axis="y",
        alpha=0.3,
    )

    for bar in bars:

        height = bar.get_height()

        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.025,
            f"{height:.3f}",
            ha="center",
            fontsize=10,
        )

    fig.suptitle(
        "DQN vs DDQN - Independent Evaluation",
        fontsize=14,
    )

    fig.tight_layout()

    Path("figures").mkdir(exist_ok=True)

    output = Path("figures/dqn_vs_ddqn_final_eval.png")

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ----------------------------------------------
    # 打印统计结果
    # ----------------------------------------------

    print("\n===== DQN vs DDQN Statistics =====")

    for name, stats in [
        ("DQN", dqn_stats),
        ("DDQN", ddqn_stats),
    ]:

        mean, std, lower, upper = stats

        print(
            f"{name}: "
            f"mean={mean:.4f}, "
            f"std={std:.4f}, "
            f"95% CI=[{lower:.4f}, {upper:.4f}]"
        )

    print(f"\nFigure saved: {output}")


if __name__ == "__main__":
    compare_algorithms()
