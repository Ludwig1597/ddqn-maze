"""
文件: scripts/compare_ablations.py

用途:
    对比 Baseline DDQN、No Target Network、
    Small Replay Buffer 三组消融实验。

输入:
    logs/ddqn_seed{0-4}_eval.csv
    logs/ddqn_seed{0-4}_ablation_no_target_eval.csv
    logs/ddqn_seed{0-4}_ablation_small_buffer_eval.csv

输出:
    figures/ablation_comparison.png
    results/ablation_summary.csv

运行:
    python -m scripts.compare_ablations
"""

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ==================================================
# 1. 实验配置
# ==================================================

SEEDS = range(5)

# n=5，自由度=4，双侧95%置信区间的t临界值
T_CRITICAL = 2.776

GROUPS = {
    "Baseline DDQN": "",
    "No Target Network": "_ablation_no_target",
    "Small Replay Buffer": "_ablation_small_buffer",
}

METRICS = ["success_rate", "avg_return", "avg_steps"]

EXPECTED_EPISODES = list(range(50, 2001, 50))


# ==================================================
# 2. 加载单个Seed的数据
# ==================================================


def load_one(group_tag, seed):
    """
    读取一个Seed的评估CSV。

    返回:
        {
            episode: {
                success_rate: float,
                avg_return: float,
                avg_steps: float
            }
        }
    """

    filename = f"ddqn_seed{seed}{group_tag}_eval.csv"
    path = Path("logs") / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing experiment file: {path}")

    data = {}

    with path.open(encoding="utf-8", newline="") as f:

        reader = csv.DictReader(f)

        required = {
            "episode",
            "success_rate",
            "avg_return",
            "avg_steps",
        }

        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Invalid CSV columns in {path}: " f"{reader.fieldnames}")

        for row in reader:
            episode = int(row["episode"])

            if episode in data:
                raise ValueError(f"Duplicate episode {episode} in {path}")

            values = {metric: float(row[metric]) for metric in METRICS}

            if not all(math.isfinite(v) for v in values.values()):
                raise ValueError(f"Non-finite metric in {path}, " f"episode={episode}")

            data[episode] = values

    actual = sorted(data.keys())

    if actual != EXPECTED_EPISODES:
        raise ValueError(
            f"Unexpected episodes in {path}\n"
            f"Expected: {EXPECTED_EPISODES}\n"
            f"Actual: {actual}"
        )

    return data


# ==================================================
# 3. 读取全部三组、五个Seed
# ==================================================


def load_all():
    datasets = {}

    for group_name, group_tag in GROUPS.items():

        print(f"\nLoading: {group_name}")

        datasets[group_name] = []

        for seed in SEEDS:

            data = load_one(group_tag, seed)

            datasets[group_name].append(data)

            print(f"  Seed {seed}: " f"{len(data)} evaluation records")

    print("\nAll 15 CSV files validated.")

    return datasets


# ==================================================
# 4. 计算五个Seed的均值和95%置信区间
# ==================================================


def calculate_statistics(datasets):

    statistics = {}

    for group_name, seed_data in datasets.items():

        statistics[group_name] = {}

        for metric in METRICS:

            means = []
            stds = []
            lower = []
            upper = []

            for episode in EXPECTED_EPISODES:

                values = np.array(
                    [data[episode][metric] for data in seed_data], dtype=np.float64
                )

                mean = float(np.mean(values))
                std = float(np.std(values, ddof=1))

                # 95% t置信区间
                margin = T_CRITICAL * std / np.sqrt(len(values))

                means.append(mean)
                stds.append(std)
                lower.append(mean - margin)
                upper.append(mean + margin)

            statistics[group_name][metric] = {
                "mean": np.array(means),
                "std": np.array(stds),
                "lower": np.array(lower),
                "upper": np.array(upper),
            }

    return statistics


# ==================================================
# 5. 绘制消融对比图
# ==================================================


def plot_comparison(statistics):

    Path("figures").mkdir(exist_ok=True)

    episodes = np.array(EXPECTED_EPISODES)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    plot_config = [
        (
            axes[0],
            "success_rate",
            "Success Rate",
            "Success Rate",
        ),
        (
            axes[1],
            "avg_return",
            "Average Return",
            "Average Return",
        ),
    ]

    for ax, metric, title, ylabel in plot_config:

        for group_name, result in statistics.items():

            stats = result[metric]

            mean = stats["mean"]
            lower = stats["lower"]
            upper = stats["upper"]

            # 成功率置信区间限制在合法范围 [0, 1]
            if metric == "success_rate":
                lower = np.clip(lower, 0, 1)
                upper = np.clip(upper, 0, 1)

            # 五Seed均值曲线
            ax.plot(
                episodes,
                mean,
                label=group_name,
                linewidth=1.8,
            )

            # 95%置信区间
            ax.fill_between(
                episodes,
                lower,
                upper,
                alpha=0.15,
            )

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="both", alpha=0.3)
        ax.legend()

    axes[0].set_ylim(0, 1)

    axes[1].set_xlabel("Training Episode")

    fig.suptitle(
        "DDQN Ablation Study",
        fontsize=15,
    )

    fig.tight_layout()

    output = Path("figures/ablation_comparison.png")

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"\nFigure saved: {output}")


# ==================================================
# 6. 导出最终统计结果
# ==================================================


def export_summary(statistics):

    Path("results").mkdir(exist_ok=True)

    output = Path("results/ablation_summary.csv")

    with output.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "algorithm",
                "metric",
                "final_episode",
                "mean",
                "std",
                "ci95_lower",
                "ci95_upper",
            ]
        )

        for group_name, result in statistics.items():

            for metric in METRICS:

                stats = result[metric]

                writer.writerow(
                    [
                        group_name,
                        metric,
                        EXPECTED_EPISODES[-1],
                        stats["mean"][-1],
                        stats["std"][-1],
                        stats["lower"][-1],
                        stats["upper"][-1],
                    ]
                )

    print(f"Summary saved: {output}")


# ==================================================
# 7. 打印最后一次评估的对比结果
# ==================================================


def print_final_results(statistics):

    print("\n" + "=" * 75)
    print("FINAL ABLATION RESULTS | Episode 2000")
    print("=" * 75)

    print(f"{'Algorithm':<24}" f"{'Success':>12}" f"{'Return':>12}" f"{'Steps':>12}")

    for group_name, result in statistics.items():

        success = result["success_rate"]
        returns = result["avg_return"]
        steps = result["avg_steps"]

        print(
            f"{group_name:<24}"
            f"{success['mean'][-1]:>12.3f}"
            f"{returns['mean'][-1]:>12.3f}"
            f"{steps['mean'][-1]:>12.2f}"
        )

    print("=" * 75)


# ==================================================
# 8. 主程序
# ==================================================


def main():

    datasets = load_all()

    statistics = calculate_statistics(datasets)

    plot_comparison(statistics)

    export_summary(statistics)

    print_final_results(statistics)

    print("\nAblation comparison completed.")


if __name__ == "__main__":
    main()
