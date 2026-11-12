"""
文件: scripts/plot_training.py

用途:
    绘制五种子 DDQN Return 和 Success Rate 曲线。

图1:
    原始 Return + 50 Episode MA + 95% CI

图2:
    Training Success: 原始值 + 50 Episode MA + 95% CI
    Evaluation Success: 每50 Episode独立评估 + 95% CI

运行:
    python -m scripts.plot_training
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SEEDS = range(5)
WINDOW = 50
T_CRITICAL = 2.776


# ==================================================
# 1. 加载训练数据
# ==================================================


def load_training_data(seed):

    path = Path(f"logs/ddqn_seed{seed}.csv")

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    required = {"episode", "return", "success"}

    if not required.issubset(rows[0].keys()):
        raise ValueError(f"Missing columns in {path}")

    episodes = np.array(
        [int(row["episode"]) for row in rows],
        dtype=int,
    )

    returns = np.array(
        [float(row["return"]) for row in rows],
        dtype=np.float64,
    )

    successes = np.array(
        [float(row["success"]) for row in rows],
        dtype=np.float64,
    )

    if not np.array_equal(
        episodes,
        np.arange(1, len(rows) + 1),
    ):
        raise ValueError(f"Invalid episode sequence: {path}")

    if not np.all(np.isin(successes, [0, 1])):
        raise ValueError(f"Invalid success values: {path}")

    if not np.all(np.isfinite(returns)):
        raise ValueError(f"Invalid return values: {path}")

    return episodes, returns, successes


# ==================================================
# 2. 加载独立评估数据
# ==================================================


def load_eval_success_rate(seed):

    path = Path(f"logs/ddqn_seed{seed}_eval.csv")

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    episodes = np.array(
        [int(row["episode"]) for row in rows],
        dtype=int,
    )

    success_rates = np.array(
        [float(row["success_rate"]) for row in rows],
        dtype=np.float64,
    )

    if not np.all(np.isfinite(success_rates)):
        raise ValueError(f"Invalid evaluation data: {path}")

    if not np.all((success_rates >= 0) & (success_rates <= 1)):
        raise ValueError(f"Invalid success rate: {path}")

    return episodes, success_rates


# ==================================================
# 3. 移动平均
# ==================================================


def moving_average(values, window=WINDOW):

    if len(values) < window:
        raise ValueError("Not enough values for moving average")

    kernel = np.ones(window) / window

    return np.convolve(
        values,
        kernel,
        mode="valid",
    )


# ==================================================
# 4. 五种子统计
# ==================================================


def calculate_statistics(curves):

    matrix = np.asarray(curves, dtype=np.float64)

    if matrix.shape[0] != 5:
        raise ValueError("Expected 5 seeds")

    mean = np.mean(matrix, axis=0)

    std = np.std(
        matrix,
        axis=0,
        ddof=1,
    )

    standard_error = std / np.sqrt(5)

    margin = T_CRITICAL * standard_error

    lower = mean - margin
    upper = mean + margin

    return mean, lower, upper


# ==================================================
# 5. Return Curve
# ==================================================


def plot_training_return():

    raw_curves = []
    smooth_curves = []
    reference_episodes = None

    for seed in SEEDS:

        episodes, returns, _ = load_training_data(seed)

        if reference_episodes is None:
            reference_episodes = episodes
        elif not np.array_equal(episodes, reference_episodes):
            raise ValueError("Training episodes are not aligned")

        smooth_curves.append(moving_average(returns))

        raw_curves.append(returns)

    smooth_matrix = np.asarray(smooth_curves)

    mean, lower, upper = calculate_statistics(smooth_matrix)

    smooth_episodes = reference_episodes[WINDOW - 1 :]

    fig, ax = plt.subplots(figsize=(11, 6))

    # 原始 Return
    for i, returns in enumerate(raw_curves):

        ax.plot(
            reference_episodes,
            returns,
            color="tab:blue",
            linewidth=0.5,
            alpha=0.07,
            label="Raw Episode Return" if i == 0 else None,
        )

    # 每个 Seed 的移动平均
    for smooth in smooth_curves:

        ax.plot(
            smooth_episodes,
            smooth,
            color="tab:blue",
            linewidth=0.8,
            alpha=0.25,
        )

    # 五种子平均移动平均
    ax.plot(
        smooth_episodes,
        mean,
        color="black",
        linewidth=2.2,
        label="Mean 50-Episode MA",
    )

    # 95% CI
    ax.fill_between(
        smooth_episodes,
        lower,
        upper,
        color="gray",
        alpha=0.25,
        label="95% CI",
    )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Return")
    ax.set_title("DDQN Training Return - 5 Seeds")

    ax.grid(alpha=0.3)
    ax.legend()

    Path("figures").mkdir(exist_ok=True)

    output = "figures/ddqn_training_return.png"

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Saved:", output)


# ==================================================
# 6. Training + Evaluation Success Rate
# ==================================================


def plot_success_rate():

    train_raw = []
    train_smooth = []

    eval_curves = []

    train_episodes_ref = None
    eval_episodes_ref = None

    for seed in SEEDS:

        episodes, _, successes = load_training_data(seed)

        if train_episodes_ref is None:
            train_episodes_ref = episodes
        elif not np.array_equal(episodes, train_episodes_ref):
            raise ValueError("Training episodes are not aligned")

        train_raw.append(successes)

        train_smooth.append(moving_average(successes))

        eval_episodes, eval_success = load_eval_success_rate(seed)

        if eval_episodes_ref is None:
            eval_episodes_ref = eval_episodes
        elif not np.array_equal(
            eval_episodes,
            eval_episodes_ref,
        ):
            raise ValueError("Evaluation episodes are not aligned")

        eval_curves.append(eval_success)

    # Training Success Statistics
    train_matrix = np.asarray(train_smooth)

    train_mean, train_lower, train_upper = calculate_statistics(train_matrix)

    smooth_episodes = train_episodes_ref[WINDOW - 1 :]

    # Evaluation Success Statistics
    eval_mean, eval_lower, eval_upper = calculate_statistics(eval_curves)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 9),
    )

    ax1, ax2 = axes

    # ----------------------------------------------
    # Top: Training Success
    # ----------------------------------------------

    for i, raw in enumerate(train_raw):

        ax1.plot(
            train_episodes_ref,
            raw,
            color="tab:blue",
            linewidth=0.5,
            alpha=0.06,
            label="Raw Training Success" if i == 0 else None,
        )

    for smooth in train_smooth:

        ax1.plot(
            smooth_episodes,
            smooth,
            color="tab:blue",
            alpha=0.2,
            linewidth=0.8,
        )

    ax1.plot(
        smooth_episodes,
        train_mean,
        color="black",
        linewidth=2.2,
        label="Mean 50-Episode MA",
    )

    ax1.fill_between(
        smooth_episodes,
        np.clip(train_lower, 0, 1),
        np.clip(train_upper, 0, 1),
        color="gray",
        alpha=0.25,
        label="95% CI",
    )

    ax1.set_title("Training Success Rate")
    ax1.set_ylabel("Success Rate")
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.3)
    ax1.legend()

    # ----------------------------------------------
    # Bottom: Independent Evaluation Success
    # ----------------------------------------------

    for i, curve in enumerate(eval_curves):

        ax2.plot(
            eval_episodes_ref,
            curve,
            color="tab:blue",
            marker="o",
            markersize=2.5,
            linewidth=0.8,
            alpha=0.3,
            label="Seed Evaluation" if i == 0 else None,
        )

    ax2.plot(
        eval_episodes_ref,
        eval_mean,
        color="black",
        linewidth=2.2,
        marker="o",
        markersize=3,
        label="Mean Evaluation Success",
    )

    ax2.fill_between(
        eval_episodes_ref,
        np.clip(eval_lower, 0, 1),
        np.clip(eval_upper, 0, 1),
        color="gray",
        alpha=0.25,
        label="95% CI",
    )

    ax2.set_title(
        "Independent Evaluation Success Rate "
        "(epsilon = 0, 10 episodes per checkpoint)"
    )

    ax2.set_xlabel("Training Episode")
    ax2.set_ylabel("Evaluation Success Rate")
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.suptitle(
        "DDQN Training and Evaluation Success Rate - 5 Seeds",
        fontsize=14,
    )

    fig.tight_layout()

    Path("figures").mkdir(exist_ok=True)

    output = "figures/ddqn_success_rate_curve.png"

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Saved:", output)


# ==================================================
# 7. Main
# ==================================================


def main():
    plot_training_return()
    plot_success_rate()
    print("\nTraining figures completed.")


if __name__ == "__main__":
    main()
