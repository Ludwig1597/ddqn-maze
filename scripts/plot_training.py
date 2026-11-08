"""
文件: scripts/plot_training.py
用途: 绘制五种子 DDQN 训练回报曲线
运行: python -m scripts.plot_training
"""

import csv
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

SEEDS = range(5)
WINDOW = 50


def load_training_returns(seed):
    path = Path(f"logs/ddqn_seed{seed}.csv")

    with open(path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    returns = []
    for row in rows:
        ep_return = float(row["return"])
        returns.append(ep_return)
    return np.array(returns, dtype=np.float64)


# 这个函数叫做 移动平均（Moving Average），作用是：对 DDQN 训练 Return 曲线进行平滑，减少随机波动，可以更容易看出模型是否在逐渐学习。
def moving_average(values, window=50):
    # 使用卷积实现移动平均
    # np.ones(3) 是 NumPy 里创建数组的函数，意思是：生成一个包含 3 个元素、每个元素都是 1 的一维数组。
    # 这个 kernel 叫做卷积核（Convolution Kernel）
    kernel = np.ones(window) / window
    # convolve 是 Convolution（卷积）的意思：让一个小窗口沿着数据序列滑动，每次计算一个结果。
    # mode="valid"：只保留完整窗口的计算结果
    return np.convolve(values, kernel, mode="valid")


def plot_training_return():
    curves = []
    plt.figure(figsize=(11, 6))
    for seed in SEEDS:
        # TODO 2:
        # 调用 load_training_returns(seed)
        # 得到 returns
        returns = load_training_returns(seed)

        # TODO 3:
        # 调用 moving_average(returns, WINDOW)
        # 得到平滑曲线
        smooth_returns = moving_average(returns, WINDOW)

        # TODO 4:
        # 把每个 seed 的平滑曲线
        # 添加到 curves 列表
        curves.append(smooth_returns)

        # TODO 5:
        # 使用 plt.plot() 绘制每个 Seed 的曲线
        # len(smooth_returns)=2000-50+1
        # np.arange(start,end)会生成从start开始，不包含end的连续数值
        # np.arrage(50,2001)=[50,51,52,...,2000]
        # 因为移动平均窗口是 50。第一个平滑值：Episode 1～50 的平均 Return，对应横坐标 50。
        episodes = np.arange(WINDOW, WINDOW + len(smooth_returns))
        (line,) = plt.plot(
            episodes, smooth_returns, label=f"Seed {seed}", linewidth=1.2, alpha=0.7
        )
        plt.plot(
            np.arange(1, len(returns) + 1),
            returns,
            color=line.get_color(),
            linewidth=0.5,
            alpha=0.06,
        )
        # plt.plot(episodes, smooth_returns, label=f"Seed{seed}")

    # ===== Step 6: 五种子统计 =====
    # TODO 6:
    # 将 curves 转换成 NumPy 数组
    # 目标形状：(5, 1951)
    curves_array = np.array(curves)

    # TODO 7:
    # 计算五种子平均 Return
    # 提示：np.mean(..., axis=0)
    # axis=0：沿第 0 维计算，即对不同行的同一列数据求平均，得到每一列的均值。
    mean_returns = np.mean(curves_array, axis=0)

    # TODO 8:
    # 计算每个 Episode 的样本标准差
    # 提示：np.std(..., axis=0, ddof=1)
    std_returns = np.std(curves_array, axis=0, ddof=1)

    # ===== Step 7: 计算 95% 置信带 =====

    n_seeds = len(SEEDS)

    # 五种子、df=4 的双侧 t 临界值
    t_critical = 2.776

    # TODO 9:
    # 标准误：std / sqrt(n)
    standard_error = std_returns / np.sqrt(n_seeds)

    # TODO 10:
    # 置信区间上下界
    ci_lower = mean_returns - t_critical * standard_error
    ci_upper = mean_returns + t_critical * standard_error

    # ===== Step 8: 绘图 =====

    episodes = np.arange(WINDOW, WINDOW + len(mean_returns))

    # TODO 11:
    # 使用 plt.plot() 绘制 mean_returns
    # 建议黑色、linewidth=2.5
    # label="Mean Return"
    plt.plot(episodes, mean_returns, color="black", linewidth=2.5, label="Mean Return")

    # TODO 12:
    # 使用 plt.fill_between()
    # 绘制 ci_lower 到 ci_upper 之间的阴影
    # 建议 alpha=0.2
    # label="95% CI"
    plt.fill_between(
        episodes, ci_lower, ci_upper, color="gray", alpha=0.3, label="95% CI"
    )

    plt.xlabel("Episode")
    plt.ylabel("Episode Return")
    plt.title("DDQN Training Return - 5 Seeds")

    plt.legend()  # 显示图例
    plt.grid(alpha=0.3)  # 显示网格线，alpha是格线的透明度

    Path("figures").mkdir(exist_ok=True)
    # dpi=200：控制图片的分辨率，影响图片清晰度和像素尺寸。
    # bbox_inches="tight"：自动裁剪图片四周多余的空白，同时尽量保留标题、坐标轴标签等内容。
    plt.savefig("figures/ddqn_training_return.png", dpi=200, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    plot_training_return()
