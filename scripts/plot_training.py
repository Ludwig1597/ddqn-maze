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
        plt.plot(episodes, smooth_returns, label=f"Seed{seed}")

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
