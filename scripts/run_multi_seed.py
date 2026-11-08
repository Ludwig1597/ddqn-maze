"""
文件: scripts/run_multi_seed.py
用途: 统一执行 DDQN 多随机种子训练
运行: python -m scripts.run_multi_seed
"""

from pathlib import Path
from train import train


def run_seed(seed, n_episodes=2000):

    # Step 1: 构造该 Seed 的输出路径
    train_log = Path(f"logs/ddqn_seed{seed}.csv")
    eval_log = Path(f"logs/ddqn_seed{seed}_eval.csv")
    checkpoint = Path(f"checkpoints/ddqn_seed{seed}_best.pt")
    paths = [train_log, eval_log, checkpoint]
    # Step 2: 避免覆盖已有实验
    # TODO 1:
    # 检查上述三个文件是否存在
    # 如果任意文件存在，打印提示，并跳过训练
    #
    # 提示：
    # path.exists()
    # any(...)
    if any(path.exists() for path in paths):
        print(f"Seed {seed}: existing files found, skip training.")
        return

    # Step 3: 开始训练
    print(f"\n===== Training Seed {seed} =====")

    # TODO 2: 调用训练函数
    # 调用 train()
    # 传入 n_episodes 和 seed
    train(n_episodes=n_episodes, seed=seed)

    # Step 4: 检查文件
    # TODO 3:
    # 判断 train_log、eval_log、checkpoint 是否全部生成
    # 如果全部存在，打印完成信息
    if all(path.exists() for path in paths):
        print(f"Seed {seed}: training completed!")
    else:
        print(f"Seed {seed}: some output files are missing.")


if __name__ == "__main__":
    seeds = [0, 1, 2, 3, 4]

    for seed in seeds:
        run_seed(seed, n_episodes=2000)
