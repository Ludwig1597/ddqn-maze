"""
文件: scripts/summarize_multi_seed.py
用途: 汇总五个随机种子的训练统计
运行: python -m scripts.summarize_multi_seed
"""

import csv
from pathlib import Path


def summarize_seed(seed):
    path = Path(f"logs/ddqn_seed{seed}.csv")
    # Step 1: 读取 CSV
    with open(path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Step 2: 统计 Episode 数量
    episodes = len(rows)

    # TODO 1:
    # 统计全部 Episode 的 steps 总和
    # 提示：int(row["steps"])
    environment_steps = 0

    # TODO 2:
    # 统计全部 Episode 的 success 总和
    successful_episodes = 0

    for row in rows:
        environment_steps += int(row["steps"])
        successful_episodes += int(row["success"])

    # TODO 3:
    # 计算训练成功比例
    # successful_episodes / episodes
    success_rate = successful_episodes / episodes if episodes > 0 else 0.0

    print(f"\n===== Seed {seed} Summary =====")
    print("Episodes:", episodes)
    print("Environment steps:", environment_steps)
    print("Successful episodes:", successful_episodes)
    print("Training success rate:", success_rate)

    return {
        "seed": seed,
        "episodes": episodes,
        "environment_steps": environment_steps,
        "successful_episodes": successful_episodes,
        "success_rate": success_rate,
    }


if __name__ == "__main__":

    for seed in range(5):
        summarize_seed(seed)
