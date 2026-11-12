"""
文件: train.py
用途: 训练 DQN / DDQN, 并支持周期性监控回调
运行: python -m train
"""

import random
import numpy as np
import torch
import csv

# 从 Python 标准库的 pathlib 模块中，导入 Path 类，用来操作文件和文件夹路径。
from pathlib import Path

from envs.maze import CoinMaze
from agents.ddqn import DDQNAgent
from agents.dqn import DQNAgent
from evaluate import evaluate


# 探索率衰减函数
def epsilon_schedule(step):
    """
    epsilon 从 1.0 线性衰减到 0.05
    总衰减步数 20000
    """
    return max(0.05, 1 - 0.95 * step / 20000)


def train(
    n_episodes=10,
    seed=0,
    algorithm="ddqn",
    monitor_fn=None,
    monitor_interval_steps=None,
    monitor_output_path=None,
    run_tag=None,
    target_sync_interval=500,
):
    """
    参数:
    - n_episodes: 训练回合数
    - seed: 随机种子
    - algorithm: "dqn" 或 "ddqn"
    - monitor_fn: 可选监控函数
    - monitor_interval_steps: 每隔多少个环境步调用一次 monitor_fn
    - monitor_output_path: 监控结果保存路径（可选）
    """
    if (
        not isinstance(target_sync_interval, int)
        or isinstance(target_sync_interval, bool)
        or target_sync_interval < 1
    ):
        raise ValueError("target_sync_interval must be a positive integer")

    if algorithm not in {"dqn", "ddqn"}:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    if monitor_fn is not None:
        if monitor_interval_steps is None or monitor_interval_steps <= 0:
            raise ValueError("monitor_interval_steps must be a positive integer")

    # 固定随机种子
    # Python 标准库 random
    # random.randint() 随机选择动作
    random.seed(seed)
    # NumPy 旧式全局随机数生成器
    # np.random.random() 用于 ε-greedy 探索
    np.random.seed(seed)
    # PyTorch 随机数生成器
    # QNet 的权重随机初始化
    torch.manual_seed(seed)
    # 迷宫内部的随机机制
    # 随机生成金币、陷阱的位置
    env = CoinMaze(seed=seed)
    if algorithm == "ddqn":
        agent = DDQNAgent()
    elif algorithm == "dqn":
        agent = DQNAgent()
    else:
        raise ValueError(f"Unsupported algorithm:{algorithm}")

    # 统一实验名称
    run_name = f"{algorithm}_seed{seed}"
    if run_tag:
        run_name += f"_{run_tag}"
    # 保护已有日志和模型，防止误覆盖
    output_paths = [
        Path("logs") / f"{run_name}.csv",
        Path("logs") / f"{run_name}_eval.csv",
        Path("checkpoints") / f"{run_name}_best.pt",
    ]

    if any(path.exists() for path in output_paths):
        raise FileExistsError(
            f"Experiment {run_name} already exists. "
            "Refusing to overwrite existing results."
        )

    global_steps = 0
    update_count = 0
    sync_count = 0
    success_count = 0

    episode_logs = []
    eval_logs = []
    monitor_logs = []

    best_success_rate = -1.0

    checkpoint_path = Path("checkpoints") / f"{run_name}_best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(n_episodes):
        state = env.reset()
        done = False

        ep_return = 0.0
        ep_steps = 0
        ep_losses = []

        while not done:
            eps = epsilon_schedule(global_steps)

            # TODO 2:
            # 使用agent.act()选择动作
            # 会使用 ε-greedy 策略选择动作：
            # - 以 eps 的概率随机探索（Exploration）。
            # - 以 1-eps 的概率选择 QNet 预测 Q 值最大的动作（Exploitation）。
            action = agent.act(state, eps)
            # TODO 3:
            # 使用env.step()执行动作
            # 获取next_state,reward,done,info
            next_state, reward, done, info = env.step(action)

            # TODO 4:
            # 将经验存入agent.buf
            # done使用float(done)
            agent.buf.push(state, action, reward, next_state, float(done))

            # 执行一次学习更新
            loss = agent.update()
            # update()在预热阶段，此时没有执行BP
            # 在经验足够以后，它会返回实际的Huber Loss
            if loss is not None:
                update_count += 1
                ep_losses.append(loss)

            # 增加全局环境步数
            global_steps += 1
            ep_steps += 1

            # 每target_sync_interval个全局环境步同步Target
            if global_steps % target_sync_interval == 0:
                agent.sync_target()
                sync_count += 1

            # ===== 新增：训练监控钩子 =====
            if (
                monitor_fn is not None
                and monitor_interval_steps is not None
                and global_steps % monitor_interval_steps == 0
            ):
                record = monitor_fn(
                    agent=agent,
                    global_steps=global_steps,
                    episode=ep + 1,
                    algorithm=algorithm,
                    seed=seed,
                )

                if record is not None:
                    if not isinstance(record, dict):
                        raise TypeError("monitor_fn must return a dict or None")
                    monitor_logs.append(record)

            state = next_state
            ep_return += reward

        success = int(tuple(env.pos) == env.coin)
        success_count += success

        # 本回合平均Loss
        # len()：计算元素数量。
        avg_loss = sum(ep_losses) / len(ep_losses) if ep_losses else None

        # TODO 1:
        # 向episode_logs添加一个字典
        # 记录下面六个字段
        episode_logs.append(
            {
                "episode": ep + 1,
                "return": ep_return,
                "success": success,
                "steps": ep_steps,
                "epsilon": eps,
                "loss": avg_loss,
            }
        )
        print(
            # 3	最小显示宽度为 3 个字符
            # d	Decimal，表示十进制整数
            f"Episode: {ep+1:3d} |"
            # 将 ep_return 格式化为保留 3 位小数的浮点数，最小显示宽度为 7 个字符，默认右对齐。
            f"Return: {ep_return:7.3f} |"
            f"Steps: {ep_steps:3d} |"
            f"Epsilon: {eps:.3f} |"
            f"Global steps:{global_steps}"
        )

        # 每50Episode评估一次
        if (ep + 1) % 50 == 0:
            # TODO 1:
            # 调用evaluate()
            # 使用当前agent
            # episodes=10，seed=999
            # 评估使用默认epsilon=0.0
            result = evaluate(agent, episodes=10, seed=999)
            success_rate = result["success_rate"]
            # 14 TODO 1:
            # 向 eval_logs 添加一个字典
            # 包含四个字段：
            #
            # episode：ep + 1
            # success_rate：result["success_rate"]
            # avg_return：result["avg_return"]
            # avg_steps：result["avg_steps"]

            eval_logs.append(
                {
                    "episode": ep + 1,
                    "success_rate": success_rate,
                    "avg_return": result["avg_return"],
                    "avg_steps": result["avg_steps"],
                }
            )

            # TODO 2:
            # 打印当前评估成功率
            # 例如：Evaluation at episode 50: success_rate=0.200
            # fmt:off
            print(
                f"Evaluation at episode {ep+1}: " 
                f"success_rate={success_rate:.3f}"
            )
            # fmt:on
            # TODO 3:
            # 如果当前评估成功率大于best_success_rate:
            # 1. 更新best_success_rate
            # 2. 使用torch.save()保存checkpoint
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                checkpoint_data = {
                    "episode": ep + 1,
                    "seed": seed,
                    "algorithm": algorithm,
                    "eval_success_rate": success_rate,
                    "online_state_dict": agent.online.state_dict(),
                }
                torch.save(checkpoint_data, checkpoint_path)
                print(f"New best model saved to: {checkpoint_path}")

    print("\n===== Training Summary =====")
    print("Environment steps:", global_steps)
    print("Gradient updates:", update_count)
    print("Target synchronizations:", sync_count)
    print("Successful episodes:", success_count)

    # ===== 保存训练日志 =====
    log_path = Path("logs") / f"{run_name}.csv"
    # parents=True	如果上级目录不存在，也自动创建
    # exist_ok=True	如果文件夹已经存在，不要报错
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # fmt:off
    # ewline=""	让 CSV 模块自行处理换行
    # with	自动管理文件的打开和关闭
    # as f	把打开的文件对象赋给变量 f
    # 使用 "w"，意味着每次运行训练脚本，同一路径下原来的 CSV 内容会被覆盖。
    with log_path.open("w", newline="", encoding="utf-8") as f:
        # csv.DictWriter 就是专门将 Python 字典写入 CSV 文件的工具。
        # fieldnames=[]指出CSV 文件的六个列名，以及列的排列顺序。
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "return",
                "success",
                "steps",
                "epsilon",
                "loss",
            ],
        )
        # fmt:on
        # TODO 2:
        # 写入csv表头
        # 作用是写入第一行：episode,return,success,steps,epsilon,loss
        writer.writeheader()
        # TODO 3:
        # 把episode_logs中的记录全部写入文件
        writer.writerows(episode_logs)

    # ===== 保存评估日志 =====
    eval_log_path = Path("logs") / f"{run_name}_eval.csv"
    with eval_log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "success_rate",
                "avg_return",
                "avg_steps",
            ],
        )
        writer.writeheader()
        writer.writerows(eval_logs)

    # ===== 新增：保存 monitor 日志 =====
    if monitor_logs:
        if monitor_output_path is None:
            monitor_output_path = Path("results") / f"{run_name}_monitor.csv"
        else:
            monitor_output_path = Path(monitor_output_path)

        monitor_output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = list(monitor_logs[0].keys())

        with monitor_output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(monitor_logs)

        print(f"Saved monitor logs to: {monitor_output_path}")

    return agent


# 检查当前文件是否作为主程序运行
if __name__ == "__main__":
    train(n_episodes=10, seed=999, algorithm="ddqn")
