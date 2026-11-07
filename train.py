import random
import numpy as np
import torch
import csv

# 从 Python 标准库的 pathlib 模块中，导入 Path 类，用来操作文件和文件夹路径。
from pathlib import Path

from envs.maze import CoinMaze
from agents.ddqn import DDQNAgent
from evaluate import evaluate


# 探索率衰减函数
def epsilon_schedule(step):
    # TODO 1:
    # epsilon从1.0线形衰减至0.05
    # 总衰减步数为20000
    # 提示：max()、step/20000
    return max(0.05, 1 - 0.95 * step / 20000)


# n_episodes：Number of Episodes，训练回合总数。
# n_episodes=10 表示总共训练 10 个回合。
# 从 env.reset() 开始，直到 done=True，通常就算完成一个 Episode
def train(n_episodes=10, seed=0):
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
    agent = DDQNAgent()

    global_steps = 0
    update_count = 0  # 发生多少次真正的梯度更新
    sync_count = 0  # Target Network同步了多少次
    success_count = 0  # 有多少个Episode找到金币
    episode_logs = []
    eval_logs = []  # 用于存储每隔 50 Episode 产生的一条评估记录。

    best_success_rate = -1.0
    # pt是pyTorch的缩写，通常用于保存神经网络参数
    checkpoint_path = Path("checkpoints") / f"ddqn_seed{seed}_best.pt"
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

            # 每500个全局环境步同步Target
            if global_steps % 500 == 0:
                agent.sync_target()
                sync_count += 1  # 记录 Target 同步次数
            # TODO 5:
            # 更新state和ep_return
            # ep_return 表示 Episode Return（单回合累计奖励）
            state = next_state
            ep_return += reward

        success = int(tuple(env.pos) == env.coin)  # 通过环境终局位置判断是否到达金币
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
    # logs/ddqn_seed0.csv
    log_path = Path("logs") / f"ddqn_seed{seed}.csv"
    # parents=True	如果上级目录不存在，也自动创建
    # exist_ok=True	如果文件夹已经存在，不要报错
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # fmt:off
    # ewline=""	让 CSV 模块自行处理换行
    # with	自动管理文件的打开和关闭
    # as f	把打开的文件对象赋给变量 f
    # 使用 "w"，意味着每次运行训练脚本，同一路径下原来的 CSV 内容会被覆盖。
    with log_path.open("w",newline="",encoding="utf-8") as f:
        # csv.DictWriter 就是专门将 Python 字典写入 CSV 文件的工具。
        # fieldnames=[]指出CSV 文件的六个列名，以及列的排列顺序。
        writer=csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "return",
                "success",
                "steps",
                "epsilon",
                "loss"
            ]
        )
        # fmt:on
        # TODO 2:
        # 写入csv表头
        # 作用是写入第一行：episode,return,success,steps,epsilon,loss
        writer.writeheader()
        # TODO 3:
        # 把episode_logs中的记录全部写入文件
        writer.writerows(episode_logs)
    # 评估CSV路径
    eval_log_path=Path("logs")/f"ddqn_seed{seed}_eval.csv"
    with eval_log_path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "success_rate",
                "avg_return",
                "avg_steps"
            ]
        )
        writer.writeheader()
        writer.writerows(eval_logs)
    return agent


# 检查当前文件是否作为主程序运行
if __name__ == "__main__":
    train(n_episodes=10, seed=0)
