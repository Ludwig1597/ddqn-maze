import random
import numpy as np
import torch

from envs.maze import CoinMaze
from agents.ddqn import DDQNAgent


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

    for ep in range(n_episodes):
        state = env.reset()
        done = False
        ep_return = 0.0
        ep_steps = 0

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

        print("\n===== Training Summary =====")
        print("Environment steps:", global_steps)
        print("Gradient updates:", update_count)
        print("Target synchronizations:", sync_count)
        print("Successful episodes:", success_count)
    return agent


# 检查当前文件是否作为主程序运行
if __name__ == "__main__":
    train(n_episodes=10, seed=0)
