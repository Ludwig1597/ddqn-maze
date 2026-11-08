"""
文件: scripts/diagnose_checkpoint.py
用途: 分析最佳 DDQN 模型的超时轨迹
运行: python -m scripts.diagnose_checkpoint
"""

import torch
from collections import Counter

from agents.ddqn import DDQNAgent
from envs.maze import CoinMaze


def diagnose_checkpoint(
    checkpoint_path="checkpoints/ddqn_seed0_best.pt", seed=2026, episodes=100
):
    # step1: 加载训练好的模型
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",  # 设备映射位置：将加载的 Tensor 放到 CPU
        weights_only=True,  # 受限制的加载模式：主要加载权重、张量、基本类型及字典等允许的数据
    )

    # TODO 1:
    # 创建DDQNAgent
    # 加载 online_state_dict
    # 同步 Target Network
    # 将Online Network设置为eval模式
    agent = DDQNAgent()
    agent.online.load_state_dict(checkpoint["online_state_dict"])
    agent.sync_target()
    agent.online.eval()
    # step2: 创建环境
    env = CoinMaze(seed=seed)
    for ep in range(episodes):
        state = env.reset()
        done = False

        # 记录本回合经过的所有位置
        # map(int, env.pos)：把坐标中的每个元素转换成整数。
        positions = [tuple(map(int, env.pos))]
        # step3: 执行一个episode
        while not done:
            # TODO 2:
            # 使用anget.act()选择动作
            # epsilon=0.0
            action = agent.act(state, epsilon=0.0)

            # TODO 3:
            # 使用env.step()执行动作
            # 获取next_state,reward,done,info
            next_state, reward, done, info = env.step(action)

            # TODO 4:
            # 更新state
            # 把当前env.pos追加到positions
            state = next_state
            positions.append(tuple(map(int, env.pos)))
        # step4: 判断是否超时
        # 提示：env.t、env.max_steps
        # 同时确保不是找到金币或掉入陷阱
        timeout = (
            env.t >= env.max_steps
            and tuple(env.pos) != env.coin
            and tuple(env.pos) not in env.traps
        )
        # step5: 分析第一个超时回合
        if timeout:
            # TODO 6:
            #   打印 Episode 编号
            #   打印金币和陷阱位置
            #   使用 Counter 统计访问次数
            #   打印 Unique positions
            #   打印 Most visited positions
            #   打印 First 20 positions
            #   打印 Last 20 positions
            #   break
            counts = Counter(positions)  # 统计位置访问次数
            print(f"Episode: {ep+1}")
            print(f"Coin: {env.coin}")
            print(f"Traps: {env.traps}")
            print(f"Unique positions: {len(set(positions))}")
            print(f"Most visited positions: {counts.most_common(5)}")
            print(f"First 20 positions: {positions[:20]}")
            print(f"Last 20 positions: {positions[-20:]}")
            break


if __name__ == "__main__":
    diagnose_checkpoint()
