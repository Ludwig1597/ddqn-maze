# 从 Python 标准库的 collections 模块中，导入 Counter（计数器）类。
# Counter 的作用是：自动统计一组数据中，每个元素出现了多少次。
from collections import Counter
from train import train
from envs.maze import CoinMaze

# 训练一个50Episode的DDQN
agent = train(n_episodes=50, seed=0)

# 使用独立的评估环境
# 独立评估环境（Independent Evaluation Environment），意思是：
# 训练 DDQN 时使用一个迷宫环境，
# 训练结束后，再创建另一个迷宫环境，专门测试智能体的表现。
env = CoinMaze(seed=999)

for ep in range(200):
    state = env.reset()
    done = False

    # 记录走过的每一个位置
    # Trajectory 的中文意思是：轨迹、运动轨迹。
    # 在强化学习（Reinforcement Learning）中，Trajectory 通常指智能体从初始状态开始，与环境交互所经历的一系列状态、动作和奖励。
    trajectory = [tuple(env.pos)]

    while not done:
        action = agent.act(state, epsilon=0.0)
        next_state, reward, done, info = env.step(action)

        # TODO 1:
        # 将当前智能体位置加入trajectory
        trajectory.append(tuple(env.pos))

        # TODO 2:
        # 更新state，供下一次选择动作
        state = next_state

    # 找到第一个超时回合
    # 用来估算智能体撞墙或尝试非法移动的次数。
    # fmt:off
    if (env.t == env.max_steps and tuple(env.pos) != env.coin and tuple(env.pos) not in env.traps):
        blocked_steps = sum(
            # trajectory[1:]	从第二个位置取到最后
            p1 == p2 for p1, p2 in zip(trajectory, trajectory[1:])
        )
    # fmt:on
        # 超时诊断
        print("\n===== Timeout Diagnosis =====")
        print(f"Episode: {ep+1}")
        print(f"Coin position: {env.coin}")
        print(f"Trap positions: {env.traps}")
        print(f"Unique positions: {len(set(trajectory))}")  # 智能体访问过多少个不同格子
        print(f"Blocked steps: {blocked_steps}") # 有多少次动作没有改变位置
        print(f"Most visited positions: {Counter(trajectory).most_common(5)}") # 是否反复访问少数几个格子
        print(f"First 20 positions: {trajectory[:20]}") # 开始时的行动轨迹
        print(f"Last 20 positions: {trajectory[-20:]}") # 超时前是否在重复某种动作

        break
