import numpy as np
from envs.maze import CoinMaze


def evaluate(agent, episodes=20, seed=999, epsilon=0.0):
    # 独立环境，避免影响训练环境的随机状态
    env = CoinMaze(seed=seed)

    returns = []
    successes = []
    steps_list = []
    # fmt:off
    outcomes={
        "coin":0,
        "trap":0,
        "timeout":0
    }
    # fmt:on

    for ep in range(episodes):
        state = env.reset()
        done = False
        ep_return = 0.0  # 单回合累积奖励
        ep_steps = 0  # 但回合步数

        while not done:
            # TODO 1:
            # 用agent.act()选动作
            # 注意: epsilon=0.0
            action = agent.act(state, epsilon=epsilon)

            # TODO 2:
            # 执行动作，获得下一状态、奖励、done
            next_state, reward, done, info = env.step(action)

            # 评估只交互，不写经验池，不调用update()
            ep_return += reward
            ep_steps += 1
            state = next_state

        # 是否找到金币
        success = int(tuple(env.pos) == env.coin)
        if success:
            # TODO 1: 金币成功次数+1
            outcomes["coin"]+=1
        elif tuple(env.pos) in env.traps:
            # TODO 2: 陷阱终止次数+1
            outcomes["trap"]+=1
        else:
            # TODO 3: 步数超时次数+1
            outcomes["timeout"]+=1

        returns.append(ep_return)
        successes.append(success)
        steps_list.append(ep_steps)

    # TODO 3:
    # 返回一个字典，包含：
    # success_rate: 成功率平均值
    # avg_return: 平均回合奖励
    # avg_steps: 平均回合步数
    # 提示：np.mean()

    return {
        "success_rate": float(np.mean(successes)),
        "avg_return": float(np.mean(returns)),
        "avg_steps": float(np.mean(steps_list)),
        "outcomes": outcomes
    }
