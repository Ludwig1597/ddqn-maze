import numpy as np
from envs.maze import CoinMaze


def run_random_policy(episodes=100, seed=0):
    env = CoinMaze(seed=seed)
    # 创建独立的随机动作生成器
    action_rng = np.random.default_rng(seed + 1)
    # 统计数据
    returns = []
    successes = []
    steps_list = []

    outcomes = {"coin": 0, "trap": 0, "timeout": 0}

    # 开始运行多个Episode
    for ep in range(episodes):
        state = env.reset()
        done = False
        ep_return = 0.0
        ep_steps = 0

        while not done:
            # TODO 1:
            # 从0，1，2，3中均匀随机选择动作
            action = action_rng.integers(4)
            # 执行动作
            next_state, reward, done, info = env.step(action)

            ep_return += reward
            ep_steps += 1
            state = next_state

        # TODO 2:
        # 判断本回合是否找到金币
        success = int(tuple(env.pos) == env.coin)
        successes.append(success)
        returns.append(ep_return)
        steps_list.append(ep_steps)
        if success:
            outcomes["coin"] += 1
        elif tuple(env.pos) in env.traps:
            outcomes["trap"] += 1
        else:
            outcomes["timeout"] += 1
    # 汇总统计
    result = {
        "success_rate": np.mean(successes),
        "avg_return": np.mean(returns),
        "avg_steps": np.mean(steps_list),
        "outcomes": outcomes,
    }

    print("\n===== Random Policy Evaluation =====")
    print(result)

    return result


if __name__ == "__main__":
    run_random_policy(episodes=100, seed=0)
