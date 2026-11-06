from envs.maze import CoinMaze, MOVES, WALLS
import numpy as np


def manhattan_distance(pos, goal):
    # pos: 当前坐标 (r, c)
    # goal: 金币坐标 (r, c)

    # 返回当前位置到金币的曼哈顿距离。

    # TODO 1:
    # 计算行坐标差的绝对值
    # 加上列坐标差的绝对值
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])


# tie的意思是平局
# tie_rng: 打破平局的随机数生成器。
def greedy_action(env, tie_rng):
    # 当前智能体位置
    r, c = env.pos
    # 保存合法且安全的动作
    candidates = []
    # 遍历四个动作
    # action, (dr, dc) = (0, (-1, 0))
    # 把一个元组拆成三个变量
    for action, (dr, dc) in MOVES.items():
        nr = r + dr
        nc = c + dc
        # TODO 1:
        # 检查是否越界
        # 检查是否为墙
        # 检查是否在陷阱中
        # 不合法就continue
        if not (0 <= nr < env.size and 0 <= nc < env.size):
            continue
        if env.walls[nr, nc] == 1:
            continue
        next_pos = (nr, nc)
        if next_pos in env.traps:
            continue
        # 跳过该动作

        # TODO 3:
        # 计算next_pos到金币的曼哈顿距离
        # 保存(distance,action)到candidates
        distance = manhattan_distance(next_pos, env.coin)
        candidates.append((distance, action))
    # 没有安全动作
    if not candidates:
        return None
    # TODO 4:
    # 找到距离最小的动作
    # 遍历 candidates 中所有候选动作，提取每个动作对应的距离，然后找到最小距离。
    # 遍历 candidates。每次取出一个 (distance, action)。只提取其中的 distance。ß把 distance 交给外面的 min()。

    min_distance = min(distance for distance, action in candidates)
    # 如果有多个动作距离相同，使用tie_rng随机选择
    best_actions = []
    for distance, action in candidates:
        if distance == min_distance:
            best_actions.append(action)
    return int(tie_rng.choice(best_actions))


def run_greedy_policy(episodes=100, seed=0):
    env = CoinMaze(seed=seed)
    tie_rng = np.random.default_rng(seed + 1)
    returns = []
    successes = []
    steps_list = []
    outcomes = {"coin": 0, "trap": 0, "timeout": 0, "stuck": 0}

    for ep in range(episodes):
        state = env.reset()
        done = False

        ep_return = 0.0
        ep_steps = 0
        stuck = False

        while not done:
            # TODO 1:
            # 使用greedy_action()选择动作
            action = greedy_action(env=env, tie_rng=tie_rng)

            # TODO 2:
            # 如果action is None，设置stuck=True，跳出while循环
            if action is None:
                stuck = True
                break
            # TODO 3:
            # 执行动作，获取next_state,reward,done,info
            next_state, reward, done, info = env.step(action)
            # TODO 4:
            # 更新累计奖励，步数和state
            ep_return += reward
            ep_steps += 1
            state = next_state

        # TODO 5:
        # 判断本回合是否找到金币
        success = int(tuple(env.pos) == env.coin)
        # TODO 6:
        # 根据最终状态统计终止原因
        if success:
            outcomes["coin"] += 1
        elif tuple(env.pos) in env.traps:
            outcomes["trap"] += 1
        elif stuck == True:
            outcomes["stuck"] += 1
        else:
            outcomes["timeout"] += 1
        # TODO 7:
        # 保存本回合的return，success，steps
        successes.append(success)
        returns.append(ep_return)
        steps_list.append(ep_steps)

    # 计算最终统计结果
    result = {
        "success_rate": np.mean(successes),
        "avg_return": np.mean(returns),
        "avg_steps": np.mean(steps_list),
        "outcomes": outcomes,
    }

    print("\n===== Greedy Policy Evaluation =====")
    print(result)

    return result


if __name__ == "__main__":
    run_greedy_policy(episodes=100, seed=0)
