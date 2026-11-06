import numpy as np
from envs.maze import CoinMaze


def test_same_seed_sequence():
    seed = 42
    episodes = 100

    # 创建两个独立环境
    env1 = CoinMaze(seed=seed)
    env2 = CoinMaze(seed=seed)
    for ep in range(episodes):
        # 分别重置两个环境
        obs1 = env1.reset()
        obs2 = env2.reset()

        # TODO 1:
        # 比较两个环境的金币位置
        assert env1.coin == env2.coin
        # TODO 2:
        # 比较两个环境的陷阱位置
        assert env1.traps == env2.traps
        # TODO 3:
        # 比较两个环境的障碍地图
        # 提示：np.array_equal()
        assert np.array_equal(env1.walls, env2.walls)
        # TODO 4:
        # 比较两个环境的观测数组
        assert np.array_equal(obs1, obs2)

    print("Same seed reproducibility PASSED!")


if __name__ == "__main__":
    test_same_seed_sequence()
