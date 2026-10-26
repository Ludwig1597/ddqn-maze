import numpy as np

# 8个固定障碍格
WALLS = {(1, 1), (1, 2), (2, 4), (3, 1), (3, 5), (4, 3), (5, 4), (6, 6)}

class CoinMaze:
    # 创建迷宫对象，设置基础配置
    def __init__(self, size=8, max_steps=200, seed=0):
        if size != 8:
            raise ValueError("This project uses an 8x8 maze")
        # 1. 基础配置
        self.size = size
        self.max_steps = max_steps
        # 2. 随机数生成器
        self.rng = np.random.default_rng(seed)
        # 3. 创建地图
        self.walls = np.zeros((size, size), dtype=np.int8)
        # TODO 1:
        # 遍历WALLS，把障碍格在self.walls中设为1
        for r, c in WALLS:
            self.walls[r, c] = 1

        # 4. 金币候选位置
        self.coin_cands = [(7, 7), (0, 7), (7, 0), (3, 4)]
        # 5. 陷阱候选位置
        self.trap_cands = [(2, 2), (2, 5), (5, 2), (5, 6), (6, 4), (3, 6)]

        # TODO 2:
        # 创建智能体初始位置 self.pos
        # 使用np.array,位置为(0,1=0)
        self.pos = np.array([0, 0])

        # TODO 3:
        # 初始化一下下三个状态变量
        self.t = 0  # 初始步数为0，在强化学习中，t 通常表示时间步（Time Step），也就是智能体与环境交互的步数。
        self.coin = None  # 金币位置尚未确定，因此先用None，我们还没有执行 reset() 来生成金币位置
        self.traps = set()  # 初始化为空集合

    # reset()：初始化一个新的训练回合。
    # reset() 不等于重新创建整个环境对象。 固定障碍不会每回合重新生成，但金币、陷阱和智能体位置需要重置。
    def reset(self):
        # TODO 1: 重置步数为 0
        self.t = 0
        # TODO 2: 将智能体位置重置为 NumPy 数组 [0, 0]
        self.pos = np.array([0, 0])
        # TODO 3: 从 self.coin_cands 中随机选择一个金币位置
        self.coin = self.coin_cands[
            self.rng.integers(len(self.coin_cands))
        ]  # self.rng.integers(4)随机产生{0,1,2,3}
        # TODO 4: 从 self.trap_cands 中随机选择 3 个
        # self.rng.choice(6, size=3, replace=False)
        # 不允许重复，并将结果转为集合
        # 意思是从索引 0～5 中抽取，随机抽取 3 个索引，replace=False表示不放回抽样，确保不会重复
        indices = self.rng.choice(len(self.trap_cands), size=3, replace=False)
        self.traps = set()
        for i in indices:
            self.traps.add(self.trap_cands[i])

        # 暂时返回 None，等完成 obs() 后再替换
        return None

    # step()：控制智能体与环境交互。
    # step() 是强化学习中 Agent（智能体）与 Environment（环境）交互的核心接口。
    # 环境的 step() 对应 MDP 中的状态转移和奖励机制。
    # 智能体在状态 (s_t) 执行动作 (a_t)，环境产生：(s_t,a_t)=>(s_t+1,r_t+1)
