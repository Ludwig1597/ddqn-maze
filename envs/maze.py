import numpy as np

# 8个固定障碍格
WALLS = {(1, 1), (1, 2), (2, 4), (3, 1), (3, 5), (4, 3), (5, 4), (6, 6)}
# fmt: off
MOVES = {
    0: (-1, 0), # up
    1: (1, 0),  # down
    2: (0, -1), # left
    3: (0, 1)   # right
}        
# fmt: on


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
    # 文档要求 step() 接收一个动作，并返回：next_obs, reward, done, info
    # 返回值	 含义
    # next_obs	执行动作后的新观测
    # reward	本次动作获得的奖励
    # done	    本回合是否结束
    # info	    附加信息，当前可以为空字典
    def step(self, a):
        # 1. 步数+1
        self.t += 1
        # 2. 根据动作获得位置偏移
        dr, dc = MOVES[a]
        # 3. 计算目标位置
        nr = self.pos[0] + dr
        nc = self.pos[1] + dc
        # 4. 默认每步奖励
        reward = -0.01
        # 5. 检查目标格子是否可通行
        if not self._free(nr, nc):
            # TODO 1:
            # 撞墙时位置不变
            # 在reward基础上在减去0.02
            reward -= 0.02
        else:
            # TODO 2:
            # 更新智能体的位置 self.pos
            # 使用np.array
            self.pos = np.array([nr, nc])
        # 6. 检查是否到达金币
        # tuple() 会把 NumPy 数组转换为 Python 元组。
        if tuple(self.pos) == self.coin:
            # TODO 3:
            # 返回None，+1.0，True，{}
            # 到达终点金币直接获得 +1.0，不叠加步数惩罚。
            return None, 1.0, True, {}
        # 7. 检查是否踏入陷阱
        if tuple(self.pos) in self.traps:
            # TODO 4:
            # 返回None，-1.0，True，{}
            # 到达陷阱直接获得 -1.0，不叠加步数惩罚。
            return None, -1.0, True, {}
        # 8. 检查是否到达最大步数
        done = self.t >= self.max_steps
        # 9. 返回本次交互结果
        return None, reward, done, {}

    def _free(self, r, c):
        # TODO 1:
        # 如果 r 或 c 超出地图边界，返回 False
        if r < 0 or r >= self.size or c < 0 or c >= self.size:
            return False
        # TODO 2:
        # 如果 self.walls[r, c] == 1，返回 False
        if self.walls[r, c] == 1:
            return False
        # TODO 3:
        # 否则返回 True
        return True
