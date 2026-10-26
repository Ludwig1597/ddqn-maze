import numpy as np

# 8个固定障碍格
WALLS = {
    (1,1),(1,2),(2,4),(3,1),(3,5),(4,3),(5,4),(6,6)
}

class CoinMaze:
    def __init__(self,size=8,max_steps=200,seed=0):
        if size!=8:
            raise ValueError("This project uses an 8x8 maze")
        # 1. 基础配置
        self.size=size
        self.max_steps=max_steps
        # 2. 随机数生成器
        self.rng=np.random.default_rng(seed)
        # 3. 创建地图
        self.walls=np.zeros((size,size),dtype=np.int8)
        # TODE 1:
        # 遍历WALLS，把障碍格在self.walls中设为1
        for (r,c) in WALLS:
            self.walls[r,c]=1
        
        # 4. 金币候选位置
        self.coin_cands=[(7,7),(0,7),[7,0],[3,4]]
        # 5. 陷阱候选位置
        self.trap_cands=[(2,2),(2,5),(5,2),(5,6),(6,4),(3,6)]
        
        # TODE 2:
        # 创建智能体初始位置 self.pos
        # 使用np.array,位置为(0,1=0)
        self.pos=(0,0)
        
        # TODE 3:
        # 初始化一下下三个状态变量
        self.t=0 # 初始步数为0，在强化学习中，t 通常表示时间步（Time Step），也就是智能体与环境交互的步数。
        self.coin=None # 金币位置尚未确定，因此先用None，我们还没有执行 reset() 来生成金币位置
        self.traps=set() # 初始化为空集合
        
        