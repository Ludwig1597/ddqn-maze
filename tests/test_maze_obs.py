import numpy as np
from envs.maze import CoinMaze

env = CoinMaze(seed=42)
env.reset()

# 手动固定环境，便于验证
env.pos = np.array([0, 0])
env.coin = (0, 1)
env.traps = {(1, 0), (5, 6), (6, 4)}

obs = env.obs()

# 测试1: 维度与数据类型
assert obs.shape == (75,)
assert obs.dtype == np.float32

# 转回 3x5x5，方便检查
o = obs.reshape(3, 5, 5)

# 测试2: 中心格可通行
assert o[0, 2, 2] == 1.0

# 测试3: 局部视野之外的地图位置，标记为不可通行
assert o[0, 0, 2] == 0.0

# 测试4: 固定障碍(1,1)不可通行
assert o[0, 3, 3] == 0.0

# 测试5: 金币(0,1)显示在通道1上
assert o[1, 2, 3] == 1.0
assert o[1].sum() == 1.0

# 测试6: 陷阱(1,0)显示在通道2上
assert o[2, 3, 2] == 1.0
assert o[2].sum() == 1.0

print("All observation tests PASSED!")
print(f"Observation shape: {obs.shape}")
print(f"Observation dtype: {obs.dtype}")
