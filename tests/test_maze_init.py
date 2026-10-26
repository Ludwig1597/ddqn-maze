from envs.maze import CoinMaze

env=CoinMaze(seed=42)

# 检查地图维度
assert env.walls.shape==(8,8)

# 检查障碍数量
assert int(env.walls.sum())==8

# 检查初始位置
assert tuple(env.pos)==(0,0)

# 检查候选位置数量
assert len(env.coin_cands)==4
assert len(env.trap_cands)==6

# 检查动态状态
# is：比较两个变量是否引用同一个对象。
# Python 中的 None 是一个单例对象（Singleton），整个 Python 运行环境中只有一个 None 对象。
assert env.coin is None
assert env.traps==set()

print("Maze initialization test PASSED!")