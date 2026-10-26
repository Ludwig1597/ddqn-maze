from envs.maze import CoinMaze

# 创建环境并开始回合
env = CoinMaze(seed=42)
env.reset()

# 固定金币与陷阱位置
env.coin = (0, 2)
env.traps = {(1, 0)}

# 测试1: 向右移动
obs, reward, done, info = env.step(3)

assert tuple(env.pos) == (0, 1)
assert reward == -0.01
assert done is False

print("Test 1: Normal movement PASSED")

# 测试2: 向下撞墙(1,1)
obs, reward, done, info = env.step(1)

assert tuple(env.pos) == (0, 1)
# |实际值-预期值| < 允许误差
assert abs(reward - (-0.03)) < 1e-8
assert done is False

print("Test 2: Wall collision PASSED")

# 测试3: 向右走到金币(0,2)
obs, reward, done, info = env.step(3)

assert tuple(env.pos) == (0, 2)
assert reward == 1.0
assert done is True

print("Test 3: Coin reward PASSED")

# 测试4: 向下踏入陷阱
env.reset()
env.coin = (7, 7)
env.traps = {(1, 0)}

obs, reward, done, info = env.step(1)

assert tuple(env.pos) == (1, 0)
assert reward == -1.0
assert done is True

print("Test 4: Trap termination PASSED")

# 测试5: 达到最大步数
short_env = CoinMaze(max_steps=1, seed=42)
short_env.reset()

# 从(0,0)向上撞边界，消耗一次动作
obs, reward, done, info = short_env.step(0)

assert tuple(short_env.pos) == (0, 0)
assert abs(reward - (-0.03)) < 1e-8
assert done is True

print("Test 5: Time limit PASSED")

print("All step tests PASSED!")
