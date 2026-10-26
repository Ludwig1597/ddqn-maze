import numpy as np
from envs.maze import CoinMaze

env = CoinMaze(seed=42)

# 检查reset返回的观测
state = env.reset()

# 检查变量 state 是否是 NumPy 数组（ndarray）
# isinstance(对象, 类型) 判断一个对象是否属于指定类型
assert isinstance(state, np.ndarray)
assert state.shape == (75,)
assert state.dtype == np.float32

print("Test 1: Reset API PASSED")

# 检查step返回的观测
next_state, reward, done, info = env.step(3)

# ndarray 是 N-dimensional Array（N 维数组）的缩写，
# NumPy 用这个名字强调：它不仅能表示一维数组，还能表示二维、三维乃至更高维的数组
assert isinstance(next_state, np.ndarray)
assert next_state.shape == (75,)
assert next_state.dtype == np.float32
assert isinstance(info, dict)
assert done is False

print("Test 2: Step API PASSED")

# 检查真正终止时也返回观测
env.pos = np.array([0, 1])
env.coin = (0, 2)
env.traps = set()

next_state, reward, done, info = env.step(3)

assert next_state.shape == (75,)
assert reward == 1.0
assert done is True

print("Test 3: Terminal API PASSED")
print("All API tests PASSED")
