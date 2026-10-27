import numpy as np
from agents.ddqn import Replaybuffer

buf = Replaybuffer(capacity=100)

# 手工创建80条模拟数据
for i in range(80):
    # np.full() 是 NumPy 提供的一个函数，用来创建指定形状的数组，并用同一个值填满所有元素。
    # 创建一个长度为 75 的一维数组，把数组中所有元素设置为 i，使用 32 位浮点数
    s = np.full(75, i, dtype=np.float32)
    s2 = np.full(75, i + 1, dtype=np.float32)
    buf.push(s, i % 4, -0.01, s2, False)

assert len(buf) == 80

states, actions, rewards, next_states, dones = buf.sample(64)

assert states.shape == (64, 75)
assert actions.shape == (64,)
assert rewards.shape == (64,)
assert next_states.shape == (64, 75)
assert dones.shape == (64,)

assert states.dtype == np.float32
assert actions.dtype == np.int64
assert rewards.dtype == np.float32
assert next_states.dtype == np.float32
assert dones.dtype == np.float32

print("Replay Buffer test PASSED")
