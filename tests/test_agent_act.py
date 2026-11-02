import random
import torch

from envs.maze import CoinMaze
from agents.ddqn import DDQNAgent

random.seed(42)

env = CoinMaze(seed=42)
agent = DDQNAgent()
state = env.reset()

# 测试1: epsilon=0，完全使用QNet选择动作
with torch.no_grad():
    # 将迷宫状态 state 转换为 PyTorch Tensor，并增加一个 Batch 维度，以便输入 QNet。
    # 这里的 x 只是一个变量名，通常用来表示神经网络的输入数据（Input）。shape 为 (1,75)
    x = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
    expected_action = int(agent.online(x).argmax(dim=1).item())
action = agent.act(state, epsilon=0.0)
assert action == expected_action
# isinstance() 是 Python 的内置函数，用来判断一个对象是否属于某个类型（包括其子类）。
assert isinstance(action, int)

print("Test1: Greedy action PASSED")

# 测试2: epsilon=1，执行随机探索
actions = [agent.act(state, epsilon=1.0) for _ in range(100)]

assert all(0 <= a < 4 for a in actions)
# set() 是 Python 的内置函数，可以把列表转换成集合（Set）。
# 用语去除重复元素
# 意思是：断言 actions 列表中至少出现了两种不同的动作。
assert len(set(actions)) > 1
print("Test2: Random exploration PASSED")

# 测试3: 真实环境交互
next_state, reward, done, info = env.step(action)
assert next_state.shape == (75,)

print("Test3: Environment interaction PASSED")
print("All action selection tests PASSED")
