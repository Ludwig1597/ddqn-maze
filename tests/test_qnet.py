import torch
from networks.qnet import QNet
from envs.maze import CoinMaze

model = QNet()

# 三个状态，
# 这里的 3 不再表示通道数量，而是 Batch Size（批量大小）
# randn 是 PyTorch 中用于生成服从标准正态分布（Standard Normal Distribution）的随机数的函数。
# rand：Random（随机）
# n：Normal（正态分布）
states = torch.randn(3, 75)

q_values = model(states)
# 检查输出
assert q_values.shape == (3, 4)

# 检查网络参数数量
n_params = sum(p.numel() for p in model.parameters())
assert n_params == 26756

print("QNet forward test PASSED!")
print(f"Out shape: {q_values.shape}")
print(f"Parameters: {n_params}")
print(f"Q values: {q_values}")

env = CoinMaze(seed=42)
# NumPy 数组，shape=(75,)
obs = env.reset()
# NumPy → Tensor → 增加 Batch 维度
# from_numpy() 通常与原 NumPy 数组共享内存，而不是复制一份新数据
# 把 NumPy 数组 obs 转换成 PyTorch Tensor
# 然后在最前面增加一个维度（Batch 维度），以便输入 QNet 神经网络
# unsqueeze 可以理解为 增加一个长度为 1 的维度
# unsqueeze(0)表示在第 0 个位置插入新维度: (75,) → (1,75)
state = torch.from_numpy(obs).unsqueeze(0)
q_values = model(state)

assert q_values.shape == (1, 4)
# argmax()：返回最大值的位置（索引）
# 对于二维张量：q_values.shape=(1,4)
# - dim=0：沿着 Batch 维度比较。
# - dim=1：沿着动作维度比较。
action = q_values.argmax(dim=1).item()

print(f"Real observation Q values: {q_values}")
print(f"Selected action: {action}")
