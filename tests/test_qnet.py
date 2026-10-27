import torch
from networks.qnet import QNet

model = QNet()

# 三个状态，
# 这里的 3 不再表示通道数量，而是 Batch Size（批量大小）
# randn 是 PyTorch 中用于生成服从标准正态分布（Standard Normal Distribution）的随机数的函数。
# rand：Random（随机）
# n：Normal（正态分布）
states = torch.randn(3, 75)

q_value = model(states)
# 检查输出
assert q_value.shape == (3, 4)

# 检查网络参数数量
n_params = sum(p.numel() for p in model.parameters())
assert n_params == 26756

print("QNet forward test PASSED!")
print(f"Out shape: {q_value.shape}")
print(f"Parameters: {n_params}")
print(f"Q values: {q_value}")
