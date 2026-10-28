# 为了测试我们刚刚算出的 0.675，这里临时用固定 Q 值网络代替真实 QNet。
import torch
from torch import nn
from agents.ddqn import DDQNAgent


class FixedQ(nn.Module):
    def __init__(self, values):
        super().__init__()
        # nn.Parameter()：把普通 Tensor 包装成神经网络可以学习的参数。
        self.table = nn.Parameter(torch.tensor(values, dtype=torch.float32))

    def forward(self, x):
        # Python 新知识是：切片（Slicing）语法 [:n]，完整写法是[0:n],从索引 0 开始，取到索引 n 之前
        # 这个 forward() 并不是真正的 QNet 神经网络前向传播。
        # 这是为了在 DDQN 单元测试中返回预先设定好的 Q 值，让我们能够准确检查 argmax()、gather() 和 TD Target 的计算是否正确。
        return self.table[: x.shape[0]]


agent = DDQNAgent(gamma=0.95)

# fmt: off
agent.online = FixedQ([
    [0.10,0.90,0.20,0.30],
    [0.60,0.10,0.20,0.00]
])

agent.target= FixedQ([
    [0.70,0.50,0.40,0.30],
    [0.10,0.90,0.20,0.30]
])
# fmt: on

rewards = torch.tensor([0.2, -1.0])
next_states = torch.zeros((2, 75))
dones = torch.tensor([0.0, 1.0])

targets = agent.compute_target(rewards, next_states, dones)
expected = torch.tensor([0.675, -1.0])

assert targets.shape == (2,)
# 检查计算出来的 TD Target 是否与预期结果足够接近。
# torch.allclose()	检查两个 Tensor 的对应元素是否都足够接近
# atol=1e-6	绝对误差容忍值，等于 0.000001
assert torch.allclose(targets, expected, atol=1e-6)
assert targets.requires_grad is False

print("DDQN target test PASSED!")
print(f"Targets: {targets}")
