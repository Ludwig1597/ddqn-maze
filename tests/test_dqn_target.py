"""
文件：tests/test_dqn_target.py
用途：验证 DQN 和 DDQN 的 TD Target
运行：python -m tests.test_dqn_target
"""

import torch
import torch.nn as nn

from agents.dqn import DQNAgent
from agents.ddqn import DDQNAgent


class FixedQ(nn.Module):
    """返回预先指定的 Q 值，用于单元测试。"""

    def __init__(self, values):
        super().__init__()
        # PyTorch 中 register_buffer() 的用途：
        # 把一组固定的 Q 值保存到神经网络模型中，但不把它们当作需要训练的参数。
        self.register_buffer(
            "table",  # 给这个 buffer 取名为 table
            torch.tensor(values, dtype=torch.float32),  # 将输入数据转换为 Tensor
        )

    def forward(self, x):
        # 前两行：被 [:2] 选中并返回
        # batch_size = x.shape[0]
        return self.table[: x.shape[0]]


def test_dqn_target():

    # 创建两个 Agent
    dqn = DQNAgent()
    ddqn = DDQNAgent()

    online_values = [[5.0, 0.0, 1.0, 2.0], [5.0, 0.0, 1.0, 2.0]]

    target_values = [[1.0, 3.0, 2.0, 0.0], [1.0, 3.0, 2.0, 0.0]]

    # 用固定输出替换两个 Agent 的网络
    for agent in [dqn, ddqn]:
        agent.online = FixedQ(online_values)
        agent.target = FixedQ(target_values)

    # 2 个样本，每个状态 75 维
    next_states = torch.zeros(2, 75)

    rewards = torch.tensor([0.2, -1.0])
    dones = torch.tensor([0.0, 1.0])

    # TODO 1:
    # 调用 dqn.compute_target()
    dqn_targets = dqn.compute_target(rewards, next_states, dones)

    # TODO 2:
    # 调用 ddqn.compute_target()
    ddqn_targets = ddqn.compute_target(rewards, next_states, dones)

    # TODO 3:
    # 构造 expected_dqn
    # 预期：[3.05, -1.0]
    expected_dqn = torch.tensor([3.05, -1.0])

    # TODO 4:
    # 构造 expected_ddqn
    # 预期：[1.15, -1.0]
    expected_ddqn = torch.tensor([1.15, -1.0])

    # TODO 5:
    # 用 torch.allclose() 验证两个结果
    assert torch.allclose(dqn_targets, expected_dqn)
    assert torch.allclose(ddqn_targets, expected_ddqn)

    # TODO 6:
    # 检查形状均为 (2,)
    # 检查 requires_grad 为 False
    assert dqn_targets.shape == (2,)
    assert ddqn_targets.shape == (2,)

    # 相当于验证：assert dqn_targets.requires_grad == False
    assert not dqn_targets.requires_grad
    assert not ddqn_targets.requires_grad

    print("DQN and DDQN target tests PASSED!")


if __name__ == "__main__":
    test_dqn_target()
