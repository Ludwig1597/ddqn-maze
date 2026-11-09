"""
文件: tests/test_max_q.py
用途: 验证最大 Q 值估计函数
运行: python -m tests.test_max_q
"""

import torch
from agents.ddqn import DDQNAgent
from scripts.estimate_q_bias import compute_max_q


def test_max_q():
    torch.manual_seed(0)

    agent = DDQNAgent()
    states = torch.randn(200, 75)

    max_q = compute_max_q(agent, states)

    # 1. 输出形状
    assert max_q.shape == (200,)

    # 2. 所有预测值必须有限
    assert torch.isfinite(max_q).all()

    # 3. 不应该建立梯度计算图
    assert not max_q.requires_grad

    # 4. 验证计算值正确
    with torch.no_grad():
        expected = agent.online(states).max(dim=1).values

    assert torch.allclose(max_q, expected)

    print("test_max_q PASSED!")


if __name__ == "__main__":
    test_max_q()
