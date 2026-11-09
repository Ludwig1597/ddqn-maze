"""
文件: scripts/estimate_q_bias.py
用途: 计算神经网络预测的最大 Q 值
运行: python -m scripts.estimate_q_bias
"""

import torch


@torch.no_grad()
def compute_max_q(agent, states):
    """
    states: (batch_size, 75)
    返回:   (batch_size,)
    """

    # TODO 1:
    # 调用 agent.online(states)
    # 得到所有动作的 Q 值
    q_values = agent.online(states)

    # TODO 2:
    # 对动作维度取最大值
    max_q = q_values.max(dim=1).values

    # TODO 3:
    # 返回每个状态的最大 Q 值
    return max_q
