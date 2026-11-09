"""
文件: agents/dqn.py
用途: DQN 对照算法
"""

import torch
from agents.ddqn import DDQNAgent


# 自动继承 DDQN 的：
# - __init__()：初始化网络、Replay Buffer、优化器。
# - act()：ε-greedy 动作选择。
# - update()：训练与反向传播。
# - sync_target()：同步 Target Network。
class DQNAgent(DDQNAgent):
    # 只需要覆盖 compute_target()。
    @torch.no_grad()
    def compute_target(self, rewards, next_states, dones):

        # TODO 1:
        # 使用 Target Network 计算下一状态的 Q 值
        # 形状：(batch_size, 4)
        target_q_values = self.target(next_states)

        # TODO 2:
        # 找到每个样本中最大的 Q 值
        # 提示：max(dim=1).values
        # dim=1 表示沿每行的动作维度取最大值，结果形状从 (2,4) 变成 (2,)。
        # 形状：(batch_size,)
        next_q = target_q_values.max(dim=1).values

        # TODO 3:
        # 使用 Bellman 方程计算 TD Target
        # 记得使用 dones 掩码
        td_target = rewards + self.gamma * (1 - dones.float()) * next_q

        return td_target
