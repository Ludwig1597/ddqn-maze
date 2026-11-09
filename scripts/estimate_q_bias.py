
"""
文件: scripts/estimate_q_bias.py
用途: Q 值高估分析工具
"""

import numpy as np
import torch

from envs.maze import CoinMaze


@torch.no_grad()
def compute_max_q(agent, states):
    """
    states: Tensor, shape (batch_size, 75)
    返回: Tensor, shape (batch_size,)
    """
    q_values = agent.online(states)
    return q_values.max(dim=1).values


def capture_snapshot(env):
    """
    保存 CoinMaze 当前状态。
    """
    if env.coin is None:
        raise ValueError("Call env.reset() before capturing a snapshot")

    return {
        "size": env.size,
        "max_steps": env.max_steps,
        "pos": env.pos.copy(),
        "coin": tuple(env.coin),
        "traps": env.traps.copy(),
        "t": env.t,
    }


def restore_snapshot(snapshot):
    """
    根据快照创建独立的 CoinMaze 环境。
    当前只恢复同一回合的状态，不恢复后续 reset 的 RNG 序列。
    """
    env = CoinMaze(
        size=snapshot["size"],
        max_steps=snapshot["max_steps"],
        seed=0,
    )

    # 不调用 reset()，避免重新生成金币和陷阱
    env.pos = np.array(snapshot["pos"], dtype=int).copy()
    env.coin = tuple(snapshot["coin"])
    env.traps = snapshot["traps"].copy()
    env.t = int(snapshot["t"])

    return env
