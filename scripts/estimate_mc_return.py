
"""
文件: scripts/estimate_mc_return.py
用途: 从固定环境快照执行贪心策略 Rollout
      估计 Monte Carlo 折扣回报。
运行: python -m scripts.estimate_mc_return
"""

from pathlib import Path

import numpy as np
import torch

from agents.ddqn import DDQNAgent
from agents.dqn import DQNAgent

from scripts.estimate_q_bias import restore_snapshot
from scripts.sample_eval_states import load_eval_snapshots


GAMMA = 0.95
N_ROLLOUTS = 30


def rollout_once(agent, snapshot, gamma=GAMMA):
    """
    从一个完整环境快照出发，执行一次贪心 Rollout。

    返回：
        discounted_return: 折扣回报
        steps: 本次 Rollout 的交互步数
        outcome: coin / trap / timeout
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be between 0 and 1")

    # 每次都恢复一个独立环境
    env = restore_snapshot(snapshot)

    if (
        env.t >= env.max_steps
        or tuple(env.pos) == env.coin
        or tuple(env.pos) in env.traps
    ):
        raise ValueError("Snapshot must be a non-terminal state")

    state = env.obs()

    discounted_return = 0.0
    discount = 1.0
    steps = 0
    done = False

    while not done:

        # 固定使用贪心策略，不进行随机探索
        action = agent.act(state, epsilon=0.0)

        next_state, reward, done, _ = env.step(action)

        # G = r1 + gamma*r2 + gamma^2*r3 + ...
        discounted_return += discount * float(reward)

        discount *= gamma
        steps += 1
        state = next_state

    if tuple(env.pos) == env.coin:
        outcome = "coin"
    elif tuple(env.pos) in env.traps:
        outcome = "trap"
    else:
        outcome = "timeout"

    return {
        "discounted_return": float(discounted_return),
        "steps": steps,
        "outcome": outcome,
    }


def estimate_mc_return(
    agent,
    snapshot,
    n_rollouts=N_ROLLOUTS,
    gamma=GAMMA,
):
    """
    从同一个快照恢复环境，重复 n_rollouts 次。

    返回 Monte Carlo 折扣回报的均值与标准差。
    """
    if n_rollouts <= 0:
        raise ValueError("n_rollouts must be positive")

    rollouts = [
        rollout_once(agent, snapshot, gamma=gamma)
        for _ in range(n_rollouts)
    ]

    returns = np.array(
        [item["discounted_return"] for item in rollouts],
        dtype=np.float64,
    )

    steps = [item["steps"] for item in rollouts]

    outcomes = {
        "coin": sum(
            item["outcome"] == "coin" for item in rollouts
        ),
        "trap": sum(
            item["outcome"] == "trap" for item in rollouts
        ),
        "timeout": sum(
            item["outcome"] == "timeout" for item in rollouts
        ),
    }

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "returns": returns.tolist(),
        "steps": steps,
        "outcomes": outcomes,
    }


def load_trained_agent(checkpoint_path, algorithm="ddqn"):
    """
    加载已训练好的最佳 Checkpoint。

    仅用于推理，不继续训练。
    """
    if algorithm == "ddqn":
        agent = DDQNAgent()
    elif algorithm == "dqn":
        agent = DQNAgent()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    agent.online.load_state_dict(
        checkpoint["online_state_dict"]
    )

    agent.sync_target()
    agent.online.eval()
    agent.target.eval()

    return agent


def main():
    checkpoint_path = Path(
        "checkpoints/ddqn_seed0_best.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    agent = load_trained_agent(
        checkpoint_path,
        algorithm="ddqn",
    )

    snapshots = load_eval_snapshots()

    assert len(snapshots) == 200

    print("===== Monte Carlo Rollout Smoke Test =====")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Available snapshots: {len(snapshots)}")
    print(f"Rollouts per state: {N_ROLLOUTS}")

    # 本次仅验证前 2 个状态，不执行完整偏差实验
    for index, snapshot in enumerate(snapshots[:2]):

        result = estimate_mc_return(
            agent,
            snapshot,
            n_rollouts=N_ROLLOUTS,
            gamma=GAMMA,
        )

        print(f"\nState {index}")
        print(f"MC mean return: {result['mean_return']:.6f}")
        print(f"MC std return: {result['std_return']:.6f}")
        print(f"Steps: {result['steps'][0]}")
        print(f"Outcomes: {result['outcomes']}")

        # 当前环境及贪心策略都是确定性的
        assert np.allclose(
            result["returns"],
            result["returns"][0],
            atol=1e-10,
        )

    print("\nMC Rollout smoke test PASSED!")


if __name__ == "__main__":
    main()
