"""
文件: scripts/compare_q_bias.py
用途: 比较 DQN 和 DDQN 最佳模型的 Q 值估计偏差
运行:
    python -m scripts.compare_q_bias --smoke
    python -m scripts.compare_q_bias
    python -m scripts.compare_q_bias --full-30
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from scripts.estimate_q_bias import (
    compute_max_q,
    restore_snapshot,
)
from scripts.estimate_mc_return import (
    rollout_once,
    estimate_mc_return,
    load_trained_agent,
)
from scripts.sample_eval_states import load_eval_snapshots

GAMMA = 0.95
N_ROLLOUTS = 30

STATE_FIELDS = [
    "algorithm",
    "seed",
    "state_id",
    "pred_max_q",
    "mc_return",
    "delta",
    "steps",
    "outcome",
]

SUMMARY_FIELDS = [
    "algorithm",
    "seed",
    "n_states",
    "mean_max_q",
    "mean_mc_return",
    "mean_delta",
    "std_delta",
    "mean_abs_delta",
    "positive_fraction",
    "actual_rollouts_per_state",
]


def measure_agent_bias(agent, snapshots, gamma=GAMMA, full_30=False):
    """
    计算一个 Agent 在固定状态集上的估计偏差。

    full_30=False:
        确定性环境下，每状态只实际执行一次。

    full_30=True:
        每状态实际执行 30 次 Rollout。
    """

    if not snapshots:
        raise ValueError("No evaluation snapshots")

    # 从完整状态快照生成网络需要的局部观测
    states = np.stack([restore_snapshot(s).obs() for s in snapshots]).astype(np.float32)

    states_tensor = torch.from_numpy(states)

    # (N, 75) -> (N,)
    predicted_q = compute_max_q(agent, states_tensor).cpu().numpy()

    assert predicted_q.shape == (len(snapshots),)
    assert np.isfinite(predicted_q).all()

    rows = []

    for state_id, (snapshot, max_q) in enumerate(zip(snapshots, predicted_q)):
        if full_30:
            result = estimate_mc_return(
                agent,
                snapshot,
                n_rollouts=N_ROLLOUTS,
                gamma=gamma,
            )

            mc_return = result["mean_return"]
            steps = result["steps"][0]

            # 当前环境确定性，重复结果应相同
            assert np.allclose(
                result["returns"],
                result["returns"][0],
                atol=1e-10,
            )

            outcome = next(
                name for name, count in result["outcomes"].items() if count > 0
            )

        else:
            # 与30次确定性重复的均值等价
            result = rollout_once(agent, snapshot, gamma=gamma)

            mc_return = result["discounted_return"]
            steps = result["steps"]
            outcome = result["outcome"]

        delta = float(max_q) - float(mc_return)

        rows.append(
            {
                "state_id": state_id,
                "pred_max_q": float(max_q),
                "mc_return": float(mc_return),
                "delta": delta,
                "steps": steps,
                "outcome": outcome,
            }
        )

    return rows


def summarize_bias(rows):
    """
    汇总一个模型在所有评估状态上的结果。
    """

    predicted = np.array(
        [r["pred_max_q"] for r in rows],
        dtype=np.float64,
    )

    mc_returns = np.array(
        [r["mc_return"] for r in rows],
        dtype=np.float64,
    )

    deltas = predicted - mc_returns

    return {
        "n_states": len(rows),
        "mean_max_q": float(np.mean(predicted)),
        "mean_mc_return": float(np.mean(mc_returns)),
        "mean_delta": float(np.mean(deltas)),
        "std_delta": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
        "mean_abs_delta": float(np.mean(np.abs(deltas))),
        "positive_fraction": float(np.mean(deltas > 0)),
    }


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Only evaluate Seed 0 on two states",
    )

    parser.add_argument(
        "--full-30",
        action="store_true",
        help="Actually execute 30 rollouts per state",
    )

    args = parser.parse_args()

    snapshots = load_eval_snapshots()

    if len(snapshots) != 200:
        raise ValueError(f"Expected 200 snapshots, got {len(snapshots)}")

    seeds = range(1) if args.smoke else range(5)

    if args.smoke:
        snapshots = snapshots[:2]

    suffix = "_smoke" if args.smoke else "_final"

    summary_path = Path(f"results/q_bias{suffix}_summary.csv")

    states_path = Path(f"results/q_bias{suffix}_states.csv")

    # 防止重复运行覆盖之前的正式实验结果
    for path in [summary_path, states_path]:
        if path.exists():
            raise FileExistsError(
                f"{path} already exists. " "Move or rename it before rerunning."
            )

    all_states = []
    all_summary = []

    actual_rollouts = N_ROLLOUTS if args.full_30 else 1

    print("===== Q Bias Comparison =====")
    print(f"States per model: {len(snapshots)}")
    print(f"Actual rollouts per state: {actual_rollouts}")

    for algorithm in ["dqn", "ddqn"]:
        for seed in seeds:
            checkpoint_path = Path(f"checkpoints/{algorithm}_seed{seed}_best.pt")

            print(f"\nEvaluating {algorithm.upper()} Seed {seed}")

            agent = load_trained_agent(
                checkpoint_path,
                algorithm=algorithm,
            )

            rows = measure_agent_bias(
                agent,
                snapshots,
                full_30=args.full_30,
            )

            stats = summarize_bias(rows)

            for row in rows:
                all_states.append(
                    {
                        "algorithm": algorithm,
                        "seed": seed,
                        **row,
                    }
                )

            all_summary.append(
                {
                    "algorithm": algorithm,
                    "seed": seed,
                    **stats,
                    "actual_rollouts_per_state": actual_rollouts,
                }
            )

            print(
                f"Mean Max Q: {stats['mean_max_q']:.4f} | "
                f"Mean MC: {stats['mean_mc_return']:.4f} | "
                f"Delta: {stats['mean_delta']:.4f}"
            )

    write_csv(summary_path, all_summary, SUMMARY_FIELDS)

    write_csv(states_path, all_states, STATE_FIELDS)

    print("\n===== Completed =====")
    print(f"Models: {len(all_summary)}")
    print(f"State records: {len(all_states)}")
    print(f"Summary: {summary_path}")
    print(f"Details: {states_path}")


if __name__ == "__main__":
    main()
