"""
文件: scripts/track_delta_training.py
用途: 训练过程中每1000步测量 DQN/DDQN 的 Q 值偏差
运行:
    python -m scripts.track_delta_training --smoke
    python -m scripts.track_delta_training --algorithm ddqn --seed 0
    python -m scripts.track_delta_training --all
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from train import train
from scripts.sample_eval_states import load_eval_snapshots
from scripts.compare_q_bias import (
    measure_agent_bias,
    summarize_bias,
)

FIELDS = [
    "algorithm",
    "seed",
    "global_steps",
    "episode",
    "n_states",
    "mean_max_q",
    "mean_mc_return",
    "mean_delta",
    "std_delta",
    "mean_abs_delta",
    "positive_fraction",
]


def make_monitor(snapshots, output_path):
    """创建一个训练监控回调。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")

    # 先写表头；每次测量后立即落盘
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

    def monitor_fn(agent, global_steps, episode, algorithm, seed):
        # 计算固定状态集上的 Q 估计与 MC 回报
        rows = measure_agent_bias(
            agent,
            snapshots,
            full_30=False,
        )

        stats = summarize_bias(rows)

        record = {
            "algorithm": algorithm,
            "seed": seed,
            "global_steps": global_steps,
            "episode": episode,
            **stats,
        }

        # 训练中及时保存，避免中断后全部丢失
        with output_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writerow(record)

        assert np.isfinite(record["mean_delta"])

        print(
            f"\n[Delta] {algorithm.upper()} "
            f"seed={seed} steps={global_steps} "
            f"Q={stats['mean_max_q']:.4f} "
            f"MC={stats['mean_mc_return']:.4f} "
            f"Delta={stats['mean_delta']:.4f}\n"
        )

        return record

    return monitor_fn


def run_one(algorithm, seed, episodes, interval, smoke):
    snapshots = load_eval_snapshots()

    assert len(snapshots) == 200, f"Expected 200 snapshots, got {len(snapshots)}"

    if smoke:
        snapshots = snapshots[:2]
        tag = "delta_smoke"
    else:
        tag = "delta"

    output_path = Path("results") / (f"{algorithm}_seed{seed}_{tag}_curve.csv")

    monitor = make_monitor(
        snapshots=snapshots,
        output_path=output_path,
    )

    train(
        n_episodes=episodes,
        seed=seed,
        algorithm=algorithm,
        run_tag=tag,
        monitor_fn=monitor,
        monitor_interval_steps=interval,
        monitor_output_path=Path("results")
        / (f"{algorithm}_seed{seed}_{tag}_monitor.csv"),
    )

    print(f"Delta records saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--algorithm",
        choices=["dqn", "ddqn"],
        default="ddqn",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--interval", type=int, default=1000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--all", action="store_true")

    args = parser.parse_args()

    if args.smoke and args.all:
        parser.error("--smoke and --all cannot be combined")

    if args.smoke:
        run_one(
            algorithm=args.algorithm,
            seed=args.seed,
            episodes=50,
            interval=100,
            smoke=True,
        )
    elif args.all:
        for algorithm in ["dqn", "ddqn"]:
            for seed in range(5):
                run_one(
                    algorithm,
                    seed,
                    episodes=args.episodes,
                    interval=args.interval,
                    smoke=False,
                )
    else:
        run_one(
            algorithm=args.algorithm,
            seed=args.seed,
            episodes=args.episodes,
            interval=args.interval,
            smoke=False,
        )


if __name__ == "__main__":
    main()
