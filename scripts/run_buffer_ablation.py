"""
文件: scripts/run_buffer_ablation.py
用途: DDQN Replay Buffer 容量消融实验

对比:
    Baseline: 50000
    Ablation: 1000

运行:
    python -m scripts.run_buffer_ablation --smoke
    python -m scripts.run_buffer_ablation --seed 0
    python -m scripts.run_buffer_ablation --all
"""

import argparse
import csv
from pathlib import Path

from train import train

REPLAY_CAPACITY = 1000
SEEDS = range(5)
N_EPISODES = 2000


def run_one(seed, episodes=2000, smoke=False):
    if seed not in SEEDS:
        raise ValueError("seed must be 0, 1, 2, 3 or 4")

    tag = "ablation_small_buffer_smoke" if smoke else "ablation_small_buffer"

    run_name = f"ddqn_seed{seed}_{tag}"

    paths = [
        Path("logs") / f"{run_name}.csv",
        Path("logs") / f"{run_name}_eval.csv",
        Path("checkpoints") / f"{run_name}_best.pt",
    ]

    existing = [str(p) for p in paths if p.exists()]
    if existing:
        print(f"[SKIP] {run_name}: outputs already exist")
        for p in existing:
            print(f"  {p}")
        return "skipped"

    print("=" * 60)
    print("Experiment: Small Replay Buffer")
    print("Algorithm: DDQN")
    print("Seed:", seed)
    print("Episodes:", episodes)
    print("Replay capacity:", REPLAY_CAPACITY)
    print("Warmup: 1000")
    print("Batch size: 64")
    print("Target sync interval: 500")
    print("=" * 60)

    train(
        n_episodes=episodes,
        seed=seed,
        algorithm="ddqn",
        run_tag=tag,
        replay_capacity=REPLAY_CAPACITY,
        target_sync_interval=500,
    )

    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Expected output missing: {p}")

    with paths[0].open(newline="", encoding="utf-8") as f:
        train_rows = list(csv.DictReader(f))

    with paths[1].open(newline="", encoding="utf-8") as f:
        eval_rows = list(csv.DictReader(f))

    if len(train_rows) != episodes:
        raise AssertionError(f"Expected {episodes} episodes, got {len(train_rows)}")

    if len(eval_rows) != episodes // 50:
        raise AssertionError(
            f"Expected {episodes // 50} evaluations, " f"got {len(eval_rows)}"
        )

    print(f"[PASS] {run_name}")
    return "completed"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--smoke", action="store_true")

    args = parser.parse_args()

    if args.all and args.smoke:
        parser.error("--all and --smoke cannot be combined")

    if args.smoke:
        run_one(
            seed=args.seed,
            episodes=50,
            smoke=True,
        )

    elif args.all:
        for seed in SEEDS:
            run_one(seed=seed, episodes=N_EPISODES)

    else:
        run_one(seed=args.seed, episodes=N_EPISODES)


if __name__ == "__main__":
    main()
