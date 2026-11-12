"""
文件: scripts/run_target_ablation.py
用途: DDQN Target Network 消融实验

运行:
python -m scripts.run_target_ablation --smoke
python -m scripts.run_target_ablation --seed 0
python -m scripts.run_target_ablation --all
"""

import argparse
import csv
from pathlib import Path

from train import train

EXPERIMENT = "no_target"
SEEDS = range(5)
N_EPISODES = 2000
TARGET_SYNC_INTERVAL = 1


def run_one(seed, episodes=2000, smoke=False):
    if seed not in SEEDS:
        raise ValueError("seed must be 0, 1, 2, 3 or 4")

    tag = "ablation_no_target_smoke" if smoke else "ablation_no_target"

    run_name = f"ddqn_seed{seed}_{tag}"

    # 沿用 train.py 的输出路径命名规则
    paths = [
        Path("logs") / f"{run_name}.csv",
        Path("logs") / f"{run_name}_eval.csv",
        Path("checkpoints") / f"{run_name}_best.pt",
    ]

    # 防止中途发现旧文件，误覆盖实验
    existing = [str(p) for p in paths if p.exists()]
    if existing:
        print(f"[SKIP] {run_name}: outputs already exist")
        for p in existing:
            print(f"  {p}")
        return "skipped"

    print("=" * 60)
    print("Experiment:", EXPERIMENT)
    print("Algorithm: DDQN")
    print("Seed:", seed)
    print("Episodes:", episodes)
    print("Target sync interval:", TARGET_SYNC_INTERVAL)
    print("Replay capacity: 50000 (unchanged)")
    print("=" * 60)

    train(
        n_episodes=episodes,
        seed=seed,
        algorithm="ddqn",
        run_tag=tag,
        target_sync_interval=TARGET_SYNC_INTERVAL,
    )

    # 最基本的交付文件检查
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Expected output missing: {p}")

    eval_path = paths[1]
    with eval_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != episodes // 50:
        raise AssertionError(
            f"Expected {episodes // 50} evaluations, " f"got {len(rows)}"
        )

    print(f"[PASS] {run_name}")
    return "completed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke and args.all:
        parser.error("--smoke and --all cannot be combined")

    if args.smoke:
        run_one(seed=args.seed, episodes=50, smoke=True)
    elif args.all:
        for seed in SEEDS:
            run_one(seed=seed, episodes=N_EPISODES)
    else:
        run_one(seed=args.seed, episodes=N_EPISODES)


if __name__ == "__main__":
    main()
