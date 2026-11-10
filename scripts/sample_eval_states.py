"""
文件: scripts/sample_eval_states.py
用途: 生成、保存和加载 200 个固定的 Q 值评估状态
运行: python -m scripts.sample_eval_states

这里特意保存完整环境状态，而不只是 75 维观测。后续两个算法读取同一个 JSON, 就能在相同金币、陷阱、位置和剩余步数条件下进行评估。
"""

import json
from pathlib import Path

import numpy as np

from envs.maze import CoinMaze
from scripts.estimate_q_bias import (
    capture_snapshot,
    restore_snapshot,
)

N_STATES = 200
SAMPLE_STRIDE = 20
SAMPLE_SEED = 4242

OUTPUT_PATH = Path("data/q_bias_eval_states.json")


def generate_eval_snapshots(
    n_states=N_STATES,
    stride=SAMPLE_STRIDE,
    seed=SAMPLE_SEED,
):
    """
    使用均匀随机策略，每隔 stride 个环境步保存一次状态。

    返回包含 n_states 个快照的列表。
    """

    if n_states <= 0 or stride <= 0:
        raise ValueError("n_states and stride must be positive")

    env = CoinMaze(seed=seed)

    # 动作随机性与迷宫重置随机性分开管理
    action_rng = np.random.default_rng(seed + 1)

    env.reset()

    snapshots = []
    global_steps = 0

    while len(snapshots) < n_states:

        # 在动作执行前采样，保证是可以继续交互的状态
        if global_steps % stride == 0:
            snapshot = capture_snapshot(env)
            snapshots.append(snapshot)

            if len(snapshots) == n_states:
                break

        # 随机选择上、下、左、右之一
        action = int(action_rng.integers(0, 4))

        _, _, done, _ = env.step(action)

        global_steps += 1

        # 终止后进入新回合，继续收集状态
        if done:
            env.reset()

    return snapshots


def snapshot_to_record(snapshot):
    """
    将快照转换为可保存到 JSON 的 Python 字典。

    NumPy array -> list
    tuple       -> list
    set         -> 排序后的 list
    """
    return {
        "size": int(snapshot["size"]),
        "max_steps": int(snapshot["max_steps"]),
        "pos": [int(x) for x in snapshot["pos"]],
        "coin": [int(x) for x in snapshot["coin"]],
        "traps": [[int(r), int(c)] for r, c in sorted(snapshot["traps"])],
        "t": int(snapshot["t"]),
    }


def record_to_snapshot(record):
    """
    将 JSON 记录重新转换为 restore_snapshot() 接受的格式。
    """
    return {
        "size": int(record["size"]),
        "max_steps": int(record["max_steps"]),
        "pos": np.array(record["pos"], dtype=int),
        "coin": tuple(record["coin"]),
        "traps": {tuple(pos) for pos in record["traps"]},
        "t": int(record["t"]),
    }


def save_eval_snapshots(
    snapshots,
    path=OUTPUT_PATH,
    seed=SAMPLE_SEED,
    stride=SAMPLE_STRIDE,
):
    """
    保存状态集及其采样配置。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "seed": seed,
        "sampling_policy": "uniform_random",
        "stride_steps": stride,
        "n_states": len(snapshots),
        "snapshots": [snapshot_to_record(s) for s in snapshots],
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            indent=2,
            ensure_ascii=False,
        )


def load_eval_snapshots(path=OUTPUT_PATH):
    """
    从 JSON 文件加载全部环境快照。
    """
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    snapshots = [record_to_snapshot(record) for record in payload["snapshots"]]

    if len(snapshots) != payload["n_states"]:
        raise ValueError("Snapshot count mismatch")

    return snapshots


def main():
    snapshots = generate_eval_snapshots()

    assert len(snapshots) == N_STATES

    save_eval_snapshots(snapshots)

    loaded = load_eval_snapshots()

    assert len(loaded) == N_STATES

    # 验证所有快照都可以恢复为有效观测
    for snapshot in loaded:
        env = restore_snapshot(snapshot)

        assert env.obs().shape == (75,)
        assert 0 <= env.t < env.max_steps

    # 统计涉及多少种不同的金币/陷阱布局
    unique_maps = {
        (
            snapshot["coin"],
            tuple(sorted(snapshot["traps"])),
        )
        for snapshot in loaded
    }

    print("===== Fixed Evaluation States =====")
    print(f"States: {len(loaded)}")
    print(f"Sampling seed: {SAMPLE_SEED}")
    print(f"Sampling stride: {SAMPLE_STRIDE}")
    print(f"Unique maps: {len(unique_maps)}")
    print(f"Output: {OUTPUT_PATH}")
    print("All evaluation states saved successfully!")


if __name__ == "__main__":
    main()
