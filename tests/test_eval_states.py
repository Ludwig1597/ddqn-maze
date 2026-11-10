"""
文件: tests/test_eval_states.py
用途: 验证固定评估状态集的数量、可复现性和保存恢复
运行: python -m tests.test_eval_states
"""

import tempfile
from pathlib import Path

import numpy as np

from scripts.estimate_q_bias import restore_snapshot
from scripts.sample_eval_states import (
    generate_eval_snapshots,
    load_eval_snapshots,
    save_eval_snapshots,
    snapshot_to_record,
)


def test_sample_count():
    snapshots = generate_eval_snapshots(
        n_states=200,
        stride=20,
        seed=4242,
    )

    assert len(snapshots) == 200

    for snapshot in snapshots:
        env = restore_snapshot(snapshot)

        assert env.obs().shape == (75,)
        assert 0 <= env.t < env.max_steps
        assert len(env.traps) == 3

    print("test_sample_count PASSED")


def test_sampling_reproducibility():
    samples1 = generate_eval_snapshots(n_states=12, stride=20, seed=4242)

    samples2 = generate_eval_snapshots(n_states=12, stride=20, seed=4242)

    records1 = [snapshot_to_record(s) for s in samples1]

    records2 = [snapshot_to_record(s) for s in samples2]

    assert records1 == records2

    print("test_sampling_reproducibility PASSED")


def test_json_roundtrip():
    snapshots = generate_eval_snapshots(n_states=12, stride=20, seed=4242)

    # 临时目录避免测试污染正式数据文件
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "eval_states.json"

        save_eval_snapshots(
            snapshots,
            path=path,
            seed=4242,
            stride=20,
        )

        loaded = load_eval_snapshots(path)

        assert len(loaded) == len(snapshots)

        for original, recovered in zip(snapshots, loaded):
            assert snapshot_to_record(original) == snapshot_to_record(recovered)

            env1 = restore_snapshot(original)
            env2 = restore_snapshot(recovered)

            assert np.array_equal(env1.obs(), env2.obs())

            # 恢复后执行相同动作，结果应相同
            obs1, reward1, done1, _ = env1.step(3)
            obs2, reward2, done2, _ = env2.step(3)

            assert np.array_equal(obs1, obs2)
            assert reward1 == reward2
            assert done1 == done2

    print("test_json_roundtrip PASSED")


if __name__ == "__main__":
    test_sample_count()
    test_sampling_reproducibility()
    test_json_roundtrip()

    print("\nAll evaluation state tests PASSED!")
