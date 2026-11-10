"""
文件: tests/test_q_bias_plot.py
用途: 测试 Q 值偏差图的数据读取与生成
运行: python -m tests.test_q_bias_plot
"""

import csv
import tempfile
from pathlib import Path

import numpy as np

from scripts.plot_q_bias import (
    load_bias_summary,
    plot_final_q_bias,
)


def test_load_bias_summary():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "summary.csv"

        rows = []

        for algorithm, values in [
            ("dqn", [0.1, 0.2, 0.3, 0.4, 0.5]),
            ("ddqn", [0.0, 0.1, 0.2, 0.3, 0.4]),
        ]:
            for seed, value in enumerate(values):
                rows.append(
                    {
                        "algorithm": algorithm,
                        "seed": seed,
                        "n_states": 200,
                        "mean_delta": value,
                    }
                )

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "algorithm",
                    "seed",
                    "n_states",
                    "mean_delta",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        data = load_bias_summary(path)

        assert data["dqn"].shape == (5,)
        assert data["ddqn"].shape == (5,)

        assert np.allclose(
            data["dqn"],
            [0.1, 0.2, 0.3, 0.4, 0.5],
        )

        assert np.allclose(
            data["ddqn"],
            [0.0, 0.1, 0.2, 0.3, 0.4],
        )

    print("test_load_bias_summary PASSED")


def test_plot_final_q_bias():
    data = {
        "dqn": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        "ddqn": np.array([0.0, 0.1, 0.2, 0.3, 0.4]),
    }

    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "plot.png"

        stats = plot_final_q_bias(data, output_path=output)

        assert output.exists()
        assert output.stat().st_size > 0

        assert np.isclose(stats["dqn_mean"], 0.3)

        assert np.isclose(stats["ddqn_mean"], 0.2)

    print("test_plot_final_q_bias PASSED")


if __name__ == "__main__":
    test_load_bias_summary()
    test_plot_final_q_bias()

    print("\nAll Q-bias plot tests PASSED!")
