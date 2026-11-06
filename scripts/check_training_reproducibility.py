import csv
import numpy as np

from pathlib import Path
from train import train


def read_csv(path):
    # TODO 1:
    # 使用with打开csv文件
    # 使用csv.DictReader读取内容
    # 返回所有行组成的列表
    # newline=""	让 CSV 模块自行处理换行
    with open(path, "r", newline="", encoding="utf-8") as f:
        # 使用DictReader读取CSV
        # DictWriter：把字典写入 CSV。
        # DictReader：把 CSV 读取成字典。
        reader = csv.DictReader(f)
        # 将所有行转换成列表
        """list() 会把所有行收集到一个列表中。
        rows = [
            {"episode": "1", "return": "0.8", "success": "1", "steps": "20"},
            {"episode": "2", "return": "-1.2", "success": "0", "steps": "50"},
            {"episode": "3", "return": "0.5", "success": "1", "steps": "30"}
        ]
        
        Note:  CSV 读取出来的数值都是字符串。
        """
        rows = list(reader)
    # 返回所有记录
    return rows


def compare_column(rows1, rows2, column):
    if len(rows1) != len(rows2):
        return False
    # TODO 1:
    # 从rows1中提取column对应的所有数据
    # 将字符串转换为float
    # 如果遇到空字符串，转换成np。nan
    values1 = []
    for row in rows1:
        value = row[column]
        if value == "":
            values1.append(np.nan)  # np.nan表示一个非数值
        else:
            values1.append(float(value))
    # TODO 2:
    # 对rows2做相同操作
    values2 = []
    for row in rows2:
        value = row[column]
        if value == "":
            values2.append(np.nan)
        else:
            values2.append(float(value))

    # TODO 3:
    # 使用np.allclose()比较两个数组
    # equal_nan=True
    # atol=1e-8
    # rtol=1e-7
    # np.allclose()意味着我们允许非常小的浮点误差。
    # fmt:off
    return np.allclose(
        values1, 
        values2, 
        equal_nan=True, 
        atol=1e-8, 
        rtol=1e-7
    )
    # fmt:on


if __name__ == "__main__":
    seed = 42
    # ===== Run1 =====
    print("\n===== Training Run 1 =====")
    train(n_episodes=50, seed=seed)

    # 保存第一次训练的数据到内存
    run1_train = read_csv(Path(f"logs/ddqn_seed{seed}.csv"))
    run1_eval = read_csv(Path(f"logs/ddqn_seed{seed}_eval.csv"))

    # ===== Run2 =====
    print("\n===== Training Run 2 =====")
    train(n_episodes=50, seed=seed)
    # 读取第二次训练的数据
    run2_train = read_csv(Path(f"logs/ddqn_seed{seed}.csv"))
    run2_eval = read_csv(Path(f"logs/ddqn_seed{seed}_eval.csv"))

    # ===== Compare Training Results =====
    print("\n===== Training Reproducibility =====")
    # fmt:off
    train_columns=[
        "episode",
        "return",
        "success",
        "steps",
        "epsilon",
        "loss"
    ]
    # fmt:on
    training_passed = True
    for column in train_columns:
        matched = compare_column(run1_train, run2_train, column)
        print(f"{column}: {'Match' if matched else 'MISMATCH'}")
        if not matched:
            training_passed = False
    # ===== Compare Evaluation Results =====
    print("\n===== Evaluation Reproducibility ======")
    # fmt:off
    eval_columns=[
        "episode",
        "success_rate",
        "avg_return",
        "avg_steps"
    ]
    # fmt:on
    evaluation_passed = True
    for column in eval_columns:
        matched = compare_column(run1_eval, run2_eval, column)
        print(f"{column}: {'Match' if matched else 'MISMATCH'}")
        if not matched:
            evaluation_passed = False
    # ===== Final Result =====
    if training_passed and evaluation_passed:
        print("\nTraining reproducibility PASSED")
    else:
        print("\nTraining reproducibility FAILED")
    # 打印数据规模
    print("\n===== Data Summary =====")
    print(f"Run 1 training: {len(run1_train)}")
    print(f"Run 2 training: {len(run2_train)}")
    print(f"Run 1 evaluation: {len(run1_eval)}")
    print(f"Run 2 evaluation: {len(run2_eval)}")
