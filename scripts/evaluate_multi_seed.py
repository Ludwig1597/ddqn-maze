from scripts.evaluate_checkpoint import evaluate_checkpoint
import numpy as np
import csv
from pathlib import Path


def evaluate_multi_seed():
    results = []
    for seed in range(5):
        checkpoint_path = f"checkpoints/ddqn_seed{seed}_best.pt"

        # TODO 1:
        # 调用 evaluate_checkpoint()
        # episodes=200, seed=2026
        metrics = evaluate_checkpoint(checkpoint_path, episodes=200, seed=2026)

        # TODO 2:
        # 保存每个 Seed 的评估结果
        results.append(
            {
                "seed": seed,
                "success_rate": metrics["success_rate"],
                "avg_return": metrics["avg_return"],
                "avg_steps": metrics["avg_steps"],
            }
        )

    # TODO 3:
    # 提取五个模型的 success_rate
    success_rates = []
    returns = []
    steps = []

    for result in results:
        success_rates.append(result["success_rate"])
        returns.append(result["avg_return"])
        steps.append(result["avg_steps"])

    output_path = Path("results/ddqn_final_eval.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["seed", "success_rate", "avg_return", "avg_steps"]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved results to {output_path}")

    # TODO 4:
    # 计算均值与样本标准差
    mean_rate = np.mean(success_rates)
    std_rate = np.std(success_rates, ddof=1)

    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)

    mean_steps = np.mean(steps)
    std_steps = np.std(steps, ddof=1)

    # TODO 5:
    # 输出五种子对比结果
    print("\n===== Multi-Seed Evaluation =====")
    for result in results:
        print(
            f"Seed {result['seed']}: "
            f"Success Rate = {result['success_rate']:.4f}|"
            f"Avg Return = {result['avg_return']:.4f}|"
            f"Avg Steps = {result['avg_steps']:.2f}|"
        )
    # 输出五个Seed的统计汇总
    print("\n===== Summary(Mean ± Std) =====")
    print(f"Success Rate: {mean_rate:.4f} ± {std_rate:.4f}")
    print(f"Avg Return:   {mean_return:.4f} ± {std_return:.4f}")
    print(f"Avg Steps:    {mean_steps:.2f} ± {std_steps:.2f}")


if __name__ == "__main__":
    evaluate_multi_seed()
