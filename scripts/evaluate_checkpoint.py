"""_summary_
duty:
1. 加载训练好的最佳 Checkpoint。
2. 创建 DDQN Agent。
3. 恢复 Online Network 参数。
4. 同步 Target Network。
5. 使用 100 个 Episode 进行独立评估。
6. 输出成功率、平均回报和平均步数。
"""

import torch

from agents.ddqn import DDQNAgent
from evaluate import evaluate


def evaluate_checkpoint(
    checkpoint_path="checkpoints/ddqn_seed0_best.pt", episodes=100, seed=2026
):

    # TODO 1:
    # 加载 Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # TODO 2:
    # 创建 DDQNAgent
    agent = DDQNAgent()

    """_summary_
    为什么需要加载Online Network参数并且同步给Target Network

    因为 Checkpoint 只保存了 Online Network 的参数。

    新建 Agent 时：
    - Online Network 是随机初始化的。
    - Target Network 也是随机初始化的。
    第一步恢复 Online Network, 第二步将 Online Network 的参数复制给 Target Network。

    这样两个网络就恢复到了正确的初始同步状态。
    """
    # TODO 3:
    # 加载 Online Network 参数
    agent.online.load_state_dict(checkpoint["online_state_dict"])

    # TODO 4:
    # 同步 Target Network
    agent.sync_target()

    # TODO 5:
    # 打印 Checkpoint 信息
    print("Checkpoint episode:", checkpoint["episode"])
    print("Best checkpoint rate:", checkpoint["eval_success_rate"])

    # TODO 6:
    # 执行独立评估
    result = evaluate(agent, episodes=episodes, seed=seed)

    print("\n===== Independent Evaluation =====")
    print(result)

    return result


if __name__ == "__main__":
    evaluate_checkpoint(episodes=100, seed=2026)
