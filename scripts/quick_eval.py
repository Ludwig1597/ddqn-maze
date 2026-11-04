from train import train
from evaluate import evaluate


def main():
    # 训练50回合
    agent = train(n_episodes=50, seed=0)

    # 实验A：完全随机策略
    random_result = evaluate(agent, episodes=200, seed=999, epsilon=1.0)
    assert sum(random_result["outcomes"].values()) == 200
    assert random_result["outcomes"]["coin"] / 200 == random_result["success_rate"]

    # 实验B：训练后的贪心策略
    greedy_reuslt = evaluate(agent, episodes=200, seed=999, epsilon=0.0)
    assert sum(greedy_reuslt["outcomes"].values()) == 200
    assert greedy_reuslt["outcomes"]["coin"] / 200 == greedy_reuslt["success_rate"]

    print("\n===== Random Policy =====")
    print(random_result)

    print("\n===== Trained DDQN Greedy Policy =====")
    print(greedy_reuslt)

    # 使用训练后的网络进行独立评估
    # result = evaluate(agent, episodes=20, seed=999)

    # print("\n===== Evaluation Summary =====")
    # print(result)
    # print("\n")


if __name__ == "__main__":
    main()
