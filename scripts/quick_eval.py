from train import train
from evaluate import evaluate


def main():
    # 训练50回合
    agent = train(n_episodes=50, seed=0)

    # 使用训练后的网络进行独立评估
    result = evaluate(agent, episodes=20, seed=999)

    print("\n===== Evaluation Summary =====")
    print(result)
    print("\n")


if __name__ == "__main__":
    main()
