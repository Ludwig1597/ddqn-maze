# ddqn-maze
A course project for Deep Reinforcement Learning at Fudan University, applying Double Deep Q-Network (DDQN) to solve a grid-based coin collection problem.

## Overview

### ① CoinMaze 环境

第 4–8 页 · 定义状态、动作、奖励和环境交互

### ② QNet 神经网络

第 9–10 页 · 输入状态，预测四个动作的 Q 值

### ③ Replay Buffer + DDQN Agent

第 10–13 页 · 存储经验，计算 TD Target，执行 BP

### ④ 训练、评估与对照实验

第 13–19 页 · 训练智能体，比较 DQN 与 DDQN

## Details

### 实践 01：创建 DDQN 项目并初始化迷宫

**对应课程设计：第 4–7 页**：重点阅读第 2.1 节「任务场景」和第 2.4 节「环境实现与自检」，特别是第 6 页的清单 1。

**本阶段目标**: 创建 Python 项目，编写 CoinMaze 类的初始化方法 __init__()，让环境拥有地图、起点、金币候选位置、陷阱候选位置和随机数生成器。

> 本次先不实现 reset()、step() 和 75 维观测编码。  
> 完成初始化并通过测试后，再进入这些方法。

### 实践 02：实现 reset() 方法

**对应课程设计：第 6–7 页，第 2.4 节**：阅读清单 1 中的 reset() 方法。

**本阶段目标**：让迷宫能够开始一个新的回合（Episode），重置智能体位置，并随机生成金币和陷阱。

> 需要了解：NumPy 随机采样、集合推导式与随机种子的可复现性。

### 实践 03：实现迷宫的 step() 方法

**对应课程设计：第 6–7 页，第 2.4 节**：阅读第 7 页的 清单 1：环境转移逻辑，重点是 step(self, a)。

**本阶段目标**：
- 理解动作如何改变智能体位置。
- 实现边界和障碍检测。
- 根据行动结果计算奖励。
- 判断回合是否结束。
- 编写测试，验证环境转移逻辑。

> 需要了解：Python 字典映射、二维坐标、_free() 辅助方法、Termination 与 Truncation 的区别。

### 实践 04-1：实现 obs() 状态观测

**课程设计：第 7–9 页，第 3.1–3.2 节**：阅读第 8 页的「状态编码方案对比」以及「清单 2：局部视野编码」。

**本阶段目标**：实现 CoinMaze.obs()，将迷宫的局部信息转换为一个 75 维 NumPy 向量，为后续 DQN 神经网络提供输入。

> 需要了解：多维 NumPy 数组、通道（Channel）、局部坐标转换、reshape 与单元测试。

### 实践 04-2：整合 CoinMaze 接口
**课程设计：第 6–8 页，第 2.4 节与第 3.2 节**

**文档要求**： reset() 返回初始观测；step(a) 返回下一状态观测、奖励、结束标志、附加信息。

> 编写接口测试，确认 reset()、step() 真正调用 obs()，而不仅仅是 obs() 方法本身正确。

### 实践 05：正式开始实现 QNet

**课程设计：第 9–10 页，第 4.1–4.2 节**：阅读第 9 页网络结构与参数量，以及第 10 页的清单 3「Q 网络定义」。

**本阶段目标**： 让 PyTorch 神经网络接收 75 维迷宫状态，输出四个动作的 Q 值。

> 需要了解： Batch 维度、参数量计算、无激活输出层、网络前向传播测试。

### 实践 06：Replay Buffer（经验回放池）

**课程设计：第 10–14 页**：
- 第 10–11 页，第 5.1–5.2 节：DQN、经验回放和目标网络的作用。
- 第 13–14 页，第 6.1 节：训练流程中的第 5、6、7 步。

**作业原文要求**： 经验回放池容量 50,000，预热至少 1,000 条经验，每次均匀采样 64 条经验。

**本阶段目标**：独立实现能够存储与随机抽取经验的 Replay Buffer。暂时不执行 BP。

> 需要了解： deque、random.sample()、zip(*batch)、数组复制和 Batch 维度。

### 实践 07：计算 Double DQN 的 TD Target

**课程设计：第 11–13 页，第 5.3–5.4 节**：重点阅读第 12–13 页的 清单 4：DDQN 目标值计算。

**本阶段目标**： 创建 Online Network 和 Target Network，实现 DDQN 的目标 Q 值计算。

- 创建 online 和 target 两个 QNet
- 使用 load_state_dict() 同步初始化参数
- 实现 argmax() 选择动作
- 实现 gather() 评价动作
- 正确计算 done 掩码和 TD Target
- 通过手工数值测试

> 需要了解： argmax()、gather()、torch.no_grad()、Batch 维度与手工数值测试。
> 本阶段先不进行 BP。下一阶段会把 TD Target 接入 Huber Loss 和优化器。

### 实践 08：第一次真正执行 DDQN 的 BP 更新

**课程设计：第 13–15 页**：重点阅读： 第 6.1 节训练主循环的第 5–10 步，以及第 6.2 节的优化器、损失函数与梯度裁剪配置。

**本阶段目标**： 从 Replay Buffer 取出一批经验，计算预测 Q 值与 TD Target 的 Huber Loss，并让 Online QNet 更新一次参数。

![流程](./doc/img/flow.png)

> 需要了解： torch.as_tensor()、Batch 维度、SmoothL1Loss、梯度裁剪和检查参数是否真的改变。

### 实践 09：实现 ε-greedy 动作选择
**课程设计：第 14–16 页**: 建议阅读： 第 6.1 节第 3 步「选择动作」，以及第 6.3 节「探索策略与参数敏感性」。

**作业原文要求**： 使用 ε-greedy 策略，ε 从 1.0 线性下降到 0.05，衰减过程为 20,000 个全局环境步。

**本阶段目标**： 实现 agent.act(state, epsilon)，使智能体能够选择动作。

> 需要了解： Exploitation 与 Exploration、unsqueeze(0)、torch.no_grad()、动作选择的测试。

目前已经完成了 DDQN Agent 的三个重要能力：
- **预测 Q 值**：使用 Online QNet 预测四个动作的价值。
- **探索与利用**：通过 ε-greedy 在随机探索和贪心动作之间切换。
- **环境交互**：选择的动作可以交给 CoinMaze.step() 执行。

### 实践 10：Target Network 周期同步

**课程设计：第 13–15 页**：建议先读： 第 6.1 节「训练主循环」第 9–10 步，以及第 6.2 节中的目标网络配置。

**作业原文要求**： 每隔 500 个全局环境步，将 Online Network 的参数复制到 Target Network（Hard Update）。

**本次任务**： 实现一个同步方法，并验证 Target 的参数不会随 Online 的 BP 自动改变。

> 需要了解： state_dict()、load_state_dict() 与模型参数独立性的测试。

### 实践 11-1：编写 DDQN 训练主循环

**课程设计第 14–15 页 · 第 6.1–6.2 节**: 请先阅读： 第 14 页清单 5「训练主循环骨架」以及第 15 页表 9「DDQN 训练超参数基线配置」。

**作业原文要求**： 训练 2000 个 Episode，ε 从 1.0 线性下降至 0.05，每 500 个全局环境步同步 Target，每 50 回合独立评估。

**本轮目标**： 先完成训练循环，试运行 10 个 Episode，确认组件能够正确连接。

**额外补充**： 线性衰减函数、全局环境步数、训练冒烟测试（Smoke Test）。

### 实践 11-2：验证 DDQN 是否真的开始学习

**课程设计：第 14–17 页**: 建议阅读： 第 6.1 节训练主循环、第 6.2 节超参数，以及第 7.1 节评估指标。

**作业原文要求**： 预热 1000 条经验、周期同步 Target、监控回报与成功率。

**额外补充**： 为训练代码增加调试计数器，确认梯度更新确实执行，而不仅是调用了 update()。

> 当前文件： train.py

### 实践 12-1，独立评估 DDQN

**课程设计：第 16–18 页，第 7.1–7.2 节**: 阅读： 第 17–18 页关于评估指标与评估协议的内容。

**课程要求**：
- 每隔 50 个训练回合，使用 epsilon=0 评估。
- 评估过程中不允许写入 Replay Buffer。
- 最终使用固定的评估设置比较 DQN 和 DDQN。

**本阶段目标**： 区分「训练时靠探索找到金币」和「网络真正学会找到金币」。

> 额外补充： 先实现一个小规模独立评估函数，通过测试后再接入正式训练。
> python -c "from agents.ddqn import DDQNAgent; from evaluate import evaluate; print(evaluate(DDQNAgent(), episodes=20))"

### 实践 12.2：加入随机策略基准，对比 DDQN 是否真正学到了东西

**课程设计: 第 7 页及第 16–18 页**

**原文要求**： 环境需要进行随机策略基准自检；训练策略需要独立评估。

**本次目标**： 在相同的环境配置和评估回合数下，对比 Random Policy 与训练后的 Greedy Policy。

> 额外补充： 对照实验的控制变量思想，以及较大评估样本量的意义。

```
===== Random Policy =====
{'success_rate': 0.125, 'avg_return': -1.9091999999999991, 'avg_steps': 72.29}

===== Trained DDQN Greedy Policy =====
{'success_rate': 0.145, 'avg_return': -1.573700000000001, 'avg_steps': 172.015}

在当前初步实验中，DDQN 的成功率为 14.5%，略高于随机策略的 12.5%，但差异较小，尚需进一步训练和多随机种子验证。

当前数据可能反映两种不同的失败模式(只是假设)：
- Random Policy：较早掉入陷阱。
- DDQN Greedy：更多回合陷入重复移动或超时。
```

### 实践 12.3：诊断智能体为什么失败

**课程设计第 16–18 页**: 建议阅读： 第 6.4 节「失败模式排查表」和第 7.1–7.2 节「评估指标与协议」。

**作业原文要求**： 分析智能体训练失败或性能不佳的原因，记录评估指标。

**本次目标**： 搞清楚 DDQN 的平均步数为什么如此高。

> 额外补充： 将失败情况进一步分成金币成功、陷阱终止、步数超时三类。

**结果**：
![alt text](/doc/img/12-3comparison.png)

这组数据揭示了两种截然不同的失败方式。
- **Random Policy**： 大多数回合因进入陷阱而结束。
- **DDQN Greedy**： 没有进入陷阱，却有 171 个回合达到 200 步上限。

也就是说，DDQN 很可能形成了一种比较保守的策略：它不去危险区域，但也没有积极找到金币。

### 实践 12.4：追踪 DDQN 的失败路径

**课程设计：第 16 页，第 6.4 节；第 17 页，第 7.1 节**

**文档要求**： 对训练失败、Q 值异常、策略表现不佳等情况进行诊断，并在报告中分析原因。

**本次目标**： 找到一个超时回合，观察智能体到底在做什么。

> 额外补充： 记录单个 Episode 的智能体位置轨迹，判断是重复撞墙还是形成移动循环。

```
===== Timeout Diagnosis =====
Episode: 2
Coin position: (7, 7)
Trap positions: {(2, 5), (5, 6), (2, 2)}
Unique positions: 6
Blocked steps: 0
Most visited positions: [((np.int64(1), np.int64(3)), 99), ((np.int64(2), np.int64(3)), 98), ((np.int64(0), np.int64(0)), 1), ((np.int64(0), np.int64(1)), 1), ((np.int64(0), np.int64(2)), 1)]
First 20 positions: [(np.int64(0), np.int64(0)), (np.int64(0), np.int64(1)), (np.int64(0), np.int64(2)), (np.int64(0), np.int64(3)), (np.int64(1), np.int64(3)), (np.int64(2), np.int64(3)), (np.int64(1), np.int64(3)), (np.int64(2), np.int64(3)), (np.int64(1), np.int64(3)), (np.int64(2), np.int64(3)), (np.int64(1), np.int64(3)), (np.int64(2), np.int64(3)), (np.int64(1), np.int64(3)), (np.int64(2), np.int64(3)), (np.int64(1), np.int64(3)), (np.int64(2), np.int64(3)), (np.int64(1), np.int64(3)), (np.int64(2), np.int64(3)), (np.int64(1), np.int64(3)), (np.int64(2), np.int64(3))]
Last 20 positions: [(np.int64(2), np.int64(3)), (np.int64(1), np.int64(3)), (np.int64(2), np.int64(3)), (np.int64(1), np.int64(3)), (np.int64(2), np.int64(3)), (np.int64(1), np.int64(3)), (np.int64(2), np.int64(3)), (np.int64(1), np.int64(3)), (np.int64(2), np.int64(3)), (np.int64(1), np.int64(3)), (np.int64(2), np.int64(3)), (np.int64(1), np.int64(3)), (np.int64(2), np.int64(3)), (np.int64(1), np.int64(3)), (np.int64(2), np.int64(3)), (np.int64(1), np.int64(3)), (np.int64(2), np.int64(3)), (np.int64(1), np.int64(3)), (np.int64(2), np.int64(3)), (np.int64(1), np.int64(3))]
(deep-learning) jaden@Macbook-Pro ddqn-maze %
```

![alt text](/doc/img/12-4failanalysis.png)
![alt text](/doc/img/12-4failmap.png)

这里可以得出两个有依据的结论：
1. 它没有因为撞墙而超时。 Blocked steps = 0，说明动作都产生了位置变化。
2. 它形成了确定性的循环策略。 在 (1,3) 选择向下（动作 1），在 (2,3) 选择向上（动作 0），不断重复。   

之前我们观察到 DDQN 的陷阱率为 0%，现在必须更加谨慎：它可能并非真正学会了避开陷阱，而只是反复停留在一小片安全区域。

**为什么会出现这种循环？**
目前有两个主要假设。
**假设 A：训练还不充分。**
你的模型只训练了 50 个 Episode、2939 个环境步，而课程设计要求的基线训练规模是 2000 个 Episode。

当前 ε 还在较高水平，Online Network 的学习远未充分。我们不能因为早期策略有问题就直接判定算法失败。

**假设 B：75 维局部观测的信息不足。**
你现在的金币位于 (7,7)。
当智能体处于 (1,3) 或 (2,3) 时，金币都不在其 5×5 局部视野内。
因此 QNet 无法直接从当前观测中知道金币的方向，也没有显式的历史记忆。这可能促成重复决策。

### 实践 13 · 训练日志

**课程设计第 14–17 页，第 6.1 节、第 7.1 节，以及第 20 页交付清单。**

**作业要求**： 保存每回合训练记录，后续用于绘制 Return、成功率、Loss 等曲线，并进行 DQN/DDQN 对照实验。

**本次目标**： 每训练一个 Episode，就能够记录相关指标。

> 额外补充： 使用 Python 的 csv 模块保存结构化日志。
> python -c "from train import train; train(n_episodes=10, seed=0)"

