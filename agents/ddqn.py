from collections import deque
import random
import numpy as np
from networks.qnet import QNet
import torch

# Replay Buffer：负责存储和抽取经验，它本身不负责预测 Q 值，也没有需要通过 BP 学习的参数。
# 假设智能体在迷宫里寻找金币，每走一步就会产生一条经验。
# 一条经验包含五个部分：(s,a,r,s',done)，这样的五元组通常叫作 Transition（状态转移经验）
# 训练 QNet：抽取 64 条经验 → 计算 TD Target → Loss → BP → 更新参数
# 当 Replay Buffer 已满，通常采用 FIFO（先进先出）方式淘汰最旧的经验，为新经验腾出空间。


class ReplayBuffer:
    # capacity=50_000：下划线是提高可读性的写法
    def __init__(self, capacity=50_000):
        # deque(maxlen=capacity)是一个支持限制最大长度的双端队列。
        # 当容量已满，再加入新经验时，会自动删除最旧的一条。
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        # TODO 1:
        # 把一条经验存入self.buffer
        # 注意将状态数组复制后保存
        # 因为state 和 next_state 是 NumPy 数组， NumPy 数组是可变对象（Mutable Object）。
        # 如果不复制，Replay Buffer 保存的可能只是原数组的引用，而不是当时状态的独立快照。
        # 而 action、reward、done 通常是整数、浮点数和布尔值（不可变对象）。
        self.buffer.append((s.copy(), a, r, s2.copy(), done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        # 将随机抽取的经验按照字段拆分
        # 把一批经验按照字段拆开，将所有 state 放在一起、所有 action 放在一起、所有 reward 放在一起……
        # zip 英文有「拉链」的意思。Python 的 zip() 会把多个序列中相同位置的元素组合起来。
        # * 叫作可迭代对象解包（Iterable Unpacking），在函数调用时，把列表中的元素展开成多个位置参数。
        """
        batch = [                                       zip(
            (s1, a1),                                      (s1, a1),         这就完成了行列转置。
            (s2, a2),    states, actions = zip(*batch)     (s2, a2),    =>   states  = (s1, s2, s3)
            (s3, a3)                                       (s3, a3)          actions = (a1, a2, a3)
        ]                                               )
        """
        states, actions, rewards, next_states, dones = zip(*batch)

        # TODO 2:
        # 返回五个NumPy数组，顺序不变
        # states/next_states使用np.stack(...)
        # actions使用int64
        # rewards / dones 使用float32

        # NumPy 的 np.stack()：数组堆叠，将多个数组组合成一个更高维的数组
        # 64个(75,)数组 -> np.stack -> (64,75)
        states = np.stack(states)
        actions = np.asarray(actions, dtype=np.int64)
        rewards = np.asarray(rewards, dtype=np.float32)
        next_states = np.stack(next_states)
        dones = np.asarray(dones, dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        # TODO 3:
        # 返回经验回放池中的样本数量
        return len(self.buffer)


"""
DDQN (Double Deep Q-Network)
1. Online Network 选择下一状态的最优动作: a* = argmax_a Q_online(s', a)
2. Target Network 评估该动作的 Q 值: Q_next = Q_target(s', a*)
3. 计算 TD Target(自举得到的训练目标，它不是真实最优 Q 值):y = r + γ(1 - done) · Q_next
4. 当前 Q 值预测: Q_pred = Q_online(s, a)
5. 计算损失: L = HuberLoss(Q_pred, y)
6. 使用 BP 更新 Online Network 参数。
"""

# DQN 和 DDQN 都使用两个网络：
# Online Network: 参数ɵ，选择下一动作，并通过 BP 更新
# Target Network: 参数ɵ-，评估动作价值，定期从 Online 复制参数
# Online 网络负责选动作：argmax()。
# Target 网络负责评价这个动作：gather()。
# 根据奖励、折扣因子和 done 计算目标值。


class DDQNAgent:
    def __init__(self, gamma=0.95):
        self.gamma = gamma

        # TODO 1:
        # 创建Online QNet和Target QNet
        # 分别保存在self.online和self.target
        # 创建两个网络，不代表它们的参数相同
        # self.online：Online Network，负责学习 Q 值、更新参数，模式为train()
        # self.target：Target Network，负责计算 TD Target，模式为eval()
        # 目标：让 Online 的预测接近 TD Target
        self.online = QNet()
        self.target = QNet()

        # TODO 2:
        # 将online的初始参数复制到target
        # 提示：load_state_dict()和state_dict()
        # state_dict() 的意思是：获取模型的状态字典。
        # 这个字典保存了神经网络的权重、偏置等参数，以及某些层的持久状态。
        params = self.online.state_dict()
        # 两个网络分别随机初始化，因此初始参数通常不同。
        # 把字典中的参数值加载到 Target 网络对应的参数中。
        # Online QNet参数更新方式：优化器更新（执行BP）
        # Target QNet参数更新方式：定期从 Online 同步
        self.target.load_state_dict(params)

        # Target网络用于评估
        # eval()把 Target 网络切换到 Evaluation Mode（评估模式）。
        # PyTorch 的神经网络主要有两种运行模式：
        # - model.train()	训练模式	训练神经网络
        # - model.eval()	评估模式	推理、验证、评估
        # 对于当前的 QNet，调用 .eval() 通常不会改变网络的前向计算结果。
        # 但保留它是合理的，能够清楚表明 Target Network 用于推理和目标值计算。
        # PyTorch 内部可以近似理解为：self.target.train(False),只是近似理解，事实上并不会自动关闭梯度
        # 评估模式下，不通过优化器训练
        self.target.eval()

    # @ 是 Python 中的 装饰器（Decorator）语法。
    # 装饰器可以理解为：不修改函数内部代码，就给函数增加额外的功能或行为。
    # 每次执行 compute_target() 时，自动关闭 PyTorch 的梯度记录。
    # Online Q Prediction，需要梯度，用于 BP
    # TD Target，yDQN 不需要梯度，作为固定训练目标
    @torch.no_grad()
    def compute_target(self, rewards, next_states, dones):
        # TODO 3:
        # 使用online网络计算下一状态Q值
        # 用argmax()选出动作索引
        # 结果形状应是(batch_size,1)
        # DDQN 核心公式中的第一步：使用 Online QNet 选择下一状态的最优动作。
        # a*=argmax_a Q_online(s',a)
        # 意思是：把下一状态 (s') 输入 Online QNet，选择 Q 值最大的动作。
        # 1. 计算下一状态的Q值
        # self.online(next_states)调用QNet对象->nn.Module.call()PyTorch 自动处理调用过程->QNet.forward(x)x = next_states
        # self.net(x)Linear → ReLU → Linear → ReLU → Linear->返回 next_q_values Shape = (64, 4)
        # 假设 batch_size=64：
        # - next_states.shape = (64, 75)
        # - next_q_values.shape = (64, 4)
        # 因为每个状态对应 4 个动作的 Q 值。
        next_q_value = self.online(next_states)
        # 2. 选择最大 Q 值对应的动作
        # 这样得到：next_actions.shape  # (64,)
        # next_action=next_q_value.argmax(dim=1)
        # 但我们需要（64,1),gather() 需要动作索引具有 (64,1) 这样的形状，才能为每一个状态取出对应动作的 Q 值。
        # 使用 keepdim=True,Keep Dimension（保留维度） 的缩写。
        # keepdim=Flase,argmax(dim=1)=[1,2,3],形状(3,)
        # keepdim=True,argmax(dim=1)=[[1],[2],[3]],形状(3,1)
        next_action = next_q_value.argmax(dim=1, keepdim=True)

        # TODO 4:
        # 使用target网络计算下一状态Q值
        # 用gather()提取对应动作的Q值
        # 结果形状应是(batch_size,)
        # Target Network 不重新选择动作，只负责评价 Online 选出的动作。
        # Q_target(next_state,a*),a* 是 Online QNet 选出的动作。
        # gather 英文是「收集、聚集」的意思。
        # PyTorch 的 tensor.gather(dim, index) 可以根据指定的索引，沿dim维度,从 Tensor 中提取对应元素。
        # (64,4)->gather->(64,1)->squeeze(1)->(64,)
        target_q_value = self.target(next_states)
        # squeeze(1)的意思是：删除第 1 维，前提是该维度长度为 1。
        selected_q = target_q_value.gather(1, next_action).squeeze(1)

        # TODO 5:
        # 计算DDQN TD Target
        # y=r+gamma*(1-done)*next_q
        # Online 选动作 → Target 评价动作 → Bellman 公式计算 TD Target。
        # PyTorch 自动用标量 1 减去 Tensor 中dones的每个元素。
        td_target = rewards + self.gamma * (1 - dones) * selected_q

        return td_target
