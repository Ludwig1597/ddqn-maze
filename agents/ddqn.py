from collections import deque
import random
import numpy as np

# Replay Buffer：负责存储和抽取经验，它本身不负责预测 Q 值，也没有需要通过 BP 学习的参数。
# 假设智能体在迷宫里寻找金币，每走一步就会产生一条经验。
# 一条经验包含五个部分：(s,a,r,s',done)，这样的五元组通常叫作 Transition（状态转移经验）
# 训练 QNet：抽取 64 条经验 → 计算 TD Target → Loss → BP → 更新参数
# 当 Replay Buffer 已满，通常采用 FIFO（先进先出）方式淘汰最旧的经验，为新经验腾出空间。


class Replaybuffer:
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
        states=np.stack(states)
        actions=np.asarray(actions,dtype=np.int64)
        rewards=np.asarray(rewards,dtype=np.float32)
        next_states=np.stack(next_states)
        dones=np.asarray(dones,dtype=np.float32)
        return states,actions,rewards,next_states,dones

    def __len__(self):
        # TODO 3:
        # 返回经验回放池中的样本数量
        return len(self.buffer)
