import numpy as np
import torch

from envs.maze import CoinMaze
from agents.ddqn import DDQNAgent

env = CoinMaze(seed=42)
agent = DDQNAgent()
# 创建一个 NumPy 随机数生成器，并使用随机种子 42，让随机结果可以复现。
rng = np.random.default_rng(42)

# 经验不足时，不允许更新
assert agent.update() is None

state = env.reset()

# 使用随机动作采集1000条经验
# 如果我们只关心循环多少次，不关心当前是第几次，就可以把 i 替换成 _
# _ 在 Python 中只是一个普通变量名，但按照编程惯例，它表示这个变量的值不需要使用。
for _ in range(1000):
    action = int(rng.integers(4))
    # step() 方法应该会返回四个值：return next_state, reward, done, info, 第四个值不需要使用，所以用_
    next_state, reward, done, _ = env.step(action)
    agent.buf.push(state, action, reward, next_state, float(done))
    # 变量 = 值1 if 条件 else 值2
    # 如果条件为 True，取 值1。
    # 如果条件为 False，取 值2。
    state = env.reset() if done else next_state
assert len(agent.buf) == 1000

# 保存更新前的参数
# 保存更新前的参数，是为了测试 agent.update() 是否真的更新了 Online Network，以及是否错误地修改了 Target Network。
online_before = [
    # Python 的 列表推导式(List Comprehension): 表达式写在 for 前面
    # [表达式 for 变量 in 可迭代对象]
    # for p in ...：逐个取出参数，赋给变量 p。
    # p.detach()：创建一个与原 Tensor 共享数据，但不参与原计算图梯度追踪的 Tensor。
    # .clone()：复制 Tensor 的数据，得到独立的存储空间。
    # [...]：把所有复制后的 Tensor 放到一个列表中。
    p.detach().clone()
    for p in agent.online.parameters()
]
# fmt:off
target_before=[
    p.detach().clone()
    for p in agent.target.parameters()
]
# fmt:on

# 执行一次优化
loss = agent.update()

assert loss is not None
# 断言 loss 是一个有限的数值，不能是 NaN、正无穷大或负无穷大。
assert np.isfinite(loss)

# Online参数应该发生变化
# 检查执行 agent.update() 后，Online Network 是否至少有一个参数发生了变化。
# zip() 的作用是：把两个序列中的元素一一配对。
# torch.equal(before, after)作用是检查两个 Tensor 的形状和元素是否完全相同。
# any() 是 Python 的内置函数。它的意思是：只要里面有至少一个 True，就返回 True。
online_changed = any(
    not torch.equal(before, after)
    for before, after in zip(online_before, agent.online.parameters())
)
assert online_changed

# Target参数不应该通过BP变化
target_unchanged = all(
    torch.equal(before, after)
    for before, after in zip(target_before, agent.target.parameters())
)
assert target_unchanged

print("DDQN update test PASSED!")
print(f"Loss: {loss}")
print(f"Online parameters changed: {online_changed}")
print(f"Target parameters unchanged: {target_unchanged}")
