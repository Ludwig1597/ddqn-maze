"""_summary_
测试分三步：
1. 验证初始化时两个网络参数相同。
2. 手动改变 Online 参数，确认 Target 不会跟着改变。
3. 调用 sync_target()，确认 Target 重新与 Online 一致。
"""

import torch
from agents.ddqn import DDQNAgent

agent = DDQNAgent()


def params_equal(net1, net2):
    return all(
        torch.equal(p1, p2) for p1, p2 in zip(net1.parameters(), net2.parameters())
    )


# Test1: 初始参数相同
assert params_equal(agent.online, agent.target)
print("Test1: Initial synchronization PASSED")

# Test2: 修改Online参数
with torch.no_grad():
    # 取出 Online Network 的第一个参数 Tensor，然后将其中所有元素加上 0.1。
    # next() 是 Python 的内置函数，用来从迭代器中取出下一个元素。
    first_param = next(agent.online.parameters())
    # add_() 是 PyTorch 的原地操作（In-place Operation）。意思是：将 Tensor 中每一个元素加上 0.1，直接修改原来的数据。
    first_param.add_(0.1)
# Target不应该自动改变
assert not params_equal(agent.online, agent.target)
print("Test2: Network independence PASSED")

# Test3: 主动同步Target
agent.sync_target()
assert params_equal(agent.online, agent.target)
print("Test3: Hard synchronization PASSED")

print("All target synchronization tests PASSED!")
