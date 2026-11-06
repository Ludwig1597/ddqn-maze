import torch
from agents.ddqn import DDQNAgent

# 读取保存的checkpoint
# fmt:off
# torch.load() 负责把文件内容读入 Python；
# 从磁盘中读取之前保存的 DDQN 模型检查点（Checkpoint），并将里面的数据加载到 CPU 内存中
checkpoint = torch.load(
    "checkpoints/ddqn_seed0_best.pt", # 模型检查点的文件路径
    map_location="cpu", # map_location 的意思是设备映射位置
    weights_only=True # 使用受限制的加载方式，主要允许读取模型权重等常见的安全数据类型
)
# fmt:on

# 新建Agent：此时Online QNet是随机初始化的
agent = DDQNAgent()

# TODO 1:
# 使用agent.online.load_state_dict(...)
# 将checkpoint中的权重加载到OnlineQNet
agent.online.load_state_dict(checkpoint["online_state_dict"])


# TODO 2:
# 检查加载后到权重与checkpoint中的权重是否一致
"""_summary_
checkpoint["online_state_dict"]
本身也是一个字典，里面保存了 Online QNet 各层的权重和偏置。
{
    "net.0.weight": tensor(...),
    "net.0.bias": tensor(...),
    "net.2.weight": tensor(...),
    "net.2.bias": tensor(...),
    "net.4.weight": tensor(...),
    "net.4.bias": tensor(...)
}
"""
# .items() 是 Python 字典的方法
# .items() 会提供字典中所有的 Key–Value 键值对，方便使用 for 循环遍历。
for key, value in checkpoint["online_state_dict"].items():
    assert torch.equal(agent.online.state_dict()[key], value)
print("Checkpoint loading PASSED!")

# 同步Target Network
agent.sync_target()
for online_param, target_param in zip(
    agent.online.parameters(), agent.target.parameters()
):
    assert torch.equal(online_param, target_param)

print("Target network synchronization PASSED!")

# TODO 3:
# 创建形状为(1,75)的输入Tensor
# 使用agent.online(...)计算Q值
# 检查输出形状是否为(1,4)
# 为了验证刚刚从 Checkpoint 加载的 Online QNet，能否正常进行前向传播（Forward），并输出正确形状的 Q 值。
# (1,75):表示 1 个样本，每个样本有 75 个特征。
x = torch.zeros((1, 75), dtype=torch.float32)
with torch.no_grad():
    q_values = agent.online(x)
assert q_values.shape == (1, 4)
print("QNet forward test PASSED!")

print("Checkpoint loading test PASSED!")
