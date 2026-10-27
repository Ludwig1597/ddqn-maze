import torch
from torch import nn


class QNet(nn.Module):
    def __init__(self, obs_dim=75, n_act=4, hidden=128):
        # 因为QNet继承了nn.Module，因此需要super().__init__(),执行父类的初始化方法。
        super().__init__()

        self.net = nn.Sequential(
            # TODO 1: 75->128
            nn.Linear(75, 128),
            # TODO 2: ReLU
            nn.ReLU(),
            # TODO 3: 128->128
            nn.Linear(128, 128),
            # TODO 4: ReLU
            nn.ReLU(),
            # TODO 5: 128->4
            nn.Linear(128, 4),
        )

    def forward(self, x):
        # TODO 6:返回网络预测的Q值
        return self.net(x)
