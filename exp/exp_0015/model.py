import math
import torch
from torch import nn
import torch.nn.functional as F
import copy


class FactorizedNoisy(nn.Module):
    def __init__(self, in_features, out_features):
        super(FactorizedNoisy, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 学習パラメータを生成
        self.u_w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigma_w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.u_b = nn.Parameter(torch.Tensor(out_features))
        self.sigma_b = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        # 初期値設定
        stdv = 1. / math.sqrt(self.u_w.size(1))
        self.u_w.data.uniform_(-stdv, stdv)
        self.u_b.data.uniform_(-stdv, stdv)

        initial_sigma = 0.5 * stdv
        self.sigma_w.data.fill_(initial_sigma)
        self.sigma_b.data.fill_(initial_sigma)

    def forward(self, x):
        # 毎回乱数を生成
        rand_in = self._f(torch.randn(
            1, self.in_features, device=self.u_w.device))
        rand_out = self._f(torch.randn(
            self.out_features, 1, device=self.u_w.device))
        epsilon_w = torch.matmul(rand_out, rand_in)
        epsilon_b = rand_out.squeeze()

        w = self.u_w + self.sigma_w * epsilon_w
        b = self.u_b + self.sigma_b * epsilon_b
        return F.linear(x, w, b)

    def _f(self, x):
        return torch.sign(x) * torch.sqrt(torch.abs(x))


class MarioNet(nn.Module):
    def __init__(self, cfg, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim
        if h != cfg.state_height:
            raise ValueError(f'Expecting input height: 84, got: {h}')
        if w != cfg.state_width:
            raise ValueError(f'Expecting input width: 84, got: {w}')

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.values = nn.Sequential(
            FactorizedNoisy(3136, 512),
            nn.ReLU(),
            FactorizedNoisy(512, 1)
        )
        self.advantages = nn.Sequential(
            FactorizedNoisy(3136, 512),
            nn.ReLU(),
            FactorizedNoisy(512, output_dim)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        values = self.values(x)
        advantages = self.advantages(x)
        q = values + (advantages - advantages.mean(dim=1, keepdims=True))
        return q