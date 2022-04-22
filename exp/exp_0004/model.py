from torch import nn
import copy


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
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.advantages = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        values = self.values(x)
        advantages = self.advantages(x)
        q = values + (advantages - advantages.mean(dim=1, keepdims=True))
        return q
