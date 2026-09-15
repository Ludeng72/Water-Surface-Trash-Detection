import torch
from torch import nn
import torch.nn.functional as F


class SEAttention(nn.Module):
    def __init__(self, channel=512, reduction=16, activation=nn.ReLU):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            activation(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y_channel = self.avg_pool(x).view(b, c)
        y_channel = self.fc(y_channel).view(b, c, 1, 1)
        y_channel = x * y_channel.expand_as(x)

        # 计算空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y_spatial = torch.cat([avg_out, max_out], dim=1)
        y_spatial = self.spatial_attention(y_spatial)
        y_spatial = x * y_spatial.expand_as(x)

        # 结合通道和空间注意力
        return y_channel + y_spatial
