import torch
from torch import nn



class BELModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, dilation=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )       
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=6, dilation=6),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(in_channels * 5, out_channels, kernel_size=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.drop = nn.Dropout(0.2)

    def forward(self, features):
        x1 = self.conv1(features)
        x2 = self.conv2(features)
        x3 = self.conv3(features)
        x4 = self.conv4(features)
        x5 = self.conv5(features)
        group_x = torch.concat([x1, x2, x3, x4, x5], dim=1)
        x = self.drop(group_x)
        out = self.conv_out(x)

        return out


class BEBModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BEBModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        border = self.sigmoid(self.conv(features))
        return border



class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class BCModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.att = ChannelAttention(channels)

    def forward(self, features):
        att = self.att(features)
        att = features*(att+1)
        return att
