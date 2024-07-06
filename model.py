import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

class Generator(nn.Module):
    def __init__(self, color_mode='RGB'):
        super(Generator, self).__init__()
        channels = 1 if color_mode == 'L' else 3
        # 4次下采样
        self.C1 = Conv(channels, 64)
        self.D1 = DownSampling(64)
        self.C2 = Conv(64, 128)
        self.D2 = DownSampling(128)
        self.C3 = Conv(128, 256)
        self.D3 = DownSampling(256)
        self.C4 = Conv(256, 512)
        self.D4 = DownSampling(512)
        self.C5 = Conv(512, 1024)

        # 4次上采样
        self.U1 = UpSampling(1024)
        self.C6 = Conv(1024, 512)
        self.U2 = UpSampling(512)
        self.C7 = Conv(512, 256)
        self.U3 = UpSampling(256)
        self.C8 = Conv(256, 128)
        self.U4 = UpSampling(128)
        self.C9 = Conv(128, 64)

        self.Th = torch.nn.Sigmoid()
        self.amplitude_pred = nn.Conv2d(64, channels, 3, 1, 1)
        self.phase_pred = nn.Conv2d(64, channels, 3, 1, 1)

    def forward(self, x):
        # 下采样部分
        R1 = self.C1(x)
        R2 = self.C2(self.D1(R1))
        R3 = self.C3(self.D2(R2))
        R4 = self.C4(self.D3(R3))
        Y1 = self.C5(self.D4(R4))

        # 上采样部分
        # 上采样的时候需要拼接起来
        O1 = self.C6(self.U1(Y1, R4))
        O2 = self.C7(self.U2(O1, R3))
        O3 = self.C8(self.U3(O2, R2))
        O4 = self.C9(self.U4(O3, R1))

        amplitude_output = self.Th(self.amplitude_pred(O4))
        phase_output = self.Th(self.phase_pred(O4))

        return amplitude_output, phase_output

class Discriminator(nn.Module):
    def __init__(self, num_feat=64, skip_connection=True, color_mode='RGB'):
        super(Discriminator, self).__init__()
        num_in_ch = 1 if color_mode == 'L' else 3
        self.skip_connection = skip_connection
        norm = spectral_norm

        # the first convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)

        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 16, 4, 2, 1, bias=False))

        # upsample
        self.conv5 = norm(nn.Conv2d(num_feat * 16, num_feat * 8, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv7 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))

        # extra convolutions
        self.conv9 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv10 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv11 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        # upsample
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x3
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x2
        x6 = F.interpolate(x6, scale_factor=2, mode='bilinear', align_corners=False)
        x7 = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x7 = x7 + x1
        x7 = F.interpolate(x7, scale_factor=2, mode='bilinear', align_corners=False)
        x8 = F.leaky_relu(self.conv8(x7), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x8 = x8 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv9(x8), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv10(out), negative_slope=0.2, inplace=True)
        out = self.conv11(out)

        return out

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual

class Conv(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(C_in, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            nn.Dropout(0.01),
            nn.LeakyReLU(),
            nn.Conv2d(C_out, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            nn.Dropout(0.01),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)

# 下采样模块
class DownSampling(nn.Module):
    def __init__(self, C):
        super(DownSampling, self).__init__()
        self.Down = nn.Sequential(
            nn.Conv2d(C, C, 3, 2, 1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.Down(x)

# 上采样模块
class UpSampling(nn.Module):
    def __init__(self, C):
        super(UpSampling, self).__init__()
        self.Up = nn.Conv2d(C, C // 2, 1, 1)

    def forward(self, x, r):
        up = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.Up(up)
        return torch.cat((x, r), 1)
