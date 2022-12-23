import torch
import torch.nn as nn


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, is_2d=False):
        super().__init__()
        if is_2d:
            self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
                                      nn.BatchNorm2d(out_channels),
                                      nn.LeakyReLU(),
                                      nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                                      nn.BatchNorm2d(out_channels),
                                      nn.LeakyReLU())
        else:
            self.conv = nn.Sequential(nn.Conv3d(in_channels, out_channels, 4, 2, 1, bias=False),
                                      nn.BatchNorm3d(out_channels),
                                      nn.LeakyReLU(),
                                      nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
                                      nn.BatchNorm3d(out_channels),
                                      nn.LeakyReLU())

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, is_2d=False):
        super().__init__()
        if not mid_channels:
            mid_channels = in_channels
        if is_2d:
            self.up = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU())
            self.conv = nn.Sequential(nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
                                      nn.BatchNorm2d(out_channels),
                                      nn.ReLU())
        else:
            self.up = nn.Sequential(nn.ConvTranspose3d(in_channels, out_channels, 4, 2, 1, bias=False),
                                    nn.BatchNorm3d(out_channels),
                                    nn.ReLU())
            self.conv = nn.Sequential(nn.Conv3d(mid_channels, out_channels, 3, padding=1, bias=False),
                                      nn.BatchNorm3d(out_channels),
                                      nn.ReLU())

    def forward(self, x, *args):
        """
        上采样执行
        :param x: 参与上采样的参数（必要）
        :param args: 参与跳跃连接的参数（至少1个）
        :return:
        """
        x = self.up(x)
        x = torch.cat([x, *args], dim=1)
        return self.conv(x)


class UNet2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.input = nn.Sequential(nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(32),
                                   nn.LeakyReLU(),
                                   nn.Conv2d(32, 32, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(32),
                                   nn.LeakyReLU())
        self.down1 = Down(32, 64, is_2d=True)
        self.down2 = Down(64, 128, is_2d=True)
        self.down3 = Down(128, 256, is_2d=True)
        self.down4 = Down(256, 512, is_2d=True)
        self.up4 = Up(512, 256, is_2d=True)
        self.up3 = Up(256, 128, is_2d=True)
        self.up2 = Up(128, 64, is_2d=True)
        self.up1 = Up(64, 32, is_2d=True)
        if out_channels == 1:
            self.output = nn.Sequential(nn.Conv2d(32, 1, 3, padding=1, bias=False),
                                        nn.BatchNorm2d(1),
                                        nn.ReLU(),
                                        nn.Conv2d(1, 1, 1),
                                        nn.Sigmoid())
        elif out_channels > 1:
            self.output = nn.Sequential(nn.Conv2d(32, out_channels, 3, padding=1, bias=False),
                                        nn.BatchNorm2d(out_channels),
                                        nn.ReLU(),
                                        nn.Conv2d(out_channels, out_channels, 1),
                                        nn.Softmax())
        else:
            print('错误：网络参数[out_classes]设置错误，应当≥1。')
            exit(-1)

    def forward(self, x):
        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        x = self.up4(x, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        return self.output(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.input = nn.Sequential(nn.Conv3d(in_channels, 32, 3, padding=1, bias=False),
                                   nn.BatchNorm3d(32),
                                   nn.LeakyReLU(),
                                   nn.Conv3d(32, 32, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(32),
                                   nn.LeakyReLU())
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.up3 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.up1 = Up(64, 32)
        if out_channels == 1:
            self.output = nn.Sequential(nn.Conv3d(32, 1, 3, padding=1, bias=False),
                                        nn.BatchNorm3d(1),
                                        nn.ReLU(),
                                        nn.Conv3d(1, 1, 1),
                                        nn.Sigmoid())
        elif out_channels > 1:
            self.output = nn.Sequential(nn.Conv3d(32, out_channels, 3, padding=1, bias=False),
                                        nn.BatchNorm3d(out_channels),
                                        nn.ReLU(),
                                        nn.Conv3d(out_channels, out_channels, 1),
                                        nn.Softmax())
        else:
            print('错误：网络参数[out_classes]设置错误，应当≥1。')
            exit(-1)

    def forward(self, x):
        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.down3(x3)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        return self.output(x)


class DoubleEncoderSingleDecoderNetwork(nn.Module):
    def __init__(self):
        super().__init__()
