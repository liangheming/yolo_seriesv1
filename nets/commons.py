import torch
from torch import nn


class CBR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1, padding=None, groups=1):
        super(CBR, self).__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = nn.Hardswish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Focus(nn.Module):
    def __init__(self, in_channel, out_channel, kernel=1, stride=1, padding=None, groups=1):
        super(Focus, self).__init__()
        self.conv = CBR(in_channel * 4, out_channel, kernel, stride, padding, groups)

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], dim=1)
        x = self.conv(x)
        return x


class BottleNeck(nn.Module):
    def __init__(self, in_channel, out_channel, shortcut=True, groups=1, expansion=0.5):
        super(BottleNeck, self).__init__()
        inner_channel = int(out_channel * expansion)
        self.conv1 = CBR(in_channel, inner_channel, 1, 1)
        self.conv2 = CBR(inner_channel, out_channel, 3, 1, groups=groups)
        self.add = shortcut and inner_channel == out_channel

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.add:
            out = x + out
        return out


class BottleNeckCSP(nn.Module):
    def __init__(self, in_channel, out_channel, blocks=1, shortcut=True, groups=1, expansion=0.5):
        super(BottleNeckCSP, self).__init__()
        inner_channel = int(out_channel * expansion)
        self.conv1_0 = CBR(in_channel, inner_channel, 1, 1)
        self.conv2_0 = nn.Conv2d(in_channel, inner_channel, 1, 1, bias=False)
        self.conv1_n = nn.Conv2d(inner_channel, inner_channel, 1, 1, bias=False)
        self.conv3 = CBR(2 * inner_channel, out_channel, 1, 1)
        self.bn = nn.BatchNorm2d(2 * inner_channel)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.conv1_s = nn.Sequential(*[BottleNeck(inner_channel, inner_channel, shortcut, groups, expansion=1)
                                       for _ in range(blocks)])

    def forward(self, x):
        y1 = self.conv1_n(self.conv1_s(self.conv1_0(x)))
        y2 = self.conv2_0(x)
        y = self.act(self.bn(torch.cat([y1, y2], dim=1)))
        y = self.conv3(y)
        return y


class SPP(nn.Module):
    def __init__(self, in_channel, out_channel, k=(5, 9, 13)):
        super(SPP, self).__init__()
        inner_channel = in_channel // 2
        self.conv1 = CBR(in_channel, inner_channel, 1, 1)
        self.conv2 = CBR(inner_channel * (len(k) + 1), out_channel, 1, 1)
        self.pools = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat(([x] + [pool(x) for pool in self.pools]), dim=1)
        x = self.conv2(x)
        return x


if __name__ == '__main__':
    input_tensor = torch.rand(size=(1, 3, 128, 128))
    net = Focus(3, 64)
    out = net(input_tensor)
    print(out.shape)
