import functools
import torch
import torch.nn as nn


class RDB(nn.Module):
    """
    残差稠密模块
    residual dense block
    """
    def __init__(self, nf=64, gc=32):
        super().__init__()
        in_chs = [nf + i * gc for i in range(5)]
        self.conv0 = nn.Conv2d(in_chs[0], gc, 3, 1, 1)
        self.conv1 = nn.Conv2d(in_chs[1], gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_chs[2], gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_chs[3], gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(in_chs[4], nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
    def forward(self, x):
        x0 = self.lrelu(self.conv0(x))
        x_in = torch.cat((x, x0), dim=1)
        x1 = self.lrelu(self.conv1(x_in))
        x_in = torch.cat((x_in, x1), dim=1)
        x2 = self.lrelu(self.conv2(x_in))
        x_in = torch.cat((x_in, x2), dim=1)
        x3 = self.lrelu(self.conv3(x_in))
        x_in = torch.cat((x_in, x3), dim=1)
        res = self.conv4(x_in)
        out = x + 0.2 * res
        return out


class RRDB(nn.Module):
    """
    残差内残差稠密模块
    residual-in-residual dense block
    """
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.body = nn.Sequential(
            *[RDB(nf, gc) for _ in range(3)]
        )
    def forward(self, x):
        res = self.body(x)
        out = x + 0.2 * res
        return out


if __name__ == "__main__":
    dummy_in = torch.randn(4, 64, 8, 8)
    rrdb = RRDB(nf=64, gc=32)
    out = rrdb(dummy_in)
    print(f"RRDB output size {out.size()}")
