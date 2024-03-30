import numpy as np
import torch
import torch.nn as nn

class ResBlock_woBN(nn.Module):
    """
    ResBlock模块, 无BN层
    """
    def __init__(self, nf, res_scale=1.0):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.res_scale = res_scale
    
    def forward(self, x):
        res = self.relu(self.conv1(x))
        res = self.conv2(res)
        out = x + res * self.res_scale
        return out


class Upsampler(nn.Module):
    """
    上采样模块, 通过卷积和PixelShuffle进行上采样
    """
    def __init__(self, nf, scale):
        super().__init__()
        module_ls = list()
        assert scale == 2 or scale == 4
        num_blocks = int(np.log2(scale))
        for _ in range(num_blocks):
            module_ls.append(nn.Conv2d(nf, nf * 4, 3, 1, 1))
            module_ls.append(nn.PixelShuffle(2))
        self.body = nn.Sequential(*module_ls)

    def forward(self, x):
        out = self.body(x)
        return out


class EDSR(nn.Module):
    """
    EDSR 模型实现, 包括conv、无BN的resblock、pixelshuffle组成
    """
    def __init__(self, in_ch, nf, num_blocks, scale):
        super().__init__()
        self.conv_in = nn.Conv2d(in_ch, nf, 3, 1, 1)
        self.resblocks = nn.Sequential(
            *[ResBlock_woBN(nf) for _ in range(num_blocks)],
            nn.Conv2d(nf, nf, 3, 1, 1)
        )
        self.upscale = Upsampler(nf, scale)
        self.conv_out = nn.Conv2d(nf, in_ch, 3, 1, 1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.resblocks(x)
        x = self.upscale(x)
        out = self.conv_out(x)
        return out


if __name__ == "__main__":
    dummy_in = torch.randn(4, 3, 128, 128)
    edsr = EDSR(in_ch=3, nf=64, num_blocks=15, scale=4)
    out = edsr(dummy_in)
    print('EDSR input size: ', dummy_in.size())
    print('EDSR output size: ', out.size())
