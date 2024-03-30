import numpy as np
import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, nf, reduction=16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 用 1x1 conv 实现 MLP 功能
        self.mlp = nn.Sequential(
            nn.Conv2d(nf, nf // reduction, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf // reduction, nf, 1, 1, 0),
            nn.Sigmoid()
        )
    def forward(self, x):
        vec = self.gap(x)
        attn = self.mlp(vec)
        # [n,c,h,w] * [n,c,1,1]
        out = x * attn
        return out

class RCAB(nn.Module):
    """
    Residual Channel Attention Block
    残差通道注意力模块, RCAN 的基础模块
    """
    def __init__(self, nf, reduction):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1),
            ChannelAttention(nf, reduction)
        )
    def forward(self, x):
        res = self.body(x)
        out = res + x
        return out

class ResidualGroup(nn.Module):
    """
    将 RCAB 进行组合, 形成残差模块
    """
    def __init__(self, nf, reduction, n_blocks):
        super().__init__()
        self.body = nn.Sequential(
            *[RCAB(nf, reduction) for _ in range(n_blocks)],
            nn.Conv2d(nf, nf, 3, 1, 1)
        ) 
    def forward(self, x):
        res = self.body(x)
        out = res + x
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


class RCAN(nn.Module):
    def __init__(self,
                n_groups=10,
                n_blocks=20,
                in_ch=3,
                nf=64,
                reduction=16,
                scale=4):
        super().__init__()
        self.conv_in = nn.Conv2d(in_ch, nf, 3, 1, 1)
        self.body = nn.Sequential(
            *[ResidualGroup(nf, reduction, n_blocks) \
                for _ in range(n_groups)],
            nn.Conv2d(nf, nf, 3, 1, 1)
        )
        self.upper = Upsampler(nf, scale)
        self.conv_out = nn.Conv2d(nf, in_ch, 3, 1, 1)
    
    def forward(self, x):
        feat = self.conv_in(x)
        feat = self.body(feat) + feat
        upfeat = self.upper(feat)
        out = self.conv_out(upfeat)
        return out



if __name__ == "__main__":
    dummy_in = torch.randn(4, 3, 64, 64)
    rcan = RCAN(scale=4)
    out = rcan(dummy_in)
    print("RCAN output size: ", out.size())

