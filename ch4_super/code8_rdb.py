import torch
import torch.nn as nn

class ConvCat(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3):
        super().__init__()
        pad = ksize // 2
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, ksize, 1, pad),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        out = self.body(x)
        out = torch.cat((x, out), dim=1)
        return out

class ResidualDenseBlock(nn.Module):
    def __init__(self, nf, gc, num_layer):
        super().__init__()
        self.dense = nn.Sequential(*[
            ConvCat(nf + i * gc, gc, 3)
                for i in range(num_layer)
        ])
        self.fusion = nn.Conv2d(nf + num_layer * gc,
                                nf, 1, 1, 0)
    def forward(self, x):
        out = self.dense(x)
        out = self.fusion(out) + x
        return out


if __name__ == "__main__":
    x_in = torch.randn(4, 64, 16, 16)
    rdb = ResidualDenseBlock(nf=64, gc=32, num_layer=4)
    x_out = rdb(x_in)
    print('RDB output size: ', x_out.size())
