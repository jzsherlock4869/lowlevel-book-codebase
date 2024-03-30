import torch
import torch.nn as nn

class ConditionNet(nn.Module):
    def __init__(self, in_ch=3, nf=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, nf, 7, 2, 1)
        self.conv2 = nn.Conv2d(nf, nf, 3, 2, 1)
        self.conv3 = nn.Conv2d(nf, nf, 3, 2, 1)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        cond = self.avg_pool(out)
        return cond


class GFM(nn.Module):
    def __init__(self, cond_nf, in_nf, base_nf):
        super().__init__()
        self.mlp_scale = nn.Conv2d(cond_nf, base_nf, 1, 1, 0)
        self.mlp_shift = nn.Conv2d(cond_nf, base_nf, 1, 1, 0)
        self.conv = nn.Conv2d(in_nf, base_nf, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, cond):
        feat = self.conv(x)
        scale = self.mlp_scale(cond)
        shift = self.mlp_shift(cond)
        out = feat * scale + shift + feat
        out = self.relu(out)
        return out


class CSRNet(nn.Module):
    def __init__(self, in_ch=3,
                 out_ch=3,
                 base_nf=64,
                 cond_nf=32):
        super().__init__()
        self.condnet = ConditionNet(in_ch, cond_nf)
        self.gfm1 = GFM(cond_nf, in_ch, base_nf)
        self.gfm2 = GFM(cond_nf, base_nf, base_nf)
        self.gfm3 = GFM(cond_nf, base_nf, out_ch)
    def forward(self, x):
        cond = self.condnet(x)
        out = self.gfm1(x, cond)
        out = self.gfm2(out, cond)
        out = self.gfm3(out, cond)
        return out


if __name__ == "__main__":
    dummy_in = torch.randn(4, 3, 128, 128)
    csrnet = CSRNet()
    out = csrnet(dummy_in)
    print('CSRNet input size: ', dummy_in.size())
    print('CSRNet output size: ', out.size())
    n_para = sum([p.numel() for p in csrnet.parameters()])
    print(f'CSRNet total no. params: {n_para/1024:.2f}K')

