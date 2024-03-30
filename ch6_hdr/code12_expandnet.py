import torch
from torch import nn
from torch.nn import functional as F

class ConvSELU(nn.Module):
    def __init__(self, in_ch, out_ch,
                 ksize, stride, pad, dilation):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch,
                ksize, stride, pad, dilation)
        self.selu = nn.SELU(inplace=True)
    def forward(self, x):
        out = self.conv(x)
        out = self.selu(out)
        return out

class ExpandNet(nn.Module):
    def __init__(self, in_ch=3, nf=64):
        super().__init__()
        # ConvSELU 和 Conv2d 参数顺序:
        # in, out, ksize, stride, pad, dilation
        self.local_branch = nn.Sequential(
            ConvSELU(in_ch, nf, 3, 1, 1, 1),
            ConvSELU(nf, nf * 2, 3, 1, 1, 1)
        )
        self.dilation_branch = nn.Sequential(
            ConvSELU(in_ch, nf, 3, 1, 2, 2),
            ConvSELU(nf, nf, 3, 1, 2, 2),
            ConvSELU(nf, nf, 3, 1, 2, 2),
            nn.Conv2d(nf, nf, 3, 1, 2, 2)
        )
        self.global_branch = nn.Sequential(
            ConvSELU(in_ch, nf, 3, 2, 1, 1),
            *[ConvSELU(nf, nf, 3, 2, 1, 1) \
              for _ in range(5)],
            nn.Conv2d(nf, nf, 4, 1, 0, 1)
        )
        self.fusion = nn.Sequential(
            ConvSELU(nf * 4, nf, 1, 1, 0, 1),
            nn.Conv2d(nf, 3, 1, 1, 0, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        local_out = self.local_branch(x)
        dilated_out = self.dilation_branch(x)
        x256 = F.interpolate(x, (256, 256),
                mode="bilinear", align_corners=False)
        global_out = self.global_branch(x256)
        global_out = global_out.expand(*dilated_out.size())
        print("[ExpandNet] internal tensor sizes:")
        print(f"  local: {list(local_out.size())}")
        print(f"  dilation: {list(dilated_out.size())}")
        print(f"  global: {list(global_out.size())}")
        fused = torch.cat([local_out,
                           dilated_out,
                           global_out], dim=1)
        out = self.fusion(fused)
        return out


if __name__ == "__main__":
    x = torch.rand(4, 3, 256, 256)
    expandent = ExpandNet()
    out = expandent(x)
    print(f"ExpandNet input size: {x.size()}")
    print(f"ExpandNet output size: {out.size()}")

