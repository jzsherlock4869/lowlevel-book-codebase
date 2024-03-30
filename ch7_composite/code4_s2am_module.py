import torch
import torch.nn as nn
import torch.nn.functional as F
# pip install kornia
from kornia.filters import GaussianBlur2d

class ChannelAttnModule(nn.Module):
    def __init__(self, nf, reduct=16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(nf * 2, nf // reduct, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf // reduct, nf, 1, 1, 0),
            nn.Sigmoid()
        )
    def forward(self, x):
        ch_avg = self.avgpool(x)
        ch_max = self.maxpool(x)
        cat_vec = torch.cat((ch_avg, ch_max), dim=1)
        attn = self.fc(cat_vec)
        out = attn * x
        return out

class BasicLearnBlock(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.ELU(inplace=True),
            nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(nf),
            nn.ELU(inplace=True)
        )
    def forward(self, x):
        out = self.body(x)
        return out

class S2AM(nn.Module):
    def __init__(self, nf, 
                 sigma=1.0, kgauss=5, reduct=16):
        super().__init__()
        self.connection = BasicLearnBlock(nf)
        self.bg_attn = ChannelAttnModule(nf, reduct)
        self.fg_attn = ChannelAttnModule(nf, reduct)
        self.mix_attn = ChannelAttnModule(nf, reduct)
        self.gauss = GaussianBlur2d(kernel_size=(kgauss, kgauss),
                                    sigma=(sigma, sigma))
    def forward(self, feat, mask):
        ratio = mask.size()[2] // feat.size()[2]
        print(f"[S2AM] mask / feature size ratio: {ratio}")
        if ratio > 1:
            mask = F.avg_pool2d(mask,2,stride=ratio)
            mask = torch.round(mask)
        rev_mask = 1 - mask
        mask = self.gauss(mask)
        rev_mask = self.gauss(rev_mask)
        bg_out = self.bg_attn(feat) * rev_mask
        mix_out = self.mix_attn(feat)
        fg_out = self.connection(self.fg_attn(feat))
        spliced_out = (fg_out + mix_out) * mask
        out = bg_out + spliced_out
        return out


if __name__ == "__main__":
    feat = torch.randn(4, 64, 32, 32)
    mask = torch.randn(4, 1, 128, 128)
    s2am = S2AM(nf=64)
    out = s2am(feat, mask)
    print(f"S2AM out size: {out.size()}")

