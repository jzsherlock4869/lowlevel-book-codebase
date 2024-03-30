import torch
import torch.nn as nn
import torch.nn.functional as F

class SFTLayer(nn.Module):
    def __init__(self, cond_nf=32, res_nf=64):
        super().__init__()
        self.calc_scale = nn.Sequential(
            nn.Conv2d(cond_nf, cond_nf, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(cond_nf, res_nf, 1, 1, 0)
        )
        self.calc_shift = nn.Sequential(
            nn.Conv2d(cond_nf, cond_nf, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(cond_nf, res_nf, 1, 1, 0)
        )
    def forward(self, cond, feat):
        gamma = self.calc_scale(cond)
        beta = self.calc_shift(cond)
        print("[SFTLayer] gamma size: ", gamma.size())
        print("[SFTLayer] beta size: ", beta.size())
        out = feat * (gamma + 1) + beta
        return out


class ResBlockSFT(nn.Module):
    def __init__(self, cond_nf=32, res_nf=64):
        super().__init__()
        self.stf0 = SFTLayer(cond_nf, res_nf)
        self.conv0 = nn.Conv2d(res_nf, res_nf, 3, 1, 1)
        self.stf1 = SFTLayer(cond_nf, res_nf)
        self.conv1 = nn.Conv2d(res_nf, res_nf, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, cond, feat):
        out = self.stf0(cond, feat)
        out = self.relu(self.conv0(out))
        out = self.stf1(cond, out)
        out = self.conv1(out) + feat
        return cond, out


if __name__ == "__main__":
    cond = torch.randn(4, 32, 16, 16)
    feat = torch.randn(4, 64, 16, 16)
    block = ResBlockSFT(cond_nf=32, res_nf=64)
    out = block(cond, feat)
    print(f"[ResBlock SFT] cond size: {out[0].size()}, \n"\
          f"     feat size: {out[1].size()}")