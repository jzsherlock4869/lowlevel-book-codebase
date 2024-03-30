import torch
import torch.nn as nn

class DilateConvCat(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3):
        super().__init__()
        pad = ksize // 2 + 1
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, ksize, 1, pad, 2),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        out = self.body(x)
        out = torch.cat((x, out), dim=1)
        return out

class DRDB(nn.Module):
    """
    Dilated Residual Dense Block
    """
    def __init__(self, nf, gc, num_layer):
        super().__init__()
        self.dense = nn.Sequential(*[
            DilateConvCat(nf + i * gc, gc, 3)
                for i in range(num_layer)
        ])
        self.fusion = nn.Conv2d(nf + num_layer * gc,
                                nf, 1, 1, 0)
    def forward(self, x):
        out = self.dense(x)
        out = self.fusion(out) + x
        return out

class Attention(nn.Module):
    """
    Attention Module for over-/under-exposure
    """
    def __init__(self, nf=64):
        super().__init__()
        self.conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x_base, x_ref):
        x = torch.cat((x_base, x_ref), dim=1)
        out = self.conv1(x)
        out = self.lrelu(out)
        out = self.conv2(out)
        attn_map = self.sigmoid(out)
        return x_base * attn_map


class AHDRNet(nn.Module):
    def __init__(self,
                 in_ch=6,
                 out_ch=3,
                 num_dense=6,
                 num_feat=64,
                 growth_rate=32):
        super().__init__()
        self.feat_extract = nn.Conv2d(in_ch, num_feat, 3, 1, 1)
        self.attn1 = Attention(num_feat)
        self.attn2 = Attention(num_feat)
        self.fusion1 = nn.Conv2d(num_feat * 3, num_feat, 3, 1, 1)
        self.drdb1 = DRDB(num_feat, growth_rate, num_dense)
        self.drdb2 = DRDB(num_feat, growth_rate, num_dense)
        self.drdb3 = DRDB(num_feat, growth_rate, num_dense)
        self.fusion2 = nn.Sequential(
            nn.Conv2d(num_feat * 3, num_feat, 1, 1, 0),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            nn.Conv2d(num_feat, out_ch, 3, 1, 1),
            nn.Sigmoid()
        )
        self.lrelu = nn.LeakyReLU()

    def forward(self, evm, ev0, evp):
        fm = self.lrelu(self.feat_extract(evm))
        f0 = self.lrelu(self.feat_extract(ev0))
        fp = self.lrelu(self.feat_extract(evp))
        fm = self.attn1(fm, f0)
        fp = self.attn2(fp, f0)
        fcat = torch.cat([fm, f0, fp], dim=1)
        ff = self.fusion1(fcat)
        ff1 = self.drdb1(ff)
        ff2 = self.drdb2(ff1)
        ff3 = self.drdb2(ff2)
        ffcat = torch.cat([ff1, ff2, ff3], dim=1)
        res = self.fusion2(ffcat)
        out = self.conv_out(f0 + res)
        return out

if __name__ == "__main__":
    evm = torch.rand(4, 6, 64, 64)
    ev0 = torch.rand(4, 6, 64, 64)
    evp = torch.rand(4, 6, 64, 64)
    ahdrnet = AHDRNet()
    print("AHDRNet architecture: ")
    print(ahdrnet)
    out = ahdrnet(evm, ev0, evp)
    print(f"AHDRNet input size: {evm.size()} (x3)")
    print("AHDRNet output size: ", out.size())

