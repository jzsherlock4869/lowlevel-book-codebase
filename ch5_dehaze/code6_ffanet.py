import torch
import torch.nn as nn


class PixelAttention(nn.Module):
    def __init__(self, nf, reduct=8):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(nf, nf // reduct, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf // reduct, 1, 1, 1, 0),
            nn.Sigmoid()
        )
    def forward(self, x):
        attn_map = self.attn(x)
        return x * attn_map


class ChannelAttention(nn.Module):
    def __init__(self, nf, reduct=8, ret_w=False):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.attn = nn.Sequential(
            nn.Conv2d(nf, nf // reduct, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf // reduct, nf, 1, 1, 0),
            nn.Sigmoid()
        )
        self.ret_w = ret_w
    def forward(self, x):
        attn_map = self.attn(self.avgpool(x))
        if self.ret_w:
            return attn_map
        else:
            return x * attn_map


class BasicBlock(nn.Module):
    def __init__(self, nf, pa_reduct=8, ca_reduct=8):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.pix_attn = PixelAttention(nf, pa_reduct)
        self.ch_attn = ChannelAttention(nf, ca_reduct)
    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = x + out
        out = self.conv2(out)
        out = self.ch_attn(out)
        out = self.pix_attn(out)
        out = x + out
        return out


class BlockGroup(nn.Module):
    def __init__(self,
                nf, num_block,
                pa_reduct=8,
                ca_reduct=8):
        super().__init__()
        pr, cr = pa_reduct, ca_reduct
        self.group = nn.Sequential(
            *[BasicBlock(nf, pr, cr) 
                for _ in range(num_block)],
            nn.Conv2d(nf, nf, 3, 1, 1)
        )
    def forward(self, x):
        out = self.group(x)
        out = x + out
        return out


class FFANet(nn.Module):
    def __init__(self, in_ch, num_block=19):
        super().__init__()
        nf, nb = 64, num_block
        self.conv_in = nn.Conv2d(in_ch, nf, 3, 1, 1)
        self.group1 = BlockGroup(nf, nb)
        self.group2 = BlockGroup(nf, nb)
        self.group3 = BlockGroup(nf, nb)
        self.last_CA = ChannelAttention(nf * 3, ret_w=True)
        self.last_PA = PixelAttention(nf)
        self.conv_post = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_out = nn.Conv2d(nf, in_ch, 3, 1, 1)

    def forward(self, x):
        feat = self.conv_in(x)
        res1 = self.group1(feat)
        res2 = self.group2(res1)
        res3 = self.group3(res2)
        res_cat = torch.cat([res1, res2, res3], dim=1)
        attn_w = self.last_CA(res_cat)
        ws = attn_w.chunk(3, dim=1)
        res = ws[0] * res1 + ws[1] * res2 + ws[2] * res3
        res = self.last_PA(res)
        res = self.conv_out(self.conv_post(res))
        out = x + res
        return out


if __name__ == "__main__":
    x_in = torch.randn(4, 3, 128, 128)
    # 为方便展示网络结构，num_block仅设置为3
    ffanet = FFANet(in_ch=3, num_block=3)
    print("FFA-Net architecture:")
    print(ffanet)
    out = ffanet(x_in)
    print("FFA-Net input size: ", x_in.size())
    print("FFA-Net output size: ", out.size())