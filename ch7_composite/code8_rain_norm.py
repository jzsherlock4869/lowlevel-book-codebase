import torch
import torch.nn as nn
import torch.nn.functional as F


def get_masked_mean_std(feat, mask, eps=1e-5):
    masked_feat = feat * mask
    summ = torch.sum(masked_feat, dim=[2, 3], keepdim=True)
    num = torch.sum(mask, dim=[2, 3], keepdim=True)
    mean = summ / (num + eps)
    sqr = torch.sum(((feat - mean) * mask) ** 2,
                                dim=[2, 3], keepdim=True)
    std = torch.sqrt(sqr / (num + eps) + eps)
    return mean, std

class RAIN(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.fg_gamma = nn.Parameter(torch.zeros(1, nf, 1, 1))
        self.fg_beta = nn.Parameter(torch.zeros(1, nf, 1, 1))
        self.bg_gamma = nn.Parameter(torch.zeros(1, nf, 1, 1))
        self.bg_beta = nn.Parameter(torch.zeros(1, nf, 1, 1))
    def forward(self, feat, mask):
        in_size = feat.size()[2:]
        mask = F.interpolate(mask.detach(), in_size, mode='nearest')
        rev_mask = 1 - mask
        mean_bg, std_bg = get_masked_mean_std(feat, rev_mask)
        normed_bg = (feat - mean_bg) / std_bg
        affine_bg = (normed_bg * (1 + self.bg_gamma) + self.bg_beta) * rev_mask
        mean_fg, std_fg = get_masked_mean_std(feat, mask)
        # 利用背景的统计量对前景进行类似风格迁移操作
        normed_fg = (feat - mean_fg) / std_fg * std_bg + mean_bg
        affine_fg = (normed_fg * (1 + self.fg_gamma) + self.fg_beta) * mask
        out = affine_fg + affine_bg
        print(f"mean_fg: {mean_fg[0, :4, 0, 0]}")
        print(f"mean_bg: {mean_bg[0, :4, 0, 0]}")
        print(f"std_fg: {std_fg[0, :4, 0, 0]}")
        print(f"std_bg: {std_bg[0, :4, 0, 0]}")
        return out

if __name__ == "__main__":
    feat = torch.randn(4, 64, 128, 128)
    mask = torch.randint(0, 3, size=(4, 1, 128, 128))
    mask = mask.float()
    mask = torch.clamp(mask, 0, 1)
    rain_norm = RAIN(nf=64)
    out = rain_norm(feat, mask)
    print(f"RAIN output size: {out.size()}")
