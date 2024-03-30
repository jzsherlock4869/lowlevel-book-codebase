import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FFDNet(nn.Module):

    def __init__(self, in_ch, out_ch, nf=64, nb=15):
        super().__init__()
        scale = 2
        self.down = nn.PixelUnshuffle(scale)
        # 输入通道数+1为噪声强度图
        module_ls = [
            nn.Conv2d(in_ch*(scale**2) + 1, nf, 3, 1, 1),
            nn.ReLU(inplace=True)
        ]
        for _ in range(nb - 2):
            cur_layer = [
                nn.Conv2d(nf, nf, 3, 1, 1),
                nn.ReLU(inplace=True)
            ]
            module_ls = module_ls + cur_layer
        module_ls.append(nn.Conv2d(nf, out_ch*(scale**2), 3, 1, 1))
        self.body = nn.Sequential(*module_ls)
        self.up = nn.PixelShuffle(scale)

    def forward(self, x_in, sigma):
        # 保证输入尺寸pad到可以2倍下采样
        h, w = x_in.shape[-2:]
        new_h = int(np.ceil(h / 2) * 2)
        new_w = int(np.ceil(w / 2) * 2)
        pad_h, pad_w = new_h - h, new_w - w
        # F.pad 顺序为 left/right/top/bottom
        x = F.pad(x_in, (0, pad_w, 0, pad_h), "replicate")
        x = self.down(x)
        sigma_map = sigma.repeat(1, 1, new_h//2, new_w//2)
        x = self.body(torch.cat((x, sigma_map), dim=1))
        x = self.up(x)
        out = x[:, :, :h, :w]
        return out


if __name__ == "__main__":
    dummy_in = torch.randn(2, 1, 64, 64)
    sigma = torch.randn(2, 1, 1, 1)
    ffdnet = FFDNet(in_ch=1, out_ch=1, nf=64, nb=15)
    print(ffdnet)
    out = ffdnet(dummy_in, sigma)
    print('FFDNet input size: ', dummy_in.size())
    print('FFDNet output size: ', out.size())
