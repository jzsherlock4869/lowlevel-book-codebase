import torch
from torch import nn
from thop import profile

class edgeSR_MAX(nn.Module):
    def __init__(self, channels=2, ksize=5, stride=2):
        super().__init__()
        self.pixelshuffle = nn.PixelShuffle(stride)
        pad = ksize // 2
        self.filter = nn.Conv2d(1, 
                        stride ** 2 * channels,
                        ksize, 1, pad)
        nn.init.xavier_normal_(self.filter.weight, gain=1.0)
        self.filter.weight.data[:, :, pad, pad] = 1.0
    def forward(self, x):
        out = self.filter(x)
        out = self.pixelshuffle(out)
        out = torch.max(out, dim=1, keepdim=True)[0]
        return out


class edgeSR_TM(nn.Module):
    def __init__(self, channels=2, ksize=5, stride=2):
        super().__init__()
        self.pixelshuffle = nn.PixelShuffle(stride)
        self.softmax = nn.Softmax(dim=1)
        pad = ksize // 2
        self.ch = channels
        self.filter = nn.Conv2d(1, 
                        2 * stride ** 2 * channels,
                        ksize, 1, pad)
        nn.init.xavier_normal_(self.filter.weight, gain=1.0)
        self.filter.weight.data[:, :, pad, pad] = 1.0
    def forward(self, x):
        out = self.filter(x)
        out = self.pixelshuffle(out)
        k, v = torch.split(out, [self.ch, self.ch], dim=1)
        weight = self.softmax(k)
        out = torch.sum(weight * v, dim=1, keepdim=True)
        return out


class edgeSR_TR(nn.Module):
    def __init__(self, channels=2, ksize=5, stride=2):
        super().__init__()
        self.pixelshuffle = nn.PixelShuffle(stride)
        self.softmax = nn.Softmax(dim=1)
        pad = ksize // 2
        self.ch = channels
        self.filter = nn.Conv2d(1, 
                        3 * stride ** 2 * channels,
                        ksize, 1, pad)
        nn.init.xavier_normal_(self.filter.weight, gain=1.0)
        self.filter.weight.data[:, :, pad, pad] = 1.0
    def forward(self, x):
        out = self.filter(x)
        out = self.pixelshuffle(out)
        q, v, k = torch.split(out,
                              [self.ch, self.ch, self.ch], dim=1)
        weight = self.softmax(q * k)
        out = torch.sum(weight * v, dim=1, keepdim=True)
        return out


if __name__ == "__main__":
    x_in = torch.randn(1, 1, 128, 128)
    esr_max = edgeSR_MAX(2, 5, 2)
    out_max = esr_max(x_in)
    esr_tm = edgeSR_TM(2, 5, 2)
    out_tm = esr_tm(x_in)
    esr_tr = edgeSR_TR(2, 5, 2)
    out_tr = esr_tr(x_in)
    
    # 对比计算量与参数量
    flops, params = profile(esr_max, inputs=(x_in, ))
    print(f'edgeSR-MAX profile: \n {flops/1000**2:.4f}M flops, '\
          f'{params/1000:.4f}K params, '\
          f'output size: {list(out_max.size())}')
    flops, params = profile(esr_tm, inputs=(x_in, ))
    print(f'edgeSR-TM profile: \n {flops/1000**2:.4f}M flops, '\
          f'{params/1000:.4f}K params, '\
          f'output size: {list(out_tm.size())}')
    flops, params = profile(esr_tr, inputs=(x_in, ))
    print(f'edgeSR-TR profile: \n {flops/1000**2:.4f}M flops, '\
          f'{params/1000:.4f}K params, '\
          f'output size: {list(out_tr.size())}')

