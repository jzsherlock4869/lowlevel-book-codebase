import torch
import torch.nn as nn
import torch.nn.functional as F

## 1. 实现可求导的导向滤波
# 1.1 实现导向滤波中用到的均值滤波
def window_sum(x, rh, rw):
    # x.size(): [n, c, h, w]
    cum_h = torch.cumsum(x, dim=2)
    wh = 2 * rh + 1
    top = cum_h[..., rh: wh, :]
    midh = cum_h[..., wh:, :] - cum_h[..., :-wh, :]
    bot = cum_h[..., -1:, :] - cum_h[..., -wh: -rh-1, :]
    out = torch.cat([top, midh, bot], dim=2)
    cum_w = torch.cumsum(out, dim=3)
    ww = 2 * rw + 1
    left = cum_w[..., rw: ww]
    midw = cum_w[..., ww:] - cum_w[..., :-ww]
    right = cum_w[..., -1:] - cum_w[..., -ww: -rw-1]
    out = torch.cat([left, midw, right], dim=3)
    return out

class BoxFilter(nn.Module):
    def __init__(self, rh, rw):
        super().__init__()
        self.rh, self.rw = rh, rw
    def forward(self, x):
        onemap = torch.ones_like(x)
        win_sum = window_sum(x, self.rh, self.rw)
        count = window_sum(onemap, self.rh, self.rw)
        box_out = win_sum / count
        return box_out

# 1.2 快速导向滤波
class FastGuidedFilter(nn.Module):
    def __init__(self, radius, eps):
        super().__init__()
        self.r = radius
        self.eps = eps
        self.mean = BoxFilter(radius, radius)
    def forward(self, p_lr, I_lr, I_hr):
        H, W = I_hr.size()[2:]
        mean_I = self.mean(I_lr)
        mean_p = self.mean(p_lr)
        cov_Ip = self.mean(I_lr * p_lr) - mean_I * mean_p
        var_I = self.mean(I_lr * I_lr) - mean_I ** 2
        A_lr = cov_Ip / (var_I + self.eps)
        B_lr = mean_p - A_lr * mean_I
        A_hr = F.interpolate(A_lr, (H, W), mode='bilinear')
        B_hr = F.interpolate(B_lr, (H, W), mode='bilinear')
        out = A_hr * I_hr + B_hr
        return out

## 2. 实现MEF-Net网络架构
# 2.1 基础模块: AN 和 CAN 网络结构
class AdaptiveNorm(nn.Module):
    def __init__(self, nf):
        super(AdaptiveNorm, self).__init__()
        self.w0 = nn.Parameter(torch.Tensor([1.0]))
        self.w1 = nn.Parameter(torch.Tensor([0.0]))
        self.instnorm = nn.InstanceNorm2d(nf,
                                affine=True, 
                                track_running_stats=False)
    def forward(self, x):
        out = self.w0 * x + self.w1 * self.instnorm(x)
        return out

class ContextAggregationNet(nn.Module):
    def  __init__(self, num_layers=7, nf=24):
        super().__init__()
        layers = list()
        for i in range(num_layers - 1):
            in_ch = 1 if i == 0 else nf
            dil = 2 ** i if i < num_layers -2 else 1
            layers += [
                nn.Conv2d(in_ch, nf, 3,
                          stride=1,
                          padding=dil,
                          dilation=dil,
                          bias=False),
                AdaptiveNorm(nf),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        layers.append(
            nn.Conv2d(nf, 1, 1, 1, 0, bias=True)
        )
        self.body = nn.Sequential(*layers)
    def forward(self, x):
        out = self.body(x)
        return out


# 2.2 MEF-Net计算多曝光融合
class MEFNet(nn.Module):
    def __init__(self, radius=2,
                 eps=1e-4,
                 num_layers=7,
                 nf=24):
        super().__init__()
        self.lr_net = ContextAggregationNet(num_layers, nf)
        self.gf = FastGuidedFilter(radius, eps)

    def forward(self, x_lr, x_hr):
        w_lr = self.lr_net(x_lr)
        w_hr = self.gf(w_lr, x_lr, x_hr)
        w_hr = torch.abs(w_hr)
        w_hr = (w_hr + 1e-8) / torch.sum(w_hr + 1e-8, dim=0)
        o_hr = torch.sum(w_hr * x_hr, dim=0)
        o_hr = o_hr.unsqueeze(0).clamp(0, 1)
        return o_hr, w_hr


if __name__ == "__main__":
    mfs = torch.rand(4, 1, 256, 256)
    mfs_ds = F.interpolate(mfs, (64, 64), mode="bilinear")
    mefnet = MEFNet()
    o_hr, w_hr = mefnet(mfs_ds, mfs)
    print(o_hr.size())
    print(w_hr.size())

