import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ShiftByConv(nn.Module):
    def __init__(self, nf, ksize=3):
        super().__init__()
        self.weight = nn.Parameter(
            torch.zeros(1, 1, ksize, ksize, requires_grad=True)
        )
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.nf = nf
        self.ksize = ksize
        self.shift_kernel = None
    def forward(self, x):
        nc = x.size()[1]
        assert nc == self.nf
        w = self.weight.reshape(1, 1, self.ksize**2)
        is_hard = not self.training
        w = F.gumbel_softmax(w, dim=-1, hard=is_hard)
        w = w.reshape(1, 1, self.ksize, self.ksize)
        self.shift_kernel = w
        w = w.to(x.device).tile((nc, 1, 1, 1))
        pad = self.ksize // 2
        out = F.conv2d(x, w, padding=pad, groups=nc)
        return out

class GhostModule(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3,
                 intrinsic_ratio=0.5):
        super().__init__()
        self.intrinsic_ch = math.ceil(out_ch * intrinsic_ratio)
        self.ghost_ch = out_ch - self.intrinsic_ch
        pad = ksize // 2
        self.primary_conv = nn.Conv2d(in_ch, self.intrinsic_ch,
                                      ksize, 1, pad)
        self.cheap_conv = ShiftByConv(self.ghost_ch, ksize)
    def forward(self, x):
        x1 = self.primary_conv(x)
        if self.ghost_ch > self.intrinsic_ch:
            x1 = x1.repeat(1, 3, 1, 1)
        x2 = self.cheap_conv(x1[:, :self.ghost_ch, ...])
        out = torch.cat([x1[:, :self.intrinsic_ch], x2], axis=1)
        return out


if __name__ == "__main__":
    dummy_in = torch.randn(4, 16, 64, 64)
    shiftconv = ShiftByConv(nf=16)
    # 测试卷积核实现平移
    with torch.no_grad():
        shiftconv.eval()
        out = shiftconv(dummy_in)
        print("Shift Conv kernel is : \n", shiftconv.shift_kernel)
        print("Shift Conv output size: ", out.size())
    # 测试Ghost模块构建与计算
    ghostmodule = GhostModule(in_ch=16, out_ch=50, intrinsic_ratio=0.6)
    out = ghostmodule(dummy_in)
    print("Ghost module output size: ", out.size())