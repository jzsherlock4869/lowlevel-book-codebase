import torch
import torch.nn as nn

# torch.Tensor 实现 DWT 和 IDWT 操作
# 等价于上述通过4个不同的2x2卷积实现
class DWT(nn.Module):
    def __init__(self):
        super().__init__()
        self.requires_grad = False
    def forward(self, x):
        x1 = x[:, :, 0::2, 0::2] / 2
        x2 = x[:, :, 1::2, 0::2] / 2
        x3 = x[:, :, 0::2, 1::2] / 2
        x4 = x[:, :, 1::2, 1::2] / 2
        LL = x1 + x2 + x3 + x4
        LH = (x2 + x4) - (x1 + x3)
        HL = (x3 + x4) - (x1 + x2)
        HH = (x1 + x4) - (x2 + x3)
        out = torch.cat((LL, LH, HL, HH), dim=1)
        return out

class IDWT(nn.Module):
    def __init__(self):
        super().__init__()
        self.requires_grad = False
    def forward(self, x):
        n, c, h, w = x.size()
        out_c = c // 4
        out_h, out_w = h * 2, w * 2
        LL = x[:, 0*out_c: 1*out_c, ...] / 2
        LH = x[:, 1*out_c: 2*out_c, ...] / 2
        HL = x[:, 2*out_c: 3*out_c, ...] / 2
        HH = x[:, 3*out_c: 4*out_c, ...] / 2
        x1 = (LL + HH) - (LH + HL)
        x2 = (LL + LH) - (HL + HH)
        x3 = (LL + HL) - (LH + HH)
        x4 = LL + LH + HL + HH
        out = torch.zeros(n, out_c, out_h, out_w,
                                dtype=torch.float32)
        out[:, :, 0::2, 0::2] = x1
        out[:, :, 1::2, 0::2] = x2
        out[:, :, 0::2, 1::2] = x3
        out[:, :, 1::2, 1::2] = x4
        return out


# MWCNN中的下采样阶段（编码器）中的组成模块
class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilate1=2, dilate2=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, dilate1, 
                     dilation=dilate1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, dilate2,
                     dilation=dilate2),
            nn.ReLU(inplace=True)
        )
            
    def forward(self, x):
        out = self.body(x)
        return out


# MWCNN的上采样阶段（解码器）中的组成模块
class InvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilate1=2, dilate2=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, 1, dilate1,
                     dilation=dilate1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, 1, dilate2,
                     dilation=dilate2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        out = self.body(x)
        return out

# 输出模块
class OutBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, 1, 2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, 1, 1, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        )
    
    def forward(self, x):
        out = self.body(x)
        return out


class MWCNN(nn.Module):
    """
    MWCNN 主网络, 采用 DWT 和 IDWT 实现特征图下/上采样
    """
    def __init__(self, in_ch, nf):
        super().__init__()
        self.dwt = DWT()
        self.idwt = IDWT()
        self.dl0 = DownBlock(in_ch, nf, 2, 1)
        self.dl1 = DownBlock(4*nf, 2*nf, 2, 1)
        self.dl2 = DownBlock(8*nf, 4*nf, 2, 1)
        self.dl3 = DownBlock(16*nf, 8*nf, 2, 3)
        self.il3 = InvBlock(8*nf, 16*nf, 3, 2)
        self.il2 = InvBlock(4*nf, 8*nf, 2, 1)
        self.il1 = InvBlock(2*nf, 4*nf, 2, 1)
        self.outblock = OutBlock(nf, in_ch)

    def forward(self, x):
        x0 = self.dl0(x)
        x1 = self.dl1(self.dwt(x0))
        x2 = self.dl2(self.dwt(x1))
        x3 = self.dl3(self.dwt(x2))
        x3 = self.il3(x3)
        x_r2 = self.idwt(x3) + x2
        x_r1 = self.idwt(self.il2(x_r2)) + x1
        x_r0 = self.idwt(self.il1(x_r1)) + x0
        out = self.outblock(x_r0) + x
        print("[MWCNN] downscale sizes:")
        print(x0.size(), x1.size(), x2.size(), x3.size())
        print("[MWCNN] reconstruction sizes:")
        print(x_r2.size(), x_r1.size(), x_r0.size(), out.size())
        return out



if __name__ == "__main__":

    dummy_in = torch.randn(1, 3, 64, 64)
    # 测试 DWT 和 IDWT
    dwt = DWT()
    idwt = IDWT()
    dwt_out = dwt(dummy_in)
    idwt_recon = idwt(dwt_out)
    is_equal = torch.allclose(dummy_in, idwt_recon, atol=1e-6)
    print("Is equal after DWT and IDWT? ", is_equal)

    # 测试 MWCNN 网络
    mwcnn = MWCNN(in_ch=3, nf=32)
    output = mwcnn(dummy_in)