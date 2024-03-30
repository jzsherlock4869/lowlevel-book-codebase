import torch
import torch.nn as nn

class ConvBNReLU(nn.Module):
    """
    基本模块: [Conv + BN + ReLU]
    stride = 2 用来实现下采样
    groups = nframe 用来实现各帧分别处理（多帧输入层）
    """
    def __init__(self, in_ch, out_ch, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch,
                        3, stride, 1,
                        groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class Down(nn.Module):
    """
    Unet 压缩部分的下采样模块
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = ConvBNReLU(in_ch, out_ch, stride=2)
        self.conv = ConvBNReLU(out_ch, out_ch)
    def forward(self, x):
        out = self.down(x)
        out = self.conv(out)
        return out

class Up(nn.Module):
    """
    Unet 扩展部分的上采样模块
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch, in_ch)
        self.conv2 = nn.Conv2d(in_ch, out_ch * 4,
                               3, 1, 1, bias=False)
        self.upper = nn.PixelShuffle(2)
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.upper(out)
        return out

class InputFrameFusion(nn.Module):
    """
     各帧分别卷积后通过卷积融合
    """
    def __init__(self, in_ch, out_ch, nframe=5, nf=30):
        super().__init__()
        self.conv_sep = ConvBNReLU(nframe * (in_ch + 1),
                                   nframe * nf,
                                   stride=1, groups=nframe)
        self.conv_fusion = ConvBNReLU(nf * nframe, out_ch)
    def forward(self, x):
        out = self.conv_sep(x)
        out = self.conv_fusion(out)
        return out


class UnetDenoiser(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()
        nf = 32
        self.in_ch = in_ch
        self.in_fusion = InputFrameFusion(in_ch, nf, nframe=3)
        self.down1 = Down(nf, nf * 2)
        self.down2 = Down(nf * 2, nf * 4)
        self.up1 = Up(nf * 4, nf * 2)
        self.up2 = Up(nf * 2, nf)
        self.conv_last = ConvBNReLU(nf, nf)
        self.conv_out = nn.Conv2d(nf, in_ch,
                                  3, 1, 1, bias=False)
    def forward(self, x, noise_map):
        # x.size(): [n, nframe(=3), c, h, w]
        # noise_map.size(): [n, 1, h, w]
        assert x.dim() == 5 and x.size()[1] == 3
        multi_in = torch.cat(
            [x[:, 0, ...], noise_map,
             x[:, 1, ...], noise_map,
             x[:, 2, ...], noise_map], dim=1)
        print(f"[UnetDenoiser] network in size: "\
              f" {multi_in.size()}")
        feat = self.in_fusion(multi_in)
        d1 = self.down1(feat)
        d2 = self.down2(d1)
        u1 = self.up1(d2)
        u2 = self.up2(u1 + d1)
        out = self.conv_last(u2 + feat)
        res = self.conv_out(out)
        pred = x[:, 1, ...] - res
        print(f"[UnetDenoiser] \n down sizes: "\
              f" {feat.size()}-{d1.size()}-{d2.size()}")
        print(f"   up sizes: {u1.size()}-{u2.size()}-{out.size()}")
        return pred


class FastDVDNet(nn.Module):
    """
    2阶段视频去噪网络, 结构均为UnetDenoiser
    """
    def __init__(self, in_ch=3):
        super().__init__()
        self.in_ch = in_ch
        self.denoiser1 = UnetDenoiser(in_ch=in_ch)
        self.denoiser2 = UnetDenoiser(in_ch=in_ch)
    
    def forward(self, x, noise_map):
        # x size: [n, nframe(=5), c, h, w]
        assert x.size()[1] == 5
        assert x.size()[2] == self.in_ch
        # stage 1
        print("====== STAGE I =======")
        out1 = self.denoiser1(x[:, 0:3, ...], noise_map)
        out2 = self.denoiser1(x[:, 1:4, ...], noise_map)
        out3 = self.denoiser1(x[:, 2:5, ...], noise_map)
        print(f"[FastDVDNet] STAGE I out sizes: \n"\
              f"{out1.size()}, {out2.size()}, {out3.size()}")
        # stage 2
        print("====== STAGE II =======")
        stage2_in = torch.stack((out1, out2, out3), dim=1)
        out = self.denoiser2(stage2_in, noise_map)
        print(f"[FastDVDNet] STAGE II out sizes: {out.size()}")
        return out

if __name__ == "__main__":
    noisy_frames = torch.randn(4, 5, 1, 128, 128)
    noise_map = torch.randn(4, 1, 128, 128)
    fastdvd = FastDVDNet(in_ch=1)
    output = fastdvd(noisy_frames, noise_map)

