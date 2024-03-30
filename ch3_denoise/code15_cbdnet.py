import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvReLU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.relu(self.conv(x))
        return out

class UpAdd(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
    def forward(self, x1, x2):
        # x1 为小尺寸特征图，x2位大尺寸特征图
        x1 = self.deconv(x1)
        diff_h = x2.size()[2] - x1.size()[2]
        diff_w = x2.size()[3] - x1.size()[3]
        pleft = diff_w // 2
        pright = diff_w - pleft
        ptop = diff_h // 2
        pbottom = diff_h - ptop
        x1 = F.pad(x1, (pleft, pright, ptop, pbottom))
        return x1 + x2


class NoiseEstNetwork(nn.Module):
    """
    噪声估计网络，全卷积形式
    """
    def __init__(self, in_ch=3, nf=32, nb=5):
        super().__init__()
        module_ls = [ConvReLU(in_ch, nf)]
        for _ in range(nb - 2):
            module_ls.append(ConvReLU(nf, nf))
        module_ls.append(ConvReLU(nf, in_ch))
        self.est = nn.Sequential(*module_ls)
        
    def forward(self, x):
        return self.est(x)


class UnetDenoiser(nn.Module):
    """
    去噪网络, UNet结构
    """
    def __init__(self, in_ch=3, nf=64):
        super().__init__()
        self.conv_in = nn.Sequential(
            ConvReLU(in_ch * 2, nf),
            ConvReLU(nf, nf)
        )
        self.down = nn.AvgPool2d(2)
        self.conv1 = nn.Sequential(
            ConvReLU(nf, nf * 2),
            *[ConvReLU(nf * 2, nf * 2) for _ in range(2)]
        )
        self.conv2 = nn.Sequential(
            ConvReLU(nf * 2, nf * 4),
            *[ConvReLU(nf * 4, nf * 4) for _ in range(5)]
        )
        self.up1 = UpAdd(nf * 4, nf * 2)
        self.conv3 = nn.Sequential(
            *[ConvReLU(nf * 2, nf * 2) for _ in range(3)]
        )
        self.up2 = UpAdd(nf * 2, nf)
        self.conv4 = nn.Sequential(
            *[ConvReLU(nf, nf) for _ in range(2)]
        )
        self.conv_out = nn.Conv2d(nf, in_ch, 3, 1, 1)

    def forward(self, x, noise_level):
        x_in = torch.cat((x, noise_level), dim=1)
        ft = self.conv_in(x_in) # nf, 1
        d1 = self.conv1(self.down(ft))  # 2nf, 1/2
        d2 = self.conv2(self.down(d1))  # 4nf, 1/4
        u1 = self.conv3(self.up1(d2, d1))  # 2nf, 1/2
        u2 = self.conv4(self.up2(u1, ft))  # nf, 1
        res = self.conv_out(u2) # nf, 1
        out = x + res
        return out


class CBDNet(nn.Module):
    def __init__(self, 
                in_ch=3,
                nf_e=32, nb_e=5,
                nf_d=64):
        super().__init__()
        self.noise_est = NoiseEstNetwork(in_ch, nf_e, nb_e)
        self.denoiser = UnetDenoiser(in_ch, nf_d)

    def forward(self, x):
        noise_level = self.noise_est(x)
        out = self.denoiser(x, noise_level)
        return out, noise_level


if __name__ == "__main__":
    dummy_in = torch.randn(4, 3, 67, 73)
    cbdnet = CBDNet()
    pred, noise_level = cbdnet(dummy_in)
    print('pred size: ', pred.size())
    print('noise_level size: ', noise_level.size())
