import torch
import torch.nn as nn
import torch.nn.functional as F

class DecomNet(nn.Module):
    def __init__(self, nf=64, ksize=3, n_layer=5):
        super().__init__()
        layers = list()
        pad = ksize * 3 // 2
        layers.append(nn.Conv2d(4, nf, ksize*3, 1, pad))
        pad = ksize // 2
        for _ in range(n_layer):
            layers.append(nn.Conv2d(nf, nf, ksize, 1, pad))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(nf, 4, ksize, 1, pad))
        layers.append(nn.Sigmoid())
        self.body = nn.Sequential(*layers)
    def forward(self, x):
        input_max = torch.max(x, dim=1, keepdim=True)[0]
        input_im = torch.cat((x, input_max), dim=1)
        out = self.body(input_im)
        R, L = torch.split(out, [3, 1], dim=1)
        return R, L

class RelightNet(nn.Module):
    def __init__(self, nf=64, ksize=3):
        super().__init__()
        pad = ksize // 2
        self.conv0 = nn.Conv2d(4, nf, ksize, 1, pad)
        self.conv1 = nn.Conv2d(nf, nf, ksize, 2, pad)
        self.conv2 = nn.Conv2d(nf, nf, ksize, 2, pad)
        self.conv3 = nn.Conv2d(nf, nf, ksize, 2, pad)
        self.deconv1 = nn.Conv2d(nf * 2, nf, ksize, 1, pad)
        self.deconv2 = nn.Conv2d(nf * 2, nf, ksize, 1, pad)
        self.deconv3 = nn.Conv2d(nf * 2, nf, ksize, 1, pad)
        self.fusion = nn.Conv2d(nf * 3, nf, 1, 1, 0)
        self.conv_out = nn.Conv2d(nf, 1, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, R, L):
        x_in = torch.cat((R, L), dim=1)
        out0 = self.conv0(x_in)
        # 下采样，encoder过程
        out1 = self.relu(self.conv1(out0))
        out2 = self.relu(self.conv2(out1))
        out3 = self.relu(self.conv3(out2))
        # 上采样 + 跳接，decoder过程
        target_size = (out2.size()[2], out2.size()[3])
        up3 = F.interpolate(out3, size=target_size)
        up3_ex = torch.cat((up3, out2), dim=1)
        dout1 = self.relu(self.deconv1(up3_ex))
        target_size = (out1.size()[2], out1.size()[3])
        up2 = F.interpolate(dout1, size=target_size)
        up2_ex = torch.cat((up2, out1), dim=1)
        dout2 = self.relu(self.deconv2(up2_ex))
        target_size = (out0.size()[2], out0.size()[3])
        up1 = F.interpolate(dout2, size=target_size)
        up1_ex = torch.cat((up1, out0), dim=1)
        dout3 = self.relu(self.deconv3(up1_ex))
        # 特征融合
        target_size = (L.size()[2], L.size()[3])
        dout1_up = F.interpolate(dout1, size=target_size)
        dout2_up = F.interpolate(dout2, size=target_size)
        tot = torch.cat((dout1_up, dout2_up, dout3), dim=1)
        fused = self.fusion(tot)
        L_relight = self.conv_out(fused)
        return L_relight


class RetinexNet(nn.Module):
    def __init__(self, nf=64, ksize=3, n_layer=5):
        super().__init__()
        self.decom = DecomNet(nf, ksize, n_layer)
        self.relight = RelightNet(nf, ksize)
    def forward(self, x_low, x_normal):
        r_low, l_low = self.decom(x_low)
        r_normal, l_normal = self.decom(x_normal)
        l_relight = self.relight(r_low, l_low)
        return {
            "r_low": r_low,
            "l_low": l_low,
            "r_normal": r_normal,
            "l_normal": l_normal,
            "l_relight": l_relight
        }


if __name__ == "__main__":
    img_low = torch.randn(4, 3, 128, 128)
    img_normal = torch.randn(4, 3, 128, 128)
    retinexnet = RetinexNet()
    out_dict = retinexnet(img_low, img_normal)
    for k in out_dict:
        print(f"Retinex out {k} size: {out_dict[k].size()}")
