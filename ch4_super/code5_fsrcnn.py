import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

class SRCNN(nn.Module):
    def __init__(self, in_ch, nf=64):
        super().__init__()
        hid_nf = nf // 2
        self.conv1 = nn.Conv2d(in_ch, nf, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(nf, hid_nf, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(hid_nf, in_ch, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class ConvPReLU(nn.Module):
    def __init__(self, in_ch, out_ch, ksize):
        super().__init__()
        pad = ksize // 2
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, ksize, 1, pad),
            nn.PReLU(num_parameters=out_ch)
        )
    def forward(self, x):
        return self.body(x)


class FSRCNN(nn.Module):
    def __init__(self, in_ch=3, d=56, s=12, scale=4):
        super().__init__()
        self.feat_extract = ConvPReLU(in_ch, d, 5)
        self.shrink = ConvPReLU(d, s, 1)
        self.mapping = nn.Sequential(
            *[ConvPReLU(s, s, 3) for _ in range(4)]
        )
        self.expand = ConvPReLU(s, d, 1)
        self.deconv = nn.ConvTranspose2d(d, in_ch, 
                kernel_size=9, stride=scale, padding=4,
                output_padding=scale-1)

    def forward(self, x):
        out = self.feat_extract(x)
        out = self.shrink(out)
        out = self.mapping(out)
        out = self.expand(out)
        out = self.deconv(out)
        return out


if __name__ == "__main__":
    x_in = torch.randn(4, 3, 64, 64)
    x_in_x4 = F.interpolate(x_in, scale_factor=4)
    # 测试 SRCNN
    srcnn = SRCNN(in_ch=3, nf=64)
    srcnn_out = srcnn(x_in_x4)
    print('SRCNN output size: ', x_in_x4.shape)
    print('SRCNN output size: ', srcnn_out.shape)
    # 测试 FSRCNN
    fsrcnn = FSRCNN(in_ch=3, scale=4)
    fsrcnn_out = fsrcnn(x_in)
    print('FSRCNN output size: ', x_in.shape)
    print('FSRCNN output size: ', fsrcnn_out.shape)
    # 对比计算量与参数量
    flops, params = profile(srcnn, inputs=(x_in_x4, ))
    print(f'SRCNN profile: {flops/1000**3:.4f}G flops, '\
          f'{params/1000**2:.4f}M params')
    flops, params = profile(fsrcnn, inputs=(x_in, ))
    print(f'FSRCNN profile: {flops/1000**3:.4f}G flops, '\
          f'{params/1000**2:.4f}M params')

