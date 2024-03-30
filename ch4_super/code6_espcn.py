import torch
import torch.nn as nn

class ESPCN(nn.Module):
    """
    ESPCN网络, 通过PixelShuffle实现上采样
    """
    def __init__(self, in_ch, nf, factor=4):
        super().__init__()
        hid_nf = nf // 2
        out_ch = in_ch * (factor ** 2)
        self.conv1 = nn.Conv2d(in_ch, nf, 5, 1, 2)
        self.conv2 = nn.Conv2d(nf, hid_nf, 3, 1, 1)
        self.conv3 = nn.Conv2d(hid_nf, out_ch, 3, 1, 1)
        self.pixshuff = nn.PixelShuffle(factor)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.conv1(x))
        x = self.tanh(self.conv2(x))
        x = self.conv3(x)
        out = self.pixshuff(x)
        return out


if __name__ == "__main__":
    x_in = torch.randn(4, 3, 64, 64)
    espcn = ESPCN(in_ch=3, nf=64, factor=4)
    x_out = espcn(x_in)
    print('ESPCN input size: ', x_in.size())
    print('ESPCN output size: ', x_out.size())
