import torch
import torch.nn as nn

class MaxOut(nn.Module):
    def __init__(self, nf, out_nc):
        super().__init__()
        assert nf % out_nc == 0
        self.nf = nf
        self.out_nc = out_nc
    def forward(self, x):
        n, c, h, w = x.size()
        assert self.nf == c
        stacked = x.reshape((n, 
            self.out_nc, self.nf // self.out_nc, h, w))
        out = torch.max(stacked, dim=2)[0] # [n, out_nc, h, w]
        return out

class DehazeNet(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()
        nf1, nf2, nf3 = 16, 4, 16
        self.feat_extract = nn.Conv2d(in_ch, nf1, 5, 1, 0)
        self.maxout = MaxOut(nf1, nf2)
        self.multi_map = nn.ModuleList([
            nn.Conv2d(nf2, nf3, 3, 1, 1),
            nn.Conv2d(nf2, nf3, 5, 1, 2),
            nn.Conv2d(nf2, nf3, 7, 1, 3)
        ])
        self.maxpool = nn.MaxPool2d(7, stride=1)
        self.conv_out = nn.Conv2d(nf3 * 3, 1, 6)
        self.brelu = nn.Hardtanh(0, 1, inplace=True)
    
    def forward(self, x):
        batchsize = x.size()[0]
        out1 = self.feat_extract(x)
        out2 = self.maxout(out1)
        out3_1 = self.multi_map[0](out2)
        out3_2 = self.multi_map[1](out2)
        out3_3 = self.multi_map[2](out2)
        out3 = torch.cat([out3_1, out3_2, out3_3], dim=1)
        out4 = self.maxpool(out3)
        out5 = self.conv_out(out4)
        out6 = self.brelu(out5)
        # 打印各级输出的特征图大小
        print('[DehazeNet] out1 - out6 sizes:')
        print(out1.size(), out2.size())
        print(out3.size(), out4.size())
        print(out5.size(), out6.size())
        return out6.reshape(batchsize, -1)


if __name__ == "__main__":
    dummy_patch = torch.randn(4, 3, 16, 16)
    dehazenet = DehazeNet(in_ch=3)
    print("DehazeNet architecture: ")
    print(dehazenet)
    pred = dehazenet(dummy_patch)
    print('DehazeNet input size: ', dummy_patch.size())
    print('DehazeNet output size: ', pred.size())

