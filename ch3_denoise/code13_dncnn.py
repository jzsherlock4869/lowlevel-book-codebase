import torch
import torch.nn as nn

class ConvBNReLU(nn.Module):
    def __init__(self, nf, kernel_size):
        super(ConvBNReLU, self).__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(nf, nf, kernel_size=kernel_size, padding=pad)
        self.bn = nn.BatchNorm2d(nf)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DnCNN(nn.Module):
    def __init__(self, img_nc=3, nf=64, num_layers=17):
        super(DnCNN, self).__init__()
        self.in_conv = nn.Conv2d(img_nc, nf, kernel_size=3, padding=1) 
        self.body = nn.Sequential(
            *[ConvBNReLU(nf, 3) for _ in range(num_layers - 2)]
            )
        self.out_conv = nn.Conv2d(nf, img_nc, kernel_size=3, padding=1)
    def forward(self, x):
        noisy = x
        x = self.in_conv(x)
        x = self.body(x)
        pred_noise = self.out_conv(x)
        return noisy - pred_noise

if __name__ == "__main__":
    x_in = torch.randn((1, 1, 128, 128))
    dncnn = DnCNN(img_nc=1, nf=64, num_layers=17)
    print(dncnn)
    x_out = dncnn(x_in)
    print('DnCNN input size: ', x_in.size())
    print('DnCNN output size: ', x_out.size())
