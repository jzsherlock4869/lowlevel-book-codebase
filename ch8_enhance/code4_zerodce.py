import torch
import torch.nn as nn

class DCENet(nn.Module):
    def __init__(self, nf=32, n_iter=8):
        super().__init__()
        self.n_iter = n_iter
        # 编码器（encoder）
        self.conv1 = nn.Conv2d(3, nf, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf, nf, 3, 1, 1)
        # 解码器（decoder）
        self.deconv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1)
        self.deconv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1)
        self.deconv3 = nn.Conv2d(nf * 2, n_iter * 3, 3, 1, 1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 编码过程
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        # 解码过程
        xcat = torch.cat([x3, x4], dim=1)
        x5 = self.relu(self.deconv1(xcat))
        xcat = torch.cat([x2, x5], dim=1)
        x6 = self.relu(self.deconv2(xcat))
        xcat = torch.cat([x1, x6], dim=1)
        xr = self.tanh(self.deconv3(xcat))
        # 各个阶段的曲线图（curve map）
        curves = torch.split(xr, 3, dim=1)
        # 应用曲线进行提亮
        for i in range(self.n_iter):
            x = x + curves[i] * (torch.pow(x, 2) - x)
        # 返回提亮结果与曲线图
        A = torch.cat(curves, dim=1)
        return x, A


if __name__ == "__main__":
    img_low = torch.randn(4, 3, 128, 128)
    dcenet = DCENet()
    enhanced, curve_map = dcenet(img_low)
    print(f"DCE input size: ", enhanced.size())
    print(f"Enhanced size: ", enhanced.size())
    print(f"Curve map size: ", curve_map.size())
