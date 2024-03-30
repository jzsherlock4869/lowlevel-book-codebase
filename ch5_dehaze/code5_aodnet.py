import torch
import torch.nn as nn

class AODNet(nn.Module):
    def __init__(self, b=1.0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv2 = nn.Conv2d(3, 3, 3, 1, 1)
        self.conv3 = nn.Conv2d(6, 3, 5, 1, 2)
        self.conv4 = nn.Conv2d(6, 3, 7, 1, 3)
        self.conv5 = nn.Conv2d(12, 3, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.b = b

    def forward(self, x):
        out1 = self.relu(self.conv1(x))
        out2 = self.relu(self.conv2(out1))
        cat1 = torch.cat([out1, out2], dim=1)
        out3 = self.relu(self.conv3(cat1))
        cat2 = torch.cat([out2, out3], dim=1)
        out4 = self.relu(self.conv4(cat2))
        cat3 = torch.cat([out1, out2, out3, out4], dim=1)
        k_est = self.relu(self.conv5(cat3))
        output = k_est * x + k_est + self.b
        return output


if __name__ == "__main__":
    x_in = torch.randn(4, 3, 64, 64)
    aodnet = AODNet(b=1)
    out = aodnet(x_in)
    print('AODNet input size: ', x_in.size())
    print('AODNet output size: ', out.size())

