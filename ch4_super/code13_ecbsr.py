import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1x1Conv3x3(nn.Module):
    def __init__(self, in_ch, out_ch, mult=1.0):
        super().__init__()
        mid_ch = int(out_ch * mult)
        self.mid_ch = mid_ch
        conv1x1 = nn.Conv2d(in_ch, mid_ch, 1, 1, 0)
        conv3x3 = nn.Conv2d(mid_ch, out_ch, 3, 1, 1)
        self.k0, self.b0 = conv1x1.weight, conv1x1.bias
        self.k1, self.b1 = conv3x3.weight, conv3x3.bias
    def forward(self, x):
        # conv 1x1 (input, weight, bias, stride, pad)
        y0 = F.conv2d(x, self.k0, self.b0, 1, 0)
        y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
        b0_pad = self.b0.view(1, -1, 1, 1)
        y0[:, :, 0:1, :] = b0_pad
        y0[:, :, -1:, :] = b0_pad
        y0[:, :, :, 0:1] = b0_pad
        y0[:, :, :, -1:] = b0_pad
        out = F.conv2d(y0, self.k1, self.b1, 1, 0)
        return out
    def rep_params(self):
        # F.conv2d 默认 padding=0
        rep_k = F.conv2d(self.k1, self.k0.permute(1, 0, 2, 3))
        rep_b = self.b0.reshape(1, -1, 1, 1).tile(1, 1, 3, 3)
        # rep_b尺寸 3x3, k1尺寸 3x3, 输出尺寸 1x1
        rep_b = F.conv2d(rep_b, self.k1).view(-1,) + self.b1
        return rep_k, rep_b


def gen_edge_tensor(nc, mode='sobel_x'):
    mask = torch.zeros((nc, 1, 3, 3), dtype=torch.float32)
    for i in range(nc):
        if mode == 'sobel_x':
            mask[i, 0, 0, 0] = 1.0
            mask[i, 0, 1, 0] = 2.0
            mask[i, 0, 2, 0] = 1.0
            mask[i, 0, 0, 2] = -1.0
            mask[i, 0, 1, 2] = -2.0
            mask[i, 0, 2, 2] = -1.0
        elif mode == 'sobel_y':
            mask[i, 0, 0, 0] = 1.0
            mask[i, 0, 0, 1] = 2.0
            mask[i, 0, 0, 2] = 1.0
            mask[i, 0, 2, 0] = -1.0
            mask[i, 0, 2, 1] = -2.0
            mask[i, 0, 2, 2] = -1.0
        else:
            assert mode == 'laplacian'
            mask[i, 0, 0, 1] = 1.0
            mask[i, 0, 1, 0] = 1.0
            mask[i, 0, 1, 2] = 1.0
            mask[i, 0, 2, 1] = 1.0
            mask[i, 0, 1, 1] = -4.0
    return mask


class Conv1x1SobelLaplacian(nn.Module):
    def __init__(self, in_ch, out_ch, mode='sobel_x'):
        # mode: ['sobel_x', 'sobel_y', 'laplacian']
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        conv1x1 = nn.Conv2d(in_ch, out_ch, 1, 1, 0)
        self.k0, self.b0 = conv1x1.weight, conv1x1.bias
        scale = torch.randn(out_ch, 1, 1, 1) * 1e-3
        self.scale = nn.Parameter(scale)
        bias = torch.randn(out_ch) * 1e-3
        self.bias = nn.Parameter(bias)
        mask = gen_edge_tensor(out_ch, mode)
        self.mask = nn.Parameter(mask, requires_grad=False)
    def forward(self, x):
        y0 = F.conv2d(x, self.k0, self.b0, 1, 0)
        y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
        b0_pad = self.b0.view(1, -1, 1, 1)
        y0[:, :, 0:1, :] = b0_pad
        y0[:, :, -1:, :] = b0_pad
        y0[:, :, :, 0:1] = b0_pad
        y0[:, :, :, -1:] = b0_pad
        out = F.conv2d(y0, self.scale * self.mask, 
                      self.bias, 1, 0, groups=self.out_ch)
        return out
    def rep_params(self):
        k1 = torch.zeros(self.out_ch, self.out_ch, 3, 3)
        scaled_mask = self.scale * self.mask
        for i in range(self.out_ch):
            k1[i, i, :, :] = scaled_mask[i, 0, :, :]
        rep_k = F.conv2d(k1, self.k0.permute(1, 0, 2, 3))
        rep_b = self.b0.reshape(1, -1, 1, 1).tile(1, 1, 3, 3)
        rep_b = F.conv2d(rep_b, k1).view(-1,) + self.bias
        return rep_k, rep_b


class EdgeOrintedConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, mult):
        super().__init__()
        self.conv3x3 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv1x1_3x3 = Conv1x1Conv3x3(in_ch, out_ch, mult)
        self.conv1x1_sbx = Conv1x1SobelLaplacian(in_ch, out_ch, 'sobel_x')
        self.conv1x1_sby = Conv1x1SobelLaplacian(in_ch, out_ch, 'sobel_y')
        self.conv1x1_lap = Conv1x1SobelLaplacian(in_ch, out_ch, 'laplacian')
        self.prelu = nn.PReLU(num_parameters=out_ch)
    def forward(self, x):
        if self.training:
            print("[ECB] use multi-branch params")
            out = self.conv3x3(x)
            out += self.conv1x1_3x3(x)
            out += self.conv1x1_sbx(x)
            out += self.conv1x1_sby(x)
            out += self.conv1x1_lap(x)
        else:
            print("[ECB] use reparameterized params")
            rep_k, rep_b = self.rep_params()
            out = F.conv2d(x, rep_k, rep_b, 1, 1)
        out = self.prelu(out)
        return out
    def rep_params(self):
        k0, b0 = self.conv3x3.weight, self.conv3x3.bias
        k1, b1 = self.conv1x1_3x3.rep_params()
        k2, b2 = self.conv1x1_sbx.rep_params()
        k3, b3 = self.conv1x1_sby.rep_params()
        k4, b4 = self.conv1x1_lap.rep_params()
        rep_k = k0 + k1 + k2 + k3 + k4
        rep_b = b0 + b1 + b2 + b3 + b4
        return rep_k, rep_b


if __name__ == "__main__":

    # 测试边缘提取算子（sobel 和 laplacian）
    sobel_x = gen_edge_tensor(2, 'sobel_x')
    sobel_y = gen_edge_tensor(2, 'sobel_y')
    laplacian = gen_edge_tensor(2, 'laplacian')
    print("Sobel x: \n", sobel_x)
    print("Sobel y: \n", sobel_y)
    print("Laplacian: \n", laplacian)

    x_in = torch.randn(2, 64, 8, 8)
    # 测试 ECB 模块的计算与重参数化结果
    ecb = EdgeOrintedConvBlock(in_ch=64, out_ch=32, mult=2.0)
    out = ecb(x_in)
    print('ECB train mode: ', ecb.training)
    print('output train (slice): \n', out[0, 0, :4, :4])
    print('output train size: ', out.size())
    with torch.no_grad():
        ecb.eval()
        print('ECB train mode: ', ecb.training)
        out_rep = ecb(x_in)
        print('output inference (slice): \n', out_rep[0, 0, :4, :4])
        print('output inference size: ', out_rep.size())

    print('is reparam output and multi-branch the same ?',
        torch.allclose(out, out_rep, atol=1e-6))

