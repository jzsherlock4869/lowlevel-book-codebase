import torch
import torch.nn as nn
import torch.nn.functional as F

class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        kh, kw = self.kernel_size
        self.mask_weight = torch.ones(1, 1, kh, kw)
        self.slide_winsize = kh * kw
    def forward(self, feat, mask):
        with torch.no_grad():
            self.mask_weight = self.mask_weight.to(mask)
            # 更新 update_mask，邻域有前景的都置为前景
            update_mask = F.conv2d(mask, self.mask_weight,
                                    stride=self.stride,
                                    padding=self.padding,
                                    dilation=self.dilation,
                                    bias=None)
            # 计算 sum(1) / sum(M)
            mask_ratio = self.slide_winsize / (update_mask + 1e-8)
            # 用 update_mask 扩充更新输入 mask
            update_mask = torch.clamp(update_mask, 0, 1)
            # 计算输出各点的缩放比例
            mask_ratio = torch.mul(mask_ratio, update_mask)
        # 正常 Conv2d 的原始输出
        masked_feat = feat * mask
        conv_out = super().forward(masked_feat)
        # partial conv 的 mask 边缘缩放
        if self.bias is None:
            out = torch.mul(conv_out, mask_ratio)
        else:
            b = self.bias.view(1, self.out_channels, 1, 1)
            out = torch.mul(conv_out - b, mask_ratio) + b
            out = torch.mul(out, update_mask)
        # 打印相关中间变量信息
        # print("[PartialConv2d] mask_weight: \n", self.mask_weight)
        # print("[PartialConv2d] mask: \n", mask[0, 0, 4:9, 4:9])
        # print("[PartialConv2d] updated mask: \n", \
        #                         update_mask[0, 0, 4:9, 4:9])
        # print("[PartialConv2d] mask_ratio: \n", \
        #                         mask_ratio[0, 0, 4:9, 4:9])
        return out, update_mask


if __name__ == "__main__":
    feat = torch.randn(4, 64, 128, 128)
    mask = torch.randint(0, 3, size=(4, 1, 128, 128))
    mask = torch.clamp(mask, 0, 1)
    partialconv = PartialConv2d(64, 32, 3, 1, 1)
    out, new_mask = partialconv(feat, mask)
    print(f"partial conv outputs: \n"\
          f" out size: {tuple(out.size())}\n"\
          f" new_mask size: {tuple(new_mask.size())}")

