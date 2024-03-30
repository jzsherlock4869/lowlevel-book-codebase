import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.partialconv2d import PartialConv2d

class DomainEncoder(nn.Module):
    def __init__(self, nf=64, enc_dim=16):
        super().__init__()
        nfs = [nf * 2 ** i for i in range(4)]
        self.relu = nn.ReLU(inplace=True)
        # conv1 + relu
        self.conv1 = PartialConv2d(3, nfs[0], 3, 2, 0)
        # conv2 + norm2 + relu
        self.conv2 = PartialConv2d(nfs[0], nfs[1], 3, 2, 0)
        self.norm2 = nn.BatchNorm2d(nfs[1])
        # conv3 + norm3 + relu
        self.conv3 = PartialConv2d(nfs[1], nfs[2], 3, 2, 0)
        self.norm3 = nn.BatchNorm2d(nfs[2])
        # conv4 + norm4 + relu
        self.conv4 = PartialConv2d(nfs[2], nfs[3], 3, 2, 0)
        self.norm4 = nn.BatchNorm2d(nfs[3])
        # conv5 + avg_pool + conv_style
        self.conv5 = PartialConv2d(nfs[3], nfs[3], 3, 2, 0)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_style = nn.Conv2d(nfs[3], enc_dim, 1, 1, 0)
    def forward(self, img, mask):
        x, m = img, mask
        x, m = self.conv1(x, m)
        x = self.relu(x)
        x, m = self.conv2(x, m)
        x = self.relu(self.norm2(x))
        x, m = self.conv3(x, m)
        x = self.relu(self.norm3(x))
        x, m = self.conv4(x, m)
        x = self.relu(self.norm4(x))
        x, _ = self.conv5(x, m)
        x = self.avg_pool(x)
        style_code = self.conv_style(x)
        return style_code


if __name__ == "__main__":
    img_comp = torch.randn(4, 3, 128, 128)
    img_harm = torch.randn(4, 3, 128, 128)
    img_gt = torch.randn(4, 3, 128, 128)
    mask = torch.randint(0, 3, size=(4, 1, 128, 128))
    mask = mask.float()
    mask = torch.clamp(mask, 0, 1)
    rev_mask = 1 - mask
    domain_encoder = DomainEncoder()
    tri_loss_func = nn.TripletMarginLoss(margin=0.1)
    # 计算不同图像和区域的域编码向量
    bg_vec = domain_encoder(img_gt, rev_mask)
    fg_gt_vec = domain_encoder(img_gt, mask)
    fg_comp_vec = domain_encoder(img_comp, mask)
    fg_harm_vec = domain_encoder(img_harm, mask)
    # 计算Triplet损失
    triloss_1 = tri_loss_func(fg_harm_vec, bg_vec, fg_comp_vec)
    triloss_2 = tri_loss_func(fg_gt_vec, fg_harm_vec, fg_comp_vec)
    print("Triplet loss 1 : ", triloss_1)
    print("Triplet loss 2 : ", triloss_2)
