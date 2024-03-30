import torch
import torch.nn as nn

class DeepFuse(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat_extract = nn.Sequential(
            nn.Conv2d(1, 16, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 7, 1, 3)
        )
        self.recon = nn.Sequential(
            nn.Conv2d(32, 32, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 5, 1, 2)
        )
    def forward(self, x1, x2):
        f1 = self.feat_extract(x1)
        f2 = self.feat_extract(x2)
        f_fused = f1 + f2
        out = self.recon(f_fused)
        return out

def chroma_weight_fusion(x1, x2, tau=128):
    w1 = torch.abs(x1 - tau)
    w2 = torch.abs(x2 - tau)
    w_total = w1 + w2 + 1e-8
    w1, w2 = w1 / w_total, w2 / w_total
    w_fused = x1 * w1 + x2 * w2
    return w_fused

if __name__ == "__main__":
    x1_ycbcr = torch.rand(4, 3, 256, 256)
    x2_ycbcr = torch.rand(4, 3, 256, 256)
    x1_y, x1_cb, x1_cr = torch.chunk(x1_ycbcr, 3, dim=1)
    x2_y, x2_cb, x2_cr = torch.chunk(x2_ycbcr, 3, dim=1)
    print('x1 Y Cb Cr sizes: \n', \
          x1_y.size(), x1_cb.size(), x1_cr.size())
    deepfuse = DeepFuse()
    fused_y = deepfuse(x1_y, x2_y)
    print('fused Y size: \n', fused_y.size())
    fused_cb = chroma_weight_fusion(x1_cb, x2_cb)
    fused_cr = chroma_weight_fusion(x1_cr, x2_cr)
    print('fused Cb / Cr size: \n',
          fused_cb.size(), fused_cr.size())
    x_fused = torch.cat((fused_y, fused_cb, fused_cr), dim=1)
    print('fused YCbCr size: \n', x_fused.size())
