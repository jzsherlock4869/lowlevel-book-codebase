import torch
import torch.nn as nn

class DoveNetAttn(nn.Module):
    def __init__(self, enc_nf, dec_nf):
        super().__init__()
        nf = enc_nf + dec_nf
        self.attn = nn.Sequential(
            nn.Conv2d(nf, nf, 1, 1, 0),
            nn.Sigmoid()
        )
    def forward(self, enc_ft, dec_ft):
        ft = torch.cat((enc_ft, dec_ft), dim=1)
        attn_map = self.attn(ft)
        out = ft * attn_map
        return out

if __name__ == "__main__":
    enc_feat = torch.randn(4, 64, 32, 32)
    dec_feat = torch.randn(4, 64, 32, 32)
    attn = DoveNetAttn(enc_nf=64, dec_nf=64)
    fused = attn(enc_feat, dec_feat)
    print(f"DoveNet Attention out size: {fused.size()}")
