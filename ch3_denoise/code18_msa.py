import torch
import torch.nn as nn

class MSA(nn.Module):
    def __init__(self,
                 in_dim,
                 n_head=8,
                 head_dim=64,
                 dropout=0.1):
        super().__init__()
        # 多头输出总维度
        dim = n_head * head_dim
        self.n_head = n_head
        self.head_dim = head_dim
        # attention map 左乘 value, 因此需要每行归一化
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        # Q/K/V 映射层
        self.proj_qkv = nn.Linear(in_dim, dim * 3, bias=False)
        self.proj_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        # x size: [n, l, d]
        n, l, _ = x.size()
        h, d = self.n_head, self.head_dim
        qkv = self.proj_qkv(x) # [n, l, 3hd]
        q, k, v = torch.chunk(qkv, 3, dim=-1) # [n, l, hd]
        q = q.reshape(n, l, h, d).transpose(1, 2) # [n, h, l, d]
        k = k.reshape(n, l, h, d).transpose(1, 2)
        v = v.reshape(n, l, h, d).transpose(1, 2)
        # attn_map size: [n, h, l, l]
        attn_map = torch.matmul(q, k.transpose(2, 3)) / (d ** 0.5)
        attn_map = self.dropout(self.softmax(attn_map))
        out = torch.matmul(attn_map, v) # [n, h, l, d]
        out = out.transpose(1, 2).reshape(n, l, h * d)
        # [n, l, hd] -> [n, l, hd]
        out = self.proj_out(out)
        return out


if __name__ == "__main__":
    # 测试 MSA 对于向量序列的处理结果
    dummy_in = torch.randn(1, 10, 32)
    msa = MSA(in_dim=32, n_head=4, head_dim=16, dropout=0.1)
    out = msa(dummy_in)
    print(f"MSA input: {dummy_in.size()}\n   output: {out.size()}")
