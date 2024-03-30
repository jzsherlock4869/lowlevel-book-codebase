import torch
import torch.nn as nn

class CCA_Module(nn.Module):
    """
    Contrast-aware Channel Attention module
    [Block] -- ChannelStd -|
        |_____ GlobalPool -+- FC - ReLU - FC - Sigmoid - x -
        |________________________________________________|
        nf (int): 特征通道数
        sf (int): 压缩比例系数
    """
    def __init__(self, nf, sf=16):
        super(CCA_Module, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.attn = nn.Sequential(
            nn.Conv2d(nf, nf // sf, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf // sf, nf, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def stdv_channels(self, featmap):
        """
        featmap (torch.tensor): [b, c, h, w]
        """
        assert(featmap.dim() == 4)
        channel_mean = featmap.mean(3, keepdim=True)\
                              .mean(2, keepdim=True)
        channel_variance = (featmap - channel_mean)\
            .pow(2).mean(3, keepdim=True).mean(2, keepdim=True)
        return channel_variance.pow(0.5)

    def forward(self, x):
        y = self.stdv_channels(x) + self.avg_pool(x)
        y = self.attn(y)
        return x * y

# =========================== #
#           IMDB              #
# =========================== #

class IMDB_Module(nn.Module):
    """
    -- conv -- split --------------------- cat - CCA - conv -
                |__ conv -- split --------|
                            |___ conv --|
    """
    def __init__(self, n_feat, num_split=3) -> None:
        super(IMDB_Module, self).__init__()
        distill_rate = 1.0 / (num_split + 1)
        self.nf_distill = int(n_feat * distill_rate)
        self.nf_remain = n_feat - self.nf_distill
        self.level = num_split + 1

        self.conv_in = nn.Conv2d(n_feat, n_feat, 3, 1, 1)
        conv_r_ls = []
        for i in range(num_split):
            if i < num_split - 1:
                conv_r_ls.append(
                    nn.Conv2d(self.nf_remain,
                              n_feat, 3, 1, 1)
                                )
            else:
                conv_r_ls.append(
                    nn.Conv2d(self.nf_remain,
                              self.nf_distill, 3, 1, 1)
                              )
        self.conv_remains = nn.ModuleList(conv_r_ls)
        self.conv_out = nn.Conv2d(self.nf_distill * self.level,
                            n_feat, 1, 1, 0)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.cca = CCA_Module(n_feat)

    def forward(self, x):
        feat_in = self.conv_in(x)
        out_d0, out_r = torch.split(feat_in,
                                (self.nf_distill, self.nf_remain), dim=1)
        print(f"[IMDB] split no.0, out_d: "\
              f" {out_d0.size()}, out_r: {out_r.size()}")
        distill_ls = [out_d0]
        for i in range(self.level - 2):
            out_dr = self.lrelu(self.conv_remains[i](out_r))
            # out_d 和 out_r 分别为每一次分裂的 distill 和 remain 部分
            out_d, out_r = torch.split(
                                out_dr,
                                (self.nf_distill, self.nf_remain), dim=1)
            print(f"[IMDB] split no.{i + 1}, "\
                  f"out_d: {out_d.size()}, out_r: {out_r.size()}")
            distill_ls.append(out_d)
        out_d_last = self.conv_remains[self.level - 2](out_r)
        print(f"[IMDB] last conv size: {out_d_last.size()}")

        distill_ls.append(out_d_last)
        fused = torch.cat(distill_ls, dim=1)
        print(f"[IMDB] fused size: {fused.size()}")
        fused = self.cca(fused)
        print(f"[IMDB] CCA out size: {fused.size()}")
        out = self.conv_out(fused)
        print(f"[IMDB] conv1x1 size: {fused.size()}")
        return out + x


# =========================== #
#           RFDB              #
# =========================== #

class SRB(nn.Module):
    """
    浅层残差模块, 用于构建RFDB
    shallow residual block
    --- conv3 -- + --
     |___________|
    """
    def __init__(self, nf) -> None:
        super().__init__()
        self.conv = nn.Conv2d(nf, nf, 3, 1, 1)

    def forward(self, x):
        out = self.conv(x) + x
        return out


class RFDB_Module(nn.Module):
    """
     -- conv -- conv1 ------------ cat -- conv -- CCA -
             |__conv3 --- conv1 ----|
                       |__conv3 ----|
    """
    def __init__(self, n_feat, nf_distill, stage):
        super().__init__()
        self.nf_dis = nf_distill
        self.stage = stage

        conv_d_ls = [nn.Conv2d(n_feat, self.nf_dis, 1, 1, 0)]
        conv_r_ls = [SRB(n_feat)]
        for i in range(1, self.stage - 1):
            conv_d_ls.append(nn.Conv2d(n_feat, self.nf_dis, 1, 1, 0))
            conv_r_ls.append(SRB(n_feat))
        conv_d_ls.append(nn.Conv2d(n_feat, self.nf_dis, 3, 1, 1))
        self.conv_distill = nn.ModuleList(conv_d_ls)
        self.conv_remains = nn.ModuleList(conv_r_ls)
        self.conv_out = nn.Conv2d(self.nf_dis * stage, n_feat, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.cca = CCA_Module(n_feat)

    def forward(self, x):
        cur = x.clone()
        distill_ls = []
        for i in range(self.stage):
            out_d = self.conv_distill[i](cur)
            distill_ls.append(out_d)
            if i < self.stage - 1:
                cur = self.conv_remains[i](cur)
                print(f"[RFDB] stage {i}, "\
                    f"distill: {out_d.size()}, remain: {cur.size()}")
            else:
                print(f"[RFDB] stage {i}, distill: {out_d.size()}")
        fused = torch.cat(distill_ls, dim=1)
        print(f"[RFDB] fused size: {fused.size()}")
        out = self.conv_out(fused)
        print(f"[IMDB] conv1x1 size: {out.size()}")
        out = self.cca(out)
        print(f"[IMDB] CCA out size: {out.size()}")
        return out + x


if __name__ == "__main__":
    dummy_in = torch.randn(4, 32, 64, 64)
    # 测试 IMDB
    imdb = IMDB_Module(n_feat=32, num_split=3)
    imdb_out = imdb(dummy_in)
    print('IMDB output size : ', imdb_out.size())
    # 测试 RFDB
    rfdb = RFDB_Module(n_feat=32, nf_distill=8, stage=4)
    rfdb_out = rfdb(dummy_in)
    print('RFDB output size : ', rfdb_out.size())
