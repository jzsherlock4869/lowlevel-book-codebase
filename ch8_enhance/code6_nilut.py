import torch
import torch.nn as nn

class NILUT(nn.Module):
    # NILUT: neural implicit 3D LUT
    def __init__(self, in_ch=3, nf=256, n_layer=3, out_ch=3):
        super().__init__()
        layers = list()
        layers.append(nn.Linear(in_ch, nf))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(n_layer):
            layers.append(nn.Linear(nf, nf))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(nf, out_ch))
        self.body = nn.Sequential(*layers)
    def forward(self, x):
        # x size: [n, c, h, w]
        n, c, h, w = x.size()
        x = torch.permute(x, (0, 2, 3, 1))
        x = torch.reshape(x, (n, h * w, c))
        print(f"[NILUT] neural net input: {x.size()}")
        res = self.body(x)
        out = x + res
        out = torch.clamp(out, 0, 1)
        out = torch.reshape(out, (n, h, w, c))
        out = torch.permute(out, (0, 3, 1, 2))
        return out


class CNILUT(nn.Module):
    # conditional NILUT (with style encoded)
    def __init__(self, in_ch=3,
                 nf=256, n_layer=3,
                 out_ch=3, n_style=3):
        super().__init__()
        self.n_style = n_style
        layers = list()
        layers.append(nn.Linear(in_ch + n_style, nf))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(n_layer):
            layers.append(nn.Linear(nf, nf))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(nf, out_ch))
        self.body = nn.Sequential(*layers)
    def forward(self, x, style):
        # x size: [n, c, h, w]
        n, c, h, w = x.size()
        x = torch.permute(x, (0, 2, 3, 1))
        x = torch.reshape(x, (n, h * w, c))
        style_vec = torch.Tensor(style)
        print(f"[CNILUT] style vector: {style_vec}")
        style_vec = style_vec.repeat(h * w)\
                        .view(h * w, self.n_style)
        style_vec = style_vec.repeat(n, 1, 1)\
                        .view(n, h * w, self.n_style)
        style_vec = style_vec.to(x.device)
        x_style = torch.cat((x, style_vec), dim=2)
        print(f"[CNILUT] neural net input: {x_style.size()}")
        res = self.body(x_style)
        out = x + res
        out = torch.clamp(out, 0, 1)
        out = torch.reshape(out, (n, h, w, c))
        out = torch.permute(out, (0, 3, 1, 2))
        return out



if __name__ == "__main__":
    patch = torch.rand(4, 3, 64, 64)
    nilut = NILUT()
    lut_out = nilut(patch)
    print(f"NILUT input size: {patch.size()}")
    print(f"NILUT output size: {lut_out.size()}")

    style = [0.4, 0.5, 0.1]
    cnilut = CNILUT()
    clut_out = cnilut(patch, style)
    print(f"CNILUT input size: {patch.size()}")
    print(f"CNILUT output size: {clut_out.size()}")

