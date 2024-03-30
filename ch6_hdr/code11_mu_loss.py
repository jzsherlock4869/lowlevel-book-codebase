import torch
import torch.nn as nn
import torch.nn.functional as F

class HDRMuLoss(nn.Module):
    def __init__(self, mu=5000):
        super().__init__()
        self.mu = mu
    def forward(self, pred, gt):
        tensor_1_mu = torch.FloatTensor([1 + self.mu])
        Tgt_nume = torch.log(1 + self.mu * gt)
        Tgt_deno = torch.log(tensor_1_mu).to(gt.device)
        Tgt = Tgt_nume / Tgt_deno
        Tpred_nume = torch.log(1 + self.mu * pred)
        Tpred_deno = torch.log(tensor_1_mu).to(pred.device)
        Tpred = Tpred_nume / Tpred_deno
        mu_loss = F.l1_loss(Tpred, Tgt)
        return mu_loss

if __name__ == "__main__":
    muloss = HDRMuLoss(mu=5000)
    pred = torch.rand(4, 3, 64, 64)
    gt = torch.rand(4, 3, 64, 64)
    pred.requires_grad = True
    print("pred gradient is None?", pred.grad is None)
    loss = muloss(pred, gt)
    print("calc mu loss : ", loss.item())
    loss.backward()
    print("pred grad after backward (part): \n",
                pred.grad[0, 0, :3, :3])
    pred = pred - pred.grad
    loss = muloss(pred, gt)
    print("updated mu loss : ", loss.item())
