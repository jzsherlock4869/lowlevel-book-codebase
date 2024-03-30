import cv2
import numpy as np
import torch
import torch.nn.functional as F
import pyiqa

device='cpu'
niqe_metric = pyiqa.create_metric('niqe', device=device)
print('NIQE is lower better? ', niqe_metric.lower_better)
brisque_metric = pyiqa.create_metric('brisque', device=device)
print('BRISQUE is lower better? ', brisque_metric.lower_better)
psnr_metric = pyiqa.create_metric('psnr', device=device)
print('PSNR is lower better? ', psnr_metric.lower_better)

img = cv2.imread('../datasets/srdata/Set5/baby_GT.bmp')[...,::-1].copy()
H, W = img.shape[:2]

# 测试模糊损失的 NIQE (以PSNR指标作为对比)
gt_tensor = torch.FloatTensor(img).permute(2,0,1) / 255.0
gt_tensor = gt_tensor.unsqueeze(0) # [1, 3, H, W]
down2_tensor = F.interpolate(gt_tensor, scale_factor=0.5, mode='area')
down2_tensor = F.interpolate(down2_tensor, (H, W), mode='bicubic')
down4_tensor = F.interpolate(gt_tensor, scale_factor=0.25, mode='area')
down4_tensor = F.interpolate(down4_tensor, (H, W), mode='bicubic')

niqe_score = niqe_metric(down2_tensor).item()
brisque_score = brisque_metric(down2_tensor, gt_tensor).item()
psnr = psnr_metric(gt_tensor, down2_tensor).item()
print(f'down2 NIQE : {niqe_score:.4f} '\
      f'BRISQUE: {brisque_score:.4f} (PSNR: {psnr:.4f})')
niqe_score = niqe_metric(down4_tensor).item()
brisque_score = brisque_metric(down4_tensor, gt_tensor).item()
psnr = psnr_metric(gt_tensor, down4_tensor).item()
print(f'down4 NIQE : {niqe_score:.4f} '\
      f'BRISQUE: {brisque_score:.4f} (PSNR: {psnr:.4f})')
