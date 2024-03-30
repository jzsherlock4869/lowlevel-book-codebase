import cv2
import numpy as np
import torch
import torch.nn.functional as F
import pyiqa

print('all available metrics: ')
print(pyiqa.list_models())

device='cpu'
lpips_metric = pyiqa.create_metric('lpips', device=device)
print('LPIPS is lower better? ', lpips_metric.lower_better)
psnr_metric = pyiqa.create_metric('psnr', device=device)
print('PSNR is lower better? ', psnr_metric.lower_better)

img = cv2.imread('../datasets/srdata/Set5/baby_GT.bmp')[...,::-1].copy()
H, W = img.shape[:2]

# 测试模糊损失的LPIPS (以PSNR指标作为对比)
gt_tensor = torch.FloatTensor(img).permute(2,0,1) / 255.0
gt_tensor = gt_tensor.unsqueeze(0) # [1, 3, H, W]
down2_tensor = F.interpolate(gt_tensor, scale_factor=0.5, mode='area')
down2_tensor = F.interpolate(down2_tensor, (H, W), mode='bicubic')
down4_tensor = F.interpolate(gt_tensor, scale_factor=0.25, mode='area')
down4_tensor = F.interpolate(down4_tensor, (H, W), mode='bicubic')
lpips_score = lpips_metric(gt_tensor, down2_tensor).item()
psnr = psnr_metric(gt_tensor, down2_tensor).item()
print(f'LPIPS score of down2 is : '\
      f'{lpips_score:.4f} (PSNR: {psnr:.4f})')
lpips_score = lpips_metric(gt_tensor, down4_tensor).item()
psnr = psnr_metric(gt_tensor, down4_tensor).item()
print(f'LPIPS score of down4 is : '\
      f'{lpips_score:.4f} (PSNR: {psnr:.4f})')

# 测试空间变换损失的LPIPS
mat_src = np.float32([[0, 0],[0, H-1],[W-1, 0]])
mat_dst = np.float32([[0, 0],[10, H-10],[W-10, 10]])
mat_trans = cv2.getAffineTransform(mat_src, mat_dst)
warp_img = cv2.warpAffine(img, mat_trans, (W, H))
warp_tensor = torch.FloatTensor(warp_img).permute(2,0,1) / 255.0
warp_tensor = warp_tensor.unsqueeze(0) # [1, 3, H, W]
lpips_score = lpips_metric(gt_tensor, warp_tensor).item()
psnr = psnr_metric(gt_tensor, warp_tensor).item()
print(f'LPIPS score of warp is : '\
      f'{lpips_score:.4f} (PSNR: {psnr:.4f})')

