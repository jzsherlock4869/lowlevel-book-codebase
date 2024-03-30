import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import bm3d
from utils.simu_noise import add_gaussian_noise_gray

os.makedirs('results/bm3d', exist_ok=True)
img = cv2.imread('../datasets/srdata/Set12/01.png')[:,:,0]

sigma = 25
noisy = add_gaussian_noise_gray(img, sigma=sigma)

out_step1 = bm3d.bm3d(noisy, 
        sigma_psd=sigma, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
out_step2 = bm3d.bm3d(noisy, 
        sigma_psd=sigma, stage_arg=bm3d.BM3DStages.ALL_STAGES)

cv2.imwrite('results/bm3d/noisy.png', noisy)
cv2.imwrite('results/bm3d/out_step1.png', out_step1)
cv2.imwrite('results/bm3d/out_step2.png', out_step2)