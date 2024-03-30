import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils.simu_noise import add_gaussian_noise_gray

os.makedirs('results/bilateral', exist_ok=True)
img_path = '../datasets/srdata/Set12/04.png'
img = cv2.imread(img_path)[:,:,0]

# 高斯噪声
noisy_img = add_gaussian_noise_gray(img, sigma=25)

sigma_color_ls = [10, 50, 200]
sigma_space_ls = [1, 5, 20]

fig = plt.figure(figsize=(12, 12))
M, N = len(sigma_color_ls), len(sigma_space_ls)

cnt = 1
for sc in sigma_color_ls:
    for ss in sigma_space_ls:
        bi_out = cv2.bilateralFilter(noisy_img, \
                                     d=31, sigmaColor=sc, sigmaSpace=ss)
        fig.add_subplot(M, N, cnt)
        plt.imshow(bi_out, cmap='gray')
        plt.axis('off')
        plt.title(f'sigma_color={sc}, sigma_space={ss}')
        cnt += 1

plt.savefig(f'results/bilateral/bilateral.png')
plt.close()
