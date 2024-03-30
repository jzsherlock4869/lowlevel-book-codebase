import os
import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt

os.makedirs('results/wavelet', exist_ok=True)
img = cv2.imread('../datasets/srdata/Set12/02.png')[...,0]

# DWT 小波变换将图像分解到小波域
coeff = pywt.dwt2(img, 'haar')
LL, (LH, HL, HH) = coeff
print('input size: ', img.shape)
print('wavelet decompose sizes: \n',
      LL.shape, LH.shape, HL.shape, HH.shape)

# 展示分解结果
fig = plt.figure(figsize=(6, 6))
fig.add_subplot(221)
plt.imshow(LL, cmap='gray')
plt.axis('off')
plt.title('LL')
fig.add_subplot(222)
plt.imshow(np.abs(LH), cmap='gray')
plt.axis('off')
plt.title('LH')
fig.add_subplot(223)
plt.imshow(np.abs(HL), cmap='gray')
plt.axis('off')
plt.title('HL')
fig.add_subplot(224)
plt.imshow(np.abs(HH), cmap='gray')
plt.axis('off')
plt.title('HH')
plt.savefig('./results/wavelet/dwt.png')
plt.close()

# IDWT 小波系数重建原始图像
recon = pywt.idwt2(coeff, 'haar')
print('is reconstruction correct? ', np.allclose(img, recon))
