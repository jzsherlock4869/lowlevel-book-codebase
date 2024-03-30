import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.simu_noise import add_gaussian_noise_color

os.makedirs('results/nlmeans', exist_ok=True)
img_path = '../datasets/samples/lena256rgb.png'
img = cv2.imread(img_path)[...,::-1]

# 加入高斯噪声
noisy_img = add_gaussian_noise_color(img, sigma=25)

# 采用邻域为 5×5，搜索窗口为 21×21 的 NL-means 算法
nlm_out = cv2.fastNlMeansDenoisingColored(noisy_img, None, h=10, hColor=10,
                                          templateWindowSize=5,
                                          searchWindowSize=21)
nlm_diff = np.sum(np.abs(nlm_out - noisy_img), axis=2)

# 采用 5×5 的高斯滤波作为对比
gauss_out = cv2.GaussianBlur(noisy_img, (5, 5), 0)
gauss_diff = np.sum(np.abs(gauss_out - noisy_img), axis=2)

fig = plt.figure(figsize=(8, 5))
fig.add_subplot(231)
plt.imshow(noisy_img)
plt.axis('off')
plt.title('(a)')
fig.add_subplot(232)
plt.imshow(nlm_out)
plt.axis('off')
plt.title('(b)')
fig.add_subplot(233)
plt.imshow(nlm_diff, cmap='gray')
plt.axis('off')
plt.title('(c)')
fig.add_subplot(235)
plt.imshow(gauss_out)
plt.axis('off')
plt.title('(d)')
fig.add_subplot(236)
plt.imshow(gauss_diff, cmap='gray')
plt.axis('off')
plt.title('(e)')

plt.savefig('./results/nlmeans/output.png')
