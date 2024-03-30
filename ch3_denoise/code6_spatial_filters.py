import os
import cv2
import numpy as np
# 这里直接使用之前代码中写的模拟高斯噪声和椒盐噪声的函数
# 相关函数被整理并存在 ./utils/simu_noise.py 文件中
from utils.simu_noise import add_gaussian_noise_color, \
                            add_salt_pepper_noise

os.makedirs('results/spatial_filters', exist_ok=True)
img_rgb_path = '../datasets/samples/lena256rgb.png'
img = cv2.imread(img_rgb_path)

# 各种滤波器处理高斯噪声
gauss_noisy = add_gaussian_noise_color(img, sigma=50)
cv2.imwrite('results/spatial_filters/gauss_noisy.png', gauss_noisy)

# 高斯滤波
gauss_filtered = cv2.GaussianBlur(gauss_noisy, (5, 5), 0)
cv2.imwrite('results/spatial_filters/gauss_filtered.png', gauss_filtered)
# 均值滤波
mean_filtered = cv2.blur(gauss_noisy, (5, 5))
cv2.imwrite('results/spatial_filters/mean_filtered.png', mean_filtered)
# 中值滤波
median_filtered = cv2.medianBlur(gauss_noisy, 5)
cv2.imwrite('results/spatial_filters/median_filtered.png',\
            median_filtered)


# 中值滤波器和高斯滤波器处理椒盐噪声
# 添加椒盐噪声
img_gray_path = '../datasets/srdata/Set12/01.png'
img_gray = cv2.imread(img_gray_path)
speckle_noisy = add_salt_pepper_noise(img_gray, \
                        salt_ratio=0.02, pepper_ratio=0.02)
cv2.imwrite('results/spatial_filters/speckle_noisy.png', speckle_noisy)

# 中值滤波处理椒盐噪声
speckle_median_filtered = cv2.medianBlur(speckle_noisy, 5)
cv2.imwrite('results/spatial_filters/speckle_median_filtered.png',\
            speckle_median_filtered)
# 高斯滤波处理椒盐噪声
speckle_gauss_filtered = cv2.GaussianBlur(speckle_noisy, (5, 5), 0)
cv2.imwrite('results/spatial_filters/speckle_gauss_filtered.png', \
            speckle_gauss_filtered)

