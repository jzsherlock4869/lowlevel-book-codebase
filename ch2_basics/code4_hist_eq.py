import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

os.makedirs('results/hist', exist_ok=True)

# 读入RGB样例图像
img_path = '../datasets/samples/lena256rgb.png'
img_rgb = cv2.imread(img_path)[:,:,::-1].copy()

# 转为灰度值并计算其直方图
img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
hist = cv2.calcHist([img], [0], None, [256], [0, 256])

# 直方图拉伸提高对比度
stretch_out = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
# 计算直方图拉伸后图像的直方图
stretch_hist = cv2.calcHist([stretch_out], [0], None, [256], [0, 256])


# 直方图均衡化
he_out = cv2.equalizeHist(img)
# 计算均衡化后图像的直方图
he_hist = cv2.calcHist([he_out], [0], None, [256], [0, 256])

# 定义函数，将图像和直方图画到同一行中
def plot_img_hist(gs, row, img, hist):
    ax_img = plt.subplot(gs[row, 0])
    ax_hist = plt.subplot(gs[row, 1:])
    ax_img.imshow(img, 'gray', vmin=0, vmax=255)
    ax_img.axis('off')
    ax_hist.stem(hist, use_line_collection=True, markerfmt='')
    ax_hist.set_yticks([])


# 结果可视化
fig = plt.figure(figsize=(8, 5))
gs = gridspec.GridSpec(3, 4)
# 显示原始灰度图像及其直方图
plot_img_hist(gs, 0, img, hist)
# 直方图拉伸后的图像及其直方图
plot_img_hist(gs, 1, stretch_out, stretch_hist)
# 直方图均衡后的图像及其直方图
plot_img_hist(gs, 2, he_out, he_hist)

plt.savefig('./results/hist/hist_compare.png')
plt.close()