import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

os.makedirs('./results/upsample', exist_ok=True)
img = cv2.imread('../datasets/srdata/Set5/baby_GT.bmp')[:,:,::-1]
lr = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)

# 最近邻、双线性和双三次插值上采样
target_size = (512, 512)
up_nn = cv2.resize(lr, target_size, interpolation=cv2.INTER_NEAREST)
up_linear = cv2.resize(lr, target_size, interpolation=cv2.INTER_LINEAR)
up_cubic = cv2.resize(lr, target_size, interpolation=cv2.INTER_CUBIC)

# 显示并保存结果
fig = plt.figure(figsize=(15, 5))
fig.add_subplot(131)
plt.imshow(up_nn)
plt.title('nearest upsample')
fig.add_subplot(132)
plt.imshow(up_linear)
plt.title('bilinear upsample')
fig.add_subplot(133)
plt.imshow(up_cubic)
plt.title('bicubic upsample')
plt.savefig(f'results/upsample/upsample_result.png')
plt.close()
