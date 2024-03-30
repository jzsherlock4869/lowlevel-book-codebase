import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

os.makedirs('results/clahe', exist_ok=True)
# 读入RGB样例图像
img_path = '../datasets/samples/lena256rgb.png'
img_rgb = cv2.imread(img_path)[:,:,::-1].copy()
# 转为灰度值并计算其直方图
img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

clip_limit_list = [0.5, 2.0, 3.0]
tile_size_list = [8, 16]

for clip_limit in clip_limit_list:
    for tile_size in tile_size_list:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, 
                                tileGridSize=(tile_size, tile_size))
        clahe_out = clahe.apply(img)
        cv2.imwrite('results/clahe/clahe_limit{}_tile{}.png'\
                    .format(clip_limit, tile_size), clahe_out)
