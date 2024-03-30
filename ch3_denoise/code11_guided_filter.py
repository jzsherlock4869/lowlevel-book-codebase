import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.simu_noise import add_gaussian_noise_gray

def calc_mean(img, radius):
    res = cv2.boxFilter(img, cv2.CV_32F,\
            (radius, radius), borderType=cv2.BORDER_REFLECT)
    return res

def guided_filter_gray(guidance, image, radius, eps):
    image = image.astype(np.float32)
    guidance = guidance.astype(np.float32)
    # 计算导向图、输入图的均值图像
    mean_I = calc_mean(guidance, radius)
    mean_p = calc_mean(image, radius)
    # 计算导向图和输入图的协方差矩阵和导向图方差
    mean_Ip = calc_mean(image * guidance, radius)
    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = calc_mean(guidance ** 2, radius) - mean_I ** 2
    # 计算局部求解的a和b
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    # 计算导向滤波输出结果
    mean_a = calc_mean(a, radius)
    mean_b = calc_mean(b, radius)
    out = mean_a * guidance + mean_b
    return out


if __name__ == "__main__":

    os.makedirs('results/guided_filter', exist_ok=True)
    # 读取图像
    img_path = '../datasets/srdata/Set12/04.png'
    image = cv2.imread(img_path)[:,:,0]
    image = add_gaussian_noise_gray(image, sigma=25)
    guided_image = image.copy()

    # 转换为float类型
    image = image.astype(np.float32) / 255.0
    guided_image = guided_image.astype(np.float32) / 255.0

    # 进行原图导向滤波（保边平滑）
    radius_list = [11, 21, 31]
    eps_list = [0.005, 0.05, 0.5]

    fig = plt.figure(figsize=(12, 12))
    M, N = len(radius_list), len(eps_list)

    cnt = 1
    for radius in radius_list:
        for eps in eps_list:
            # guide_out = cv2.ximgproc.guidedFilter(guided_image, image, radius, eps)
            guide_out = guided_filter_gray(guided_image, image, radius, eps)
            # 滤波结果保存
            guide_out_u8 = np.uint8(guide_out * 255.0)
            fig.add_subplot(M, N, cnt)
            plt.imshow(guide_out_u8, cmap='gray')
            plt.axis('off')
            plt.title(f'radius={radius}, eps={eps}')
            cnt += 1

    plt.savefig(f'results/guided_filter/guided.png')
    plt.close()
