import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def mef_ssim(expo_imgs, fused_img, win_size=7, p=2):
    eps = 1e-10
    num_expo = len(expo_imgs)
    fused_img = fused_img.astype(np.float32)
    radius = win_size // 2
    H, W, C = fused_img.shape
    expos = np.stack(expo_imgs, axis=0).astype(np.float32)
    ssim_map = np.zeros((H, W, C), dtype=np.float32)
    for i in range(radius, H - radius):
        for j in range(radius, W - radius):
            fpatch = fused_img[i-radius:i+radius+1,
                               j-radius:j+radius+1, ...]
            c_hat = np.zeros((1, 1, C), dtype=np.float32)
            s_sum_nume = np.zeros((1, 1, C), dtype=np.float32)
            s_sum_deno = np.zeros((1, 1, C), dtype=np.float32)
            # 逐个取出曝光帧
            for eid in range(num_expo):
                x_k = expos[eid,
                            i-radius:i+radius+1,
                            j-radius:j+radius+1, ...]
                mu_k = np.mean(x_k, axis=(0,1), keepdims=True)
                # 计算均值归0后的结果
                x_tilde_k = x_k - mu_k
                c_k = np.linalg.norm(x_tilde_k,
                            axis=(0,1), keepdims=True)
                c_hat = np.maximum(c_k, c_hat)
                # 计算结构（structure）并更新目标结构的分子分母
                s_k = x_tilde_k / (c_k + eps)
                w_k = c_k ** p
                s_sum_nume = s_sum_nume + w_k * s_k
                s_sum_deno = s_sum_deno + w_k
            # 合成目标x hat
            s_bar = s_sum_nume / (s_sum_deno + eps)
            s_bar_norm = np.linalg.norm(s_bar,
                            axis=(0,1), keepdims=True)
            s_hat = s_bar / (s_bar_norm + eps)
            x_hat = c_hat * s_hat
            # 计算类 SSIM 相似度
            mu_x_hat = np.mean(x_hat,
                           axis=(0,1), keepdims=True)
            mu_y = np.mean(fpatch,
                           axis=(0,1), keepdims=True)
            sigma2_x_hat = np.mean(x_hat**2,
                                   axis=(0,1), keepdims=True) \
                            - mu_x_hat**2
            sigma2_y = np.mean(fpatch**2,
                               axis=(0,1), keepdims=True) \
                            - mu_y**2
            sigma_x_hat_y = np.mean(x_hat * fpatch,
                               axis=(0,1), keepdims=True) \
                            - mu_x_hat * mu_y
            C1 = (0.03 * 255) ** 2
            mef_ssim_patch = (2 * sigma_x_hat_y + C1) \
                        / (sigma2_x_hat + sigma2_y + C1)
            ssim_map[i, j, :] = mef_ssim_patch
    return ssim_map

if __name__ == "__main__":
    expo1 = cv2.imread('../datasets/hdr/multi_expo/grandcanal_A.jpg')
    expo2 = cv2.imread('../datasets/hdr/multi_expo/grandcanal_B.jpg')
    expo3 = cv2.imread('../datasets/hdr/multi_expo/grandcanal_C.jpg')
    fused = cv2.imread('results/expofusion/fused.png')
    mef_ssim_map = mef_ssim([expo1, expo2, expo3], fused)
    os.makedirs('./results/mefssim/', exist_ok=True)
    fig = plt.figure()
    plt.imshow(np.mean(mef_ssim_map, axis=2),
               cmap='gray', vmin=0, vmax=1)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('./results/mefssim/mef_ssim_map.png')
    print('MEF-SSIM score is :', np.mean(mef_ssim_map))
