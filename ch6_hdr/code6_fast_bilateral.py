import os
import cv2
import numpy as np
from scipy.ndimage import convolve
from scipy.interpolate import interpn
import matplotlib.pyplot as plt

def fast_bilateral(img, sigma_s, sigma_r):
    H, W = img.shape
    VMIN, VMAX = np.min(img), np.max(img)
    VR = VMAX - VMIN
    print('[Fast Bilateral] downscaled image info: ')
    print(f'H x W: {H}x{W}, vmin: {VMIN}, vmax: {VMAX}')
    h = int((H - 1) / sigma_s) + 2
    w = int((W - 1) / sigma_s) + 2
    vr = int(VR / sigma_r) + 2
    data_tensor = np.zeros((h, w, vr))
    weight_tensor = np.zeros((h, w, vr))
    # 下采样并建立3D网格
    ds_v_map = np.round((img - VMIN) / sigma_r).astype(int)
    for i in range(H):
        for j in range(W):
            val = img[i, j]
            ds_v = ds_v_map[i, j]
            ds_i = np.round(i / sigma_s).astype(int)
            ds_j = np.round(j / sigma_s).astype(int)
            data_tensor[ds_i, ds_j, ds_v] += val
            weight_tensor[ds_i, ds_j, ds_v] += 1
    # 生成 3D 卷积核
    kernel_3d = np.zeros((3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                d2 = (i - 1) ** 2 \
                   + (j - 1) ** 2 \
                   + (k - 1) ** 2
                val = np.exp(-d2/2)
                kernel_3d[i, j, k] = val
    # 3D 空间内卷积（2D 坐标空间 + 1D 值域空间）
    fdata = convolve(data_tensor, kernel_3d, mode='constant')
    fweight = convolve(weight_tensor, kernel_3d, mode='constant')
    norm_data = fdata / (fweight + 1e-10)
    norm_data[fweight == 0] = 0
    print('[Fast Bilateral] grid size: ', norm_data.shape)
    # 对处理结果进行插值，得到处理后的图像
    ds_points = (np.arange(h), np.arange(w), np.arange(vr))
    samples = np.zeros((H, W, 3))
    for i in range(H):
        for j in range(W):
            val = img[i, j]
            ds_i = i / sigma_s
            ds_j = j / sigma_s
            ds_v = (val - VMIN) / sigma_r
            samples[i, j, :] = [ds_i, ds_j, ds_v]
    output = interpn(ds_points, norm_data, samples)
    return output


def fast_bilateral_tone(hdr_img,
                        contrast,
                        sigma_s,
                        sigma_r,
                        gamma_coeff):
    R, G, B = cv2.split(hdr_img)
    luma = (20.0 * R + 40.0 * G + 1.0 * B) / 61.0
    log_luma = np.log10(luma.astype(np.float64))
    log_base = fast_bilateral(log_luma, sigma_s, sigma_r)
    log_detail = log_luma - log_base
    vmax, vmin = np.max(log_base), np.min(log_base)
    comp_fact = np.log10(contrast) / (vmax - vmin)
    log_abs_scale = vmax * comp_fact
    log_out = log_base * comp_fact + log_detail - log_abs_scale
    luma_out = 10 ** (log_out)
    Rout = R / luma * luma_out
    Gout = G / luma * luma_out
    Bout = B / luma * luma_out
    out_img = cv2.merge((Rout, Gout, Bout))
    out_img = np.power(out_img, 1/gamma_coeff)
    out_img = np.clip(out_img, a_min=0, a_max=1)
    return out_img


if __name__ == "__main__":

    os.makedirs('results/fastbilateral/', exist_ok=True)
    # 测试快速双边滤波
    img = cv2.imread('../datasets/srdata/Set12/07.png')[..., 0]
    h, w = img.shape
    sigma = 25
    gaussian_noise = np.float32(np.random.randn(*(img.shape))) * sigma
    noisy_img = img + gaussian_noise
    noisy_img = np.clip((noisy_img).round(), 0, 255).astype(np.uint8)
    sigma_s, sigma_r = 5, 50
    out = fast_bilateral(noisy_img, sigma_s, sigma_r)
    out = np.clip(out, a_min=0, a_max=255).astype(np.uint8)
    cv2.imwrite('results/fastbilateral/bi_input.png', noisy_img)
    cv2.imwrite('results/fastbilateral/bi_output.png', out)

    # 测试快速双边滤波 HDR 影调映射
    hdr_path = '../datasets/hdr/memorial.hdr'
    hdr_img = cv2.imread(hdr_path,
                    flags = cv2.IMREAD_ANYDEPTH)[...,::-1]
    tonemapped = fast_bilateral_tone(hdr_img, 5, 10, 0.4, gamma_coeff=1.2)
    tonemapped = np.clip(tonemapped * 255, a_min=0, a_max=255)
    tonemapped = tonemapped.astype(np.uint8)[...,::-1]
    cv2.imwrite('results/fastbilateral/tone_output.png', tonemapped)

