import os
import cv2
import numpy as np
import time
from utils.pyramid import build_gaussian_pyr, \
        build_laplacian_pyr, collapse_laplacian_pyr
from utils.remapping import remapping_tone

def fast_local_laplace_filters(img, sigma, beta, n_samples, max_value):
    # 计算所需参数
    h, w = img.shape[:2]
    n_level = int(np.ceil(np.log2(min(h, w))))
    print("[LLF] pyramid level total: ", n_level)
    samples = np.linspace(0, max_value, n_samples)
    step = 1 / n_samples
    # 建立输入高斯金字塔，初始化输出的拉普拉斯金字塔
    gauss_pyr = build_gaussian_pyr(img, n_level)
    zero_img = np.zeros_like(img)
    out_laplace_pyr = build_laplacian_pyr(zero_img, n_level)
    print("[LLF] set pyramid last level using gauss_pyr")
    out_laplace_pyr[-1] = gauss_pyr[-1]
    for g0 in samples:
        print(f"[LLF] current sample g0: {g0:.4f}")
        remapped = remapping_tone(img, g0, sigma, beta)
        cur_lap_pyr = build_laplacian_pyr(remapped, n_level)
        for lvl in range(n_level - 1):
            gauss_layer = gauss_pyr[lvl]
            # 获取需要被该g0插值计算的取值的坐标和系数
            region = (np.abs(gauss_layer - g0) < step)
            coeff = 1 - np.abs(gauss_layer - g0) / step
            coeff = coeff * region
            cur_out = cur_lap_pyr[lvl] * coeff
            out_laplace_pyr[lvl] += cur_out
    img_out = collapse_laplacian_pyr(out_laplace_pyr)
    img_out = np.clip(img_out, 0, max_value)
    return img_out


if __name__ == "__main__":

    os.makedirs('results/llf', exist_ok=True)
    # test image: https://www.pauldebevec.com/Research/HDR/
    hdr_path = '../datasets/hdr/memorial.hdr'

    hdr_img = cv2.imread(hdr_path, flags = cv2.IMREAD_ANYDEPTH)
    h, w = hdr_img.shape[:2]
    hdr_img = cv2.resize(hdr_img, (w//2, h//2),
                         interpolation=cv2.INTER_AREA)

    hdr_img = hdr_img / hdr_img.max()
    hdr_img = np.power(hdr_img, 1/2.2)

    beta = 0.01
    sigma = 0.05
    max_value = 1
    n_samples = 12
    start_time = time.time()
    tone_out = fast_local_laplace_filters(hdr_img,
                        sigma, beta, n_samples, max_value)
    end_time = time.time()
    mini = np.percentile(tone_out, 1)
    maxi = np.percentile(tone_out, 99)
    tone_out = np.clip((tone_out - mini) / (maxi - mini), 0, 1)
    tone_out_8u = (tone_out  * 255.0).astype(np.uint8)
    tone_in_8u = (hdr_img  * 255.0).astype(np.uint8)
    cv2.imwrite('results/llf/fastllf_out.png', tone_out_8u)
    cv2.imwrite('results/llf/fastllf_in.png', tone_in_8u)
    # 显示 fastLLF 算法的运行时间
    print(f'total running time: {end_time - start_time:.2f}s')


