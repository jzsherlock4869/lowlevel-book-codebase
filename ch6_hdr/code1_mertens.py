import os
import cv2
import numpy as np
from glob import glob
import utils.pyramid as P

def calc_saturation(img_rgb):
    r, g, b = cv2.split(img_rgb / 255.0)
    mean = (r + g + b) / 3.0
    saturation = np.sqrt(((r-mean) **2 + (g-mean) **2 + (b-mean) **2) / 3.0)
    return saturation

def calc_contrast(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    contrast = np.abs(cv2.Laplacian(gray, \
                ddepth=cv2.CV_16S, ksize=3)).astype(np.float32)
    contrast = (contrast - np.min(contrast))
    contrast = contrast / np.max(contrast)
    return contrast

def calc_exposedness(img_rgb, sigma=0.2):
    r, g, b = cv2.split(img_rgb / 255.0)
    r_res =  np.exp(-(r - 0.5) ** 2 / (2 * sigma ** 2))
    g_res =  np.exp(-(g - 0.5) ** 2 / (2 * sigma ** 2))
    b_res =  np.exp(-(b - 0.5) ** 2 / (2 * sigma ** 2))
    exposedness = r_res * g_res * b_res
    return exposedness

def get_weightmaps(img_ls, weight_sat, weight_con, weight_expo):
    sum_tot = None
    weightmaps = list()
    for img in img_ls:
        saturation = calc_saturation(img)
        contrast = calc_contrast(img)
        exposedness = calc_exposedness(img)
        cur_weightmap = (saturation ** weight_sat) \
                      * (contrast ** weight_con) \
                      * (exposedness ** weight_expo) + 1e-8
        weightmaps.append(cur_weightmap)
    weightmaps = np.stack(weightmaps, axis=0)
    sum_tot = np.sum(weightmaps, axis=0)
    weightmaps = weightmaps / sum_tot
    return weightmaps

def exposure_fusion(img_dir, out_dir, pyr_level=10, indexes=[1.0, 1.0, 1.0]):
    weight_sat, weight_con, weight_expo = indexes
    img_paths = list(glob(os.path.join(img_dir, '*')))
    img_paths = sorted(img_paths)
    num_expo = len(img_paths)

    imgs = [cv2.imread(img_path)[:,:,::-1] for img_path in img_paths]
    weightmaps = get_weightmaps(imgs, weight_sat, weight_con, weight_expo)
    img_lap_pyrs = [P.build_laplacian_pyr(img, pyr_level) for img in imgs]
    weight_gau_pyrs = [P.build_gaussian_pyr(weight, pyr_level) \
                            for weight in weightmaps]
    output_pyr = list()
    for lvl in range(pyr_level):
        cur_fused = None
        for i in range(num_expo):
            cur_weight = np.expand_dims(weight_gau_pyrs[i][lvl], axis=2) * 1.0
            if cur_fused is None:
                cur_fused = img_lap_pyrs[i][lvl] * cur_weight
            else:
                cur_fused += img_lap_pyrs[i][lvl] * cur_weight
        output_pyr.append(cur_fused)
    fused = P.collapse_laplacian_pyr(output_pyr)
    fused = (np.clip(fused, 0, 255)).astype(np.uint8)[:,:,::-1]
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, 'fused.png'), fused)
    for idx in range(num_expo):
        w = (weightmaps[idx] * 255).astype(np.uint8)
        fname = os.path.basename(img_paths[idx]).split('.')[0]
        cv2.imwrite(os.path.join(out_dir, f'weight_{idx}_{fname}.png'), w)
    return


if __name__ == "__main__":
    img_dir = '../datasets/hdr/multi_expo/'
    out_dir = './results/expofusion'
    pyr_level = 9
    indexes = [1, 1, 1]
    exposure_fusion(img_dir, out_dir, pyr_level, indexes)
