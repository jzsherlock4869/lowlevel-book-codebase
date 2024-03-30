import os
import cv2
import numpy as np
import pywt
from utils.simu_noise import add_gaussian_noise_gray

def calc_visu_thr(N, sigma):
    thr = sigma * np.sqrt(2 * np.log(N))
    return thr

def calc_bayes_thr(coeff, sigma):
    eps = 1e-6
    signal_var = np.mean(coeff ** 2) - sigma ** 2
    signal_var = np.sqrt(max(signal_var, 0)) + eps
    thr = sigma ** 2 / signal_var
    return thr

def shrinkage(coeff, thr, mode="soft"):
    assert mode in {"soft", "hard"}
    out = coeff.copy()
    out[np.abs(coeff) < thr] = 0
    if mode == "soft":
        shrinked = (np.abs(out[np.abs(coeff) > thr]) - thr)
        sign = np.sign(out[np.abs(coeff) > thr])
        out[np.abs(coeff) > thr] = sign * shrinked
    return out

def wavelet_denoise(img, wave, level, sigma,
                    shrink_mode="soft", thr_mode="visu"):
    assert thr_mode in {"visu", "bayes"}
    dwt_out = pywt.wavedec2(img, wavelet=wave, level=level)
    dn_out = [dwt_out[0]]
    n_level = len(dwt_out) - 1
    if thr_mode == "visu":
        thr = calc_visu_thr(img.size, sigma)
    for lvl in range(1, n_level + 1):
        cur_lvl = list()
        for sub in range(len(dwt_out[lvl])):
            coeff = dwt_out[lvl][sub]
            if thr_mode == "bayes":
                thr = calc_bayes_thr(coeff, sigma)
            out = shrinkage(coeff, thr, mode=shrink_mode)
            cur_lvl.append(out)
        dn_out.append(tuple(cur_lvl))
    recon = pywt.waverec2(dn_out, wavelet=wave)
    return recon


if __name__ == "__main__":

    os.makedirs('results/wavelet', exist_ok=True)
    img = cv2.imread('../datasets/srdata/Set12/02.png')[...,0]
    noisy = add_gaussian_noise_gray(img, sigma=15)
    cv2.imwrite(f'./results/wavelet/noisy_15.png', noisy)

    for mode in ["hard", "soft"]:
        for thr in ["visu", "bayes"]:
            denoised = wavelet_denoise(img,
                            wave="haar", level=3, sigma=15,
                            shrink_mode=mode, thr_mode=thr)
            cv2.imwrite(f'./results/wavelet/dn_{mode}_{thr}.png', denoised)


