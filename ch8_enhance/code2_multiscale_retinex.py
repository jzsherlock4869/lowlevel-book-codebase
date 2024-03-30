import cv2
import numpy as np
import os

def single_scale_retinex(img, sigma):
    img = img.astype(np.float32) + 1.0
    img_blur = cv2.GaussianBlur(img, ksize=[0, 0], sigmaX=sigma)
    retinex = np.log(img / 256.0) - np.log(img_blur / 256.0)
    for cidx in range(retinex.shape[-1]):
        retinex[:, :, cidx] = cv2.normalize(retinex[:, :, cidx],\
                                     None, 0, 255, cv2.NORM_MINMAX)
    retinex = np.clip(retinex, a_min=0, a_max=255).astype(np.uint8)
    return retinex


def multi_scale_retinex(img, sigma_ls):
    img = img.astype(np.float32) + 1.0
    retinex_sum = None
    num_scale = len(sigma_ls)
    for sigma in sigma_ls:
        img_blur = cv2.GaussianBlur(img, ksize=[0, 0], sigmaX=sigma)
        retinex = np.log(img / 256.0) - np.log(img_blur / 256.0)
        if retinex_sum is None:
            retinex_sum = retinex
        else:
            retinex_sum += retinex
    retinex = retinex_sum / num_scale
    for cidx in range(retinex.shape[-1]):
        retinex[:, :, cidx] = cv2.normalize(retinex[:, :, cidx],\
                                    None, 0, 255, cv2.NORM_MINMAX)
    retinex = np.clip(retinex, a_min=0, a_max=255).astype(np.uint8)
    return retinex


def color_restoration(img, retinex):
    A = np.log(np.sum(img, axis=2, keepdims=True))
    retinex = retinex * (np.log(125.0 * img) - A)
    return retinex


def simplest_color_balance(img, dark_percent, light_percent):
    N = img.shape[0] * img.shape[1]
    dark_thr, light_thr = int(dark_percent * N), int(light_percent * N)
    res = img.copy()
    for cidx in range(img.shape[-1]):
        cur_ch = res[:, :, cidx]
        sorted_ch = sorted(cur_ch.flatten())
        mini, maxi = sorted_ch[dark_thr], sorted_ch[-light_thr]
        res[:, :, cidx] = np.clip((cur_ch - mini) / (maxi - mini),
                            a_min=0, a_max=1) * 255.0
    return res


def MSR_color_restoration(img, sigma_ls, dark_percent, light_percent):
    img = img.astype(np.float32) + 1.0
    retinex_sum = None
    num_scale = len(sigma_ls)
    for sigma in sigma_ls:
        img_blur = cv2.GaussianBlur(img, ksize=[0, 0], sigmaX=sigma)
        retinex = np.log(img) - np.log(img_blur)
        if retinex_sum is None:
            retinex_sum = retinex
        else:
            retinex_sum += retinex
    retinex = retinex_sum / num_scale
    output = color_restoration(img, retinex)
    output = simplest_color_balance(output, dark_percent, light_percent)
    output = np.clip(output, a_min=0, a_max=255).astype(np.uint8)
    return output


def MSR_color_preservation(img, sigma_ls, dark_percent, light_percent):
    img = img.astype(np.float32) + 1.0
    retinex_sum = None
    gray = np.mean(img, axis=2, keepdims=True)
    num_scale = len(sigma_ls)
    for sigma in sigma_ls:
        gray_blur = cv2.GaussianBlur(gray, ksize=[0, 0], sigmaX=sigma)
        gray_blur = np.expand_dims(gray_blur, axis=2)
        retinex = np.log(gray) - np.log(gray_blur)
        if retinex_sum is None:
            retinex_sum = retinex
        else:
            retinex_sum += retinex
    retinex = retinex_sum / num_scale
    retinex = simplest_color_balance(retinex, dark_percent, light_percent)
    B = np.max(img, axis=2, keepdims=True)
    gray_ratio = retinex / gray
    A = np.minimum(255.0 / B, gray_ratio)
    output = (img * A).astype(np.uint8)
    return output


if __name__ == "__main__":

    os.makedirs('./results/retinex', exist_ok=True)
    impath = "../datasets/lowlight/IMG_20200419_191100.jpg"
    lowlight = cv2.imread(impath)[:,:,::-1]
    h, w = lowlight.shape[:2]
    img = cv2.resize(lowlight, (w//4, h//4))
    sigma_ls = [15, 80, 250]
    for sigma in sigma_ls:
        ssr_out = single_scale_retinex(img, sigma=sigma)
        cv2.imwrite(f'./results/retinex/ssr_out_sigma{sigma}.png',\
                                            ssr_out[:,:,::-1])
    msr_out = multi_scale_retinex(img, sigma_ls=sigma_ls)
    cv2.imwrite(f'./results/retinex/msr_out.png', msr_out[:,:,::-1])
    msrcr_out = MSR_color_restoration(img, sigma_ls, 0.02, 0.02)
    cv2.imwrite(f'./results/retinex/msrcr_out.png', msrcr_out[:,:,::-1])
    msrcp_out = MSR_color_preservation(img, sigma_ls, 0.02, 0.02)
    cv2.imwrite(f'./results/retinex/msrcp_out.png', msrcp_out[:,:,::-1])

