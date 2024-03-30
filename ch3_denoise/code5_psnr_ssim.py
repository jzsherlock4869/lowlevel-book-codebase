import os
import cv2
import numpy as np

def calc_psnr(img1, img2, peak_value=255.0):
    img1, img2 = np.float64(img1), np.float64(img2)
    mse = np.mean((img1 - img2) ** 2)
    psnr = 10 * np.log10((peak_value ** 2) / mse)
    return psnr

def calc_ssim(img1, img2, win_size=11, sigma=1.5, L=255.0):
    assert img1.shape == img2.shape
    if img1.ndim == 2:
        img1 = np.expand_dims(img1, axis=2)
        img2 = np.expand_dims(img2, axis=2)
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    img1, img2 = np.float64(img1), np.float64(img2)
    ssim_ls = list()
    winr = (win_size - 1) // 2
    for ch_id in range(img1.shape[-1]):
        cur_img1 = img1[:, :, ch_id]
        cur_img2 = img2[:, :, ch_id]
        mu1 = cv2.GaussianBlur(cur_img1, \
                ksize=[win_size, win_size], sigmaX=sigma)
        mu2 = cv2.GaussianBlur(cur_img2, \
                ksize=[win_size, win_size], sigmaX=sigma)
        mu11 = cv2.GaussianBlur(cur_img1**2, \
                ksize=[win_size, win_size], sigmaX=sigma)
        mu22 = cv2.GaussianBlur(cur_img2**2, \
                ksize=[win_size, win_size], sigmaX=sigma)
        mu12 = cv2.GaussianBlur(cur_img1*cur_img2, \
                ksize=[win_size, win_size], sigmaX=sigma)
        sigma1_2 = mu11 - mu1 ** 2
        sigma2_2 = mu22 - mu2 ** 2
        sigma12 = mu12 - mu1 * mu2
        nume = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
        deno = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_2 + sigma2_2 + C2)
        ssim_map = nume / deno
        ssim = np.mean(ssim_map[winr:-winr, winr:-winr])
        ssim_ls.append(ssim)
    return np.mean(ssim_ls)

if __name__ == "__main__":

    os.makedirs('results/psnr_ssim', exist_ok=True)
    img_path = '../datasets/samples/baboon256rgb.png'
    img = cv2.imread(img_path)
    img_blur5 = cv2.blur(img, (5, 5))
    img_blur10 = cv2.blur(img, (10, 10))
    img_ratio = img * 0.8 + 100
    img_minus5 = img - 10.0
    img_noisy25 = img + np.random.randn(*img.shape) * 25

    cv2.imwrite('results/psnr_ssim/img_blur5.png',\
                np.clip(img_blur5, 0, 255))
    cv2.imwrite('results/psnr_ssim/img_blur10.png',\
                np.clip(img_blur10, 0, 255))
    cv2.imwrite('results/psnr_ssim/img_ratio.png',\
                np.clip(img_ratio, 0, 255))
    cv2.imwrite('results/psnr_ssim/img_minus5.png',\
                np.clip(img_minus5, 0, 255))
    cv2.imwrite('results/psnr_ssim/img_noisy25.png',\
                np.clip(img_noisy25, 0, 255))

    print("====== PSNR ======")
    psnr_blur5 = calc_psnr(img_blur5, img)
    psnr_blur10 = calc_psnr(img_blur10, img)
    psnr_ratio = calc_psnr(img_ratio, img)
    psnr_minus5 = calc_psnr(img_minus5, img)
    psnr_noisy25 = calc_psnr(img_noisy25, img)

    print("blur5 PSNR: ", psnr_blur5)
    print("blur10 PSNR: ", psnr_blur10)
    print("ratio PSNR: ", psnr_ratio)
    print("minus5 PSNR: ", psnr_minus5)
    print("noisy25 PSNR: ", psnr_noisy25)

    print("====== SSIM ======")
    ssim_blur5 = calc_ssim(img_blur5, img)
    ssim_blur10 = calc_ssim(img_blur10, img)
    ssim_ratio = calc_ssim(img_ratio, img)
    ssim_minus5 = calc_ssim(img_minus5, img)
    ssim_noisy25 = calc_ssim(img_noisy25, img)

    print("blur5 SSIM: ", ssim_blur5)
    print("blur10 SSIM: ", ssim_blur10)
    print("ratio SSIM: ", ssim_ratio)
    print("minus5 SSIM: ", ssim_minus5)
    print("noisy25 SSIM: ", ssim_noisy25)
