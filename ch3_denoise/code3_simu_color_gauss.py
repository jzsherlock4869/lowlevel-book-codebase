import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_gaussian_noise_color(img_rgb, sigma=15):
    h, w, c = img_rgb.shape
    gaussian_noise = np.random.randn(h, w, c) * sigma
    noisy = img_rgb + gaussian_noise
    noisy = np.round(np.clip(noisy, a_min=0, a_max=255))
    return noisy.astype(np.uint8)

if __name__ == "__main__":
    img_path = '../datasets/srdata/Set5/head_GT.bmp'
    img = cv2.imread(img_path)[:,:,::-1]
    gaussian_sigma_ls = [15, 25, 50]
    N = len(gaussian_sigma_ls)
    fig = plt.figure(figsize=(10, 3))
    for sid, sigma in enumerate(gaussian_sigma_ls):
        noisy = add_gaussian_noise_color(img, sigma)
        fig.add_subplot(1, N, sid + 1)
        plt.imshow(noisy)
        plt.axis('off')
        plt.title(f'Gaussian, sigma={sigma}')
    plt.savefig(f'results/noise_simu/gaussian_color.png')
    plt.close()
