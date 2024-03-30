import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_gaussian_noise_gray(img_gray, sigma=15):
    h, w = img_gray.shape
    gaussian_noise = np.random.randn(h, w) * sigma
    noisy = img_gray + gaussian_noise
    noisy = np.round(np.clip(noisy, a_min=0, a_max=255))
    return noisy.astype(np.uint8)


if __name__ == "__main__":
    os.makedirs('./results/noise_simu', exist_ok=True)
    img_path = '../datasets/srdata/Set12/05.png'
    img = cv2.imread(img_path)[:,:,0]
    gaussian_sigma_ls = [15, 25, 50]
    N = len(gaussian_sigma_ls)
    fig = plt.figure(figsize=(10, 6))
    for sid, sigma in enumerate(gaussian_sigma_ls):
        noisy = add_gaussian_noise_gray(img, sigma)
        fig.add_subplot(2, N, sid + 1)
        plt.imshow(noisy, cmap='gray')
        plt.axis('off')
        plt.title(f'Gaussian, sigma={sigma}')
        fig.add_subplot(2, N, N + sid + 1)
        plt.imshow(np.abs(noisy * 1.0 - img), cmap='gray')
        plt.axis('off')
    plt.savefig(f'results/noise_simu/gaussian.png')
    plt.close()
