import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_poisson_noise_gray(img_gray, scale=0.5):
    noisy = np.random.poisson(img_gray)
    poisson_noise = noisy - img_gray
    noisy = img_gray + scale * poisson_noise
    noisy = np.round(np.clip(noisy, a_min=0, a_max=255))
    return noisy.astype(np.uint8)


if __name__ == "__main__":
    os.makedirs('./results/noise_simu', exist_ok=True)
    img_path = '../datasets/srdata/Set12/05.png'
    img = cv2.imread(img_path)[:,:,0]
    poisson_scale_ls = [0.3, 1.2, 1.8]
    N = len(poisson_scale_ls)
    fig = plt.figure(figsize=(10, 6))
    for sid, s in enumerate(poisson_scale_ls):
        noisy = add_poisson_noise_gray(img, scale=s)
        fig.add_subplot(2, N, sid + 1)
        plt.imshow(noisy, cmap='gray')
        plt.axis('off')
        plt.title(f'Poisson, scale={s}')
        fig.add_subplot(2, N, N + sid + 1)
        plt.imshow(np.abs(noisy * 1.0 - img), cmap='gray')
        plt.axis('off')
    plt.savefig(f'results/noise_simu/poisson.png')
    plt.close()

