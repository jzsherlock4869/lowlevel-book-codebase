import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def unsharpen_mask(img, sigma, w):
    blur = cv2.GaussianBlur(img, ksize=[0, 0], sigmaX=sigma)
    usm = cv2.addWeighted(img, 1 + w, blur, -w, 0)
    return usm

if __name__ == "__main__":

    os.makedirs('results/usm', exist_ok=True)
    img = cv2.imread('../datasets/srdata/Set5/butterfly_GT.bmp')[:,:,::-1]
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)

    # USM 锐化参数
    sigma_list = [1.0, 5.0, 10.0]
    w_list = [0.1, 0.8, 1.5]

    fig = plt.figure(figsize=(10, 10))
    M, N = len(sigma_list), len(w_list)

    cnt = 1
    for sigma in sigma_list:
        for w in w_list:
            usm_out = unsharpen_mask(img, sigma, w)
            fig.add_subplot(M, N, cnt)
            plt.imshow(usm_out)
            plt.axis('off')
            plt.title(f'sigma={sigma}, w={w}')
            cnt += 1

    plt.savefig(f'results/usm/usm_result.png')
    plt.close()