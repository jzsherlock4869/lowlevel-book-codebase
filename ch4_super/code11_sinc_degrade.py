import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy import special

def circular_lowpass_kernel(cutoff=3, kernel_size=7, pad_to=0):
    """
    2D sinc filter
    代码来源 (official realESRGAN): 
    https://github.com/XPixelGroup/
    BasicSR/blob/master/basicsr/data/degradations.py
    """
    assert kernel_size % 2 == 1, \
                'Kernel size must be an odd number.'
    kernel = np.fromfunction(
        lambda x, y: cutoff * special.j1(cutoff * np.sqrt(
            (x - (kernel_size - 1) / 2)**2 
            + (y - (kernel_size - 1) / 2)**2))
            / (2 * np.pi * np.sqrt(
            (x - (kernel_size - 1) / 2)**2 
            + (y - (kernel_size - 1) / 2)**2) + 1e-10),
            [kernel_size, kernel_size])
    kernel[(kernel_size - 1) // 2,
           (kernel_size - 1) // 2] = cutoff**2 / (4 * np.pi)
    kernel = kernel / np.sum(kernel)
    if pad_to > kernel_size:
        pad_size = (pad_to - kernel_size) // 2
        kernel = np.pad(kernel, 
                    ((pad_size, pad_size), (pad_size, pad_size)))
    return kernel


def simulate_ringing(img, cutoff=np.pi/3, ksize=31, pad_to=0):
    """
        img: torch.Tensor [h, w, c]
        cutoff, ksize, pad_to: 对应与sinc函数参数
    """
    # 获取 kernel
    sinc_kernel = circular_lowpass_kernel(cutoff, ksize, pad_to)
    ksize = max(ksize, pad_to)
    sinc = torch.Tensor(sinc_kernel).reshape(1, 1, ksize, ksize)
    # 计算卷积
    pad = ksize // 2
    h, w, c = img.size()
    img = img.permute(2,0,1).unsqueeze(0)
    img = F.pad(img, (pad, pad, pad, pad), mode='reflect')
    img = img.transpose(0, 1) # [c, 1, h_paded, w_paded]
    ringed = F.conv2d(img, sinc).squeeze(1).permute(1, 2, 0)
    return ringed


if __name__ == "__main__":
    img = cv2.imread('../datasets/srdata/Set14/flowers.bmp')
    img_ten = torch.FloatTensor(img)
    ring_out = simulate_ringing(img_ten).numpy()
    ring_out = np.clip(ring_out, a_min=0, a_max=255)
    os.makedirs('./results/degrade/', exist_ok=True)
    cv2.imwrite('./results/degrade/sinc.png', ring_out.astype(np.uint8))