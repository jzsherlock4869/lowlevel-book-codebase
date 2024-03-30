import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def invert_img(img):
    return 255 - img

def calc_dark_channel(img, patch_size):
    h, w = img.shape[:2]
    min_ch = np.min(img, axis=2)
    dark_channel = np.zeros((h, w))
    r = patch_size // 2
    for i in range(h):
        for j in range(w):
            top, bottom = max(0, i - r), min(i + r, h - 1)
            left, right = max(0, j - r), min(j + r, w - 1)
            dark_channel[i, j] = np.min(min_ch[top:bottom + 1, left:right + 1])
    return dark_channel

def calc_A(img, dark, topk=100):
    h, w = img.shape[:2]
    img_vec = img.reshape(h * w, 3)
    dark_vec = dark.reshape(h * w)
    idx = np.argsort(dark_vec)[::-1][:topk]
    candidate_vec = img_vec[idx, :]
    sum_vec = np.sum(candidate_vec, axis=1)
    atm_light = candidate_vec[np.argmax(sum_vec), :]
    return atm_light.astype(np.float32)

def calc_t(img, A, patch_size, omega):
    dark = calc_dark_channel(img / A, patch_size)
    t = 1 - omega * dark
    t[t < 0.5] = (t[t < 0.5] ** 2) * 2
    return t.astype(np.float32)

def calc_J(img, t, A):
    A = np.expand_dims(A, axis=[0, 1])
    t = np.expand_dims(t, axis=2)
    J = (img - A) / t + A
    J = np.clip(J, a_min=0, a_max=255).astype(np.uint8)
    return J

def lowlight_enhance(img, patch_size=3, omega=0.8):
    rev_img = invert_img(img)
    dark = calc_dark_channel(rev_img, patch_size)
    A = calc_A(rev_img, dark)
    t = calc_t(rev_img, A, patch_size, omega)
    J = calc_J(rev_img, t, A)
    enhanced = invert_img(J)
    return enhanced, rev_img, dark, t


if __name__ == "__main__":
    impath = "../datasets/lowlight/IMG_20200419_191100.jpg"
    lowlight = cv2.imread(impath)[:,:,::-1]
    h, w = lowlight.shape[:2]
    lowlight = cv2.resize(lowlight, (w//4, h//4))
    enhanced, rev_img, dark, trans = lowlight_enhance(lowlight)
    os.makedirs('results/invert_dehaze', exist_ok=True)
    cv2.imwrite(f'results/invert_dehaze/input.png', lowlight[:,:,::-1])
    cv2.imwrite(f'results/invert_dehaze/out.png', enhanced[:,:,::-1])
    cv2.imwrite(f'results/invert_dehaze/rev_in.png', rev_img[:,:,::-1])
    cv2.imwrite(f'results/invert_dehaze/dark_channel.png', dark)
    cv2.imwrite(f'results/invert_dehaze/trans.png',\
                np.clip(trans * 255, 0, 255).astype(np.uint8))

