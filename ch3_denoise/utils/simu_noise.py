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

def add_gaussian_noise_gray(img_gray, sigma=15):
    h, w = img_gray.shape
    gaussian_noise = np.random.randn(h, w) * sigma
    noisy = img_gray + gaussian_noise
    noisy = np.round(np.clip(noisy, a_min=0, a_max=255))
    return noisy.astype(np.uint8)

def add_gaussian_noise_color(img_rgb, sigma=15):
    h, w, c = img_rgb.shape
    gaussian_noise = np.random.randn(h, w, c) * sigma
    noisy = img_rgb + gaussian_noise
    noisy = np.round(np.clip(noisy, a_min=0, a_max=255))
    return noisy.astype(np.uint8)

def add_salt_pepper_noise(img,
                          salt_ratio=0.01,
                          pepper_ratio=0.01):
    speckle_noisy = img.copy()
    h, w = speckle_noisy.shape[:2]
    num_salt = np.ceil(img.size * salt_ratio)
    salt_rid = np.random.choice(h, int(num_salt))
    salt_cid = np.random.choice(w, int(num_salt))
    speckle_noisy[salt_rid, salt_cid, ...] = 255
    num_pepper = np.ceil(img.size * pepper_ratio)
    pepper_rid = np.random.choice(h, int(num_pepper))
    pepper_cid = np.random.choice(w, int(num_pepper))
    speckle_noisy[pepper_rid, pepper_cid, ...] = 0
    return speckle_noisy

