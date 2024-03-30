import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


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

def estimate_atmospheric_light(img, dark, percent=0.1):
    h, w = img.shape[:2]
    img_vec = img.reshape(h * w, 3)
    dark_vec = dark.reshape(h * w)
    topk = int(h * w * percent / 100)
    idx = np.argsort(dark_vec)[::-1][:topk]
    atm_light = np.max(img_vec[idx, :], axis=0)  # [topk, 3] -> [3]
    return atm_light.astype(np.float32)

def calc_transmission(img, atmo, patch_size, omega, t0, radius=11, eps=1e-3):
    dark = calc_dark_channel(img / atmo, patch_size)
    trans = 1 - omega * dark
    trans = np.maximum(trans, t0)
    guide = img.astype(np.float32) / 255.0
    trans = trans.astype(np.float32)
    trans_refined = cv2.ximgproc.guidedFilter(
                    guide, trans, radius=radius, eps=eps)
    return trans_refined

def calc_scene_radiance(img, trans, atmo):
    atmo = np.expand_dims(atmo, axis=[0, 1])
    trans = np.expand_dims(trans, axis=2)
    radiance = (img - atmo) / trans + atmo
    radiance = np.clip(radiance, a_min=0, a_max=255).astype(np.uint8)
    return radiance

def dehaze_dark_channel_prior(img, cfg, return_all=True):
    dark_channel = calc_dark_channel(img, patch_size=cfg["patch_size"])
    atmo_light = estimate_atmospheric_light(img, 
                            dark_channel, percent=cfg["atmo_est_percent"])
    trans_est = calc_transmission(img, atmo_light, cfg["patch_size"],
                                  cfg["omega"], cfg["t0"],
                                  cfg["guide_radius"], cfg["guide_eps"])
    dehazed = calc_scene_radiance(img, trans_est, atmo_light)

    if return_all:
        return {
            "dark_channel": dark_channel,
            "atmo_light": atmo_light,
            "trans_est": trans_est,
            "dehazed": dehazed
        }
    else:
        return dehazed


if __name__ == "__main__":
    # 测试去雾效果
    cfg = {
        "patch_size": 5,
        "atmo_est_percent": 0.1,
        "omega": 1.0,
        "t0": 0.4,
        "guide_radius": 21,
        "guide_eps": 0.1
    }

    img_path = "../datasets/hazy/IMG_20200405_163759.jpg"

    img = cv2.imread(img_path)[:,:,::-1]
    h, w = img.shape[:2]
    img = cv2.resize(img, (w // 3, h // 3))

    output = dehaze_dark_channel_prior(img, cfg, return_all=True)
    dark_channel = output["dark_channel"]
    atmo_light = output["atmo_light"]
    trans_est = output["trans_est"]
    dehazed = output["dehazed"]

    print('estimated atmosphere light : ', atmo_light)
    plt.figure()
    plt.imshow(img)
    plt.title('hazy image')
    plt.xticks([]), plt.yticks([])

    plt.figure()
    plt.imshow(dark_channel, cmap='gray')
    plt.title('dark channel')
    plt.xticks([]), plt.yticks([])

    plt.figure()
    plt.imshow(trans_est, cmap='gray')
    plt.title('transmission map')
    plt.xticks([]), plt.yticks([])

    plt.figure()
    plt.imshow(dehazed)
    plt.title('dehazed')
    plt.xticks([]), plt.yticks([])
    plt.show()
