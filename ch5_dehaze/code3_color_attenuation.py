import cv2
import numpy as np
import matplotlib.pyplot as plt

def miminum_filter(smap, ksize):
    h, w = smap.shape
    output = np.zeros_like(smap)
    hw = ksize // 2
    for i in range(h):
        for j in range(w):
            t, b = max(0, i - hw), min(i + hw, h - 1)
            l, r = max(0, j - hw), min(j + hw, w - 1)
            output[i, j] = np.min(smap[t:b+1, l:r+1])
    return output

def calc_depth_map(img,
                   minfilt_ksize,
                   guide_radius,
                   guide_eps):
    img = img.astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    satu, value = hsv[..., 1], hsv[..., 2]
    epsilon = np.random.normal(0, 0.041337, img.shape[:2])
    depth = 0.121779 + 0.959710 * value - 0.780245 * satu + epsilon
    depth_mini = miminum_filter(depth, minfilt_ksize)
    depth_mini = depth_mini.astype(np.float32)
    depth_guide = cv2.ximgproc.guidedFilter(img,
            depth_mini, radius=guide_radius, eps=guide_eps)
    return depth_guide, depth_mini, depth

def estimate_atmospheric_light(img, depth, percent=0.1):
    h, w = img.shape[:2]
    n_sample = int(h * w * percent / 100)
    depth_v = depth.reshape((h * w))
    img_v = img.reshape((h * w, 3))
    loc = np.argsort(depth_v)[::-1] # descending
    # 找到深度最大的 n_sample 个备选点
    Acand = img_v[loc[:n_sample], :] # [n_sample, 3]
    Anorm = np.linalg.norm(Acand, axis=1)
    loc = np.argsort(Anorm)[::-1]
    select_num = min(n_sample, 20)
    # 筛选norm最大的点（最多20个）取最大值
    Asele = Acand[loc[:select_num], :]
    A = np.max(Asele, axis=0)
    return A

def calc_scene_radiance(img, trans, atmo):
    atmo = np.expand_dims(atmo, axis=[0, 1])
    trans = np.expand_dims(trans, axis=2)
    radiance = (img - atmo) / trans + atmo
    radiance = np.clip(radiance, a_min=0, a_max=1)
    radiance = (radiance * 255).astype(np.uint8)
    return radiance

def dehaze_color_attenuation(img, cfg, return_all=True):
    img = img / 255.0
    depth, _, _ = calc_depth_map(img,
                   cfg["minfilt_ksize"],
                   cfg["guide_radius"],
                   cfg["guide_eps"])
    trans_est = np.exp( - cfg["beta"] * depth)
    t_min, t_max = cfg["t_min"], cfg["t_max"]
    trans_est = np.clip(trans_est, a_min=t_min, a_max=t_max)
    atmo = estimate_atmospheric_light(img, depth, cfg["percent"])
    dehazed = calc_scene_radiance(img, trans_est, atmo)
    if return_all:
        return {
            "dehazed": dehazed,
            "depth": depth,
            "trans_est": trans_est,
            "atmo_light": atmo
        }
    else:
        return dehazed


if __name__ == "__main__":
    cfg = {
        "minfilt_ksize": 21,
        "guide_radius": 11,
        "guide_eps": 5e-3,
        "beta": 1.0,
        "t_min": 0.05,
        "t_max": 1.0,
        "percent": 0.1
    }

    img_path = "../datasets/hazy/IMG_20201002_102129.jpg"

    img = cv2.imread(img_path)[:,:,::-1]
    h, w = img.shape[:2]
    img = cv2.resize(img, (w // 3, h // 3))

    # ================================ #
    # 测试基于颜色衰减先验的 depth map     #
    # ================================ #
    dmap_guide, dmap_mini, dmap \
          = calc_depth_map(img / 255.0,
                            minfilt_ksize=21,
                            guide_radius=11,
                            guide_eps=5e-3)
    plt.figure()
    plt.imshow(dmap, cmap='gray')
    plt.title('color attenuation depth')
    plt.xticks([]), plt.yticks([])
    plt.figure()
    plt.imshow(dmap_mini, cmap='gray')
    plt.title('minimum filtered depth')
    plt.xticks([]), plt.yticks([])
    plt.figure()
    plt.imshow(dmap_guide, cmap='gray')
    plt.title('guided filtered depth')
    plt.xticks([]), plt.yticks([])
    plt.show()

    # ================================ #
    #         测试去雾效果               #
    # ================================ #
    output = dehaze_color_attenuation(img, cfg, return_all=True)
    depth = output["depth"]
    trans_est = output["trans_est"]
    atmo_light = output["atmo_light"]
    dehazed = output["dehazed"]

    print('estimated atmosphere light : ', atmo_light)
    plt.figure()
    plt.imshow(img)
    plt.title('hazy image')
    plt.xticks([]), plt.yticks([])
    plt.figure()
    plt.imshow(depth, cmap='gray')
    plt.title('depth')
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
