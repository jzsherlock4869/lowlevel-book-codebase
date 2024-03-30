import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def reinhard_tonemapping(img,
                   mid_gray=0.18,
                   num_scale=8,
                   alpha=0.35355,
                   ratio=1.6,
                   phi=8.0,
                   epsilon=0.05):
    img = img / img.max()
    H, W = img.shape[:2]
    B, G, R = cv2.split(img)
    Lw = 0.27 * R + 0.67 * G + 0.06 * B
    delta = 1e-10
    mLw = np.exp(np.mean(np.log(Lw + delta)))
    L = (mid_gray / mLw) * Lw
    v1_ls = list()
    for i in range(num_scale):
        sigma = alpha * (ratio ** i)
        local_gauss = cv2.GaussianBlur(L, ksize=[0,0], sigmaX=sigma)
        v1_ls.append(local_gauss)
    v_ls = list()
    for i in range(num_scale - 1):
        nume = np.abs(v1_ls[i] - v1_ls[i + 1])
        deno = mid_gray * (2**phi) / (ratio**i) + v1_ls[i]
        v_ls.append(nume / deno)

    v1ls_np = np.stack(v1_ls, axis=2)
    vls_np = np.stack(v_ls, axis=2)
    vsm = v1_ls[-1].copy()

    for i in range(H):
        for j in range(W):
            vec = vls_np[i, j, :]
            for cur_s in range(num_scale - 1):
                if vec[cur_s] > epsilon:
                    vsm[i, j] = v1ls_np[i, j, cur_s]
                    break
    Ld = np.clip(L / (1 + vsm), a_min=0, a_max=1)
    out_color = np.expand_dims(Ld / (Lw + 1e-10), axis=2) * img
    out_color = np.clip(out_color, a_min=0, a_max=1)
    return out_color



if __name__ == "__main__":

    os.makedirs('results/reinhard', exist_ok=True)
    # test image: https://www.pauldebevec.com/Research/HDR/
    hdr_path = '../datasets/hdr/memorial.hdr'
    img = cv2.imread(hdr_path, flags = cv2.IMREAD_ANYDEPTH)
    out = reinhard_tonemapping(img)
    inp = img / img.max()
    inp = np.power(inp, 1/2.2)

    tone_in_8u = (inp  * 255.0).astype(np.uint8)
    tone_out_8u = (out  * 255.0).astype(np.uint8)
    cv2.imwrite('results/reinhard/reinhard_in.png', tone_in_8u)
    cv2.imwrite('results/reinhard/reinhard_out.png', tone_out_8u)
