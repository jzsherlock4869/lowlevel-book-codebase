import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils.pyramid as P

def laplace_fusion(img1, img2, weight1, pyr_level=10, verbose=True):
    """
    args:
        img1, img2: 输入图像, 范围0-255
        weight1: img1的融合权重, 取值0-1之间, img2的权重由1-weight1计算得到
        pyr_level: 金字塔融合层数，如果超过最大值，则置为最大值
    return:
        融合后图像, 范围0-255
    """
    assert img1.shape == img2.shape
    assert img1.shape[:2] == weight1.shape[:2]
    weight2 = 1.0 - weight1
    h, w = img1.shape[:2]
    max_level = int(np.log2(min(h, w)))
    pyr_level = min(max_level, pyr_level)
    print(f"[laplace_fusion] max pyr_level: {max_level},"
          f" set pyr_level: {pyr_level}")
    lap_pyr1 = P.build_laplacian_pyr(img1, pyr_level)
    lap_pyr2 = P.build_laplacian_pyr(img2, pyr_level)
    w_pyr1 = P.build_gaussian_pyr(weight1, pyr_level)
    w_pyr2 = P.build_gaussian_pyr(weight2, pyr_level)
    fused_lap_pyr = list()
    if verbose:
        part1_pyr = list()
        part2_pyr = list()
    for lvl in range(pyr_level):
        w1 = np.expand_dims(w_pyr1[lvl], axis=2) * 1.0
        w2 = np.expand_dims(w_pyr2[lvl], axis=2) * 1.0
        fused_layer = lap_pyr1[lvl] * w1 + lap_pyr2[lvl] * w2
        fused_lap_pyr.append(fused_layer)
        if verbose:
            part1_pyr.append(lap_pyr1[lvl] * w1)
            part2_pyr.append(lap_pyr2[lvl] * w2)
    fused = P.collapse_laplacian_pyr(fused_lap_pyr)
    fused = (np.clip(fused, 0, 255)).astype(np.uint8)
    if verbose:
        return fused, part1_pyr, part2_pyr
    else:
        return fused

if __name__ == "__main__":

    fg = cv2.imread("../datasets/composite/plane/source.jpg")
    bg = cv2.imread("../datasets/composite/plane/target.jpg")
    mask = cv2.imread("../datasets/composite/plane/mask.jpg")[:,:,0] / 255.0
    fused, pyr1, pyr2 = laplace_fusion(fg, bg,
                            mask, pyr_level=5, verbose=True)
    cv2.imwrite('results/laplacian.png', fused)

    fig = plt.figure(figsize=(8, 3))
    n_layers = len(pyr1)
    for i in range(n_layers):
        fig.add_subplot(2, n_layers, i + 1)
        plt.imshow(pyr1[i][..., 0])
        plt.xticks([]), plt.yticks([])
        plt.title(f"FG layer {i + 1}")
        fig.add_subplot(2, n_layers, i + 1 + n_layers)
        plt.imshow(pyr2[i][..., 0])
        plt.xticks([]), plt.yticks([])
        plt.title(f"BG layer {i + 1}")
    plt.show()
