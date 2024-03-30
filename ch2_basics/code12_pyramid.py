import os
import numpy as np
import cv2


def build_gaussian_pyr(img, pyr_level=5):
    # 建立高斯金字塔
    gauss_pyr = list()
    cur = img.copy()
    gauss_pyr.append(cur)
    # 逐步下采样并加入金字塔
    for i in range(1, pyr_level):
        cur_h, cur_w = cur.shape[:2]
        cur = cv2.pyrDown(cur, 
            dstsize=(int(np.round(cur_w / 2)), int(np.round(cur_h / 2))))
        gauss_pyr.append(cur)
    return gauss_pyr

def build_laplacian_pyr(img, pyr_level=5):
    # 建立拉普拉斯金字塔
    laplace_pyr = list()
    cur = img.copy()
    # 逐步下采样，并计算差值，加入拉普拉斯塔
    for i in range(pyr_level - 1):
        cur_h, cur_w = cur.shape[:2]
        down = cv2.pyrDown(cur, 
                dstsize=(int(np.round(cur_w / 2)), int(np.round(cur_h / 2))))
        up = cv2.pyrUp(down, dstsize=(cur_w, cur_h))
        lap_layer = cur.astype(np.float32) - up
        laplace_pyr.append(lap_layer)
        cur = down
    # 最后一层为高斯降采样，非差值
    laplace_pyr.append(cur)
    return laplace_pyr

def collapse_laplacian_pyr(laplace_pyr):
    # 从拉普拉斯金字塔重建图像
    pyr_level = len(laplace_pyr)
    # 逐步上采样，并加入差剖面图
    tmp = laplace_pyr[-1].astype(np.float32)
    for i in range(1, pyr_level):
        lvl = pyr_level - 1 - i
        cur = laplace_pyr[lvl]
        cur_h, cur_w, _ = cur.shape
        up = cv2.pyrUp(tmp, dstsize=(cur_w, cur_h))
        tmp = cv2.add(cur, up)
    return tmp


if __name__ == "__main__":
    os.makedirs('results/pyramid', exist_ok=True)
    img_path = '../datasets/samples/cat2.png'
    image = cv2.imread(img_path)
    print(f"input image size: ", image.shape)

    # 测试结果
    gauss_pyr = build_gaussian_pyr(image)
    # 打印高斯金字塔各层图像尺寸，并进行保存
    for idx, layer in enumerate(gauss_pyr):
        print(f"[Gaussian Pyramid] layer: {idx}, size: {layer.shape}")
        cv2.imwrite(f'./results/pyramid/gauss_lyr{idx}.png', layer)

    laplacian_pyr = build_laplacian_pyr(image)
    # 打印拉普拉斯金字塔各层图像尺寸，并进行保存
    os.makedirs('./results/pyramid/', exist_ok=True)
    for idx, layer in enumerate(laplacian_pyr):
        print(f"[Laplacian Pyramid] layer: {idx}, size: {layer.shape}")
        cv2.imwrite(f'./results/pyramid/laplace_lyr{idx}.png',
                     np.abs(layer).astype(np.uint8))
