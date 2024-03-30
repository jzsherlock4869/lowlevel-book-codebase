import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def optical_flow_warp(img1, img2, method='deepflow'):
    assert img1.shape == img2.shape
    h, w = img1.shape[:2]
    if img1.ndim == 3 and img1.shape[-1] == 3:
        # 将图像转换为灰度图, 用于光流计算
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # 需要安装下列package
    # pip install opencv-contrib-python
    # pip list | grep opencv
    #     opencv-contrib-python       4.7.0.72
    #     opencv-python               4.7.0.72
    if method == 'deepflow':
        of_func = cv2.optflow.createOptFlow_DeepFlow()
    elif method == 'farneback':
        of_func = cv2.optflow.createOptFlow_Farneback()
    elif method == 'pcaflow':
        of_func = cv2.optflow.createOptFlow_PCAFlow()
    # 计算得到的光流是后一帧到前一帧的变化量，为了将前一帧映射为后一帧，需要反向
    # 参考：https://docs.opencv.org/4.7.0/dc/d6b/group__video__track.html
    flow = -1.0 * of_func.calc(img1_gray, img2_gray, None)
    remap_mat = np.zeros_like(flow)
    # flow size: [h, w, 2], 通道0：dx，通道1：dy
    # 1 x w
    remap_mat[:, :, 0] = flow[:, :, 0] + np.arange(w)
    # h x 1
    remap_mat[:, :, 1] = flow[:, :, 1] + np.arange(h)[:, np.newaxis]
    res = cv2.remap(img1, remap_mat, None, cv2.INTER_LINEAR)

    # 将光流向量图像转换为BGR彩色图像
    vis_hsv = np.zeros((h, w, 3), dtype=np.uint8)
    vis_hsv[..., 1] = 255
    mag, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    vis_hsv[..., 0] = angle * 255 / np.pi / 2
    vis_hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_vis = cv2.cvtColor(vis_hsv, cv2.COLOR_HSV2BGR)
    return res, flow, flow_vis

if __name__ == "__main__":

    os.makedirs('results/optical_flow', exist_ok=True)
    frame0 = cv2.imread('../datasets/frames/frame_1.png')
    frame1 = cv2.imread('../datasets/frames/frame_2.png')
    H, W = frame0.shape[:2]
    h, w = H // 2, W // 2
    frame0 = cv2.resize(frame0, (w, h), interpolation=cv2.INTER_AREA)
    frame1 = cv2.resize(frame1, (w, h), interpolation=cv2.INTER_AREA)

    for method in ['farneback', 'pcaflow', 'deepflow']:
        warped_frame0, optical_flow, flow_vis = \
            optical_flow_warp(frame0, frame1, method)
        cv2.imwrite(f'results/optical_flow/{method}_warped_frame0.png',\
                warped_frame0)
        cv2.imwrite(f'results/optical_flow/{method}_flowmap.png',\
                flow_vis)
