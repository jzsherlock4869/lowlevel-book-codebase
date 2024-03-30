import os
import cv2
import matplotlib.pyplot as plt

os.makedirs('results/colorspace', exist_ok=True)

# 读取示例图像（默认为BGR格式）
img_path = '../datasets/samples/lena256rgb.png'
img_bgr = cv2.imread(img_path)

# 将图像从BGR转换到YCrCb空间
img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
# 将图像从BGR转换到HSV空间, _FULL用于映射到255范围
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV_FULL)
# 将图像从BGR转换到CIE-L*a*b*空间
img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

def vis_3channels(img, labels, save_path):
    fig = plt.figure(figsize=(8, 3))
    for i in range(3):
        ax = fig.add_subplot(1, 3, i + 1)
        im = ax.imshow(img[:,:,i], cmap='jet', vmin=0, vmax=255)
        fig.colorbar(im, ax=ax, fraction=0.045)
        ax.set_axis_off()
        ax.set_title(labels[i] + ' channel')
    plt.savefig(save_path)
    plt.close()

vis_3channels(img_ycrcb, ['Y','Cr','Cb'], './results/colorspace/ycrcb.png')
vis_3channels(img_hsv, ['H','S','V'], './results/colorspace/hsv.png')
vis_3channels(img_lab, ['L*','a*','b*'], './results/colorspace/lab.png')
