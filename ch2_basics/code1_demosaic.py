import os
import cv2
import numpy as np

os.makedirs('results/demosaic', exist_ok=True)
bayer_path = "../datasets/samples/sample_bayer_zebra.bmp"
bayer = cv2.imread(bayer_path, cv2.IMREAD_UNCHANGED)

# 用bilinear进行demosaicing
img_bgr_bilinear = cv2.cvtColor(bayer, cv2.COLOR_BayerGB2BGR)
cv2.imwrite('results/demosaic/demosaic_bilinear.png', img_bgr_bilinear)
# 用VNG方法做demosaicing（Variable Number of Gradients）
img_bgr_vng = cv2.cvtColor(bayer, cv2.COLOR_BayerGB2BGR_VNG)
cv2.imwrite('results/demosaic/demosaic_vng.png', img_bgr_vng)
