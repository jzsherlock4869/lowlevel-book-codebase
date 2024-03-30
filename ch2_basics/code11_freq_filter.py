import os
import cv2
import numpy as np

os.makedirs('results/fft_test', exist_ok=True)

img_path = '../datasets/samples/butterfly256rgb.png'
img_bgr = cv2.imread(img_path)
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) / 255.0
h, w = img.shape
# 通过二维FFT转到频域
spec = np.fft.fftshift(np.fft.fft2(img))
# 频域低通、高通滤波
low_mask = np.zeros((h, w), dtype=np.float32)
cv2.circle(low_mask, (w // 2, h // 2), 10, 1, -1)
high_mask = 1 - low_mask
# 频域mask与频谱相乘
lp_spec = spec * low_mask
hp_spec = spec * high_mask
# 计算低通和高通滤波后的振幅谱并保存
lp_amp = np.abs(lp_spec)
hp_amp = np.abs(hp_spec)
cv2.imwrite('./results/fft_test/lowpass_amp.png', \
            np.clip(lp_amp, 0, 200) / 200 * 255)
cv2.imwrite('./results/fft_test/highpass_amp.png', \
            np.clip(hp_amp, 0, 200) / 200 * 255)
# FFT反变换回空域并取实部，得到频域滤波后的图像
lp_img = np.fft.ifft2(np.fft.fftshift(lp_spec)).real
hp_img = np.fft.ifft2(np.fft.fftshift(hp_spec)).real
cv2.imwrite('./results/fft_test/lowpass_img.png', lp_img * 255)
cv2.imwrite('./results/fft_test/highpass_img.png', hp_img * 255)
