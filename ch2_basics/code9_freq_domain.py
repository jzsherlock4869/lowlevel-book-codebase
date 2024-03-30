import os
import cv2
import numpy as np

os.makedirs('results/fft_test', exist_ok=True)

# 图像及其频谱图
img_path = '../datasets/samples/butterfly256rgb.png'
img_bgr = cv2.imread(img_path)
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
cv2.imwrite('results/fft_test/butterfly_gray.png', img)
img = img / 255.0
# 通过二维FFT转到频域
spec = np.fft.fftshift(np.fft.fft2(img))
print(f"frequency domain, size is {spec.shape}, type is {spec.dtype}")
# 计算振幅谱和相位谱
amp, phase = np.abs(spec), np.angle(spec)
print(f"amplitude max: {np.max(amp):.4f}, min: {np.min(amp):.4f}")
print(f"amplitude max: {np.max(phase):.4f}, min: {np.min(phase):.4f}")

cv2.imwrite('./results/fft_test/amp.png', \
                np.clip(amp, 0, 200) / 200 * 255)
cv2.imwrite('./results/fft_test/phase.png', \
                (phase + np.pi) / (2 * np.pi) * 255)
