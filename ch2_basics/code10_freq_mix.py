import os
import cv2
import numpy as np

os.makedirs('results/fft_test', exist_ok=True)

# 读取两张不同的示例图像，并进行FFT变换
img_path_1 = '../datasets/samples/butterfly256rgb.png'
img_path_2 = '../datasets/samples/lena256rgb.png'
img_bgr_1 = cv2.imread(img_path_1)
img_bgr_2 = cv2.imread(img_path_2)
img1 = cv2.cvtColor(img_bgr_1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img_bgr_2, cv2.COLOR_BGR2GRAY)
cv2.imwrite('results/fft_test/mix_1.png', img1)
cv2.imwrite('results/fft_test/mix_2.png', img2)
img1, img2 = img1 / 255.0, img2 / 255.0
spec1 = np.fft.fftshift(np.fft.fft2(img1))
spec2 = np.fft.fftshift(np.fft.fft2(img2))
# 分别计算振幅谱和相位谱
amp1, phase1 = np.abs(spec1), np.angle(spec1)
amp2, phase2 = np.abs(spec2), np.angle(spec2)

# 分别生成两个新频谱：A振幅+B相位，以及 B振幅+A相位
amp1phase2 = np.zeros_like(spec1)
amp1phase2.real = amp1 * np.cos(phase2)
amp1phase2.imag = amp1 * np.sin(phase2)
amp2phase1 = np.zeros_like(spec2)
amp2phase1.real = amp2 * np.cos(phase1)
amp2phase1.imag = amp2 * np.sin(phase1)
# 反变换后保存生成的图像
img_amp1phase2 = np.fft.ifft2(np.fft.fftshift(amp1phase2)).real
cv2.imwrite('./results/fft_test/mix_amp1phase2.png', img_amp1phase2 * 255)
img_amp2phase1 = np.fft.ifft2(np.fft.fftshift(amp2phase1)).real
cv2.imwrite('./results/fft_test/mix_amp2phase1.png', img_amp2phase1 * 255)
