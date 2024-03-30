import cv2
import numpy as np
import matplotlib.pyplot as plt

def estimate_trans(I, A, tmin=0.4):
    A = np.expand_dims(A, axis=[1]) # [3, 1]
    norm_A = np.linalg.norm(A)
    IA = np.dot(I, A) / norm_A # [N, 1]
    norm_I2 = np.linalg.norm(I, axis=1, keepdims=True) ** 2
    IR = norm_I2 - IA ** 2
    IR = np.sqrt(IR)
    h = (norm_A - IA) / (IR + 1e-8)
    cov_IA = np.cov(IA.flatten(), h.flatten())
    cov_IR = np.cov(IR.flatten(), h.flatten())
    eta = (cov_IA / cov_IR)[1, 0]
    t_map = 1 - (IA - eta * IR) / norm_A
    t_map = (t_map - t_map.min()) / (t_map.max() - t_map.min())
    t_map = np.clip(t_map, a_min=tmin, a_max=1.0)
    return t_map

def dehaze_fattal(img, atmo, tmin=0.3):
    H, W, C = img.shape
    I = img.reshape((H * W, C))
    t_map = estimate_trans(I, atmo, tmin)
    J = (I - (1 - t_map) * np.expand_dims(atmo, axis=[0])) / (t_map + 1e-16)
    dehazed = J.reshape((H, W, C))
    t_map = t_map.reshape((H, W))
    return dehazed, t_map


if __name__ == "__main__":

    img_path = "../datasets/hazy/house-input.bmp"
    img = cv2.imread(img_path)[:,:,::-1] / 255.0
    atmo = np.array([210., 217., 223.]) / 255.0

    out, tmap = dehaze_fattal(img, atmo)
    out = np.clip(out, 0, 1)

    plt.figure()
    plt.imshow(img)
    plt.title('hazy input')
    plt.xticks([]), plt.yticks([])
    
    plt.figure()
    plt.imshow(out)
    plt.title('fattal dehaze output')
    plt.xticks([]), plt.yticks([])
    
    plt.figure()
    plt.imshow(tmap, cmap='gray')
    plt.title('transmission map')
    plt.xticks([]), plt.yticks([])
    plt.show()