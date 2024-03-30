import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_rect_mask(img):
    if img.ndim == 3:
        img = img[:,:,::-1]
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    roi = cv2.selectROI(img)
    x, y, w, h = roi
    x, y, w, h = int(x), int(y), int(w), int(h)
    mask[y: y + h, x: x + w] = 1.0
    crop = img[y: y + h, x: x + w]
    return mask, roi, crop

if __name__ == "__main__":
    img1 = cv2.imread("data_samples/lena256rgb.png")[:,:,::-1]
    mask, roi, crop = get_rect_mask(img1)
    print("ROI selected is : ", roi)
    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.title("generated rect mask")
    plt.figure()
    plt.imshow(crop)
    plt.title("cropped ROI image")
    plt.show()
