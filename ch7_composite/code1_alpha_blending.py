import cv2
import numpy as np

def blur_mask(mask, sigma=10):
    alpha_mask = cv2.GaussianBlur(mask, ksize=[0, 0], sigmaX=sigma)
    return alpha_mask

def alpha_blend(img1, img2, alpha_mask):
    if img1.ndim == 3:
        alpha_mask = np.expand_dims(alpha_mask, axis=2)
    blend = img1 * alpha_mask + img2 * (1 - alpha_mask)
    return blend

if __name__ == "__main__":
    fg = cv2.imread("../datasets/composite/plane/source.jpg")
    bg = cv2.imread("../datasets/composite/plane/target.jpg")
    mask = cv2.imread("../datasets/composite/plane/mask.jpg")[:,:,0]
    mask = mask / 255.0
    alpha_mask = blur_mask(mask, sigma=10)
    copy_paste = alpha_blend(fg, bg, mask)
    alpha_blend = alpha_blend(fg, bg, alpha_mask)
    cv2.imwrite('./results/copy_paste.png', copy_paste)
    cv2.imwrite('./results/alpha_blend.png', alpha_blend)
