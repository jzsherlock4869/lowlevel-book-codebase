import cv2
import numpy as np
import os

os.makedirs('./results/awb', exist_ok=True)

def gray_world_awb(img):
    img = img.astype(np.float32)
    avg = np.mean(img, axis=(0, 1))
    r_gain = avg[1] / avg[0]
    b_gain = avg[1] / avg[2]
    img[:,:,0] *= r_gain
    img[:,:,2] *= b_gain
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

if __name__ == "__main__":
    img_path = '../datasets/samples/awb_input.jpg'
    rgb_in = cv2.imread(img_path)[:,:,::-1]
    gray_world_out = gray_world_awb(rgb_in)
    cv2.imwrite('results/awb/gray_world_out.png', \
                gray_world_out[:,:,::-1])