import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_salt_pepper_noise(img,
                          salt_ratio=0.01,
                          pepper_ratio=0.01):
    speckle_noisy = img.copy()
    h, w = speckle_noisy.shape[:2]
    num_salt = np.ceil(img.size * salt_ratio)
    salt_rid = np.random.choice(h, int(num_salt))
    salt_cid = np.random.choice(w, int(num_salt))
    speckle_noisy[salt_rid, salt_cid, ...] = 255
    num_pepper = np.ceil(img.size * pepper_ratio)
    pepper_rid = np.random.choice(h, int(num_pepper))
    pepper_cid = np.random.choice(w, int(num_pepper))
    speckle_noisy[pepper_rid, pepper_cid, ...] = 0
    return speckle_noisy


if __name__ == "__main__":

    img_path = '../datasets/srdata/Set12/05.png'
    img = cv2.imread(img_path)[:,:,0]

    # 设置不同参数
    salt_pepper_ratio = [
        (0.01, 0.01),
        (0.05, 0.01),
        (0.01, 0.05)]

    N = len(salt_pepper_ratio)
    fig = plt.figure(figsize=(10, 3))
    for sid, s in enumerate(salt_pepper_ratio):
        noisy = add_salt_pepper_noise(img,
                                    salt_ratio=s[0],
                                    pepper_ratio=s[1])
        fig.add_subplot(1, N, sid + 1)
        plt.imshow(noisy, cmap='gray')
        plt.axis('off')
        plt.title(f'salt {s[0]}, pepper {s[1]}')
    plt.savefig(f'results/noise_simu/salt_pepper.png')
    plt.close()

