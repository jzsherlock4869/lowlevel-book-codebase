import os
import cv2
import numpy as np

os.makedirs('results/transform', exist_ok=True)
img_path = '../datasets/samples/cat2.png'
image = cv2.imread(img_path)

def transform_image(img, points1, points2):
    height, width = img.shape[0], img.shape[1]
    matrix = cv2.getPerspectiveTransform(points1, points2)
    res = cv2.warpPerspective(img, matrix, (width, height))
    return res, matrix

points1 = np.float32([[0, 0], [0, 255], [255, 0], [255, 255]])
points2 = np.float32([[10, 80], [80, 180], [200, 20], [240, 240]])
res, matrix = transform_image(image, points1, points2)
print("perspective matrix : \n", matrix)
cv2.imwrite('results/transform/perspective.png', res)

