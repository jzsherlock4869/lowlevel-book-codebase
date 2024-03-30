import os
import cv2
import numpy as np

os.makedirs('results/transform', exist_ok=True)
img_path = '../datasets/samples/cat1.png'
image = cv2.imread(img_path)

def set_affine_matrix(op_types=['scale'], params=[(1, 1)]):
    assert len(op_types) == len(params)
    tot_affine_mat = np.eye(3, 3, dtype=np.float32)
    op_mat_ls = list()
    for op_type, param in zip(op_types, params):
        if op_type == 'scale':
            scale_x, scale_y = param
            affine_mat = np.array(
                [[scale_x, 0, 0],
                [0, scale_y, 0],
                [0, 0, 1]], dtype=np.float32)
        if op_type == 'translation':
            trans_x, trans_y = param
            affine_mat = np.array(
                [[1, 0, trans_x],
                [0, 1, trans_y],
                [0, 0, 1]], dtype=np.float32)
        if op_type == 'rot':
            theta = param
            cost, sint = np.cos(theta), np.sin(theta)
            affine_mat = np.array(
                [[cost, sint, 0],
                [-sint, cost, 0],
                [0, 0, 1]], dtype=np.float32)
        if op_type == 'shear':
            phi_x, phi_y = param
            tant_x, tant_y = np.tan(phi_x), np.tan(phi_y)
            affine_mat = np.array(
                [[1, tant_x, 0],
                [tant_y, 1, 0],
                [0, 0, 1]], dtype=np.float32)
        op_mat_ls.append(affine_mat[:2, :])
        tot_affine_mat = affine_mat.dot(tot_affine_mat)
    return tot_affine_mat[:2, :], op_mat_ls


test_types = ['translation', 'scale', 'rot', 'shear']
test_params = [(15, 2), (1.1, 0.8), 0.15, (-0.3, 0.2)]

affine_mat, op_mat_ls = set_affine_matrix(test_types, test_params)
print(affine_mat)

height, width = image.shape[:2]
affined = cv2.warpAffine(image, affine_mat, (width, height), flags=cv2.INTER_LINEAR)
cv2.imwrite('results/transform/affined_out.png', affined)

inter_img = image
for idx in range(len(test_types)):
    print(test_types[idx])
    print(op_mat_ls[idx])
    inter_img = cv2.warpAffine(inter_img, op_mat_ls[idx], (width, height), flags=cv2.INTER_LINEAR)
    cv2.imwrite('results/transform/affined_out_step{}_{}.png'\
                .format(idx, test_types[idx]), inter_img)

