import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from utils.select_roi import get_rect_mask
from collections import OrderedDict


def neighbor_coords(coord):
    i, j = coord
    neighbors = [
        (i - 1, j), (i + 1, j),
        (i, j - 1), (i, j + 1)
    ]
    return neighbors

def mask_to_coord(mask):
    h, w = mask.shape
    np_coords = np.nonzero(mask)
    omega_coords = OrderedDict()
    num_pix = len(np_coords[0])
    for idx in range(num_pix):
        coord = (np_coords[0][idx], np_coords[1][idx])
        omega_coords[coord] = idx
    edge_coords = OrderedDict()
    edge_pix_idx = 0
    for i, j in omega_coords:
        cur_coord = (i, j)
        for nb in neighbor_coords(cur_coord):
            if nb not in omega_coords:
                edge_coords[cur_coord] = edge_pix_idx
                edge_pix_idx += 1
                break
    return omega_coords, edge_coords

def construct_matrix(omega_coords):
    num_pt = len(omega_coords)
    coeff_mat = sparse.lil_matrix((num_pt, num_pt))
    for i in range(num_pt):
        coeff_mat[i, i] = 4
    for cur_coord in omega_coords:
        for nb in neighbor_coords(cur_coord):
            if nb in omega_coords:
                coeff_mat[omega_coords[cur_coord], omega_coords[nb]] = -1
    return coeff_mat

def calc_target_vec(fg, bg, omega_coords, edge_coords):
    num_pt = len(omega_coords)
    target_vec = np.zeros(num_pt)
    for idx, cur_coord in enumerate(omega_coords):
        div = fg[cur_coord[0], cur_coord[1]] * 4.0
        for nb in neighbor_coords(cur_coord):
            div = div - fg[nb[0], nb[1]]
        if cur_coord in edge_coords:
            for nb in neighbor_coords(cur_coord):
                if nb not in omega_coords:
                    div += bg[nb[0], nb[1]]
        target_vec[idx] = div
    return target_vec

def calc_target_vec_mix(fg, bg, omega_coords, edge_coords):
    num_pt = len(omega_coords)
    target_vec = np.zeros(num_pt)
    for idx, cur_coord in enumerate(omega_coords):
        center_fg = fg[cur_coord[0], cur_coord[1]]
        center_bg = bg[cur_coord[0], cur_coord[1]]
        div = 0
        for nb in neighbor_coords(cur_coord):
            grad_fg = center_fg * 1.0 - fg[nb[0], nb[1]]
            grad_bg = center_bg * 1.0 - bg[nb[0], nb[1]]
            if abs(grad_fg) > abs(grad_bg):
                div += grad_fg
            else:
                div += grad_bg
        if cur_coord in edge_coords:
            for nb in neighbor_coords(cur_coord):
                if nb not in omega_coords:
                    div += bg[nb[0], nb[1]]
        target_vec[idx] = div
    return target_vec

def solve_poisson_eq(coeff_mat, target_vec):
    res = spsolve(coeff_mat, target_vec)
    res = np.clip(res, a_min=0, a_max=255)
    return res

def paste_result(bg, value_vec, omega_coords):
    bg_out = bg.copy()
    for idx, cur_coord in enumerate(omega_coords):
        bg_out[cur_coord[0], cur_coord[1]] = value_vec[idx]
    return bg_out

def poisson_blend(fg, bg, mask, blend_type="mix"):
    assert fg.ndim == bg.ndim, "need the same ndim for FG/BG"
    if fg.ndim == 2:
        fg = np.expand_dims(fg, axis=2)
        bg = np.expand_dims(bg, axis=2)
    mask[:, [0, -1]] = 0
    mask[[0, -1], :] = 0
    omega_coords, edge_coords = mask_to_coord(mask)
    coeff_mat = construct_matrix(omega_coords)
    output = np.zeros_like(bg)
    for ch_idx in range(fg.shape[-1]):
        cur_fg = fg[:, :, ch_idx]
        cur_bg = bg[:, :, ch_idx]
        if blend_type == "import":
            target_vec = calc_target_vec(cur_fg, cur_bg, omega_coords, edge_coords)
        elif blend_type == "mix":
            target_vec = calc_target_vec_mix(cur_fg, cur_bg, omega_coords, edge_coords)
        else:
            raise NotImplementedError\
                (f"blend_type should be import | mix, {blend_type} unsupport.")
        new_fg = solve_poisson_eq(coeff_mat, target_vec)
        cur_fused = paste_result(cur_bg, new_fg, omega_coords)
        output[:, :, ch_idx] = cur_fused
    return output


if __name__ == "__main__":
    fg = cv2.imread("../datasets/composite/plane/source.jpg")
    bg = cv2.imread("../datasets/composite/plane/target.jpg")
    mask = cv2.imread("../datasets/composite/plane/mask.jpg")[:,:,0]
    # 两种不同的泊松融合
    fused = poisson_blend(fg, bg, mask, blend_type="import")
    fused_mix = poisson_blend(fg, bg, mask, blend_type="mix")
    cv2.imwrite('./results/poisson_out.png', fused)
    cv2.imwrite('./results/poisson_out_mix.png', fused_mix)
