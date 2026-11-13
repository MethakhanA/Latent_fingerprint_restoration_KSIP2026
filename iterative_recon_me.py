import os
from glob import glob

import math
import cv2 as cv
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
import copy

from skimage.morphology import convex_hull_image

from framework.blockattribute import BlockAttribute

def check_8_con(pos_list, ks_map):
    filter_pos_list = []
    for pos in pos_list:
        row, col = pos
        start_row, start_col = row-1, col-1
        buffer = False
        for i in range(3):
            for j in range(3):
                if ks_map[start_row+i, start_col+j]<=0:
                    buffer = True
                    break
            if buffer:
                break
        if not buffer:
            filter_pos_list.append(pos)
    return filter_pos_list

def angle_diff(a: float, b: float) -> float:
    """
    Computes the shortest unoriented angle difference between two angles in radians.
    
    The result will always be in the range [0, pi/2].
    Angles separated by pi (or odd multiples of pi) represent opposite directions
    on the same line, resulting in a difference of 0.
    """
    # Normalize the signed difference into (-pi, pi]
    diff = (a - b + math.pi) % (2 * math.pi) - math.pi
    
    # Wrap onto the line interval [0, pi/2]
    abs_diff = abs(diff)
    if abs_diff > math.pi / 2:
        abs_diff = math.pi - abs_diff
        
    return abs_diff

def visualize_3x3_orientation(orientations, row_map_index, col_map_index, magnitude, pos, block_size=64):
    """
    Visualizes a single 3x3 block image patch and overlays orientation lines 
    from a provided 1D list of 9 angles.
    """
    pos_r, pos_c = pos[0], pos[1]
    
    # Get the starting coordinate of the top-left block
    r_pos = row_map_index[pos_r - 1]
    c_pos = col_map_index[pos_c - 1]
    
    # Crop the 3x3 image patch
    patch_img = magnitude[r_pos:r_pos+(block_size*3), c_pos:c_pos+(block_size*3)]
    
    plt.figure(figsize=(8, 8))
    plt.imshow(patch_img, cmap='hot', origin='upper')
    plt.title(f"3x3 Patch Orientations (Center: [{pos_r}, {pos_c}])", fontsize=14)
    
    # Define line length to fit nicely inside each block (e.g., 80% of block size)
    line_length = block_size * 0.8 
    
    # Loop through the 3x3 grid
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            # Map the 2D grid offset to the 1D list index (0 to 8)
            idx = (dr + 1) * 3 + (dc + 1)
            
            # Skip if orientation list is too short or angle is None
            if idx >= len(orientations) or orientations[idx] is None:
                continue
                
            angle = orientations[idx]
            
            # Find the center pixel of the current sub-block inside the large patch
            center_y = (dr + 1) * block_size + (block_size / 2)
            center_x = (dc + 1) * block_size + (block_size / 2)
            
            # Calculate the line endpoints using sine and cosine
            dy = (line_length / 2) * np.sin(angle)
            dx = (line_length / 2) * np.cos(angle)
            
            y1, x1 = center_y - dy, center_x - dx
            y2, x2 = center_y + dy, center_x + dx
            
            plt.plot([x1, x2], [y1, y2], color='cyan', linewidth=2, marker='o', markersize=4)

    plt.axis('off')
    plt.tight_layout()
    plt.show()

def distance_p2p(pos1, pos2):
    y1, x1 = pos1
    y2, x2 = pos2
    d_y = np.abs(y1-y2)
    d_x = np.abs(x1-x2)
    d_r = np.sqrt(d_y**2+d_x**2)
    return d_r

def peak_2_peak_min_distance(couple_peak_pos:tuple, couple_peak_comp_pos:tuple):
    pos, inv_pos = couple_peak_pos
    comp_pos, comp_inv_pos = couple_peak_comp_pos
    d_r = distance_p2p(pos, comp_pos)
    d_r_temp = distance_p2p(pos, comp_inv_pos)
    if d_r_temp<d_r:
        d_r = d_r_temp
    return d_r








def fft_peak_mean_var(magnitude_block, lpf_radius=15, hpf_radius=3):
    filter = FreqFilter(magnitude_block.shape)
    LPF = filter.getLPF(15, f_type="Ideal")
    HPF = filter.getHPF(3, f_type="Ideal")
    banded = magnitude_block * LPF * HPF

    values = banded[banded != 0]
    if len(values)<4:
        return 0
    Q = np.var(values)*np.max(values)/np.mean(values)
    return Q

def generate_pmv_mask(pmv_map, limit_ratio=0.1):
    """
    Generates a binary mask from the PMV map using a hard limit threshold 
    and applies a convex hull to the result.
    """
    hard_limit_thresh = limit_ratio * np.max(pmv_map)
    base_mask = (pmv_map >= hard_limit_thresh)
    
    if np.any(base_mask):
        hard_limit_mask = convex_hull_image(base_mask)
    else:
        # Fallback if the map is entirely blank/zeros
        hard_limit_mask = np.zeros_like(base_mask, dtype=bool)
        
    return hard_limit_mask

def get8connect_index(pos:tuple):
    row, col = pos
    OFFSETS_8 = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1)
    ]
    return [(row+OFFSETS_8[idx][0], col+OFFSETS_8[idx][1]) for idx in range(8)]

if __name__ == "__main__":
    

    from framework.reflection_reconstruction import find_orientation_crossings

    from framework.blockbase_pipeline import BlockBaseFrameWork
    from framework.grouping_framework import BlockGroup
    from framework.orientation_estimation import ban_bandpass
    from framework.orientation_estimation import local_multipeak
    from framework.crossing_field import ban_bandpass_gaussian, exclude_boundary, visualize_crossings, find_orientation_crossings

    from framework.blockattribute import find_orientation, find_orientation_support_v

    from framework.grouping_framework import map_clustering_peak_threshold

    # from framework.utils.plot_all import plot_all
    from framework.utils.freqfilter import FreqFilter
    from framework.utils.kurtosis import fft_kurtosis
    from framework.utils.plot_all import plot_all
    from framework.utils.normalize_range import normalize_image

    # ---- Define Params ----
    o_block_size = 64
    no_block_size = 16
    bp_r = (3, 16)
    gss_f_size = 3
    # ---- ---- ----

    # out_path = r"C:\work\image_processing\latent_fingerprint\model\data\TV_img"
    out_path = r"C:\work\image_processing\latent_fingerprint\model\data\TV_img"
    save_dir = r"C:\work\image_processing\latent_fingerprint\model\data\output"
    for file in tqdm(glob(os.path.join(out_path, '*'))):
        # file = r"C:\work\image_processing\latent_fingerprint\model\data\data_tv\overlap_rim.png"
        # if os.path.exists(os.path.join(save_dir, os.path.basename(file))):
        #     continue


        img = cv.imread(file, 0) # This is a Texture TV image.
        img = cv.bitwise_not(img)
        # ---- Run BlockBase PipeLine ----
        BBF = BlockBaseFrameWork(img, overlap_block_size=o_block_size, nonoverlap_block_size=no_block_size, zeromean=True, window_func='Gaussian', blur_edge=False)
        row_map_index, col_map_index = BBF.row_map_block_index_list, BBF.col_map_block_index_list
        row_o_index, col_o_index = BBF.row_o_block_index_list, BBF.col_o_block_index_list
        row_no_index, col_no_index = BBF.row_no_block_index_list, BBF.col_no_block_index_list
        pad_img = BBF.get_img()
        BBF.stft()
        magnitude = BBF.getMagnitude().astype(np.float32)
        # plot_all(magnitude[row_map_index[index]:row_map_index[index]+o_block_size, col_map_index[index]:col_map_index[index]+o_block_size], cmap='hot', title_list=["Original Magnitude"])
        # ---- ---- ----
        
        # ---- Apply Gaussian Bandpass ----
        # Between frequency 3-15
        magnitude = BBF.apply_func_map(magnitude, ban_bandpass_gaussian, bp_r[0], bp_r[1], gss_f_size)
        
        # ---- Create PMV map
        pmv_map = np.zeros((len(row_map_index), len(col_map_index)))
        pmv_map = BBF.apply_func_map(magnitude, fft_peak_mean_var, output_is_img=False, output_vector=pmv_map)
        pmv_map = normalize_image(pmv_map)
        pmv_mask = generate_pmv_mask(pmv_map, limit_ratio=0.1)

        # ---- Crop Kurtosis and picture for better Visualization ----
        block_pad = np.array(BBF.fp_pad)//no_block_size
        # ks_map = ks_map[block_pad[0]:-block_pad[1], block_pad[2]:-block_pad[3]]
        # pad_img = pad_img[BBF.fp_pad[0]:-BBF.fp_pad[1], BBF.fp_pad[2]:-BBF.fp_pad[3]]
        
        # plot_all([pad_img, ks_map], cmap=['gray', 'hot'])

        # # /// C1: ---- Create Activation map from watershed clustering
        # bg_list = map_clustering_watershed(ks_map)
        # max_pos_list = [item.max_pos for item in bg_list] # each item is a Block Group
        # mp_grp = BlockGroup([itm1[0] for itm1 in max_pos_list], [itm2[1] for itm2 in max_pos_list])
        # # ---- ---- ----
        
        # /// C2: ---- Create Activation map from localmultipeak and banningpeak
        max_pos_list = local_multipeak(pmv_map, mean_radius_ban=None, radius_ban=2, max_peak_count=np.inf)
        # max_pos_list = banning_peak(ks_map, mean_radius_ban=None)
        # -- Check if 8 connectivity have 0 kurtosis
        max_pos_list = check_8_con(max_pos_list, pmv_map)
        row_mp_list, col_mp_list = [item[0] for item in max_pos_list], [item[1] for item in max_pos_list]
        mp_grp = BlockGroup(row_mp_list, col_mp_list)
        # ---- ---- ----
        
        # ---- Create Activation Map
        activation_map = mp_grp.generate_activation_map(pmv_map.shape[0], pmv_map.shape[1])
        activation_map = exclude_boundary(activation_map)
        activation_map = activation_map*pmv_mask        
        # plot_all([pad_img, pmv_map, activation_map], cmap=['gray', 'hot', 'gray'], title_list=["Original", "PMV Map", "Starting Peak Pos"])
        
        # ---- Create Attribute map ----
        orientation_map = np.empty((len(row_map_index), len(col_map_index)), dtype=np.ndarray)
        orientation_map = BBF.apply_func_map(magnitude, find_orientation, output_is_img=False, output_vector=orientation_map)
        # ---- ---- ----
        o_map = orientation_map
        # ---- ---- ----

        
        # # 1. Run your original function (unmodified)
        raw_crossings = find_orientation_crossings(o_map, d=2.0, dist_tol=1.5, cluster_tol=3.0, min_lines=2, max_crossings=100, inside_image_only=True, activation_map=activation_map)
        crossings = []
        for crossing in raw_crossings:
            cy, cx = crossing[0] # Continuous float coordinates
            icy, icx = int(round(cy)), int(round(cx))
            if 0 <= icy < pmv_mask.shape[0] and 0 <= icx < pmv_mask.shape[1]:
                if pmv_mask[icy, icx]: # Only keep if inside mask
                    crossings.append(crossing)


        visualize_crossings(o_map, crossings, background_image=pad_img, cmap='gray')
        
        # r_idx, c_idx = 14, 14
        # r_idx, c_idx = 2, 0
        
        # # Calculate offsets using map_index for magnitude and no_index for pad_img
        # m_row, m_col = row_map_index[r_idx], col_map_index[c_idx]
        # p_row, p_col = row_no_index[r_idx], col_no_index[c_idx]

        # # Extract regions (Assuming o_block_size is 64 and no_block_size is 16)
        # ori = find_orientation_support_v(magnitude[m_row:m_row+64, m_col:m_col+64])
        # print(ori)
        # plot_all([pad_img[p_row:p_row+16, p_col:p_col+16], magnitude[m_row:m_row+64, m_col:m_col+64]])
        # ---- ---- ----
        
        # ---- CreateOrientation Tensor Map
        attribute_map = np.empty((len(row_map_index), len(col_map_index)), dtype=np.ndarray)
        # each cell will contain couple_peak
        # You can get orientation from couple_peak[index]["orientation"]
        attribute_map = BBF.apply_func_map(magnitude, find_orientation_support_v, output_is_img=False, output_vector=attribute_map)
        # ---- ---- ----

        # Let us start with the non multi thread approach
        for crossing in crossings:
            # Crossing is in the form of
            # (crossingpoint), (index), (index), ...
            crossing_pos = crossing[0] # Exact crossing position
            source_indexes = crossing[1:] # All sources indexes (row, col)

            # Initiate Param
            member = [source_index for source_index in source_indexes]
            
            while True:
                pass