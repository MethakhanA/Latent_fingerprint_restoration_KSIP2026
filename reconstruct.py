import os
from glob import glob

'''
# ---- First, let us clarify the steps to preparation

# --
First we find the crossing point, find all the block that line crossed
then, we train the data from outer->inner
'''
import math
import cv2 as cv
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl




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

if __name__ == "__main__":
    

    from framework.reflection_reconstruction import find_orientation_crossings

    from framework.blockbase_pipeline import BlockBaseFrameWork
    from framework.grouping_framework import BlockGroup
    from framework.orientation_estimation import ban_bandpass
    from framework.orientation_estimation import banning_peak, local_multipeak
    from framework.crossing_field import ban_bandpass_gaussian, exclude_boundary, visualize_crossings, find_orientation_crossings

    from framework.blockattribute import find_orientation

    from framework.grouping_framework import map_clustering_peak_threshold

    from framework.utils.plot_all import plot_all
    from framework.utils.freqfilter import FreqFilter
    from framework.utils.kurtosis import fft_kurtosis
    from framework.utils.plot_all import plot_all
    out_path = r"C:\work\image_processing\latent_fingerprint\model\data\U_roll_500"

    # out_path = r"D:\work\image_processing\Latent_fingerprint\segment\data\data_b"
    # ---- Define Params ----
    o_block_size = 64
    no_block_size = 16
    bp_r = (3, 16)
    gss_f_size = 3
    # ---- ---- ----
    for file in tqdm(glob(os.path.join(out_path, '*'))):

        # file = r"D:\work\image_processing\Latent_fingerprint\segment\data\data_tv\Good_Quality.png"
        # file = r"C:\work\image_processing\latent_fingerprint\model\data\U_roll\00002302_U_500_roll_01.png"
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
        magnitude = BBF.apply_func_map(magnitude, ban_bandpass_gaussian, bp_r[0], bp_r[1], gss_f_size)
        # plot_all(magnitude[row_map_index[index]:row_map_index[index]+o_block_size, col_map_index[index]:col_map_index[index]+o_block_size], cmap='hot', title_list=["Gaussian Bandpass Magnitude"])
        
        # ---- Create kurtosis map ----
        ks_map = np.zeros((len(row_map_index), len(col_map_index)))
        ks_map = BBF.apply_func_map(magnitude, fft_kurtosis, output_is_img=False, output_vector=ks_map)
        # ---- ---- ----
        
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
        max_pos_list = local_multipeak(ks_map, mean_radius_ban=None, radius_ban=2, max_peak_count=np.inf)
        # max_pos_list = banning_peak(ks_map, mean_radius_ban=None)
        # -- Check if 8 connectivity have 0 kurtosis
        max_pos_list = check_8_con(max_pos_list, ks_map)
        row_mp_list, col_mp_list = [item[0] for item in max_pos_list], [item[1] for item in max_pos_list]
        mp_grp = BlockGroup(row_mp_list, col_mp_list)
        # ---- ---- ----
        
        # ---- Create Activation Map
        activation_map = mp_grp.generate_activation_map(ks_map.shape[0], ks_map.shape[1])
        activation_map = exclude_boundary(activation_map)        
        plot_all([pad_img, ks_map, activation_map], cmap=['gray', 'hot', 'gray'])

        # ---- Create Attribute map ----
        orientation_map = np.empty((len(row_map_index), len(col_map_index)), dtype=np.ndarray)
        orientation_map = BBF.apply_func_map(magnitude, find_orientation, output_is_img=False, output_vector=orientation_map)
        # ---- ---- ----
        o_map = orientation_map
        # o_map = orientation_map[block_pad[0]:-block_pad[1], block_pad[2]:-block_pad[3]]
        # magnitude = magnitude[block_pad[0]*64:-block_pad[1]*64, block_pad[2]*64:-block_pad[3]*64]
        # print(o_map.shape, activation_map.shape)
        # ---- ---- ----

        
        # print(f"Still here!\n")
        # # 1. Run your original function (unmodified)
        crossings = find_orientation_crossings(o_map, d=2.0, dist_tol=1.5, cluster_tol=3.0, min_lines=2, max_crossings=100, inside_image_only=True, activation_map=activation_map)
        visualize_crossings(o_map, crossings, background_image=pad_img, cmap='gray')
        
        for crossing in crossings:
            cross_pos = crossings[0]
            support_pos = crossings[1:]
            # ---- Each 8 connectivity check for double peak
            # As per current implementation,
            # -> crop to half image
            # -> use local multipeak
            # -> if double peak is already in the crop, Don't search
            