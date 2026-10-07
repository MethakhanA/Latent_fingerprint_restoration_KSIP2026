import os
from glob import glob

from tqdm import tqdm
import numpy as np
import cv2 as cv

import matplotlib.pyplot as plt

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

from framework.utils.TV import TV_preprocessing

def simple_func(block_img):
    return block_img
if __name__ == "__main__":
    out_path = r"D:\work\image_processing\Latent_fingerprint\segment\data\data_tv"
    # out_path = r"D:\work\image_processing\Latent_fingerprint\segment\data\data_b"
    # ---- Define Params ----
    o_block_size = 64
    no_block_size = 16
    bp_r = (3, 16)
    gss_f_size = 3
    # ---- ---- ----
    for file in tqdm(glob(os.path.join(out_path, '*'))):
        file = r"D:\work\image_processing\Latent_fingerprint\segment\data\data_tv\Good_Quality.png"
        # file = r"C:\work\image_processing\latent_fingerprint\automatic_segment\data\data_b\00002310_2A_X_L01_BP_S05_500PPI_8BPC_1CH_LP01-1_1.png"
        img = cv.imread(file, 0) # This is a Texture TV image.
        # ---- Run BlockBase PipeLine ----
        BBF = BlockBaseFrameWork(img, overlap_block_size=o_block_size, nonoverlap_block_size=no_block_size, zeromean=True, window_func='Gaussian', blur_edge=True)
        row_map_index, col_map_index = BBF.row_map_block_index_list, BBF.col_map_block_index_list
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
        ks_map = ks_map[block_pad[0]:-block_pad[1], block_pad[2]:-block_pad[3]]
        pad_img = pad_img[BBF.fp_pad[0]:-BBF.fp_pad[1], BBF.fp_pad[2]:-BBF.fp_pad[3]]
        
        # plot_all([pad_img, ks_map], cmap=['gray', 'hot'])

        # # /// C1: ---- Create Activation map from watershed clustering
        # bg_list = map_clustering_watershed(ks_map)
        # max_pos_list = [item.max_pos for item in bg_list] # each item is a Block Group
        # mp_grp = BlockGroup([itm1[0] for itm1 in max_pos_list], [itm2[1] for itm2 in max_pos_list])
        # # ---- ---- ----
        
        # /// C2: ---- Create Activation map from localmultipeak and banningpeak
        max_pos_list = local_multipeak(ks_map, mean_radius_ban=None, radius_ban=2, max_peak_count=10)
        # max_pos_list = banning_peak(ks_map, mean_radius_ban=None)
        row_mp_list, col_mp_list = [item[0] for item in max_pos_list], [item[1] for item in max_pos_list]
        mp_grp = BlockGroup(row_mp_list, col_mp_list)
        # ---- ---- ----
        
        # ---- Create Activation Map
        activation_map = mp_grp.generate_activation_map(ks_map.shape[0], ks_map.shape[1])
        activation_map = exclude_boundary(activation_map)        
        # plot_all([pad_img, ks_map, activation_map], cmap=['gray', 'hot', 'gray'])

        # ---- Create Attribute map ----
        orientation_map = np.empty((len(row_map_index), len(col_map_index)), dtype=np.ndarray)
        orientation_map = BBF.apply_func_map(magnitude, find_orientation, output_is_img=False, output_vector=orientation_map)
        # ---- ---- ----
        
        o_map = orientation_map[block_pad[0]:-block_pad[1], block_pad[2]:-block_pad[3]]
        # print(o_map.shape, activation_map.shape)
        # ---- ---- ----

        # ---- Find Crossing Field ----
        result = find_orientation_crossings(o_map, activation_map=activation_map)
        visualize_crossings(o_map, result, background_image=pad_img, cmap='gray')
        # ---- ---- ----
        
        # ---- Assign to BlockGroup
        # Result is in the form of ((crossing), support1, support2, ...)
        # print(result)
        crossing_list = []
        for crossing in result:
            crossing = list(crossing)
            # -- Unpack Variable
            crossing_point = crossing[0]
            support_point = crossing[1:]
            # -- Assign to BlockGroup
            BG = BlockGroup()
            BG.set_center_pos(crossing_point)
            for support in support_point:
                # -- Find Cluster
                cluster_list = map_clustering_peak_threshold(ks_map, [support])
                # plot_all(cluster_list[0])
                BG.add_member(support[0], support[1], BlockGroup(fromActiMap=True, activation_map=cluster_list[0]))
            crossing_list.append(BG)
        # ---- ---- ----
        # magnitude = magnitude[BBF.fp_pad[0]:-BBF.fp_pad[1], BBF.fp_pad[2]:-BBF.fp_pad[3]]
        # ---- Reconstruct Each Picture
        for crossing_group in crossing_list:
            # --- This is Each Picture
            member_list = crossing_group.get_member_list()
            o_cluster_acti_map = np.zeros_like(ks_map, dtype=np.uint8)
            for member in member_list:
                m_row, m_col = member
                cluster_group = crossing_group.get_member_attribute(m_row, m_col)
                cluster_acti_map = cluster_group.generate_activation_map(ks_map.shape[0], ks_map.shape[1])
                cluster_acti_map = cluster_acti_map.astype(np.uint8)
                o_cluster_acti_map = np.bitwise_or(o_cluster_acti_map, cluster_acti_map)
            plot_all(o_cluster_acti_map)
            # -- Apply to each func
            o_cluster_acti_map = o_cluster_acti_map.astype(bool)
            # pad the activation map first
            o_cluster_acti_map = np.pad(
                o_cluster_acti_map, 
                pad_width=(
                    (block_pad[0], block_pad[1]),  # Pad Top, Bottom (Axis 0)
                    (block_pad[2], block_pad[3])   # Pad Left, Right (Axis 1)
                ), 
                mode='constant', 
                constant_values=0
            )
            recon_map = BBF.apply_func_map(magnitude, simple_func, activation_map=o_cluster_acti_map)
            # plot_all(recon_map, cmap='hot')
            BBF.setMagnitude(recon_map)
            BBF.istft()
            output = BBF.get_output_img()
            plot_all([img, output])
        # ---- ---- ----