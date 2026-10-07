import os
from glob import glob
import math

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from skimage.feature import peak_local_max
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.ndimage import maximum_filter

from framework.blockbase_pipeline import BlockBaseFrameWork
from framework.utils.plot_all import plot_all
from framework.utils.freqfilter import FreqFilter

# Select 5 Top Peak
# vector -> Magnitude, Orientation (0-2pi), frequency (distance), harmonic (Peak at 2*frequency)

def banning_peak(block_img, radius_ban=3, max_peak_count=5, mean_radius_ban=(3, 16)):
    ''' - Finding 5 First peak by searching from maxima -> ban discovered peak -> find new peak
        - Add mean prevention thresholding.
    '''
    row, col = block_img.shape
    y_indices, x_indices = np.ogrid[:row, :col] # create coordinate grid
    
    peak_loc = []
    x_center, y_center = col//2, row//2
    if mean_radius_ban is not None:
        block_img = ban_bandpass(block_img, mean_radius_ban[0], mean_radius_ban[1])
    for i in range(max_peak_count):
        peak_val = np.max(block_img)
        y_center, x_center = np.where(block_img==peak_val)
        for j in range(len(y_center)):
            cy = y_center[j]
            cx = x_center[j]
            peak_loc.append([int(cy), int(cx)]) # Same format as peak_local_max from skimage.feature
        # Ban in radius
        for index in range(len(x_center)):
            ban_circular(block_img, x_center[index], y_center[index], radius_ban, (y_indices, x_indices))
    return peak_loc

def local_multipeak(block_img, radius_ban=3, max_peak_count=5, mean_radius_ban=(3, 16)):
    if mean_radius_ban is not None:
        block_img = ban_bandpass(block_img, mean_radius_ban[0], mean_radius_ban[1])
    peak_pos = peak_local_max(block_img, radius_ban, num_peaks=max_peak_count, exclude_border=True)
    if len(peak_pos)==0:
        return None
    if len(peak_pos) > 0:
        # --- Sort Peak
        peak_vals = block_img[tuple(peak_pos.T)]
        sorted_indices = np.argsort(peak_vals)[::-1]
        peak_pos = peak_pos[sorted_indices]
        peak_pos = [tuple(p) for p in peak_pos]
    return peak_pos

# ---- Hierarchical Clustering
def find_peaks_adaptive_hierarchical(image, max_distance, neighborhood_size=3, noise_floor=None):
    """
    Finds peak locations using local maximum candidate detection followed by 
    hierarchical agglomerative clustering. Works across varied intensity scales.

    Parameters:
        image (np.ndarray): 2D numpy array.
        max_distance (float): Spatial distance threshold to merge adjacent local peaks.
        neighborhood_size (int): Size of the local search window (3x3 default).
        noise_floor (float or None): Optional minimum absolute value to ignore 
                                     flat/zero background noise.
    """
    # 1. Identify all pixels that are strictly local maxima relative to their neighbors
    local_max_mask = (image == maximum_filter(image, size=neighborhood_size))
    
    # 2. Filter out flat zero/background regions
    if noise_floor is not None:
        local_max_mask &= (image > noise_floor)
    else:
        # Ignore completely flat areas (where max == min locally)
        local_max_mask &= (image > image.min())

    rows, cols = np.where(local_max_mask)
    if len(rows) == 0:
        return []

    coords = np.column_stack((rows, cols))
    if len(coords) == 1:
        return [(int(rows[0]), int(cols[0]))]

    # 3. Group nearby local maxima using hierarchical clustering
    Z = linkage(coords, method='complete', metric='euclidean')
    cluster_labels = fcluster(Z, criterion='distance', t=max_distance)

    # 4. Resolve each cluster to its highest local point
    peaks = []
    for label in np.unique(cluster_labels):
        cluster_coords = coords[cluster_labels == label]
        intensities = image[cluster_coords[:, 0], cluster_coords[:, 1]]
        
        # Pick the point with highest intensity in this cluster
        max_idx = np.argmax(intensities)
        peaks.append((int(cluster_coords[max_idx, 0]), int(cluster_coords[max_idx, 1])))

    return peaks

def ban_circular(block_img, centerx, centery, radius_ban, grid=None):
    row, col = block_img.shape
    if grid is None:
        y_indices, x_indices = np.ogrid[:row, :col]
    else:
        y_indices, x_indices = grid
    # Ban radius
    distance_sq = ((x_indices-centerx)**2 + (y_indices-centery)**2)
    mask = distance_sq <= radius_ban**2
    block_img[mask]=0
    return block_img
def ban_bandpass(block_img, radius1, radius2):
    filter = FreqFilter(block_img.shape)
    bandwidth = radius2-radius1
    bandcenter = radius1+(bandwidth/2)
    BPF = filter.getBPF(radius1=radius1, radius2=radius2)
    output = block_img*BPF
    return output


# if __name__ == "__main__":
#     path = r"D:\work\image_processing\Latent_fingerprint\segment\data"
#     for file in glob(os.path.join(path, '*')):
#         img = cv.imread(file, 0)
#         o_block_size = 64
#         no_block_size = 32
#         BBF = BlockBaseFrameWork(img, overlap_block_size=o_block_size, nonoverlap_block_size=no_block_size, zeromean=True, window_func='Gaussian', blur_edge=True)
        
#         row_map_index_list, col_map_index_list = BBF.row_map_block_index_list, BBF.col_map_block_index_list
        
#         BBF.stft()
#         magnitude = BBF.getMagnitude()
#         img = BBF.get_img()
#         plot_all(img)
        # i = 9
        # j = 7
        # block_img = magnitude[row_map_index_list[i]:row_map_index_list[i]+o_block_size, col_map_index_list[j]:col_map_index_list[j]+o_block_size]
        # dp_loc = banning_peak(block_img)
        # plm_loc = peak_local_max(block_img, 3, num_peaks=10)
        # plot_all(block_img, cmap='hot')
        # break
        
        # orient_field = BBF.apply_func_map(magnitude, orientation_estimation)
        # plt.imshow(orient_field, 'hot')
        # plt.show()