import numpy as np
from collections import deque
import cv2 as cv
from tqdm import tqdm

# from skimage.feature import peak_local_max
from sklearn.cluster import DBSCAN
from scipy.ndimage import label

from framework.blockbase_pipeline import BlockBaseFrameWork
from  framework.orientation_estimation import local_multipeak, banning_peak
# from blockattribute import find_Attribute_multi

from framework.utils.concave_hull import concave_hull
from framework.utils.check_angle_rad import check_angle_rad
# ---- Visualization Hook ----
from framework.utils.plot_all import plot_all
# ----
class BlockGroup:
    def __init__(self, row_index_list=None, col_index_list=None, block_attribute_list=None, fromActiMap=False, activation_map=None):
        if fromActiMap:
            row_index_list = []
            col_index_list = []
            for i in range(activation_map.shape[0]):
                for j in range(activation_map.shape[1]):
                    if activation_map[i][j]==1:
                        row_index_list.append(i)
                        col_index_list.append(j)
        self.rw_idx_lst = row_index_list
        self.cl_idx_lst = col_index_list
        self.blk_attrib_lst = block_attribute_list
        self.max_pos = None
        if not row_index_list is None:
            self.__gen_id()
        else:
            self.group_id = {}
    def __gen_id(self):
        group_id = {}
        blk_attrib_lst = self.blk_attrib_lst
        attrib_idx = 0
        for idx in range(len(self.rw_idx_lst)):
            r_idx = self.rw_idx_lst[idx]
            c_idx = self.cl_idx_lst[idx]
            if blk_attrib_lst is None:
                group_id[(r_idx, c_idx)] = None
            else:
                group_id[(r_idx, c_idx)] = blk_attrib_lst[attrib_idx]
            attrib_idx += 1
        self.group_id = group_id
    def add_member(self, row_index, col_index, attribute=None):
        self.group_id[(row_index, col_index)] = attribute
    def remove_member(self, row_index, col_index):
        del self.group_id[(row_index, col_index)]
    def generate_activation_map(self, max_row, max_col):
        atv_map = np.zeros((max_row, max_col), dtype=bool)
        for pos in self.group_id:
            row_idx, col_idx = pos
            atv_map[row_idx, col_idx] = True
        return atv_map
    def get_member_attribute(self, row_index, col_index):
        return self.group_id[(row_index, col_index)]
    def get_member_list(self):
        return [id for id in self.group_id]
    def get_concave_hull(self):
        points = self.get_member_list()
        hull_points = concave_hull(points)
        if hull_points is None:
            return points
        return hull_points
    def set_max_pos(self, max_pos:tuple, insert=False):
        if max_pos in self.group_id:
            self.max_pos = max_pos
            return True
        if insert:
            self.add_member(max_pos[0], max_pos[1])
            return True
        else:
            return False
    def set_center_pos(self, center_pos:tuple):
        self.center_pos = center_pos         

def map_clustering_n_iterative(vector_map, falloff_threshold=0.9, kernel=None, connectivity=8, custom_peak_pos=None):
    "Iterative n-connectivity kernel block clustering"
    if kernel is None:
        if connectivity==8:
            kernel = np.array([[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]], dtype=float)
        elif connectivity==4:
            kernel = np.array([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]], dtype=float)
    map_row, map_col = vector_map.shape
    k_row, k_col = kernel.shape
    if k_row%2==0 or k_col%2==0:
        raise ValueError("Kernel must have odd shape for it to have center!")
    ctr_k_row, ctr_k_col = k_row//2, k_col//2
    if custom_peak_pos is None:
        peak_pos = local_multipeak(vector_map, radius_ban=2, max_peak_count=10)
    else:
        peak_pos = custom_peak_pos # put your custom peak pos here
    if peak_pos is None:
        return None, None
    BG_list = [] # block group 
    display_map_list = []
    for pos in tqdm(peak_pos):
        # ---- Visualization tag
        # display_map = np.zeros_like(vector_map)
        # ------------------
        r_idx, c_idx = pos
        # Initiate row and column list
        BG = BlockGroup([r_idx], [c_idx])
        BG.set_max_pos((r_idx, c_idx)) # set max pos
        # iteratively find row and col index until not fit in falloff_threshold
        index = 0
        delta = BG.get_member_list()
        while True:
            change = False
            for ctr_pos in delta:
                temp_kernel = kernel.copy()
                # center of kernel in map coordinate
                ctr_row, ctr_col = ctr_pos
                # ---- Visualization tag
                # display_map[ctr_row, ctr_col] = 1
                # ----------------------
                start_row, stop_row =  ctr_row-ctr_k_row, ctr_row+ctr_k_row
                start_col, stop_col = ctr_col-ctr_k_col, ctr_col+ctr_k_col
                if start_row < 0:
                    rl_start_row = 0
                    temp_kernel = temp_kernel[(rl_start_row-start_row):, :]
                else:
                    rl_start_row = start_row
                if stop_row >= map_row:
                    rl_stop_row = map_row-1
                    temp_kernel = temp_kernel[:-(stop_row-rl_stop_row), :]
                else:
                    rl_stop_row = stop_row
                if start_col < 0:
                    rl_start_col = 0
                    temp_kernel = temp_kernel[:, (rl_start_col-start_col):]
                else:
                    rl_start_col = start_col
                if stop_col >= map_col:
                    rl_stop_col = map_col-1
                    temp_kernel = temp_kernel[:, :-(stop_col-rl_stop_col)]
                else:
                    rl_stop_col = stop_col
                
                output = temp_kernel*vector_map[rl_start_row:rl_stop_row+1, rl_start_col:rl_stop_col+1]
                
                thres = falloff_threshold*vector_map[ctr_row, ctr_col]
                    
                _, output = cv.threshold(output, thres, 1, cv.THRESH_BINARY)

                rel_ctr_row = ctr_row - rl_start_row
                rel_ctr_col = ctr_col - rl_start_col
                output[rel_ctr_row, rel_ctr_col] = 0
                rel_pos = np.where(output>0)
                if len(rel_pos[0])==0:
                    # Do nothing, skip to next loop
                    continue
                for i in range(len(rel_pos[0])):
                    rl_row, rl_col = rl_start_row+rel_pos[0][i], rl_start_col+rel_pos[1][i]
                    if not (rl_row, rl_col) in BG.get_member_list():
                        BG.add_member(rl_row, rl_col)
                        if not change:
                            change = True
            temp = BG.get_member_list()
            set_delta = set(delta)
            delta = [item for item in temp if item not in set_delta]
            #--------- Visualization tag
            # plot_all([vector_map, display_map]) # Uncomment ts
            #------------------------
            if not change:
                # There is no change
                break
        BG_list.append(BG)
        # display_map_list.append(display_map)
    return BG_list

def map_clustering_watershed(vector_map, connectivity=8, merge_threshold=0.5):
    """Clusters pixels hierarchically from peaks downward (monotonically non-increasing),

    excluding 0-intensity background, with per-step visualization.

    :param vector_map: 2D numpy array (grayscale image)
    :param connectivity: 4 or 8 for neighbor checking
    :param merge_threshold: Fraction of overlap required to merge two
    clusters
    :return: List of clusters, each containing a list of (y, x) tuples
    """
    rows, cols = vector_map.shape

    if connectivity == 4:
        # 4 connectivity
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        # 8 Connectivity
        offsets = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]

    def get_neighbors(y, x):
        neighbors = []
        for dy, dx in offsets:
            ny, nx = y + dy, x + dx
            if 0 <= ny < rows and 0 <= nx < cols:
                neighbors.append((ny, nx))
        return neighbors

    # 1. Filter out zero-intensity background and sort coordinates descending by intensity
    y_coords, x_coords = np.nonzero(vector_map)
    intensities = vector_map[y_coords, x_coords]

    sort_idx = np.argsort(intensities)[::-1]
    sorted_pixels = list(zip(y_coords[sort_idx], x_coords[sort_idx]))

    visited_by_any = set()
    clusters = []

    # 2 & 3. Identify Peaks and Perform Downward Region Growing
    for peak_y, peak_x in sorted_pixels:
        if (peak_y, peak_x) in visited_by_any:
            continue

        current_cluster_set = set([(peak_y, peak_x)])
        queue = deque([(peak_y, peak_x)])

        while queue:
            curr_y, curr_x = queue.popleft()
            curr_intensity = vector_map[curr_y, curr_x]

            # # --- VISUALIZATION HOOK ---
            # vis_array = np.zeros_like(vector_map)
            # for cy, cx in current_cluster_set:
            #     vis_array[cy, cx] = vector_map[cy, cx]

            # plot_all(vis_array)
            # # --------------------------

            for ny, nx in get_neighbors(curr_y, curr_x):
                n_intensity = vector_map[ny, nx]

                # Exclude background pixels and already included pixels
                if n_intensity == 0 or (ny, nx) in current_cluster_set:
                    continue

                # Monotonic rule: Neighbor intensity must be less than or equal to current pixel
                if n_intensity <= curr_intensity:
                    current_cluster_set.add((ny, nx))
                    visited_by_any.add((ny, nx))
                    queue.append((ny, nx))
        # print((peak_y, peak_x))
        # plot_all(vis_array)
        clusters.append([current_cluster_set, (peak_y, peak_x)])
        # Add Breaking Condition

    # 4. Merge Similar Clusters
    merged_clusters = []
    peak_pos_list = []
    for cluster, peak_pos in clusters:
        merged = False
        for i, existing_cluster in enumerate(merged_clusters):
            intersection = cluster.intersection(existing_cluster)

            min_size = min(len(cluster), len(existing_cluster))
            if len(intersection) / min_size > merge_threshold:
                merged_clusters[i] = existing_cluster.union(cluster)
                merged = True
                break

        if not merged:
            merged_clusters.append(cluster)
            peak_pos_list.append(peak_pos)
    # return [list(c) for c in merged_clusters]
    # return [BlockGroup([item[0] for item in list(c)], [item[1] for item in list(c)]) for c in merged_clusters]
    result = [BlockGroup(*zip(*c)) for c in merged_clusters]
    for i in range(len(result)):
        result[i].set_max_pos(peak_pos_list[i])
    return result
# Example
def map_clustering_tol(vector_map, tol=0.8, min_samp=2):
    # ---- Note: Very Slow
    dbscan = DBSCAN(eps=tol, min_samples=min_samp)
    labels = dbscan.fit_predict(vector_map)
    cluster_ids = [label for label in np.unique(labels) if label != -1]
    # masks = [(labels==c_id) for c_id in cluster_ids]
    # [BlockGroup(fromActiMap=True, activation_map=)]
    return [BlockGroup(fromActiMap=True, activation_map=(labels==c_id)) for c_id in cluster_ids]
def map_clustering_threshold(vector_map, parts=4):
    # ---- Don't Forget to add max cluster
    bins = np.linspace(0.0, 1.0, parts + 1) # into n parts
    lower = bins[:-1, None, None]
    upper = bins[1:, None, None]
    masks = (vector_map >= lower) & (vector_map < upper)
    return [BlockGroup(fromActiMap=True, activation_map=item) for item in masks]
def map_clustering_peak_threshold(vector_map, peak_pos, thresh=0.8):
    # Sort peaks in descending order based on their value in vector_map
    sorted_peaks = sorted(peak_pos, key=lambda pos: vector_map[tuple(pos)], reverse=True)
    
    # Precompute global image mean as the minimum threshold floor
    img_mean = np.mean(vector_map)
    
    clusters = []
    structure = np.ones((3,) * vector_map.ndim)
    
    for pos in sorted_peaks:
        pos_tuple = tuple(pos)
        peak_val = vector_map[pos_tuple]
        
        # Prevent cutoff from dropping below the image mean
        cutoff = max(thresh * peak_val, img_mean)
        
        # If the peak itself is below the image mean, it produces an empty mask
        if peak_val < img_mean:
            clusters.append(np.zeros_like(vector_map, dtype=np.uint8))
            continue

        binary_mask = vector_map >= cutoff
        
        # Extract connected component corresponding to this peak
        labeled_map, _ = label(binary_mask, structure=structure)
        target_label = labeled_map[pos_tuple]
        
        if target_label == 0:
            cluster_map = np.zeros_like(vector_map, dtype=np.uint8)
        else:
            cluster_map = (labeled_map == target_label).astype(np.uint8)
            
        clusters.append(cluster_map)
        
    return clusters
def map_clustering_kernel(vector_map, kernel=None):
    pass
if __name__ == "__main__":
    from utils.freqfilter import FreqFilter
    from utils.plot_all import plot_all
    ##---- Clustering Example
    size = 32
    radius1, radius2 = 3, 16
    filter = FreqFilter((size, size))
    BPF = filter.getBPF(radius1=radius1, radius2=radius2)
    BPF = BPF.astype(np.float32)
    BPF = cv.GaussianBlur(BPF, (3, 3), sigmaX=0)
    # plot_all(BPF)
    # dat_list = map_clustering_watershed(BPF)
    # mask = map_clustering_threshold(BPF)
    # mask = map_clustering_peak_threshold(BPF, )
    # print(mask)
    # plot_all([mask[i] for i in range(len(mask))])
    # ---- ---- ----
    
    ##---- 