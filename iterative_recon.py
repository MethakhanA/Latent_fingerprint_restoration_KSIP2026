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

def fft_var(magnitude_block, lpf_radius=15, hpf_radius=3):
    filter = FreqFilter(magnitude_block.shape)
    LPF = filter.getLPF(15, f_type="Ideal")
    HPF = filter.getHPF(3, f_type="Ideal")
    banded = magnitude_block * LPF * HPF

    values = banded[banded != 0]
    return np.var(values)

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

import matplotlib.patches as patches
def _to_hw(size):
    """Normalize an int or (height, width) tuple into a (h, w) tuple."""
    if size is None:
        return None
    if isinstance(size, (int, np.integer)):
        return (int(size), int(size))
    return (int(size[0]), int(size[1]))


def plot_orientation_map(
    orientation_2d,
    background_image=None,
    overlap_block_size=None,
    non_overlap_block_size=None,
    directed=False,
    show_vectors=True,
    show_block_grid=False,
    overlay_alpha=0.45,
    missing_color=None,
    vector_color="black",
    padded_blocks="auto",
):
    """
    Plots a 2D orientation map in radians, optionally overlaid on a background image
    using overlapping and non-overlapping block dimensions.

    Parameters:
    - orientation_2d: 2D list or NumPy array of angles in radians (None = no orientation).
    - background_image: Optional 2D (grayscale) or 3D (RGB/RGBA) NumPy array.
    - overlap_block_size: Int or (h, w) tuple for the full analysis window size (W).
    - non_overlap_block_size: Int or (h, w) tuple for the core block / step size (S).
    - directed: False wraps angles to [0, pi); True wraps angles to [0, 2*pi).
    - show_vectors: If True, draws orientation line segments or arrows at block centers.
    - show_block_grid: If True, draws the non-overlapping tile grid and an example
                       overlapping window boundary.
    - overlay_alpha: Opacity (0 to 1) of the orientation heatmap when background_image is set.
    - missing_color: Color for None/NaN cells. Defaults to 'lightgray' without a background
                     image, or 'none' (transparent) when background_image is provided.
    - vector_color: Color of the orientation lines/arrows.
    - padded_blocks: 'auto', True, or False. Determines whether block (0, 0) is centered on
                     the non-overlapping tile (True, padded image) or the full overlapping
                     window (False, unpadded sliding window).
    """
    angles = np.array(orientation_2d, dtype=float)
    rows, cols = angles.shape

    period = 2 * np.pi if directed else np.pi
    angles = np.mod(angles, period)

    # Resolve block sizes (W = overlap window size, S = non-overlap step/core size)
    W = _to_hw(overlap_block_size)
    S = _to_hw(non_overlap_block_size)

    if S is None and W is not None:
        S = W
    elif W is None and S is not None:
        W = S
    elif W is None and S is None:
        if background_image is not None:
            img_h, img_w = background_image.shape[:2]
            S = (img_h / rows, img_w / cols)
        else:
            S = (1, 1)
        W = S

    S_h, S_w = S
    W_h, W_w = W

    # Determine center coordinates of each block in image pixel space
    if padded_blocks == "auto":
        if background_image is not None and (W_h != S_h or W_w != S_w):
            img_h, img_w = background_image.shape[:2]
            unpadded_err = abs(img_w - ((cols - 1) * S_w + W_w)) + abs(img_h - ((rows - 1) * S_h + W_h))
            padded_err = abs(img_w - (cols * S_w)) + abs(img_h - (rows * S_h))
            use_padded = padded_err < unpadded_err
        else:
            use_padded = True
    else:
        use_padded = bool(padded_blocks)

    # Pixel (0, 0) spans [-0.5, 0.5], so a block of size B starting at 0 is centered at (B - 1) / 2
    start_x = (S_w - 1) / 2.0 if use_padded else (W_w - 1) / 2.0
    start_y = (S_h - 1) / 2.0 if use_padded else (W_h - 1) / 2.0

    x_centers = start_x + np.arange(cols) * S_w
    y_centers = start_y + np.arange(rows) * S_h

    # Configure colormap and missing-value rendering
    if missing_color is None:
        missing_color = "none" if background_image is not None else "lightgray"

    cmap = plt.get_cmap("twilight").copy()
    if str(missing_color).lower() in ("none", "transparent"):
        cmap.set_bad(alpha=0.0)
    else:
        cmap.set_bad(color=missing_color)

    fig, ax = plt.subplots(figsize=(7, 6))

    # 1. Plot background image if provided
    if background_image is not None:
        bg_cmap = "gray" if background_image.ndim == 2 else None
        ax.imshow(background_image, cmap=bg_cmap, origin="upper")

    # 2. Plot orientation heatmap aligned to the non-overlapping block tiles
    extent = [
        x_centers[0] - S_w / 2.0,
        x_centers[-1] + S_w / 2.0,
        y_centers[-1] + S_h / 2.0,
        y_centers[0] - S_h / 2.0,
    ]
    alpha = overlay_alpha if background_image is not None else 1.0
    im = ax.imshow(
        angles,
        cmap=cmap,
        vmin=0,
        vmax=period,
        origin="upper",
        extent=extent,
        alpha=alpha,
    )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if directed:
        cbar.set_ticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        cbar.set_ticklabels(["0", "π/2", "π", "3π/2", "2π"])
    else:
        cbar.set_ticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
        cbar.set_ticklabels(["0", "π/4", "π/2", "3π/4", "π"])
    cbar.set_label("Orientation (radians)")

    # 3. Overlay orientation vectors in image data coordinates
    if show_vectors:
        X, Y = np.meshgrid(x_centers, y_centers)
        U = np.cos(angles)
        V = np.sin(angles)
        vec_len = 0.75 * min(S_w, S_h)

        quiver_kwargs = dict(
            angles="xy",
            scale_units="xy",
            scale=1.0 / vec_len,
            pivot="mid",
            color=vector_color,
        )
        if not directed:
            quiver_kwargs.update(headwidth=0, headlength=0, headaxislength=0)

        ax.quiver(X, Y, U, V, **quiver_kwargs)

    # 4. Optionally draw non-overlapping tile grid and an example overlapping block
    if show_block_grid:
        x_edges = np.append(x_centers - S_w / 2.0, x_centers[-1] + S_w / 2.0)
        y_edges = np.append(y_centers - S_h / 2.0, y_centers[-1] + S_h / 2.0)
        ax.vlines(x_edges, y_edges[0], y_edges[-1], colors="white", linestyles=":", linewidth=0.8, alpha=0.8)
        ax.hlines(y_edges, x_edges[0], x_edges[-1], colors="white", linestyles=":", linewidth=0.8, alpha=0.8)

        if W_w != S_w or W_h != S_h:
            # Highlight the first overlapping window (W_w x W_h) for visual reference
            rect = patches.Rectangle(
                (x_centers[0] - W_w / 2.0, y_centers[0] - W_h / 2.0),
                W_w,
                W_h,
                linewidth=1.5,
                edgecolor="yellow",
                facecolor="none",
                linestyle="--",
                label=f"Overlap window ({W_w}×{W_h})",
            )
            ax.add_patch(rect)
            ax.legend(loc="upper right", framealpha=0.7)

    ax.set_title("2D Orientation Map")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    plt.tight_layout()
    plt.show()


from numpy.lib.stride_tricks import sliding_window_view
def sobel_ignore_none(arr: np.ndarray, keep_none: bool = True):
    """
    Apply a 2D Sobel filter to a dtype=object array containing numbers and None.

    Parameters
    ----------
    arr : np.ndarray
        2D array (dtype=object) with numeric values and None.
    keep_none : bool
        If True, returns dtype=object arrays with original None positions preserved.
        If False, returns float64 arrays with np.nan at original None positions.

    Returns
    -------
    gx, gy, magnitude : tuple of np.ndarray
    """
    # Convert None -> np.nan and numbers -> float64
    data = np.array(arr, dtype=np.float64)
    none_mask = np.isnan(data)

    # Standard 3x3 Sobel kernels
    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float64)
    ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float64)

    # Pad borders with np.nan so edge cells ignore out-of-bounds neighbors
    padded = np.pad(data, pad_width=1, mode="constant", constant_values=np.nan)
    windows = sliding_window_view(padded, (3, 3))

    # Ignore None (NaN) cells during the weighted sum
    gx = np.nansum(windows * kx, axis=(-2, -1))
    gy = np.nansum(windows * ky, axis=(-2, -1))
    magnitude = np.hypot(gx, gy)

    # Restore None / NaN at cells that were originally None
    gx[none_mask] = np.nan
    gy[none_mask] = np.nan
    magnitude[none_mask] = np.nan

    if keep_none:
        def to_object(a):
            out = a.astype(object)
            out[none_mask] = None
            return out

        return to_object(gx), to_object(gy), to_object(magnitude)

    return gx, gy, magnitude

def laplacian_ignore_none(
    arr: np.ndarray,
    diagonal: bool = False,
    keep_none: bool = False,
) -> np.ndarray:
    """
    Apply a 2D Laplacian filter to a dtype=object array containing numbers and None.

    Parameters
    ----------
    arr : np.ndarray
        2D array (dtype=object) with numeric values and None.
    diagonal : bool
        If False (default), uses the 4-connected kernel (center -4).
        If True, uses the 8-connected kernel including diagonals (center -8).
    keep_none : bool
        If False (default), returns a float64 array with np.nan at original None
        positions (ready for matplotlib.pyplot.imshow).
        If True, returns a dtype=object array with None preserved.

    Returns
    -------
    np.ndarray
    """
    data = np.array(arr, dtype=np.float64)
    none_mask = np.isnan(data)

    if diagonal:
        kernel = np.array([[1,  1, 1],
                           [1, -8, 1],
                           [1,  1, 1]], dtype=np.float64)
    else:
        kernel = np.array([[0,  1, 0],
                           [1, -4, 1],
                           [0,  1, 0]], dtype=np.float64)

    padded = np.pad(data, pad_width=1, mode="constant", constant_values=np.nan)
    windows = sliding_window_view(padded, (3, 3))

    # Substituting NaN neighbors with the window's center value cancels their
    # difference (neighbor - center = 0) and automatically adjusts the center
    # weight from -4 (or -8) to match the number of valid neighbors.
    filled = np.where(np.isnan(windows), data[:, :, None, None], windows)
    lap = np.sum(filled * kernel, axis=(-2, -1))

    lap[none_mask] = np.nan

    if keep_none:
        out = lap.astype(object)
        out[none_mask] = None
        return out

    return lap


if __name__ == "__main__":
    

    from framework.reflection_reconstruction import find_orientation_crossings

    from framework.blockbase_pipeline import BlockBaseFrameWork
    from framework.grouping_framework import BlockGroup
    from framework.orientation_estimation import ban_bandpass
    from framework.orientation_estimation import local_multipeak
    from framework.crossing_field import ban_bandpass_gaussian, exclude_boundary, visualize_crossings, find_orientation_crossings

    from framework.blockattribute import find_orientation, find_orientation_support_v, find_frequency

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
    # save_dir = r"C:\work\image_processing\latent_fingerprint\model\data\output"
    out_path = r"D:\work\image_processing\Latent_fingerprint\SFP_latent_fingerprint_enh\segmentation\Latent_fingerprint_segmentation_KSIP2026\data\sd302h_original_500ppi"
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

        # ---- Create KS map
        ks_map = np.zeros((len(row_map_index), len(col_map_index)))
        ks_map = BBF.apply_func_map(magnitude, fft_peak_mean_var, output_is_img=False, output_vector=ks_map)
        ks_map = normalize_image(ks_map)


        # ---- Create VAR map
        var_map = np.zeros((len(row_map_index), len(col_map_index)))
        var_map = BBF.apply_func_map(magnitude, fft_peak_mean_var, output_is_img=False, output_vector=var_map)
        var_map = normalize_image(var_map)
        # plot_all([img, pmv_map, var_map], cmap=['gray', 'hot', 'hot']) # PMV and VAR map is the same



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
        plot_all([pad_img, pmv_map, pmv_mask], cmap=['gray', 'hot', 'gray'], title_list=["Original", "PMV Map", "Starting Peak Pos"])
        
        # ---- Create max pos Orientation map ----
        orientation_map = np.empty((len(row_map_index), len(col_map_index)), dtype=np.ndarray)
        orientation_map = BBF.apply_func_map(magnitude, find_orientation, output_is_img=False, output_vector=orientation_map)
        # ---- ---- ----
        # ---- Create max pos Frequency map ----
        freq_map = np.empty((len(row_map_index), len(col_map_index)), dtype=np.float32)
        freq_map = BBF.apply_func_map(magnitude, find_frequency, output_is_img=False, output_vector=freq_map)
        freq_map[freq_map>15] = 0
        
        plot_orientation_map(orientation_map, background_image=pad_img, overlap_block_size=64, non_overlap_block_size=16, overlay_alpha=0.3)

        # # ---- Try doing sobel and laplacian
        # gx, gy, magnitude = sobel_ignore_none(orientation_map, keep_none=False)
        # lap = laplacian_ignore_none(orientation_map)
        # plot_all([pad_img, lap, magnitude, freq_map])
        # # ---- ---- ----

        # # 1. Run your original function (unmodified)
        raw_crossings = find_orientation_crossings(orientation_map, d=3.0, dist_tol=1, cluster_tol=1, min_lines=60, max_crossings=10000, inside_image_only=True, activation_map=pmv_mask)
        avg = 0
        crossings = []
        for crossing in raw_crossings:
            cy, cx = crossing[0] # Continuous float coordinates
            icy, icx = int(round(cy)), int(round(cx))
            if 0 <= icy < pmv_mask.shape[0] and 0 <= icx < pmv_mask.shape[1]:
                if pmv_mask[icy, icx]: # Only keep if inside mask
                    crossings.append(crossing)
                    avg += len(crossing)-1
        total = len(crossings)
        if total == 0:
            avg = 0
        else:
            avg = avg/len(crossings)
        print(f"Total Number of Crossing under current criterion are :{total}")
        print(f"The average number of lines is {avg}")
        
        visualize_crossings(orientation_map, crossings, background_image=pad_img, cmap='gray')
        
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
        
        # # ---- CreateOrientation Tensor Map
        # attribute_map = np.empty((len(row_map_index), len(col_map_index)), dtype=np.ndarray)
        # # each cell will contain couple_peak
        # # You can get orientation from couple_peak[index]["orientation"]
        # attribute_map = BBF.apply_func_map(magnitude, find_orientation_support_v, output_is_img=False, output_vector=attribute_map)
        # # ---- ---- ----

        # # Let us start with the non multi thread approach
        # for crossing in crossings:
        #     # Crossing is in the form of
        #     # (crossingpoint), (index), (index), ...
        #     crossing_pos = crossing[0] # Exact crossing position
        #     source_indexes = crossing[1:] # All sources indexes (row, col)

        #     # Initiate Param
        #     member = [source_index for source_index in source_indexes] # All source.
            
        #     while True:
        #         pass