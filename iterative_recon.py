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

from cluster_growth import grow_reconstruct_crossing_iterative

def project_line_supercover(p0: tuple[float, float], p1: tuple[float, float], shape: tuple[int, int] = None) -> list[tuple[int, int]]:
    """
    Finds ALL grid blocks (row, col) that a continuous geometric line segment 
    from p0=(y0, x0) to p1=(y1, x1) physically intersects.
    
    Prevents pi/4 (45°) and pi/2 (90°) angle snapping artifacts.
    """
    y0, x0 = p0
    y1, x1 = p1

    curr_y, curr_x = int(np.floor(y0)), int(np.floor(x0))
    end_y, end_x = int(np.floor(y1)), int(np.floor(x1))

    dy = y1 - y0
    dx = x1 - x0

    step_y = 1 if dy > 0 else (-1 if dy < 0 else 0)
    step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)

    # Parametric t step sizes along continuous ray L(t) = P0 + t * dP
    if dy != 0:
        t_max_y = (curr_y + (1 if step_y > 0 else 0) - y0) / dy
        t_delta_y = abs(1.0 / dy)
    else:
        t_max_y = float('inf')
        t_delta_y = float('inf')

    if dx != 0:
        t_max_x = (curr_x + (1 if step_x > 0 else 0) - x0) / dx
        t_delta_x = abs(1.0 / dx)
    else:
        t_max_x = float('inf')
        t_delta_x = float('inf')

    blocks = []
    H, W = shape if shape is not None else (float('inf'), float('inf'))
    max_steps = int(abs(curr_x - end_x) + abs(curr_y - end_y)) + 50

    for _ in range(max_steps):
        if 0 <= curr_y < H and 0 <= curr_x < W:
            if not blocks or blocks[-1] != (curr_y, curr_x):
                blocks.append((curr_y, curr_x))

        if curr_x == end_x and curr_y == end_y:
            break

        # Move to whichever cell boundary (X or Y) is encountered first along ray
        if t_max_x < t_max_y:
            t_max_x += t_delta_x
            curr_x += step_x
        elif t_max_y < t_max_x:
            t_max_y += t_delta_y
            curr_y += step_y
        else: # Exact corner boundary intersection
            t_max_x += t_delta_x
            t_max_y += t_delta_y
            curr_x += step_x
            curr_y += step_y

    return blocks

def get_crossing_trajectories(crossings_output: list, shape: tuple[int, int] = None) -> list[dict]:
    """
    Projects rays from source block cell centers (sy + 0.5, sx + 0.5) 
    to exact floating-point crossing coordinates (cy, cx).
    """
    results = []

    for item in crossings_output:
        cy, cx = item[0]
        source_blocks = item[1:]
        line_paths = []

        for sy, sx in source_blocks:
            # Ray origin at center of source block
            p0 = (sy + 0.5, sx + 0.5)
            p1 = (cy, cx)
            
            # Extract intersected grid blocks along continuous vector
            path_blocks = project_line_supercover(p0, p1, shape=shape)
            line_paths.append(path_blocks)

        results.append({
            'crossing_point': (cy, cx),
            'source_blocks': source_blocks,
            'line_paths': line_paths
        })

    return results

def visualize_crossing_blocks(shape: tuple[int, int], 
                              trajectories_data: list[dict], 
                              background_image=None, 
                              activation_map=None, 
                              cmap='gray', 
                              block_size=16):
    """
    Plots continuous ray vector lines and intersected grid blocks overlayed 
    directly on a full-resolution pixel background image.

    Parameters:
        shape: (H, W) dimensions in block grid space.
        trajectories_data: Output list from `get_crossing_trajectories`.
        background_image: 2D image array in pixel space (e.g. pad_img).
        activation_map: 2D block activation array (H, W).
        cmap: Colormap for background_image.
        block_size: Pixel size per block cell (default: 16).
    """
    H, W = shape
    pix_W = W * block_size
    pix_H = H * block_size

    fig, ax = plt.subplots(figsize=(10, 10))

    # 1. Background image overlay (spans native pixel coordinates)
    if background_image is not None:
        img_h, img_w = background_image.shape[:2]
        ax.imshow(background_image, cmap=cmap, origin='upper', 
                  extent=[0, img_w, img_h, 0], alpha=0.75)
    elif activation_map is not None:
        ax.imshow(activation_map, cmap='Blues', origin='upper', 
                  extent=[0, pix_W, pix_H, 0], alpha=0.3)
    else:
        grid_bg = np.zeros((pix_H, pix_W))
        ax.imshow(grid_bg, cmap='gray_r', alpha=0.05, origin='upper',
                  extent=[0, pix_W, pix_H, 0])

    # Optional: Overlay activation map on top of background image if both provided
    if background_image is not None and activation_map is not None:
        ax.imshow(activation_map, cmap='Blues', origin='upper', 
                  extent=[0, pix_W, pix_H, 0], alpha=0.3)

    # 2. Block Grid Lines on exact 16-pixel boundaries
    ax.set_xticks(np.arange(0, pix_W + 1, block_size))
    ax.set_yticks(np.arange(0, pix_H + 1, block_size))
    ax.grid(True, which='major', color='lightgray', linestyle='--', linewidth=0.7)

    color_palette = mpl.colormaps['tab10']

    # 3. Intersected Blocks & Continuous Trajectories
    for cross_idx, item in enumerate(trajectories_data):
        cy_block, cx_block = item['crossing_point']
        paths = item['line_paths']
        sources = item['source_blocks']

        # Continuous crossing point in pixel coordinates
        cx_pix = cx_block * block_size
        cy_pix = cy_block * block_size

        # Floating-point crossing marker (Red star)
        ax.plot(cx_pix, cy_pix, '*', markersize=16, color='red', zorder=6,
                label='Crossing Point (float)' if cross_idx == 0 else "")

        for line_idx, (path, (sy, sx)) in enumerate(zip(paths, sources)):
            color = color_palette((cross_idx * 3 + line_idx) % 10)

            # Highlight each intersected 16x16 pixel block cell
            for y, x in path:
                rect = plt.Rectangle((x * block_size, y * block_size), 
                                     block_size, block_size, 
                                     facecolor=color, alpha=0.35, 
                                     edgecolor=color, linewidth=1.2, zorder=3)
                ax.add_patch(rect)

            # Source block center in pixel coordinates
            sx_pix = (sx + 0.5) * block_size
            sy_pix = (sy + 0.5) * block_size

            # True continuous ray line: Source Center -> Crossing Point
            ax.plot([sx_pix, cx_pix], [sy_pix, cy_pix], '-', color=color, linewidth=2, zorder=5,
                    label=f'Ray {cross_idx+1}-{line_idx+1}')

            # Mark Source Block Center
            ax.plot(sx_pix, sy_pix, 's', color='darkgreen', markersize=7, zorder=6,
                    label='Source Center' if (cross_idx == 0 and line_idx == 0) else "")

    # 4. Axes & Image Coordinate Setup (Row Y down, Col X right)
    ax.set_xlim(-block_size, pix_W + block_size)
    ax.set_ylim(pix_H + block_size, -block_size)

    ax.set_xlabel("Col X (Pixels)")
    ax.set_ylabel("Row Y (Pixels)")
    ax.set_title(f"Continuous Ray Trajectories & Intersected Blocks ({block_size}x{block_size} Grid)")
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))
    plt.tight_layout()
    plt.show()

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

def find_crossing_point(blocks, orientations):
    """
    Finds the intersection point of multiple lines given their origins and orientations.
    
    Args:
        blocks: List of tuples (y, x) representing the source points.
        orientations: List of floats representing the angles (theta) in radians.
        
    Returns:
        Tuple (cross_y, cross_x) representing the crossing point, 
        or None if no lines intersect (e.g., all are parallel).
    """
    if len(blocks) < 2 or len(blocks) != len(orientations):
        return None
        
    blocks = np.array(blocks)
    ys = blocks[:, 0]
    xs = blocks[:, 1]
    thetas = np.array(orientations, dtype=float)
    
    # Line equation components: A*x + B*y = C
    A = np.sin(thetas)
    B = -np.cos(thetas)
    C = xs * A + ys * B
    
    cross_xs = []
    cross_ys = []
    
    N = len(blocks)
    # Calculate all pairwise intersections
    for i in range(N - 1):
        for j in range(i + 1, N):
            Det = A[i] * B[j] - A[j] * B[i]
            
            # Skip if lines are practically parallel
            if abs(Det) > 1e-6:
                cross_x = (C[i] * B[j] - C[j] * B[i]) / Det
                cross_y = (A[i] * C[j] - A[j] * C[i]) / Det
                
                cross_xs.append(cross_x)
                cross_ys.append(cross_y)
                
    if not cross_xs:
        return None # No valid intersections found
        
    # Average the intersections if there are more than 2 lines
    avg_y = float(np.mean(cross_ys))
    avg_x = float(np.mean(cross_xs))
    
    return (avg_y, avg_x)

def visualize_single_crossing_centroid(ori_array, crossing_data, centroid_block, crossing_idx, extension_length=8.0, background_image=None, cmap='gray', block_size=16, offset=24):
    """
    Visualizes a single original crossing and its new centroid on the full image.
    Pops up in a separate Matplotlib window for each crossing.
    Includes dashed lines to show how the orientation projection changed.
    """
    H = len(ori_array)
    W = len(ori_array[0]) if H > 0 else 0
    
    pix_W = W * block_size
    pix_H = H * block_size
    
    # Create a new figure for this specific crossing
    fig, ax = plt.subplots(figsize=(10, 10))
    if hasattr(fig.canvas.manager, 'set_window_title'):
        fig.canvas.manager.set_window_title(f"Crossing {crossing_idx} Comparison")
        
    ax.set_xlim(-block_size + offset, pix_W + block_size + offset)
    ax.set_ylim(-block_size + offset, pix_H + block_size + offset)
    ax.invert_yaxis()
    
    # 1. Render background image
    if background_image is not None:
        img_h, img_w = background_image.shape[:2]
        ax.imshow(background_image, cmap=cmap, origin='upper', 
                  extent=[0, img_w, img_h, 0], alpha=0.75)
    
    # 2. Pixel Grid lines
    ax.set_xticks(np.arange(offset, pix_W + offset + 1, block_size))
    ax.set_yticks(np.arange(offset, pix_H + offset + 1, block_size))
    ax.grid(which='major', color='lightgray', linestyle='-', linewidth=0.5, alpha=0.7)
    
    # 3. Orientation Field
    for y in range(H):
        for x in range(W):
            theta = ori_array[y][x]
            if theta is not None:
                line_len = (block_size / 2) - 1 
                dx = np.cos(theta) * line_len
                dy = np.sin(theta) * line_len
                cx_pixel = offset + x * block_size + (block_size / 2)
                cy_pixel = offset + y * block_size + (block_size / 2)
                ax.plot([cx_pixel - dx, cx_pixel + dx], 
                        [cy_pixel - dy, cy_pixel + dy], 
                        color='white' if background_image is not None else 'dimgray', 
                        linewidth=1.5)
    
    # 4. Plot this specific Crossing, its Centroid, and Shifts
    cy_block, cx_block = crossing_data[0] 
    blocks = crossing_data[1:]
    color = '#d62728' # Red for the old data
    new_color = '#00FF00' # Bright Green for the new data
    
    # Apply offset for Old Crossing
    old_cx_pixel = offset + cx_block * block_size
    old_cy_pixel = offset + cy_block * block_size
    
    # Apply offset for New Centroid
    cent_cy_block, cent_cx_block = centroid_block
    new_cx_pixel = offset + cent_cx_block * block_size
    new_cy_pixel = offset + cent_cy_block * block_size
    
    added_old_label = False
    added_new_label = False
    
    # Draw Source Blocks and both projection lines
    for (by, bx) in blocks:
        bx_center = offset + bx * block_size + (block_size / 2)
        by_center = offset + by * block_size + (block_size / 2)
        ax.scatter(bx_center, by_center, color=color, marker='s', s=40, edgecolor='black', zorder=4)
        
        # --- OLD Projection Line (Solid, Faded Red) ---
        dist_pixel_old = np.hypot(old_cx_pixel - bx_center, old_cy_pixel - by_center)
        if dist_pixel_old > 0:
            ext_pixel = extension_length * block_size
            ex_old = old_cx_pixel + ((old_cx_pixel - bx_center) / dist_pixel_old) * ext_pixel
            ey_old = old_cy_pixel + ((old_cy_pixel - by_center) / dist_pixel_old) * ext_pixel
            ax.plot([bx_center, ex_old], [by_center, ey_old], color=color, linestyle='-', 
                    alpha=0.3, linewidth=2.0, label='Old Projection' if not added_old_label else "")
            added_old_label = True

        # --- NEW Projection Line (Dashed, Bright Green) ---
        dist_pixel_new = np.hypot(new_cx_pixel - bx_center, new_cy_pixel - by_center)
        if dist_pixel_new > 0:
            ext_pixel = extension_length * block_size
            ex_new = new_cx_pixel + ((new_cx_pixel - bx_center) / dist_pixel_new) * ext_pixel
            ey_new = new_cy_pixel + ((new_cy_pixel - by_center) / dist_pixel_new) * ext_pixel
            ax.plot([bx_center, ex_new], [by_center, ey_new], color=new_color, linestyle='--', 
                    alpha=0.8, linewidth=2.0, label='New Projection' if not added_new_label else "")
            added_new_label = True

    # Plot Old Crossing (Star)
    ax.scatter(old_cx_pixel, old_cy_pixel, color=color, marker='*', s=300, edgecolor='black', alpha=0.5, zorder=5, label='Old Crossing')
    
    # Plot New Centroid (Plus)
    ax.scatter(new_cx_pixel, new_cy_pixel, color=new_color, marker='P', s=200, edgecolor='black', zorder=6, label='New Centroid')
    
    # Draw connecting shift line between Old Crossing and New Centroid
    ax.plot([old_cx_pixel, new_cx_pixel], [old_cy_pixel, new_cy_pixel], color='yellow', linestyle=':', linewidth=2, zorder=5, label='Shift Vector')

    ax.set_title(f"Crossing #{crossing_idx} | Old vs Centroid Shift (Offset={offset})")
    ax.set_xlabel("X (Pixels)")
    ax.set_ylabel("Y (Pixels)")
    
    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.tight_layout()
    plt.show(block=False)

def mask_block_magnitude(patch_magnitude, double_peak_positions, thresh=0.8):
    """
    Applies map_clustering_peak_threshold for symmetric double peaks 
    and returns the masked frequency magnitude patch.
    """
    clusters = map_clustering_peak_threshold(patch_magnitude, double_peak_positions, thresh=thresh)
    if not clusters:
        return patch_magnitude
        
    combined_mask = np.clip(np.sum(clusters, axis=0), 0, 1).astype(np.float32)
    return patch_magnitude * combined_mask

def mask_block_magnitude_watershed(patch_magnitude, double_peak_positions, thresh=0.8):
    """
    Applies marker-based watershed segmentation for symmetric double peaks 
    using the updated BlockAttribute class, and returns the masked frequency patch.
    """
    # 1. Instantiate the BlockAttribute with the current frequency patch
    BA = BlockAttribute(patch_magnitude)
    
    # 2. Manually inject the tracked peak pair to act as the watershed markers.
    # This forces the watershed algorithm to only grow from the selected trajectory.
    pos, inv_pos = double_peak_positions
    BA.couple_peak = [{"position": (pos, inv_pos)}]
    
    # 3. Run the new watershed segmentation
    binary_mask, labeled_clusters, paired_masks = BA.segment_peaks_watershed()
    
    # 4. Fallback if the segmentation fails or returns empty
    if binary_mask is None or np.count_nonzero(binary_mask) == 0:
        return patch_magnitude
        
    # 5. Convert the 255-based uint8 mask to a 0.0-1.0 float32 multiplier
    combined_mask = (binary_mask > 0).astype(np.float32)
    
    return patch_magnitude * combined_mask

def convert_to_triangular_mask(mask, ori):
    """
    Converts a single isolated pair mask into bounded triangular sectors (wedges) that:
    1. Match the max angular spread of the original peak.
    2. Terminate at the maximum radial distance (frequency) of the original pair.
    
    Args:
        mask (np.ndarray): 2D binary mask.
        ori (float): Orientation angle of the mask in degrees.
        
    Returns:
        np.ndarray: The new triangular mask.
    """
    if mask is None or not np.any(mask):
        return mask

    h, w = mask.shape
    cy, cx = h // 2, w // 2
    
    # Find coordinates of the original pair mask pixels
    Y, X = np.where(mask > 0)
    
    # Exclude center pixel to avoid division by zero
    valid = (Y != cy) | (X != cx)
    Y_valid, X_valid = Y[valid], X[valid]
    
    if len(Y_valid) == 0:
        return mask.copy()
    
    # 1. Compute maximum angular spread (theta width)
    theta_mask = np.degrees(np.arctan2(Y_valid - cy, X_valid - cx))
    diff = np.abs(theta_mask - ori) % 180.0
    diff = np.minimum(diff, 180.0 - diff)
    max_width = np.max(diff)
    max_width = max(max_width, 1.5)  # Enforce minimum angular width
    
    # 2. Compute maximum frequency radius (r max)
    radial_distances = np.hypot(Y_valid - cy, X_valid - cx)
    r_max = np.max(radial_distances)
    
    # 3. Create polar grid for the entire block
    YY, XX = np.indices((h, w))
    dy, dx = YY - cy, XX - cx
    grid_r = np.hypot(dy, dx)
    grid_theta = np.degrees(np.arctan2(dy, dx))
    
    # Calculate angular difference on grid
    diff_grid = np.abs(grid_theta - ori) % 180.0
    diff_grid = np.minimum(diff_grid, 180.0 - diff_grid)
    
    # 4. Combine angular condition AND radial cap condition
    angle_condition = (diff_grid <= max_width)
    radius_condition = (grid_r <= r_max)
    
    new_mask = (angle_condition & radius_condition).astype(mask.dtype)
    
    # Keep DC component active
    new_mask[cy, cx] = 1
    
    return new_mask

# def mask_block_magnitude_triangle(patch_magnitude, double_peak_positions, thresh=0.8):
#     """
#     Applies marker-based watershed segmentation for symmetric double peaks 
#     using the updated BlockAttribute class, and returns the masked frequency patch.
#     """
#     # 1. Instantiate the BlockAttribute with the current frequency patch
#     BA = BlockAttribute(patch_magnitude)
    
#     # 2. Manually inject the tracked peak pair to act as the watershed markers.
#     # This forces the watershed algorithm to only grow from the selected trajectory.
#     pos, inv_pos = double_peak_positions
#     BA.couple_peak = [{"position": (pos, inv_pos)}]
    
#     # 3. Run the new watershed segmentation
#     binary_mask, labeled_clusters, paired_masks = BA.segment_peaks_watershed()
    
#     # 4. Fallback if the segmentation fails or returns empty
#     if binary_mask is None or np.count_nonzero(binary_mask) == 0:
#         return patch_magnitude

#     binary_mask = convert_to_triangular_mask()
    
#     # 5. Convert the 255-based uint8 mask to a 0.0-1.0 float32 multiplier
#     combined_mask = (binary_mask > 0).astype(np.float32)
    
#     return patch_magnitude * combined_mask


def run_safe_istft(BBF, freq_map):
    bbf_copy = copy.deepcopy(BBF)
    bbf_copy.setMagnitude(freq_map)
    bbf_copy.istft()
    return bbf_copy.get_output_img()

def animate_reconstruction_and_growth(ori_array, history, crossing_idx, background_image=None, 
                                      cmap='gray', block_size=16, offset=24, extension_length=8.0):
    """
    Renders frame-by-frame animation:
    Left Subplot  -> Reconstructed Spatial Image from masked iSTFT
    Right Subplot -> Cluster growth and Centroid trajectory

    #### ---- 
    """
    H = len(ori_array)
    W = len(ori_array[0]) if H > 0 else 0
    pix_W, pix_H = W * block_size, H * block_size
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    if hasattr(fig.canvas.manager, 'set_window_title'):
        fig.canvas.manager.set_window_title(f"Crossing {crossing_idx} - iSTFT Reconstruction & Growth")
        
    def update(frame):
        ax1.clear()
        ax2.clear()
        
        state = history[frame]
        step = state['step']
        cluster = state['cluster']
        centroid_block = state['centroid']
        reconstructed_img = state['reconstructed_img']
        
        # --- SUBPLOT 1: Spatial Image Reconstruction ---
        ax1.imshow(reconstructed_img, cmap='gray')
        ax1.set_title(f"iSTFT Spatial Reconstruction (Step {step})")
        ax1.axis('off')
        
        # --- SUBPLOT 2: Cluster Growth & Orientation Field ---
        ax2.set_xlim(-block_size + offset, pix_W + block_size + offset)
        ax2.set_ylim(-block_size + offset, pix_H + block_size + offset)
        ax2.invert_yaxis()
        
        if background_image is not None:
            img_h, img_w = background_image.shape[:2]
            ax2.imshow(background_image, cmap=cmap, origin='upper', 
                       extent=[0, img_w, img_h, 0], alpha=0.5)
            
        ax2.set_xticks(np.arange(offset, pix_W + offset + 1, block_size))
        ax2.set_yticks(np.arange(offset, pix_H + offset + 1, block_size))
        ax2.grid(which='major', color='lightgray', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Draw base orientation field
        for y in range(H):
            for x in range(W):
                theta = ori_array[y][x]
                if theta is not None:
                    line_len = (block_size / 2) - 1 
                    dx = np.cos(theta) * line_len
                    dy = np.sin(theta) * line_len
                    cx_pixel = offset + x * block_size + (block_size / 2)
                    cy_pixel = offset + y * block_size + (block_size / 2)
                    ax2.plot([cx_pixel - dx, cx_pixel + dx], 
                             [cy_pixel - dy, cy_pixel + dy], 
                             color='white' if background_image is not None else 'dimgray', 
                             linewidth=1.0, alpha=0.3)
        
        # Apply offset to centroid
        cent_cy_block, cent_cx_block = centroid_block
        new_cx_pixel = offset + cent_cx_block * block_size
        new_cy_pixel = offset + cent_cy_block * block_size
        
        # Plot Cluster Blocks & Projections
        for (by, bx) in cluster:
            bx_center = offset + bx * block_size + (block_size / 2)
            by_center = offset + by * block_size + (block_size / 2)
            
            is_new = (by, bx) in state['new_blocks'] and step > 0
            block_color = 'cyan' if is_new else '#d62728'
            
            ax2.scatter(bx_center, by_center, color=block_color, marker='s', s=50, edgecolor='black', zorder=4)
            
            dist_pixel = np.hypot(new_cx_pixel - bx_center, new_cy_pixel - by_center)
            if dist_pixel > 0:
                ext_pixel = extension_length * block_size
                ex = new_cx_pixel + ((new_cx_pixel - bx_center) / dist_pixel) * ext_pixel
                ey = new_cy_pixel + ((new_cy_pixel - by_center) / dist_pixel) * ext_pixel
                ax2.plot([bx_center, ex], [by_center, ey], color='#00FF00', linestyle='--', alpha=0.5, linewidth=1.5)

        # Plot Trajectory & Current Centroid
        traj_x = [offset + h['centroid'][1] * block_size for h in history[:frame+1]]
        traj_y = [offset + h['centroid'][0] * block_size for h in history[:frame+1]]
        ax2.plot(traj_x, traj_y, color='yellow', linestyle='-', linewidth=2, marker='o', markersize=4, zorder=5)
        ax2.scatter(new_cx_pixel, new_cy_pixel, color='#00FF00', marker='P', s=200, edgecolor='black', zorder=6)
        
        ax2.set_title(f"Cluster Expansion | Step {step}/{len(history)-1} | Blocks: {len(cluster)}")
        ax2.set_xlabel("X (Pixels)")
        ax2.set_ylabel("Y (Pixels)")
        plt.tight_layout()

    anim = FuncAnimation(fig, update, frames=len(history), interval=900, repeat=False)
    plt.show()
    return anim

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

if __name__ == "__main__":
    

    from framework.reflection_reconstruction import find_orientation_crossings

    from framework.blockbase_pipeline import BlockBaseFrameWork
    from framework.grouping_framework import BlockGroup
    from framework.orientation_estimation import ban_bandpass
    from framework.orientation_estimation import banning_peak, local_multipeak
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
    # out_path = r"C:\work\image_processing\latent_fingerprint\model\data\TV_img"
    out_path = r"C:\work\image_processing\latent_fingerprint\model\data\data_tv"
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
        plot_all([pad_img, pmv_map, activation_map], cmap=['gray', 'hot', 'gray'], title_list=["Original", "PMV Map", "Starting Peak Pos"])
        
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
        attribute_map = BBF.apply_func_map(magnitude, find_orientation_support_v, output_is_img=False, output_vector=attribute_map)
        # ---- ---- ----



        
        best_reconstruction = None
        max_cluster_size = 0
        animations = []
        for crossing_idx, crossing in enumerate(crossings):
            # Grow cluster
            growth_history = grow_reconstruct_crossing_iterative(
                BBF=BBF,
                crossing=crossing,
                attribute_map=attribute_map,
                ks_map=pmv_map,
                pmv_mask=pmv_mask,
                min_kurtosis=0.5,
                max_steps=30,
                mask_thresh=0.7,
                mask_func=mask_block_magnitude, 
                istft_func=run_safe_istft
            )
            
            # 2. Render animation
            anim = animate_reconstruction_and_growth(
                ori_array=o_map,
                history=growth_history,
                crossing_idx=crossing_idx,
                background_image=pad_img,
                cmap='gray',
                block_size=no_block_size,
                offset=24
            )
            
            # 3. Save as GIF
            gif_filename = os.path.join("", f"{os.path.basename(file)}_crossing_{crossing_idx}.gif")
            
            print(f"Saving animation for crossing {crossing_idx}...")
            # fps controls the speed (frames per second). Lower is slower.
            anim.save(gif_filename, writer='pillow', fps=2) 
            print(f"Saved: {gif_filename}")

        # # ---- Save the best reconstruction to a directory ----
        # if best_reconstruction is not None:
        #     # Define output directory (e.g., alongside data_tv)
        #     # save_dir = os.path.join(os.path.dirname(out_path), "reconstructed_outputs")
        #     # os.makedirs(save_dir, exist_ok=True)
            
        #     # Normalize the iSTFT output to 0-255 uint8 for saving
        #     output_img_normalized = cv.normalize(best_reconstruction, None, 0, 255, cv.NORM_MINMAX)
        #     output_img_normalized = output_img_normalized.astype(np.uint8)
            
        #     filename = os.path.basename(file)
        #     # save_path = os.path.join(save_dir, f"recon_{filename}")
        #     save_path = os.path.join(save_dir, filename)
            
        #     cv.imwrite(save_path, output_img_normalized)






        # all_centroids = []

        # MIN_KURTOSIS_THRESHOLD = 0.5  # Adjust based on your ks_map distribution
        # MAX_GROWTH_STEPS = 6

        # animations = []

        # for crossing_idx, crossing in enumerate(crossings):
        #     # 1. Grow cluster, apply frequency peak mask, and compute iSTFT frame-by-frame
        #     growth_history = grow_reconstruct_crossing_iterative(
        #         BBF=BBF,
        #         crossing=crossing,
        #         attribute_map=attribute_map,
        #         ks_map=pmv_map,
        #         pmv_mask=pmv_mask,     # <--- Passed mask here
        #         min_kurtosis=0.5,
        #         max_steps=20,
        #         mask_thresh=0.7,
        #         # dist_thresh=7.0       # <--- Set distance threshold here
        #     )
            
        #     # 2. Render side-by-side animation window
        #     anim = animate_reconstruction_and_growth(
        #         ori_array=o_map,
        #         history=growth_history,
        #         crossing_idx=crossing_idx,
        #         background_image=pad_img,
        #         cmap='gray',
        #         block_size=no_block_size,
        #         offset=24
        #     )
        #     animations.append(anim)

        # plt.show()
        #     # 2. Extract the iteration (state) that yielded the biggest cluster
        #     best_state = max(growth_history, key=lambda state: len(state['cluster']))
        #     best_reconstructed_img = best_state['reconstructed_img']
        #     cluster_size = len(best_state['cluster'])
            
        #     # 3. Normalize the iSTFT float32 output into a standard 0-255 image for saving
        #     norm_img = cv.normalize(best_reconstructed_img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
            
        #     # 4. Construct file name and save to disk
        #     # base_name = os.path.splitext(os.path.basename(file))[0]
        #     # out_filename = f"{base_name}_crossing_{crossing_idx}_size_{cluster_size}.png"
        #     out_filename = os.path.basename(file)
        #     out_filepath = os.path.join(save_dir, out_filename)
            
        #     cv.imwrite(out_filepath, norm_img)
            
        # print(f"Finished! All best reconstructions saved to: {save_dir}")