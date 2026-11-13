import numpy as np
from sklearn.ensemble import IsolationForest
import math

def angle_diff(a: float, b: float) -> float:
    diff = (a - b + math.pi) % (2 * math.pi) - math.pi
    abs_diff = abs(diff)
    if abs_diff > math.pi / 2:
        abs_diff = math.pi - abs_diff
    return abs_diff

def calculate_pairwise_crossings(blocks, angles):
    """Calculates all pairwise crossing points for a set of blocks and angles."""
    crossings = []
    block_pairs = []
    
    if len(blocks) < 2:
        return crossings, block_pairs
        
    ys, xs = np.array([b[0] for b in blocks]), np.array([b[1] for b in blocks])
    thetas = np.array(angles, dtype=float)
    
    A = np.sin(thetas)
    B = -np.cos(thetas)
    C = xs * A + ys * B
    
    N = len(blocks)
    for i in range(N - 1):
        for j in range(i + 1, N):
            Det = A[i] * B[j] - A[j] * B[i]
            if abs(Det) > 1e-6:
                cross_x = (C[i] * B[j] - C[j] * B[i]) / Det
                cross_y = (A[i] * C[j] - A[j] * C[i]) / Det
                crossings.append((cross_y, cross_x))
                block_pairs.append((i, j))
                
    return crossings, block_pairs

def get_peak_radius(peak):
    """Calculates the frequency radius (distance from center) of a peak."""
    if "radius" in peak:
        return peak["radius"]
    # If radius isn't explicitly saved, calculate it from the symmetric position tuple
    p1, p2 = peak["position"]
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1]) / 2.0

def grow_reconstruct_crossing_iterative(BBF, crossing, attribute_map, ks_map, pmv_mask,
                                      min_kurtosis=0.5, max_steps=8, mask_thresh=0.8,
                                      mask_func=None, istft_func=None, freq_tolerance=3.0):
    """
    Cluster growth using BFS queue, Isolation Forest outlier detection, 
    and frequency-radius based orientation fallback.
    """
    o_blocksize = BBF._BlockBaseFrameWork__overlap_blocksize
    H, W = ks_map.shape
    
    initial_crossing_pt = crossing[0]
    parent_blocks = list(crossing[1:]) 
    
    active_cluster = set(parent_blocks)
    centroid_history = [initial_crossing_pt]
    selected_peaks = {}
    
    for (r, c) in parent_blocks:
        if attribute_map[r, c] is not None:
            selected_peaks[(r, c)] = attribute_map[r, c][0]
    
    original_freq_map = BBF.getMagnitude()
    working_freq_map = np.zeros_like(original_freq_map)
    
    for (r, c) in parent_blocks:
        if (r, c) in selected_peaks:
            r_start, c_start = BBF.row_map_block_index_list[r], BBF.col_map_block_index_list[c]
            patch = original_freq_map[r_start:r_start + o_blocksize, c_start:c_start + o_blocksize].copy()
            pos_p, inv_pos_p = selected_peaks[(r, c)]["position"]
            working_freq_map[r_start:r_start + o_blocksize, c_start:c_start + o_blocksize] = \
                mask_func(patch, [pos_p, inv_pos_p], thresh=mask_thresh)

    reconstructed_img = istft_func(BBF, working_freq_map)
    history = [{
        'step': 0, 'cluster': list(active_cluster), 'centroid': initial_crossing_pt,
        'new_blocks': list(active_cluster), 'reconstructed_img': reconstructed_img
    }]
    
    l_8_neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1),  (1, 0),  (1, 1)]
    iso_forest = IsolationForest(contamination=0.2, random_state=42)

    for step in range(1, max_steps + 1):
        child_candidates = []
        child_to_parent = {} 
        
        for (r, c) in parent_blocks:
            if (r, c) not in selected_peaks: continue
            parent_angle = selected_peaks[(r, c)]["angle"]
            
            for dr, dc in l_8_neighbors:
                nr, nc = r + dr, c + dc
                if (0 <= nr < H and 0 <= nc < W and 
                    (nr, nc) not in active_cluster and (nr, nc) not in child_candidates and 
                    ks_map[nr, nc] >= min_kurtosis and pmv_mask[nr, nc] and 
                    attribute_map[nr, nc] is not None):
                    
                    child_to_parent[(nr, nc)] = (r, c)
                    
                    best_peak = min(attribute_map[nr, nc], key=lambda p: angle_diff(p["angle"], parent_angle))
                    selected_peaks[(nr, nc)] = best_peak
                    child_candidates.append((nr, nc))
                        
        if not child_candidates:
            break
            
        accepted_children = []
        current_centroid = centroid_history[-1]

        if len(child_candidates) >= 2:
            child_angles = [selected_peaks[b]["angle"] for b in child_candidates]
            crossings, pairs = calculate_pairwise_crossings(child_candidates, child_angles)
            
            if crossings:
                preds = iso_forest.fit_predict(crossings)
                inliers = [crossings[i] for i in range(len(crossings)) if preds[i] == 1]
                
                if inliers:
                    current_centroid = (np.mean([p[0] for p in inliers]), np.mean([p[1] for p in inliers]))
                
                for child_idx, child in enumerate(child_candidates):
                    child_crossing_indices = [i for i, pair in enumerate(pairs) if child_idx in pair]
                    outlier_count = sum(1 for idx in child_crossing_indices if preds[idx] == -1)
                    
                    if child_crossing_indices and outlier_count > len(child_crossing_indices) / 2:
                        
                        # ---------------------------------------------------------
                        # NEW DISTANCE LOGIC: Frequency Radius Matching
                        # ---------------------------------------------------------
                        parent_block = child_to_parent[child]
                        parent_radius = get_peak_radius(selected_peaks[parent_block])
                        
                        best_radius_diff = float('inf')
                        best_peak = None
                        
                        for candidate_peak in attribute_map[child[0], child[1]]:
                            cand_radius = get_peak_radius(candidate_peak)
                            radius_diff = abs(cand_radius - parent_radius)
                            
                            if radius_diff < best_radius_diff:
                                best_radius_diff = radius_diff
                                best_peak = candidate_peak
                        
                        # Discard if the best match is still on a vastly different frequency bin
                        if best_radius_diff > freq_tolerance:
                            del selected_peaks[child]
                            continue
                        else:
                            selected_peaks[child] = best_peak
                            
                    accepted_children.append(child)
            else:
                accepted_children = child_candidates
        else:
            accepted_children = child_candidates

        if not accepted_children:
            break

        active_cluster.update(accepted_children)
        parent_blocks = accepted_children 
        
        for (nr, nc) in accepted_children:
            r_start = BBF.row_map_block_index_list[nr]
            c_start = BBF.col_map_block_index_list[nc]
            patch = original_freq_map[r_start:r_start + o_blocksize, c_start:c_start + o_blocksize].copy()
            
            pos_p, inv_pos_p = selected_peaks[(nr, nc)]["position"]
            masked_patch = mask_func(patch, [pos_p, inv_pos_p], thresh=mask_thresh)
            working_freq_map[r_start:r_start + o_blocksize, c_start:c_start + o_blocksize] = masked_patch

        reconstructed_img = istft_func(BBF, working_freq_map)
        centroid_history.append(current_centroid)
        
        history.append({
            'step': step,
            'cluster': list(active_cluster),
            'centroid': current_centroid,
            'new_blocks': accepted_children,
            'reconstructed_img': reconstructed_img
        })
        
    return history