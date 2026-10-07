import math
import numpy as np
from framework.orientation_estimation import local_multipeak, find_peaks_adaptive_hierarchical
from framework.utils.check_angle_rad import check_angle_rad
from framework.utils.plot_all import plot_all
'''
---- ---- ----
Under Major Reworking.
Fix Before using
-> Each find function will return multiple vector
-> Double Peak Will be Filter
-> if a peak does not have its double. Then that peak will be ignored.

---- ---- ----

'''
import matplotlib.pyplot as plt
def plot_image_peaks(image, peaks, title="Detected Peaks", cmap='viridis', marker_color='red', marker='x'):
    """
    Plots a 2D image with peak locations overlaid.

    Parameters:
        image (np.ndarray): 2D numpy array representing the image.
        peaks (list of tuple): List of (row, col) coordinate tuples.
        title (str): Title for the plot.
        cmap (str): Matplotlib colormap for the image.
        marker_color (str): Color of the peak markers.
        marker (str): Marker style ('x', '+', 'o', etc.).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Display the image
    im = ax.imshow(image, cmap=cmap, origin='upper')
    fig.colorbar(im, ax=ax, label='Intensity')

    # Overlay peaks
    if peaks:
        # Note: Matplotlib scatter expects x=cols, y=rows
        rows, cols = zip(*peaks)
        ax.scatter(cols, rows, color=marker_color, marker=marker, s=100, linewidths=2, label=f'Peaks (n={len(peaks)})')
        ax.legend(loc='upper right', framealpha=0.8)

    ax.set_title(title)
    ax.set_xlabel("Column Index (X)")
    ax.set_ylabel("Row Index (Y)")
    plt.tight_layout()
    plt.show()

class BlockAttribute:
    def __init__(self, block_img, peak_pos=None):
        # ---- Declare Variable
        self.block_img = block_img
        self.row, self.col = block_img.shape
        self.x_center, self.y_center = self.col/2, self.row/2
        # ---- Initiate function
        # if peak_pos is None:
        #     self.find_peak_pos(radius_ban=3)
        # self.filter_double_peak() # --- Check for double peak
        # self.find_direction(is_degree=False) # --- Check for direction
        # self.find_magnitude() # --- Find magnitude
        # self.find_distance() # --- Find Frequency or Distance
        # self.find_direction() # --- Find if there is Harmonic
        # ----

    def find_peak_pos(self, radius_ban=2, max_peak_count=np.inf):
        # ---- Crop to half image, Local Multipeak, filter doublepeak
        row, col = self.row, self.col
        # crop_block_img = self.block_img[:row//2, :]
        crop_block_img = self.block_img
        peak_pos = local_multipeak(crop_block_img, radius_ban, max_peak_count)
        # peak_pos = find_peaks_adaptive_hierarchical(crop_block_img, 2, 3, 0)
        # print(peak_pos)
        if peak_pos is None:
            self.peak_pos = None
            return None
        temp_peak_pos = peak_pos.copy()
        for pos in peak_pos:
            inv_pos = (row-pos[0], col-pos[1])
            if inv_pos in peak_pos:
                continue
            else:
                # peak_pos.append(inv_pos)
                temp_peak_pos.append(inv_pos)
        # plot_image_peaks(crop_block_img, temp_peak_pos)
        self.peak_pos = temp_peak_pos
        return temp_peak_pos
    
    def filter_double_peak(self):
        """Redesign data structure -> list(dict) sorted from highest to lowest pixel value."""
        peak_pos = self.peak_pos
        if peak_pos is None or len(peak_pos) == 0:
            self.couple_peak = None
            return None

        row, col = self.row, self.col
        peak_set = set(peak_pos)
        visited = set()
        couple_peak = []

        # Sort positions by pixel value in descending order
        sorted_peaks = sorted(
            peak_pos,
            key=lambda pos: self.block_img[pos[0], pos[1]],
            reverse=True
        )

        for pos in sorted_peaks:
            if pos in visited:
                continue

            visited.add(pos)

            # Mark the inverse position so it gets skipped if encountered later
            inv_pos = (row - pos[0], col - pos[1])
            if inv_pos in peak_set:
                visited.add(inv_pos)
                couple_peak.append({
                    "position": (pos, inv_pos)
                })

        self.couple_peak = couple_peak
        return couple_peak

    
    """def find_peak_pos(self, radius_ban=2, max_peak_count=np.inf):
        # ---- Find peak position from local multipeak. (Better than banning peak)
        peak_pos = local_multipeak(self.block_img, radius_ban, max_peak_count)
        self.peak_pos = peak_pos
        # print(f"Ts peak pos is {peak_pos}")
        return peak_pos
    def filter_double_peak(self):
        # ---- Couple Double peak together
        peak_pos = self.peak_pos
        row, col = self.row, self.col
        couple_peak = {}
        if peak_pos is None:
            self.couple_peak = None
            return None
        for pos in peak_pos:
            inv_c_pos = (row-pos[0]-1, col-pos[1]-1) # change to possible list
            if inv_c_pos in peak_pos:
                couple_peak[(tuple(pos), inv_c_pos)] = {}
                continue
            else:
                inv_pos_list = [(inv_c_pos[0]+dy, inv_c_pos[1]+dx) for dy in (-1,0,1) for dx in (-1,0,1) if dy or dx]
                for inv_pos in inv_pos_list:
                    if inv_pos in peak_pos:
                        # print(f"Was here btw")
                        # print((tuple(pos), inv_pos))
                        couple_peak[(tuple(pos), inv_pos)] = {}
                        break
        self.couple_peak = couple_peak
        return couple_peak
    """
    def find_direction(self, is_degree=False):
        row, col = self.row, self.col
        couple_peak = self.couple_peak
        if couple_peak is None:
            return None
        for index in range(len(couple_peak)):
            (pos, inv_pos) = couple_peak[index]["position"]
        # for (pos, inv_pos) in self.couple_peak:
            if pos[0]==(row//2):
                inv_pos = (pos[0], col-pos[1])
            elif pos[1]==(col//2):
                inv_pos = (row-pos[0], pos[1])
            else:
                inv_pos = (row-pos[0], col-pos[1])
            p1, p2 = sorted([pos, inv_pos])
            # 2. Differences in image coordinates (row=y, col=x)
            del_row = p2[0] - p1[0]  # dy >= 0 because of sorting
            del_col = p2[1] - p1[1]  # dx
            # 3. Compute base angle in radians [0, pi]
            angle = math.atan2(del_row, del_col)
            couple_peak[index]["angle"] = angle
        self.couple_peak = couple_peak
        return couple_peak

    def find_direction_peak_pos(self):
        # Find Direction from max pos
        row, col = self.row, self.col
        pos = np.unravel_index(np.argmax(self.block_img), self.block_img.shape)
        if pos[0]==(row//2):
            inv_pos = (pos[0], col-pos[1])
        elif pos[1]==(col//2):
            inv_pos = (row-pos[0], pos[1])
        else:
            inv_pos = (row-pos[0], col-pos[1])
        p1, p2 = sorted([pos, inv_pos])
        # 2. Differences in image coordinates (row=y, col=x)
        del_row = p2[0] - p1[0]  # dy >= 0 because of sorting
        del_col = p2[1] - p1[1]  # dx
        # 3. Compute base angle in radians [0, pi]
        angle = math.atan2(del_row, del_col)
        # if (del_row>=0 and del_col>=0) or (del_row<0 and del_col<0):
        #     angle = -angle
        return angle
    
    def find_magnitude(self):
        # ---- Pack magnitude into array
        couple_peak = self.couple_peak
        block_img = self.block_img
        for index in range(len(couple_peak)):
            (pos, inv_pos) = couple_peak[index]["position"]
            couple_peak[index]["magnitude"] = block_img[pos[0], pos[1]]
        self.couple_peak = couple_peak
        return couple_peak

    def find_distance(self):
        # ---- find distance from center
        x_center, y_center = self.x_center, self.y_center
        couple_peak = self.couple_peak
        for index in range(len(couple_peak)):
            (pos, inv_pos) = couple_peak[index]["position"]
            d_y = pos[0]-y_center
            d_x = pos[1]-x_center
            d_r = np.sqrt(d_x**2+d_y**2)
            # couple_peak[(pos, inv_pos)]["distance"] = d_r
            couple_peak[index]["distance"] = d_r
        self.couple_peak = couple_peak
        return couple_peak

    def find_harmonic(self):
        # ---- find if there is harmonic
        row, col = self.row, self.col # --- Unpack Variable
        couple_peak = self.couple_peak
        block_img = self.block_img
        # for (pos, inv_pos) in couple_peak:
        for index in range(len(couple_peak)):
            (pos, inv_pos) = couple_peak[index]["position"]
            # angle = couple_peak[(pos, inv_pos)]["angle"]
            angle = couple_peak[index]["angle"]
            h_y, h_x = int(2*pos[0]*np.sin(angle)), int(2*pos[1]*np.cos(angle))
            if h_y>=row or h_x>=col:
                harmonic = None
            else:
                harmonic = block_img[h_y, h_x]
            couple_peak[index]["harmonic"] = harmonic
            # couple_peak[(pos, inv_pos)]["harmonic"] = harmonic # --- assign value
        self.couple_peak = couple_peak
        return couple_peak

    def getattribute(self):
        return self.couple_peak
    
def find_orientation(block_img):
    BA = BlockAttribute(block_img)
    return BA.find_direction_peak_pos()
    couple_peak = BA.getattribute()
    if couple_peak is None or len(couple_peak)==0:
        # no peak
        # print(couple_peak)
        # plot_all(block_img)
        return None
    return couple_peak[list(couple_peak)[0]]["angle"]
def find_orientation_support_v(block_img):
    BA = BlockAttribute(block_img)
    BA.find_peak_pos()
    BA.filter_double_peak()
    BA.find_direction()
    couple_peak = BA.getattribute()
    return couple_peak


# def find_Attribute_multi(block_img, peak_count=5, filter_double_peak=True, tol=0.8):
#     # peak_pos = local_multipeak(block_img, 3, peak_count)
#     if filter_double_peak:
#         peak_count = peak_count*2 # double peak
#     peak_pos = local_multipeak(block_img, 2, peak_count)
#     if peak_pos is None:
#         return None
#     # output_vector = np.zeros((len(peak_pos), 4))
#     output_vector = []
#     for index in range(len(peak_pos)):
#         BA = BlockAttribute(block_img, peak_pos[index])
#         direction = BA.find_direction()
#         magnitude = BA.find_magnitude()
#         frequency = BA.find_distance()
#         harmonic = BA.find_harmonic()
#         # output_vector[index] = np.array([direction, magnitude, frequency, harmonic])
#         # check for redundancy
#         if filter_double_peak:
#             is_double_peak = False
#             for vector in output_vector:
#                 if check_angle_rad(vector[0], direction) and math.isclose(vector[1], magnitude, rel_tol=tol) and math.isclose(vector[2], frequency, rel_tol=tol):
#                    is_double_peak = True
#                    break
#             if not is_double_peak:
#                 output_vector.append([peak_pos[index][0], peak_pos[index][1], direction, magnitude, frequency, harmonic])
#         else:
#             output_vector.append([peak_pos[index][0], peak_pos[index][1], direction, magnitude, frequency, harmonic])
#     return np.array(output_vector)
