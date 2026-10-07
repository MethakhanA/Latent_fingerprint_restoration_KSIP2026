import sys
import math
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QFileDialog, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from framework.reflection_reconstruction import find_orientation_crossings

from framework.blockbase_pipeline import BlockBaseFrameWork
from framework.grouping_framework import BlockGroup
from framework.orientation_estimation import ban_bandpass
from framework.orientation_estimation import banning_peak, local_multipeak
from framework.crossing_field import ban_bandpass_gaussian, exclude_boundary, visualize_crossings, find_orientation_crossings

from framework.blockattribute import find_orientation, find_orientation_support_v

from framework.grouping_framework import map_clustering_peak_threshold

from framework.utils.plot_all import plot_all
from framework.utils.freqfilter import FreqFilter
from framework.utils.kurtosis import fft_kurtosis
from framework.utils.plot_all import plot_all


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

def distance_p2p(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def peak_2_peak_min_distance(couple_peak_pos: tuple, couple_peak_comp_pos: tuple):
    pos, inv_pos = couple_peak_pos
    comp_pos, comp_inv_pos = couple_peak_comp_pos
    d_r = distance_p2p(pos, comp_pos)
    d_r_temp = distance_p2p(pos, comp_inv_pos)
    if d_r_temp < d_r:
        d_r = d_r_temp
    return d_r

def visualize_crossings_on_ax(ax, ori_array, crossings, extension_length=8.0, background_image=None, cmap='gray', block_size=16, offset=24):
    ax.clear()
    if ori_array is None:
        ax.set_title("No data loaded")
        return []
        
    H = len(ori_array)
    W = len(ori_array[0]) if H > 0 else 0
    pix_W, pix_H = W * block_size, H * block_size
    
    ax.set_xlim(-block_size + offset, pix_W + block_size + offset)
    ax.set_ylim(pix_H + block_size + offset, -block_size + offset) # Inverted Y axis
    
    if background_image is not None:
        img_h, img_w = background_image.shape[:2]
        ax.imshow(background_image, cmap=cmap, origin='upper', extent=[0, img_w, img_h, 0], alpha=0.75)
    
    ax.set_xticks(np.arange(offset, pix_W + offset + 1, block_size))
    ax.set_yticks(np.arange(offset, pix_H + offset + 1, block_size))
    ax.grid(which='major', color='lightgray', linestyle='-', linewidth=0.5, alpha=0.7)
    
    clickable_sources = []

    for y in range(H):
        for x in range(W):
            theta = ori_array[y][x]
            if theta is not None:
                line_len = (block_size / 2) - 1 
                dx = np.cos(theta) * line_len
                dy = np.sin(theta) * line_len
                cx_pixel = offset + x * block_size + (block_size / 2)
                cy_pixel = offset + y * block_size + (block_size / 2)
                ax.plot([cx_pixel - dx, cx_pixel + dx], [cy_pixel - dy, cy_pixel + dy], 
                        color='white' if background_image is not None else 'dimgray', linewidth=1.5)
    
    colors = ['#d62728', '#2ca02c', '#1f77b4', '#9467bd', '#ff7f0e']
    for i, crossing_data in enumerate(crossings):
        cy_block, cx_block = crossing_data[0] 
        blocks = crossing_data[1:]
        color = colors[i % len(colors)]
        
        cx_pixel = offset + cx_block * block_size
        cy_pixel = offset + cy_block * block_size
        ax.scatter(cx_pixel, cy_pixel, color=color, marker='*', s=300, edgecolor='black', zorder=5)
        
        for (by, bx) in blocks:
            bx_center = offset + bx * block_size + (block_size / 2)
            by_center = offset + by * block_size + (block_size / 2)
            clickable_sources.append((bx_center, by_center, by, bx))
            ax.scatter(bx_center, by_center, color=color, marker='s', s=40, edgecolor='black', zorder=4)
            
            dist_pixel = np.hypot(cx_pixel - bx_center, cy_pixel - by_center)
            if dist_pixel > 0:
                ext_pixel = extension_length * block_size
                ex = cx_pixel + ((cx_pixel - bx_center) / dist_pixel) * ext_pixel
                ey = cy_pixel + ((cy_pixel - by_center) / dist_pixel) * ext_pixel
                ax.plot([bx_center, ex], [by_center, ey], color=color, linestyle='-', alpha=0.8, linewidth=2.0)

    ax.set_title("Main View: Orientation Crossings")
    return clickable_sources

def visualize_3x3_on_ax(ax, orientations, row_map_index, col_map_index, magnitude, pos, title, block_size=64):
    ax.clear()
    if magnitude is None:
        return
        
    pos_r, pos_c = pos[0], pos[1]
    r_pos = row_map_index[pos_r - 1]
    c_pos = col_map_index[pos_c - 1]
    
    patch_img = magnitude[r_pos:r_pos+(block_size*3), c_pos:c_pos+(block_size*3)]
    ax.imshow(patch_img, cmap='hot', origin='upper')
    ax.set_title(title, fontsize=10)
    
    line_length = block_size * 0.8 
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            idx = (dr + 1) * 3 + (dc + 1)
            if idx >= len(orientations) or orientations[idx] is None:
                continue
                
            angle = orientations[idx]
            center_y = (dr + 1) * block_size + (block_size / 2)
            center_x = (dc + 1) * block_size + (block_size / 2)
            dy = (line_length / 2) * np.sin(angle)
            dx = (line_length / 2) * np.cos(angle)
            
            y1, x1 = center_y - dy, center_x - dx
            y2, x2 = center_y + dy, center_x + dx
            ax.plot([x1, x2], [y1, y2], color='cyan', linewidth=2, marker='o', markersize=4)
    ax.axis('off')

# ==========================================
# 2. Main PyQt Application
# ==========================================

class AppGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BlockBase Pipeline Analyzer")
        self.setGeometry(100, 100, 1400, 800)
        
        # Pipeline Data State
        self.ori_array = None
        self.crossings = []
        self.bg_img = None
        self.attribute_map = None
        self.row_map_idx = None
        self.col_map_idx = None
        self.magnitude = None
        self.block_size = 16
        self.offset = 24
        self.clickable_sources = []
        
        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # --- Top Bar (Controls) ---
        top_bar = QHBoxLayout()
        self.btn_load = QPushButton("Select Image from Folder")
        self.btn_load.clicked.connect(self.open_image)
        self.lbl_status = QLabel("Ready. Please select an image.")
        top_bar.addWidget(self.btn_load)
        top_bar.addWidget(self.lbl_status)
        top_bar.addStretch()
        main_layout.addLayout(top_bar)
        
        # --- Visualization Area ---
        content_layout = QHBoxLayout()
        
        # 1. Left Panel (Main Canvas + Navigation Toolbar)
        left_layout = QVBoxLayout()
        
        self.fig_main = Figure(figsize=(8, 8))
        self.canvas_main = FigureCanvas(self.fig_main)
        self.ax_main = self.fig_main.add_subplot(111)
        self.canvas_main.mpl_connect('button_press_event', self.on_click)
        
        # Add the matplotlib toolbar to the left layout above the canvas
        self.toolbar = NavigationToolbar(self.canvas_main, self)
        left_layout.addWidget(self.toolbar)
        left_layout.addWidget(self.canvas_main)
        
        content_layout.addLayout(left_layout, stretch=2)
        
        # 2. Right Panel (3x3 views)
        right_layout = QVBoxLayout()
        self.fig_first = Figure(figsize=(4, 4))
        self.canvas_first = FigureCanvas(self.fig_first)
        self.ax_first = self.fig_first.add_subplot(111)
        right_layout.addWidget(self.canvas_first)
        
        self.fig_mod = Figure(figsize=(4, 4))
        self.canvas_mod = FigureCanvas(self.fig_mod)
        self.ax_mod = self.fig_mod.add_subplot(111)
        right_layout.addWidget(self.canvas_mod)
        
        content_layout.addLayout(right_layout, stretch=1)
        main_layout.addLayout(content_layout)
        
        self.draw_initial_empty_state()

    def draw_initial_empty_state(self):
        self.ax_main.clear()
        self.ax_main.axis('off')
        self.ax_main.text(0.5, 0.5, "No Image Loaded", ha='center', va='center')
        self.canvas_main.draw()
        
        for ax, canvas in [(self.ax_first, self.canvas_first), (self.ax_mod, self.canvas_mod)]:
            ax.clear()
            ax.axis('off')
            canvas.draw()

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif)")
        if file_path:
            self.lbl_status.setText(f"Processing: {file_path}")
            QApplication.processEvents() # Force UI update before heavy processing
            self.run_pipeline(file_path)
            self.lbl_status.setText(f"Loaded: {file_path}")

    def run_pipeline(self, file_path):
        # (Ensure your custom pipeline functions/classes are defined/imported here)
        img = cv.imread(file_path, 0)
        if img is None:
            self.lbl_status.setText("Failed to load image!")
            return
            
        img = cv.bitwise_not(img)
        
        o_block_size, no_block_size = 64, 16
        bp_r, gss_f_size = (3, 16), 3
        
        BBF = BlockBaseFrameWork(img, overlap_block_size=o_block_size, nonoverlap_block_size=no_block_size, zeromean=True, window_func='Gaussian', blur_edge=False)
        self.row_map_idx, self.col_map_idx = BBF.row_map_block_index_list, BBF.col_map_block_index_list
        pad_img = BBF.get_img()
        BBF.stft()
        
        magnitude = BBF.getMagnitude().astype(np.float32)
        magnitude = BBF.apply_func_map(magnitude, ban_bandpass_gaussian, bp_r[0], bp_r[1], gss_f_size)
        self.magnitude = magnitude
        
        ks_map = np.zeros((len(self.row_map_idx), len(self.col_map_idx)))
        ks_map = BBF.apply_func_map(magnitude, fft_kurtosis, output_is_img=False, output_vector=ks_map)
        
        max_pos_list = local_multipeak(ks_map, mean_radius_ban=None, radius_ban=2, max_peak_count=np.inf)
        max_pos_list = check_8_con(max_pos_list, ks_map)
        mp_grp = BlockGroup([itm[0] for itm in max_pos_list], [itm[1] for itm in max_pos_list])
        
        activation_map = mp_grp.generate_activation_map(ks_map.shape[0], ks_map.shape[1])
        activation_map = exclude_boundary(activation_map)        
        
        orientation_map = np.empty((len(self.row_map_idx), len(self.col_map_idx)), dtype=np.ndarray)
        orientation_map = BBF.apply_func_map(magnitude, find_orientation, output_is_img=False, output_vector=orientation_map)
        self.ori_array = orientation_map
        
        self.crossings = find_orientation_crossings(self.ori_array, d=2.0, dist_tol=1.5, cluster_tol=3.0, min_lines=2, max_crossings=100, inside_image_only=True, activation_map=activation_map)
        
        attribute_map = np.empty((len(self.row_map_idx), len(self.col_map_idx)), dtype=np.ndarray)
        self.attribute_map = BBF.apply_func_map(magnitude, find_orientation_support_v, output_is_img=False, output_vector=attribute_map)
        
        self.bg_img = pad_img
        self.block_size = no_block_size
        
        # Reset Matplotlib toolbar history before drawing new image
        self.toolbar.update()
        
        # Draw results onto GUI
        self.draw_main_view()

    def draw_main_view(self):
        self.clickable_sources = visualize_crossings_on_ax(
            self.ax_main, self.ori_array, self.crossings, 
            background_image=self.bg_img, block_size=self.block_size, offset=self.offset
        )
        self.canvas_main.draw()
        
        self.ax_first.clear()
        self.ax_first.set_title("Click a source block to see First Peak")
        self.ax_first.axis('off')
        self.canvas_first.draw()
        
        self.ax_mod.clear()
        self.ax_mod.set_title("Click a source block to see Modified Peak")
        self.ax_mod.axis('off')
        self.canvas_mod.draw()

    def on_click(self, event):
        # Prevent clicks from updating panels if the Zoom/Pan tool is active!
        if self.toolbar.mode != '':
            return

        if event.inaxes != self.ax_main or self.attribute_map is None:
            return
            
        click_x, click_y = event.xdata, event.ydata
        if click_x is None or click_y is None:
            return
            
        threshold = self.block_size / 2.0
        closest_dist = float('inf')
        clicked_block = None
        
        for (px, py, by, bx) in self.clickable_sources:
            dist = math.hypot(click_x - px, click_y - py)
            if dist < threshold and dist < closest_dist:
                closest_dist = dist
                clicked_block = (by, bx)
                
        if clicked_block:
            self.update_side_panels(clicked_block)

    def update_side_panels(self, pos):
        by, bx = pos
        l1, l2 = range(by-1, by+2), range(bx-1, bx+2) 
        try:
            couple_peak_list = [self.attribute_map[y, x] for y in l1 for x in l2]
            first_peak_list = [item[0]["angle"] if item else None for item in couple_peak_list]
            
            visualize_3x3_on_ax(self.ax_first, first_peak_list, self.row_map_idx, self.col_map_idx, 
                                self.magnitude, pos, title=f"First Peak List [{by},{bx}]", block_size=64)
            self.canvas_first.draw()
            
            center_data = self.attribute_map[by, bx][0]
            center_pos, inv_center_pos = center_data["position"]
            center_angle = center_data["angle"]
            
            modify_peak_list = []
            for index in range(len(couple_peak_list)):
                couple_peak = couple_peak_list[index]
                if index == 4: 
                    modify_peak_list.append(center_angle)
                    continue
                
                min_d_r = 100
                rl_angle = None
                if couple_peak is not None:
                    for jdex in range(len(couple_peak)):
                        peak = couple_peak[jdex]
                        pos_p, inv_pos_p = peak["position"]
                        d_r = peak_2_peak_min_distance((pos_p, inv_pos_p), (center_pos, inv_center_pos))
                        if d_r < min_d_r:
                            min_d_r = d_r
                            rl_angle = peak["angle"]
                
                modify_peak_list.append(rl_angle)
                
            visualize_3x3_on_ax(self.ax_mod, modify_peak_list, self.row_map_idx, self.col_map_idx, 
                                self.magnitude, pos, title=f"Modified Peak List [{by},{bx}]", block_size=64)
            self.canvas_mod.draw()
            
        except Exception as e:
            print(f"Error plotting 3x3 block: {e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AppGUI()
    window.show()
    sys.exit(app.exec_())