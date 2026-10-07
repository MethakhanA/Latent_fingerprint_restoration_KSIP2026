import sys
import os
import cv2 as cv
import numpy as np
import csv
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QFileDialog, QGraphicsView, 
                             QGraphicsScene, QGraphicsPixmapItem, QCheckBox, 
                             QLabel, QScrollArea, QFrame, QMessageBox)
from PyQt5.QtGui import QImage, QPixmap, QPen, QColor, QBrush, QPainter
from PyQt5.QtCore import Qt

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

from dataset_preparation import *
class ZoomableView(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHint(QPainter.Antialiasing if hasattr(Qt, 'QPainter') else 0)
        # Enable panning with the left mouse button
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        # Zoom exactly where the mouse cursor is pointing
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

    def wheelEvent(self, event):
        # Calculate zoom factor based on mouse wheel scroll direction
        zoom_in_factor = 1.15
        zoom_out_factor = 1.0 / zoom_in_factor

        # event.angleDelta().y() > 0 means scrolling up (zoom in)
        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor

        self.scale(zoom_factor, zoom_factor)
class CrossingAnnotator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crossing Ground Truth Annotator")
        self.resize(1200, 800)

        self.block_size = 16
        self.image_folder = ""
        self.image_files = []
        self.current_idx = -1
        
        self.current_pad_img = None
        self.current_trajectories = []
        self.crossing_graphic_items = {} # Maps crossing_idx to list of QGraphicsItems

        self.gt_csv_path = "ground_truth_crossings.csv"

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # --- LEFT PANEL: Image Viewer ---
        self.scene = QGraphicsScene()
        self.view = ZoomableView(self.scene) 
        layout.addWidget(self.view, stretch=3)

        # --- RIGHT PANEL: Controls ---
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        layout.addWidget(control_panel, stretch=1)

        # 1. Folder Selection & Navigation
        self.btn_load_dir = QPushButton("1. Load Image Folder")
        self.btn_load_dir.clicked.connect(self.load_directory)
        control_layout.addWidget(self.btn_load_dir)

        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("< Prev")
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_next = QPushButton("Next >")
        self.btn_next.clicked.connect(self.next_image)
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.btn_next)
        control_layout.addLayout(nav_layout)

        self.lbl_info = QLabel("No folder loaded.")
        control_layout.addWidget(self.lbl_info)

        # Separator line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        control_layout.addWidget(line)

        # 2. CSV Output Selection
        self.btn_set_csv = QPushButton("2. Set Output CSV Location")
        self.btn_set_csv.clicked.connect(self.select_csv_path)
        control_layout.addWidget(self.btn_set_csv)

        self.lbl_csv_path = QLabel(f"Save Path: {os.path.basename(self.gt_csv_path)}")
        self.lbl_csv_path.setWordWrap(True)
        control_layout.addWidget(self.lbl_csv_path)

        # Separator line
        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setFrameShadow(QFrame.Sunken)
        control_layout.addWidget(line2)

        # 3. Dynamic Crossing List
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setAlignment(Qt.AlignTop)
        scroll.setWidget(self.scroll_content)
        control_layout.addWidget(scroll)

    def select_csv_path(self):
        """Opens a file dialog to choose or create a CSV save location."""
        path, _ = QFileDialog.getSaveFileName(
            self, 
            "Select Ground Truth CSV File", 
            self.gt_csv_path, 
            "CSV Files (*.csv)"
        )
        if path:
            if not path.endswith('.csv'):
                path += '.csv'
            self.gt_csv_path = path
            self.lbl_csv_path.setText(f"Save Path: {os.path.basename(path)}")

    def load_directory(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self.image_folder = folder
            valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')
            self.image_files = sorted([f for f in os.listdir(folder) if f.lower().endswith(valid_exts)])
            
            if self.image_files:
                self.current_idx = 0
                self.load_image(self.current_idx)
            else:
                self.lbl_info.setText("No valid images found in folder.")

    def run_your_pipeline(self, filepath):
        """
        INSERT YOUR PIPELINE HERE.
        This function takes the raw image path, runs your BlockBaseFrameWork,
        and returns exactly what the GUI needs to render the frame.
        """
        # ==========================================
        # PASTE YOUR FULL SCRIPT LOGIC HERE
        # ==========================================
        o_block_size = 64
        no_block_size = 16
        bp_r = (3, 16)
        gss_f_size = 3
        # 1. Load image
        img = cv.imread(filepath, 0)
        img = cv.bitwise_not(img)
        
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
        
        # r_idx, c_idx = 2, 0
        
        # # Calculate offsets using map_index for magnitude and no_index for pad_img
        # m_row, m_col = row_map_index[r_idx], col_map_index[c_idx]
        # p_row, p_col = row_no_index[r_idx], col_no_index[c_idx]

        # # Extract regions (Assuming o_block_size is 64 and no_block_size is 16)
        # ori = find_orientation(magnitude[m_row:m_row+64, m_col:m_col+64])
        # print(ori)
        # plot_all([pad_img[p_row:p_row+16, p_col:p_col+16], magnitude[m_row:m_row+64, m_col:m_col+64]])
        # ---- ---- ----

        # print(f"Total number of crossing is {len(crossings)}")
        # # 2. Extract block trajectories per orientation ray
        trajectories_data = get_crossing_trajectories(crossings)
        
        return pad_img, trajectories_data

    def load_image(self, idx):
        if idx < 0 or idx >= len(self.image_files):
            return

        filename = self.image_files[idx]
        filepath = os.path.join(self.image_folder, filename)
        self.lbl_info.setText(f"Processing: {filename} ({idx+1}/{len(self.image_files)})")
        QApplication.processEvents() 

        # 1. Run the heavy pipeline
        self.current_pad_img, self.current_trajectories = self.run_your_pipeline(filepath)

        # 2. Render Image to Qt Scene
        self.scene.clear()
        self.crossing_graphic_items.clear()
        
        h, w = self.current_pad_img.shape
        qimg = QImage(self.current_pad_img.data, w, h, w, QImage.Format_Grayscale8)
        self.scene.addPixmap(QPixmap.fromImage(qimg))
        
        # Define the offset here
        offset = 24

        # 3. Draw grid lines shifted by the offset
        pen_grid = QPen(QColor(200, 200, 200, 100))
        pen_grid.setStyle(Qt.DashLine)
        # Horizontal grid lines
        for y in range(offset, h, self.block_size):
            self.scene.addLine(0, y, w, y, pen_grid)
        # Vertical grid lines
        for x in range(offset, w, self.block_size):
            self.scene.addLine(x, 0, x, h, pen_grid)

        # 4. Draw Crossings with Offset
        colors = [QColor(214, 39, 40), QColor(44, 160, 44), QColor(31, 119, 180), 
                  QColor(148, 103, 189), QColor(255, 127, 14)]
                  
        for c_idx, traj in enumerate(self.current_trajectories):
            color = colors[c_idx % len(colors)]
            self.crossing_graphic_items[c_idx] = []
            
            # Apply offset to crossing point
            cy_block, cx_block = traj['crossing_point']
            cx_pix = offset + cx_block * self.block_size
            cy_pix = offset + cy_block * self.block_size
            
            pen_ray = QPen(color, 2)
            brush_block = QBrush(QColor(color.red(), color.green(), color.blue(), 70))
            
            for path, (sy, sx) in zip(traj['line_paths'], traj['source_blocks']):
                # Apply offset to Intersected blocks
                for by, bx in path:
                    rect = self.scene.addRect(offset + bx * self.block_size, 
                                              offset + by * self.block_size, 
                                              self.block_size, self.block_size, 
                                              QPen(Qt.NoPen), brush_block)
                    self.crossing_graphic_items[c_idx].append(rect)
                
                # Apply offset to Source Center
                sx_pix = offset + (sx + 0.5) * self.block_size
                sy_pix = offset + (sy + 0.5) * self.block_size
                
                # Draw Line and Source box
                line = self.scene.addLine(sx_pix, sy_pix, cx_pix, cy_pix, pen_ray)
                src = self.scene.addRect(sx_pix - 3, sy_pix - 3, 6, 6, QPen(Qt.black), QBrush(color))
                self.crossing_graphic_items[c_idx].extend([line, src])

            # Draw Crossing Center 
            cross_marker = self.scene.addEllipse(cx_pix - 8, cy_pix - 8, 16, 16, QPen(Qt.black), QBrush(color))
            self.crossing_graphic_items[c_idx].append(cross_marker)

        # 5. Populate Right Control Panel
        self.populate_crossing_controls()
        self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def populate_crossing_controls(self):
        # Clear previous controls
        for i in reversed(range(self.scroll_layout.count())): 
            widget = self.scroll_layout.itemAt(i).widget()
            if widget: widget.deleteLater()

        for c_idx, traj in enumerate(self.current_trajectories):
            row_frame = QFrame()
            row_layout = QHBoxLayout(row_frame)
            row_layout.setContentsMargins(0,0,0,0)

            # Checkbox to toggle visibility
            chk = QCheckBox(f"Crossing {c_idx+1}")
            chk.setChecked(True)
            chk.stateChanged.connect(lambda state, idx=c_idx: self.toggle_visibility(idx, state))
            
            # Button to save as ground truth
            btn_save = QPushButton("Save as True")
            btn_save.setStyleSheet("background-color: #e0f7fa; font-weight: bold;")
            btn_save.clicked.connect(lambda checked, idx=c_idx, c_data=traj: self.save_ground_truth(idx, c_data))

            row_layout.addWidget(chk)
            row_layout.addWidget(btn_save)
            self.scroll_layout.addWidget(row_frame)

    def toggle_visibility(self, c_idx, state):
        is_visible = (state == Qt.Checked)
        for item in self.crossing_graphic_items[c_idx]:
            item.setVisible(is_visible)

    def save_ground_truth(self, c_idx, traj_data):
        # Prompt to select a CSV path if not set yet
        if not self.gt_csv_path:
            self.select_csv_path()
            if not self.gt_csv_path:
                return  # User cancelled dialog

        filename = self.image_files[self.current_idx]
        cy, cx = traj_data['crossing_point']
        
        # Ensure target directory exists
        csv_dir = os.path.dirname(self.gt_csv_path)
        if csv_dir and not os.path.exists(csv_dir):
            os.makedirs(csv_dir, exist_ok=True)

        # Append data to selected CSV file
        file_exists = os.path.isfile(self.gt_csv_path)
        with open(self.gt_csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(['filename', 'crossing_idx', 'block_cy', 'block_cx'])
            writer.writerow([filename, c_idx, cy, cx])
        
        QMessageBox.information(
            self, 
            "Saved", 
            f"Saved crossing #{c_idx+1} for {filename}\nto: {self.gt_csv_path}\n\nMoving to next image..."
        )
        self.next_image()

    def prev_image(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_image(self.current_idx)

    def next_image(self):
        if self.current_idx < len(self.image_files) - 1:
            self.current_idx += 1
            self.load_image(self.current_idx)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CrossingAnnotator()
    window.show()
    sys.exit(app.exec_())