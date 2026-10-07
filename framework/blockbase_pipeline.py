# Path related Lib
import os
import math
from glob import glob

# Operation related Lib
import numpy as np
from tqdm import tqdm

# Image related Lib
import cv2 as cv
import matplotlib.pyplot as plt

# Local Lib
# from methlib.general import plot_all, normalize_range
# from methlib.intensity_transform import log_transform
from framework.utils.fourier import Fourier2D
from framework.utils.blur_edge import blurEdge

class BlockBaseFrameWork:
    def __init__(self, img=None, overlap_block_size:int=64, nonoverlap_block_size:int=32, zeromean:bool=False, window_func=None, blur_edge=False, blur_size=21, erode_size=21):
        """Initialize the block-based framework for image processing."""
        self.__img = img
        self.__zeromean = zeromean
        self.__window_func = window_func
        self.__blur_edge = blur_edge
        self.__blur_edge_size = (blur_size, erode_size)
        
        # Initialize map holders
        self.__spa_map_img = None
        self.__freq_map_img = None
        self.__phase_map_img = None
        self.__mean_arr = None
        self.__output_img = None

        self.__padding(overlap_block_size, nonoverlap_block_size) # Pad
        self.generate_block_index() # Generate block index
        self.__generate_spatial_map()
   
    def __crop_center(self, img: np.ndarray, center_block_size: int):
        if self.__nonoverlap_blocksize == self.__overlap_blocksize:
            return img
        
        row, col = img.shape
        s_row = (row - center_block_size) // 2
        s_col = (col - center_block_size) // 2
        
        img_crop = img[s_row : s_row + center_block_size, 
                       s_col : s_col + center_block_size].copy()
        return img_crop
   
    def __padding(self, overlap_block_size=64, nonoverlap_block_size=32):
        """Pad image by specifying overlap and non-overlap sizes.
        
        Args:
            overlap_block_size (int): The big block size (e.g., 64)
            nonoverlap_block_size (int): The small block size (e.g., 32)
        """
        img = self.__img
        
        self.__nonoverlap_blocksize = nonoverlap_block_size
        self.__overlap_blocksize = overlap_block_size
        
        rows, cols = img.shape
        max_size = max(rows, cols)
        new_size = int(math.ceil(max_size / nonoverlap_block_size) * nonoverlap_block_size) + overlap_block_size
        
        pad_top = (new_size - rows) // 2
        pad_bottom = new_size - rows - pad_top
        pad_left = (new_size - cols) // 2
        pad_right = new_size - cols - pad_left
        
        self.__img = np.pad(
            img,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            "constant",
            constant_values=np.mean(img),
        )
        # blur mask
        if self.__blur_edge:
            blur_size, erode_size = self.__blur_edge_size
            blur_mask = np.ones_like(img)*255
            blur_mask = np.pad(
                img,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                "constant",
                constant_values=0,
            )
            self.__img = blurEdge(self.__img, blur_mask, blur_size, erode_size)
        self.fp_pad = [pad_top, pad_bottom, pad_left, pad_right]

   
    def __unpad(self, input_img):
        """Unpad image to its original dimensions."""
        top, bottom, left, right = self.fp_pad
        h, w = input_img.shape
        return input_img[top : h - bottom, left : w - right]
   
    def generate_block_index(self):
        o_blocksize = self.__overlap_blocksize
        no_blocksize = self.__nonoverlap_blocksize
        
        start_index = (o_blocksize - no_blocksize) // 2
        stop_index_row = self.__img.shape[0] - (o_blocksize - (start_index + no_blocksize))
        
        # ROW indices
        self.row_no_block_index_list = range(start_index, stop_index_row, no_blocksize)
        self.row_o_block_index_list = range(0, int(o_blocksize * (stop_index_row - start_index) / no_blocksize), no_blocksize)
        self.row_map_block_index_list = range(0, len(self.row_no_block_index_list) * o_blocksize, o_blocksize)
        
        # COL indices (Note: relies on stop_index_row for square processing based on original logic)
        self.col_no_block_index_list = range(start_index, stop_index_row, no_blocksize)
        self.col_o_block_index_list = range(0, int(o_blocksize * (stop_index_row - start_index) / no_blocksize), no_blocksize)
        self.col_map_block_index_list = range(0, len(self.col_no_block_index_list) * o_blocksize, o_blocksize)
   
    def __crop_map(self, map_img):
        o_blocksize = self.__overlap_blocksize
        no_blocksize = self.__nonoverlap_blocksize
        
        crop_img = np.zeros_like(self.__img, np.float32)
        
        for row_index in range(len(self.row_no_block_index_list)):
            start_no_row = self.row_no_block_index_list[row_index]
            stop_no_row = start_no_row + no_blocksize
            
            start_map_row = self.row_map_block_index_list[row_index]
            stop_map_row = start_map_row + o_blocksize
            
            for col_index in range(len(self.col_no_block_index_list)):
                start_no_col = self.col_o_block_index_list[col_index]
                stop_no_col = start_no_col + no_blocksize
                
                start_map_col = self.col_map_block_index_list[col_index]
                stop_map_col = start_map_col + o_blocksize
                
                patch = map_img[start_map_row:stop_map_row, start_map_col:stop_map_col]
                crop_img[start_no_row:stop_no_row, start_no_col:stop_no_col] = patch
                
        return crop_img
            
    def __generate_spatial_map(self):
        o_blocksize = self.__overlap_blocksize
        
        rows_len = len(self.row_no_block_index_list)
        cols_len = len(self.col_no_block_index_list)
        
        spa_map_img = np.zeros((o_blocksize * rows_len, o_blocksize * cols_len), np.float32)
        
        for row_index in tqdm(range(rows_len)):
            start_o_row = self.row_o_block_index_list[row_index]
            stop_o_row = start_o_row + o_blocksize
            
            start_map_row = self.row_map_block_index_list[row_index]
            stop_map_row = start_map_row + o_blocksize

            for col_index in range(cols_len):
                start_o_col = self.col_o_block_index_list[col_index]
                stop_o_col = start_o_col + o_blocksize
                
                start_map_col = self.col_map_block_index_list[col_index]
                stop_map_col = start_map_col + o_blocksize
                
                patch = self.__img[start_o_row:stop_o_row, start_o_col:stop_o_col]
                spa_map_img[start_map_row:stop_map_row, start_map_col:stop_map_col] = patch
        
        self.__spa_map_img = spa_map_img

    def stft(self):
        # Unpack Variable
        o_blocksize = self.__overlap_blocksize
        no_blocksize = self.__nonoverlap_blocksize
        spa_map_img = self.__spa_map_img
        
        rows_len = len(self.row_no_block_index_list)
        cols_len = len(self.col_no_block_index_list)
        
        freq_map_img = np.zeros((o_blocksize * rows_len, o_blocksize * cols_len), np.float32)
        phase_map_img = np.zeros((o_blocksize * rows_len, o_blocksize * cols_len), np.float32)
        
        if self.__zeromean:
            mean_arr = np.zeros((rows_len, cols_len))
            
        for row_index in tqdm(range(rows_len)):
            start_o_row = self.row_o_block_index_list[row_index]
            stop_o_row = start_o_row + o_blocksize
            
            start_map_row = self.row_map_block_index_list[row_index]
            stop_map_row = start_map_row + o_blocksize

            for col_index in range(cols_len):
                start_o_col = self.col_o_block_index_list[col_index]
                stop_o_col = start_o_col + o_blocksize
                
                start_map_col = self.col_map_block_index_list[col_index]
                stop_map_col = start_map_col + o_blocksize
                
                patch = spa_map_img[start_map_row:stop_map_row, start_map_col:stop_map_col]
                    
                FFT = Fourier2D(patch, self.__zeromean, self.__window_func)
                FFT.fft()
                
                if self.__zeromean:
                    mean_arr[row_index, col_index] = FFT.avg
                    
                freq_map_img[start_map_row:stop_map_row, start_map_col:stop_map_col] = FFT.getMagnitude()
                phase_map_img[start_map_row:stop_map_row, start_map_col:stop_map_col] = FFT.getPhase()
                
        self.__freq_map_img = freq_map_img
        self.__phase_map_img = phase_map_img
        if self.__zeromean:    
            self.__mean_arr = mean_arr
  
    def istft(self):
        """Invert STFT. Needs testing on real picture."""
        o_blocksize = self.__overlap_blocksize
        no_blocksize = self.__nonoverlap_blocksize
        
        output_img = np.zeros_like(self.__img, np.float32)
        
        for row_index in range(len(self.row_no_block_index_list)):
            start_no_row = self.row_no_block_index_list[row_index]
            stop_no_row = start_no_row + no_blocksize
            
            start_map_row = self.row_map_block_index_list[row_index]
            stop_map_row = start_map_row + o_blocksize
            
            for col_index in range(len(self.col_no_block_index_list)):
                start_no_col = self.col_no_block_index_list[col_index]
                stop_no_col = start_no_col + no_blocksize
                
                start_map_col = self.col_map_block_index_list[col_index]
                stop_map_col = start_map_col + o_blocksize
                
                f_patch = self.__freq_map_img[start_map_row:stop_map_row, start_map_col:stop_map_col]
                ph_patch = self.__phase_map_img[start_map_row:stop_map_row, start_map_col:stop_map_col]

                
                FFT = Fourier2D(f_patch, frequency=True, window_func=self.__window_func)
                FFT.setPhase(ph_patch)
                FFT.ifft()
                
                out_patch = FFT.getOutputImg()
                if self.__zeromean:
                    out_patch += self.__mean_arr[row_index, col_index]
                    
                out_patch = self.__crop_center(out_patch, no_blocksize)
                output_img[start_no_row:stop_no_row, start_no_col:stop_no_col] = out_patch
        
        # Unpad all images
        self.__img = self.__unpad(self.__img)
        self.__freq_map_img = self.__unpad(self.__freq_map_img)
        self.__output_img = self.__unpad(output_img)
    
    def apply_func_map(self, map_img, func, *args, output_is_img=True, output_vector=None, activation_map=None, custom_row_index=None, custom_col_index=None, iteration_size=1):
        o_blocksize = self.__overlap_blocksize
        rows = len(self.row_map_block_index_list)
        cols = len(self.col_map_block_index_list)
        # If don't specify then Activate Every fucking thing.
        if activation_map is None:
            activation_map = np.full((rows, cols), True, dtype=bool)

        # Incase you don't want the output to be picture, but whatever vector you so desire
        if output_is_img:
            output_img = np.zeros((o_blocksize * rows, o_blocksize * cols), np.float32)
            # output_img = np.zeros_like(map_img)
        else:
            output_img = output_vector
        
        row_block_index_list = self.row_map_block_index_list
        col_block_index_list = self.col_map_block_index_list
        it_size = o_blocksize
        
        if not output_is_img:
            if custom_row_index is None:
                custom_row_index = range(rows)
                custom_col_index = range(cols)
            row_output_block_index_list = custom_row_index
            col_output_block_index_list = custom_col_index
            opt_it_size = iteration_size
        else:
            row_output_block_index_list = row_block_index_list
            col_output_block_index_list = col_block_index_list
            opt_it_size = it_size
        for row_index in range(rows):
            # Map index
            start_map_row = row_block_index_list[row_index]
            stop_map_row = start_map_row + it_size
            # Output index, Done this for shorter computation in for loop
            o_start_map_row = row_output_block_index_list[row_index]
            o_stop_map_row = o_start_map_row+opt_it_size
            for col_index in range(cols):
                # try:
                if activation_map[row_index][col_index]==True:
                    # Map index
                    start_map_col = col_block_index_list[col_index]
                    stop_map_col = start_map_col + it_size
                    # Output index
                    o_start_map_col = col_output_block_index_list[col_index]
                    o_stop_map_col = o_start_map_col+opt_it_size
                    patch = map_img[start_map_row:stop_map_row, start_map_col:stop_map_col]
                    if output_is_img:
                        output = func(patch, *args)
                        output_img[o_start_map_row:o_stop_map_row, o_start_map_col:o_stop_map_col] = output
                    else:
                        # Note that instead of allocating np.zeroes. Use the np.empty instead so you can Shove whatever the fuck inside the position.
                        output = func(patch, *args)
                        output_img[o_start_map_row, o_start_map_col] = output
                # except:
                #     pass
        return output_img    
        
    def get_img(self):
        return self.__img
    
    # def get_mask(self):
    #     return self.__mask
    
    def getSpatial(self):
        return self.__spa_map_img
    
    def getMagnitude(self):
        """
        This map should not be shown in plt if o_block_size/no_block_size
        is greater than 2, otherwise the program will freeze.
        """
        return self.__freq_map_img
    
    def getPhase(self):
        return self.__phase_map_img
    
    def getVisualizeMagnitude(self):
        """Visualize the current magnitude (No overlap block)"""
        vis_magnitude = self.__crop_map(self.__freq_map_img)
        return self.__unpad(vis_magnitude)
    
    def get_output_img(self):
        return self.__output_img
    
    def setSpatial(self, spa_map):
        self.__spa_map_img = spa_map

    def setMagnitude(self, freq_map):
        self.__freq_map_img = freq_map

if __name__ == "__main__":
    from orientation_estimation import local_multipeak
    from grouping_framework import map_clustering_peak_threshold
    def test_inv(block_img, start, stop):
        peak_pos = local_multipeak(block_img, radius_ban=2, max_peak_count=np.inf)
        if peak_pos is None:
            return np.zeros_like(block_img)
        cluster = map_clustering_peak_threshold(block_img, peak_pos, 0.7)
        clst_map = np.zeros_like(cluster[0])
        for i in range(start, stop):
            clst_map+= cluster[i]
        # clst_map = clst_map/4
        return block_img*clst_map
    from utils.plot_all import plot_all
    # img_path = r"D:\work\image_processing\Latent_fingerprint\SFP_latent_fingerprint_enh\segmentation\Latent_fingerprint_segmentation_KSIP2026\data\sd302h_original_500ppi\00002302_2B_X_L01_BP_S04_500PPI_8BPC_1CH_LP05-1_1.png"
    # img_path = r"C:\work\image_processing\latent_fingerprint\automatic_segment\data\00002313_1C_L_L02_BP_S10_500PPI_8BPC_1CH_LP03-1_1.png"
    # mask_path = r"D:\work\image_processing\Latent_fingerprint\SFP_latent_fingerprint_enh\latent_fingerprint_enhancement-master\data\masks_machine\rtp2013_11_1_T_2.png"
    folder_path = r"D:\work\image_processing\Latent_fingerprint\segment\data_TV"
    folder_path = r"D:\work\image_processing\Latent_fingerprint\segment\data"
    # img_path = r"D:\work\image_processing\Latent_fingerprint\segment\data\00002310_1C_L_L01_BP_S05_500PPI_8BPC_1CH_LP04-1_1.png"
    
    for img_path in glob(os.path.join(folder_path, '*')):
        img = cv.imread(img_path, 0)
        # plot_all(img)
        # mask = cv.imread(mask_path, 0)
        BBF = BlockBaseFrameWork(img, overlap_block_size=64, nonoverlap_block_size=16, zeromean=True, window_func="Gaussian")
        BBF.stft()
        magnitude = BBF.getMagnitude()
        # kurtosis = BBF.apply_func_map(magnitude, log_transform)
        # BBF.setMagnitude(kurtosis)

        ##---- Test ISTFT
        # BBF.istft()
        # output = BBF.get_output_img()
        # # output = normalize_range(output, (np.min(output), np.max(output)), (0, 255))
        # plot_all([img, output], title_list=["Before", "after"])
        # plot_all([img, img-output], title_list=["Original", "Delta"])
        # ---- ---- ----
        
        from grouping_framework import map_clustering_watershed, map_clustering_n_iterative, BlockGroup
        from crossing_field import ban_bandpass_gaussian
        import plotly.graph_objects as go

        start, stop = 2, 4
        # Gaussian Bandpass 3->16 with gaussian 3
        ban_magnitude = BBF.apply_func_map(magnitude, ban_bandpass_gaussian, 3, 16, 3)
        output_magnitude = BBF.apply_func_map(ban_magnitude, test_inv, start, stop)
        
        
        r_idx, c_idx = 10, 10
        row_map_list, col_map_list = BBF.row_map_block_index_list, BBF.col_map_block_index_list
        block = ban_magnitude[row_map_list[r_idx]:row_map_list[r_idx]+64, col_map_list[c_idx]:col_map_list[c_idx]+64]
        
        # from blockattribute import find_Attribute_multi
        # from orientation_estimation import local_multipeak
        # from grouping_framework import map_clustering_peak_threshold
        # ---- Find 5 peak
        peak_pos = local_multipeak(block, radius_ban=2, max_peak_count=np.inf)
        cluster = map_clustering_peak_threshold(block, peak_pos, 0.7)
        
        clst_map = np.zeros_like(cluster[0])
        # ---- Change for cluster position
        
        for i in range(start, stop):
        # for i in range(len(cluster)):
            clst_map+= cluster[i]
        plot_all([item for item in cluster])
        plt.subplot(1, 3, 1)
        plt.title("Ban Magnitude")
        plt.imshow(block, cmap='hot')
        plt.subplot(1 ,3, 2)
        plt.title("Cluster Map")
        plt.imshow(clst_map, cmap='gray')
        plt.subplot(1, 3, 3)
        plt.title("Cluster Overlay")
        plt.imshow(block, cmap='hot')
        plt.imshow(clst_map, alpha=0.5, cmap='gray')
        plt.show()
        
        
        BBF.setMagnitude(output_magnitude)
        BBF.istft()
        output = BBF.get_output_img()
        plot_all([img, output])
        # plt.imshow(img, alpha=0.5, cmap='gray')
        # plt.imshow(output, cmap='gray')
        # plt.show()
        
        
        
        
        
        # # ---- Watershed Clustering
        # bg_list = map_clustering_watershed(block, 4)
        # # bg_list = map_clustering_n_iterative(block, 0.8)
        # print(f"The number of cluster is : {len(bg_list)}")
        # klst = [bg_list[i].generate_activation_map(block.shape[0], block.shape[1]) for i in range(len(bg_list))]
        # # klst.insert(0, block)
        # # plot_all(block, cmap='hot')
        # fig1 = go.Figure(data=[go.Surface(z=block, colorscale='Jet')])
        # fig1.update_layout(
        #     title='1. 3D FFT Magnitude Spectrum',
        #     scene=dict(xaxis_title='Freq X', yaxis_title='Freq Y', zaxis_title='Magnitude'),
        # )
        # fig1.show()
        # for i in range(len(klst)):
        #     plot_all([block, klst[i]], cmap='hot', title_list=["Original", f"cluster {i}"])
        #     # ---- ---- ----