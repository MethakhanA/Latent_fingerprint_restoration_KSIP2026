import numpy as np
class FreqFilter:
    def __init__(self, filter_shape):
        self.__height = filter_shape[0]
        self.__width = filter_shape[1]
    def __distanceMap(self, center_pos=None, scale=True):
        # -> Preset Position Index
        v = np.arange(0, self.__height)
        u = np.arange(0, self.__width)
        # -> Create Mesh Grid
        uv, vv = np.meshgrid(u, v)
        # -> Target Position
        if center_pos is None:
            center_u, center_v = (self.__width // 2, self.__height // 2)
        else:
            center_u, center_v = center_pos
        # -> Scaling Factor
        if scale:
            scale_u = self.__width / max(self.__width, self.__height)
            scale_v = self.__height / max(self.__width, self.__height)
        else:
            scale_u, scale_v = 1, 1
        # -> Distance Map (Euclidean)
        self.__distance_map = np.sqrt(((uv-center_u)/scale_u)**2 + ((vv-center_v)/scale_v)**2)
    def __LPF(self, freq_cutoff, f_type="Ideal", n_order=2):
        if f_type == "Ideal":
            self.__filter = np.where(self.__distance_map <= freq_cutoff, 1, 0)
        elif f_type == "Gaussian":
            self.__filter = np.exp(-self.__distance_map**2 / (2*freq_cutoff**2))
        elif f_type == "Butterworth":
            self.__filter = 1 / (1 + (self.__distance_map/freq_cutoff)**(2*n_order))
    def getLPF(self, freq_cutoff, f_type="Ideal", n_order=2, center_pos=None, scale=True):
        self.__distanceMap(center_pos, scale)
        self.__LPF(freq_cutoff, f_type, n_order)
        return self.__filter
    def idealFunction(distance_map, freq_cutoff):
        ideal_func = np.where(distance_map <= freq_cutoff, 1, 0)
        return ideal_func
    def gaussianFunction(distance_map, freq_cutoff):
        gauss_func = np.exp(-distance_map**2 / (2*freq_cutoff**2))
        return gauss_func
    def butterworthFunction(distance_map, freq_cutoff, n_order):
        bw_func = 1 / (1 + (distance_map/freq_cutoff)**(2*n_order))
        return bw_func
    def getHPF(self, freq_cutoff, f_type="Ideal", n_order=2, center_pos=None, scale=True):
        lowpass_filter = self.getLPF(freq_cutoff, f_type, n_order, center_pos, scale)
        self.__filter = 1 - lowpass_filter
        return self.__filter
    def __laplacian(self):
        self.__filter = -4 * (np.pi**2) * (self.__distance_map**2)
        return self.__filter
    def getLaplacian(self):
        self.__distanceMap()
        self.__laplacian()
        return self.__filter
    def __BPF(self, band_center, band_width, f_type="Ideal", n_order=2):
        eps = 1e-6
        if f_type == "Ideal":
            self.__filter = np.where((self.__distance_map >= (band_center-band_width/2))&(self.__distance_map <= (band_center+band_width/2)), 1, 0)
        elif f_type == "Gaussian":
            self.__filter = np.exp(-((self.__distance_map**2-band_center**2) / ((self.__distance_map*band_width)+eps))**2)
        elif f_type == "Butterworth":
            self.__filter = 1/(1 + ((self.__distance_map**2-band_center**2) / ((self.__distance_map*band_width)+eps))**(2*n_order))
    def getBPF(self, band_center=None, band_width=None, f_type="Ideal", n_order=2, radius1=None, radius2=None):
        self.__distanceMap()
        if band_center is None or band_width is None:
            band_width = radius2-radius1
            band_center = radius1+(band_width/2)
        self.__BPF(band_center, band_width, f_type, n_order)
        # bandpass_filter = self.getBPF(band_center, band_width, f_type, n_order)
        # self.__filter = 1 - bandpass_filter
        return self.__filter
    def getSelectiveFilter(self, pos_list, radius_list, pass_filter=False, f_type="Ideal", n_order=2):
        # -> Check if pos_list and radius_list are in List form
        assert isinstance(radius_list, list), "\"radius_list\" must be in list form, e.g., [r1, r2, ...]."
        assert isinstance(pos_list, list), "\"pos_list\" must be in list form, e.g., [(pos_v, pos_u), ...]."
        # -> Empty selective filter
        select_filter = np.ones((self.__height, self.__width))
        # -> Create each notch filter
        for pos, r in zip(pos_list, radius_list):
            # -> Create HPF (notch)
            hp_filter = self.getHPF(r, f_type, n_order, center_pos=pos, scale=False)
            # -> Find mirrored position
            mpos_v = self.__height - pos[0] - (1 if self.__height % 2 != 0 else 0)
            mpos_u = self.__width - pos[1] - (1 if self.__width % 2 != 0 else 0)
            mirror_pos = (mpos_v, mpos_u)
            # -> Create HPF for mirrored position
            hp_mfilter = self.getHPF(r, f_type, n_order, center_pos=mirror_pos, scale=False)
            # -> Merge Filter
            select_filter = select_filter * hp_filter * hp_mfilter
        if pass_filter:
            select_filter = 1 - select_filter
        return select_filter
