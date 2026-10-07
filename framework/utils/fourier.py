import numpy as np
from scipy import fftpack
from scipy.signal.windows import gaussian
import cv2 as cv
import matplotlib.pyplot as plt


def log_transform(image:np.ndarray, c=None, to_uint8=True):
    # preventing overflow
    image = image.astype(np.float32)
    if c is None:
        c = 255/np.log(1+np.max(image))
    # transform
    out_image = c*np.log(1+image)
    # turn to uint8 if selected
    if to_uint8:
        return out_image.astype(np.uint8)
    else:
        return out_image
class Fourier2D:
    def __init__(self, input_img, zero_mean=False, window_func=None, frequency=False):
        self.__input_img = input_img
        self.__zero_mean = zero_mean
        self.__window_func = window_func
        self.__img_height = input_img.shape[0]
        self.__img_width = input_img.shape[1]
        if frequency:
            self.setMagnitude(input_img)
            self.__windowing(input_img)
    def __zeroMean(self):
        '''
        Make the picture be zero mean

        '''
        if self.__zero_mean:
            avg = self.__input_img.mean()
            output_img = self.__input_img -avg
            self.avg = avg
        else:
            output_img = self.__input_img.copy()
        return output_img
    def __windowing(self, input_img):
        if self.__window_func is None:
            output_img = input_img
        elif self.__window_func == "Gaussian":
            gauss_std = lambda k_size : 0.3*((k_size-1)*0.5)+0.8
            
            std_vert = gauss_std(self.__img_height)
            std_horz = gauss_std(self.__img_width)
            
            winfunc_vert = gaussian(self.__img_height, std_vert).reshape((1, -1))
            winfunc_horz = gaussian(self.__img_width, std_horz).reshape((1, -1))
            self.__window = winfunc_vert * winfunc_horz.T
            output_img = input_img * self.__window.T
        else:
            window_func = self.__window_func(input_img) # function that return image size window function
            output_img = input_img*window_func
        return output_img
            
    def fft(self):
        '''
        transform and shift the center
        '''
        # zero mean
        preproc_img = self.__zeroMean() # do function zero mean on self
        preproc_img = self.__windowing(preproc_img)
        
        
        # do the fast fourier transform
        fft_complex = fftpack.fft2(preproc_img)
        # split transform into magnitude and phase
        self.__fft_magnitude = np.abs(fft_complex)
        self.__fft_phase = np.arctan2(fft_complex.imag, fft_complex.real)
        # shift the quadrant
        self.__fft_magnitude = fftpack.fftshift(self.__fft_magnitude)
    def ifft(self):
        '''
        inverse transform and shift the center back
        '''
        # For inverse window function
            # self.__fft_magnitude = self.__fft_magnitude*self.__inv_window
            
        # Invert shift Magnitude plot
        ifft_magnitude = fftpack.ifftshift(self.__fft_magnitude)
        
        # Combine magnitude and phase
        ifft_real = ifft_magnitude*np.cos(self.__fft_phase)
        ifft_imag = ifft_magnitude*np.sin(self.__fft_phase)
        # combine
        ifft_complex = ifft_real+(ifft_imag*1j)
        # invert fft
        output_complex = fftpack.ifft2(ifft_complex)
        # Get image data from the real part(Imag is a false part)
        self.__output_img = output_complex.real
        if not self.__window_func is None:
            self.__inv_window = 1/self.__window
            self.__output_img = self.__output_img*self.__inv_window
    def getOutputImg(self):
        '''
        Get the image out of the objects
        '''
        return self.__output_img
    def showMagnitude(self, log_scale=False):
        '''
            Plot the magnitude of FFT
        '''
        v_magnitude = self.__fft_magnitude.copy()
        if log_scale:
            v_magnitude = log_transform(v_magnitude, to_uint8=True)
        plt.figure()
        plt.imshow(v_magnitude, cmap='hot')
    def getMagnitude(self):
        '''
        Get the magnitude fft out of the object
        '''
        # get the magnitude of itself
        return self.__fft_magnitude
    def getPhase(self):
        return self.__fft_phase
    def setMagnitude(self, fft_magnitude):
        '''
        Set the magnitude of picture to something else?
        '''
        # set the magnitude to something else
        self.__fft_magnitude = fft_magnitude
    def setPhase(self, fft_phase):
        self.__fft_phase = fft_phase