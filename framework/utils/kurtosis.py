import numpy as np
from scipy.stats import kurtosis

from framework.utils.freqfilter import FreqFilter

# Find Kurtosis
def fft_kurtosis(magnitude_block, lpf_radius=15, hpf_radius=3):
    """Kurtosis of a magnitude block, restricted to a ring band
    (excludes the DC peak in the center and far-out high-frequency noise).

    magnitude_block: 2D array from BlockBaseFrameWork.getMagnitude(), sliced
    to one overlap-sized block (as returned directly by the pipeline).
    """
    filter = FreqFilter(magnitude_block.shape)
    LPF = filter.getLPF(15, f_type="Ideal")
    HPF = filter.getHPF(3, f_type="Ideal")
    banded = magnitude_block * LPF * HPF

    values = banded[banded != 0]
    if values.size < 4:
        return 0.0

    k = kurtosis(values, fisher=False)
    if np.isnan(k):
        return 0.0
    return float(k)
