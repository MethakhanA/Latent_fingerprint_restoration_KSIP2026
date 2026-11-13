import numpy as np

def normalize_image(img, target_min=0, target_max=255, dtype=np.uint8):
    """
    Normalizes an image array from any input range to [target_min, target_max].
    
    Parameters:
        img (np.ndarray): Input image array (float, int, any range).
        target_min (float): Minimum value of output range.
        target_max (float): Maximum value of output range.
        dtype (np.dtype): Desired output data type (e.g., np.uint8, np.float32).
        
    Returns:
        np.ndarray: Normalized image array.
    """
    img_float = img.astype(np.float64)
    min_val = np.min(img_float)
    max_val = np.max(img_float)
    
    # Handle flat/constant images to prevent division by zero
    if max_val == min_val:
        return np.full_like(img, target_min, dtype=dtype)
    
    # Min-Max Normalization formula: (x - min) / (max - min) * (target_max - target_min) + target_min
    normalized = (img_float - min_val) / (max_val - min_val) * (target_max - target_min) + target_min
    
    return normalized.astype(dtype)