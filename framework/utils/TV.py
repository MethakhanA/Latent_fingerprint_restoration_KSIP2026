import cv2 as cv
import numpy as np
from skimage.restoration import denoise_tv_chambolle
import matplotlib.pyplot as plt

def normalize(input_img):
    norm_img = (255 * (input_img - input_img.min()) / (input_img.max() - input_img.min())).astype(np.uint8)
    return norm_img

def TV_preprocessing(img):
    input_img_float = img.astype(np.float64)
    cartoon_img = denoise_tv_chambolle(input_img_float, weight=40)
    texture_img = input_img_float - cartoon_img.astype(np.float64)
    texture_img = normalize(texture_img).astype(np.uint8)
    return texture_img
    
if __name__ == "__main__":
    from glob import glob
    import os
    from tqdm import tqdm
    path = r"C:\work\image_processing\latent_fingerprint\Segmentation\data\input\sd302h_500ppi_original_mate"
    output_path = r"C:\work\image_processing\latent_fingerprint\model\data\TV_img"
    for file in tqdm(glob(os.path.join(path, '*'))):
        img = cv.imread(file)
        out = TV_preprocessing(img)
        cv.imwrite(os.path.join(output_path, os.path.basename(file)), out)