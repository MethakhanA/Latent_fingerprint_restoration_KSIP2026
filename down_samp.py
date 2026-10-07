


import os
from glob import glob
from tqdm import tqdm
import cv2 as cv

if __name__ == "__main__":
    path = r"data/U_roll"
    out_path = r"C:\work\image_processing\latent_fingerprint\model\data\U_roll_500"
    for file in tqdm(glob(os.path.join(path, '*'))):
        img = cv.imread(file, 0)
        img = cv.pyrDown(img)
        cv.imwrite(os.path.join(out_path, os.path.basename(file)), img)
