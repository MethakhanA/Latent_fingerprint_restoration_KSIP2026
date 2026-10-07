import numpy as np
import cv2 as cv
def blurEdge(finger_img, gt_img, blur_size=21, erode_size=21):
    mask_bin = (gt_img > 0).astype(np.uint8)
    stre = cv.getStructuringElement(cv.MORPH_ELLIPSE, (erode_size, erode_size))
    mask_eroded = cv.erode(mask_bin, stre)
    mean_val = int(cv.mean(finger_img, mask=mask_eroded)[0])
    mask_f = mask_eroded.astype(np.float32)
    alpha = cv.GaussianBlur(mask_f, (blur_size, blur_size), 0)
    output = (alpha * finger_img) + ((1.0 - alpha) * mean_val)
    return output.astype(np.uint8)