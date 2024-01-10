import numpy as np
import torch
from torch import nn
import utils
import cv2
import os

# --------------------------------------------
# generate motion blurred image
# --------------------------------------------
def motion_blur(image, degree=10, angle=20):
    image = np.array(image)
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    blurred = np.array(blurred)
    return blurred



if __name__ == '__main__':
    gt_root = '/data/liujiaxin/data/SUFDD/Microtubule/Train/Raw'
    gt_paths = utils.get_image_paths(gt_root)
    degree = 5
    angle = 90
    for gt_path in gt_paths:
        gt_image = cv2.imread(gt_path, -1)
        blurred_image = motion_blur(gt_image, degree, angle)
        blur_path = gt_path.replace('Raw', 'MBD'+str(degree)+'A'+str(angle))
        blur_path = blur_path.replace('Raw', 'MBD'+str(degree)+'A'+str(angle))
        utils.mkdirs(os.path.dirname(blur_path))
        utils.tifsave(blurred_image, blur_path)