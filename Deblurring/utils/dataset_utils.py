import torch
import numpy as np
import tifffile as TIFF
import os

class MixUp_AUG:
    def __init__(self):
        self.dist = torch.distributions.beta.Beta(torch.tensor([0.6]), torch.tensor([0.6]))

    def aug(self, rgb_gt, rgb_noisy):
        bs = rgb_gt.size(0)
        indices = torch.randperm(bs)
        rgb_gt2 = rgb_gt[indices]
        rgb_noisy2 = rgb_noisy[indices]

        lam = self.dist.rsample((bs,1)).view(-1,1,1,1).cuda()

        rgb_gt    = lam * rgb_gt + (1-lam) * rgb_gt2
        rgb_noisy = lam * rgb_noisy + (1-lam) * rgb_noisy2

        return rgb_gt, rgb_noisy


# --------------------------------------------
# get single image of size HxWxn_channles (tif)
# --------------------------------------------
def read_tif(path):
    # read image by tifffile
    # return: Numpy float32, HWC
    img = TIFF.imread(path)  # read tif img
    img = np.array(img, dtype='float32').squeeze()
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


# --------------------------------------------
# save .tif image
# --------------------------------------------
def tifsave(img, img_path):
    img = np.squeeze(img)
    img = img.astype('float32')
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    TIFF.imwrite(img_path, img, dtype='float32')


# ------------------------------------
# image normalization and background elimination
# ------------------------------------
def preprocess(img, eliback, prctile):
    # background elimination
    if eliback:
        img = elimback(img)
    # normalization image
    if prctile:
        img =prctile_norm(img, 0.1, 99.9)
    else:
        img = prctile_norm(img)
    return img

# ------------------------------------
# make the value of pixel in img smaller than 100 to be 0.
# ------------------------------------
def elimback(x):
    if x.max() < 100:
        return np.array(x, dtype='float32')   
    else:
        x = x - 100.0
        y = np.where(x > 0, x, 0)
        return np.array(y, dtype='float32')

# ------------------------------------
# normalize the picture
# ------------------------------------
def prctile_norm(img, min_prc=0, max_prc=100):
    img_norm = np.zeros(img.shape, dtype='float32')
    imgshape = img.shape
    if len(imgshape) == 3:
        for i in range(img.shape[2]):
            x = img[..., i]
            y = (x - np.percentile(x, min_prc)) / (np.percentile(x, max_prc) - np.percentile(x, min_prc) + 1e-7)
            y[y > 1] = 1
            y[y < 0] = 0
            img_norm[..., i] = y
    else:
        y = (img - np.percentile(img, min_prc)) / (np.percentile(img, max_prc) - np.percentile(img, min_prc) + 1e-7)
        y[y > 1] = 1
        y[y < 0] = 0
        img_norm = y
    return np.array(img_norm, dtype='float32')



# ------------------------------------
# write csv file
# ------------------------------------
def write_lines(file, info):
    with open(file, 'a') as f:
        f.writelines(info)


# ------------------------------------
# convert 2/3/4-dimensional torch tensor to numpy
# ------------------------------------
def tensor2numpy(img):
    # img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    img = img.data.float().cpu().numpy()
    # img = img.data.squeeze().cpu().numpy()
    if img.ndim == 4:
        img = np.transpose(img, (0, 2, 3, 1))
    return img.squeeze()


'''
# --------------------------------------------
# get image pathes
# --------------------------------------------
'''

IMG_EXTENSIONS = ['.tif', '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
    
def get_image_paths(dataroot):
    paths = None  # return None if dataroot is None
    if isinstance(dataroot, str):
        if not os.path.isdir(dataroot):
            if is_image_file(dataroot):
                paths = [dataroot]
        else:
            paths = sorted(_get_paths_from_images(dataroot))
    elif isinstance(dataroot, list):
        paths = []
        for i in dataroot:
            paths += sorted(_get_paths_from_images(i))
    return paths


def _get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images