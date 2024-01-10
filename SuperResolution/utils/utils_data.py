import numpy as np
import torch
from torch import nn

def upsampling(x, scale_factor=2, up_mode='bilinear'):
    while len(x.shape) < 4:
        x = np.expand_dims(x, axis=0)
    x = torch.from_numpy(x)
    up = nn.Upsample(scale_factor=scale_factor, mode=up_mode, align_corners=True)
    x_up = up(x)
    x_up = x_up.detach().numpy().squeeze()
    return x_up

if __name__ == '__main__':
    import utils_image as util
    import tifffile as TIFF
    H_root = "/data/ljx/data/SUFDD/F_actin/9/Val/Noise"
    save_root = "/data/ljx/data/SUFDD/F_actin/9/Val/SudoSR"
    util.mkdir(save_root)
    path_H = util.get_image_paths(H_root)
    for path in path_H:
        img = util.read_tif(path)
        img = img.squeeze()
        path = path.replace('Noise', 'SudoSR')
        path = path.replace('Raw', 'SudoSR')
        img = upsampling(img)
        util.tifsave(img, path)
        