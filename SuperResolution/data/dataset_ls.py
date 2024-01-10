from email import utils
import torch.utils.data as data
from torchvision import transforms
import numpy as np
import random
import utils.utils_image as util


class DatasetLS(data.Dataset):
    '''
    # -----------------------------------------
    # Get L for unsupervised task.
    # Only "dataroot_L" is needed.
    # -----------------------------------------
    # e.g., train Neighbor2Neighbor-denoising
    # -----------------------------------------
    '''

    def __init__(self, opt, select_img_nums=0, dataseed=0):
        super(DatasetLS, self).__init__()
        print('Read L and LS for unsupervisedtask. Only "dataroot_L" and "dataroot_LS" are needed.')
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = opt['patch_size']

        # ------------------------------------
        # get the path of L and SL
        # ------------------------------------
        dataroot_l = opt['dataroot_L']
        dataroot_sl = opt['dataroot_SL']
        print(f'get Low images from {dataroot_l} and {dataroot_sl}.')
        self.paths_L = util.get_image_paths(opt['dataroot_L'])
        self.paths_SL = util.get_image_paths(opt['dataroot_SL'])
        assert self.paths_L, 'Error: L paths are empty.'
        assert self.paths_SL, 'Error: SL paths are empty.'
        
        # ------------------------------------
        # if using part of dataset, get the assigned imgs.
        # ------------------------------------
        if select_img_nums:
            random.seed(dataseed)
            img_ids = random.sample(range(0, len(self.paths_L)), select_img_nums)
            self.paths_L = self.paths_L[img_ids]
        
        # ------------------------------------
        # if paths_L is not equal to paths_SL, make them equal.
        # ------------------------------------
        if len(self.paths_L) != len(self.paths_SL):
            self.paths_L *= int(np.ceil(float(len(self.paths_SL)) / len(self.paths_L)))
            random.shuffle(self.paths_L)
            self.paths_L = self.paths_L[:int(len(self.paths_SL))]


    def __getitem__(self, index):
        # ------------------------------------
        # get L and SL image
        # ------------------------------------
        L_path = self.paths_L[index]
        SL_path = self.paths_SL[index]
        img_L = util.read_tif(L_path)
        img_SL = util.read_tif(SL_path)
        # ------------------------------------
        # preproess image: background elimination and nrmalization
        # ------------------------------------
        img_L = util.process(img_L, self.opt['eliback'], self.opt['prctile'])
        img_SL = util.process(img_SL, self.opt['Seliback'], self.opt['Sprctile'])
        # ------------------------------------
        # if train, get L and SL patch.
        # ------------------------------------
        if self.opt['phase'] == 'train':
            img_L = util.get_img_patch(img_L, self.patch_size)
            img_SL = util.get_img_patch(img_SL, self.patch_size)

        # ------------------------------------
        # HWC to CHW, numpy to tensor
        # ------------------------------------
        transformer = transforms.Compose([transforms.ToTensor()])
        img_L = transformer(img_L)
        img_SL = transformer(img_SL)

        return {'L': img_L, 'SL': img_SL, 'L_path': L_path, 'SL_path': SL_path}

    def __len__(self):
        return len(self.paths_L)