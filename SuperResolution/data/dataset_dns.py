from email import utils
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
import random
import utils.utils_image as util


class DatasetDNS(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H for supervised task.
    # If only "paths_H" is provided, sythesize L on-the-fly.
    # If sythesize L, train and test useing the same noise type
    # -----------------------------------------
    # e.g., train SwinIR-denoising, test Neighbor2Neighbor-denoising
    # -----------------------------------------
    '''

    def __init__(self, opt, select_img_nums=0, dataseed=0):
        super(DatasetDNS, self).__init__()
        print('Read L/H for supervised denoising task.')
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = opt['patch_size'] if self.opt['patch_size'] else 256
        self.sigma = opt['sigma'] if opt['sigma'] else 0.1

        # ------------------------------------
        # get the path of L/H and SL/SH
        # ------------------------------------
        dataroot_l = opt['dataroot_L']
        dataroot_h = opt['dataroot_H']
        dataroot_sl = opt['dataroot_SL']
        dataroot_sh = opt['dataroot_SH']
        print(f'get Low images from {dataroot_l}.')
        print(f'get High images from {dataroot_h}.')
        print(f'get source Low images from {dataroot_sl}.')
        print(f'get source High images from {dataroot_sh}.')
        self.paths_H = util.get_image_paths(opt['dataroot_H'])
        self.paths_L = util.get_image_paths(opt['dataroot_L'])
        self.paths_SH = util.get_image_paths(opt['dataroot_SH'])
        self.paths_SL = util.get_image_paths(opt['dataroot_SL'])
        
        assert self.paths_H, 'Error: H paths are empty.'
        assert self.paths_SH, 'Error: SH paths are empty.'
        if self.paths_L and self.paths_H:
            assert len(self.paths_L) == len(self.paths_H), 'L/H mismatch - {}, {}.'.format(len(self.paths_L), len(self.paths_H))
        if self.paths_SL and self.paths_SH:
            assert len(self.paths_SL) == len(self.paths_SH), 'L/H mismatch - {}, {}.'.format(len(self.paths_SL), len(self.paths_SH))
        
        # ------------------------------------
        # if using part of dataset, get the assigned imgs.
        # ------------------------------------
        if select_img_nums:
            random.seed(dataseed)
            img_ids = random.sample(range(0, len(self.paths_H)), select_img_nums)
            if self.paths_L:
                self.paths_L = self.paths_L[img_ids]
            self.paths_H = self.paths_H[img_ids]
        
        # ------------------------------------
        # if paths_L is not equal to paths_SL, make them equal.
        # ------------------------------------
        if len(self.paths_H) != len(self.paths_SH):
            self.paths_H *= int(np.ceil(float(len(self.paths_SH)) / len(self.paths_H)))
            self.paths_H.sort()
            ids = random.sample(range(0, len(self.paths_H)), len(self.paths_SH))
            self.paths_H = self.paths_H[ids]
            if self.paths_L:
                self.paths_L *= int(np.ceil(float(len(self.paths_SH)) / len(self.paths_L)))
                self.paths_L.sort()
                self.paths_L = self.paths_L[ids]


    def __getitem__(self, index):
        L_path = None
        SL_path = None
        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        SH_path = self.paths_SH[index]
        img_H = util.read_tif(H_path)
        img_SH = util.read_tif(SH_path)
        # ------------------------------------
        # preproess image: background elimination and nrmalization
        # ------------------------------------
        # img_H = util.process(img_H, self.opt['eliback'], self.opt['prctile'])
        # img_SH = util.process(img_SH, self.opt['Seliback'], self.opt['Sprctile'])
        img_H = util.process(img_H, False, self.opt['prctile'])
        img_SH = util.process(img_SH, False, self.opt['Sprctile'])
        # ------------------------------------
        # get L image
        # ------------------------------------
        if self.paths_L:
            L_path = self.paths_L[index]
            img_L = util.read_tif(L_path)
            # ------------------------------------
            # preproess image: background elimination and nrmalization
            # ------------------------------------
            img_L = util.process(img_L, self.opt['eliback'], self.opt['prctile'])
        else:
            # --------------------------------
            # add noise
            # --------------------------------
            img_L = img_H.clone()
            noise = torch.randn(img_L.size()).mul_(self.sigma)
            img_L.add_(noise)
        # ------------------------------------
        # get SL image
        # ------------------------------------
        if self.paths_SL:
            SL_path = self.paths_L[index]
            img_SL = util.read_tif(SL_path)
            # ------------------------------------
            # preproess image: background elimination and nrmalization
            # ------------------------------------
            img_SL = util.process(img_SL, self.opt['Seliback'], self.opt['Sprctile'])
        else:
            # --------------------------------
            # add noise
            # --------------------------------
            img_SL = img_SH.clone()
            noise = torch.randn(img_SL.size()).mul_(self.sigma)
            img_SL.add_(noise)

        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
        if self.opt['phase'] == 'train':
            img_L = util.get_img_patch(img_L, self.patch_size)
            img_H = util.get_img_patch(img_H, self.patch_size)
            img_SL = util.get_img_patch(img_SL, self.patch_size)
            img_SH = util.get_img_patch(img_SH, self.patch_size)
            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            mode = random.randint(0, 7)
            img_L, img_H = util.augment_img(img_L, mode=mode), util.augment_img(img_H, mode=mode)
            mode = random.randint(0, 7)
            img_SL, img_SH = util.augment_img(img_SL, mode=mode), util.augment_img(img_SH, mode=mode)

        # ------------------------------------
        # HWC to CHW, numpy to tensor
        # ------------------------------------
        transformer = transforms.Compose([transforms.ToTensor()])
        img_L, img_H = transformer(img_L), transformer(img_H)
        img_SL, img_SH = transformer(img_SL), transformer(img_SH)

        if L_path is None:
            L_path = H_path
        if SL_path is None:
            SL_path = SH_path

        return {'L': img_L, 'H': img_H, 'SL': img_SL, 'SH': img_SH, 
        'L_path': L_path, 'H_path': H_path, 'SL_path': SL_path, 'H_path': SH_path}

    def __len__(self):
        return len(self.paths_H)