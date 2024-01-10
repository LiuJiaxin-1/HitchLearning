from email import utils
import numpy as np
import torch.utils.data as data
from torchvision import transforms
import random
import utils.utils_image as util


class DatasetSR(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H for supervised task.
    # If only "paths_H" is provided, sythesize bicubicly downsampled L on-the-fly.
    # -----------------------------------------
    # e.g., train SwinIR-super-resolution, test SwinIR-super-resolution
    # -----------------------------------------
    '''

    def __init__(self, opt, select_img_nums=0, dataseed=0):
        super(DatasetSR, self).__init__()
        print('Read L/H for supervised super-resolution task.')
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.sf = opt['scale'] if opt['scale'] else 2
        self.patch_size = opt['patch_size'] if self.opt['patch_size'] else 256
        self.L_size = self.patch_size // self.sf

        # ------------------------------------
        # get the path of L and H
        # ------------------------------------
        dataroot_l = opt['dataroot_L']
        dataroot_h = opt['dataroot_H']
        print(f'get Low images from {dataroot_l}.')
        print(f'get High images from {dataroot_h}.')
        self.paths_H = util.get_image_paths(opt['dataroot_H'])
        self.paths_L = util.get_image_paths(opt['dataroot_L'])
        
        assert self.paths_H, 'Error: H paths are empty.'
        if self.paths_L and self.paths_H:
            assert len(self.paths_L) == len(self.paths_H), 'L/H mismatch - {}, {}.'.format(len(self.paths_L), len(self.paths_H))
        
        # ------------------------------------
        # if using part of dataset, get the assigned imgs.
        # ------------------------------------
        if select_img_nums:
            random.seed(dataseed)
            img_ids = random.sample(range(0, len(self.paths_H)), select_img_nums)
            if self.paths_L:
                self.paths_L = [self.paths_L[ids] for ids in img_ids]
                # self.paths_L = list(np.array(self.paths_L)[img_ids])
            self.paths_H = [self.paths_H[ids] for ids in img_ids]
            # self.paths_H = list(np.array(self.paths_H)[img_ids])
        
        if self.opt['phase'] != 'train':
            if self.paths_L:
                self.paths_L.sort()
            self.paths_H.sort()


    def __getitem__(self, index):
        L_path = None
        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = util.read_tif(H_path)
        # ------------------------------------
        # preproess image: background elimination and nrmalization
        # ------------------------------------
        # img_H = util.process(img_H, self.opt['eliback'], self.opt['prctile'])
        # img_H = util.preprocess(img_H, False, self.opt['prctile'])
        img_H = util.preprocess(img_H, False, False)

        # ------------------------------------
        # get L image
        # ------------------------------------
        if self.paths_L:
            L_path = self.paths_L[index]
            img_L = util.read_tif(L_path)
            # ------------------------------------
            # preproess image: background elimination and nrmalization
            # ------------------------------------
            img_L = util.preprocess(img_L, self.opt['eliback'], self.opt['prctile'])
        else:
            # --------------------------------
            # sythesize L image via matlab's bicubic
            # --------------------------------
            H, W = img_H.shape[:2]
            img_L = util.imresize_np(img_H, 1 / self.sf, True)

        # ------------------------------------
        # if train, get L patch
        # ------------------------------------
        if self.opt['phase'] == 'train':
            H, W, C = img_L.shape
        
            # --------------------------------
            # randomly crop the L patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.L_size))
            rnd_w = random.randint(0, max(0, W - self.L_size))
            img_L = img_L[rnd_h:rnd_h + self.L_size, rnd_w:rnd_w + self.L_size, :]

            # --------------------------------
            # crop corresponding H patch
            # --------------------------------
            rnd_h_H, rnd_w_H = int(rnd_h * self.sf), int(rnd_w * self.sf)
            img_H = img_H[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            mode = random.randint(0, 7)
            img_L, img_H = util.augment_img(img_L, mode=mode), util.augment_img(img_H, mode=mode)

        # ------------------------------------
        # HWC to CHW, numpy to tensor
        # ------------------------------------
        # transformer = transforms.Compose([transforms.ToTensor()])
        img_L, img_H = util.single2tensor3(img_L), util.single2tensor3(img_H)

        if L_path is None:
            L_path = H_path

        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path, 'index':index}

    def __len__(self):
        return len(self.paths_H)