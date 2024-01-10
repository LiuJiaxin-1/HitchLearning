from email import utils
import numpy as np
import torch.utils.data as data
from torchvision import transforms
import random
import utils.utils_image as util


class DatasetSRS(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H for supervised task.
    # If only "paths_H" is provided, sythesize bicubicly downsampled L on-the-fly.
    # -----------------------------------------
    # e.g., train SwinIR-super-resolution, test SwinIR-super-resolution
    # -----------------------------------------
    '''

    def __init__(self, opt, select_img_nums=0, dataseed=0):
        super(DatasetSRS, self).__init__()
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
                self.paths_L = [self.paths_L[ids] for ids in img_ids]
                # self.paths_L = list(np.array(self.paths_L)[img_ids])
            self.paths_H = [self.paths_H[ids] for ids in img_ids]
            # self.paths_H = list(np.array(self.paths_H)[img_ids])
        
        # ------------------------------------
        # if paths_L is not equal to paths_SL, make them equal.
        # ------------------------------------
        if len(self.paths_H) != len(self.paths_SH):
            self.paths_H *= int(np.ceil(float(len(self.paths_SH)) / len(self.paths_H)))
            self.paths_H.sort()
            if self.paths_L:
                self.paths_L *= int(np.ceil(float(len(self.paths_SH)) / len(self.paths_L)))
                self.paths_L.sort()
            ids = random.sample(range(0, len(self.paths_H)), len(self.paths_SH))
            if len(self.paths_H) > len(self.paths_SH):
                self.paths_H = [self.paths_H[id] for id in ids]
                if self.paths_L:
                    self.paths_L = [self.paths_L[id] for id in ids]


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
        # img_H = util.preprocess(img_H, self.opt['eliback'], self.opt['prctile'])
        # img_H = util.preprocess(img_H, False, self.opt['prctile'])
        # img_SH = util.preprocess(img_SH, False, self.opt['Sprctile'])
        img_H = util.preprocess(img_H, False,False)
        img_SH = util.preprocess(img_SH, False, False)
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
        # get SL image
        # ------------------------------------
        if self.paths_SL:
            SL_path = self.paths_SL[index]
            img_SL = util.read_tif(SL_path)
            # ------------------------------------
            # preproess image: background elimination and nrmalization
            # ------------------------------------
            img_SL = util.preprocess(img_SL, self.opt['Seliback'], self.opt['Sprctile'])
        else:
            # --------------------------------
            # sythesize SL image via matlab's bicubic
            # --------------------------------
            H, W = img_H.shape[:2]
            img_SL = util.imresize_np(img_SH, 1 / self.sf, True)

        # ------------------------------------
        # if train, get L patch
        # ------------------------------------
        if self.opt['phase'] == 'train':        
            # --------------------------------
            # randomly crop the L patch
            # --------------------------------
            H, W, C = img_L.shape
            rnd_h = random.randint(0, max(0, H - self.L_size))
            rnd_w = random.randint(0, max(0, W - self.L_size))
            img_L = img_L[rnd_h:rnd_h + self.L_size, rnd_w:rnd_w + self.L_size, :]

            # --------------------------------
            # crop corresponding H patch
            # --------------------------------
            rnd_h_H, rnd_w_H = int(rnd_h * self.sf), int(rnd_w * self.sf)
            img_H = img_H[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

        
            # --------------------------------
            # randomly crop the SL patch
            # --------------------------------
            H, W, C = img_SL.shape
            rnd_h = random.randint(0, max(0, H - self.L_size))
            rnd_w = random.randint(0, max(0, W - self.L_size))
            img_SL = img_SL[rnd_h:rnd_h + self.L_size, rnd_w:rnd_w + self.L_size, :]

            # --------------------------------
            # crop corresponding SH patch
            # --------------------------------
            rnd_h_H, rnd_w_H = int(rnd_h * self.sf), int(rnd_w * self.sf)
            img_SH = img_SH[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

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
        # transformer = transforms.Compose([transforms.ToTensor()])
        # img_L, img_H = transformer(img_L), transformer(img_H)
        # img_SL, img_SH = transformer(img_SL), transformer(img_SH)
        img_L, img_H = util.single2tensor3(img_L), util.single2tensor3(img_H)
        img_SL, img_SH = util.single2tensor3(img_SL), util.single2tensor3(img_SH)

        if L_path is None:
            L_path = H_path
        if SL_path is None:
            SL_path = SH_path

        return {'L': img_L, 'H': img_H, 'SL': img_SL, 'SH': img_SH, 'index':index,
        'L_path': L_path, 'H_path': H_path, 'SL_path': SL_path, 'SH_path': SH_path}

    def __len__(self):
        return len(self.paths_H)