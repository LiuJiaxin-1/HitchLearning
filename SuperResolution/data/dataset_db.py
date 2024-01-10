from email import utils
import torch.utils.data as data
from torchvision import transforms
import random
import scipy
import scipy.ndimage
import utils.utils_image as util


class DatasetDB(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H for supervised task.
    # If only "paths_H" is provided, sythesize L on-the-fly.
    # If sythesize L, train and test useing the same blur method
    # -----------------------------------------
    # e.g., train supervised deblur network, test deblur network
    # -----------------------------------------
    '''

    def __init__(self, opt, select_img_nums=0, dataseed=0):
        super(DatasetDB, self).__init__()
        print('Get L/H for image-to-image mapping. Both "paths_L" and "paths_H" are needed.')
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = opt['patch_size'] if self.opt['patch_size'] else 256
        self.sigma = opt['sigma'] if opt['sigma'] else 1

        # ------------------------------------
        # get the path of L/H
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
                self.paths_L = self.paths_L[img_ids]
            self.paths_H = self.paths_H[img_ids]

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
        img_H = util.process(img_H, False, self.opt['prctile'])
        
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
            # sythesize L image via scipy
            # --------------------------------
            img_L = scipy.ndimage.gaussian_filter(img_H, sigma=self.sigma)

        # ------------------------------------
        # if train, get L patch
        # ------------------------------------
        if self.opt['phase'] == 'train':
            img_L = util.get_img_patch(img_L, self.patch_size)
            img_H = util.get_img_patch(img_H, self.patch_size)
            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            mode = random.randint(0, 7)
            img_L, img_H = util.augment_img(img_L, mode=mode), util.augment_img(img_H, mode=mode)

        # ------------------------------------
        # HWC to CHW, numpy to tensor
        # ------------------------------------
        transformer = transforms.Compose([transforms.ToTensor()])
        img_L, img_H = transformer(img_L), transformer(img_H)

        if L_path is None:
            L_path = H_path

        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)