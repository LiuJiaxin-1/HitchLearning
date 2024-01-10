from email import utils
import torch.utils.data as data
from torchvision import transforms
import random
import utils.utils_image as util


class DatasetL(data.Dataset):
    '''
    # -----------------------------------------
    # Get L for unsupervised task.
    # Only "dataroot_L" is needed.
    # -----------------------------------------
    # e.g., train Neighbor2Neighbor-denoising
    # -----------------------------------------
    '''

    def __init__(self, opt, select_img_nums=0, dataseed=0):
        super(DatasetL, self).__init__()
        print('Read L for unsupervised task. Only "dataroot_L" is needed.')
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = opt['patch_size']

        # ------------------------------------
        # get the path of L
        # ------------------------------------
        dataroot_l = opt['dataroot_L']
        print(f'get Low images from {dataroot_l}.')
        self.paths_L = util.get_image_paths(opt['dataroot_L'])
        assert self.paths_L, 'Error: L paths are empty.'
        # ------------------------------------
        # if using part of dataset, get the assigned imgs.
        # ------------------------------------
        if select_img_nums:
            random.seed(dataseed)
            img_ids = random.sample(range(0, len(self.paths_L)), select_img_nums)
            self.paths_L = self.paths_L[img_ids]
        
        if self.opt['phase'] != 'train':
            self.paths_L.sort()


    def __getitem__(self, index):
        L_path = None

        # ------------------------------------
        # get L image
        # ------------------------------------
        L_path = self.paths_L[index]
        img_L = util.read_tif(L_path)
        # ------------------------------------
        # preproess image: background elimination and nrmalization
        # ------------------------------------
        img_L = util.process(img_L, self.opt['eliback'], self.opt['prctile'])
        
        # ------------------------------------
        # if train, get L patch
        # ------------------------------------
        if self.opt['phase'] == 'train':
            img_L = util.get_img_patch(img_L, self.patch_size)

        # ------------------------------------
        # HWC to CHW, numpy to tensor
        # ------------------------------------
        transformer = transforms.Compose([transforms.ToTensor()])
        img_L = transformer(img_L)

        return {'L': img_L, 'L_path': L_path}

    def __len__(self):
        return len(self.paths_L)

