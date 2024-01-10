import os
import numpy as np
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from pdb import set_trace as stx
import random
import utils
import random

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif', 'tif'])

def Aug(inp_img, tar_img, aug):
    if aug==1:
            inp_img = inp_img.flip(1)
            tar_img = tar_img.flip(1)
    elif aug==2:
        inp_img = inp_img.flip(2)
        tar_img = tar_img.flip(2)
    elif aug==3:
        inp_img = torch.rot90(inp_img,dims=(1,2))
        tar_img = torch.rot90(tar_img,dims=(1,2))
    elif aug==4:
        inp_img = torch.rot90(inp_img,dims=(1,2), k=2)
        tar_img = torch.rot90(tar_img,dims=(1,2), k=2)
    elif aug==5:
        inp_img = torch.rot90(inp_img,dims=(1,2), k=3)
        tar_img = torch.rot90(tar_img,dims=(1,2), k=3)
    elif aug==6:
        inp_img = torch.rot90(inp_img.flip(1),dims=(1,2))
        tar_img = torch.rot90(tar_img.flip(1),dims=(1,2))
    elif aug==7:
        inp_img = torch.rot90(inp_img.flip(2),dims=(1,2))
        tar_img = torch.rot90(tar_img.flip(2),dims=(1,2))
    return inp_img, tar_img


class DataLoaderTrain(Dataset):
    def __init__(self, data_dir, raw, gt, img_options=None, nums=0):
        super(DataLoaderTrain, self).__init__()

        raw_dir = data_dir
        gt__dir = data_dir.replace(raw, gt)
        self.inp_filenames = utils.get_image_paths(raw_dir)
        self.tar_filenames = utils.get_image_paths(gt__dir)
        self.sizex         = len(self.inp_filenames)  # get the size of input
        
        if nums:
            random.seed(nums)
            img_ids = random.sample(range(0, self.sizex), nums)
            self.inp_filenames = [self.inp_filenames[ids] for ids in img_ids]
            self.tar_filenames = [self.tar_filenames[ids] for ids in img_ids]
            self.sizex         = len(self.inp_filenames)

        self.img_options = img_options

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = utils.read_tif(inp_path)
        tar_img = utils.read_tif(tar_path)
        inp_img = utils.preprocess(inp_img, True, True)
        tar_img = utils.preprocess(tar_img, False, True)

        w,h,_ = tar_img.shape
        padw = ps-w if w<ps else 0
        padh = ps-h if h<ps else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw!=0 or padh!=0:
            inp_img = TF.pad(inp_img, (0,0,padw,padh), padding_mode='reflect')
            tar_img = TF.pad(tar_img, (0,0,padw,padh), padding_mode='reflect')

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        hh, ww = tar_img.shape[1], tar_img.shape[2]

        rr     = random.randint(0, hh-ps)
        cc     = random.randint(0, ww-ps)
        aug    = random.randint(0, 8)

        # Crop patch
        inp_img = inp_img[:, rr:rr+ps, cc:cc+ps]
        tar_img = tar_img[:, rr:rr+ps, cc:cc+ps]

        # Data Augmentations
        inp_img, tar_img = Aug(inp_img, tar_img, aug)
        
        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return filename, tar_img, inp_img


class DataLoaderSrcTrg(Dataset):
    def __init__(self, src_dir, trg_dir, sraw, sgt, traw, tgt, img_options=None):
        super(DataLoaderSrcTrg, self).__init__()

        # -----------------------
        # get src and trg image paths
        # -----------------------
        raw_src_dir = src_dir
        gt__src_dir = src_dir.replace(sraw, sgt)
        self.inp_src_filenames = utils.get_image_paths(raw_src_dir)
        self.tar_src_filenames = utils.get_image_paths(gt__src_dir)
        raw_trg_dir = trg_dir
        gt__trg_dir = trg_dir.replace(traw, tgt)
        self.inp_trg_filenames = utils.get_image_paths(raw_trg_dir)
        self.tar_trg_filenames = utils.get_image_paths(gt__trg_dir)

        self.img_options = img_options
        self.src_sizex       = len(self.inp_src_filenames)  # get the size of src input
        self.trg_sizex       = len(self.inp_trg_filenames)  # get the size of trg input

        # -----------------------
        # copy trg images making trg images equal to src images 
        # -----------------------
        self.inp_trg_filenames = sorted(self.inp_trg_filenames * int(np.ceil(self.src_sizex / self.trg_sizex)))
        self.tar_trg_filenames = sorted(self.tar_trg_filenames * int(np.ceil(self.src_sizex / self.trg_sizex)))
        if len(self.inp_trg_filenames) > self.src_sizex:
            self.inp_trg_filenames = self.inp_trg_filenames[:self.src_sizex]
            self.tar_trg_filenames = self.tar_trg_filenames[:self.src_sizex]

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.src_sizex

    def __getitem__(self, index):
        index_ = index % self.src_sizex
        ps = self.ps

        inp_src_path = self.inp_src_filenames[index_]
        tar_src_path = self.tar_src_filenames[index_]
        inp_trg_path = self.inp_trg_filenames[index_]
        tar_trg_path = self.tar_trg_filenames[index_]

        inp_src_img = utils.read_tif(inp_src_path)
        tar_src_img = utils.read_tif(tar_src_path)
        inp_trg_img = utils.read_tif(inp_trg_path)
        tar_trg_img = utils.read_tif(tar_trg_path)

        inp_src_img = utils.preprocess(inp_src_img, True, True)
        tar_src_img = utils.preprocess(tar_src_img, False, True)
        inp_trg_img = utils.preprocess(inp_trg_img, True, True)
        tar_trg_img = utils.preprocess(tar_trg_img, False, True)

        sw,sh,_ = tar_src_img.shape
        tw,th,_ = tar_trg_img.shape

        spadw = ps-sw if sw<ps else 0
        spadh = ps-sh if sh<ps else 0
        tpadw = ps-tw if tw<ps else 0
        tpadh = ps-th if th<ps else 0

        # Reflect Pad in case image is smaller than patch_size
        if spadw!=0 or spadh!=0:
            inp_src_img = TF.pad(inp_src_img, (0,0,spadw,spadh), padding_mode='reflect')
            tar_src_img = TF.pad(tar_src_img, (0,0,spadw,spadh), padding_mode='reflect')
        if tpadw!=0 or tpadh!=0:
            inp_trg_img = TF.pad(inp_trg_img, (0,0,tpadw,tpadh), padding_mode='reflect')
            tar_trg_img = TF.pad(tar_trg_img, (0,0,tpadw,tpadh), padding_mode='reflect')

        inp_src_img = TF.to_tensor(inp_src_img)
        tar_src_img = TF.to_tensor(tar_src_img)
        inp_trg_img = TF.to_tensor(inp_trg_img)
        tar_trg_img = TF.to_tensor(tar_trg_img)

        shh, sww = tar_src_img.shape[1], tar_src_img.shape[2]
        thh, tww = tar_trg_img.shape[1], tar_trg_img.shape[2]

        srr     = random.randint(0, shh-ps)
        scc     = random.randint(0, sww-ps)
        trr     = random.randint(0, thh-ps)
        tcc     = random.randint(0, tww-ps)
        saug    = random.randint(0, 8)
        taug    = random.randint(0, 8)

        # Crop patch
        inp_src_img = inp_src_img[:, srr:srr+ps, scc:scc+ps]
        tar_src_img = tar_src_img[:, srr:srr+ps, scc:scc+ps]
        inp_trg_img = inp_trg_img[:, trr:trr+ps, tcc:tcc+ps]
        tar_trg_img = tar_trg_img[:, trr:trr+ps, tcc:tcc+ps]

        # Data Augmentations
        inp_src_img, tar_src_img = Aug(inp_src_img, tar_src_img, saug)
        inp_trg_img, tar_trg_img = Aug(inp_trg_img, tar_trg_img, taug)
        
        filename = os.path.splitext(os.path.split(tar_trg_path)[-1])[0]

        return filename, tar_src_img, inp_src_img, tar_trg_img, inp_trg_img


class DataLoaderVal(Dataset):
    def __init__(self, data_dir, raw, gt, img_options=None):
        super(DataLoaderVal, self).__init__()

        raw_dir = data_dir
        gt__dir = data_dir.replace(raw, gt)
        self.inp_filenames = utils.get_image_paths(raw_dir)
        self.tar_filenames = utils.get_image_paths(gt__dir)

        self.img_options = img_options
        self.sizex       = len(self.tar_filenames)  # get the size of target

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = utils.read_tif(inp_path)
        tar_img = utils.read_tif(tar_path)
        inp_img = utils.preprocess(inp_img, True, True)
        tar_img = utils.preprocess(tar_img, False, True)

        # Validate on center crop
        if self.ps is not None:
            inp_img = TF.center_crop(inp_img, (ps,ps))
            tar_img = TF.center_crop(tar_img, (ps,ps))

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return filename, tar_img, inp_img

class DataLoaderTest(Dataset):
    def __init__(self, inp_dir, img_options):
        super(DataLoaderTest, self).__init__()

        inp_files = sorted(os.listdir(inp_dir))
        self.inp_filenames = [os.path.join(inp_dir, x) for x in inp_files if is_image_file(x)]

        self.inp_size = len(self.inp_filenames)
        self.img_options = img_options

    def __len__(self):
        return self.inp_size

    def __getitem__(self, index):

        path_inp = self.inp_filenames[index]
        filename = os.path.splitext(os.path.split(path_inp)[-1])[0]
        inp = Image.open(path_inp)

        inp = TF.to_tensor(inp)
        return inp, filename
