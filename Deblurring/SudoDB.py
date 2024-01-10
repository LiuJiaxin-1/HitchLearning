import os
from config import Config 
opt = Config('/home/liujiaxin/Program/MPRNet-DB/Deblurring/sudodb.yml')

gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import torch
torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as calculate_ssim
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import mean_squared_error as calculate_mse
from skimage.metrics import mean_absolute_error as calculate_mae

import random
import time
import numpy as np

import utils
from Datasets.data import get_training_data, get_validation_data
from model.MPRNet import MPRNet
import model.losses as losses
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from pdb import set_trace as stx

######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

start_epoch = 1
mode = opt.MODEL.MODE
session = opt.MODEL.SESSION

result_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'results', session)
model_dir  = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'models',  session)

utils.mkdir(result_dir)
utils.mkdir(model_dir)

train_dir = opt.TRAINING.TRAIN_DIR
val_dir   = opt.TRAINING.VAL_DIR
data_dir = train_dir
src_name = 'ER'
data_name = 'F_actin'
######### Model ###########
model_restoration = MPRNet(in_c=1, out_c=1, n_feat=96, scale_unetfeats=48, scale_orsnetfeats=32, num_cab=8, kernel_size=3, reduction=4, bias=False)
model_restoration.cuda()

device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
  print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")


new_lr = opt.OPTIM.LR_INITIAL

optimizer = optim.Adam(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999),eps=1e-8)


######### Scheduler ###########
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS-warmup_epochs, eta_min=opt.OPTIM.LR_MIN)
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()

######### Val ###########
path_chk_rest    = utils.get_last_path(model_dir, '_best.pth')
utils.load_checkpoint(model_restoration,path_chk_rest)


if len(device_ids)>1:
    model_restoration = nn.DataParallel(model_restoration, device_ids = device_ids)

######### Loss ###########
criterion_char = losses.CharbonnierLoss()
criterion_edge = losses.EdgeLoss()

######### DataLoaders ###########
val_dataset = get_validation_data(data_dir, 'MBD5A90', 'GT', {'patch_size':None})
val_loader = DataLoader(dataset=val_dataset, batch_size=24, shuffle=False, num_workers=48, drop_last=False, pin_memory=True)


best_psnr = 0
best_epoch = 0

#### Evaluation ####
model_restoration.eval()
avg_psnr = 0.0
avg_ssim = 0.0
avg_mse = 0.0
avg_mae = 0.0
save_dir = os.path.join(data_dir, 'SudoMDB')
utils.mkdir(save_dir)
for ii, data_val in enumerate((val_loader), 0):
    target = data_val[0].cuda()
    input_ = data_val[1].cuda()
    f_name = data_val[2]

    with torch.no_grad():
        restored = model_restoration(input_)
    restored = torch.clamp(restored[0],0,1)

    for res,tar,name in zip(restored,target,f_name):
        name = name.split('_', 1)[-1]
        res = utils.dataset_utils.tensor2numpy(res)
        tar = utils.dataset_utils.tensor2numpy(tar)
        sudo_db_img_path = os.path.join(save_dir, 'SudoMDB_' + name +'.tif')
        utils.dataset_utils.tifsave(res, sudo_db_img_path)
        image_metrics = os.path.join(data_dir, data_name + '_SudoMDB_' + src_name +'.csv')
        if not os.path.exists(image_metrics):
            utils.dataset_utils.write_lines(image_metrics, f"{'ImageName'},{'MSE'},{'MAE'},{'PSNR'},{'SSIM'}\n")
        # -----------------------
        # calculate PSNR, SSIM and MSE
        # -----------------------
        current_psnr = calculate_psnr(tar, res)
        current_ssim = calculate_ssim(tar, res)
        current_mse = calculate_mse(tar, res)
        current_mae = calculate_mae(tar, res)
        utils.dataset_utils.write_lines(image_metrics, 
        f"{name},{current_mse},{current_mae},{current_psnr},{current_ssim}\n")
        print('{:>10s} | PSNR:{:<4.2f}dB | SSIM:{:<4.4f} | MSE:{:<4.5f} | MAE:{:<4.5f}'
            .format(name, current_psnr, current_ssim, current_mse, current_mae))
        avg_psnr += current_psnr
        avg_ssim += current_ssim
        avg_mse += current_mse
    

