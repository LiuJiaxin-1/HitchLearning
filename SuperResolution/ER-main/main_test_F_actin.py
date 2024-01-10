'''
Using the trained model to create Sudo SR for target datasets
'''

import os.path
import math
import argparse
from re import L
import time
import random
from tkinter import image_names
import numpy as np
from collections import OrderedDict
import logging
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from skimage.metrics import structural_similarity as calculate_ssim
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import mean_squared_error as calculate_mse
from skimage.metrics import mean_absolute_error as calculate_mae

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils import utils_csv as write_csv
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model


def main(image_name='F_actin', mode='Src', dataseed=0, json_path='./options/train_F-actin_x2_psnr.json'):
    def create_sudoSR(data_loader, current_step, mode, image_root, images_metrics):
        avg_psnr = 0.0
        avg_ssim = 0.0
        avg_mse = 0.0
        avg_mae = 0.0
        idx = 0
        for data in data_loader:
            model.feed_data(data=data, train_flag=False)
            model.test()
            visuals = model.current_results()
            L_imgs = util.tensor2numpy(visuals['L'])
            E_imgs = util.tensor2numpy(visuals['E'])
            H_imgs = util.tensor2numpy(visuals['H'])
            if len(E_imgs.shape) == 3:
                for image_path, E_img, H_img in zip(data['H_path'], E_imgs, H_imgs):
                    idx += 1
                    image_name_ext = os.path.basename(image_path)
                    img_name, ext = os.path.splitext(image_name_ext)
                    img_name = img_name.split('_', 1)[-1]
                    img_dir = os.path.join(image_root, img_name)
                    util.mkdir(img_dir)
                    save_E_img_path = os.path.join(img_dir, '{:s}_{:d}_{:s}_E{:s}'.format(img_name, current_step, mode, ext))
                    util.tifsave(E_img, save_E_img_path)
                    # -----------------------
                    # calculate PSNR, SSIM , MSE and MAE
                    # -----------------------
                    current_psnr = calculate_psnr(H_img, E_img)
                    current_ssim = calculate_ssim(H_img, E_img)
                    current_mse = calculate_mse(H_img, E_img)
                    current_mae = calculate_mae(H_img, E_img)
                    # current_psnr_norm = calculate_psnr(sr, util.prctile_norm(sudo_sr))
                    # current_ssim_norm = calculate_ssim(sr, util.prctile_norm(sudo_sr))
                    # current_mse_norm = calculate_mse(sr, util.prctile_norm(sudo_sr))
                    # current_mae_norm = calculate_mae(sr, util.prctile_norm(sudo_sr))
                    # write image_mode.csv
                    write_csv.write_lines(images_metrics, f"{img_name},{mode},{current_mse},{current_mae},{current_psnr},{current_ssim}\n")
                    # write_csv.write_lines(image_norm_metrics, f"{img_name},{current_mse_norm},{current_mae_norm},{current_psnr_norm},{current_ssim_norm}\n")
                    logger.info('{:->4d}--> {:>10s} | PSNR:{:<4.2f}dB | SSIM:{:<4.4f} | MSE:{:<4.5f} | MAE:{:<4.5f}'
                    .format(idx, img_name, current_psnr, current_ssim, current_mse, current_mae))
                    # logger.info('{:->4d}--> {:>10s} | N_PSNR:{:<4.2f}dB | N_SSIM:{:<4.4f} | N_MSE:{:<4.5f} | N_MAE:{:<4.5f}'
                    # .format(idx, img_name, current_psnr_norm, current_ssim_norm, current_mse_norm, current_mae_norm))
                    avg_psnr += current_psnr
                    avg_ssim += current_ssim
                    avg_mse += current_mse
                    avg_mae += current_mae

            elif len(E_imgs.shape) == 2:
                image_path = data['H_path'][0]
                E_img = E_imgs
                H_img = H_imgs
                idx += 1
                image_name_ext = os.path.basename(image_path)
                img_name, ext = os.path.splitext(image_name_ext)
                img_name = img_name.split('_', 1)[-1]
                img_dir = os.path.join(image_root, img_name)
                util.mkdir(img_dir)
                save_E_img_path = os.path.join(img_dir, '{:s}_{:d}_{:s}_E.{:s}'.format(img_name, current_step, mode, ext))
                util.tifsave(E_img, save_E_img_path)
                # -----------------------
                # calculate PSNR, SSIM , MSE and MAE
                # -----------------------
                current_psnr = calculate_psnr(H_img, E_img)
                current_ssim = calculate_ssim(H_img, E_img)
                current_mse = calculate_mse(H_img, E_img)
                current_mae = calculate_mae(H_img, E_img)
                # current_psnr_norm = calculate_psnr(sr, util.prctile_norm(sudo_sr))
                # current_ssim_norm = calculate_ssim(sr, util.prctile_norm(sudo_sr))
                # current_mse_norm = calculate_mse(sr, util.prctile_norm(sudo_sr))
                # current_mae_norm = calculate_mae(sr, util.prctile_norm(sudo_sr))
                # write image_mode.csv
                write_csv.write_lines(images_metrics, f"{img_name},{mode},{current_mse},{current_mae},{current_psnr},{current_ssim}\n")
                # write_csv.write_lines(image_norm_metrics, f"{img_name},{current_mse_norm},{current_mae_norm},{current_psnr_norm},{current_ssim_norm}\n")
                logger.info('{:->4d}--> {:>10s} | PSNR:{:<4.2f}dB | SSIM:{:<4.4f} | MSE:{:<4.5f} | MAE:{:<4.5f}'
                .format(idx, img_name, current_psnr, current_ssim, current_mse, current_mae))
                # logger.info('{:->4d}--> {:>10s} | N_PSNR:{:<4.2f}dB | N_SSIM:{:<4.4f} | N_MSE:{:<4.5f} | N_MAE:{:<4.5f}'
                # .format(idx, img_name, current_psnr_norm, current_ssim_norm, current_mse_norm, current_mae_norm))
                avg_psnr += current_psnr
                avg_ssim += current_ssim
                avg_mse += current_mse
                avg_mae += current_mae
        avg_psnr = avg_psnr / idx
        avg_ssim = avg_ssim / idx
        avg_mse = avg_mse / idx
        avg_mae = avg_mae / idx
        # testing log
        logger.info('<Average PSNR : {:<.2f}dB,  Average SSIM : {:<.4f}, Average MSE : {:<.4f}, Average MAE : {:<.4f}\n'
        .format(avg_psnr, avg_ssim, avg_mse, avg_mae))
        write_csv.write_lines(images_metrics, f"{mode + '_Average'},{avg_mse},{avg_mae},{avg_psnr},{avg_ssim}\n")


    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    opt['path']['logs'] = os.path.join(opt['path']['logs'], image_name)
    opt['path']['options'] = os.path.join(opt['path']['options'], image_name)
    opt['path']['models'] = os.path.join(opt['path']['models'], image_name, mode)
    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))
    

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)
    border = opt['scale']
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    if opt['rank'] == 0:
        logger_name = 'SudoSR'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['logs'], logger_name + '_' + mode + '.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    # ----------------------------------------
    # configure metrics file
    # ----------------------------------------
    metrics_path = opt['path']['metrics']
    images_metrics = os.path.join(metrics_path, image_name + '.csv')
    if not os.path.exists(images_metrics):
        write_csv.write_lines(images_metrics, f"{'ImageName'},{'Mode'},{'MSE'},{'MAE'},{'PSNR'},{'SSIM'}\n")
    # ----------------------------------------
    # image save path
    # ----------------------------------------
    images_path = opt['path']['images']
    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'Trg_test':
            test_set = define_Dataset(dataset_opt, 0, dataseed)
            test_loader = DataLoader(
                                     test_set, 
                                     batch_size=dataset_opt['dataloader_batch_size'],
                                     shuffle=False,
                                     num_workers=dataset_opt['dataloader_num_workers'],
                                     drop_last=False,
                                     pin_memory=True
                                    )
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''
    model = define_Model(opt, mode)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    model.init_train()
    if opt['rank'] == 0:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        logger.info(model.info_network())
        logger.info(model.info_params())

    '''
    # ----------------------------------------
    # Step--4 (main create)
    # ----------------------------------------
    '''
    # -------------------------------
    # 1) create target_train_SR images
    # -------------------------------
    # create_sudoSR(train_loader, opt)
    
    
    # -------------------------------
    # 2) create target_val_SR images
    # -------------------------------
    create_sudoSR(test_loader, current_step, mode, images_path, images_metrics)
    
    
    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        filename = image_name + mode
        option.save(opt, filename) 
    
       

if __name__ == '__main__':
    main(image_name='F_actin', mode='Src', dataseed=0, json_path='/home/liujiaxin/Program/N2022-SR/options/F_actin/run-d4/train_F-actin_x2_Srctest.json')