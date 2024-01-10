from audioop import avg
import os.path
import math
import argparse
from re import L
from statistics import mean
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

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils import utils_csv as write_csv
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model


'''
# --------------------------------------------
# training code for MSRResNet
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
# github: https://github.com/cszn/KAIR
# --------------------------------------------
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''


def main(mode, image_name, srcname, finetune=False, image_type='F_actin', dataseed=0, test_dataroot=None, train_img_nums=0, test_img_nums=0,json_path='./options/train_F-actin_x2_psnr.json'):

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
    # opt['rank'] = current_step
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # update datasets
    if not mode in ['AllTrg','PartTrg', 'Src']:
        opt['datasets']['test']['dataroot_L'] = test_dataroot['L']
        opt['datasets']['test']['dataroot_H'] = test_dataroot['H']
    if mode in ['Trg', 'Finetune', 'SrcTrg', 'ours']:
        opt['datasets']['train']['dataroot_L'] = test_dataroot['L']
        opt['datasets']['train']['dataroot_H'] = test_dataroot['H'].replace('SR', 'SudoSrc'+srcname)
    train_nums = {'AllTrg': 0, 'PartTrg': train_img_nums, 'Trg': test_img_nums, 'Src': 0, 
                  'Finetune': test_img_nums, 'SrcTrg': test_img_nums, 'ours': test_img_nums}
    test_nums = {'AllTrg': 0, 'PartTrg': 0, 'Trg': test_img_nums, 'Src': test_img_nums, 
                  'Finetune': test_img_nums, 'SrcTrg': test_img_nums, 'ours': test_img_nums}

    # update model
    if finetune:
        if mode in ['Finetune', 'ours', 'SrcTrg'] and not opt['path']['pretrained_netG'] and not opt['path']['pretrained_netE']:
            model_path = opt['path']['models']
            model_path = model_path.replace(image_name, image_type)
            model_path = model_path.replace(mode, 'Src')
            init_iter_G, init_path_G = option.find_last_checkpoint(model_path, net_type='G')
            init_iter_E, init_path_E = option.find_last_checkpoint(model_path, net_type='E')
            opt['path']['pretrained_netG'] = init_path_G
            opt['path']['pretrained_netE'] = init_path_E
            init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(model_path, net_type='optimizerG')
            opt['path']['pretrained_optimizerG'] = init_path_optimizerG
            current_step = 0
            init_optimizer = False
    else:
        init_optimizer = True     
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



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
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['logs'], logger_name + '_' + mode + '.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    # ----------------------------------------
    # configure metrics file
    # ----------------------------------------
    metrics_path = opt['path']['metrics']
    if opt['rank'] == 0:
        mode_metrics = os.path.join(metrics_path, mode + '.csv')
        images_metrics = os.path.join(metrics_path, image_type + '.csv')
        avg_metrics = os.path.join(metrics_path, mode + '_avg.csv')
        if not os.path.exists(mode_metrics):
            write_csv.write_lines(mode_metrics, f"{'ImageName'},{'MSE'},{'PSNR'},{'SSIM'}\n")
        if not os.path.exists(images_metrics):
            write_csv.write_lines(images_metrics, f"{'ImageName'},{'Mode'},{'MSE'},{'PSNR'},{'SSIM'}\n")
        if not os.path.exists(avg_metrics):
            write_csv.write_lines(avg_metrics, f"{'EPOCH'},{'AVGMSE'},{'AVGPSNR'},{'AVGSSIM'},{'LOSS'}\n")

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        filename = image_name + mode
        option.save(opt, filename) 


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
        if phase == 'train':
            train_set = define_Dataset(dataset_opt, train_nums[mode], dataseed)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=False, seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                          shuffle=True,
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                          drop_last=False,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=False,
                                          pin_memory=True)

        elif phase == 'test':
            test_set = define_Dataset(dataset_opt, test_nums[mode], dataseed)
            test_loader = DataLoader(test_set, batch_size=dataset_opt['dataloader_batch_size'],
                                     shuffle=False, num_workers=dataset_opt['dataloader_num_workers'],
                                     drop_last=False, pin_memory=True)
        elif phase == 'update':
            pass
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt, mode)
    model.init_train(init_optimizer=init_optimizer)
    if opt['rank'] == 0:
        logger.info(model.info_network())
        logger.info(model.info_params())

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''
    avg_loss = 0.0
    best_psnr_all = []
    best_ssim_all = []
    best_mse_all = []
    best_psnr = 0.0
    best_ssim = 0.0
    best_mse = 1.0
    for epoch in range(1, opt['train']['epoch'] + 1):  # keep running
    # for epoch in range(100000):
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        # Lambda = epoch / opt['train']['epoch'] * opt['train']['increase_ratio']
        total_loss = 0.0
        for i, train_data in enumerate(train_loader):
            current_step += 1

            # -------------------------------
            # 1) feed patch pairs
            # -------------------------------
            model.feed_data(data=train_data, train_flag=True)

            # -------------------------------
            # 2) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step)

            # -------------------------------
            # 3) training information
            # -------------------------------
            logs = model.current_log()  # such as loss
            message = '<mode:{}, epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(mode, epoch, current_step, model.current_learning_rate())
            for k, v in logs.items():  # merge log information into message
                message += '{:s}: {:.3e} '.format(k, v)
            logger.info(message)
            total_loss += logs['G_loss']
            
            # -------------------------------
            # 4) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)      

        

        # -------------------------------
        # 5) save model
        # -------------------------------
        if epoch % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
            logger.info('Saving the model.')
            model.save(current_step)              


        if epoch % opt['train']['checkpoint_test'] == 0 or epoch == 1:
            # -------------------------------
            # 7) testing
            # -------------------------------
            avg_psnr = 0.0
            avg_ssim = 0.0
            avg_mse = 0.0
            psnr = []
            ssim = []
            mse = []
            idx = 0

            for test_data in test_loader:
                model.feed_data(data=test_data, train_flag=False)
                model.test()
                visuals = model.current_results()
                L_imgs = util.tensor2numpy(visuals['L'])
                E_imgs = util.tensor2numpy(visuals['E'])
                H_imgs = util.tensor2numpy(visuals['H'])
                if len(E_imgs.shape) == 3:
                    for image_path, L_img, E_img, H_img in zip(test_data['H_path'], L_imgs, E_imgs, H_imgs):
                        # -----------------------
                        # set saving dir for img and metrics
                        # -----------------------
                        idx += 1
                        image_name_ext = os.path.basename(image_path)
                        img_name, ext = os.path.splitext(image_name_ext)
                        img_name = img_name.split('_', 1)[-1]
                        img_dir = os.path.join(opt['path']['images'], img_name)
                        util.mkdir(img_dir)
                        matric_path = os.path.join(metrics_path, img_name)
                        util.mkdir(matric_path)
                        image_mode_metrics = os.path.join(matric_path, mode + '_' + img_name + '.csv')
                        if not os.path.exists(image_mode_metrics):
                            write_csv.write_lines(image_mode_metrics, f"{'EPOCH'},{'MSE'},{'PSNR'},{'SSIM'}\n")

                        # -----------------------
                        # save image L, image H and estimated image E
                        # -----------------------
                        save_L_img_path = os.path.join(img_dir, '{:s}_L.{:s}'.format(img_name, ext))
                        save_H_img_path = os.path.join(img_dir, '{:s}_H.{:s}'.format(img_name, ext))
                        save_E_img_path = os.path.join(img_dir, '{:s}_{:d}_{:s}_E.{:s}'.format(img_name, epoch, mode, ext))
                        if epoch == 1:
                            if not os.path.exists(save_L_img_path):
                                util.tifsave(L_img, save_L_img_path)
                            if not os.path.exists(save_H_img_path):
                                util.tifsave(H_img, save_H_img_path)
                        util.tifsave(E_img, save_E_img_path)

                        # -----------------------
                        # calculate PSNR, SSIM and MSE
                        # -----------------------
                        # E_img_norm = util.prctile_norm(E_img)
                        current_psnr = calculate_psnr(H_img, E_img, data_range=1)
                        current_ssim = calculate_ssim(H_img, E_img, data_range=1)
                        current_mse = calculate_mse(H_img, E_img)
                        psnr.append(current_psnr)
                        ssim.append(current_ssim)
                        mse.append(current_mse)
                        # write image_mode.csv
                        write_csv.write_lines(image_mode_metrics, f"{epoch:04d},{current_mse},{current_psnr},{current_ssim}\n")
                        logger.info('{:->4d}--> {:>10s} | PSNR:{:<4.2f}dB | SSIM:{:<4.4f} | MSE:{:<4.5f}'
                        .format(idx, image_name_ext, current_psnr, current_ssim, current_mse))
                
                elif len(E_imgs.shape) == 2:
                    image_path = test_data['H_path'][0]
                    L_img = L_imgs
                    E_img = E_imgs
                    H_img = H_imgs
                    # -----------------------
                    # set saving dir for img and metrics
                    # -----------------------
                    idx += 1
                    image_name_ext = os.path.basename(image_path)
                    img_name, ext = os.path.splitext(image_name_ext)
                    img_name = img_name.split('_', 1)[-1]
                    img_dir = os.path.join(opt['path']['images'], img_name)
                    util.mkdir(img_dir)
                    matric_path = os.path.join(metrics_path, img_name)
                    util.mkdir(matric_path)
                    image_mode_metrics = os.path.join(matric_path, mode + '_' + img_name + '.csv')
                    if not os.path.exists(image_mode_metrics):
                        write_csv.write_lines(image_mode_metrics, f"{'EPOCH'},{'MSE'},{'PSNR'},{'SSIM'}\n")

                    # -----------------------
                    # save image L, image H and estimated image E
                    # -----------------------
                    save_L_img_path = os.path.join(img_dir, '{:s}_L.{:s}'.format(img_name, ext))
                    save_H_img_path = os.path.join(img_dir, '{:s}_H.{:s}'.format(img_name, ext))
                    save_E_img_path = os.path.join(img_dir, '{:s}_{:d}_{:s}_E.{:s}'.format(img_name, epoch, mode, ext))
                    if epoch == 1:
                        if not os.path.exists(save_L_img_path):
                            util.tifsave(L_img, save_L_img_path)
                        if not os.path.exists(save_H_img_path):
                            util.tifsave(H_img, save_H_img_path)
                    util.tifsave(E_img, save_E_img_path)

                    # -----------------------
                    # calculate PSNR, SSIM and MSE
                    # -----------------------
                    # E_img_norm = util.prctile_norm(E_img)
                    current_psnr = calculate_psnr(H_img, E_img, data_range=1)
                    current_ssim = calculate_ssim(H_img, E_img, data_range=1)
                    current_mse = calculate_mse(H_img, E_img)
                    psnr.append(current_psnr)
                    ssim.append(current_ssim)
                    mse.append(current_mse)
                    # write image_mode.csv
                    write_csv.write_lines(image_mode_metrics, f"{epoch:04d},{current_mse},{current_psnr},{current_ssim}\n")
                    logger.info('{:->4d}--> {:>10s} | PSNR:{:<4.2f}dB | SSIM:{:<4.4f} | MSE:{:<4.5f}'
                    .format(idx, image_name_ext, current_psnr, current_ssim, current_mse))
                    # if mode == 'ours':
                    #     write_csv.write_lines(images_metrics, "\n")
                

            avg_psnr = mean(psnr)
            avg_ssim = mean(ssim)
            avg_mse = mean(mse)

            if avg_psnr >= best_psnr:
                best_psnr = avg_psnr
                best_ssim = avg_ssim
                best_mse = avg_mse
                best_epoch = epoch
                best_step = current_step
                best_psnr_all = psnr
                best_ssim_all = ssim
                best_mse_all = mse

            # testing log
            avg_loss = total_loss/(i+1)
            logger.info('<epoch:{:3d}, iter:{:8,d}, Average LOSS : {:<.4f}, Average PSNR : {:<.2f}dB,  Average SSIM : {:<.4f}, Average MSE : {:<.4f}\n'
            .format(epoch, current_step, avg_loss, avg_psnr, avg_ssim, avg_mse))
            write_csv.write_lines(avg_metrics, f"{epoch},{avg_mse},{avg_psnr},{avg_ssim},{avg_loss}\n")

    logger.info('<best epoch:{:3d}, best step:{:8,d}, Best PSNR : {:<.2f}dB,  Best SSIM : {:<.4f}, Best MSE : {:<.4f}\n'
    .format(best_epoch, best_step, best_psnr, best_ssim, best_mse))
    for j in range(len(test_set.paths_L)):
        image_path = test_set.paths_L[j]
        image_name_ext = os.path.basename(image_path)
        img_name, ext = os.path.splitext(image_name_ext)
        img_name = img_name.split('_', 1)[-1]
        write_csv.write_lines(images_metrics, f"{img_name},{mode},{best_mse_all[j]},{best_psnr_all[j]},{best_ssim_all[j]}\n")
    
       

if __name__ == '__main__':
    pass