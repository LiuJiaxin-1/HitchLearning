from audioop import avg
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

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils import utils_csv as write_csv
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model_H import define_Model


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


def main(mode, image_name, srcname, image_type='F_actin', dataseed=0, test_dataroot=None, train_img_nums=0, test_img_nums=0,json_path='./options/train_F-actin_x2_psnr.json'):

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
        # loss_matrics = os.path.join(metrics_path, mode + '_loss.csv')
        avg_metrics = os.path.join(metrics_path, mode + '_avg.csv')
        if not os.path.exists(mode_metrics):
            write_csv.write_lines(mode_metrics, f"{'ImageName'},{'MSE'},{'PSNR'},{'SSIM'}\n")
        if not os.path.exists(images_metrics):
            write_csv.write_lines(images_metrics, f"{'ImageName'},{'Mode'},{'MSE'},{'PSNR'},{'SSIM'}\n")
        # if not os.path.exists(loss_matrics):
        #     write_csv.write_lines(loss_matrics, f"{'Epoch'},{'Loss'}\n")
        if not os.path.exists(avg_metrics):
            write_csv.write_lines(avg_metrics, f"{'EPOCH'},{'AVGMSE'},{'AVGPSNR'},{'AVGSSIM'},{'LOSS'}\n")

        
        # image_metrics =os.path.join(metrics_path, image_name + '.csv') 

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    # print('Random seed: {}'.format(seed))
    logger.info('Random seed: {}'.format(seed))
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
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
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
    if mode in ['Finetune', 'SrcTrg', 'ours']:
        init_optimizer = False
    else:
        init_optimizer = True
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
    for epoch in range(1, opt['train']['epoch'] + 1):  # keep running
    # for epoch in range(100000):
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        # Lambda = epoch / opt['train']['epoch'] * opt['train']['increase_ratio']
        total_loss = 0.0
        for i, train_data in enumerate(train_loader):
            # for train_data in train_loader:

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

            # write_csv.write_lines(loss_matrics, f"{epoch:04d},{total_loss/(i+1)}\n")    

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
                idx = 0

                for test_data in test_loader:
                    idx += 1
                    image_name_ext = os.path.basename(test_data['L_path'][0])
                    img_name, ext = os.path.splitext(image_name_ext)
                    img_name = img_name.split('_', 1)[-1]

                    img_dir = os.path.join(opt['path']['images'], img_name)
                    util.mkdir(img_dir)

                    matric_path = os.path.join(metrics_path, img_name)
                    util.mkdir(matric_path)
                    image_mode_metrics = os.path.join(matric_path, mode + '_' + img_name + '.csv')
                    if not os.path.exists(image_mode_metrics):
                        write_csv.write_lines(image_mode_metrics, f"{'EPOCH'},{'MSE'},{'PSNR'},{'SSIM'}\n")

                    model.feed_data(data=test_data, train_flag=False)
                    model.test()

                    visuals = model.current_visuals()
                    L_img = util.tensor2numpy(visuals['L'])
                    E_img = util.tensor2numpy(visuals['E'])
                    H_img = util.tensor2numpy(visuals['H'])

                    # -----------------------
                    # save image L, image H and estimated image E
                    # -----------------------
                    if epoch == 0 and mode == 'AllTrg':
                        save_img_path = os.path.join(img_dir, '{:s}_L.{:s}'.format(img_name, ext))
                        util.tifsave(L_img, save_img_path)
                        save_img_path = os.path.join(img_dir, '{:s}_H.{:s}'.format(img_name, ext))
                        util.tifsave(H_img, save_img_path)
                    save_img_path = os.path.join(img_dir, '{:s}_{:d}_{:s}_E.{:s}'.format(img_name, epoch, mode, ext))
                    util.tifsave(E_img, save_img_path)

                    # -----------------------
                    # calculate PSNR, SSIM and MSE
                    # -----------------------
                    current_psnr = calculate_psnr(H_img, E_img)
                    current_ssim = calculate_ssim(H_img, E_img)
                    current_mse = calculate_mse(H_img, E_img)
                    # write image_mode.csv
                    write_csv.write_lines(image_mode_metrics, f"{epoch:04d},{current_mse},{current_psnr},{current_ssim}\n")
                    logger.info('{:->4d}--> {:>10s} | PSNR:{:<4.2f}dB | SSIM:{:<4.4f} | MSE:{:<4.5f}'
                    .format(idx, image_name_ext, current_psnr, current_ssim, current_mse))
                    if epoch == opt['train']['epoch']:
                        write_csv.write_lines(mode_metrics, f"{img_name},{current_mse},{current_psnr},{current_ssim}\n")
                        write_csv.write_lines(images_metrics, f"{img_name},{mode},{current_mse},{current_psnr},{current_ssim}\n")
                        # if mode == 'ours':
                        #     write_csv.write_lines(images_metrics, "\n")
                    

                    avg_psnr += current_psnr
                    avg_ssim += current_ssim
                    avg_mse += current_mse

                avg_psnr = avg_psnr / idx
                avg_ssim = avg_ssim / idx
                avg_mse = avg_mse / idx
                avg_loss = total_loss/(i+1)


                # testing log
                logger.info('<epoch:{:3d}, iter:{:8,d}, Average LOSS : {:<.4f}, Average PSNR : {:<.2f}dB,  Average SSIM : {:<.4f}, Average MSE : {:<.4f}\n'
                .format(epoch, current_step, avg_loss, avg_psnr, avg_ssim, avg_mse))
                write_csv.write_lines(avg_metrics, f"{epoch},{avg_mse},{avg_psnr},{avg_ssim},{avg_loss}\n")

            
    
    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        filename = image_name + mode
        option.save(opt, filename) 
    
       

if __name__ == '__main__':
    import os
    import utils.utils_image as util
    from main_train_psnr import main
    json_path = './options/train_F-actin_x2_psnr.json'
    image_type = 'F_actin'
    modes = ['AllTrg', 'PartTrg', 'Trg', 'Src', 'Finetune', 'SrcTrg', 'ours']
    test_root_L = "/home/liujiaxin/Program/N2022/data/SUFDD/F_actin/9/Val/Noise" 
    test_root_H = "/home/liujiaxin/Program/N2022/data/SUFDD/F_actin/9/Val/SR" 
    test_paths_L = util.get_image_paths(test_root_L)
    test_paths_H = util.get_image_paths(test_root_H)
    for i in range(len(test_paths_L)):
        test_L, test_H = test_paths_L[i], test_paths_H[i]
        test_data = {'L': test_L, 'H': test_H}
        mode = 'ours'
        image_name_ext = os.path.basename(test_L)
        image_name, ext = os.path.splitext(image_name_ext)
        image_name = image_name.split('_', 1)[-1]
        if mode in ['AllTrg', 'PartTrg']:
            image_name = image_name.split('_')[0] + '_' + image_name.split('_')[1]
        main(mode=mode, image_name=image_name, image_type=image_type, test_dataroot=test_data, test_img_nums=1, train_img_nums=10, dataseed=1, json_path=json_path)
