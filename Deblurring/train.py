import os
from config import Config 
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import logging
import shutil
import random
import time
import numpy as np

from skimage.metrics import structural_similarity as calculate_ssim
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import mean_squared_error as calculate_mse
from skimage.metrics import mean_absolute_error as calculate_mae
from statistics import mean

import utils
from Datasets.data import get_training_data, get_srctrg_data, get_validation_data
from model.MPRNet import MPRNet
import model.losses as losses
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from pdb import set_trace as stx

def deblur_train(test_dir, option_path, dataname, img_name, nums=0):

    torch.backends.cudnn.benchmark = True
    ######### Get options and set env ###########
    opt = Config(option_path)
    gpus = ','.join([str(i) for i in opt.GPU])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    

    ######### Set Seeds ###########
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    ######### Set train info ###########
    start_epoch = 1
    mode = opt.MODEL.MODE
    session = opt.MODEL.SESSION

    ######### Set path and save info ###########
    result_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'results')
    model_dir  = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'models')
    utils.mkdir(result_dir)
    utils.mkdir(model_dir)

    logger_dir  = os.path.join(result_dir, 'logs')
    images_dir  = os.path.join(result_dir, 'images')
    metric_dir = os.path.join(result_dir, 'metrics')
    utils.mkdir(logger_dir)
    utils.mkdir(images_dir)
    utils.mkdir(metric_dir)

    best_img_dir = os.path.join(images_dir, 'best_images')
    utils.mkdir(best_img_dir)

    best_metric = os.path.join(metric_dir, dataname + '_' + mode +'.csv')
    if not os.path.exists(best_metric):
        utils.write_lines(best_metric, f"{'ImageName'},{'Mode'},{'MSE'},{'MAE'},{'PSNR'},{'SSIM'}\n")

    logger_name = 'MDB_' + img_name + '_' + mode
    utils.logger_info(logger_name, os.path.join(logger_dir, logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    # logger.info(option.dict2str(opt))

    src_dir = opt.TRAINING.SRC_DIR
    trg_dir = opt.TRAINING.TRG_DIR
    val_dir = opt.TRAINING.VAL_DIR
    
    # -----------------------
    # update datasets
    # -----------------------
    if mode in ['Trg', 'Finetune']:
        src_dir = test_dir
        val_dir = test_dir
    elif mode in ['ours', 'SrcTrg']:
        trg_dir = test_dir
        val_dir = test_dir

    ######### Model ###########
    # -----------------------
    # Model Para
    # in_c = 1
    # n_feat  = 96
    # scale_unetfeats=48, 
    # scale_orsnetfeats=32, 
    # num_cab=8, 
    # kernel_size=3, 
    # reduction=4, 
    # bias=False
    # -----------------------
    model_restoration  = MPRNet(in_c=opt.MODEL.IN_CHAN, 
                                out_c=opt.MODEL.IN_CHAN, 
                                n_feat=opt.MODEL.N_FEAT, 
                                scale_unetfeats=opt.MODEL.SCALE_UNETFEATS, 
                                scale_orsnetfeats=opt.MODEL.SCALE_ORSNETFEATS, 
                                num_cab=opt.MODEL.NUM_CAB, 
                                kernel_size=opt.MODEL.KERNEL_SIZE, 
                                reduction=opt.MODEL.REDUCTION, 
                                bias=opt.MODEL.BIAS)
    model_restoration.cuda()

    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 1:
        logger.info("Let's use {} GPUs!".format(torch.cuda.device_count()))

    new_lr = opt.OPTIM.LR_INITIAL

    optimizer = optim.Adam(model_restoration.parameters(), 
                           lr=new_lr, betas=(0.9, 0.999),
                           eps=1e-8)

    ######### Scheduler ###########
    warmup_epochs = opt.OPTIM.WARM_EPOCH
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                            opt.OPTIM.NUM_EPOCHS-warmup_epochs, 
                                                            eta_min=opt.OPTIM.LR_MIN)
    scheduler = GradualWarmupScheduler(optimizer, 
                                       multiplier=1, 
                                       total_epoch=warmup_epochs, 
                                       after_scheduler=scheduler_cosine)
    scheduler.step()

    ######### Resume ###########
    # -----------------------
    # update model
    # -----------------------
    if mode == 'Finetune':
        model_path = model_dir.replace(mode, 'Src')
        path_chk_rest = utils.get_last_path(model_path, '_best.pth')
        utils.load_checkpoint(model_restoration,path_chk_rest)
        logger.info('------------------------------------------------------------------------------')
        logger.info("==> Finetune model based on {}".format(path_chk_rest))
        logger.info('------------------------------------------------------------------------------')
    if mode not in ['AllTrg', 'PartTrg', 'Src']:
        model_dir = os.path.join(model_dir, img_name)
        utils.mkdir(model_dir)

    if opt.TRAINING.RESUME:
        path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
        utils.load_checkpoint(model_restoration,path_chk_rest)
        start_epoch = utils.load_start_epoch(path_chk_rest) + 1
        utils.load_optim(optimizer, path_chk_rest)

        for i in range(1, start_epoch):
            scheduler.step()
        new_lr = scheduler.get_lr()[0]
        logger.info('------------------------------------------------------------------------------')
        logger.info("==> Resuming Training with learning rate:{}".format(new_lr))
        logger.info('------------------------------------------------------------------------------')
    

    if len(device_ids)>1:
        model_restoration = nn.DataParallel(model_restoration, device_ids = device_ids)

    ######### Loss ###########
    criterion_char = losses.CharbonnierLoss()
    criterion_edge = losses.EdgeLoss()

    ######### DataLoaders ###########
    if mode not in ['ours', 'SrcTrg']:
        train_dataset = get_training_data(src_dir, 
                                          opt.TRAINING.TRAIN_RAW_NAME, 
                                          opt.TRAINING.TRAIN_GT_NAME, 
                                          {'patch_size':opt.TRAINING.TRAIN_PS},
                                          opt.TRAINING.ELIBACK,
                                          nums)
    else:
        train_dataset = get_srctrg_data(src_dir,
                                        trg_dir,
                                        opt.TRAINING.TRAIN_RAW_NAME,
                                        opt.TRAINING.TRAIN_GT_NAME,
                                        opt.TRAINING.VAL_RAW_NAME,
                                        opt.TRAINING.VAL_SUDOGT_NAME,
                                        {'patch_size':opt.TRAINING.TRAIN_PS})
    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=opt.OPTIM.BATCH_SIZE, 
                              shuffle=True, 
                              num_workers=opt.OPTIM.NUM_WORKERS, 
                              drop_last=False, 
                              pin_memory=True)

    val_dataset = get_validation_data(val_dir, 
                                      opt.TRAINING.VAL_RAW_NAME,
                                      opt.TRAINING.VAL_GT_NAME,
                                      {'patch_size':None})
    val_loader = DataLoader(dataset=val_dataset, 
                            batch_size=12, 
                            shuffle=False, 
                            num_workers=24, 
                            drop_last=False, 
                            pin_memory=True)

    logger.info('===> Start Epoch {} End Epoch {}'.format(start_epoch, opt.OPTIM.NUM_EPOCHS + 1))
    logger.info('===> Loading datasets')

    avg_raw_psnr = 0.0
    avg_raw_ssim = 0.0
    avg_raw_mse = 0.0
    avg_raw_mae = 0.0
    best_epoch = 0
    best_psnr = 0.0
    best_ssim = 0.0
    best_mse = 1.0
    best_mae = float('inf')
    best_psnr_all = []
    best_ssim_all = []
    best_mse_all = []
    best_mae_all = []

    for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        epoch_loss = 0
        train_id = 1

        model_restoration.train()
        for i, data in enumerate(tqdm(train_loader), 0):

            # -----------------------
            # zero_grad
            # -----------------------
            for param in model_restoration.parameters():
                param.grad = None

            # -----------------------
            # get training data
            # -----------------------
            gts_ = data[1].cuda()
            raws = data[2].cuda()
            if mode in ['ours', 'SrcTrg']:
                gts_trg_ = data[3].cuda()
                raws_trg = data[4].cuda()
                if mode == 'ours':
                    raws = utils.FDA_source_to_target_low(raws, raws_trg, opt.TRAINING.LB)
                gts_ = torch.cat((gts_, gts_trg_), 0)
                raws = torch.cat((raws, raws_trg), 0)
            
            loss_char = 0.0
            loss_edge = 0.0

            restored = model_restoration(raws)
    
            # -----------------------
            # Compute loss at each stage
            # -----------------------
            loss_char_ = [criterion_char(restored[j],gts_) for j in range(len(restored))]
            loss_edge_ = [criterion_edge(restored[j],gts_) for j in range(len(restored))]
            for char, edge in zip(loss_char_, loss_edge_):
                loss_char += char
                loss_edge += edge
            loss = (loss_char) + (0.05*loss_edge)
        
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_restoration.parameters(), 0.01)
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step()
        
        logger.info("------------------------------------------------------------------")
        logger.info("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".
        format(epoch, time.time()-epoch_start_time, epoch_loss, scheduler.get_lr()[0]))
        logger.info("------------------------------------------------------------------")
        
        torch.save({'epoch': epoch, 
                    'state_dict': model_restoration.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, os.path.join(model_dir,"model_latest.pth"))

        #### Evaluation ####
        if epoch%opt.TRAINING.VAL_AFTER_EVERY == 0:
            model_restoration.eval()
            img_nums = 0
            avg_psnr = 0.0
            avg_ssim = 0.0
            avg_mse = 0.0
            avg_mae = 0.0
            psnr = []
            ssim = []
            mse = []
            mae = []
            for ii, data_val in enumerate((val_loader), 0):
                fs_n = data_val[0]
                gts_ = data_val[1].cuda()
                raws = data_val[2].cuda()

                with torch.no_grad():
                    restored = model_restoration(raws)
                # restored = torch.clamp(restored[0],0,1)
                restored = restored[0]

                for raw, gt_, res, f_n in zip(raws, gts_, restored, fs_n):
                    img_nums += 1
                    f_n = f_n.split('_', 1)[-1]
                    raw = utils.tensor2numpy(raw)
                    gt_ = utils.tensor2numpy(gt_)
                    res = utils.tensor2numpy(res)

                    # -----------------------
                    # save image of raw, res, gt_
                    # -----------------------
                    image_dir = os.path.join(images_dir, f_n)
                    utils.mkdir(image_dir)
                    if epoch == opt.TRAINING.VAL_AFTER_EVERY:
                        save_raw_path = os.path.join(image_dir, 'MB_{:s}.tif'.format(f_n))
                        save_gt__path = os.path.join(image_dir, 'GT_{:s}.tif'.format(f_n))
                        if not os.path.exists(save_raw_path):
                            utils.tifsave(raw, save_raw_path)
                        if not os.path.exists(save_gt__path):
                            utils.tifsave(gt_, save_gt__path)
                    save_db_path = os.path.join(image_dir, 'MDB_{:s}_{:d}_{:s}.tif'.format(f_n, epoch, mode))
                    utils.tifsave(res, save_db_path)

                    image_metric_dir = metric_dir
                    image_metric = os.path.join(image_metric_dir,f_n + '_' + mode +'.csv')
                    if not os.path.exists(image_metric):
                        utils.write_lines(image_metric, f"{'EPOCH'},{'MSE'},{'MAE'},{'PSNR'},{'SSIM'}\n")
                    # -----------------------
                    # calculate PSNR, SSIM and MSE
                    # -----------------------
                    if epoch == opt.TRAINING.VAL_AFTER_EVERY:
                        raw_psnr = calculate_psnr(gt_, raw)
                        raw_ssim = calculate_ssim(gt_, raw)
                        raw_mse  = calculate_mse(gt_, raw)
                        raw_mae  = calculate_mae(gt_, raw)
                        avg_raw_psnr += raw_psnr
                        avg_raw_ssim += raw_ssim
                        avg_raw_mse += raw_mse
                        avg_raw_mae += raw_mae
                        utils.write_lines(best_metric, f"{f_n},{'blur'},{raw_mse},{raw_mae},{raw_psnr},{raw_ssim}\n")
                        # print("[--Blur-- Image:%s  PSNR: %.4f      SSIM: %.4f     MSE: %.4f      MAE: %.4f]"% (f_n, raw_psnr, raw_ssim, raw_mse, raw_mae))

                    current_psnr = calculate_psnr(gt_, res)
                    current_ssim = calculate_ssim(gt_, res)
                    current_mse  = calculate_mse(gt_, res)
                    current_mae  = calculate_mae(gt_, res)
                    psnr.append(current_psnr)
                    ssim.append(current_ssim)
                    mse.append(current_mse)
                    mae.append(current_mae)
                    utils.write_lines(image_metric, f"{epoch:04d},{current_mse},{current_mae},{current_psnr},{current_ssim}\n")
                    # print("[---DB--- Image:%s  PSNR: %.4f      SSIM: %.4f     MSE: %.4f      MAE: %.4f]"% (f_n, current_psnr, current_ssim, current_mse, current_mae))

            avg_psnr = mean(psnr)
            avg_ssim = mean(ssim)
            avg_mse = mean(mse)
            avg_mae = mean(mae)
            if epoch == opt.TRAINING.VAL_AFTER_EVERY:
                avg_raw_psnr /= img_nums
                avg_raw_ssim /= img_nums
                avg_raw_mse /= img_nums
                avg_raw_mae /= img_nums
                if mode in ['AllTrg', 'PartTrg', 'Src']:
                    utils.write_lines(best_metric, f"{'Average'},{'blur'},{avg_raw_mse},{avg_raw_mae},{avg_raw_psnr},{avg_raw_ssim}\n")

            if best_psnr < avg_psnr:
                best_psnr = avg_psnr
                best_epoch = epoch
                best_ssim = avg_ssim
                best_mse = avg_mse
                best_mae = avg_mae
                best_psnr_all = psnr
                best_ssim_all = ssim
                best_mse_all = mse
                best_mae_all = mae
                torch.save({'epoch': epoch, 
                            'state_dict': model_restoration.state_dict(),
                            'optimizer' : optimizer.state_dict()
                            }, os.path.join(model_dir,"model_best.pth"))


            logger.info("############################################################")
            logger.info("[epoch      %d  PSNR: %.4f      SSIM: %.4f      MSE: %.4f      MAE: %.4f]"
            % (epoch, avg_psnr, avg_ssim, avg_mse, avg_mae))
            logger.info("[best_epoch %d  Best_PSNR: %.4f Best_SSIM: %.4f Best_MSE: %.4f Best_MAE: %.4f]"
            % (best_epoch, best_psnr, best_ssim, best_mse, best_mae))
            logger.info("[BlurImage  %d  Blur_PSNR: %.4f Blur_SSIM: %.4f Blur_MSE: %.4f Blur_MAE: %.4f]"
            % (epoch, avg_raw_psnr, avg_raw_ssim, avg_raw_mse, avg_raw_mae))
            logger.info("############################################################")

            torch.save({'epoch': epoch, 
                        'state_dict': model_restoration.state_dict(),
                        'optimizer' : optimizer.state_dict()
                        }, os.path.join(model_dir,f"model_epoch_{epoch}.pth"))
    
    #### save best info ####
    for j in range(val_dataset.sizex):
        f_p = val_dataset.inp_filenames[j]
        f_n = os.path.splitext(os.path.split(f_p)[-1])[0]
        f_n = f_n.split('_', 1)[-1]
        psnr_best = best_psnr_all[j]
        ssim_best = best_ssim_all[j]
        mse_best  = best_mse_all[j]
        mae_best  = best_mae_all[j]
        utils.write_lines(best_metric, f"{f_n},{mode},{mse_best},{mae_best},{psnr_best},{ssim_best}\n")
        img_path = os.path.join(images_dir, f_n, 'MDB_{:s}_{:d}_{:s}.tif'.format(f_n, epoch, mode))
        best_path = os.path.join(best_img_dir, 'Best_{:s}_{:s}.tif'.format(f_n, mode))
        shutil.copy(img_path, best_path)

    if mode in ['AllTrg', 'PartTrg', 'Src']:
        utils.write_lines(best_metric, f"{'Average'},{mode},{mean(best_mse_all)},{mean(best_mae_all)},{mean(best_psnr_all)},{mean(best_ssim_all)}\n")   
