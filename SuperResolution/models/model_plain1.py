from collections import OrderedDict
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam

from models.select_network import define_G
from models.model_plain import ModelPlain

from utils.utils_image import generate_mask_pair, generate_subimages
from utils.utils_model import test_mode
from utils.utils_regularizers import regularizer_orth, regularizer_clip


class ModelPlain1(ModelPlain):
    """Train with two sub-images"""
    def __init__(self, opt):
        super(ModelPlain1, self).__init__(opt)
        self.Lambda1 = self.opt_train['train']['Lambda1']
        self.Lambda2 = self.opt_train['train']['Lambda2']

    # ----------------------------------------
    # feed L data
    # ----------------------------------------
    def feed_data(self, data, need_H=False):
        self.L = data['L'].to(self.device)
        if need_H:
            self.H = data['H'].to(self.device)

    # ----------------------------------------
    # feed L to netG
    # ----------------------------------------
    def netG_forward(self, seed):
        self.mask1, self.mask2 = generate_mask_pair(self.L, seed)
        self.L_sub1 = generate_subimages(self.L, self.mask1)
        self.L_sub2 = generate_subimages(self.L, self.mask2)
        with torch.no_grad():
            self.E = self.netG(self.L)
        self.E_sub1 = generate_subimages(self.E, self.mask1)
        self.E_sub2 = generate_subimages(self.E, self.mask2)
        self.L_sub1_E = self.netG(self.L_sub1)
        self.T = self.L_sub2

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step, Lambda):
        self.G_optimizer.zero_grad()
        self.netG_forward(current_step)
        diff = self.L_sub1_E - self.T
        exp_diff = self.E_sub1 - self.E_sub2
        loss1 = self.G_lossfn(self.L_sub1_E, self.T)
        loss2 = Lambda * self.G_lossfn(diff, exp_diff)
        G_loss = self.Lambda1 * loss1 + self.Lambda2 * loss2
        G_loss.backward()

        self.G_optimizer.step()

        # self.log_dict['G_loss'] = G_loss.item()/self.E.size()[0]  # if `reduction='sum'`
        self.log_dict['loss1'] = loss1.item()
        self.log_dict['loss2'] = loss2.item()
        self.log_dict['G_loss'] = G_loss.item()

    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.netG_forward()
        self.netG.train()

    # ----------------------------------------
    # test / inference x8
    # ----------------------------------------
    def testx8(self):
        self.netG.eval()
        with torch.no_grad():
            self.E = test_mode(self.netG, self.L, mode=3, sf=self.opt['scale'], modulo=1)
        self.netG.train()

    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict

    # ----------------------------------------
    # get L, E, H image
    # ----------------------------------------
    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach()[0].float().cpu()
        out_dict['E'] = self.E.detach()[0].float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach()[0].float().cpu()
        return out_dict

    # ----------------------------------------
    # get L, E, H batch images
    # ----------------------------------------
    def current_results(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach().float().cpu()
        out_dict['E'] = self.E.detach().float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach().float().cpu()
        return out_dict

    """
    # ----------------------------------------
    # Information of netG
    # ----------------------------------------
    """

    # ----------------------------------------
    # print network
    # ----------------------------------------
    def print_network(self):
        msg = self.describe_network(self.netG)
        print(msg)

    # ----------------------------------------
    # print params
    # ----------------------------------------
    def print_params(self):
        msg = self.describe_params(self.netG)
        print(msg)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self):
        msg = self.describe_network(self.netG)
        return msg

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msg = self.describe_params(self.netG)
        return msg
