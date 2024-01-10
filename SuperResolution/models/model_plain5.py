from models.model_plain4 import ModelPlain4
import torch

class ModelPlain5(ModelPlain4):
    """Train with two inputs (L, C) and with pixel loss"""

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data, train_flag, need_H=True):
        self.L = data['L'].to(self.device)
        if train_flag:
            self.SL = data['SL'].to(self.device)
            self.L = torch.cat((self.L, self.SL), 0)
        if need_H:
            self.H = data['H'].to(self.device)
            if train_flag:
                self.SH = data['SH'].to(self.device)
                self.H = torch.cat((self.H, self.SH), 0)

    # # ----------------------------------------
    # # feed (L, C) to netG and get E
    # # ----------------------------------------
    # def netG_forward(self):
    #     self.E = self.netG(self.L)

