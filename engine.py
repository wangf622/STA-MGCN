import os.path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import util
import shutil
from tensorboardX import SummaryWriter
import time


class Trainer_1(object):
    def __init__(self, device, model, model_name, lrate, wdecay, decay):
        self.device = device
        self.model = model
        self.model.to(self.device).reset_parameters()
        self.model_name = model_name
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        # self.optimizer = optim.AdamW(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        # self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[30, 60, 90], gamma=0.1, last_epoch=-1)
        # self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=decay, verbose=True)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, eta_min=1e-6, T_max=32, verbose=True, last_epoch=-1)
        # self.scheduler = optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=50, gamma=0.5)
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=.3, patience=10, threshold=1e-3,
        #                                                       min_lr=1e-5, verbose=True)
        # self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.optimizer, T_0=16, T_mult=2, eta_min=1e-6, last_epoch=-1, verbose=True)
        # self.criterion = nn.MSELoss().to(self.device)
        self.criterion = nn.SmoothL1Loss().to(self.device)
        # self.criterion = util.masked_mae
        self.clip = 10
        # self.scaler = scaler

    def train(self, input, target):
        self.model.train()
        self.optimizer.zero_grad()  # input 为输入得mini_batch，再训练之前清空每一个mini_batch的梯度

        output = self.model(input)
        loss = self.criterion(output, target)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(preds=output, labels=target, null_val=0.0).item()
        mape = util.masked_mape(preds=output, labels=target, null_val=0.0).item()
        rmse = util.masked_rmse(preds=output, labels=target, null_val=0.0).item()

        return loss.item(), mae, mape, rmse

    def eval(self, input, target):
        self.model.eval()

        output = self.model(input)
        # loss = self.criterion(output, target)
        loss = self.criterion(output, target)

        mae = util.masked_mae(preds=output, labels=target, null_val=0.0).item()
        mape = util.masked_mape(preds=output, labels=target, null_val=0.0).item()
        rmse = util.masked_rmse(preds=output, labels=target, null_val=0.0).item()

        return loss.item(), mae, mape, rmse
