import os.path as osp
from subprocess import run
from itsdangerous import exc

from mmcv.runner import Hook
from torch.utils.data import DataLoader
import torch
import mmcv
import torch.distributed as dist
from mmcv.runner import get_dist_info
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type
import torch.nn as nn


class LossWeightDecay(Hook):
    def __init__(self, 
                 loss_name = None, 
                 warm_up = False, 
                 early_stop = False,
                 start_cosine= False,
                 logger = None,
                 **prcisebn_kwargs) -> None:
        super(LossWeightDecay).__init__()
        self.loss_name = loss_name
        self.warm_up = warm_up
        self.early_stop = early_stop
        self.start_cosine = start_cosine
        self.logger = logger
        self.max_loss_weight = None

    def before_train_epoch(self, runner):
        self.logger.info('\nStart Running LossWeightDecay!')
        try:
            loss = getattr(runner.model.module, self.loss_name)
        except:
            raise AttributeError
        if self.max_loss_weight is None:
            self.max_loss_weight = loss.loss_weight 
        if self.warm_up:
            assert(self.warm_up <= runner.max_epochs)
            if runner.epoch <= self.warm_up:
                loss.loss_weight = runner.epoch / self.warm_up * self.max_loss_weight 
        
        if self.early_stop:
            assert(self.early_stop <= runner.max_epochs)
            if runner.epoch > self.early_stop:
                loss.loss_weight = 0

        if self.start_cosine:
            assert(self.start_cosine <= runner.max_epochs)
            if runner.epoch > self.start_cosine:
                loss.loss_weight = torch.cos( 1.57 * torch.tensor((runner.epoch - self.start_cosine) \
                    /(runner.max_epochs - self.start_cosine + 1e-8))) * self.max_loss_weight

        self.logger.info('\nLoss Weight is: {}'.format(getattr(runner.model.module, self.loss_name).loss_weight))
        self.logger.info('\nEnd Running LossWeightDecay!')