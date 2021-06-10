import os.path as osp

from mmcv.runner import Hook
from torch.utils.data import DataLoader
import torch
import mmcv
import torch.distributed as dist
from mmcv.runner import get_dist_info
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type
import torch.nn as nn

BN_MODULE_TYPES: Tuple[Type[nn.Module]] = (
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
)

def _scaled_all_reduce(tensors):
    """Performs the scaled all_reduce operation on the provided tensors.

    The input tensors are modified in-place. Currently supports only the sum
    reduction operator. The reduced values are scaled by the inverse size of the
    process group (equivalent to cfg.NUM_GPUS).
    """
    # There is no need for reduction in the single-proc case
    rank, world_size = get_dist_info()
    if world_size == 1:
        return tensors
    # Queue the reductions
    reductions = []
    for tensor in tensors:
        reduction = torch.distributed.all_reduce(tensor, async_op=True)
        reductions.append(reduction)
    # Wait for reductions to finish
    for reduction in reductions:
        reduction.wait()
    # Scale the results
    for tensor in tensors:
        tensor.mul_(1.0 / world_size)
    return tensors


@torch.no_grad()
def _precisebn_stats(model, data_loader, num_samples, logger = None, dist = False):
    if dist:
        rank, world_size = get_dist_info()
        num_samples = max(num_samples, data_loader.batch_size * world_size)
        num_iter = int(num_samples / data_loader.batch_size / world_size)
        time.sleep(2)
    else:
        num_iter = int(num_samples / data_loader.batch_size)
    num_iter = min(num_iter, len(data_loader))

    bns = [m for m in model.modules() if isinstance(m, BN_MODULE_TYPES)]
    assert len(bns) != 0, 'Your model has no BN module, Please turn off precise bn!'
    if logger is not None:
        if (dist == False) or ( dist and rank == 0):
            logger.info(r'Model has {} BN modules'.format(len(bns)))
    running_means = [torch.zeros_like(bn.running_mean) for bn in bns]
    running_vars = [torch.zeros_like(bn.running_var) for bn in bns]
    momentums = [bn.momentum for bn in bns]

    for bn in bns:
        bn.momentum = 1.0

    results = []
    dataset = data_loader.dataset
    if (dist == False) or ( dist and rank == 0):
        prog_bar = mmcv.ProgressBar(num_samples)
    for i, data in enumerate(data_loader):
        if i >= num_iter:
            return
        with torch.no_grad():
            data.pop('gt_label')
            result = model(return_loss=False, **data)
            for i, bn in enumerate(bns):
                running_means[i] += bn.running_mean / num_iter
                running_vars[i] += bn.running_var / num_iter

        if (dist == False) or ( dist and rank == 0):
            batch_size = data['img'].size(0) * world_size if dist else data['img'].size(0)
            for _ in range(batch_size):
                prog_bar.update()

    if dist:
        running_means = _scaled_all_reduce(running_means)
        running_vars = _scaled_all_reduce(running_vars)
    #print("Debug Rank{} :".format(rank), running_means[0][0:5])
    
    for i, bn in enumerate(bns):
        bn.running_mean = running_means[i]
        bn.running_var = running_vars[i]
        bn.momentum = momentums[i]

class PrecisebnHook(Hook):
    """Evaluation hook.

    Args:
        dataloader (DataLoader): A PyTorch dataloader.
        num_samples (int): The number of precise samples.
        interval (int): Evaluation interval (by epochs). Default: 1.
    """

    def __init__(self, dataloader, num_samples = 1024, interval = 1, logger = None, **prcisebn_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError('dataloader must be a pytorch DataLoader, but got'
                            f' {type(dataloader)}')
        self.dataloader = dataloader
        self.num_samples = num_samples
        self.interval = interval
        self.eval_kwargs = prcisebn_kwargs
        self.logger = logger

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        
        self.logger.info('\nStart Running Precise BatchNorm2D !')
        _precisebn_stats(runner.model, self.dataloader, self.num_samples)
        self.logger.info('\nEnd Running Precise BatchNorm2D !') 

class DistPrecisebnHook(PrecisebnHook):

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        
        self.logger.info('\nStart Running Precise BatchNorm2D !')
        _precisebn_stats(runner.model, self.dataloader, self.num_samples, logger = self.logger,dist = True)
        self.logger.info('\nEnd Running Precise BatchNorm2D !')