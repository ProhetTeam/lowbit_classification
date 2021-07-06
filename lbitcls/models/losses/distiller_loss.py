import torch
from torch._C import set_flush_denormal
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import loss

from ..builder import LOSSES
from .utils import weight_reduce_loss


def distill_kl(feat_student, feat_teacher, temperature):
    p_s = F.log_softmax(feat_student/temperature, dim = 1)
    p_t = F.softmax(feat_teacher/temperature, dim = 1)
    loss = F.kl_div(p_s, p_t, size_average=False) * (temperature*2) / feat_student.shape[0]
    return loss

@LOSSES.register_module()
class DistillKL(nn.Module):

    def __init__(self, temperature = 4, loss_weight=1.0):
        super(DistillKL, self).__init__()
        self.temperature = temperature
        self.loss_weight = loss_weight
        self.cls_criterion = distill_kl

    def forward(self,
                feat_student, 
                feat_teacher,
                **kwargs):
        losses = dict()
        loss_kl = self.loss_weight * self.cls_criterion(
            feat_student, 
            feat_teacher, 
            self.temperature,
            **kwargs)
        losses.update({'KLloss':loss_kl})
        return losses

@LOSSES.register_module()
class FeatureLossV1(nn.Module):
    def __init__(self, 
                 beta = 0.1,
                 loss_weight = 1.0):
        super(FeatureLossV1, self).__init__()
        self.loss_weight = loss_weight
        self.beta = beta  

    def forward(self,
                feat_student,
                feat_teacher,
                return_value = False,
                **kwargs):
        losses = dict()

        feat_loss_weight = self.loss_weight * \
            torch.exp(feat_teacher.abs() * self.beta + self.dist(feat_teacher))
        feat_loss = (feat_loss_weight * (feat_student - feat_teacher)).abs().mean()
        if return_value:
            return feat_loss
        else:
            losses.update({'FeatLossV1':feat_loss})
            return losses
    
    def dist(self, data, bins = 600):
        bin_size = (data.max() - data.min()) / bins
        hisc = data.detach().histc(bins = bins) / data.numel()
        xbin_idx = ((data - data.min()) / bin_size).floor().clamp(0, bins - 1)
        res = hisc[xbin_idx.long()]
        res = res/res.sum()
        return res