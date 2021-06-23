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
