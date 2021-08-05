from mmcv.runner.checkpoint import weights_to_cpu
from lowbit_classification.lbitcls.models import losses
from logging import setLoggerClass
from numpy import mod
from torch._C import set_flush_denormal
import torch.nn as nn
import torch
from torch.nn.modules import loss
from ..builder import CLASSIFIERS, DISTILLERS, build_classifier, build_backbone, build_head, build_neck, build_loss
from ..classifiers.base import BaseClassifier
from QuanTransformer.quantrans.quantops.ABQAT import ABQATConv2d
from QuanTransformer.quantrans.quantops.LSQPlus import LSQDPlusConv2d

from .base import BaseDistiller
from mmcv import Config
from QuanTransformer.quantrans import build_mtransformer
from mmcv.runner import load_checkpoint
from lowbit_classification.lbitcls.models.distillers.distillers import KDDistiller


@DISTILLERS.register_module()
class StageDistiller(KDDistiller):
    def __init__(self, 
                student_model, 
                teacher_model, 
                distill_loss = dict(type = 'DistillKL', temperature = 4),
                stage_loss = dict(type='FeatureLossV1')
                ):
       super(StageDistiller, self).__init__(student_model, teacher_model, distill_loss=distill_loss)
       self.stage_loss = build_loss(stage_loss)
    
    def submodule_forward(self, model, img):
        feats = model.backbone(img)
        if model.with_neck:
            if isinstance(feats, list):
                before_logits = model.neck(feats[-1])
            else:
                before_logits = model.neck(feats) 
        return feats, before_logits
    
    def forward_train(self, img, gt_label, **kwargs):
        student_feats, stu_before_logits = self.submodule_forward(self.student_model, img)
        losses = dict()

        loss_stu_cls, student_logits = self.student_model.head.forward_train(stu_before_logits, gt_label)
        with torch.no_grad():
            self.teacher_model.eval()
            teacher_feats, tech_before_logits = self.submodule_forward(self.teacher_model, img)
            loss_teach_cls, teacher_logits = self.teacher_model.head.forward_train(tech_before_logits, gt_label)
        
        loss_st_KD = self.distill_loss(student_logits, teacher_logits)
        
        if isinstance(student_feats, list):
            raise NotImplementedError
        else:
            loss_stage = self.stage_loss(student_feats, teacher_feats, True) + \
                self.stage_loss(stu_before_logits, tech_before_logits, True)

        losses.update(loss_stu_cls)
        losses.update(loss_st_KD)
        losses.update({self.stage_loss._get_name()+'loss':loss_stage})
        return losses