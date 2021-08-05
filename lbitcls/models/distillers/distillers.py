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

@DISTILLERS.register_module()
class KDDistiller(BaseDistiller):

    def __init__(self, 
                 student_model, 
                 teacher_model,
                 distill_loss = dict(type = 'DistillKL', temperature = 4)):
        super(KDDistiller, self).__init__()

        self.student_config = student_model
        self.teacher_config = teacher_model

        def _model_generator(config):
            model = build_classifier(config.model)
            if hasattr(config, "quant_transformer") \
                and config.quant_transformer is not None:
                model_transformer = build_mtransformer(config.quant_transformer)
                return model_transformer(model, logger= None)
            else:
                return model

        self.student_model = _model_generator(self.student_config) 
        self.teacher_model = _model_generator(self.teacher_config)
        self.distill_loss = build_loss(distill_loss)
        self.init_weights()

    def init_weights(self, pretrained=None):
        def _init_weights(model, config):
            if hasattr(config, 'pre_train') and config['pre_train'] is not None:
                load_checkpoint(model, config['pre_train'], map_location = 'cpu')
            else:
                model.init_weights()
        _init_weights(self.student_model, self.student_config)
        _init_weights(self.teacher_model, self.teacher_config)

    def extract_feat(self, img):
        pass

    def forward_train(self, img, gt_label, **kwargs):
        r"""1. Extrace student features """
        student_feat = self.student_model.extract_feat(img)
        
        losses = dict()
        loss_stu_cls, feat_stu_cls = self.student_model.head.forward_train(student_feat, gt_label)

        with torch.no_grad():
            self.teacher_model.eval()
            teacher_feat = self.teacher_model.extract_feat(img)
            loss_teach_cls, feat_teach_cls = self.teacher_model.head.forward_train(teacher_feat, gt_label)
        
        loss_st_KD = self.distill_loss(feat_stu_cls, feat_teach_cls)
        
        losses.update(loss_stu_cls)
        #losses.update(loss_teach_cls)
        losses.update(loss_st_KD)

        return losses

    def simple_test(self, img):
        x = self.student_model.extract_feat(img)
        return self.student_model.head.simple_test(x)