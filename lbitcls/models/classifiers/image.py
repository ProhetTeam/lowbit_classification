import torch.nn as nn
import torch

import copy
import warnings
from ..utils.augment import Augments

from ..builder import CLASSIFIERS, build_backbone, build_head, build_neck
from .base import BaseClassifier
from QuanTransformer.quantrans.quantops.ABQAT import ABQATConv2d
from QuanTransformer.quantrans.quantops.LSQPlus import LSQDPlusConv2d

@CLASSIFIERS.register_module()
class ImageClassifier(BaseClassifier):

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None):
        super(ImageClassifier, self).__init__(init_cfg)

        if pretrained is not None:
            warnings.warn('DeprecationWarning: pretrained is a deprecated \
                key, please consider using init_cfg')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if head is not None:
            self.head = build_head(head)

        self.augments = None
        if train_cfg is not None:
            augments_cfg = train_cfg.get('augments', None)
            if augments_cfg is not None:
                self.augments = Augments(augments_cfg)
            else:
                # Considering BC-breaking
                mixup_cfg = train_cfg.get('mixup', None)
                cutmix_cfg = train_cfg.get('cutmix', None)
                assert mixup_cfg is None or cutmix_cfg is None, \
                    'If mixup and cutmix are set simultaneously,' \
                    'use augments instead.'
                if mixup_cfg is not None:
                    warnings.warn('The mixup attribute will be deprecated. '
                                  'Please use augments instead.')
                    cfg = copy.deepcopy(mixup_cfg)
                    cfg['type'] = 'BatchMixup'
                    # In the previous version, mixup_prob is always 1.0.
                    cfg['prob'] = 1.0
                    self.augments = Augments(cfg)
                if cutmix_cfg is not None:
                    warnings.warn('The cutmix attribute will be deprecated. '
                                  'Please use augments instead.')
                    cfg = copy.deepcopy(cutmix_cfg)
                    cutmix_prob = cfg.pop('cutmix_prob')
                    cfg['type'] = 'BatchCutMix'
                    cfg['prob'] = cutmix_prob
                    self.augments = Augments(cfg)

    def extract_feat(self, img):
        """Directly extract features from the backbone + neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self, img, gt_label, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            gt_label (Tensor): It should be of shape (N, 1) encoding the
                ground-truth label of input images for single label task. It
                shoulf be of shape (N, C) encoding the ground-truth label
                of input images for multi-labels task.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if self.augments is not None:
            img, gt_label = self.augments(img, gt_label)

        x = self.extract_feat(img)

        losses = dict()
        loss = self.head.forward_train(x, gt_label)
        losses.update(loss)

        loss_extra = {}
        for _, layer in self.named_modules():
            if isinstance(layer, ABQATConv2d):
                if hasattr(layer, 'weight_quant_loss'):
                    try:
                        loss_extra['weight_quant_loss'] += layer.weight_quant_loss
                    except:
                        loss_extra['weight_quant_loss'] = layer.weight_quant_loss

                if hasattr(layer, 'quant_error_loss'):
                    try:
                        loss_extra['quant_error_loss'] += layer.quant_error_loss
                    except:
                        loss_extra['quant_error_loss'] = layer.quant_error_loss
                        
                if hasattr(layer, 'act_quant_loss'):
                    try:
                        loss_extra['act_quant_loss'] += layer.act_quant_loss
                    except:
                        loss_extra['act_quant_loss'] = layer.act_quant_loss 
                        
            if isinstance(layer, LSQDPlusConv2d):
                if hasattr(layer, 'weight_quant_loss'):
                    try:
                        loss_extra['weight_quant_loss'] += layer.weight_quant_loss
                    except:
                        loss_extra['weight_quant_loss'] = layer.weight_quant_loss  
                if hasattr(layer, 'quant_error_loss'):
                    try:
                        loss_extra['quant_error_loss'] += layer.quant_error_loss
                    except:
                        loss_extra['quant_error_loss'] = layer.quant_error_loss
            
                if hasattr(layer, 'act_quant_loss'):
                    try:
                        loss_extra['act_quant_loss'] += layer.act_quant_loss
                    except:
                        loss_extra['act_quant_loss'] = layer.act_quant_loss 
                        
        losses.update(loss_extra) 
        return losses

    def simple_test(self, img):
        """Test without augmentation."""
        x = self.extract_feat(img)
        return self.head.simple_test(x)