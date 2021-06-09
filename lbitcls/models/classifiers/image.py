import torch.nn as nn
import torch
from ..builder import CLASSIFIERS, build_backbone, build_head, build_neck
from .base import BaseClassifier
from thirdparty.mtransformer.ABQAT import ABQATConv2d
from thirdparty.mtransformer.LSQPlus import LSQDPlusConv2d

@CLASSIFIERS.register_module()
class ImageClassifier(BaseClassifier):

    def __init__(self, backbone, neck=None, head=None, pretrained=None):
        super(ImageClassifier, self).__init__()
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if head is not None:
            self.head = build_head(head)

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(ImageClassifier, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_head:
            self.head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone + neck
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self, img, gt_label, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            gt_label (Tensor): of shape (N, 1) encoding the ground-truth label
                of input images.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
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
