import torch.nn as nn
import torch
from ..builder import CLASSIFIERS, build_backbone, build_head, build_neck
from .base import BaseClassifier
from lbitcls.mtransformer.DSQ import DSQConv, DSQConvV2


@CLASSIFIERS.register_module()
class DSQImageClassifier(BaseClassifier):

    def __init__(self, backbone, neck=None, head=None, pretrained=None, 
            alpha_weight = 1.0):
        super(DSQImageClassifier, self).__init__()
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if head is not None:
            self.head = build_head(head)
        self.alpha_weight = alpha_weight
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(DSQImageClassifier, self).init_weights(pretrained)
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

        loss_alpha = {"loss_alpha":0.}
        counter = 0
        for name, layer in self.named_modules():
            if isinstance(layer, (DSQConv, DSQConvV2)):
                if hasattr(layer, 'alphaW'):
                    loss_alpha['loss_alpha'] += torch.abs(layer.alphaW)
                    counter += 1
                if hasattr(layer, 'alphaB'):
                    loss_alpha['loss_alpha'] += torch.abs(layer.alphaB)
                    counter += 1
                if hasattr(layer, 'alphaA'):
                    loss_alpha['loss_alpha'] += torch.abs(layer.alphaA)
                    counter += 1
        loss_alpha['loss_alpha'] = loss_alpha['loss_alpha'] / counter
        losses.update(loss_alpha)
        
        return losses

    def simple_test(self, img):
        """Test without augmentation."""
        x = self.extract_feat(img)
        return self.head.simple_test(x)
