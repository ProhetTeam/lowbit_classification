import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from ..builder import HEADS
from .cls_head import ClsHead
from mmcv.cnn import ConvModule, kaiming_init, normal_init

@HEADS.register_module()
class LinearClsHead(ClsHead):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 drop_out_ratio = None,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 topk=(1, )):
        super(LinearClsHead, self).__init__(loss=loss, topk=topk)
        self.in_channels = in_channels
        self.num_classes = num_classes

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')
        self.drop_out_ratio = drop_out_ratio 
        self._init_layers()

    def _init_layers(self):
        self.fc = nn.Linear(self.in_channels, self.num_classes)
        if self.drop_out_ratio is not None:
            self.drop_out = nn.Dropout(p = self.drop_out_ratio)

    def init_weights(self):
        normal_init(self.fc, mean=0, std=0.01, bias=0)

    def simple_test(self, img):
        """Test without augmentation."""
        cls_score = self.fc(img)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = F.softmax(cls_score, dim=1) if cls_score is not None else None
        if torch.onnx.is_in_onnx_export():
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred

    def forward_train(self, x, gt_label):
        if self.drop_out_ratio is not None:
            cls_score = self.fc(self.drop_out(x))
        else:
            cls_score = self.fc(x)
        losses = self.loss(cls_score, gt_label)
        return losses

@HEADS.register_module()
class ConvClsHead(ClsHead):
    """Convolution classifier head for MobileNetV3. 
    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        mid_channels (int): Number of channels in the hidden stage.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        loss (dict): Config of classification loss.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 mid_channels,
                 conv_cfg=None,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 topk=(1, )):
        super(ConvClsHead, self).__init__(loss=loss, topk=topk)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.conv_cfg = conv_cfg
        # TODO: add 'act_cfg‘？

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self._init_layers()

    def _init_layers(self):
        self.conv = ConvModule(
            in_channels=self.in_channels,
            out_channels=self.mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=self.conv_cfg,
            norm_cfg=None,
            act_cfg=dict(type='HSwish'))
        self.fc = nn.Linear(self.mid_channels, self.num_classes)

    def init_weights(self):
        # self.conv is automatically initialized inside ConvModule
        normal_init(self.fc, mean=0, std=0.01, bias=0)

    def simple_test(self, img):
        """Test without augmentation."""
        img = img.view(img.size(0), img.size(1), 1, 1)
        conv_feat = self.conv(img)
        conv_feat = conv_feat.view(conv_feat.size(0), -1)
        cls_score = self.fc(conv_feat)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = F.softmax(cls_score, dim=1) if cls_score is not None else None
        if torch.onnx.is_in_onnx_export():
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred

    def forward_train(self, x, gt_label):
        # TODO: Can not process tuple
        x = x.view(x.size(0), x.size(1), 1, 1)
        conv_feat = self.conv(x)
        conv_feat = conv_feat.view(conv_feat.size(0), -1)
        cls_score = self.fc(conv_feat)
        losses = self.loss(cls_score, gt_label)
        return losses