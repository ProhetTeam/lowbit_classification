from .cls_head import ClsHead
from .linear_head import LinearClsHead, ConvClsHead
from .stacked_head import StackedLinearClsHead

__all__ = ['ClsHead', 'LinearClsHead', 'ConvClsHead', 'StackedLinearClsHead']
