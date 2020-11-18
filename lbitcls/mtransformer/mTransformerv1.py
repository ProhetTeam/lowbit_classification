from .Basemtransformer import Basemtransformer
import time
import copy
import types
import inspect 
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from .Basemtransformer import Basemtransformer
from .builder import MTRANSFORMERS
from .builder import build_quanlayer

@MTRANSFORMERS.register_module()
class mTransformerV1(Basemtransformer, nn.Module):
    def __init__(self, **kwargs):
        super(mTransformerV1, self).__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __call__(self, model, replace_layer = True, **kwargs):
        
        return model

        

