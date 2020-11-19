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
from .utils import dict_merge

@MTRANSFORMERS.register_module()
class mTransformerV2(Basemtransformer, nn.Module):
    def __init__(self, 
                 quan_policy = dict(),
                 first_layer = None, 
                 last_layer = None,
                 **kwargs):
        super(mTransformerV2, self).__init__()
        self.first_layer = first_layer
        self.last_layer = last_layer

        self.register_dict = OrderedDict()
        for key, value in quan_policy.items():
            assert(hasattr(nn, key))
            self.register_dict[getattr(nn, key)] = value
        self.layer_idx = 0

    def __call__(self, model, exclude_layers =[], logger = None, **kwargs):
        r""" Convert float Model to quantization Model
        Args:
            model(nn.Module): Standard Model
            excludes_layers(list): Some layers u dnot want to quatify
            lagger: logger 
        Return:
            New Model: replace with quantization layers
        return model
        """
        if len(self.register_dict) == 0:
            logger.info(f'There is NO layer to be quantified!')
            return model

        for module_name in model._modules:
            if len(model._modules[module_name]._modules) > 0:
                self.__call__(model._modules[module_name], exclude_layers, logger, **kwargs)
            else:
                if type(getattr(model, module_name)) not in self.register_dict:
                    continue
                if module_name in exclude_layers:
                    continue
                logger.info(f"\nTransform Layer Name :{module_name} ; {type(getattr(model, module_name)).__name__} -> {self.register_dict[type(getattr(model, module_name))]['type']}")
                
                current_layer = getattr(model, module_name)
                sig = inspect.signature(type(getattr(model, module_name)))
                new_kwargs = {}
                for key in sig.parameters:
                    if sig.parameters[key].default != inspect.Parameter.empty:
                        continue
                    assert(hasattr(current_layer, key))
                    new_kwargs[key] = getattr(current_layer, key)
                
                ## First layer quantization policy
                quan_args = self.register_dict[type(getattr(model, module_name))]
                if self.layer_idx == 0 and self.first_layer is not None:
                    quan_args = self.first_layer
                
                ## Last layer quantization policy
                if isinstance(current_layer, nn.Linear) and self.last_layer is not None:
                    quan_args = self.last_layer 

                new_kwargs = {**quan_args, **new_kwargs} #merge two args
                new_quan_layer = build_quanlayer(new_kwargs)
                dict_merge(new_quan_layer.__dict__, current_layer.__dict__)
                setattr(model, module_name, new_quan_layer)

                self.layer_idx += 1
        return model
