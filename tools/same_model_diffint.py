import argparse
import copy
import os, sys
import os.path as osp
import time
import cv2
import numpy as np
import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import checkpoint, init_dist
from mmcv.runner import load_checkpoint
from argparse import ArgumentParser, Namespace

from lbitcls import __version__
from lbitcls import models
from lbitcls.datasets import build_dataset
from lbitcls.utils import collect_env, get_root_logger
from lbitcls.models import build_classifier
from lbitcls.apis import set_random_seed, train_classifier
from lbitcls.datasets.pipelines import Resize, CenterCrop

from thirdparty.mtransformer import build_mtransformer
from functools import partial
from thirdparty.model_analysis_tool.MultiModelCmp import MultiModelCmp
from lbitcls.apis import init_model, inference_model


def infer(model, img):
    out= model(img, return_loss=False)
    return out


configs = ['thirdparty/configs/benchmark/config1_res18_float_1m_b64.py',
           'thirdparty/configs/LSQDPlus/config4_res18_lsqdplus_int4_updatelr4x_weightloss_4m.py',
           'thirdparty/configs/LSQDPlus/config6_res18_lsqdplus_int3_allchangenoweightloss_4m.py',]
checkpoints = ['thirdparty/modelzoo/res18.pth',
               'work_dirs/LSQDPlus/config4_res18_lsqdplus_int4_updatelr4x_weightloss_4m/latest.pth',
               'work_dirs/LSQDPlus/config6_res18_lsqdplus_int3_allchangenoweightloss_4m/latest.pth']
extra_names = ['r18-fp32', 
               'r18-int4',
               'r18-int3']


def main():
    parser = ArgumentParser()
    parser.add_argument('--img', help='Image file', default= '1260,3f000afab0a06')
    parser.add_argument('--save-path', 
        type = str, default= "./model_analysis.html", help = "html save path")
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    models = []
    for idx, config in enumerate(configs):
        models.append(init_model(config, checkpoints[idx], device = args.device))

    r""" 1. Float32 Model analysis """
    model_analysis_tool = MultiModelCmp(models, 
                                        smaple_num = 20, 
                                        max_data_length = 2e4, 
                                        bin_size = 0.01, 
                                        save_path = args.save_path,
                                        extra_names = extra_names,
                                        use_torch_plot = True)
    model_analysis_tool(inference_model, img = args.img)
    model_analysis_tool.down_html()


if __name__ == '__main__':
    main()
