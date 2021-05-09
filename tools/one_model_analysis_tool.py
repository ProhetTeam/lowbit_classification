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
from mmcv.runner import init_dist
from mmcv.runner import load_checkpoint
from argparse import ArgumentParser

from lbitcls import __version__
from lbitcls.datasets import build_dataset
from lbitcls.utils import collect_env, get_root_logger
from lbitcls.models import build_classifier
from lbitcls.apis import set_random_seed, train_classifier
from lbitcls.datasets.pipelines import Resize, CenterCrop

from thirdparty.mtransformer import build_mtransformer
from functools import partial
from thirdparty.model_analysis_tool.ModelAnalyticalToolV2 import QModelAnalysis
from thirdparty.model_analysis_tool.OneModelAnalyticalTool import OneModelAnalysis
from lbitcls.apis import init_model, inference_model

def infer(model, img):
    out= model(img, return_loss=False)
    return out

def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help = 'Int checkpoint file')
    parser.add_argument('--save-path', type = str, default= "./model_analysis.html", help = "html save path")
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    model = init_model(args.config, args.checkpoint, device = args.device)

    r""" 1. Float32 Model analysis """
    model_analysis_tool = OneModelAnalysis(model, 
                                         smaple_num = 20, 
                                         max_data_length = 2e4, 
                                         bin_size = 0.01, 
                                         save_path = args.save_path,
                                         use_torch_plot = True)
    model_analysis_tool(inference_model, img = args.img)
    model_analysis_tool.down_html()
    
if __name__ == '__main__':
    main()
