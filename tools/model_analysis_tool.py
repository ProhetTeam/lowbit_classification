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

#import git
#repo = git.Repo('.', search_parent_directories=True)
#sys.path.insert(0, str(repo.working_tree_dir))

from lbitcls import __version__
from lbitcls.datasets import build_dataset
from lbitcls.utils import collect_env, get_root_logger
from lbitcls.models import build_classifier
from lbitcls.apis import set_random_seed, train_classifier
from lbitcls.datasets.pipelines import Resize, CenterCrop

from thirdparty.mtransformer import build_mtransformer

from thirdparty.model_analysis_tool.ModelAnalyticalTool import ModelAnalyticalTool
from functools import partial
def parse_args():
    parser = argparse.ArgumentParser(description='visuliza a classifier')
    parser.add_argument('--config', default="/data/code/quant_project_zy/lowbit_classification/thirdparty/configs/visconfig_mobilenetv2_LSQ_int4.py", help='train config file path')
    #parser.add_argument('--img', default="/data/code/quant_project_zy/lowbit_classification", help='train config file path')
    parser.add_argument('--img', default="/data/code/quant_project_zy/lowbit_classification/vis_model.jpg", help='train config file path')
    
    # parser.add_argument('--checkpoint', default="", help='the dir to save logs and models')
    # parser.add_argument('--device', default='cuda:0', help='device use for analysis')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # model = build_classifier(cfg.model)
    # if hasattr(cfg, "quant_transformer"):
    #     model_transformer = build_mtransformer(cfg.quant_transformer)
    #     model = model_transformer(model, logger= logger)
    # load_checkpoint(model, cfg.load_from, strict=False)
    # model.eval()

    # args = parse_args()

    cfg = Config.fromfile(args.config)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    
    if args.img is None:
        img = torch.rand(1,3,224,224)
    else:
        img = mmcv.imread(args.img)
        img = mmcv.imresize(img,size=(256, 256))
        img_height, img_width = img.shape[:2]

        y1 = max(0, int(round((img_height - 224) / 2.)))
        x1 = max(0, int(round((img_width - 224) / 2.)))
        y2 = min(img_height, y1 + 224) - 1
        x2 = min(img_width, x1 + 224) - 1

        # crop the image
        img = mmcv.imcrop(img, bboxes=np.array([x1, y1, x2, y2]))
        img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0)
        #img = CenterCrop(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    logger.info(f"os.enviro:\n {os.environ} \n")

    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)

    # log some basic info
    #logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    # if args.seed is not None:
    #     logger.info(f'Set random seed to {args.seed}, '
    #                 f'deterministic: {args.deterministic}')
    #     set_random_seed(args.seed, deterministic=args.deterministic)

    model = build_classifier(cfg.model)
    if hasattr(cfg, "quant_transformer"):
        model_transformer = build_mtransformer(cfg.quant_transformer)
        model = model_transformer(model, logger= logger)

    load_checkpoint(model, cfg.load_from, strict=False)
    model.eval()
    r""" 1. Float32 Model analysis """
    model_analysis_tool = ModelAnalyticalTool(model, is_quant = True, save_path = './model_analysis_res18_int3_apot.html')

    model_analysis_tool.weight_dist_analysis(smaple_num = 8)
    
    infer_func = partial(infer)
    model_analysis_tool.activation_dist_analysis(infer_func, smaple_num = 8, img = img)

    model_analysis_tool.down_html()

def infer(model, img):
    # read img
    
    out= model(img, return_loss=False)
    # norm
    # out = model(img)
    
    return out

if __name__ == '__main__':
    main()
