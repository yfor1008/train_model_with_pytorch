#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @File   : infer_for_valid.py
# @Author : yuanwenjin
# @Mail   : xxxx@mail.com
# @Date   : 2020/12/30 11:26:05
# @Docs   : 对测试集进行测试
'''

import os
import time
import argparse
import logging
import torch
import numpy as np

from PIL import Image
from timm.data.transforms_factory import create_transform

from timm.models import create_model
from timm.data import resolve_data_config
from timm.utils import setup_default_logging

parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--model', '-m', metavar='MODEL', default='dpn92',
                    help='model architecture (default: dpn92)')
parser.add_argument('--img_size', default=None, type=int,
                    metavar='N', help='Input image dimension')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--crop_pct', default=1.0, type=float,
                    metavar='N', help='需默认为1.0， 否则为0.875')
parser.add_argument('--num_classes', type=int, default=1000,
                    help='Number classes in dataset')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--log_path', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

def main():

    args = parser.parse_args()

    setup_default_logging(log_path=args.log_path)
    _logger = logging.getLogger('inference')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create model
    model = create_model(
        args.model,
        num_classes=args.num_classes,
        in_chans=3,
        checkpoint_path=args.checkpoint)

    data_config = resolve_data_config(vars(args), model=model) # 默认配置 mean, std
    img_transform = create_transform(
        data_config['input_size'],
        is_training=False,
        mean=data_config['mean'],
        std=data_config['std'],
        crop_pct=data_config['crop_pct']
    )

    model = model.to(device)
    model.eval()

    # confusion matrix
    conf_matrix = np.zeros((args.num_classes, args.num_classes))

    # predict
    class_dir = os.listdir(args.data)
    for cl in class_dir:
        imgs = os.listdir(os.path.join(args.data, cl))
        for img in imgs:
            im = Image.open(os.path.join(args.data, cl, img))
            im1 = img_transform(im).unsqueeze(0)
            im1 = im1.to(device)
            outputs = model(im1)
            idx = torch.max(outputs, 1)[1].tolist()[0]
            probs = torch.nn.functional.softmax(outputs, dim=1)[0].tolist()
            _logger.info('%s: %d, %.2f' % (os.path.join(args.data, cl, img), idx, probs[idx]))
            conf_matrix[int(cl), idx] += 1

    _logger.info(conf_matrix)

if __name__ == "__main__":
    main()
