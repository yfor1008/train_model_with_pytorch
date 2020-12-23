#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @File   : get_model_param_nums.py
# @Author : yuanwenjin
# @Mail   : xxxx@mail.com
# @Date   : 2020/12/23 15:36:45
# @Docs   : 获取模型参数量
'''

import os
import torch
from torchsummary import summary
import timm

# 设置device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设置环境变量，改变模型存储位置
ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
model_dir = '/home/train_datas/pretrained_models'
os.environ.setdefault(ENV_XDG_CACHE_HOME, model_dir) # os.putenv 不起作用

model_name = 'resnet18'
# model_name = 'resnet50'
model = timm.create_model(model_name, pretrained=True).to(device)
model.eval()

# 模型参数
summary(model, input_size=(3, 224, 224), batch_size=-1) # 需指定输入大小
