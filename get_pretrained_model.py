#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @File   : get_pretrained_model.py
# @Author : yuanwenjin
# @Mail   : xxxx@mail.com
# @Date   : 2020/12/23 11:47:03
# @Docs   : 获取预训练好的模型
'''

import os
import timm

# 设置环境变量，改变模型存储位置
ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
model_dir = '/home/train_datas/pretrained_models'
os.environ.setdefault(ENV_XDG_CACHE_HOME, model_dir) # os.putenv 不起作用

model_name = 'resnet18'
# model_name = 'resnet50'
model = timm.create_model(model_name, pretrained=True) # 模型保存在 model_dir/torch/hub/checkpoints/
model.eval()

# 输出模型参数
print('Model %s created, param count: %d' % (model_name, sum([m.numel() for m in model.parameters()]))) # 获取模型参数量
