#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @File   : get_model_list.py
# @Author : yuanwenjin
# @Mail   : xxxx@mail.com
# @Date   : 2020/12/23 11:33:28
# @Docs   : 获取支持模型
'''

from pprint import pprint
import timm

# model_names = timm.list_models('*') # 所有模型
# pprint(model_names)

model_names = timm.list_models(pretrained=True) # 有预训练的模型
pprint(model_names)
