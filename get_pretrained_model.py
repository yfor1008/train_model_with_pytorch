#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @File   : get_pretrained_model.py
# @Author : yuanwenjin
# @Mail   : xxxx@mail.com
# @Date   : 2020/12/23 11:47:03
# @Docs   : 获取预训练好的模型
'''

import timm
model_name = 'resnet18'
model = timm.create_model(model_name, pretrained=True)
model.eval()
print('Model %s created, param count: %d' % (model_name, sum([m.numel() for m in model.parameters()]))) # 获取模型参数量
# 模型默认会保存在 ~/.cache/torch/checkpoints/ 目录(在容器内，则保存在 /root/.cache/torch/checkpoints/)
