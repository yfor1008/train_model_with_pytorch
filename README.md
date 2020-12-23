# train_model_with_pytorch
使用 pytorch 训练模型

参考：[https://github.com/rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models)

## 使用

- 查看支持模型

```python
from pprint import pprint
import timm
model_names = timm.list_models('*') # 所有模型
pprint(model_names)
# list_models(filter='', module='', pretrained=False, exclude_filters='')
# filter：对模型名称进行筛选，支持正则
# module：对模型模块进行筛选
# pretrained：筛选是否有预训练模型
# exclude_filters：排除模型
```

- 获取预训练模型

```python
import timm
model_name = 'resnet18'
model = timm.create_model(model_name, pretrained=True)
model.eval()
print('Model %s created, param count: %d' % (model_name, sum([m.numel() for m in model.parameters()]))) # 获取模型参数量
# 模型默认会保存在 ~/.cache/torch/checkpoints/ 目录(在容器内，则保存在 /root/.cache/torch/checkpoints/)
```

