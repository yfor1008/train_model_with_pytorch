# train_model_with_pytorch
使用 pytorch 训练模型

参考：[https://github.com/rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models)

## 使用

- inference

```bash
python inference.py ../pytorch-image-models/.data/vision/imagenet/ --model resnet18 --pretrained --img 224
```

这里 `../pytorch-image-models/.data/vision/imagenet/` 为测试样本所在的目录，其下每个目录为一个类别。

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
```

- 模型参数

```python
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
```

