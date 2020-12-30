# train_model_with_pytorch
使用 pytorch 训练模型

参考：[https://github.com/rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models)

## 使用

### 训练自己数据

- 数据准备

数据以文件夹的方式存储，需分为 `train` 和 `valid` 2个文件夹，每个文件夹中的每一个文件夹为一个类别，如下所示：

```
train_data_dir
  	train_dir
  		class0_dir
  			img0, img1, ..., imgN0
  		class1_dir
  			img0, img1, ..., imgN1
  		...
  		classM_dir
  			img0, img1, ..., imgNm
  	valid_dir
  		class0_dir
  			img0, img1, ..., imgN0
  		class1_dir
  			img0, img1, ..., imgN1
  		...
  		classM_dir
  			img0, img1, ..., imgNm
```

- 训练模型

```bash
# 从0开始训练
python train.py train_data_dir --model model_name --proj project_name --output out_dir
# 从预训练模型fineturn
python train.py train_data_dir --model model_name --proj project_name --pretrained --output out_dir
# 从指定checkpoint开始训练
python train.py train_data_dir --model model_name --proj project_name --initial cpt_path --output out_dir
```

这里使用了 [trains(clearml)](https://github.com/allegroai/clearml) 工具来记录实验过程，`project_name` 为其需要的项目名。

`--output` 指定 checkpoints 存放位置。

### inference

- 使用预训练模型

```bash
python inference.py ../pytorch-image-models/.data/vision/imagenet/ --model resnet18 --pretrained --img 224
```

这里 `../pytorch-image-models/.data/vision/imagenet/` 为测试样本所在的目录，其下每个目录为一个类别。

- 使用训练好模型：

```bash
python infer_for_single_image.py data_dir --img_size 224 --num_classes class_num --checkpoint cp_path
python infer_for_valid.py class_dir --img_size 224 --num_classes class_num --checkpoint cp_path
# data_dir，其下图像
# class_dir，其下每个目录为一个类别
```

### 查看支持模型

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

### 获取预训练模型

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

### 模型参数

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

