# 0. Intro



## 0.1 目录

[TOC]

## 0.2 论文、代码、数据集与阅读笔记

- paper: [https://openreview.net/forum?id=LzQQ89U1qm_](https://openreview.net/forum?id=LzQQ89U1qm_)

- codes: https://github.com/thuml/Anomaly-Transformer
- datasets(不完整): https://drive.google.com/drive/folders/1gisthCoE-RrKJ0j3KPV7xiibhHWT9qRm?usp=sharing
- datasets()
- my review: 

# 1. Startup

初始条件介绍和必要准备工作，代码来自https://github.com/thuml/Anomaly-Transformer，论文数据来自作者提供的[Google Cloud](https://drive.google.com/drive/folders/1gisthCoE-RrKJ0j3KPV7xiibhHWT9qRm?usp=sharing)

## 初始环境信息

显卡：耕升GTX 1660 6GB

CPU：Intel i7-10700 2.90GHz

内存：16GB DDR4

系统：Ubuntu 20.04.1 内核5.15.0-89-generic (非虚拟机)

CUDA：release 11.5

显卡驱动信息：

```bash
Thu Nov 30 16:24:15 2023       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce GTX 1660        Off | 00000000:01:00.0  On |                  N/A |
| 77%   78C    P0              96W / 120W |   5636MiB /  6144MiB |     99%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
```

## 安装Pytorch 1.8.0

在已经安装conda 22.9.0，并用conda创建了python 3.6虚拟环境（环境命名为Anomaly-Transformer）的前提下，尝试使用`conda`安装pytorch（失败，网络原因导致较大文件下载失败，`conda`参照了[这个链接](https://mirror.tuna.tsinghua.edu.cn/help/anaconda/)换清华源仍然无法解决）

```bash
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

因此尝试pip安装（成功，可以正常使用`import torch`）

```bash
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```

一开始比较困扰的就是CUDA版本对应的问题，但后面看似乎`cudatoolkit`版本和机器安装的`CUDA`版本不用完全对应也能安装并使用上pytorch.

# 2. 论文实验复现

将论文提出方法应用到SDM、PSM、MSL、SMAP、SWaT共计五个数据集，复现文章评估数据。

## 2.1 SMD

作为第一个登场的脚本，很明显是要报一堆大大小小的错的。还好问题都不大，搞定了后面就畅通无阻了。

### 首次运行SMD.sh

```bash
(Anomaly-Transformer) username@username-ubuntu:/media/username/folder/Dev/Anomaly-Transformer$ bash ./scripts/SMD.sh
./scripts/SMD.sh: line 2: $'\r': command not found
Traceback (most recent call last):
  File "main.py", line 7, in <module>
    from solver import Solver
  File "/media/username/folder/Dev/Anomaly-Transformer/solver.py", line 9, in <module>
    from data_factory.data_loader import get_loader_segment
  File "/media/username/folder/Dev/Anomaly-Transformer/data_factory/data_loader.py", line 11, in <module>
    import pandas as pd
ModuleNotFoundError: No module named 'pandas'
Traceback (most recent call last):
  File "main.py", line 7, in <module>
    from solver import Solver
  File "/media/username/folder/Dev/Anomaly-Transformer/solver.py", line 9, in <module>
    from data_factory.data_loader import get_loader_segment
  File "/media/username/folder/Dev/Anomaly-Transformer/data_factory/data_loader.py", line 11, in <module>
    import pandas as pd
ModuleNotFoundError: No module named 'pandas'
```

**问题定位与解决**：可见问题主要都是package缺失，缺失package和安装命令如下：

- `sklearn`: 命令行输入`pip install scikit-learn`
- `pandas` : 命令行输入`pip install pandas`

### 再次运行SMD.sh

```bash
(Anomaly-Transformer) username@username-ubuntu:/media/username/folder/Dev/Anomaly-Transformer$ bash ./scripts/SMD.sh
./scripts/SMD.sh: line 2: $'\r': command not found
------------ Options -------------
anormly_ratio: 0.5
batch_size: 256
data_path: dataset/SMD
dataset: SMD
input_c: 38
k: 3
lr: 0.0001
mode: train
model_save_path: checkpoints
num_epochs: 10
output_c: 38
pretrained_model: None
win_size: 100
-------------- End ----------------
Traceback (most recent call last):
  File "main.py", line 52, in <module>
    main(config)
  File "main.py", line 18, in main
    solver = Solver(vars(config))
  File "/media/username/folder/Dev/Anomaly-Transformer/solver.py", line 74, in __init__
    dataset=self.dataset)
  File "/media/username/folder/Dev/Anomaly-Transformer/data_factory/data_loader.py", line 204, in get_loader_segment
    dataset = SMDSegLoader(data_path, win_size, step, mode)
  File "/media/username/folder/Dev/Anomaly-Transformer/data_factory/data_loader.py", line 166, in __init__
    data = np.load(data_path + "/SMD_train.npy")
  File "/home/username/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/numpy/lib/npyio.py", line 416, in load
    fid = stack.enter_context(open(os_fspath(file), "rb"))
FileNotFoundError: [Errno 2] No such file or directory: 'dataset/SMD/SMD_train.npy'
------------ Options -------------
anormly_ratio: 0.5
batch_size: 256
data_path: dataset/SMD
dataset: SMD
input_c: 38
k: 3
lr: 0.0001
mode: test
model_save_path: checkpoints
num_epochs: 10
output_c: 38
pretrained_model: 20
win_size: 100
-------------- End ----------------
Traceback (most recent call last):
  File "main.py", line 52, in <module>
    main(config)
  File "main.py", line 18, in main
    solver = Solver(vars(config))
  File "/media/username/folder/Dev/Anomaly-Transformer/solver.py", line 74, in __init__
    dataset=self.dataset)
  File "/media/username/folder/Dev/Anomaly-Transformer/data_factory/data_loader.py", line 204, in get_loader_segment
    dataset = SMDSegLoader(data_path, win_size, step, mode)
  File "/media/username/folder/Dev/Anomaly-Transformer/data_factory/data_loader.py", line 166, in __init__
    data = np.load(data_path + "/SMD_train.npy")
  File "/home/username/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/numpy/lib/npyio.py", line 416, in load
    fid = stack.enter_context(open(os_fspath(file), "rb"))
FileNotFoundError: [Errno 2] No such file or directory: 'dataset/SMD/SMD_train.npy'

```

**问题定位与解决**：问题主要为数据集文件找不到：`'dataset/SMD/SMD_train.npy'`，根据该提示将下载的数据集文件([Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/9605612594f0423f891e/) or [Google Cloud](https://drive.google.com/drive/folders/1gisthCoE-RrKJ0j3KPV7xiibhHWT9qRm?usp=sharing))整理后按照如下结构存放：

```
Anomoly_Transformer/
├── dataset/
│     ├── SMD/
│     │    ├── SMD_test.npy
│     │    ├── SMD_train.npy
│     │    └── ......
│     ├── PSM/
│     │    ├── test.csv
│     │    ├── train.csv
│     │    └── ......
│     ├── MSL/
│     │    ├── MSL_test.npy
│     │    └── ......
│     └── SMAP/
│          ├── SMAP_test.npy
│          └── ......
└── ......
```

### 第三次及之后运行SMD.sh

```bash
(Anomaly-Transformer) username@username-ubuntu:/media/username/folder/Dev/Anomaly-Transformer$ bash ./scripts/SMD.sh
./scripts/SMD.sh: line 2: $'\r': command not found
------------ Options -------------
anormly_ratio: 0.5
batch_size: 256
data_path: dataset/SMD
dataset: SMD
input_c: 38
k: 3
lr: 0.0001
mode: train
model_save_path: checkpoints
num_epochs: 10
output_c: 38
pretrained_model: None
win_size: 100
-------------- End ----------------
======================TRAIN MODE======================
Traceback (most recent call last):
  File "main.py", line 52, in <module>
    main(config)
  File "main.py", line 21, in main
    solver.train()
  File "/media/username/folder/Dev/Anomaly-Transformer/solver.py", line 161, in train
    self.win_size)).detach())) + torch.mean(
  File "/media/username/folder/Dev/Anomaly-Transformer/solver.py", line 13, in my_kl_loss
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
RuntimeError: CUDA out of memory. Tried to allocate 80.00 MiB (GPU 0; 5.79 GiB total capacity; 3.97 GiB already allocated; 49.75 MiB free; 4.11 GiB reserved in total by PyTorch)
------------ Options -------------
anormly_ratio: 0.5
batch_size: 256
data_path: dataset/SMD
dataset: SMD
input_c: 38
k: 3
lr: 0.0001
mode: test
model_save_path: checkpoints
num_epochs: 10
output_c: 38
pretrained_model: 20
win_size: 100
-------------- End ----------------
Traceback (most recent call last):
  File "main.py", line 52, in <module>
    main(config)
  File "main.py", line 23, in main
    solver.test()
  File "/media/username/folder/Dev/Anomaly-Transformer/solver.py", line 210, in test
    os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')))
  File "/home/username/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/serialization.py", line 579, in load
    with _open_file_like(f, 'rb') as opened_file:
  File "/home/username/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/serialization.py", line 230, in _open_file_like
    return _open_file(name_or_buffer, mode)
  File "/home/username/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/serialization.py", line 211, in __init__
    super(_open_file, self).__init__(open(name, mode))
FileNotFoundError: [Errno 2] No such file or directory: 'checkpoints/SMD_checkpoint.pth'

```

**问题定位与解决**：

- 问题1：CUDA out of memory: `RuntimeError: CUDA out of memory. Tried to allocate 80.00 MiB (GPU 0; 5.79 GiB total capacity; 3.97 GiB already allocated; 49.75 MiB free; 4.11 GiB reserved in total by PyTorch)`，初步认为是CUDA显存分配问题，模型所需显存没有得到满足。
- 问题2：模型checkpoint文件缺失：由于训练未成功进行，使得模型checkpoint文件沒有成功生成，从而在test阶段想要读取模型时无法读取。

因此应该围绕CUDA显存分配优化进行研究。

解决过程：

- 在[博文](https://blog.csdn.net/m0_50502579/article/details/126059178)中找到方案1：减小batch_size
- 尝试将启动命令中训练与测试的`batch_size`均从`256`改为`128`，然后重新运行`./scripts/SMD.sh`
- 仍然爆显存：`RuntimeError: CUDA out of memory. Tried to allocate 40.00 MiB (GPU 0; 5.79 GiB total capacity; 3.89 GiB already allocated; 82.94 MiB free; 4.05 GiB reserved in total by PyTorch)`
- 尝试修改`batch_size`为`64`，然后重新运行`./scripts/SMD.sh`
- 问题依旧，尝试修改`batch_size`为`32`，然后重新运行`./scripts/SMD.sh`
- 成功开始训练，迹象为观察到如下训练过程打印的epoch信息：

```bash
======================TRAIN MODE======================
        speed: 0.1335s/iter; left time: 283.1503s
        speed: 0.1289s/iter; left time: 260.5845s
Epoch: 1 cost time: 29.139591455459595
Epoch: 1, Steps: 222 | Train Loss: -40.3103769 Vali Loss: -46.1086967 
Validation loss decreased (inf --> -46.108697).  Saving model ...
Updating learning rate to 0.0001
        speed: 0.2505s/iter; left time: 475.7060s
        speed: 0.1302s/iter; left time: 234.2822s
Epoch: 2 cost time: 28.97248649597168
```

在经过四个epoch后停止，进入test阶段，并输出了最终实验结果：

```bash
Threshold : 0.06388568006455485
pred:    (708400,)
gt:      (708400,)
pred:  (708400,)
gt:    (708400,)
Accuracy : 0.9926, Precision : 0.8927, Recall : 0.9329, F-score : 0.9124 
```

完整的训练、测试过程控制台输出如下：

```bash
(Anomaly-Transformer) username@username-ubuntu:/media/username/folder/Dev/Anomaly-Transformer$ bash ./scripts/SMD.sh
------------ Options -------------
anormly_ratio: 0.5
batch_size: 32
data_path: dataset/SMD
dataset: SMD
input_c: 38
k: 3
lr: 0.0001
mode: train
model_save_path: checkpoints
num_epochs: 10
output_c: 38
pretrained_model: None
win_size: 100
-------------- End ----------------
======================TRAIN MODE======================
        speed: 0.1335s/iter; left time: 283.1503s
        speed: 0.1289s/iter; left time: 260.5845s
Epoch: 1 cost time: 29.139591455459595
Epoch: 1, Steps: 222 | Train Loss: -40.3103769 Vali Loss: -46.1086967 
Validation loss decreased (inf --> -46.108697).  Saving model ...
Updating learning rate to 0.0001
        speed: 0.2505s/iter; left time: 475.7060s
        speed: 0.1302s/iter; left time: 234.2822s
Epoch: 2 cost time: 28.97248649597168
Epoch: 2, Steps: 222 | Train Loss: -47.4852449 Vali Loss: -46.8629997 
EarlyStopping counter: 1 out of 3
Updating learning rate to 5e-05
        speed: 0.2555s/iter; left time: 428.5185s
        speed: 0.1307s/iter; left time: 206.1918s
Epoch: 3 cost time: 29.593196392059326
Epoch: 3, Steps: 222 | Train Loss: -47.8205990 Vali Loss: -47.0798451 
EarlyStopping counter: 2 out of 3
Updating learning rate to 2.5e-05
        speed: 0.2540s/iter; left time: 369.4981s
        speed: 0.1327s/iter; left time: 179.8330s
Epoch: 4 cost time: 29.744439840316772
Epoch: 4, Steps: 222 | Train Loss: -47.9206608 Vali Loss: -47.1366013 
EarlyStopping counter: 3 out of 3
Early stopping
------------ Options -------------
anormly_ratio: 0.5
batch_size: 32
data_path: dataset/SMD
dataset: SMD
input_c: 38
k: 3
lr: 0.0001
mode: test
model_save_path: checkpoints
num_epochs: 10
output_c: 38
pretrained_model: 20
win_size: 100
-------------- End ----------------
======================TEST MODE======================
/home/username/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Threshold : 0.06388568006455485
pred:    (708400,)
gt:      (708400,)
pred:  (708400,)
gt:    (708400,)
Accuracy : 0.9926, Precision : 0.8927, Recall : 0.9329, F-score : 0.9124 
```

## 2.2 PSM

### 首次运行PSM.sh

成功结束，测试结果摘要如下：

```bash
======================TEST MODE======================
Threshold : 0.0011754722148179996
pred:    (87800,)
gt:      (87800,)
pred:  (87800,)
gt:    (87800,)
Accuracy : 0.9882, Precision : 0.9697, Recall : 0.9883, F-score : 0.9789
```



完整执行过程如下：

```bash

nomaly-Transformer$ bash ./scripts/PSM.sh
------------ Options -------------
anormly_ratio: 1.0
batch_size: 32
data_path: dataset/PSM
dataset: PSM
input_c: 25
k: 3
lr: 0.0001
mode: train
model_save_path: checkpoints
num_epochs: 3
output_c: 25
pretrained_model: None
win_size: 100
-------------- End ----------------
test: (87841, 25)
train: (132481, 25)
test: (87841, 25)
train: (132481, 25)
test: (87841, 25)
train: (132481, 25)
test: (87841, 25)
train: (132481, 25)
======================TRAIN MODE======================
        speed: 0.1336s/iter; left time: 1644.4129s
        speed: 0.1275s/iter; left time: 1556.9812s
        speed: 0.1276s/iter; left time: 1545.0969s
        speed: 0.1277s/iter; left time: 1534.3159s
        speed: 0.1279s/iter; left time: 1524.0642s
        speed: 0.1279s/iter; left time: 1510.8930s
        speed: 0.1279s/iter; left time: 1498.2409s
        speed: 0.1279s/iter; left time: 1485.3983s
        speed: 0.1279s/iter; left time: 1472.8507s
        speed: 0.1280s/iter; left time: 1460.4547s
        speed: 0.1280s/iter; left time: 1448.4515s
        speed: 0.1282s/iter; left time: 1437.4174s
        speed: 0.1284s/iter; left time: 1427.1299s
        speed: 0.1284s/iter; left time: 1414.0394s
        speed: 0.1284s/iter; left time: 1401.1101s
        speed: 0.1285s/iter; left time: 1389.8639s
        speed: 0.1284s/iter; left time: 1375.8026s
        speed: 0.1283s/iter; left time: 1361.4072s
        speed: 0.1284s/iter; left time: 1349.9602s
        speed: 0.1284s/iter; left time: 1336.6942s
        speed: 0.1282s/iter; left time: 1322.4420s
        speed: 0.1283s/iter; left time: 1310.0931s
        speed: 0.1283s/iter; left time: 1297.5508s
        speed: 0.1282s/iter; left time: 1283.9510s
        speed: 0.1283s/iter; left time: 1271.7995s
        speed: 0.1283s/iter; left time: 1259.0883s
        speed: 0.1283s/iter; left time: 1245.8020s
        speed: 0.1283s/iter; left time: 1233.0175s
        speed: 0.1283s/iter; left time: 1220.1547s
        speed: 0.1283s/iter; left time: 1207.7776s
        speed: 0.1284s/iter; left time: 1195.7177s
        speed: 0.1282s/iter; left time: 1181.4120s
        speed: 0.1283s/iter; left time: 1168.8951s
        speed: 0.1282s/iter; left time: 1155.4520s
        speed: 0.1283s/iter; left time: 1143.4009s
        speed: 0.1284s/iter; left time: 1131.5084s
        speed: 0.1283s/iter; left time: 1117.7446s
        speed: 0.1282s/iter; left time: 1104.4219s
        speed: 0.1282s/iter; left time: 1091.2835s
        speed: 0.1283s/iter; left time: 1078.9449s
        speed: 0.1282s/iter; left time: 1065.9970s
Epoch: 1 cost time: 531.1504812240601
Epoch: 1, Steps: 4137 | Train Loss: -48.0091480 Vali Loss: -48.8543076 
Validation loss decreased (inf --> -48.854308).  Saving model ...
Updating learning rate to 0.0001
        speed: 1.2588s/iter; left time: 10290.9493s
        speed: 0.1282s/iter; left time: 1035.4373s
        speed: 0.1282s/iter; left time: 1022.6818s
        speed: 0.1283s/iter; left time: 1010.5991s
        speed: 0.1282s/iter; left time: 996.7476s
        speed: 0.1283s/iter; left time: 984.5289s
        speed: 0.1282s/iter; left time: 971.1445s
        speed: 0.1282s/iter; left time: 958.5275s
        speed: 0.1282s/iter; left time: 945.7043s
        speed: 0.1283s/iter; left time: 933.1298s
        speed: 0.1282s/iter; left time: 919.9409s
        speed: 0.1282s/iter; left time: 907.2530s
        speed: 0.1282s/iter; left time: 894.4075s
        speed: 0.1282s/iter; left time: 881.5417s
        speed: 0.1283s/iter; left time: 869.0542s
        speed: 0.1283s/iter; left time: 856.4396s
        speed: 0.1284s/iter; left time: 844.1700s
        speed: 0.1283s/iter; left time: 830.4890s
        speed: 0.1282s/iter; left time: 816.9863s
        speed: 0.1283s/iter; left time: 804.9645s
        speed: 0.1282s/iter; left time: 791.7809s
        speed: 0.1283s/iter; left time: 779.5495s
        speed: 0.1283s/iter; left time: 766.7960s
        speed: 0.1282s/iter; left time: 753.4139s
        speed: 0.1282s/iter; left time: 740.1944s
        speed: 0.1282s/iter; left time: 727.6711s
        speed: 0.1284s/iter; left time: 715.8365s
        speed: 0.1282s/iter; left time: 701.6651s
        speed: 0.1283s/iter; left time: 689.5141s
        speed: 0.1282s/iter; left time: 676.0763s
        speed: 0.1282s/iter; left time: 663.4497s
        speed: 0.1282s/iter; left time: 650.6223s
        speed: 0.1284s/iter; left time: 638.5416s
        speed: 0.1282s/iter; left time: 625.1153s
        speed: 0.1283s/iter; left time: 612.6931s
        speed: 0.1283s/iter; left time: 599.7292s
        speed: 0.1282s/iter; left time: 586.6867s
        speed: 0.1284s/iter; left time: 574.6260s
        speed: 0.1283s/iter; left time: 561.4819s
        speed: 0.1283s/iter; left time: 548.4647s
        speed: 0.1283s/iter; left time: 535.5813s
Epoch: 2 cost time: 530.542858839035
Epoch: 2, Steps: 4137 | Train Loss: -48.9527894 Vali Loss: -48.9326362 
EarlyStopping counter: 1 out of 3
Updating learning rate to 5e-05
        speed: 1.2538s/iter; left time: 5062.9567s
        speed: 0.1284s/iter; left time: 505.7279s
        speed: 0.1284s/iter; left time: 492.9298s
        speed: 0.1283s/iter; left time: 479.5802s
        speed: 0.1282s/iter; left time: 466.2639s
        speed: 0.1283s/iter; left time: 453.8794s
        speed: 0.1284s/iter; left time: 441.3263s
        speed: 0.1282s/iter; left time: 428.0605s
        speed: 0.1284s/iter; left time: 415.8170s
        speed: 0.1283s/iter; left time: 402.4540s
        speed: 0.1282s/iter; left time: 389.4098s
        speed: 0.1283s/iter; left time: 376.9801s
        speed: 0.1283s/iter; left time: 364.0838s
        speed: 0.1283s/iter; left time: 351.2112s
        speed: 0.1282s/iter; left time: 338.1965s
        speed: 0.1283s/iter; left time: 325.5066s
        speed: 0.1282s/iter; left time: 312.6431s
        speed: 0.1284s/iter; left time: 300.1481s
        speed: 0.1283s/iter; left time: 287.0474s
        speed: 0.1284s/iter; left time: 274.4572s
        speed: 0.1282s/iter; left time: 261.2532s
        speed: 0.1282s/iter; left time: 248.4272s
        speed: 0.1282s/iter; left time: 235.6939s
        speed: 0.1282s/iter; left time: 222.7621s
        speed: 0.1282s/iter; left time: 209.9875s
        speed: 0.1282s/iter; left time: 197.1853s
        speed: 0.1282s/iter; left time: 184.3661s
        speed: 0.1283s/iter; left time: 171.6811s
        speed: 0.1285s/iter; left time: 159.0394s
        speed: 0.1283s/iter; left time: 145.9588s
        speed: 0.1283s/iter; left time: 133.1463s
        speed: 0.1283s/iter; left time: 120.3105s
        speed: 0.1282s/iter; left time: 107.4170s
        speed: 0.1283s/iter; left time: 94.6591s
        speed: 0.1282s/iter; left time: 81.8221s
        speed: 0.1282s/iter; left time: 68.9838s
        speed: 0.1282s/iter; left time: 56.1650s
        speed: 0.1282s/iter; left time: 43.3404s
        speed: 0.1283s/iter; left time: 30.5237s
        speed: 0.1282s/iter; left time: 17.6921s
        speed: 0.1282s/iter; left time: 4.8703s
Epoch: 3 cost time: 530.5573189258575
Epoch: 3, Steps: 4137 | Train Loss: -48.9824078 Vali Loss: -48.9623636 
EarlyStopping counter: 2 out of 3
Updating learning rate to 2.5e-05
------------ Options -------------
anormly_ratio: 1.0
batch_size: 32
data_path: dataset/PSM
dataset: PSM
input_c: 25
k: 3
lr: 0.0001
mode: test
model_save_path: checkpoints
num_epochs: 10
output_c: 25
pretrained_model: 20
win_size: 100
-------------- End ----------------
test: (87841, 25)
train: (132481, 25)
test: (87841, 25)
train: (132481, 25)
test: (87841, 25)
train: (132481, 25)
test: (87841, 25)
train: (132481, 25)
======================TEST MODE======================
/home/username/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Threshold : 0.0011754722148179996
pred:    (87800,)
gt:      (87800,)
pred:  (87800,)
gt:    (87800,)
Accuracy : 0.9882, Precision : 0.9697, Recall : 0.9883, F-score : 0.9789

```



## 2.3 MSL

### 首次运行MSL.sh

```bash
(Anomaly-Transformer) username@username-ubuntu:/media/username/folder/Dev/Anomaly-Transformer$ bash ./scripts/MSL.sh
------------ Options -------------
anormly_ratio: 1.0
batch_size: 32
data_path: dataset/MSL
dataset: MSL
input_c: 55
k: 3
lr: 0.0001
mode: train
model_save_path: checkpoints
num_epochs: 3
output_c: 55
pretrained_model: None
win_size: 100
-------------- End ----------------
test: (73729, 55)
train: (58317, 55)
test: (73729, 55)
train: (58317, 55)
test: (73729, 55)
train: (58317, 55)
test: (73729, 55)
train: (58317, 55)
Traceback (most recent call last):
  File "main.py", line 52, in <module>
    main(config)
  File "main.py", line 18, in main
    solver = Solver(vars(config))
  File "/media/username/folder/Dev/Anomaly-Transformer/solver.py", line 85, in __init__
    self.build_model()
  File "/media/username/folder/Dev/Anomaly-Transformer/solver.py", line 90, in build_model
    self.model = AnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3)
  File "/media/username/folder/Dev/Anomaly-Transformer/model/AnomalyTransformer.py", line 77, in __init__
    ) for l in range(e_layers)
  File "/media/username/folder/Dev/Anomaly-Transformer/model/AnomalyTransformer.py", line 77, in <listcomp>
    ) for l in range(e_layers)
  File "/media/username/folder/Dev/Anomaly-Transformer/model/attn.py", line 29, in __init__
    self.distances = torch.zeros((window_size, window_size)).cuda()
  File "/home/username/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/cuda/__init__.py", line 170, in _lazy_init
    torch._C._cuda_init()
RuntimeError: No CUDA GPUs are available
------------ Options -------------
anormly_ratio: 1.0
batch_size: 32
data_path: dataset/MSL
dataset: MSL
input_c: 55
k: 3
lr: 0.0001
mode: test
model_save_path: checkpoints
num_epochs: 10
output_c: 55
pretrained_model: 20
win_size: 100
-------------- End ----------------
test: (73729, 55)
train: (58317, 55)
test: (73729, 55)
train: (58317, 55)
test: (73729, 55)
train: (58317, 55)
test: (73729, 55)
train: (58317, 55)
Traceback (most recent call last):
  File "main.py", line 52, in <module>
    main(config)
  File "main.py", line 18, in main
    solver = Solver(vars(config))
  File "/media/username/folder/Dev/Anomaly-Transformer/solver.py", line 85, in __init__
    self.build_model()
  File "/media/username/folder/Dev/Anomaly-Transformer/solver.py", line 90, in build_model
    self.model = AnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3)
  File "/media/username/folder/Dev/Anomaly-Transformer/model/AnomalyTransformer.py", line 77, in __init__
    ) for l in range(e_layers)
  File "/media/username/folder/Dev/Anomaly-Transformer/model/AnomalyTransformer.py", line 77, in <listcomp>
    ) for l in range(e_layers)
  File "/media/username/folder/Dev/Anomaly-Transformer/model/attn.py", line 29, in __init__
    self.distances = torch.zeros((window_size, window_size)).cuda()
  File "/home/username/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/cuda/__init__.py", line 170, in _lazy_init
    torch._C._cuda_init()
RuntimeError: No CUDA GPUs are available

```

运行失败，原因为`RuntimeError: No CUDA GPUs are available`，不知道为什么GPU不可用了。尝试看看GPU是否可用。

```bash
(Anomaly-Transformer) username@username-ubuntu:/media/username/folder/Dev/Anomaly-Transformer$ python
Python 3.6.13 |Anaconda, Inc.| (default, Jun  4 2021, 14:25:59) 
[GCC 7.5.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> print(torch.cuda.device_count())
1
>>> print(torch.cuda.is_available())
True
```

结果正常？？？再尝试运行`MSL.sh`仍然有问题，检查`MSL.sh`本身，发现第一行有问题：

```bash
export CUDA_VISIBLE_DEVICES=7
```

因为电脑只有一张显卡，序号不应该是7，应该为0。修改后再运行`MSL.sh`，正常了...(大无语，干嘛突然写个7，其他脚本的明明都是0)

### 再次运行MSL.sh

修改GPU序号后正常运行，测试结果摘要如下

```bash
======================TEST MODE======================
Threshold : 0.0012788161612115718
pred:    (73700,)
gt:      (73700,)
pred:  (73700,)
gt:    (73700,)
Accuracy : 0.9863, Precision : 0.9186, Recall : 0.9545, F-score : 0.9362 
```

完整执行过程如下：

```bash
(Anomaly-Transformer) username@username-ubuntu:/media/username/folder/Dev/Anomaly-Transformer$ bash ./scripts/MSL.sh
------------ Options -------------
anormly_ratio: 1.0
batch_size: 32
data_path: dataset/MSL
dataset: MSL
input_c: 55
k: 3
lr: 0.0001
mode: train
model_save_path: checkpoints
num_epochs: 3
output_c: 55
pretrained_model: None
win_size: 100
-------------- End ----------------
test: (73729, 55)
train: (58317, 55)
test: (73729, 55)
train: (58317, 55)
test: (73729, 55)
train: (58317, 55)
test: (73729, 55)
train: (58317, 55)
======================TRAIN MODE======================
        speed: 0.1424s/iter; left time: 763.5606s
        speed: 0.1358s/iter; left time: 714.4375s
        speed: 0.1387s/iter; left time: 716.0470s
        speed: 0.1354s/iter; left time: 685.1058s
        speed: 0.1357s/iter; left time: 673.0052s
        speed: 0.1402s/iter; left time: 681.4136s
        speed: 0.1380s/iter; left time: 656.8551s
        speed: 0.1355s/iter; left time: 631.5301s
        speed: 0.1360s/iter; left time: 620.2718s
        speed: 0.1350s/iter; left time: 602.2681s
        speed: 0.1344s/iter; left time: 586.1590s
        speed: 0.1352s/iter; left time: 576.0738s
        speed: 0.1372s/iter; left time: 570.8950s
        speed: 0.1358s/iter; left time: 551.6753s
        speed: 0.1326s/iter; left time: 525.1665s
        speed: 0.1336s/iter; left time: 515.8068s
        speed: 0.1349s/iter; left time: 507.2338s
        speed: 0.1348s/iter; left time: 493.3304s
Epoch: 1 cost time: 247.81738114356995
Epoch: 1, Steps: 1820 | Train Loss: -47.0458832 Vali Loss: -46.7697310 
Validation loss decreased (inf --> -46.769731).  Saving model ...
Updating learning rate to 0.0001
        speed: 1.1043s/iter; left time: 3910.1910s
        speed: 0.1326s/iter; left time: 456.2046s
        speed: 0.1330s/iter; left time: 444.4217s
        speed: 0.1336s/iter; left time: 433.0193s
        speed: 0.1396s/iter; left time: 438.4048s
        speed: 0.1384s/iter; left time: 420.9290s
        speed: 0.1358s/iter; left time: 399.4554s
        speed: 0.1363s/iter; left time: 387.1776s
        speed: 0.1354s/iter; left time: 371.1777s
        speed: 0.1354s/iter; left time: 357.5956s
        speed: 0.1351s/iter; left time: 343.2376s
        speed: 0.1355s/iter; left time: 330.8024s
        speed: 0.1362s/iter; left time: 318.7658s
        speed: 0.1367s/iter; left time: 306.3689s
        speed: 0.1363s/iter; left time: 291.7184s
        speed: 0.1358s/iter; left time: 277.2149s
        speed: 0.1362s/iter; left time: 264.4203s
        speed: 0.1352s/iter; left time: 248.9006s
Epoch: 2 cost time: 246.60186314582825
Epoch: 2, Steps: 1820 | Train Loss: -48.5221037 Vali Loss: -47.3841785 
EarlyStopping counter: 1 out of 3
Updating learning rate to 5e-05
        speed: 1.1290s/iter; left time: 1942.9595s
        speed: 0.1394s/iter; left time: 225.9686s
        speed: 0.1351s/iter; left time: 205.5129s
        speed: 0.1406s/iter; left time: 199.7962s
        speed: 0.1332s/iter; left time: 175.9856s
        speed: 0.1326s/iter; left time: 161.8460s
        speed: 0.1314s/iter; left time: 147.2958s
        speed: 0.1334s/iter; left time: 136.1576s
        speed: 0.1319s/iter; left time: 121.5173s
        speed: 0.1389s/iter; left time: 114.0600s
        speed: 0.1306s/iter; left time: 94.1768s
        speed: 0.1396s/iter; left time: 86.6974s
        speed: 0.1352s/iter; left time: 70.4401s
        speed: 0.1373s/iter; left time: 57.8087s
        speed: 0.1379s/iter; left time: 44.2689s
        speed: 0.1322s/iter; left time: 29.2235s
        speed: 0.1308s/iter; left time: 15.8220s
        speed: 0.1308s/iter; left time: 2.7468s
Epoch: 3 cost time: 245.10069799423218
Epoch: 3, Steps: 1820 | Train Loss: -48.7357392 Vali Loss: -47.5481951 
EarlyStopping counter: 2 out of 3
Updating learning rate to 2.5e-05
------------ Options -------------
anormly_ratio: 1.0
batch_size: 32
data_path: dataset/MSL
dataset: MSL
input_c: 55
k: 3
lr: 0.0001
mode: test
model_save_path: checkpoints
num_epochs: 10
output_c: 55
pretrained_model: 20
win_size: 100
-------------- End ----------------
test: (73729, 55)
train: (58317, 55)
test: (73729, 55)
train: (58317, 55)
test: (73729, 55)
train: (58317, 55)
test: (73729, 55)
train: (58317, 55)
======================TEST MODE======================
/home/username/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Threshold : 0.0012788161612115718
pred:    (73700,)
gt:      (73700,)
pred:  (73700,)
gt:    (73700,)
Accuracy : 0.9863, Precision : 0.9186, Recall : 0.9545, F-score : 0.9362 
```

## 2.4 SMAP

### 首次运行SMAP.sh

这次留了个心眼看看脚本第一行的GPU编号是否正确，没有问题，得到测试结果摘要如下：

```bash
======================TEST MODE======================
Threshold : 0.0005670388956787038
pred:    (427600,)
gt:      (427600,)
pred:  (427600,)
gt:    (427600,)
Accuracy : 0.9906, Precision : 0.9360, Recall : 0.9943, F-score : 0.9642 

```

完整执行过程如下（跑得太久，机器都快烤熟了）：

```bash
(Anomaly-Transformer) username@username-ubuntu:/media/username/folder/Dev/Anomaly-Transformer$ bash ./scripts/SMAP.sh
------------ Options -------------
anormly_ratio: 1.0
batch_size: 32
data_path: dataset/SMAP
dataset: SMAP
input_c: 25
k: 3
lr: 0.0001
mode: train
model_save_path: checkpoints
num_epochs: 3
output_c: 25
pretrained_model: None
win_size: 100
-------------- End ----------------
test: (427617, 25)
train: (135183, 25)
test: (427617, 25)
train: (135183, 25)
test: (427617, 25)
train: (135183, 25)
test: (427617, 25)
train: (135183, 25)
======================TRAIN MODE======================
        speed: 0.1343s/iter; left time: 1687.4161s
        speed: 0.1316s/iter; left time: 1641.0434s
        speed: 0.1304s/iter; left time: 1612.4711s
        speed: 0.1304s/iter; left time: 1599.8954s
        speed: 0.1315s/iter; left time: 1600.4497s
        speed: 0.1304s/iter; left time: 1573.3904s
        speed: 0.1306s/iter; left time: 1563.0526s
        speed: 0.1313s/iter; left time: 1557.9986s
        speed: 0.1308s/iter; left time: 1539.4388s
        speed: 0.1302s/iter; left time: 1518.6118s
        speed: 0.1312s/iter; left time: 1518.1523s
        speed: 0.1305s/iter; left time: 1495.9211s
        speed: 0.1306s/iter; left time: 1484.8276s
        speed: 0.1306s/iter; left time: 1471.1267s
        speed: 0.1297s/iter; left time: 1447.8968s
        speed: 0.1304s/iter; left time: 1442.7176s
        speed: 0.1299s/iter; left time: 1424.9521s
        speed: 0.1303s/iter; left time: 1415.9097s
        speed: 0.1309s/iter; left time: 1409.4994s
        speed: 0.1311s/iter; left time: 1398.0175s
        speed: 0.1318s/iter; left time: 1392.9594s
        speed: 0.1302s/iter; left time: 1362.8479s
        speed: 0.1371s/iter; left time: 1421.0785s
        speed: 0.1292s/iter; left time: 1326.9539s
        speed: 0.1303s/iter; left time: 1324.7232s
        speed: 0.1308s/iter; left time: 1316.6065s
        speed: 0.1309s/iter; left time: 1304.8822s
        speed: 0.1306s/iter; left time: 1289.0181s
        speed: 0.1322s/iter; left time: 1291.2752s
        speed: 0.1315s/iter; left time: 1271.0320s
        speed: 0.1302s/iter; left time: 1245.4013s
        speed: 0.1310s/iter; left time: 1240.1241s
        speed: 0.1309s/iter; left time: 1225.9448s
        speed: 0.1300s/iter; left time: 1204.4843s
        speed: 0.1308s/iter; left time: 1198.7496s
        speed: 0.1329s/iter; left time: 1205.0089s
        speed: 0.1319s/iter; left time: 1183.1681s
        speed: 0.1301s/iter; left time: 1153.9812s
        speed: 0.1295s/iter; left time: 1135.2198s
        speed: 0.1307s/iter; left time: 1132.5606s
        speed: 0.1312s/iter; left time: 1124.1417s
        speed: 0.1296s/iter; left time: 1097.1383s
Epoch: 1 cost time: 553.0207221508026
Epoch: 1, Steps: 4222 | Train Loss: -47.6426614 Vali Loss: -48.1685601 
Validation loss decreased (inf --> -48.168560).  Saving model ...
Updating learning rate to 0.0001
        speed: 5.5265s/iter; left time: 46118.9601s
        speed: 0.1295s/iter; left time: 1067.8902s
        speed: 0.1297s/iter; left time: 1056.0485s
        speed: 0.1296s/iter; left time: 1042.8939s
        speed: 0.1328s/iter; left time: 1055.0172s
        speed: 0.1347s/iter; left time: 1056.7791s
        speed: 0.1300s/iter; left time: 1006.6005s
        speed: 0.1293s/iter; left time: 988.4494s
        speed: 0.1292s/iter; left time: 975.1445s
        speed: 0.1294s/iter; left time: 963.6841s
        speed: 0.1292s/iter; left time: 948.7126s
        speed: 0.1292s/iter; left time: 936.1060s
        speed: 0.1291s/iter; left time: 922.7592s
        speed: 0.1291s/iter; left time: 909.7682s
        speed: 0.1291s/iter; left time: 896.7182s
        speed: 0.1290s/iter; left time: 882.9915s
        speed: 0.1291s/iter; left time: 870.9842s
        speed: 0.1289s/iter; left time: 856.8340s
        speed: 0.1289s/iter; left time: 843.7893s
        speed: 0.1291s/iter; left time: 831.9244s
        speed: 0.1292s/iter; left time: 819.9622s
        speed: 0.1297s/iter; left time: 809.7022s
        speed: 0.1293s/iter; left time: 794.2553s
        speed: 0.1292s/iter; left time: 781.1323s
        speed: 0.1292s/iter; left time: 767.8188s
        speed: 0.1293s/iter; left time: 755.7132s
        speed: 0.1292s/iter; left time: 742.2035s
        speed: 0.1293s/iter; left time: 729.9284s
        speed: 0.1294s/iter; left time: 717.5859s
        speed: 0.1293s/iter; left time: 703.9006s
        speed: 0.1292s/iter; left time: 690.3800s
        speed: 0.1291s/iter; left time: 677.3184s
        speed: 0.1293s/iter; left time: 665.2619s
        speed: 0.1292s/iter; left time: 651.7931s
        speed: 0.1292s/iter; left time: 638.8412s
        speed: 0.1293s/iter; left time: 626.6065s
        speed: 0.1292s/iter; left time: 612.8525s
        speed: 0.1291s/iter; left time: 599.8719s
        speed: 0.1292s/iter; left time: 587.1467s
        speed: 0.1292s/iter; left time: 574.4164s
        speed: 0.1293s/iter; left time: 561.6398s
        speed: 0.1291s/iter; left time: 548.2016s
Epoch: 2 cost time: 546.9801330566406
Epoch: 2, Steps: 4222 | Train Loss: -48.5213919 Vali Loss: -48.2957534 
EarlyStopping counter: 1 out of 3
Updating learning rate to 5e-05
        speed: 5.4772s/iter; left time: 22582.6544s
        speed: 0.1305s/iter; left time: 524.9768s
        speed: 0.1301s/iter; left time: 510.4672s
        speed: 0.1292s/iter; left time: 494.0773s
        speed: 0.1291s/iter; left time: 480.7389s
        speed: 0.1293s/iter; left time: 468.5919s
        speed: 0.1292s/iter; left time: 455.1257s
        speed: 0.1292s/iter; left time: 442.3313s
        speed: 0.1293s/iter; left time: 429.8049s
        speed: 0.1293s/iter; left time: 416.6019s
        speed: 0.1291s/iter; left time: 403.3154s
        speed: 0.1292s/iter; left time: 390.4252s
        speed: 0.1291s/iter; left time: 377.3882s
        speed: 0.1292s/iter; left time: 364.6556s
        speed: 0.1293s/iter; left time: 352.1070s
        speed: 0.1291s/iter; left time: 338.6508s
        speed: 0.1292s/iter; left time: 325.8527s
        speed: 0.1291s/iter; left time: 312.7774s
        speed: 0.1292s/iter; left time: 300.1695s
        speed: 0.1291s/iter; left time: 286.9356s
        speed: 0.1291s/iter; left time: 274.0536s
        speed: 0.1299s/iter; left time: 262.8002s
        speed: 0.1324s/iter; left time: 254.5154s
        speed: 0.1298s/iter; left time: 236.6313s
        speed: 0.1328s/iter; left time: 228.8171s
        speed: 0.1327s/iter; left time: 215.4497s
        speed: 0.1304s/iter; left time: 198.6513s
        speed: 0.1295s/iter; left time: 184.3195s
        speed: 0.1299s/iter; left time: 171.8900s
        speed: 0.1292s/iter; left time: 157.9532s
        speed: 0.1290s/iter; left time: 144.8993s
        speed: 0.1292s/iter; left time: 132.2070s
        speed: 0.1293s/iter; left time: 119.3879s
        speed: 0.1291s/iter; left time: 106.2423s
        speed: 0.1291s/iter; left time: 93.3420s
        speed: 0.1291s/iter; left time: 80.4567s
        speed: 0.1292s/iter; left time: 67.5827s
        speed: 0.1292s/iter; left time: 54.6509s
        speed: 0.1293s/iter; left time: 41.7594s
        speed: 0.1292s/iter; left time: 28.8161s
        speed: 0.1292s/iter; left time: 15.8970s
        speed: 0.1291s/iter; left time: 2.9699s
Epoch: 3 cost time: 547.1292362213135
Epoch: 3, Steps: 4222 | Train Loss: -48.6120459 Vali Loss: -48.3690009 
EarlyStopping counter: 2 out of 3
Updating learning rate to 2.5e-05
------------ Options -------------
anormly_ratio: 1.0
batch_size: 32
data_path: dataset/SMAP
dataset: SMAP
input_c: 25
k: 3
lr: 0.0001
mode: test
model_save_path: checkpoints
num_epochs: 10
output_c: 25
pretrained_model: 20
win_size: 100
-------------- End ----------------
test: (427617, 25)
train: (135183, 25)
test: (427617, 25)
train: (135183, 25)
test: (427617, 25)
train: (135183, 25)
test: (427617, 25)
train: (135183, 25)
======================TEST MODE======================
/home/username/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Threshold : 0.0005670388956787038
pred:    (427600,)
gt:      (427600,)
pred:  (427600,)
gt:    (427600,)
Accuracy : 0.9906, Precision : 0.9360, Recall : 0.9943, F-score : 0.9642 
```

## 2.5 SWaT

此数据集较为特殊，体现在其获取和使用上。在数据集获取上，根据协议无法与他人分享，需要自行申请。因此我前往iTrust官网申请，只选择SWaT数据集即可（[填表链接](https://docs.google.com/forms/d/e/1FAIpQLSdwOIR-LuFnSu5cIAzun5OQtWXcsOhmC7NtTbb-LBI1MyOcug/viewform?usp=sf_link)），并等一了天得到邮件回复。在数据集使用上，作者没有编写训练脚本，因此需要自己对数据集进行处理并使用模型进行训练和测试。

拿到数据集后，根据论文附件K中Table 13推断论文使用的是2015年版本的数据集，即SWaT共享的Google Drive中，`SWAT/SWaT.A1&A2_Dec 2015/Physical/`下的`SWaT_dataset_Attack_v0.xlsx`（作测试集）和`SWaT_dataset_Normal_v1.xlsx`（作训练集）。下面进行简单的处理。

### 数据集处理

```python
# 1. 使用表格软件打开两者，分别删除第一行(第一行不是标题，只有P1,P2等字符，第二行的标题需要保留)后均保存为csv文件
# 2. 将两者用Python进行简单检查，转为numpy矩阵并保存为npy文件，代码如下：

import numpy as np
import pandas as pd

swat_train_pd = pd.read_csv('./dataset/SWaT/SWaT_Dataset_Normal_v1.csv')
swat_test_pd = pd.read_csv('./dataset/SWaT/SWaT_Dataset_Attack_v0.csv')

print(swat_train_pd.shape)
print(swat_test_pd.shape)
print(swat_test_pd['Normal/Attack'].unique())
print(swat_test_pd.head())
"""
(495000, 53)
(449919, 53)
['Normal' 'Attack' 'A ttack']
                 Timestamp    FIT101    LIT101  ...  P602  P603  Normal/Attack
0   28/12/2015 10:00:00 AM  2.427057  522.8467  ...     1     1         Normal
1   28/12/2015 10:00:01 AM  2.446274  522.8860  ...     1     1         Normal
2   28/12/2015 10:00:02 AM  2.489191  522.8467  ...     1     1         Normal
3   28/12/2015 10:00:03 AM  2.534350  522.9645  ...     1     1         Normal
4   28/12/2015 10:00:04 AM  2.569260  523.4748  ...     1     1         Normal

[5 rows x 53 columns]
"""

swat_test_pd = swat_test_pd.replace('Normal',0).replace('Attack',1).replace('A ttack',1)
swat_test_label_np = swat_test_pd.iloc[:,52].values
swat_test_np = swat_test_pd.drop([' Timestamp','Normal/Attack'], axis=1).values
swat_train_np = swat_train_pd.drop([' Timestamp','Normal/Attack'], axis=1).values

print(swat_train_np.shape)
print(swat_test_np.shape)
print(swat_test_label_np.shape)
"""
(495000, 51)
(449919, 51)
(449919,)
"""

np.save('./dataset/SWaT/swat_test_label.npy',swat_test_label_np)
np.save('./dataset/SWaT/swat_train.npy',swat_train_np)
np.save('./dataset/SWaT/swat_test.npy',swat_test_np)
```

然后新建训练测试脚本`./scripts/SWaT.sh`，内容是从`./scripts/Start.sh`复制的

```bash
export CUDA_VISIBLE_DEVICES=0

python main.py --anormly_ratio 0.5 --num_epochs 3    --batch_size 32  --mode train --dataset SWaT  --data_path dataset/SWaT --input_c 51    --output_c 51
python main.py --anormly_ratio 0.1  --num_epochs 10        --batch_size 32     --mode test    --dataset SWaT   --data_path dataset/SWaT  --input_c 51    --output_c 51  --pretrained_model 10
```

接着为SWaT数据集添加dataloder。编辑`./data_factory/data_loader.py`，添加一个`SwatSegLoader`类并修改原有`get_loader_segment`函数：

```python
'''
Loader for SWaT dataset
'''
class SwatSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/swat_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/swat_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/swat_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

            
"""
Add a new line about the SWaT dataset
"""            
def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', dataset='KDD'):
    if (dataset == 'SMD'):
        dataset = SMDSegLoader(data_path, win_size, step, mode)
    elif (dataset == 'MSL'):
        dataset = MSLSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'SMAP'):
        dataset = SMAPSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'PSM'):
        dataset = PSMSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'SWaT'): # added this
        dataset = SwatSegLoader(data_path, win_size, 1, mode)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader

```

### **首次运行SWaT.sh**

得到测试结果摘要如下：

```bash
======================TEST MODE======================
Threshold : 0.0031170047065244427
pred:    (449900,)
gt:      (449900,)
pred:  (449900,)
gt:    (449900,)
Accuracy : 0.9775, Precision : 0.8841, Recall : 0.9371, F-score : 0.9099 
```

完整执行过程如下（跑了大概两个小时）：

```bash
(Anomaly-Transformer) username@username-ubuntu:/media/username/3E6E20236E1FD28F/Dev/Anomaly-Transformer$ bash ./scripts/SWaT.sh
------------ Options -------------
anormly_ratio: 0.1
batch_size: 32
data_path: dataset/SWaT
dataset: SWaT
input_c: 51
k: 3
lr: 0.0001
mode: test
model_save_path: checkpoints
num_epochs: 10
output_c: 51
pretrained_model: 10
win_size: 100
-------------- End ----------------
test: (449919, 51)
train: (496800, 51)
test: (449919, 51)
train: (496800, 51)
test: (449919, 51)
train: (496800, 51)
test: (449919, 51)
train: (496800, 51)
======================TEST MODE======================
/home/username/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Threshold : 0.0032192275498528246
pred:    (449900,)
gt:      (449900,)
pred:  (449900,)
gt:    (449900,)
Accuracy : 0.9771, Precision : 0.8965, Recall : 0.9172, F-score : 0.9067 
(Anomaly-Transformer) username@username-ubuntu:/media/username/3E6E20236E1FD28F/Dev/Anomaly-Transformer$ bash ./scripts/SWaT.sh
------------ Options -------------
anormly_ratio: 0.5
batch_size: 32
data_path: dataset/SWaT
dataset: SWaT
input_c: 51
k: 3
lr: 0.0001
mode: train
model_save_path: checkpoints
num_epochs: 3
output_c: 51
pretrained_model: None
win_size: 100
-------------- End ----------------
test: (449919, 51)
train: (495000, 51)
test: (449919, 51)
train: (495000, 51)
test: (449919, 51)
train: (495000, 51)
test: (449919, 51)
train: (495000, 51)
======================TRAIN MODE======================
        speed: 0.1428s/iter; left time: 6612.5713s
        speed: 0.1354s/iter; left time: 6253.0974s
        speed: 0.1352s/iter; left time: 6230.7968s
        speed: 0.1355s/iter; left time: 6233.4215s
        speed: 0.1390s/iter; left time: 6378.5659s
        speed: 0.1403s/iter; left time: 6425.6394s
        speed: 0.1348s/iter; left time: 6161.5534s
        speed: 0.1359s/iter; left time: 6194.7005s
        speed: 0.1356s/iter; left time: 6168.4861s
        speed: 0.1360s/iter; left time: 6174.7164s
        speed: 0.1395s/iter; left time: 6319.8666s
        speed: 0.1389s/iter; left time: 6276.0570s
        speed: 0.1371s/iter; left time: 6183.6705s
        speed: 0.1401s/iter; left time: 6302.9684s
        speed: 0.1351s/iter; left time: 6067.7535s
        speed: 0.1351s/iter; left time: 6050.1461s
        speed: 0.1365s/iter; left time: 6101.5390s
        speed: 0.1378s/iter; left time: 6144.6511s
        speed: 0.1372s/iter; left time: 6104.3526s
        speed: 0.1382s/iter; left time: 6137.7459s
        speed: 0.1381s/iter; left time: 6117.2903s
        speed: 0.1371s/iter; left time: 6061.2191s
        speed: 0.1365s/iter; left time: 6021.1155s
        speed: 0.1345s/iter; left time: 5915.7146s
        speed: 0.1349s/iter; left time: 5921.9364s
        speed: 0.1355s/iter; left time: 5933.2968s
        speed: 0.1354s/iter; left time: 5914.8026s
        speed: 0.1424s/iter; left time: 6210.5447s
        speed: 0.1359s/iter; left time: 5911.6384s
        speed: 0.1374s/iter; left time: 5964.3514s
        speed: 0.1348s/iter; left time: 5838.2409s
        speed: 0.1346s/iter; left time: 5816.1968s
        speed: 0.1346s/iter; left time: 5802.6837s
        speed: 0.1346s/iter; left time: 5788.3701s
        speed: 0.1353s/iter; left time: 5804.4665s
        speed: 0.1394s/iter; left time: 5966.4150s
        speed: 0.1423s/iter; left time: 6074.2209s
        speed: 0.1420s/iter; left time: 6049.9245s
        speed: 0.1345s/iter; left time: 5715.0185s
        speed: 0.1344s/iter; left time: 5698.0888s
        speed: 0.1411s/iter; left time: 5968.1633s
        speed: 0.1415s/iter; left time: 5970.2509s
        speed: 0.1408s/iter; left time: 5928.4565s
        speed: 0.1426s/iter; left time: 5990.2664s
        speed: 0.1357s/iter; left time: 5686.5547s
        speed: 0.1365s/iter; left time: 5704.3444s
        speed: 0.1358s/iter; left time: 5664.3863s
        speed: 0.1392s/iter; left time: 5791.4018s
        speed: 0.1372s/iter; left time: 5693.4794s
        speed: 0.1362s/iter; left time: 5638.4746s
        speed: 0.1376s/iter; left time: 5682.1785s
        speed: 0.1382s/iter; left time: 5695.7126s
        speed: 0.1381s/iter; left time: 5673.8861s
        speed: 0.1428s/iter; left time: 5854.1201s
        speed: 0.1353s/iter; left time: 5535.4870s
        speed: 0.1358s/iter; left time: 5538.6886s
        speed: 0.1359s/iter; left time: 5530.9134s
        speed: 0.1360s/iter; left time: 5520.0075s
        speed: 0.1369s/iter; left time: 5545.7690s
        speed: 0.1353s/iter; left time: 5466.8770s
        speed: 0.1356s/iter; left time: 5465.3225s
        speed: 0.1344s/iter; left time: 5403.5386s
        speed: 0.1346s/iter; left time: 5398.9813s
        speed: 0.1364s/iter; left time: 5455.0500s
        speed: 0.1344s/iter; left time: 5362.2032s
        speed: 0.1346s/iter; left time: 5357.8926s
        speed: 0.1349s/iter; left time: 5356.4118s
        speed: 0.1358s/iter; left time: 5375.5669s
        speed: 0.1393s/iter; left time: 5503.1650s
        speed: 0.1394s/iter; left time: 5492.4838s
        speed: 0.1436s/iter; left time: 5641.9017s
        speed: 0.1382s/iter; left time: 5417.8900s
        speed: 0.1361s/iter; left time: 5320.7390s
        speed: 0.1370s/iter; left time: 5342.9212s
        speed: 0.1406s/iter; left time: 5470.8222s
        speed: 0.1363s/iter; left time: 5289.7796s
        speed: 0.1354s/iter; left time: 5240.4470s
        speed: 0.1352s/iter; left time: 5218.9477s
        speed: 0.1415s/iter; left time: 5446.3757s
        speed: 0.1396s/iter; left time: 5360.8597s
        speed: 0.1403s/iter; left time: 5371.5712s
        speed: 0.1355s/iter; left time: 5174.8979s
        speed: 0.1347s/iter; left time: 5132.1327s
        speed: 0.1356s/iter; left time: 5153.9954s
        speed: 0.1357s/iter; left time: 5144.1700s
        speed: 0.1364s/iter; left time: 5156.1387s
        speed: 0.1370s/iter; left time: 5164.2171s
        speed: 0.1385s/iter; left time: 5208.6199s
        speed: 0.1450s/iter; left time: 5438.2515s
        speed: 0.1492s/iter; left time: 5580.7061s
        speed: 0.1395s/iter; left time: 5203.9755s
        speed: 0.1372s/iter; left time: 5103.0719s
        speed: 0.1402s/iter; left time: 5202.6695s
        speed: 0.1426s/iter; left time: 5274.6904s
        speed: 0.1393s/iter; left time: 5141.1819s
        speed: 0.1393s/iter; left time: 5126.7243s
        speed: 0.1356s/iter; left time: 4976.0627s
        speed: 0.1359s/iter; left time: 4975.5315s
        speed: 0.1387s/iter; left time: 5061.0901s
        speed: 0.1418s/iter; left time: 5162.7360s
        speed: 0.1377s/iter; left time: 4997.3233s
        speed: 0.1360s/iter; left time: 4922.3810s
        speed: 0.1459s/iter; left time: 5267.9064s
        speed: 0.1375s/iter; left time: 4949.4103s
        speed: 0.1393s/iter; left time: 5001.8771s
        speed: 0.1465s/iter; left time: 5245.6815s
        speed: 0.1383s/iter; left time: 4938.3803s
        speed: 0.1372s/iter; left time: 4884.7422s
        speed: 0.1372s/iter; left time: 4871.4987s
        speed: 0.1378s/iter; left time: 4877.6222s
        speed: 0.1345s/iter; left time: 4748.5213s
        speed: 0.1344s/iter; left time: 4730.7891s
        speed: 0.1346s/iter; left time: 4725.0120s
        speed: 0.1359s/iter; left time: 4756.4741s
        speed: 0.1351s/iter; left time: 4715.7454s
        speed: 0.1344s/iter; left time: 4676.1957s
        speed: 0.1359s/iter; left time: 4716.7439s
        speed: 0.1349s/iter; left time: 4668.0923s
        speed: 0.1355s/iter; left time: 4673.7719s
        speed: 0.1369s/iter; left time: 4708.6927s
        speed: 0.1346s/iter; left time: 4615.7443s
        speed: 0.1419s/iter; left time: 4851.2301s
        speed: 0.1420s/iter; left time: 4842.0394s
        speed: 0.1390s/iter; left time: 4726.7629s
        speed: 0.1400s/iter; left time: 4745.2550s
        speed: 0.1417s/iter; left time: 4789.7382s
        speed: 0.1388s/iter; left time: 4677.9071s
        speed: 0.1359s/iter; left time: 4566.2085s
        speed: 0.1362s/iter; left time: 4561.5282s
        speed: 0.1412s/iter; left time: 4717.2056s
        speed: 0.1399s/iter; left time: 4657.4927s
        speed: 0.1398s/iter; left time: 4639.9572s
        speed: 0.1386s/iter; left time: 4586.4466s
        speed: 0.1437s/iter; left time: 4742.9178s
        speed: 0.1414s/iter; left time: 4653.5077s
        speed: 0.1389s/iter; left time: 4555.4527s
        speed: 0.1407s/iter; left time: 4600.7729s
        speed: 0.1353s/iter; left time: 4410.3650s
        speed: 0.1359s/iter; left time: 4415.5378s
        speed: 0.1348s/iter; left time: 4368.9311s
        speed: 0.1359s/iter; left time: 4389.8059s
        speed: 0.1355s/iter; left time: 4362.4974s
        speed: 0.1355s/iter; left time: 4348.9004s
        speed: 0.1363s/iter; left time: 4360.6657s
        speed: 0.1350s/iter; left time: 4307.7063s
        speed: 0.1354s/iter; left time: 4305.2038s
        speed: 0.1359s/iter; left time: 4309.3123s
        speed: 0.1353s/iter; left time: 4276.3862s
        speed: 0.1357s/iter; left time: 4274.6798s
        speed: 0.1354s/iter; left time: 4249.9516s
        speed: 0.1356s/iter; left time: 4245.4502s
        speed: 0.1366s/iter; left time: 4261.3997s
        speed: 0.1374s/iter; left time: 4272.8442s
        speed: 0.1353s/iter; left time: 4194.5101s
Epoch: 1 cost time: 2127.7168984413147
Epoch: 1, Steps: 15466 | Train Loss: -48.4473046 Vali Loss: -47.3290840 
Validation loss decreased (inf --> -47.329084).  Saving model ...
Updating learning rate to 0.0001
        speed: 6.2369s/iter; left time: 192301.2652s
        speed: 0.1412s/iter; left time: 4340.9789s
        speed: 0.1374s/iter; left time: 4207.7656s
        speed: 0.1395s/iter; left time: 4258.3836s
        speed: 0.1371s/iter; left time: 4172.0284s
        speed: 0.1385s/iter; left time: 4202.1003s
        speed: 0.1396s/iter; left time: 4219.4023s
        speed: 0.1378s/iter; left time: 4153.6058s
        speed: 0.1369s/iter; left time: 4110.4457s
        speed: 0.1371s/iter; left time: 4103.2520s
        speed: 0.1402s/iter; left time: 4181.8363s
        speed: 0.1373s/iter; left time: 4081.6900s
        speed: 0.1351s/iter; left time: 4003.4332s
        speed: 0.1390s/iter; left time: 4105.8047s
        speed: 0.1424s/iter; left time: 4190.9918s
        speed: 0.1417s/iter; left time: 4156.5764s
        speed: 0.1403s/iter; left time: 4102.5730s
        speed: 0.1420s/iter; left time: 4138.0644s
        speed: 0.1429s/iter; left time: 4147.9949s
        speed: 0.1402s/iter; left time: 4055.5878s
        speed: 0.1436s/iter; left time: 4141.6097s
        speed: 0.1432s/iter; left time: 4115.6612s
        speed: 0.1436s/iter; left time: 4112.7574s
        speed: 0.1425s/iter; left time: 4066.4701s
        speed: 0.1379s/iter; left time: 3921.2554s
        speed: 0.1488s/iter; left time: 4216.4488s
        speed: 0.1372s/iter; left time: 3873.5076s
        speed: 0.1406s/iter; left time: 3955.3746s
        speed: 0.1341s/iter; left time: 3759.9923s
        speed: 0.1341s/iter; left time: 3745.7212s
        speed: 0.1340s/iter; left time: 3730.7106s
        speed: 0.1341s/iter; left time: 3717.9604s
        speed: 0.1340s/iter; left time: 3703.4859s
        speed: 0.1340s/iter; left time: 3690.3612s
        speed: 0.1340s/iter; left time: 3677.0124s
        speed: 0.1340s/iter; left time: 3662.8140s
        speed: 0.1340s/iter; left time: 3650.3931s
        speed: 0.1340s/iter; left time: 3636.0835s
        speed: 0.1341s/iter; left time: 3624.2639s
        speed: 0.1341s/iter; left time: 3611.4985s
        speed: 0.1391s/iter; left time: 3732.2035s
        speed: 0.1365s/iter; left time: 3649.8741s
        speed: 0.1396s/iter; left time: 3719.0167s
        speed: 0.1363s/iter; left time: 3616.9123s
        speed: 0.1385s/iter; left time: 3659.8609s
        speed: 0.1389s/iter; left time: 3657.7603s
        speed: 0.1378s/iter; left time: 3614.8408s
        speed: 0.1432s/iter; left time: 3741.8558s
        speed: 0.1415s/iter; left time: 3683.3259s
        speed: 0.1433s/iter; left time: 3714.9921s
        speed: 0.1375s/iter; left time: 3552.5034s
        speed: 0.1403s/iter; left time: 3610.4658s
        speed: 0.1388s/iter; left time: 3559.0740s
        speed: 0.1391s/iter; left time: 3551.4850s
        speed: 0.1344s/iter; left time: 3419.0121s
        speed: 0.1383s/iter; left time: 3502.4784s
        speed: 0.1405s/iter; left time: 3544.0996s
        speed: 0.1440s/iter; left time: 3619.6918s
        speed: 0.1463s/iter; left time: 3663.1171s
        speed: 0.1437s/iter; left time: 3582.2442s
        speed: 0.1425s/iter; left time: 3538.2324s
        speed: 0.1430s/iter; left time: 3537.9648s
        speed: 0.1382s/iter; left time: 3405.0349s
        speed: 0.1349s/iter; left time: 3308.9421s
        speed: 0.1352s/iter; left time: 3304.2378s
        speed: 0.1352s/iter; left time: 3289.0986s
        speed: 0.1359s/iter; left time: 3292.5206s
        speed: 0.1351s/iter; left time: 3260.5215s
        speed: 0.1359s/iter; left time: 3266.8184s
        speed: 0.1381s/iter; left time: 3305.8850s
        speed: 0.1421s/iter; left time: 3387.8256s
        speed: 0.1362s/iter; left time: 3233.5133s
        speed: 0.1406s/iter; left time: 3323.1694s
        speed: 0.1390s/iter; left time: 3270.9122s
        speed: 0.1481s/iter; left time: 3470.9108s
        speed: 0.1480s/iter; left time: 3452.5954s
        speed: 0.1434s/iter; left time: 3332.1595s
        speed: 0.1444s/iter; left time: 3340.4601s
        speed: 0.1429s/iter; left time: 3290.3965s
        speed: 0.1418s/iter; left time: 3251.0263s
        speed: 0.1432s/iter; left time: 3269.5059s
        speed: 0.1446s/iter; left time: 3286.1990s
        speed: 0.1414s/iter; left time: 3200.7980s
        speed: 0.1359s/iter; left time: 3061.4477s
        speed: 0.1357s/iter; left time: 3043.1181s
        speed: 0.1374s/iter; left time: 3067.6164s
        speed: 0.1344s/iter; left time: 2988.2089s
        speed: 0.1344s/iter; left time: 2974.6237s
        speed: 0.1345s/iter; left time: 2962.6316s
        speed: 0.1391s/iter; left time: 3051.0582s
        speed: 0.1375s/iter; left time: 3002.8196s
        speed: 0.1360s/iter; left time: 2955.7436s
        speed: 0.1369s/iter; left time: 2962.3081s
        speed: 0.1407s/iter; left time: 3030.2290s
        speed: 0.1372s/iter; left time: 2940.8313s
        speed: 0.1365s/iter; left time: 2911.8519s
        speed: 0.1359s/iter; left time: 2885.3581s
        speed: 0.1359s/iter; left time: 2871.7553s
        speed: 0.1348s/iter; left time: 2835.1846s
        speed: 0.1355s/iter; left time: 2837.3081s
        speed: 0.1349s/iter; left time: 2810.6872s
        speed: 0.1380s/iter; left time: 2860.7424s
        speed: 0.1382s/iter; left time: 2850.8490s
        speed: 0.1380s/iter; left time: 2833.0666s
        speed: 0.1358s/iter; left time: 2775.4763s
        speed: 0.1364s/iter; left time: 2772.4910s
        speed: 0.1417s/iter; left time: 2866.6913s
        speed: 0.1408s/iter; left time: 2834.4955s
        speed: 0.1414s/iter; left time: 2833.6041s
        speed: 0.1386s/iter; left time: 2761.7363s
        speed: 0.1373s/iter; left time: 2722.3473s
        speed: 0.1450s/iter; left time: 2861.6037s
        speed: 0.1379s/iter; left time: 2707.0912s
        speed: 0.1358s/iter; left time: 2652.5041s
        speed: 0.1386s/iter; left time: 2694.2524s
        speed: 0.1394s/iter; left time: 2694.5749s
        speed: 0.1407s/iter; left time: 2706.0995s
        speed: 0.1428s/iter; left time: 2731.5554s
        speed: 0.1428s/iter; left time: 2718.7514s
        speed: 0.1411s/iter; left time: 2670.9368s
        speed: 0.1419s/iter; left time: 2673.1494s
        speed: 0.1371s/iter; left time: 2569.1856s
        speed: 0.1363s/iter; left time: 2540.2892s
        speed: 0.1393s/iter; left time: 2582.3330s
        speed: 0.1404s/iter; left time: 2588.2281s
        speed: 0.1390s/iter; left time: 2547.5025s
        speed: 0.1358s/iter; left time: 2475.4478s
        speed: 0.1347s/iter; left time: 2442.8927s
        speed: 0.1389s/iter; left time: 2504.9989s
        speed: 0.1393s/iter; left time: 2498.3219s
        speed: 0.1385s/iter; left time: 2469.3112s
        speed: 0.1439s/iter; left time: 2551.8338s
        speed: 0.1393s/iter; left time: 2455.4148s
        speed: 0.1361s/iter; left time: 2385.9529s
        speed: 0.1380s/iter; left time: 2405.8499s
        speed: 0.1386s/iter; left time: 2401.7565s
        speed: 0.1405s/iter; left time: 2421.6689s
        speed: 0.1346s/iter; left time: 2305.7041s
        speed: 0.1358s/iter; left time: 2312.4812s
        speed: 0.1366s/iter; left time: 2313.5609s
        speed: 0.1446s/iter; left time: 2433.9046s
        speed: 0.1391s/iter; left time: 2327.8682s
        speed: 0.1357s/iter; left time: 2257.7770s
        speed: 0.1372s/iter; left time: 2267.6875s
        speed: 0.1361s/iter; left time: 2236.3395s
        speed: 0.1362s/iter; left time: 2224.1642s
        speed: 0.1359s/iter; left time: 2205.6602s
        speed: 0.1355s/iter; left time: 2185.4923s
        speed: 0.1360s/iter; left time: 2180.5024s
        speed: 0.1358s/iter; left time: 2164.1690s
        speed: 0.1347s/iter; left time: 2132.9385s
        speed: 0.1351s/iter; left time: 2124.7418s
        speed: 0.1361s/iter; left time: 2127.1795s
        speed: 0.1354s/iter; left time: 2103.8047s
Epoch: 2 cost time: 2142.760348558426
Epoch: 2, Steps: 15466 | Train Loss: -48.7614085 Vali Loss: -47.4089206 
Validation loss decreased (-47.329084 --> -47.408921).  Saving model ...
Updating learning rate to 5e-05
        speed: 6.1778s/iter; left time: 94934.8721s
        speed: 0.1406s/iter; left time: 2146.9605s
        speed: 0.1405s/iter; left time: 2131.2590s
        speed: 0.1406s/iter; left time: 2118.3086s
        speed: 0.1407s/iter; left time: 2105.8900s
        speed: 0.1407s/iter; left time: 2091.9465s
        speed: 0.1406s/iter; left time: 2076.6571s
        speed: 0.1406s/iter; left time: 2062.1350s
        speed: 0.1404s/iter; left time: 2044.6462s
        speed: 0.1415s/iter; left time: 2047.4681s
        speed: 0.1482s/iter; left time: 2129.4256s
        speed: 0.1492s/iter; left time: 2127.9783s
        speed: 0.1365s/iter; left time: 1934.1541s
        speed: 0.1399s/iter; left time: 1968.6168s
        speed: 0.1400s/iter; left time: 1955.1173s
        speed: 0.1475s/iter; left time: 2045.0873s
        speed: 0.1462s/iter; left time: 2012.1008s
        speed: 0.1486s/iter; left time: 2030.5059s
        speed: 0.1392s/iter; left time: 1889.0985s
        speed: 0.1359s/iter; left time: 1830.0555s
        speed: 0.1362s/iter; left time: 1821.0789s
        speed: 0.1380s/iter; left time: 1830.8140s
        speed: 0.1400s/iter; left time: 1842.7255s
        speed: 0.1408s/iter; left time: 1840.0198s
        speed: 0.1404s/iter; left time: 1821.1826s
        speed: 0.1390s/iter; left time: 1788.3899s
        speed: 0.1352s/iter; left time: 1725.9144s
        speed: 0.1353s/iter; left time: 1713.4649s
        speed: 0.1352s/iter; left time: 1699.5077s
        speed: 0.1351s/iter; left time: 1683.8779s
        speed: 0.1349s/iter; left time: 1668.2950s
        speed: 0.1350s/iter; left time: 1656.3314s
        speed: 0.1349s/iter; left time: 1641.9323s
        speed: 0.1358s/iter; left time: 1638.1283s
        speed: 0.1408s/iter; left time: 1684.5690s
        speed: 0.1397s/iter; left time: 1658.0442s
        speed: 0.1364s/iter; left time: 1605.3649s
        speed: 0.1355s/iter; left time: 1581.1990s
        speed: 0.1354s/iter; left time: 1565.9528s
        speed: 0.1353s/iter; left time: 1551.1118s
        speed: 0.1355s/iter; left time: 1539.6940s
        speed: 0.1426s/iter; left time: 1606.9790s
        speed: 0.1432s/iter; left time: 1599.1447s
        speed: 0.1396s/iter; left time: 1544.6739s
        speed: 0.1375s/iter; left time: 1508.0367s
        speed: 0.1357s/iter; left time: 1474.7991s
        speed: 0.1387s/iter; left time: 1493.3745s
        speed: 0.1369s/iter; left time: 1460.2498s
        speed: 0.1450s/iter; left time: 1532.1255s
        speed: 0.1354s/iter; left time: 1417.6454s
        speed: 0.1367s/iter; left time: 1417.3438s
        speed: 0.1383s/iter; left time: 1419.4605s
        speed: 0.1394s/iter; left time: 1417.0308s
        speed: 0.1381s/iter; left time: 1389.8998s
        speed: 0.1359s/iter; left time: 1354.2720s
        speed: 0.1379s/iter; left time: 1360.3001s
        speed: 0.1400s/iter; left time: 1366.8987s
        speed: 0.1357s/iter; left time: 1311.7372s
        speed: 0.1355s/iter; left time: 1296.3165s
        speed: 0.1400s/iter; left time: 1325.6356s
        speed: 0.1396s/iter; left time: 1307.1830s
        speed: 0.1438s/iter; left time: 1332.1646s
        speed: 0.1425s/iter; left time: 1306.2933s
        speed: 0.1405s/iter; left time: 1273.5351s
        speed: 0.1383s/iter; left time: 1240.3545s
        speed: 0.1399s/iter; left time: 1240.1732s
        speed: 0.1349s/iter; left time: 1182.6134s
        speed: 0.1357s/iter; left time: 1176.3826s
        speed: 0.1369s/iter; left time: 1172.9671s
        speed: 0.1396s/iter; left time: 1181.7322s
        speed: 0.1503s/iter; left time: 1257.1756s
        speed: 0.1516s/iter; left time: 1253.5654s
        speed: 0.1357s/iter; left time: 1108.4112s
        speed: 0.1349s/iter; left time: 1088.4797s
        speed: 0.1383s/iter; left time: 1102.0619s
        speed: 0.1379s/iter; left time: 1084.9288s
        speed: 0.1415s/iter; left time: 1098.7484s
        speed: 0.1359s/iter; left time: 1042.0906s
        speed: 0.1380s/iter; left time: 1044.4410s
        speed: 0.1410s/iter; left time: 1052.5578s
        speed: 0.1362s/iter; left time: 1003.2852s
        speed: 0.1407s/iter; left time: 1022.6825s
        speed: 0.1375s/iter; left time: 985.7831s
        speed: 0.1369s/iter; left time: 967.7417s
        speed: 0.1386s/iter; left time: 965.7697s
        speed: 0.1374s/iter; left time: 943.5992s
        speed: 0.1373s/iter; left time: 929.1964s
        speed: 0.1411s/iter; left time: 940.4838s
        speed: 0.1358s/iter; left time: 891.6402s
        speed: 0.1361s/iter; left time: 880.2868s
        speed: 0.1361s/iter; left time: 866.7270s
        speed: 0.1399s/iter; left time: 876.9626s
        speed: 0.1391s/iter; left time: 857.7881s
        speed: 0.1390s/iter; left time: 843.3105s
        speed: 0.1417s/iter; left time: 845.6014s
        speed: 0.1393s/iter; left time: 817.4279s
        speed: 0.1448s/iter; left time: 835.1416s
        speed: 0.1418s/iter; left time: 803.5788s
        speed: 0.1396s/iter; left time: 777.3048s
        speed: 0.1352s/iter; left time: 739.0622s
        speed: 0.1342s/iter; left time: 720.3296s
        speed: 0.1356s/iter; left time: 714.3432s
        speed: 0.1349s/iter; left time: 697.1035s
        speed: 0.1357s/iter; left time: 687.6707s
        speed: 0.1371s/iter; left time: 680.7852s
        speed: 0.1348s/iter; left time: 656.0860s
        speed: 0.1394s/iter; left time: 664.3335s
        speed: 0.1424s/iter; left time: 664.4477s
        speed: 0.1430s/iter; left time: 652.9596s
        speed: 0.1432s/iter; left time: 639.7950s
        speed: 0.1415s/iter; left time: 618.0999s
        speed: 0.1405s/iter; left time: 599.4433s
        speed: 0.1417s/iter; left time: 590.2657s
        speed: 0.1402s/iter; left time: 570.0331s
        speed: 0.1411s/iter; left time: 559.8401s
        speed: 0.1404s/iter; left time: 543.0194s
        speed: 0.1401s/iter; left time: 527.8371s
        speed: 0.1402s/iter; left time: 513.9767s
        speed: 0.1402s/iter; left time: 500.0044s
        speed: 0.1385s/iter; left time: 480.0465s
        speed: 0.1368s/iter; left time: 460.6511s
        speed: 0.1387s/iter; left time: 453.0378s
        speed: 0.1385s/iter; left time: 438.5996s
        speed: 0.1406s/iter; left time: 431.1120s
        speed: 0.1376s/iter; left time: 408.1366s
        speed: 0.1430s/iter; left time: 409.8844s
        speed: 0.1467s/iter; left time: 405.8109s
        speed: 0.1465s/iter; left time: 390.8402s
        speed: 0.1357s/iter; left time: 348.3139s
        speed: 0.1357s/iter; left time: 334.8425s
        speed: 0.1374s/iter; left time: 325.2861s
        speed: 0.1424s/iter; left time: 322.7762s
        speed: 0.1408s/iter; left time: 305.0053s
        speed: 0.1422s/iter; left time: 293.8833s
        speed: 0.1435s/iter; left time: 282.3321s
        speed: 0.1438s/iter; left time: 268.4056s
        speed: 0.1446s/iter; left time: 255.5926s
        speed: 0.1386s/iter; left time: 230.9797s
        speed: 0.1357s/iter; left time: 212.7075s
        speed: 0.1370s/iter; left time: 201.0489s
        speed: 0.1391s/iter; left time: 190.1014s
        speed: 0.1348s/iter; left time: 170.7725s
        speed: 0.1383s/iter; left time: 161.4011s
        speed: 0.1370s/iter; left time: 146.2138s
        speed: 0.1350s/iter; left time: 130.5128s
        speed: 0.1350s/iter; left time: 117.0379s
        speed: 0.1348s/iter; left time: 103.3958s
        speed: 0.1349s/iter; left time: 89.9780s
        speed: 0.1350s/iter; left time: 76.5349s
        speed: 0.1350s/iter; left time: 63.0486s
        speed: 0.1350s/iter; left time: 49.5276s
        speed: 0.1349s/iter; left time: 36.0312s
        speed: 0.1349s/iter; left time: 22.5222s
        speed: 0.1350s/iter; left time: 9.0454s
Epoch: 3 cost time: 2149.257043838501
Epoch: 3, Steps: 15466 | Train Loss: -48.9121188 Vali Loss: -47.4772334 
EarlyStopping counter: 1 out of 3
Updating learning rate to 2.5e-05
------------ Options -------------
anormly_ratio: 0.1
batch_size: 32
data_path: dataset/SWaT
dataset: SWaT
input_c: 51
k: 3
lr: 0.0001
mode: test
model_save_path: checkpoints
num_epochs: 10
output_c: 51
pretrained_model: 10
win_size: 100
-------------- End ----------------
test: (449919, 51)
train: (495000, 51)
test: (449919, 51)
train: (495000, 51)
test: (449919, 51)
train: (495000, 51)
test: (449919, 51)
train: (495000, 51)
======================TEST MODE======================
/home/username/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Threshold : 0.0031170047065244427
pred:    (449900,)
gt:      (449900,)
pred:  (449900,)
gt:    (449900,)
Accuracy : 0.9775, Precision : 0.8841, Recall : 0.9371, F-score : 0.9099 

```



## NeurIPS-TS

这个Bencmark比较特殊，需要搭建测试平台:https://github.com/datamllab/tods

### 使用系统的Python[失败]

按照步骤克隆仓库并运行`pip install -e .`时出现报错，一些包安装存在问题：

```bash
Getting requirements to build wheel ... error
  error: subprocess-exited-with-error
  
  × Getting requirements to build wheel did not run successfully.
  │ exit code: 1
  ╰─> [154 lines of output]
      <string>:15: DeprecationWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html
      <string>:51: DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.
      <string>:54: DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.
      <string>:51: DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.
      performance hint: statsmodels/tsa/regime_switching/_hamilton_filter.pyx:83:5: Exception check on 'shamilton_filter_log_iteration' will always require the GIL to be acquired.
      Possible solutions:
          1. Declare the function as 'noexcept' if you control the definition and you're sure you don't want the function to raise exceptions.
          2. Use an 'int' return type on the function to allow an error code to be returned.

```

**问题定位和解决**：安装依赖`statsmodels==0.11.1`出现问题，尝试降低版本，在根目录的`setup.py`中修改`statsmodels==0.11.0rc1`，再次执行`pip install -e .`时成功了。



新的问题：

```bash
ERROR: Could not find a version that satisfies the requirement tensorflow==2.4 (from tods) (from versions: 2.5.0, 2.5.1, 2.5.2, 2.5.3, 2.6.0rc0, 2.6.0rc1, 2.6.0rc2, 2.6.0, 2.6.1, 2.6.2, 2.6.3, 2.6.4, 2.6.5, 2.7.0rc0, 2.7.0rc1, 2.7.0, 2.7.1, 2.7.2, 2.7.3, 2.7.4, 2.8.0rc0, 2.8.0rc1, 2.8.0, 2.8.1, 2.8.2, 2.8.3, 2.8.4, 2.9.0rc0, 2.9.0rc1, 2.9.0rc2, 2.9.0, 2.9.1, 2.9.2, 2.9.3, 2.10.0rc0, 2.10.0rc1, 2.10.0rc2, 2.10.0rc3, 2.10.0, 2.10.1, 2.11.0rc0, 2.11.0rc1, 2.11.0rc2, 2.11.0, 2.11.1, 2.12.0rc0, 2.12.0rc1, 2.12.0, 2.12.1, 2.13.0rc0, 2.13.0rc1, 2.13.0rc2, 2.13.0, 2.13.1, 2.14.0rc0, 2.14.0rc1, 2.14.0, 2.14.1, 2.15.0rc0, 2.15.0rc1, 2.15.0, 2.15.0.post1)
ERROR: No matching distribution found for tensorflow==2.4
```

**问题定位和解决**：`tensorflow 2.4`版本找不着，尝试在`setup.py`中修改为`tensorflow==2.5`



新的问题：

```bash
ERROR: Could not find a version that satisfies the requirement keras-nightly~=2.5.0.dev (from tensorflow) (from versions: none)
ERROR: No matching distribution found for keras-nightly~=2.5.0.dev
```

**问题定位和解决**：`keras-nightly~=2.5.0.dev`也找不着，手动去`pypi.org`官网下载安装https://pypi.org/project/keras-nightly/#history，我下载了这个的whl：https://pypi.org/project/keras-nightly/2.5.0.dev2021032900/，在终端使用`pip install ./keras_nightly-2.5.0.dev2021032900-py2.py3-none-any.whl `，成功。



最后似乎没有完全成功？：

```bash
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
spyder 5.3.3 requires pyqt5<5.16, which is not installed.
spyder 5.3.3 requires pyqtwebengine<5.16, which is not installed.
daal4py 2021.6.0 requires daal==2021.4.0, which is not installed.
anaconda-project 0.11.1 requires ruamel-yaml, which is not installed.
pylint 2.14.5 requires typing-extensions>=3.10.0; python_version < "3.10", but you have typing-extensions 3.7.4.3 which is incompatible.
imageio 2.19.3 requires pillow>=8.3.2, but you have pillow 7.1.2 which is incompatible.
conda-repo-cli 1.0.20 requires clyent==1.2.1, but you have clyent 1.2.2 which is incompatible.
conda-repo-cli 1.0.20 requires nbformat==5.4.0, but you have nbformat 5.5.0 which is incompatible.
conda-repo-cli 1.0.20 requires PyYAML==6.0, but you have pyyaml 5.4.1 which is incompatible.
conda-repo-cli 1.0.20 requires requests==2.28.1, but you have requests 2.26.0 which is incompatible.
bokeh 2.4.3 requires typing-extensions>=3.10.0, but you have typing-extensions 3.7.4.3 which is incompatible.
black 22.6.0 requires typing-extensions>=3.10.0.0; python_version < "3.10", but you have typing-extensions 3.7.4.3 which is incompatible.
astroid 2.11.7 requires typing-extensions>=3.10; python_version < "3.10", but you have typing-extensions 3.7.4.3 which is incompatible.
Successfully installed GitPython-3.1.24 absl-py-0.15.0 aiosignal-1.3.1 astunparse-1.6.3 cachetools-5.3.2 combo-0.1.3 custom-inherit-2.3.2 dateparser-1.1.8 flatbuffers-1.12 frozendict-1.2 frozenlist-1.4.0 gast-0.4.0 gitdb-4.0.11 google-auth-2.25.1 google-auth-oauthlib-0.4.6 google-pasta-0.2.0 gputil-1.4.0 grpcio-1.34.1 grpcio-testing-1.32.0 grpcio-tools-1.34.1 h5py-3.1.0 jsonpath-ng-1.5.3 jsonschema-4.0.1 keras-2.4.0 keras-preprocessing-1.1.2 liac-arff-2.5.0 more-itertools-8.5.0 nimfa-1.4.0 numpy-1.19.5 oauthlib-3.2.2 openml-0.11.0 opt-einsum-3.3.0 pandas-1.3.4 pillow-7.1.2 protobuf-3.20.3 pyarrow-14.0.1 pyod-1.0.5 pytypes-1.0b10 pyyaml-5.4.1 ray-2.8.1 requests-2.26.0 requests-oauthlib-1.3.1 rfc3339-validator-0.1.4 rfc3986-validator-0.1.1 rsa-4.9 scikit-learn-0.24.2 scipy-1.7.1 simplejson-3.12.0 six-1.15.0 smmap-5.0.1 statsmodels-0.11.0rc1 stumpy-1.4.0 tamu_axolotl-2021.2.11.1 tamu_d3m-2022.5.23 tensorboard-2.11.2 tensorboard-data-server-0.6.1 tensorboard-plugin-wit-1.8.1 tensorboardX-2.6.2.2 tensorflow-2.5.0 tensorflow-estimator-2.5.0 termcolor-1.1.0 tods-0.0.2 typing-extensions-3.7.4.3 typing-inspect-0.7.1 tzlocal-5.2 webcolors-1.11.1 wrapt-1.12.1 xgboost-2.0.2 xmltodict-0.13.0
```

### 使用Conda虚拟环境

在Pycharm中打开项目并新建conda interpreter, python 版本为3.8 (项目要求 Python 3.6 && pip 19+)

将根目录的`setup.py`更改过的依赖项版本还原

打开终端并确保在tods根目录且使用了conda的虚拟环境python，执行`pip install -e .`

这次安装无伤速通！总之就是以后再也不要用系统的python interpreter跑项目了！希望有时间能打包个docker镜像造福人类。

在根目录新建`test_example.py`如下并运行：

```python
import pandas as pd

from tods import schemas as schemas_utils
from tods import generate_dataset, evaluate_pipeline

table_path = 'datasets/anomaly/raw_data/yahoo_sub_5.csv'
target_index = 6 # what column is the target
metric = 'F1_MACRO' # F1 on both label 0 and 1

# Read data and generate dataset
df = pd.read_csv(table_path)
dataset = generate_dataset(df, target_index)

# Load the default pipeline
pipeline = schemas_utils.load_default_pipeline()

# Run the pipeline
pipeline_result = evaluate_pipeline(dataset, pipeline, metric)
print(pipeline_result)
```

成功输出结果。

```json
{'method_called': 'evaluate',
 'outputs': "[{'outputs.0':      d3mIndex  anomaly"
            '0           0        0'
            '1           1        0'
            '2           2        0'
            '3           3        0'
            '4           4        0'
            '...       ...      ...'
            '1395     1395        0'
            '1396     1396        0'
            '1397     1397        1'
            '1398     1398        1'
            '1399     1399        0'
            ''
            "[1400 rows x 2 columns]}, {'outputs.0':      d3mIndex  anomaly"
            '0           0        0'
            '1           1        0'
            '2           2        0'
            '3           3        0'
            '4           4        0'
            '...       ...      ...'
            '1395     1395        0'
            '1396     1396        0'
            '1397     1397        1'
            '1398     1398        1'
            '1399     1399        0'
            ''
            '[1400 rows x 2 columns]}]',
 'pipeline': '<d3m.metadata.pipeline.Pipeline object at 0x7fcab1e73cd0>',
 'scores': '     metric     value  normalized  randomSeed  fold'
           '0  F1_MACRO  0.708549    0.708549           0     0',
 'status': 'COMPLETED'}
```



## 2.6 总结



| Dataset \ Metrics | Accuracy | Precision | Recall | F1-score |
| :---------------: | :------: | :-------: | :----: | :------: |
|    SMD / Ours     |  99.26   |   89.27   | 93.29  |  91.24   |
|    SMD / Paper    |    \     |   89.40   | 95.45  |  92.33   |
|    MSL / Ours     |  98.63   |   91.86   | 95.45  |  93.62   |
|    MSL / Paper    |    \     |   92.09   | 95.15  |  93.59   |
|    SMAP / Ours    |  99.06   |   93.60   | 99.43  |  96.42   |
|   SMAP / Paper    |    \     |   94.13   | 99.40  |  96.69   |
|    SWaT / Ours    |  97.75   |   88.41   | 93.71  |  90.99   |
|   SWaT / Paper    |    \     |   91.55   | 96.73  |  94.07   |
|    PSM / Ours     |  98.82   |   96.97   | 98.83  |  97.89   |
|    PSM / Paper    |    \     |   96.91   | 98.90  |  97.89   |

可见所有数据集与论文第4章Table 1所给数据并无较大出入，且所有F1-Score仍然如Table 1标注所示，领先于其余对比算法。

## UCR dataset

下载于：https://compete.hexagon-ml.com/media/data/multi-dataset-time-series-anomaly-detection-39/data.zip

# 3. 分析与设计

一些思考：能不能跳出时间序列的限制，例如将代码文本作为输入，输出其是否存在异常。首先为了能让代码片段输入，肯定需要进行一定的编码，例如现在大模型流行使用的tokenizer方法。但tokenizer是将单个词映射为定长向量，而一个代码片通常由多个可视为词的符号组成，且词之间具有严密的逻辑关系。



## 3.1 Anomaly ratio $r$ 及其局限性

论文为每个数据集设定了不同的异常比例$r$，用于确定一个Anomaly Score 阈值$\delta$，使得验证集的异常点占比达到预设的$r$. 这存在需要人工经验取值的问题，况且异常比例$r$在验证集、训练集和测试集的情况很有可能存在不同，我认为这主要是由于时间序列的连续特性无法进行随机采样得到验证集导致的，论文代码中对于验证集的选取也受限于连续特性。另一方面，论文方法是无监督方法，设置阈值是无监督异常检测任务中难以避免的一个操作，而从这个角度进行基于标签数据的有监督学习改进是困难的，因为现实中的时序异常数据很难打标签。

### 3.1.1 尝试通过改变重建损失计算利用标签数据训练

简单二分策略的使用混合重建损失的结果：
```bash
Anomaly Ratio : 50.0
Threshold : 0.0
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.5032, Precision : 0.7028, Recall : 0.2204, F-score : 0.3356 

Threshold : 0.0
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.5032, Precision : 0.7028, Recall : 0.2204, F-score : 0.3356 
```





## 3.2 对网络异常检测数据的适用性

### 3.2.1 在NSL-KDD数据集上训练与测试

#### 简单二分策略

将normal标签设为0，其余均为1，然后在不同的anomaly ratio下测试算法在NSLKDD数据集上的效果。

##### r=0.5%

``` bash
(Anomaly-Transformer) username@username-ubuntu:/media/username/3E6E20236E1FD28F/Dev/Anomaly-Transformer$ bash ./scripts/NSLKDD.sh
------------ Options -------------
anormly_ratio: 0.5
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD
input_c: 122
k: 3
lr: 0.0001
mode: train
model_save_path: checkpoints
num_epochs: 3
output_c: 122
pretrained_model: None
win_size: 100
-------------- End ----------------
======================TRAIN MODE======================

------------ Options -------------
anormly_ratio: 0.5
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD
input_c: 122
k: 3
lr: 0.0001
mode: test
model_save_path: checkpoints
num_epochs: 10
output_c: 122
pretrained_model: 10
win_size: 100
-------------- End ----------------

======================TEST MODE======================
/home/username/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Threshold : 0.02469959240406773
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.4585, Precision : 0.9481, Recall : 0.0514, F-score : 0.0975 
```



##### r=1.0%

```bash
(Anomaly-Transformer) username@username-ubuntu:/media/username/3E6E20236E1FD28F/Dev/Anomaly-Transformer$ bash ./scripts/NSLKDD.sh
------------ Options -------------
anormly_ratio: 1.0
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD
input_c: 122
k: 3
lr: 0.0001
mode: train
model_save_path: checkpoints
num_epochs: 3
output_c: 122
pretrained_model: None
win_size: 100
-------------- End ----------------
test: (22544, 122)
train: (125973, 122)
test: (22544, 122)
train: (125973, 122)
test: (22544, 122)
train: (125973, 122)
test: (22544, 122)
train: (125973, 122)
======================TRAIN MODE======================
        speed: 0.1369s/iter; left time: 1601.7550s
        speed: 0.1303s/iter; left time: 1512.0029s
        speed: 0.1306s/iter; left time: 1502.6880s
        speed: 0.1307s/iter; left time: 1490.6938s
        speed: 0.1308s/iter; left time: 1478.8884s
        speed: 0.1308s/iter; left time: 1465.4615s
        speed: 0.1308s/iter; left time: 1452.1905s
        speed: 0.1309s/iter; left time: 1440.6744s
        speed: 0.1313s/iter; left time: 1431.1096s
        speed: 0.1313s/iter; left time: 1418.0322s
        speed: 0.1312s/iter; left time: 1404.4908s
        speed: 0.1313s/iter; left time: 1392.5284s
        speed: 0.1313s/iter; left time: 1378.5787s
        speed: 0.1313s/iter; left time: 1365.6873s
        speed: 0.1312s/iter; left time: 1351.8481s
        speed: 0.1313s/iter; left time: 1339.4321s
        speed: 0.1312s/iter; left time: 1325.7825s
        speed: 0.1312s/iter; left time: 1312.7482s
        speed: 0.1312s/iter; left time: 1299.0108s
        speed: 0.1313s/iter; left time: 1286.7188s
        speed: 0.1312s/iter; left time: 1273.0868s
        speed: 0.1312s/iter; left time: 1260.0926s
        speed: 0.1316s/iter; left time: 1250.9483s
        speed: 0.1335s/iter; left time: 1254.8993s
        speed: 0.1328s/iter; left time: 1235.7682s
        speed: 0.1308s/iter; left time: 1204.1539s
        speed: 0.1309s/iter; left time: 1191.3958s
        speed: 0.1309s/iter; left time: 1178.2278s
        speed: 0.1322s/iter; left time: 1176.9927s
        speed: 0.1315s/iter; left time: 1157.5977s
        speed: 0.1315s/iter; left time: 1144.1452s
        speed: 0.1314s/iter; left time: 1130.5234s
        speed: 0.1314s/iter; left time: 1117.5005s
        speed: 0.1314s/iter; left time: 1104.2098s
        speed: 0.1314s/iter; left time: 1090.8631s
        speed: 0.1315s/iter; left time: 1078.5724s
        speed: 0.1314s/iter; left time: 1064.4714s
        speed: 0.1315s/iter; left time: 1052.1539s
        speed: 0.1353s/iter; left time: 1069.2737s
Epoch: 1 cost time: 517.9919922351837
Epoch: 1, Steps: 3934 | Train Loss: -47.0414631 Vali Loss: -47.4802924 
Validation loss decreased (inf --> -47.480292).  Saving model ...
Updating learning rate to 0.0001
        speed: 0.4840s/iter; left time: 3759.9627s
        speed: 0.1332s/iter; left time: 1021.6347s
        speed: 0.1323s/iter; left time: 1001.5881s
        speed: 0.1319s/iter; left time: 985.3198s
        speed: 0.1345s/iter; left time: 990.9520s
        speed: 0.1343s/iter; left time: 976.3390s
        speed: 0.1390s/iter; left time: 996.3020s
        speed: 0.1346s/iter; left time: 951.4199s
        speed: 0.1360s/iter; left time: 947.8869s
        speed: 0.1341s/iter; left time: 921.2710s
        speed: 0.1379s/iter; left time: 933.7204s
        speed: 0.1326s/iter; left time: 883.9982s
        speed: 0.1394s/iter; left time: 915.8344s
        speed: 0.1355s/iter; left time: 876.6080s
        speed: 0.1428s/iter; left time: 909.5229s
        speed: 0.1436s/iter; left time: 900.0828s
        speed: 0.1437s/iter; left time: 886.3455s
        speed: 0.1433s/iter; left time: 869.4626s
        speed: 0.1445s/iter; left time: 862.6468s
        speed: 0.1449s/iter; left time: 850.2270s
        speed: 0.1418s/iter; left time: 818.0399s
        speed: 0.1367s/iter; left time: 774.7034s
        speed: 0.1367s/iter; left time: 761.4599s
        speed: 0.1367s/iter; left time: 747.4475s
        speed: 0.1364s/iter; left time: 732.2058s
        speed: 0.1368s/iter; left time: 721.0366s
        speed: 0.1367s/iter; left time: 706.8142s
        speed: 0.1367s/iter; left time: 693.1174s
        speed: 0.1367s/iter; left time: 679.1568s
        speed: 0.1367s/iter; left time: 665.7158s
        speed: 0.1368s/iter; left time: 652.5808s
        speed: 0.1367s/iter; left time: 638.4532s
        speed: 0.1362s/iter; left time: 622.4852s
        speed: 0.1364s/iter; left time: 609.4713s
        speed: 0.1364s/iter; left time: 595.9573s
        speed: 0.1364s/iter; left time: 582.4890s
        speed: 0.1364s/iter; left time: 568.5825s
        speed: 0.1364s/iter; left time: 554.9945s
        speed: 0.1365s/iter; left time: 541.9608s
Epoch: 2 cost time: 539.6562712192535
Epoch: 2, Steps: 3934 | Train Loss: -48.5144279 Vali Loss: -48.1329151 
EarlyStopping counter: 1 out of 3
Updating learning rate to 5e-05
        speed: 0.4813s/iter; left time: 1845.6853s
        speed: 0.1364s/iter; left time: 509.2899s
        speed: 0.1362s/iter; left time: 495.1138s
        speed: 0.1364s/iter; left time: 482.0215s
        speed: 0.1365s/iter; left time: 468.8174s
        speed: 0.1364s/iter; left time: 455.0160s
        speed: 0.1364s/iter; left time: 441.3130s
        speed: 0.1364s/iter; left time: 427.6231s
        speed: 0.1364s/iter; left time: 414.0394s
        speed: 0.1364s/iter; left time: 400.4002s
        speed: 0.1363s/iter; left time: 386.2983s
        speed: 0.1363s/iter; left time: 372.6786s
        speed: 0.1364s/iter; left time: 359.5014s
        speed: 0.1364s/iter; left time: 345.8320s
        speed: 0.1364s/iter; left time: 332.1094s
        speed: 0.1363s/iter; left time: 318.1958s
        speed: 0.1364s/iter; left time: 304.7477s
        speed: 0.1362s/iter; left time: 290.8440s
        speed: 0.1366s/iter; left time: 277.9560s
        speed: 0.1363s/iter; left time: 263.7950s
        speed: 0.1364s/iter; left time: 250.2387s
        speed: 0.1364s/iter; left time: 236.6038s
        speed: 0.1363s/iter; left time: 222.8828s
        speed: 0.1363s/iter; left time: 209.2962s
        speed: 0.1364s/iter; left time: 195.7429s
        speed: 0.1363s/iter; left time: 181.9378s
        speed: 0.1364s/iter; left time: 168.4278s
        speed: 0.1362s/iter; left time: 154.6098s
        speed: 0.1360s/iter; left time: 140.7701s
        speed: 0.1364s/iter; left time: 127.5001s
        speed: 0.1359s/iter; left time: 113.5145s
        speed: 0.1362s/iter; left time: 100.1328s
        speed: 0.1363s/iter; left time: 86.5670s
        speed: 0.1362s/iter; left time: 72.8662s
        speed: 0.1361s/iter; left time: 59.1943s
        speed: 0.1363s/iter; left time: 45.6561s
        speed: 0.1361s/iter; left time: 31.9842s
        speed: 0.1363s/iter; left time: 18.3962s
        speed: 0.1364s/iter; left time: 4.7725s
Epoch: 3 cost time: 536.1699142456055
Epoch: 3, Steps: 3934 | Train Loss: -48.7206043 Vali Loss: -48.3033336 
EarlyStopping counter: 2 out of 3
Updating learning rate to 2.5e-05
------------ Options -------------
anormly_ratio: 1.0
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD
input_c: 122
k: 3
lr: 0.0001
mode: test
model_save_path: checkpoints
num_epochs: 10
output_c: 122
pretrained_model: 10
win_size: 100
-------------- End ----------------
test: (22544, 122)
train: (125973, 122)
test: (22544, 122)
train: (125973, 122)
test: (22544, 122)
train: (125973, 122)
test: (22544, 122)
train: (125973, 122)
======================TEST MODE======================
/home/username/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Threshold : 0.007011290364898737
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.4537, Precision : 0.8757, Recall : 0.0468, F-score : 0.0888 

/home/username/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Threshold : 0.0031170047065244427
pred:    (449900,)
gt:      (449900,)
pred:  (449900,)
gt:    (449900,)
Accuracy : 0.9775, Precision : 0.8841, Recall : 0.9371, F-score : 0.9099 

(Anomaly-Transformer) username@username-ubuntu:/media/username/3E6E20236E1FD28F/Dev/Anomaly-Transformer$ bash ./scripts/NSLKDD.sh
------------ Options -------------
anormly_ratio: 1.0
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD
input_c: 122
k: 3
lr: 0.0001
mode: train
model_save_path: checkpoints
num_epochs: 3
output_c: 122
pretrained_model: None
win_size: 100
-------------- End ----------------
test: (22544, 122)
train: (125973, 122)
test: (22544, 122)
train: (125973, 122)
test: (22544, 122)
train: (125973, 122)
test: (22544, 122)
train: (125973, 122)
======================TRAIN MODE======================
        speed: 0.1369s/iter; left time: 1601.7550s
        speed: 0.1303s/iter; left time: 1512.0029s
        speed: 0.1306s/iter; left time: 1502.6880s
        speed: 0.1307s/iter; left time: 1490.6938s
        speed: 0.1308s/iter; left time: 1478.8884s
        speed: 0.1308s/iter; left time: 1465.4615s
        speed: 0.1308s/iter; left time: 1452.1905s
        speed: 0.1309s/iter; left time: 1440.6744s
        speed: 0.1313s/iter; left time: 1431.1096s
        speed: 0.1313s/iter; left time: 1418.0322s
        speed: 0.1312s/iter; left time: 1404.4908s
        speed: 0.1313s/iter; left time: 1392.5284s
        speed: 0.1313s/iter; left time: 1378.5787s
        speed: 0.1313s/iter; left time: 1365.6873s
        speed: 0.1312s/iter; left time: 1351.8481s
        speed: 0.1313s/iter; left time: 1339.4321s
        speed: 0.1312s/iter; left time: 1325.7825s
        speed: 0.1312s/iter; left time: 1312.7482s
        speed: 0.1312s/iter; left time: 1299.0108s
        speed: 0.1313s/iter; left time: 1286.7188s
        speed: 0.1312s/iter; left time: 1273.0868s
        speed: 0.1312s/iter; left time: 1260.0926s
        speed: 0.1316s/iter; left time: 1250.9483s
        speed: 0.1335s/iter; left time: 1254.8993s
        speed: 0.1328s/iter; left time: 1235.7682s
        speed: 0.1308s/iter; left time: 1204.1539s
        speed: 0.1309s/iter; left time: 1191.3958s
        speed: 0.1309s/iter; left time: 1178.2278s
        speed: 0.1322s/iter; left time: 1176.9927s
        speed: 0.1315s/iter; left time: 1157.5977s
        speed: 0.1315s/iter; left time: 1144.1452s
        speed: 0.1314s/iter; left time: 1130.5234s
        speed: 0.1314s/iter; left time: 1117.5005s
        speed: 0.1314s/iter; left time: 1104.2098s
        speed: 0.1314s/iter; left time: 1090.8631s
        speed: 0.1315s/iter; left time: 1078.5724s
        speed: 0.1314s/iter; left time: 1064.4714s
        speed: 0.1315s/iter; left time: 1052.1539s
        speed: 0.1353s/iter; left time: 1069.2737s
Epoch: 1 cost time: 517.9919922351837
Epoch: 1, Steps: 3934 | Train Loss: -47.0414631 Vali Loss: -47.4802924 
Validation loss decreased (inf --> -47.480292).  Saving model ...
Updating learning rate to 0.0001
        speed: 0.4840s/iter; left time: 3759.9627s
        speed: 0.1332s/iter; left time: 1021.6347s
        speed: 0.1323s/iter; left time: 1001.5881s
        speed: 0.1319s/iter; left time: 985.3198s
        speed: 0.1345s/iter; left time: 990.9520s
        speed: 0.1343s/iter; left time: 976.3390s
        speed: 0.1390s/iter; left time: 996.3020s
        speed: 0.1346s/iter; left time: 951.4199s
        speed: 0.1360s/iter; left time: 947.8869s
        speed: 0.1341s/iter; left time: 921.2710s
        speed: 0.1379s/iter; left time: 933.7204s
        speed: 0.1326s/iter; left time: 883.9982s
        speed: 0.1394s/iter; left time: 915.8344s
        speed: 0.1355s/iter; left time: 876.6080s
        speed: 0.1428s/iter; left time: 909.5229s
        speed: 0.1436s/iter; left time: 900.0828s
        speed: 0.1437s/iter; left time: 886.3455s
        speed: 0.1433s/iter; left time: 869.4626s
        speed: 0.1445s/iter; left time: 862.6468s
        speed: 0.1449s/iter; left time: 850.2270s
        speed: 0.1418s/iter; left time: 818.0399s
        speed: 0.1367s/iter; left time: 774.7034s
        speed: 0.1367s/iter; left time: 761.4599s
        speed: 0.1367s/iter; left time: 747.4475s
        speed: 0.1364s/iter; left time: 732.2058s
        speed: 0.1368s/iter; left time: 721.0366s
        speed: 0.1367s/iter; left time: 706.8142s
        speed: 0.1367s/iter; left time: 693.1174s
        speed: 0.1367s/iter; left time: 679.1568s
        speed: 0.1367s/iter; left time: 665.7158s
        speed: 0.1368s/iter; left time: 652.5808s
        speed: 0.1367s/iter; left time: 638.4532s
        speed: 0.1362s/iter; left time: 622.4852s
        speed: 0.1364s/iter; left time: 609.4713s
        speed: 0.1364s/iter; left time: 595.9573s
        speed: 0.1364s/iter; left time: 582.4890s
        speed: 0.1364s/iter; left time: 568.5825s
        speed: 0.1364s/iter; left time: 554.9945s
        speed: 0.1365s/iter; left time: 541.9608s
Epoch: 2 cost time: 539.6562712192535
Epoch: 2, Steps: 3934 | Train Loss: -48.5144279 Vali Loss: -48.1329151 
EarlyStopping counter: 1 out of 3
Updating learning rate to 5e-05
        speed: 0.4813s/iter; left time: 1845.6853s
        speed: 0.1364s/iter; left time: 509.2899s
        speed: 0.1362s/iter; left time: 495.1138s
        speed: 0.1364s/iter; left time: 482.0215s
        speed: 0.1365s/iter; left time: 468.8174s
        speed: 0.1364s/iter; left time: 455.0160s
        speed: 0.1364s/iter; left time: 441.3130s
        speed: 0.1364s/iter; left time: 427.6231s
        speed: 0.1364s/iter; left time: 414.0394s
        speed: 0.1364s/iter; left time: 400.4002s
        speed: 0.1363s/iter; left time: 386.2983s
        speed: 0.1363s/iter; left time: 372.6786s
        speed: 0.1364s/iter; left time: 359.5014s
        speed: 0.1364s/iter; left time: 345.8320s
        speed: 0.1364s/iter; left time: 332.1094s
        speed: 0.1363s/iter; left time: 318.1958s
        speed: 0.1364s/iter; left time: 304.7477s
        speed: 0.1362s/iter; left time: 290.8440s
        speed: 0.1366s/iter; left time: 277.9560s
        speed: 0.1363s/iter; left time: 263.7950s
        speed: 0.1364s/iter; left time: 250.2387s
        speed: 0.1364s/iter; left time: 236.6038s
        speed: 0.1363s/iter; left time: 222.8828s
        speed: 0.1363s/iter; left time: 209.2962s
        speed: 0.1364s/iter; left time: 195.7429s
        speed: 0.1363s/iter; left time: 181.9378s
        speed: 0.1364s/iter; left time: 168.4278s
        speed: 0.1362s/iter; left time: 154.6098s
        speed: 0.1360s/iter; left time: 140.7701s
        speed: 0.1364s/iter; left time: 127.5001s
        speed: 0.1359s/iter; left time: 113.5145s
        speed: 0.1362s/iter; left time: 100.1328s
        speed: 0.1363s/iter; left time: 86.5670s
        speed: 0.1362s/iter; left time: 72.8662s
        speed: 0.1361s/iter; left time: 59.1943s
        speed: 0.1363s/iter; left time: 45.6561s
        speed: 0.1361s/iter; left time: 31.9842s
        speed: 0.1363s/iter; left time: 18.3962s
        speed: 0.1364s/iter; left time: 4.7725s
Epoch: 3 cost time: 536.1699142456055
Epoch: 3, Steps: 3934 | Train Loss: -48.7206043 Vali Loss: -48.3033336 
EarlyStopping counter: 2 out of 3
Updating learning rate to 2.5e-05
------------ Options -------------
anormly_ratio: 1.0
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD
input_c: 122
k: 3
lr: 0.0001
mode: test
model_save_path: checkpoints
num_epochs: 10
output_c: 122
pretrained_model: 10
win_size: 100
-------------- End ----------------
test: (22544, 122)
train: (125973, 122)
test: (22544, 122)
train: (125973, 122)
test: (22544, 122)
train: (125973, 122)
test: (22544, 122)
train: (125973, 122)
======================TEST MODE======================
/home/username/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Threshold : 0.007011290364898737
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.4537, Precision : 0.8757, Recall : 0.0468, F-score : 0.0888 

```

#####  r=50.0%

```bash
------------ Options -------------
anormly_ratio: 50.0
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD
input_c: 122
k: 3
lr: 0.0001
mode: train
model_save_path: checkpoints
num_epochs: 10
output_c: 122
pretrained_model: None
win_size: 100
-------------- End ----------------
test: (22544, 122)
train: (125973, 122)
test: (22544, 122)
train: (125973, 122)
test: (22544, 122)
train: (125973, 122)
test: (22544, 122)
train: (125973, 122)
======================TRAIN MODE======================
        speed: 0.1355s/iter; left time: 5316.9313s
        speed: 0.1356s/iter; left time: 5307.0847s
        speed: 0.1361s/iter; left time: 5313.6697s
        speed: 0.1319s/iter; left time: 5136.0505s
        speed: 0.1325s/iter; left time: 5148.3551s
        speed: 0.1319s/iter; left time: 5109.8676s
        speed: 0.1323s/iter; left time: 5113.9623s
        speed: 0.1322s/iter; left time: 5096.4198s
        speed: 0.1315s/iter; left time: 5054.2963s
        speed: 0.1322s/iter; left time: 5068.8596s
        speed: 0.1324s/iter; left time: 5062.5358s
        speed: 0.1340s/iter; left time: 5111.2675s
        speed: 0.1310s/iter; left time: 4983.8302s
        speed: 0.1316s/iter; left time: 4993.7247s
        speed: 0.1310s/iter; left time: 4958.1371s
        speed: 0.1310s/iter; left time: 4944.1936s
        speed: 0.1310s/iter; left time: 4931.2664s
        speed: 0.1311s/iter; left time: 4920.0523s
        speed: 0.1310s/iter; left time: 4905.3923s
        speed: 0.1311s/iter; left time: 4894.8455s
        speed: 0.1310s/iter; left time: 4879.8760s
        speed: 0.1311s/iter; left time: 4869.0741s
        speed: 0.1310s/iter; left time: 4851.3908s
        speed: 0.1311s/iter; left time: 4841.5871s
        speed: 0.1311s/iter; left time: 4828.5362s
        speed: 0.1311s/iter; left time: 4816.3585s
        speed: 0.1310s/iter; left time: 4800.7019s
        speed: 0.1309s/iter; left time: 4784.6071s
        speed: 0.1310s/iter; left time: 4773.5566s
        speed: 0.1310s/iter; left time: 4761.9220s
        speed: 0.1310s/iter; left time: 4748.1666s
        speed: 0.1310s/iter; left time: 4734.0354s
        speed: 0.1310s/iter; left time: 4720.0640s
        speed: 0.1310s/iter; left time: 4708.9502s
        speed: 0.1310s/iter; left time: 4694.6541s
        speed: 0.1311s/iter; left time: 4685.8344s
        speed: 0.1309s/iter; left time: 4665.9214s
        speed: 0.1309s/iter; left time: 4654.0228s
        speed: 0.1310s/iter; left time: 4642.1672s
Epoch: 1 cost time: 518.1424803733826
Epoch: 1, Steps: 3934 | Train Loss: -46.8131543 Vali Loss: -47.3336469 
Validation loss decreased (inf --> -47.333647).  Saving model ...
Updating learning rate to 0.0001
        speed: 0.4610s/iter; left time: 16276.7726s
        speed: 0.1310s/iter; left time: 4612.5694s
        speed: 0.1310s/iter; left time: 4599.6580s
        speed: 0.1310s/iter; left time: 4587.0812s
        speed: 0.1310s/iter; left time: 4572.4280s
        speed: 0.1312s/iter; left time: 4565.2154s
        speed: 0.1310s/iter; left time: 4546.2637s
        speed: 0.1310s/iter; left time: 4534.1559s
        speed: 0.1310s/iter; left time: 4519.9412s
        speed: 0.1310s/iter; left time: 4506.3366s
        speed: 0.1310s/iter; left time: 4493.2462s
        speed: 0.1309s/iter; left time: 4479.1329s
        speed: 0.1309s/iter; left time: 4465.1925s
        speed: 0.1309s/iter; left time: 4451.1384s
        speed: 0.1309s/iter; left time: 4439.9778s
        speed: 0.1309s/iter; left time: 4425.5404s
        speed: 0.1309s/iter; left time: 4412.6608s
        speed: 0.1309s/iter; left time: 4399.0118s
        speed: 0.1309s/iter; left time: 4386.5066s
        speed: 0.1310s/iter; left time: 4374.8416s
        speed: 0.1309s/iter; left time: 4360.6469s
        speed: 0.1310s/iter; left time: 4348.4842s
        speed: 0.1309s/iter; left time: 4333.9263s
        speed: 0.1309s/iter; left time: 4321.6431s
        speed: 0.1310s/iter; left time: 4309.9051s
        speed: 0.1309s/iter; left time: 4295.7049s
        speed: 0.1309s/iter; left time: 4282.2557s
        speed: 0.1309s/iter; left time: 4268.6591s
        speed: 0.1309s/iter; left time: 4256.5062s
        speed: 0.1309s/iter; left time: 4241.6757s
        speed: 0.1309s/iter; left time: 4228.4744s
        speed: 0.1309s/iter; left time: 4215.9839s
        speed: 0.1309s/iter; left time: 4203.7866s
        speed: 0.1310s/iter; left time: 4192.8665s
        speed: 0.1310s/iter; left time: 4179.2397s
        speed: 0.1310s/iter; left time: 4165.1929s
        speed: 0.1309s/iter; left time: 4151.0590s
        speed: 0.1310s/iter; left time: 4140.1505s
        speed: 0.1310s/iter; left time: 4126.8890s
Epoch: 2 cost time: 515.0924828052521
Epoch: 2, Steps: 3934 | Train Loss: -48.4168298 Vali Loss: -47.9248486 
EarlyStopping counter: 1 out of 3
Updating learning rate to 5e-05
        speed: 0.4590s/iter; left time: 14398.8618s
        speed: 0.1309s/iter; left time: 4094.5120s
        speed: 0.1309s/iter; left time: 4079.1813s
        speed: 0.1309s/iter; left time: 4067.8194s
        speed: 0.1309s/iter; left time: 4054.6873s
        speed: 0.1309s/iter; left time: 4040.4983s
        speed: 0.1309s/iter; left time: 4027.7394s
        speed: 0.1309s/iter; left time: 4015.9503s
        speed: 0.1309s/iter; left time: 4002.8357s
        speed: 0.1319s/iter; left time: 4020.1789s
        speed: 0.1328s/iter; left time: 4032.3251s
        speed: 0.1311s/iter; left time: 3969.1873s
        speed: 0.1311s/iter; left time: 3956.2238s
        speed: 0.1311s/iter; left time: 3943.2807s
        speed: 0.1314s/iter; left time: 3937.5022s
        speed: 0.1317s/iter; left time: 3935.3920s
        speed: 0.1325s/iter; left time: 3944.7383s
        speed: 0.1329s/iter; left time: 3942.8348s
        speed: 0.1313s/iter; left time: 3883.1325s
        speed: 0.1340s/iter; left time: 3950.0506s
        speed: 0.1334s/iter; left time: 3919.4091s
        speed: 0.1309s/iter; left time: 3832.1620s
        speed: 0.1309s/iter; left time: 3818.9089s
        speed: 0.1309s/iter; left time: 3806.7406s
        speed: 0.1309s/iter; left time: 3791.2263s
        speed: 0.1309s/iter; left time: 3779.5752s
        speed: 0.1309s/iter; left time: 3765.7227s
        speed: 0.1309s/iter; left time: 3753.4292s
        speed: 0.1309s/iter; left time: 3739.4836s
        speed: 0.1309s/iter; left time: 3726.6624s
        speed: 0.1309s/iter; left time: 3714.4806s
        speed: 0.1310s/iter; left time: 3703.1740s
        speed: 0.1309s/iter; left time: 3688.7920s
        speed: 0.1309s/iter; left time: 3673.9377s
        speed: 0.1311s/iter; left time: 3667.6697s
        speed: 0.1309s/iter; left time: 3649.3432s
        speed: 0.1310s/iter; left time: 3637.1319s
        speed: 0.1309s/iter; left time: 3622.2054s
        speed: 0.1309s/iter; left time: 3608.6637s
Epoch: 3 cost time: 516.3497984409332
Epoch: 3, Steps: 3934 | Train Loss: -48.6391877 Vali Loss: -48.1122985 
EarlyStopping counter: 2 out of 3
Updating learning rate to 2.5e-05
        speed: 0.4590s/iter; left time: 12595.1135s
        speed: 0.1309s/iter; left time: 3578.3130s
        speed: 0.1309s/iter; left time: 3566.5477s
        speed: 0.1309s/iter; left time: 3552.5281s
        speed: 0.1310s/iter; left time: 3541.1830s
        speed: 0.1309s/iter; left time: 3526.7139s
        speed: 0.1309s/iter; left time: 3514.5533s
        speed: 0.1309s/iter; left time: 3501.3495s
        speed: 0.1310s/iter; left time: 3490.4514s
        speed: 0.1309s/iter; left time: 3474.1579s
        speed: 0.1309s/iter; left time: 3461.5496s
        speed: 0.1309s/iter; left time: 3448.6966s
        speed: 0.1309s/iter; left time: 3434.1434s
        speed: 0.1309s/iter; left time: 3422.2355s
        speed: 0.1309s/iter; left time: 3407.6903s
        speed: 0.1310s/iter; left time: 3396.7607s
        speed: 0.1309s/iter; left time: 3381.6889s
        speed: 0.1309s/iter; left time: 3369.2955s
        speed: 0.1309s/iter; left time: 3355.4160s
        speed: 0.1309s/iter; left time: 3343.8095s
        speed: 0.1309s/iter; left time: 3329.5964s
        speed: 0.1309s/iter; left time: 3316.2136s
        speed: 0.1309s/iter; left time: 3303.9051s
        speed: 0.1309s/iter; left time: 3290.4389s
        speed: 0.1309s/iter; left time: 3277.7411s
        speed: 0.1309s/iter; left time: 3264.3852s
        speed: 0.1310s/iter; left time: 3254.0794s
        speed: 0.1310s/iter; left time: 3241.5225s
        speed: 0.1310s/iter; left time: 3226.8819s
        speed: 0.1310s/iter; left time: 3213.5240s
        speed: 0.1310s/iter; left time: 3201.2858s
        speed: 0.1309s/iter; left time: 3185.9262s
        speed: 0.1309s/iter; left time: 3172.8487s
        speed: 0.1309s/iter; left time: 3160.0894s
        speed: 0.1310s/iter; left time: 3148.5221s
        speed: 0.1309s/iter; left time: 3133.3825s
        speed: 0.1309s/iter; left time: 3121.3162s
        speed: 0.1309s/iter; left time: 3106.9576s
        speed: 0.1309s/iter; left time: 3095.5211s
Epoch: 4 cost time: 514.96653175354
Epoch: 4, Steps: 3934 | Train Loss: -48.7457672 Vali Loss: -48.3127411 
EarlyStopping counter: 3 out of 3
Early stopping
------------ Options -------------
anormly_ratio: 50.0
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD
input_c: 122
k: 3
lr: 0.0001
mode: test
model_save_path: checkpoints
num_epochs: 10
output_c: 122
pretrained_model: 20
win_size: 100
-------------- End ----------------
test: (22544, 122)
train: (125973, 122)
test: (22544, 122)
train: (125973, 122)
test: (22544, 122)
train: (125973, 122)
test: (22544, 122)
train: (125973, 122)
======================TEST MODE======================
/home/username/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Threshold : 0.0
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.5032, Precision : 0.7028, Recall : 0.2204, F-score : 0.3356 
```

##### r=60.0%

```bash
Threshold : 0.0
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.5284, Precision : 0.6548, Recall : 0.3625, F-score : 0.4666 
```



```bash
------------ Options -------------
anormly_ratio: 60.0
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD
input_c: 122
k: 3
lr: 0.0001
mode: train
model_save_path: checkpoints
num_epochs: 10
output_c: 122
pretrained_model: None
win_size: 100
-------------- End ----------------
test: (22544, 122)
train: (125973, 122)
test: (22544, 122)
train: (125973, 122)
test: (22544, 122)
train: (125973, 122)
test: (22544, 122)
train: (125973, 122)
======================TRAIN MODE======================
        speed: 0.1388s/iter; left time: 5446.8367s
        speed: 0.1314s/iter; left time: 5141.6661s
        speed: 0.1315s/iter; left time: 5133.9673s
        speed: 0.1315s/iter; left time: 5121.2421s
        speed: 0.1365s/iter; left time: 5300.2137s
        speed: 0.1374s/iter; left time: 5322.4496s
        speed: 0.1333s/iter; left time: 5149.1546s
        speed: 0.1323s/iter; left time: 5099.9489s
        speed: 0.1311s/iter; left time: 5038.8110s
        speed: 0.1310s/iter; left time: 5023.0063s
        speed: 0.1445s/iter; left time: 5524.1188s
        speed: 0.1505s/iter; left time: 5740.0504s
        speed: 0.1497s/iter; left time: 5694.0534s
        speed: 0.1498s/iter; left time: 5684.9724s
        speed: 0.1495s/iter; left time: 5657.5665s
        speed: 0.1501s/iter; left time: 5666.0902s
        speed: 0.1508s/iter; left time: 5676.4843s
        speed: 0.1447s/iter; left time: 5432.0509s
        speed: 0.1438s/iter; left time: 5383.0355s
        speed: 0.1459s/iter; left time: 5446.3010s
        speed: 0.1380s/iter; left time: 5138.5641s
        speed: 0.1528s/iter; left time: 5676.7783s
        speed: 0.1533s/iter; left time: 5678.8169s
        speed: 0.1487s/iter; left time: 5494.3238s
        speed: 0.1487s/iter; left time: 5478.8813s
        speed: 0.1354s/iter; left time: 4973.0808s
        speed: 0.1330s/iter; left time: 4874.2198s
        speed: 0.1327s/iter; left time: 4849.0647s
        speed: 0.1334s/iter; left time: 4863.0441s
        speed: 0.1332s/iter; left time: 4840.3917s
        speed: 0.1392s/iter; left time: 5045.4379s
        speed: 0.1484s/iter; left time: 5363.5204s
        speed: 0.1490s/iter; left time: 5370.1684s
        speed: 0.1318s/iter; left time: 4736.2374s
        speed: 0.1317s/iter; left time: 4719.7045s
        speed: 0.1317s/iter; left time: 4705.7837s
        speed: 0.1317s/iter; left time: 4693.7737s
        speed: 0.1380s/iter; left time: 4903.0555s
        speed: 0.1551s/iter; left time: 5496.2793s
Epoch: 1 cost time: 553.4214758872986
Epoch: 1, Steps: 3934 | Train Loss: -47.2930476 Vali Loss: -47.4361793 
Validation loss decreased (inf --> -47.436179).  Saving model ...
Updating learning rate to 0.0001
        speed: 0.5466s/iter; left time: 19300.5711s
        speed: 0.1308s/iter; left time: 4605.8315s
        speed: 0.1311s/iter; left time: 4601.2093s
        speed: 0.1311s/iter; left time: 4589.1366s
        speed: 0.1417s/iter; left time: 4945.4468s
        speed: 0.1421s/iter; left time: 4945.5076s
        speed: 0.1327s/iter; left time: 4604.5278s
        speed: 0.1344s/iter; left time: 4650.3773s
        speed: 0.1396s/iter; left time: 4815.6763s
        speed: 0.1351s/iter; left time: 4649.1411s
        speed: 0.1332s/iter; left time: 4568.2611s
        speed: 0.1401s/iter; left time: 4793.6377s
        speed: 0.1325s/iter; left time: 4520.7700s
        speed: 0.1468s/iter; left time: 4991.2005s
        speed: 0.1412s/iter; left time: 4786.1676s
        speed: 0.1330s/iter; left time: 4496.7579s
        speed: 0.1337s/iter; left time: 4508.0182s
        speed: 0.1333s/iter; left time: 4479.6438s
        speed: 0.1326s/iter; left time: 4442.6618s
        speed: 0.1321s/iter; left time: 4413.8824s
        speed: 0.1310s/iter; left time: 4364.2314s
        speed: 0.1426s/iter; left time: 4734.3299s
        speed: 0.1338s/iter; left time: 4430.3676s
        speed: 0.1325s/iter; left time: 4372.0640s
        speed: 0.1327s/iter; left time: 4367.8091s
        speed: 0.1325s/iter; left time: 4345.4439s
        speed: 0.1327s/iter; left time: 4341.0565s
        speed: 0.1326s/iter; left time: 4324.5873s
        speed: 0.1356s/iter; left time: 4406.5620s
        speed: 0.1464s/iter; left time: 4743.1841s
        speed: 0.1395s/iter; left time: 4506.8946s
        speed: 0.1423s/iter; left time: 4583.7955s
        speed: 0.1461s/iter; left time: 4691.0209s
        speed: 0.1415s/iter; left time: 4530.2409s
        speed: 0.1423s/iter; left time: 4538.9798s
        speed: 0.1390s/iter; left time: 4421.9290s
        speed: 0.1395s/iter; left time: 4421.7746s
        speed: 0.1370s/iter; left time: 4330.0266s
        speed: 0.1368s/iter; left time: 4311.1386s
Epoch: 2 cost time: 539.1221182346344
Epoch: 2, Steps: 3934 | Train Loss: -48.4677824 Vali Loss: -47.9757537 
EarlyStopping counter: 1 out of 3
Updating learning rate to 5e-05
        speed: 0.4911s/iter; left time: 15407.6395s
        speed: 0.1325s/iter; left time: 4144.1985s
        speed: 0.1348s/iter; left time: 4202.4603s
        speed: 0.1361s/iter; left time: 4230.0066s
        speed: 0.1315s/iter; left time: 4074.3328s
        speed: 0.1355s/iter; left time: 4182.4872s
        speed: 0.1483s/iter; left time: 4562.3436s
        speed: 0.1509s/iter; left time: 4627.7957s
        speed: 0.1495s/iter; left time: 4572.1388s
        speed: 0.1501s/iter; left time: 4574.2171s
        speed: 0.1498s/iter; left time: 4548.7989s
        speed: 0.1459s/iter; left time: 4416.7031s
        speed: 0.1436s/iter; left time: 4332.7850s
        speed: 0.1434s/iter; left time: 4311.9022s
        speed: 0.1465s/iter; left time: 4390.5848s
        speed: 0.1476s/iter; left time: 4409.8262s
        speed: 0.1477s/iter; left time: 4398.3314s
        speed: 0.1445s/iter; left time: 4286.7148s
        speed: 0.1463s/iter; left time: 4327.3261s
        speed: 0.1437s/iter; left time: 4235.5724s
        speed: 0.1437s/iter; left time: 4219.7474s
        speed: 0.1462s/iter; left time: 4280.5025s
        speed: 0.1448s/iter; left time: 4223.6321s
        speed: 0.1444s/iter; left time: 4198.9182s
        speed: 0.1446s/iter; left time: 4190.8026s
        speed: 0.1442s/iter; left time: 4164.7457s
        speed: 0.1446s/iter; left time: 4159.9640s
        speed: 0.1440s/iter; left time: 4129.8085s
        speed: 0.1447s/iter; left time: 4133.1227s
        speed: 0.1444s/iter; left time: 4111.4583s
        speed: 0.1449s/iter; left time: 4110.1645s
        speed: 0.1440s/iter; left time: 4072.0241s
        speed: 0.1438s/iter; left time: 4050.8166s
        speed: 0.1444s/iter; left time: 4055.0368s
        speed: 0.1441s/iter; left time: 4031.9231s
        speed: 0.1442s/iter; left time: 4020.3876s
        speed: 0.1445s/iter; left time: 4012.2267s
        speed: 0.1448s/iter; left time: 4007.0133s
        speed: 0.1445s/iter; left time: 3985.0134s
Epoch: 3 cost time: 565.8869743347168
Epoch: 3, Steps: 3934 | Train Loss: -48.6567995 Vali Loss: -48.1764615 
EarlyStopping counter: 2 out of 3
Updating learning rate to 2.5e-05
        speed: 0.5192s/iter; left time: 14247.0282s
        speed: 0.1448s/iter; left time: 3958.9565s
        speed: 0.1448s/iter; left time: 3943.9087s
        speed: 0.1443s/iter; left time: 3916.2532s
        speed: 0.1443s/iter; left time: 3901.8156s
        speed: 0.1440s/iter; left time: 3880.5454s
        speed: 0.1444s/iter; left time: 3874.2425s
        speed: 0.1448s/iter; left time: 3871.7225s
        speed: 0.1443s/iter; left time: 3842.9239s
        speed: 0.1441s/iter; left time: 3824.7977s
        speed: 0.1441s/iter; left time: 3810.2506s
        speed: 0.1442s/iter; left time: 3797.6073s
        speed: 0.1442s/iter; left time: 3782.7356s
        speed: 0.1443s/iter; left time: 3771.7759s
        speed: 0.1444s/iter; left time: 3759.7137s
        speed: 0.1440s/iter; left time: 3736.1705s
        speed: 0.1444s/iter; left time: 3730.0108s
        speed: 0.1440s/iter; left time: 3707.2472s
        speed: 0.1445s/iter; left time: 3706.0638s
        speed: 0.1438s/iter; left time: 3671.9295s
        speed: 0.1442s/iter; left time: 3669.4479s
        speed: 0.1446s/iter; left time: 3664.2420s
        speed: 0.1446s/iter; left time: 3648.3342s
        speed: 0.1446s/iter; left time: 3634.8287s
        speed: 0.1474s/iter; left time: 3691.6503s
        speed: 0.1511s/iter; left time: 3767.4017s
        speed: 0.1494s/iter; left time: 3711.8324s
        speed: 0.1441s/iter; left time: 3564.2086s
        speed: 0.1439s/iter; left time: 3545.6107s
        speed: 0.1468s/iter; left time: 3603.3738s
        speed: 0.1452s/iter; left time: 3548.3801s
        speed: 0.1446s/iter; left time: 3520.4837s
        speed: 0.1441s/iter; left time: 3491.6876s
        speed: 0.1440s/iter; left time: 3476.8049s
        speed: 0.1448s/iter; left time: 3481.9156s
        speed: 0.1466s/iter; left time: 3508.7926s
        speed: 0.1447s/iter; left time: 3450.3449s
        speed: 0.1440s/iter; left time: 3418.7299s
        speed: 0.1439s/iter; left time: 3401.0288s
Epoch: 4 cost time: 569.6859128475189
Epoch: 4, Steps: 3934 | Train Loss: -48.7182953 Vali Loss: -48.2986017 
EarlyStopping counter: 3 out of 3
Early stopping
------------ Options -------------
anormly_ratio: 60.0
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD
input_c: 122
k: 3
lr: 0.0001
mode: test
model_save_path: checkpoints
num_epochs: 10
output_c: 122
pretrained_model: 20
win_size: 100
-------------- End ----------------
test: (22544, 122)
train: (125973, 122)
test: (22544, 122)
train: (125973, 122)
test: (22544, 122)
train: (125973, 122)
test: (22544, 122)
train: (125973, 122)
======================TEST MODE======================
/home/username/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Threshold : 0.0
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.5284, Precision : 0.6548, Recall : 0.3625, F-score : 0.4666 


```

#### 移除训练集异常点

简单去除训练集异常点数据：

```bash
#train ar=0.5%
#test ar=60%
Threshold : 8.954137840471525e-22
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.4903, Precision : 0.6855, Recall : 0.1930, F-score : 0.3012 
#train ar=60%
#test ar=60%
Threshold : 1.6401431994555087e-32
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.4899, Precision : 0.6622, Recall : 0.2119, F-score : 0.3210 
```

对异常点进行KNN插补：



#### 一对多策略 OvR

单独将每个类标为1，其余标为0，每个类的model checkpoint 使用各自的anomaly ratio单独训练。

```
------------ Options -------------
anormly_ratio: 20.0
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_0
input_c: 122
k: 3
lr: 0.0001
mode: train
model_save_path: checkpoints
num_epochs: 3
output_c: 122
pretrained_model: None
win_size: 100
-------------- End ----------------
Threshold : 0.0
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.7100, Precision : 0.2341, Recall : 0.1776, F-score : 0.2020 

------------ Options -------------
anormly_ratio: 1.0
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_1
input_c: 122
k: 3
lr: 0.0001
mode: train
model_save_path: checkpoints
num_epochs: 3
output_c: 122
pretrained_model: None
win_size: 100
-------------- End ----------------
Threshold : 0.002423033353406936
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9736, Precision : 0.0107, Recall : 0.0094, F-score : 0.0100 

------------ Options -------------
anormly_ratio: 5.0
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_2
input_c: 122
k: 3
lr: 0.0001
mode: train
model_save_path: checkpoints
num_epochs: 3
output_c: 122
pretrained_model: None
win_size: 100
-------------- End ----------------
Threshold : 2.4234104793409644e-19
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9197, Precision : 0.0644, Recall : 0.0604, F-score : 0.0623 

------------ Options -------------
anormly_ratio: 5.0
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_3
input_c: 122
k: 3
lr: 0.0001
mode: train
model_save_path: checkpoints
num_epochs: 3
output_c: 122
pretrained_model: None
win_size: 100
-------------- End ----------------
Threshold : 2.2883160614427485e-21
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9076, Precision : 0.0818, Recall : 0.0675, F-score : 0.0740 

------------ Options -------------
anormly_ratio: 1.0
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_4
input_c: 122
k: 3
lr: 0.0001
mode: train
model_save_path: checkpoints
num_epochs: 3
output_c: 122
pretrained_model: None
win_size: 100
-------------- End ----------------
Threshold : 0.006830912414006861
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9606, Precision : 0.0408, Recall : 0.0151, F-score : 0.0221 

------------ Options -------------
anormly_ratio: 1.0
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_5
input_c: 122
k: 3
lr: 0.0001
mode: train
model_save_path: checkpoints
num_epochs: 3
output_c: 122
pretrained_model: None
win_size: 100
-------------- End ----------------
Threshold : 0.007120268438011376
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9576, Precision : 0.0259, Recall : 0.0082, F-score : 0.0124 

------------ Options -------------
anormly_ratio: 1.0
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_6
input_c: 122
k: 3
lr: 0.0001
mode: train
model_save_path: checkpoints
num_epochs: 3
output_c: 122
pretrained_model: None
win_size: 100
-------------- End ----------------
Threshold : 0.002397903576493261
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9557, Precision : 0.0319, Recall : 0.0123, F-score : 0.0178 

------------ Options -------------
anormly_ratio: 0.01
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_7
input_c: 122
k: 3
lr: 0.0001
mode: train
model_save_path: checkpoints
num_epochs: 3
output_c: 122
pretrained_model: None
win_size: 100
-------------- End ----------------
Threshold : 0.6700110692559966
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9985, Precision : 0.0000, Recall : 0.0000, F-score : 0.0000 

------------ Options -------------
anormly_ratio: 1.0
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_8
input_c: 122
k: 3
lr: 0.0001
mode: train
model_save_path: checkpoints
num_epochs: 3
output_c: 122
pretrained_model: None
win_size: 100
-------------- End ----------------
Threshold : 0.006463531367480735
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9737, Precision : 0.0124, Recall : 0.0084, F-score : 0.0100 

------------ Options -------------
anormly_ratio: 5.0
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_9
input_c: 122
k: 3
lr: 0.0001
mode: train
model_save_path: checkpoints
num_epochs: 3
output_c: 122
pretrained_model: None
win_size: 100
-------------- End ----------------
Threshold : 9.709074460615489e-17
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9228, Precision : 0.0520, Recall : 0.0487, F-score : 0.0503 

------------ Options -------------
anormly_ratio: 1.0
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_10
input_c: 122
k: 3
lr: 0.0001
mode: train
model_save_path: checkpoints
num_epochs: 3
output_c: 122
pretrained_model: None
win_size: 100
-------------- End ----------------
Threshold : 0.0072950472310184325
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9819, Precision : 0.0127, Recall : 0.0169, F-score : 0.0145 

------------ Options -------------
anormly_ratio: 1.0
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_11
input_c: 122
k: 3
lr: 0.0001
mode: train
model_save_path: checkpoints
num_epochs: 3
output_c: 122
pretrained_model: None
win_size: 100
-------------- End ----------------
Threshold : 0.006282573062926521
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9598, Precision : 0.0256, Recall : 0.0088, F-score : 0.0131 

------------ Options -------------
anormly_ratio: 0.1
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_12
input_c: 122
k: 3
lr: 0.0001
mode: train
model_save_path: checkpoints
num_epochs: 3
output_c: 122
pretrained_model: None
win_size: 100
-------------- End ----------------
Threshold : 0.0748524039611232
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9941, Precision : 0.0108, Recall : 0.0244, F-score : 0.0149 

------------ Options -------------
anormly_ratio: 0.5
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_13
input_c: 122
k: 3
lr: 0.0001
mode: train
model_save_path: checkpoints
num_epochs: 3
output_c: 122
pretrained_model: None
win_size: 100
-------------- End ----------------
Threshold : 0.028188115973026333
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9874, Precision : 0.0129, Recall : 0.0150, F-score : 0.0139 

------------ Options -------------
anormly_ratio: 0.5
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_14
input_c: 122
k: 3
lr: 0.0001
mode: train
model_save_path: checkpoints
num_epochs: 3
output_c: 122
pretrained_model: None
win_size: 100
-------------- End ----------------
Threshold : 0.012299377284944485
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9887, Precision : 0.0000, Recall : 0.0000, F-score : 0.0000 

------------ Options -------------
anormly_ratio: 0.01
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_15
input_c: 122
k: 3
lr: 0.0001
mode: train
model_save_path: checkpoints
num_epochs: 3
output_c: 122
pretrained_model: None
win_size: 100
-------------- End ----------------
Threshold : 0.7865708318292497
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9984, Precision : 0.0000, Recall : 0.0000, F-score : 0.0000 

------------ Options -------------
anormly_ratio: 1.0
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_16
input_c: 122
k: 3
lr: 0.0001
mode: train
model_save_path: checkpoints
num_epochs: 3
output_c: 122
pretrained_model: None
win_size: 100
-------------- End ----------------
Threshold : 0.0023502711369655835
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9731, Precision : 0.0036, Recall : 0.0030, F-score : 0.0033 

------------ Options -------------
anormly_ratio: 0.5
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_17
input_c: 122
k: 3
lr: 0.0001
mode: train
model_save_path: checkpoints
num_epochs: 3
output_c: 122
pretrained_model: None
win_size: 100
-------------- End ----------------
Threshold : 0.02240190408192611
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9866, Precision : 0.0122, Recall : 0.0142, F-score : 0.0131 

------------ Options -------------
anormly_ratio: 1.0
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_18
input_c: 122
k: 3
lr: 0.0001
mode: train
model_save_path: checkpoints
num_epochs: 3
output_c: 122
pretrained_model: None
win_size: 100
-------------- End ----------------
Threshold : 0.007351175076328215
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9767, Precision : 0.0000, Recall : 0.0000, F-score : 0.0000 

------------ Options -------------
anormly_ratio: 0.5
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_19
input_c: 122
k: 3
lr: 0.0001
mode: train
model_save_path: checkpoints
num_epochs: 3
output_c: 122
pretrained_model: None
win_size: 100
-------------- End ----------------
Threshold : 0.023260270589962658
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9855, Precision : 0.0115, Recall : 0.0127, F-score : 0.0121 

------------ Options -------------
anormly_ratio: 0.05
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_20
input_c: 122
k: 3
lr: 0.0001
mode: train
model_save_path: checkpoints
num_epochs: 3
output_c: 122
pretrained_model: None
win_size: 100
-------------- End ----------------
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9965, Precision : 0.0000, Recall : 0.0000, F-score : 0.0000 

------------ Options -------------
anormly_ratio: 0.05
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_21
input_c: 122
k: 3
lr: 0.0001
mode: train
model_save_path: checkpoints
num_epochs: 3
output_c: 122
pretrained_model: None
win_size: 100
-------------- End ----------------
Threshold : 0.10992800116911451
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9961, Precision : 0.0000, Recall : 0.0000, F-score : 0.0000 

------------ Options -------------
anormly_ratio: 0.05
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_22
input_c: 122
k: 3
lr: 0.0001
mode: train
model_save_path: checkpoints
num_epochs: 3
output_c: 122
pretrained_model: None
win_size: 100
-------------- End ----------------
Threshold : 0.13109133851528165
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9963, Precision : 0.0000, Recall : 0.0000, F-score : 0.0000 

------------ Options -------------
anormly_ratio: 0.01
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_23
input_c: 122
k: 3
lr: 0.0001
mode: train
model_save_path: checkpoints
num_epochs: 3
output_c: 122
pretrained_model: None
win_size: 100
-------------- End ----------------
Threshold : 0.8262347285803151
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9996, Precision : 0.0000, Recall : 0.0000, F-score : 0.0000 

------------ Options -------------
anormly_ratio: 0.05
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_24
input_c: 122
k: 3
lr: 0.0001
mode: train
model_save_path: checkpoints
num_epochs: 3
output_c: 122
pretrained_model: None
win_size: 100
-------------- End ----------------
Threshold : 0.09242837175354189
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9966, Precision : 0.0000, Recall : 0.0000, F-score : 0.0000 

------------ Options -------------
anormly_ratio: 0.01
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_25
input_c: 122
k: 3
lr: 0.0001
mode: train
model_save_path: checkpoints
num_epochs: 3
output_c: 122
pretrained_model: None
win_size: 100
-------------- End ----------------
Threshold : 0.36379067861726466
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9995, Precision : 0.0000, Recall : 0.0000, F-score : 0.0000 

------------ Options -------------
anormly_ratio: 0.05
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_26
input_c: 122
k: 3
lr: 0.0001
mode: train
model_save_path: checkpoints
num_epochs: 3
output_c: 122
pretrained_model: None
win_size: 100
-------------- End ----------------
Threshold : 0.1089880059286936
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9964, Precision : 0.0000, Recall : 0.0000, F-score : 0.0000 

------------ Options -------------
anormly_ratio: 0.05
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_27
input_c: 122
k: 3
lr: 0.0001
mode: train
model_save_path: checkpoints
num_epochs: 3
output_c: 122
pretrained_model: None
win_size: 100
-------------- End ----------------
Threshold : 0.04048785941675266
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9976, Precision : 0.0000, Recall : 0.0000, F-score : 0.0000 

------------ Options -------------
anormly_ratio: 0.01
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_28
input_c: 122
k: 3
lr: 0.0001
mode: train
model_save_path: checkpoints
num_epochs: 3
output_c: 122
pretrained_model: None
win_size: 100
-------------- End ----------------
Threshold : 0.6861327379285682
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9990, Precision : 0.0000, Recall : 0.0000, F-score : 0.0000 

------------ Options -------------
anormly_ratio: 0.01
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_29
input_c: 122
k: 3
lr: 0.0001
mode: train
model_save_path: checkpoints
num_epochs: 3
output_c: 122
pretrained_model: None
win_size: 100
-------------- End ----------------
Threshold : 0.7651060473798674
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9992, Precision : 0.0000, Recall : 0.0000, F-score : 0.0000 

------------ Options -------------
anormly_ratio: 0.01
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_30
input_c: 122
k: 3
lr: 0.0001
mode: train
model_save_path: checkpoints
num_epochs: 3
output_c: 122
pretrained_model: None
win_size: 100
-------------- End ----------------
Threshold : 0.6199230782323712
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9992, Precision : 0.0000, Recall : 0.0000, F-score : 0.0000 

------------ Options -------------
anormly_ratio: 0.01
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_31
input_c: 122
k: 3
lr: 0.0001
mode: train
model_save_path: checkpoints
num_epochs: 3
output_c: 122
pretrained_model: None
win_size: 100
-------------- End ----------------
Threshold : 0.5765992528795936
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9993, Precision : 0.0714, Recall : 0.2500, F-score : 0.1111 

------------ Options -------------
anormly_ratio: 0.01
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_32
input_c: 122
k: 3
lr: 0.0001
mode: train
model_save_path: checkpoints
num_epochs: 3
output_c: 122
pretrained_model: None
win_size: 100
-------------- End ----------------

```



# 4. Optuna超参数优化



首先安装环境

```bash
pip install optuna
pip install optuna-dashboard
```

然后参照对NSLKDD数据的命令运行（共开8个线程在8张GPU执行）：
```bash
python optuna_optimization.py --cuda=0 --dataset=NSLKDD --n_trials=30 --host & python optuna_optimization.py --cuda=1 --dataset=NSLKDD --n_trials=30 & python optuna_optimization.py --cuda=2 --dataset=NSLKDD --n_trials=30 & python optuna_optimization.py --cuda=3 --dataset=NSLKDD --n_trials=30 & python optuna_optimization.py --cuda=4 --dataset=NSLKDD --n_trials=30 & python optuna_optimization.py --cuda=5 --dataset=NSLKDD --n_trials=30 & python optuna_optimization.py --cuda=6 --dataset=NSLKDD --n_trials=30 & python optuna_optimization.py --cuda=7 --dataset=NSLKDD --n_trials=30
```

控制台web端这样打开：
```bash
optuna-dashboard sqlite:///db.sqlite3
```

这里面的`db.sqlite3`换成你存的数据库文件的名字，例如`anomaly_transformer_swat_study.db`

# x. References

- 论文解读1：https://zhuanlan.zhihu.com/p/553509779
- Detection Adjustment:https://blog.csdn.net/a571625338/article/details/127979281
- 网络安全数据集调研：https://zhuanlan.zhihu.com/p/149130456

- 在KDD99数据集进行入侵检测1：https://cloud.tencent.com/developer/article/1621977

- 代码debug相关
  - CodeBERT代码bug修复：https://juejin.cn/post/7034105242841153550
  - 基于Transformer的微软DeepBug代码bug修复：https://arxiv.org/pdf/2105.09352.pdf
  - 微软代码智能benchmark（含代码纠错Bugs2fix）：https://github.com/microsoft/CodeXGLUE
    - 微软代码智能benchmark中文介绍：https://www.msra.cn/zh-cn/news/features/codexglue
