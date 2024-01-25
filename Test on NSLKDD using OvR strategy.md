# Test on NSLKDD using OvR strategy

test sets includes 37 class of attack types, and we encode it into 37 label list using OvR strategy. The model checkpoint is trained at once with anomaly ration=1. And the testing results are as follows.

| #    | Accuracy | Precision | Recall | F-score | Anomaly Label   |
| ---- | -------- | --------- | ------ | ------- | --------------- |
| 1    | 0.7897   | 0.2722    | 0.0105 | 0.0203  | neptune         |
| 2    | 0.9790   | 0.0247    | 0.0126 | 0.0167  | saint           |
| 3    | 0.9497   | 0.0798    | 0.0131 | 0.0225  | mscan           |
| 4    | 0.9400   | 0.1369    | 0.0187 | 0.0329  | guess_passwd    |
| 5    | 0.9638   | 0.0309    | 0.0076 | 0.0121  | smurf           |
| 6    | 0.9608   | 0.0432    | 0.0095 | 0.0156  | apache2         |
| 7    | 0.9612   | 0.0679    | 0.0150 | 0.0246  | satan           |
| 8    | 0.9919   | 0.0000    | 0.0000 | 0.0000  | buffer_overflow |
| 9    | 0.9770   | 0.0062    | 0.0028 | 0.0039  | back            |
| 10   | 0.9519   | 0.0793    | 0.0138 | 0.0235  | warezmaster     |
| 11   | 0.9850   | 0.0062    | 0.0056 | 0.0059  | snmpgetattack   |
| 12   | 0.9631   | 0.0432    | 0.0102 | 0.0166  | processtable    |
| 13   | 0.9911   | 0.0062    | 0.0244 | 0.0099  | pod             |
| 14   | 0.9871   | 0.0123    | 0.0150 | 0.0136  | httptunnel      |
| 15   | 0.9896   | 0.0000    | 0.0000 | 0.0000  | nmap            |
| 16   | 0.9921   | 0.0000    | 0.0000 | 0.0000  | ps              |
| 17   | 0.9784   | 0.0245    | 0.0121 | 0.0162  | snmpguess       |
| 18   | 0.9868   | 0.0185    | 0.0213 | 0.0198  | ipsweep         |
| 19   | 0.9798   | 0.0000    | 0.0000 | 0.0000  | mailbomb        |
| 20   | 0.9863   | 0.0309    | 0.0318 | 0.0313  | portsweep       |
| 21   | 0.9920   | 0.0000    | 0.0000 | 0.0000  | multihop        |
| 22   | 0.9921   | 0.0062    | 0.0588 | 0.0112  | named           |
| 23   | 0.9922   | 0.0000    | 0.0000 | 0.0000  | sendmail        |
| 24   | 0.9927   | 0.0000    | 0.0000 | 0.0000  | loadmodule      |
| 25   | 0.9922   | 0.0000    | 0.0000 | 0.0000  | xterm           |
| 26   | 0.9927   | 0.0000    | 0.0000 | 0.0000  | worm            |
| 27   | 0.9923   | 0.0000    | 0.0000 | 0.0000  | teardrop        |
| 28   | 0.9923   | 0.0000    | 0.0000 | 0.0000  | rootkit         |
| 29   | 0.9924   | 0.0000    | 0.0000 | 0.0000  | xlock           |
| 30   | 0.9927   | 0.0000    | 0.0000 | 0.0000  | perl            |
| 31   | 0.9925   | 0.0000    | 0.0000 | 0.0000  | land            |
| 32   | 0.9926   | 0.0000    | 0.0000 | 0.0000  | xsnoop          |
| 33   | 0.9927   | 0.0000    | 0.0000 | 0.0000  | sqlattack       |
| 34   | 0.9927   | 0.0000    | 0.0000 | 0.0000  | ftp_write       |
| 35   | 0.9928   | 0.0000    | 0.0000 | 0.0000  | imap            |
| 36   | 0.9927   | 0.0000    | 0.0000 | 0.0000  | udpstorm        |
| 37   | 0.9927   | 0.0000    | 0.0000 | 0.0000  | phf             |



## Complete console outputs

```bash
(Anomaly-Transformer) ranlychan@ranlychan-ubuntu:/media/ranlychan/3E6E20236E1FD28F/Dev/Anomaly-Transformer$ bash ./scripts/NSLKDD.sh 
------------ Options -------------
anormly_ratio: 0.5
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_0
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
/home/ranlychan/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Threshold : 0.025000463612377857
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.7897, Precision : 0.2722, Recall : 0.0105, F-score : 0.0203 
------------ Options -------------
anormly_ratio: 0.5
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_1
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
/home/ranlychan/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Threshold : 0.025000463612377857
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9790, Precision : 0.0247, Recall : 0.0126, F-score : 0.0167 
------------ Options -------------
anormly_ratio: 0.5
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_2
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
/home/ranlychan/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Threshold : 0.025000463612377857
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9497, Precision : 0.0798, Recall : 0.0131, F-score : 0.0225 
------------ Options -------------
anormly_ratio: 0.5
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_3
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
/home/ranlychan/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Threshold : 0.025000463612377857
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9400, Precision : 0.1369, Recall : 0.0187, F-score : 0.0329 
------------ Options -------------
anormly_ratio: 0.5
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_4
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
/home/ranlychan/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Threshold : 0.025000463612377857
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9638, Precision : 0.0309, Recall : 0.0076, F-score : 0.0121 
------------ Options -------------
anormly_ratio: 0.5
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_5
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
/home/ranlychan/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Threshold : 0.025000463612377857
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9608, Precision : 0.0432, Recall : 0.0095, F-score : 0.0156 
------------ Options -------------
anormly_ratio: 0.5
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_6
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
/home/ranlychan/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Threshold : 0.025000463612377857
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9612, Precision : 0.0679, Recall : 0.0150, F-score : 0.0246 
------------ Options -------------
anormly_ratio: 0.5
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_7
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
/home/ranlychan/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Threshold : 0.025000463612377857
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9919, Precision : 0.0000, Recall : 0.0000, F-score : 0.0000 
------------ Options -------------
anormly_ratio: 0.5
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_8
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
/home/ranlychan/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Threshold : 0.025000463612377857
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9770, Precision : 0.0062, Recall : 0.0028, F-score : 0.0039 
------------ Options -------------
anormly_ratio: 0.5
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_9
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
/home/ranlychan/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Threshold : 0.025000463612377857
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9519, Precision : 0.0793, Recall : 0.0138, F-score : 0.0235 
------------ Options -------------
anormly_ratio: 0.5
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_10
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
/home/ranlychan/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Threshold : 0.025000463612377857
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9850, Precision : 0.0062, Recall : 0.0056, F-score : 0.0059 
------------ Options -------------
anormly_ratio: 0.5
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_11
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
/home/ranlychan/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Threshold : 0.025000463612377857
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9631, Precision : 0.0432, Recall : 0.0102, F-score : 0.0166 
------------ Options -------------
anormly_ratio: 0.5
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_12
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
/home/ranlychan/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Threshold : 0.025000463612377857
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9911, Precision : 0.0062, Recall : 0.0244, F-score : 0.0099 
------------ Options -------------
anormly_ratio: 0.5
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_13
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
/home/ranlychan/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Threshold : 0.025000463612377857
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9871, Precision : 0.0123, Recall : 0.0150, F-score : 0.0136 
------------ Options -------------
anormly_ratio: 0.5
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_14
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
/home/ranlychan/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
CThreshold : 0.025000463612377857
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9896, Precision : 0.0000, Recall : 0.0000, F-score : 0.0000 
------------ Options -------------
anormly_ratio: 0.5
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_15
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
/home/ranlychan/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Threshold : 0.025000463612377857
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9921, Precision : 0.0000, Recall : 0.0000, F-score : 0.0000 
------------ Options -------------
anormly_ratio: 0.5
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_16
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
/home/ranlychan/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Threshold : 0.025000463612377857
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9784, Precision : 0.0245, Recall : 0.0121, F-score : 0.0162 
------------ Options -------------
anormly_ratio: 0.5
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_17
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
/home/ranlychan/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Threshold : 0.025000463612377857
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9868, Precision : 0.0185, Recall : 0.0213, F-score : 0.0198 
------------ Options -------------
anormly_ratio: 0.5
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_18
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
/home/ranlychan/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Threshold : 0.025000463612377857
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9798, Precision : 0.0000, Recall : 0.0000, F-score : 0.0000 
------------ Options -------------
anormly_ratio: 0.5
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_19
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
/home/ranlychan/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Threshold : 0.025000463612377857
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9863, Precision : 0.0309, Recall : 0.0318, F-score : 0.0313 
------------ Options -------------
anormly_ratio: 0.5
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_20
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
/home/ranlychan/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Threshold : 0.025000463612377857
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9920, Precision : 0.0000, Recall : 0.0000, F-score : 0.0000 
------------ Options -------------
anormly_ratio: 0.5
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_21
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
/home/ranlychan/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Threshold : 0.025000463612377857
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9921, Precision : 0.0062, Recall : 0.0588, F-score : 0.0112 
------------ Options -------------
anormly_ratio: 0.5
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_22
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
/home/ranlychan/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Threshold : 0.025000463612377857
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9922, Precision : 0.0000, Recall : 0.0000, F-score : 0.0000 
------------ Options -------------
anormly_ratio: 0.5
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_23
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
/home/ranlychan/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Threshold : 0.025000463612377857
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9927, Precision : 0.0000, Recall : 0.0000, F-score : 0.0000 
------------ Options -------------
anormly_ratio: 0.5
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_24
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
/home/ranlychan/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Threshold : 0.025000463612377857
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9922, Precision : 0.0000, Recall : 0.0000, F-score : 0.0000 
------------ Options -------------
anormly_ratio: 0.5
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_25
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
/home/ranlychan/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Threshold : 0.025000463612377857
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9927, Precision : 0.0000, Recall : 0.0000, F-score : 0.0000 
------------ Options -------------
anormly_ratio: 0.5
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_26
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
/home/ranlychan/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Threshold : 0.025000463612377857
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9923, Precision : 0.0000, Recall : 0.0000, F-score : 0.0000 
------------ Options -------------
anormly_ratio: 0.5
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_27
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
/home/ranlychan/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Threshold : 0.025000463612377857
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9923, Precision : 0.0000, Recall : 0.0000, F-score : 0.0000 
------------ Options -------------
anormly_ratio: 0.5
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_28
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
/home/ranlychan/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Threshold : 0.025000463612377857
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9924, Precision : 0.0000, Recall : 0.0000, F-score : 0.0000 
------------ Options -------------
anormly_ratio: 0.5
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_29
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
/home/ranlychan/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Threshold : 0.025000463612377857
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9927, Precision : 0.0000, Recall : 0.0000, F-score : 0.0000 
------------ Options -------------
anormly_ratio: 0.5
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_30
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
/home/ranlychan/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Threshold : 0.025000463612377857
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9925, Precision : 0.0000, Recall : 0.0000, F-score : 0.0000 
------------ Options -------------
anormly_ratio: 0.5
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_31
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
/home/ranlychan/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Threshold : 0.025000463612377857
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9926, Precision : 0.0000, Recall : 0.0000, F-score : 0.0000 
------------ Options -------------
anormly_ratio: 0.5
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_32
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
/home/ranlychan/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Threshold : 0.025000463612377857
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9927, Precision : 0.0000, Recall : 0.0000, F-score : 0.0000 
------------ Options -------------
anormly_ratio: 0.5
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_33
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
/home/ranlychan/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Threshold : 0.025000463612377857
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9927, Precision : 0.0000, Recall : 0.0000, F-score : 0.0000 
------------ Options -------------
anormly_ratio: 0.5
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_34
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
/home/ranlychan/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Threshold : 0.025000463612377857
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9928, Precision : 0.0000, Recall : 0.0000, F-score : 0.0000 
------------ Options -------------
anormly_ratio: 0.5
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_35
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
/home/ranlychan/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Threshold : 0.025000463612377857
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9927, Precision : 0.0000, Recall : 0.0000, F-score : 0.0000 
------------ Options -------------
anormly_ratio: 0.5
batch_size: 32
data_path: dataset/NSLKDD
dataset: NSLKDD_36
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
/home/ranlychan/anaconda3/envs/Anomaly-Transformer/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
Threshold : 0.025000463612377857
pred:    (22500,)
gt:      (22500,)
pred:  (22500,)
gt:    (22500,)
Accuracy : 0.9927, Precision : 0.0000, Recall : 0.0000, F-score : 0.0000 
```

