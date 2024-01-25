"""
Optuna example that optimizes multi-layer perceptrons using PyTorch.

In this example, we optimize the validation accuracy of fashion product recognition using
PyTorch and FashionMNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole FashionMNIST dataset,
we here use a small subset of it.

"""
import os
import argparse
import torch
import optuna
from optuna.trial import TrialState
from solver import Solver

# 1.使用trial中定义和包含的参数来定义模型
# 2.使用trial与模型定义目标函数，对于神经网络来说，一次训练后进行一次预测验证，验证指标就可以作为适应度
# 3.创建一次study进行参数优化搜索

class Argument(object):
    def __init__(self, trial, input_c=122, output_c=122, pretrained_model=None,dataset='credit',mode='train',data_path='./dataset/creditcard_ts.csv',model_save_path='checkpoints',cuda_num=0):
        self.lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        self.num_epochs = trial.suggest_int("num_epochs", 1, 30, log=True)
        self.k = trial.suggest_int("k", 1, 10, log=True)
        self.win_size = trial.suggest_int("win_size", 50, 200, log=True)
        self.input_c = input_c
        self.output_c = output_c
        self.batch_size = trial.suggest_int("batch_size", 16, 256, log=True)
        self.anormly_ratio = trial.suggest_float("anormly_ratio", 1e-3, 100, log=True)
        self.pretrained_model = None
        self.dataset = dataset
        self.mode = mode #['train', 'test']
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.cuda_num=cuda_num
        self.device=None

    def reset_lr(self,lr):
        self.lr=lr

    def reset_num_epochs(self,num_epochs):
        self.num_epochs=num_epochs

    def reset_k(self,k):
        self.k=k

    def reset_win_size(self,win_size):
        self.win_size=win_size

    def reset_batch_size(self,batch_size):
        self.batch_size=batch_size

    def reset_anormly_ratio(self,anormly_ratio):
        self.anormly_ratio=anormly_ratio

    def set_pretrained_model_from_num_epochs(self):
        self.pretrained_model = self.num_epochs

# --anormly_ratio 0.5 --num_epochs 3    --batch_size 32  --mode train --dataset SWaT  --data_path dataset/SWaT --input_c 51    --output_c 51
# --anormly_ratio 0.1  --num_epochs 10        --batch_size 32     --mode test    --dataset SWaT   --data_path dataset/SWaT  --input_c 51    --output_c 51  --pretrained_model 10

cuda_num = 0

def swat_objective(trial):
    train_arg = Argument(trial,input_c=51,output_c=51, mode='train', dataset='SWaT', data_path='dataset/SWaT', cuda_num = cuda_num)
    solver_train = Solver(vars(train_arg))

    test_arg = Argument(trial,input_c=51,output_c=51, mode='test', dataset='SWaT', data_path='dataset/SWaT', cuda_num = cuda_num)
    test_arg.set_pretrained_model_from_num_epochs
    solver_test = Solver(vars(test_arg))

    solver_train.train()
    accuracy, precision, recall, f_score = solver_test.test()

    # 单目标
    # Define weights based on the importance of each metric
    weight_accuracy = 0.1
    weight_precision = 0.3
    weight_recall = 0.3
    weight_f_score = 0.3

    # Combine metrics using a weighted sum
    combined_metric = (
        weight_accuracy * accuracy +
        weight_precision * precision +
        weight_recall * recall +
        weight_f_score * f_score
    )

    # 返回多个指标可实现多目标优化
    return combined_metric

def nslkdd_objective(trial):
    train_arg = Argument(trial,input_c=122,output_c=122, mode='train', dataset='NSLKDD', data_path='dataset/NSLKDD', cuda_num = cuda_num)
    solver_train = Solver(vars(train_arg))

    test_arg = Argument(trial,input_c=122,output_c=122, mode='test', dataset='NSLKDD', data_path='dataset/NSLKDD', cuda_num = cuda_num)
    test_arg.set_pretrained_model_from_num_epochs
    solver_test = Solver(vars(test_arg))

    solver_train.train()
    accuracy, precision, recall, f_score = solver_test.test()

    # # 单目标
    # # Define weights based on the importance of each metric
    # weight_accuracy = 0.1
    # weight_precision = 0.3
    # weight_recall = 0.3
    # weight_f_score = 0.3

    # # Combine metrics using a weighted sum
    # combined_metric = (
    #     weight_accuracy * accuracy +
    #     weight_precision * precision +
    #     weight_recall * recall +
    #     weight_f_score * f_score
    # )

    # 返回多个指标可实现多目标优化
    return accuracy, precision, recall, f_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_num', type=int, default=0)
    parser.add_argument('--host', type=bool, default=False)
    parser.add_argument('--dataset', type=str, default='SWaT')
    parser.add_argument('--n_trials', type=int, default=30)
    config = parser.parse_args()
    cuda_num = config.cuda_num

    study_name = "anomaly_transformer_"+config.dataset+"_study" 
    storage_name = "sqlite:///anomaly_transformer_swat_study.db".format(study_name)

    print(type(config.host))
    print(config.host)

    if(config.host):
        print('Process is host, creating study')
        study = optuna.create_study(directions=["maximize","maximize","maximize","maximize"],study_name=study_name, storage=storage_name)
    else:
        print('Process is NOT host, loading')
        study = optuna.load_study(
        study_name=study_name, storage=storage_name
    )
        
    if(config.dataset == 'SWaT'):
        study.optimize(swat_objective, n_trials=config.n_trials, timeout=9000)
    elif(config.dataset == 'NSLKDD'):
        study.optimize(nslkdd_objective, n_trials=config.n_trials, timeout=9000)
    else:
        raise Exception('No such dataset or not implemented yet')
    # study = optuna.create_study(direction="maximize")
    

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))