export CUDA_VISIBLE_DEVICES=0

# python main.py --anormly_ratio 0.5 --num_epochs 3    --batch_size 32  --mode train --dataset SWaT  --data_path dataset/SWaT --input_c 51    --output_c 51
# python main.py --anormly_ratio 0.1  --num_epochs 10        --batch_size 32     --mode test    --dataset SWaT   --data_path dataset/SWaT  --input_c 51    --output_c 51  --pretrained_model 10

# Accuracy : 0.9914, Precision : 0.9412, Recall : 0.9907, F-score : 0.9653 
# [I 2024-01-13 09:06:35,663] Trial 30 finished with value: 0.9683116573503561 and parameters: {'lr': 0.0002357128774064241, 'num_epochs': 13, 'k': 8, 'win_size': 111, 'batch_size': 47, 'anormly_ratio': 0.20225878617481274}. Best is trial 16 with value: 0.9703533569341025.

python main.py --anormly_ratio 0.20225878617481274 --num_epochs 13    --batch_size 47  --mode train --dataset SWaT  --data_path dataset/SWaT --input_c 51    --output_c 51 --win_size 111 --k 8 --lr 0.0002357128774064241 --cuda_num 0
python main.py --anormly_ratio 0.20225878617481274  --num_epochs 13        --batch_size 47     --mode test    --dataset SWaT   --data_path dataset/SWaT  --input_c 51    --output_c 51  --win_size 111 --pretrained_model 13 --k 8 --lr 0.0002357128774064241 --cuda_num 0