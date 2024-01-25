export CUDA_VISIBLE_DEVICES=0

python main.py --anormly_ratio 60 --num_epochs 10    --batch_size 64  --mode train --dataset NSLKDD  --data_path dataset/NSLKDD --input_c 122    --output_c 122
python main.py --anormly_ratio 60  --num_epochs 10        --batch_size 64     --mode test    --dataset NSLKDD   --data_path dataset/NSLKDD  --input_c 122    --output_c 122  --pretrained_model 20


#python main.py --anormly_ratio 20 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_0 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 1 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_1 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 5 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_2 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 5 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_3 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 1 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_4 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 1 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_5 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 1 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_6 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 0.01 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_7 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 1 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_8 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 5 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_9 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 1 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_10 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 1 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_11 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 0.1 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_12 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 0.5 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_13 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 0.5 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_14 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 0.01 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_15 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 1 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_16 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 0.5 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_17 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 1 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_18 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 0.5 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_19 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 0.05 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_20 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 0.05--num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_21 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 0.05 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_22 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 0.01 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_23 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 0.05 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_24 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 0.01 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_25 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 0.05 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_26 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 0.05 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_27 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 0.01 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_28 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 0.01 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_29 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 0.01 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_30 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 0.01 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_31 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 0.01 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_32 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 0.01 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_33 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 0.01 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_34 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 0.01 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_35 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 0.01 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_36 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20

## concat train and test label ratio
#normal             0.518823
#neptune            0.308860
#satan              0.029411
#ipsweep            0.025182
#smurf              0.022294
#portsweep          0.020792
#nmap               0.010544
#back               0.008854
#guess_passwd       0.008645
#mscan              0.006706
#warezmaster        0.006491
#teardrop           0.006087
#warezclient        0.005993
#apache2            0.004962
#processtable       0.004612
#snmpguess          0.002229
#saint              0.002148
#mailbomb           0.001973
#pod                0.001629
#snmpgetattack      0.001199
#httptunnel         0.000896
#buffer_overflow    0.000337
#land               0.000168
#multihop           0.000168
#rootkit            0.000155
#named              0.000114
#ps                 0.000101
#sendmail           0.000094
#xterm              0.000088
#imap               0.000081
#loadmodule         0.000074
#ftp_write          0.000074
#xlock              0.000061
#phf                0.000040
#perl               0.000034
#xsnoop             0.000027
#spy                0.000013
#worm               0.000013
#sqlattack          0.000013
#udpstorm           0.000013

#python main.py --anormly_ratio 20 --num_epochs 3 --batch_size 32 --mode train --dataset NSLKDD_0 --data_path dataset/NSLKDD --input_c 122 --output_c 122 && python main.py --anormly_ratio 20 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_0 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 1 --num_epochs 3 --batch_size 32 --mode train --dataset NSLKDD_1 --data_path dataset/NSLKDD --input_c 122 --output_c 122 && python main.py --anormly_ratio 1 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_1 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 5 --num_epochs 3 --batch_size 32 --mode train --dataset NSLKDD_2 --data_path dataset/NSLKDD --input_c 122 --output_c 122 && python main.py --anormly_ratio 5 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_2 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 5 --num_epochs 3 --batch_size 32 --mode train --dataset NSLKDD_3 --data_path dataset/NSLKDD --input_c 122 --output_c 122 && python main.py --anormly_ratio 5 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_3 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 1 --num_epochs 3 --batch_size 32 --mode train --dataset NSLKDD_4 --data_path dataset/NSLKDD --input_c 122 --output_c 122 && python main.py --anormly_ratio 1 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_4 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 1 --num_epochs 3 --batch_size 32 --mode train --dataset NSLKDD_5 --data_path dataset/NSLKDD --input_c 122 --output_c 122 && python main.py --anormly_ratio 1 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_5 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 1 --num_epochs 3 --batch_size 32 --mode train --dataset NSLKDD_6 --data_path dataset/NSLKDD --input_c 122 --output_c 122 && python main.py --anormly_ratio 1 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_6 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 0.01 --num_epochs 3 --batch_size 32 --mode train --dataset NSLKDD_7 --data_path dataset/NSLKDD --input_c 122 --output_c 122 && python main.py --anormly_ratio 0.01 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_7 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 1 --num_epochs 3 --batch_size 32 --mode train --dataset NSLKDD_8 --data_path dataset/NSLKDD --input_c 122 --output_c 122 && python main.py --anormly_ratio 1 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_8 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 5 --num_epochs 3 --batch_size 32 --mode train --dataset NSLKDD_9 --data_path dataset/NSLKDD --input_c 122 --output_c 122 && python main.py --anormly_ratio 5 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_9 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 1 --num_epochs 3 --batch_size 32 --mode train --dataset NSLKDD_10 --data_path dataset/NSLKDD --input_c 122 --output_c 122 && python main.py --anormly_ratio 1 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_10 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 1 --num_epochs 3 --batch_size 32 --mode train --dataset NSLKDD_11 --data_path dataset/NSLKDD --input_c 122 --output_c 122 && python main.py --anormly_ratio 1 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_11 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 0.1 --num_epochs 3 --batch_size 32 --mode train --dataset NSLKDD_12 --data_path dataset/NSLKDD --input_c 122 --output_c 122 && python main.py --anormly_ratio 0.1 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_12 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 0.5 --num_epochs 3 --batch_size 32 --mode train --dataset NSLKDD_13 --data_path dataset/NSLKDD --input_c 122 --output_c 122 && python main.py --anormly_ratio 0.5 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_13 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 0.5 --num_epochs 3 --batch_size 32 --mode train --dataset NSLKDD_14 --data_path dataset/NSLKDD --input_c 122 --output_c 122 && python main.py --anormly_ratio 0.5 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_14 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 0.01 --num_epochs 3 --batch_size 32 --mode train --dataset NSLKDD_15 --data_path dataset/NSLKDD --input_c 122 --output_c 122 && python main.py --anormly_ratio 0.01 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_15 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 1 --num_epochs 3 --batch_size 32 --mode train --dataset NSLKDD_16 --data_path dataset/NSLKDD --input_c 122 --output_c 122 && python main.py --anormly_ratio 1 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_16 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 0.5 --num_epochs 3 --batch_size 32 --mode train --dataset NSLKDD_17 --data_path dataset/NSLKDD --input_c 122 --output_c 122 && python main.py --anormly_ratio 0.5 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_17 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 1 --num_epochs 3 --batch_size 32 --mode train --dataset NSLKDD_18 --data_path dataset/NSLKDD --input_c 122 --output_c 122 && python main.py --anormly_ratio 1 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_18 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 0.5 --num_epochs 3 --batch_size 32 --mode train --dataset NSLKDD_19 --data_path dataset/NSLKDD --input_c 122 --output_c 122 && python main.py --anormly_ratio 0.5 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_19 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 0.05 --num_epochs 3 --batch_size 32 --mode train --dataset NSLKDD_20 --data_path dataset/NSLKDD --input_c 122 --output_c 122 && python main.py --anormly_ratio 0.05 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_20 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 0.05 --num_epochs 3 --batch_size 32 --mode train --dataset NSLKDD_21 --data_path dataset/NSLKDD --input_c 122 --output_c 122 && python main.py --anormly_ratio 0.05 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_21 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 0.05 --num_epochs 3 --batch_size 32 --mode train --dataset NSLKDD_22 --data_path dataset/NSLKDD --input_c 122 --output_c 122 && python main.py --anormly_ratio 0.05 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_22 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 0.01 --num_epochs 3 --batch_size 32 --mode train --dataset NSLKDD_23 --data_path dataset/NSLKDD --input_c 122 --output_c 122 && python main.py --anormly_ratio 0.01 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_23 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 0.05 --num_epochs 3 --batch_size 32 --mode train --dataset NSLKDD_24 --data_path dataset/NSLKDD --input_c 122 --output_c 122 && python main.py --anormly_ratio 0.05 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_24 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 0.01 --num_epochs 3 --batch_size 32 --mode train --dataset NSLKDD_25 --data_path dataset/NSLKDD --input_c 122 --output_c 122 && python main.py --anormly_ratio 0.01 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_25 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 0.05 --num_epochs 3 --batch_size 32 --mode train --dataset NSLKDD_26 --data_path dataset/NSLKDD --input_c 122 --output_c 122 && python main.py --anormly_ratio 0.05 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_26 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 0.05 --num_epochs 3 --batch_size 32 --mode train --dataset NSLKDD_27 --data_path dataset/NSLKDD --input_c 122 --output_c 122 && python main.py --anormly_ratio 0.05 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_27 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 0.01 --num_epochs 3 --batch_size 32 --mode train --dataset NSLKDD_28 --data_path dataset/NSLKDD --input_c 122 --output_c 122 && python main.py --anormly_ratio 0.01 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_28 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 0.01 --num_epochs 3 --batch_size 32 --mode train --dataset NSLKDD_29 --data_path dataset/NSLKDD --input_c 122 --output_c 122 && python main.py --anormly_ratio 0.01 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_29 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 0.01 --num_epochs 3 --batch_size 32 --mode train --dataset NSLKDD_30 --data_path dataset/NSLKDD --input_c 122 --output_c 122 && python main.py --anormly_ratio 0.01 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_30 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 0.01 --num_epochs 3 --batch_size 32 --mode train --dataset NSLKDD_31 --data_path dataset/NSLKDD --input_c 122 --output_c 122 && python main.py --anormly_ratio 0.01 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_31 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 0.01 --num_epochs 3 --batch_size 32 --mode train --dataset NSLKDD_32 --data_path dataset/NSLKDD --input_c 122 --output_c 122 && python main.py --anormly_ratio 0.01 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_32 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 0.01 --num_epochs 3 --batch_size 32 --mode train --dataset NSLKDD_33 --data_path dataset/NSLKDD --input_c 122 --output_c 122 && python main.py --anormly_ratio 0.01 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_33 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 0.01 --num_epochs 3 --batch_size 32 --mode train --dataset NSLKDD_34 --data_path dataset/NSLKDD --input_c 122 --output_c 122 && python main.py --anormly_ratio 0.01 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_34 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 0.01 --num_epochs 3 --batch_size 32 --mode train --dataset NSLKDD_35 --data_path dataset/NSLKDD --input_c 122 --output_c 122 && python main.py --anormly_ratio 0.01 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_35 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20
#python main.py --anormly_ratio 0.01 --num_epochs 3 --batch_size 32 --mode train --dataset NSLKDD_36 --data_path dataset/NSLKDD --input_c 122 --output_c 122 && python main.py --anormly_ratio 0.01 --num_epochs 10 --batch_size 32 --mode test --dataset NSLKDD_36 --data_path dataset/NSLKDD --input_c 122 --output_c 122 --pretrained_model 20