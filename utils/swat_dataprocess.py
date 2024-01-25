import numpy as np
import pandas as pd

swat_train_pd = pd.read_csv('../dataset/SWaT/SWaT_Dataset_Normal_v1.csv')
swat_test_pd = pd.read_csv('../dataset/SWaT/SWaT_Dataset_Attack_v0.csv')

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

# the column 'Normal/Attack' contains special value 'A ttack' with a space in it
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

np.save('../dataset/SWaT/swat_test_label.npy', swat_test_label_np)
np.save('../dataset/SWaT/swat_train.npy', swat_train_np)
np.save('../dataset/SWaT/swat_test.npy', swat_test_np)


