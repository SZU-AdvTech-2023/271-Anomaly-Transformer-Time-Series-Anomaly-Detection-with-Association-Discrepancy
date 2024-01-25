import numpy as np

d1 = np.load('../prediction_test/NSLKDD_0_pred_test_list.npy')
d2 = np.load('../dataset/NSLKDD/nsl_kdd_test_label_encoded.npy')
d3 = np.load('../prediction_test/NSLKDD_pred_test_list_1.npy')
d4 = np.load('../dataset/NSLKDD/nsl_kdd_test_label_encoded_ovr.npy')

# print(np.count_nonzero(d2 == 0 ))
# print(np.count_nonzero(d2 == 1 ))
# print(np.count_nonzero(d2 == 1 ) / len(d2))
#
# d3
# # print(d1)
# print(np.count_nonzero(d3[0] == 0 )) # ground truth normal
# print(np.count_nonzero(d3[0] == 1 )) # ground truth abnormal
# print(np.count_nonzero(d3[1] == 1 )) # prediction abnormal before adjustment
# print(np.count_nonzero(d3[2] == 1 )) # prediction abnormal after adjustment


# anomaly ratio
for labels in d4:
    print(np.count_nonzero(labels == 1)/len(labels) *100.0)