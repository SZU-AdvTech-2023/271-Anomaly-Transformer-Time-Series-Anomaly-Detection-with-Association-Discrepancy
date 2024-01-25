from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

# gt;pred(before-fixed);pred(fixed);
msl_ptl = np.load('../prediction_test/MSL_pred_test_list.npy')
psm_ptl = np.load('../prediction_test/PSM_pred_test_list.npy')
smd_ptl = np.load('../prediction_test/SMD_pred_test_list.npy')
swat_ptl = np.load('../prediction_test/SWaT_pred_test_list_1.npy')
smap_ptl = np.load('../prediction_test/SMAP_pred_test_list.npy')

datas = [msl_ptl,psm_ptl,smd_ptl,swat_ptl,smap_ptl]
names = ['MSL','PSM','SMD','SWaT','SMAP']

for i in range(len(datas)):
    auc_score = roc_auc_score(datas[i][0], datas[i][2])
    print(names[i]+' AUC score='+str(auc_score))
    fpr, tpr, thresholds = roc_curve(datas[i][0], datas[i][2])
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}')  # 绘制ROC曲线，标注AUC的值
    # 随即分类器没有分类能力，其FPR=TPR。随机分类器的性能通常表示为ROC曲线上的对角线
    plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random Classifier')  # 绘制随机分类器的ROC曲线
    plt.xlabel('False Positive Rate')  # x轴标签为FPR
    plt.ylabel('True Positive Rate')  # y轴标签为TPR
    plt.title(names[i]+' ROC Curve')  # 设置标题
    plt.legend()
    plt.savefig('../pics/'+names[i]+'_roc_curve.png')
    plt.show()

# msl_auc_score = roc_auc_score(msl_ptl[0],msl_ptl[2]) #y_test, y_score
# psm_auc_score = roc_auc_score(psm_ptl[0],psm_ptl[2])
# smd_auc_score = roc_auc_score(smd_ptl[0],smd_ptl[2])
# swat_auc_score = roc_auc_score(swat_ptl[0],swat_ptl[2])
# smap_auc_score = roc_auc_score(smap_ptl[0],smap_ptl[2])
#
# fpr, tpr, thresholds = roc_curve(msl_ptl[0], msl_ptl[2])
#
# print(msl_auc_score)
# print(psm_auc_score)
# print(smd_auc_score)
# print(swat_auc_score)
# print(smap_auc_score)


