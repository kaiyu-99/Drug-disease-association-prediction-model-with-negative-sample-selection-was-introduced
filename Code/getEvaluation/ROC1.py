# 根据结果绘制ROC曲线
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

# data = pd.read_csv(r'D:/drug_disease_project/evaluation/IBKBN_result_2and3/13_10_3_550+1000.csv', names=['actual', 'predict', 'score'])
data=pd.read_csv('D:/drug_disease_project/Lab2_detail_NS/1_1_1_2_BN/11_1_2_550+50.csv',names=['actual', 'predict', 'score'])
data_positive = data.iloc[:6349, :]
data_negative = data.iloc[6349:, :]

print(len(data_positive))
print(len(data_negative))
error = 0
for index in range(len(data)):
    if data.iloc[index, 0] == 1 and data.iloc[index, 1] == 2:
        error += 1
        data.iloc[index, 2] = 1.0 - data.iloc[index, 2]
    if data.iloc[index, 0] == 2 and data.iloc[index, 1] == 2:
        data.iloc[index, 2] = 1.0 - data.iloc[index, 2]
    if data.iloc[index, 0] == 2 and data.iloc[index, 1] == 1:
        error += 1

data_values = data.sort_values(by='score', ascending=False)
thresh = sorted(list(set(list(data_values['score']))))



def calculate_indicator(thresh, samples):
    all_TPR = []
    all_FPR = []
    all_P = []
    all_R = []
    count = 0

    for th in np.linspace(0, 1, 100):
        TP = FP = FN = TN = 0
        predict = np.where(data_positive['score'] >= th, 1, 2)
        for i in range(len(data_positive)):
            if data_positive.iloc[i, 0] == 1 and predict[i] == 1:
                TP += 1
            if data_positive.iloc[i, 0] == 1 and predict[i] == 2:
                FN += 1
        predict = np.where(data_negative['score'] >= th, 1, 2)
        for j in range(len(data_negative)):
            if data_negative.iloc[j, 0] == 2 and predict[j] == 2:
                TN += 1
            if data_negative.iloc[j, 0] == 2 and predict[j] == 1:
                FP += 1

        TPR = TP / (TP + FN)  # y
        FPR = FP / (TN + FP)  # x
        print(FPR, TPR)
        print()

        P = TP / (TP + FP)  # Precision y
        R = TP / (TP + FN)  # Recall x


        all_TPR.append(TPR)
        all_FPR.append(FPR)

        all_P.append(P)
        all_R.append(R)
    return all_FPR, all_TPR
    # return all_P, all_R

y_true = data_values['actual']

y_pred = data_values['score']


sum_AUC = 0
M = data_positive.shape[0]  # 正样本个数
N = data_negative.shape[0]  # 负样本个数
print("负样本个数：",N)


actual_class = list(data['actual'])

predict_score = list(data['score'])
x = 0
for j in range(len(actual_class)):
    if actual_class[j] == 1:
        for k in range(len(actual_class)):
            if actual_class[k] == 2:
                if predict_score[j] > predict_score[k]:
                    x += 1
                elif predict_score[j] == predict_score[k]:
                    x += 0.5
                else:
                    x += 0
AUC = round(x/(M*N), 4)
print(AUC)


fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred, pos_label=1)
print(metrics.auc(fpr, tpr))
AUC=metrics.auc(fpr, tpr)
# print(metrics.average_precision_score(y_true, y_pred, pos_label=1))
#
# plt.plot(fpr, tpr, label='AUC=' + str(metrics.auc(fpr, tpr)))
# plt.plot([0, 1], [0, 1])
# plt.plot(fpr,tpr,label="auc="+str(auc),color='darkorange')
plt.plot(fpr, tpr, label='ROC curve(AUC=%0.4f)' %AUC, color='darkorange')

# plt.plot([0, 1], [0, 1])
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('1-SP', fontsize=12)
plt.ylabel('SN', fontsize=12)
plt.legend(loc='lower right')  # 说明所在位置
# plt.show()
plt.savefig('D:/drug_disease_project/Lab2_detail_NS/pic/BN/BN_ROC_1_1_1_2_550+50.png',bbox_inches='tight')
