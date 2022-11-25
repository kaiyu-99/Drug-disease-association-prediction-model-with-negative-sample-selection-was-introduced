#draw pr curve
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt


# label_num=3
label_num=2
# ---------------------------------所有PR曲线花在同一图上--------------------------
for p in range(label_num):
    # data = pd.read_csv(r'D:/drug_disease_project/multi_curve/BN_1_3_' + str(p + 1) + '.csv',
    #                    names=['actual', 'predict', 'score'])
    data = pd.read_csv(r'D:/drug_disease_project/Lab2_detail_NS/1_2_1_3/IBK_13_1_' + str(p + 2)+'.csv',
                       names=['actual', 'predict', 'score'])

# data=pd.read_csv('D:/drug_disease_project/RandomNS/50+600.csv', names=['actual', 'predict', 'score'])
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



    all_P = [0]
    all_R = [1]
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
        P = TP / (TP + FP)  # Precision y
        R = TP / (TP + FN)  # Recall x
        print(P,R)
        all_P.append(P)
        all_R.append(R)
    all_R.append(0)
    all_P.append(1)



    y_true = data_values['actual']
    y_pred = data_values['score']

    sum_AUC = 0
    M = data_positive.shape[0]  # 正样本个数
    N = data_negative.shape[0]  # 负样本个数
    print("负样本个数：", N)

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
    AUC = round(x / (M * N), 4)
    print(AUC)

    precision,recall, threshold = metrics.precision_recall_curve(y_true, y_pred, pos_label=1)
    #集合所有的点
    x_r = np.array(all_R)
    y_p = np.array(all_P)
    #求面积
    #根据预测分数计算平均精度 (AP)。
    #AP 将precision-recall 曲线总结为在每个阈值处实现的精度的加权平均值，将前一个阈值的召回率增加用作权重：
    AUPR=metrics.average_precision_score(y_true, y_pred, pos_label=1)

    #画线
    plt.plot(x_r, y_p, label='T' + str(p + 2) + '(AUPR=%0.4f)' % AUPR)
    # plt.plot(recall, precision, label='T' + str(p + 1) + '(AUPR=%0.4f)' % AUPR)
plt.rcParams.update({'font.size': 15})
# plt.plot([0, 1], [0, 1])
plt.rc('font',family='Times New Roman')
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
# plt.plot([0, 1], [0, 1])
plt.xlim([0.0,1.01])
plt.ylim([0.0,1.01])
plt.xlabel('Recall',fontsize=15)
plt.ylabel('Precision',fontsize=15)
plt.legend(loc='lower left')  # 说明所在位置
plt.title('1:3(KNN)_PR')
# plt.show()
plt.savefig('D:/drug_disease_project/Lab2_detail_NS/pic/IBK/IBK_PR_1_3.png',bbox_inches='tight')
plt.show()
