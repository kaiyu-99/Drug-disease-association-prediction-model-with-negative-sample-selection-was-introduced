# 根据结果绘制ROC曲线
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

# choose_color = ['#CD853F', '#A52A2A', '#FF1493']
# label_num=3
label_num=2
# ---------------------------------所有ROC曲线花在同一图上--------------------------

for p in range(label_num):
    # data = pd.read_csv(r'D:/drug_disease_project/multi_curve/IBK_1_1_' + str(p + 1) + '.csv',
    #     #     #                    names=['actual', 'predict', 'score'])
    data = pd.read_csv(r'D:/drug_disease_project/Lab2_detail_NS/1_2_1_3/IBK_13_1_' + str(p + 2) + '.csv',
                       names=['actual', 'predict', 'score'])
    # data = pd.read_csv(r'D:/drug_disease_project/evaluation/IBKBN_result_2and3/13_10_3_550+1000.csv', names=['actual', 'predict', 'score'])
    #循环打开那三个文件test
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

    all_TPR = []
    all_FPR = []
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
        all_TPR.append(TPR)
        all_FPR.append(FPR)
    all_TPR.append(0)
    all_FPR.append(0)


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

    x = np.array(all_FPR)
    y = np.array(all_TPR)
    fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred, pos_label=1)
    # print(metrics.auc(fpr, tpr))
    AUC=metrics.auc(fpr, tpr)
    # plt.plot([0, 1], [0, 1])
    # plt.plot(fpr,tpr,label="auc="+str(auc),color='darkorange')
    # plt.plot(fpr, tpr, label='T'+str(p+1)+'(AUC=%0.4f)' %AUC)
    plt.plot(fpr, tpr, label='T' + str(p + 2) + '(AUC=%0.4f)' % AUC)
    # plt.plot(x, y, label='T' + str(p + 1) + '(AUC=%0.4f)' % AUC)
plt.rcParams.update({'font.size': 15})
# plt.plot([0, 1], [0, 1])
plt.rc('font',family='Times New Roman')
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlim([0.0, 1.01])
plt.ylim([0.0, 1.01])
plt.xlabel('1-SP', fontsize=15)
plt.ylabel('SN', fontsize=15)
plt.legend(loc='lower right')  # 说明所在位置
# plt.title('1:1(kNN)_ROC')
plt.title('1:3(KNN)_ROC')
# plt.savefig('D:/drug_disease_project/evaluation/drawPic/savedPic/IBK_ROC_1_1.png',bbox_inches='tight')
plt.savefig('D:/drug_disease_project/Lab2_detail_NS/pic/IBK/IBK_ROC_1_3.png',bbox_inches='tight')
plt.show()
















