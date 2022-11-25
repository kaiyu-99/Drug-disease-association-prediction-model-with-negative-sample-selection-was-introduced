import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from _pydecimal import Decimal, Context, ROUND_HALF_UP
import xlwt

#求出混淆矩阵以及计算性能指标，存入excel中
matrix = []
# for y in range(400):
for y in range(1):
    matrix.append([])
print(matrix)
k = 0

for i in range(17, 18, 1):
    for j in range(1000, 1050, 50):
            # save_data = pd.read_csv('D:/drug_disease_project/evaluation/IBKBN_result_2and3/11_10_400+1000.csv', header=None)
            # save_data = pd.read_csv(
            #     'D:/drug_disease_project/Lab2_detail_NS/1_2_1_3/1_3_1_3_'+str(i)+'+'+str(j)+'.csv', header=None)
            save_data = pd.read_csv('D:/drug_disease_project/CompareToOldEvalution/'+str(i)+'_400+1000_RF_new.csv',
            header=None)
            y_true = save_data[0].values
            print(y_true)
            y_pred = save_data[1].values
            print(y_pred)
            # 求出混淆矩阵 正样本1负样本2
            C2 = confusion_matrix(y_true, y_pred, labels=[1, 2])
            print(C2)
            # 统计TP TN FP FN
            tp, fn, fp, tn = C2.ravel()
            acc = accuracy_score(y_true, y_pred)
            print("ACC:", acc)

            # FN FP
            def calc(TP, FN, FP, TN):
                SN = TP / (TP + FN)  # recall
                SP = TN / (TN + FP)
                precision = TP / (TP + FP)
                # ACC = (TP + TN) / (TP + TN + FN + FP)
                f1_score = 2 * SN * precision / (SN + precision)
                fz = TP * TN - FP * FN
                fm = (TP + FN) * (TP + FP) * (TN + FP) * (TN + FN)
                MCC = fz / pow(fm, 0.5)
                return SN, SP, precision, f1_score, MCC



            SN, SP, Precision, F1_score, MCC = calc(tp, fn, fp, tn)
            ACC = Context(prec=3, rounding=ROUND_HALF_UP).create_decimal(acc)
            print(ACC)
            SN = Context(prec=3, rounding=ROUND_HALF_UP).create_decimal(SN)
            SP = Context(prec=3, rounding=ROUND_HALF_UP).create_decimal(SP)
            print(SP)
            Precision = Context(prec=3, rounding=ROUND_HALF_UP).create_decimal(Precision)
            F1_score = Context(prec=3, rounding=ROUND_HALF_UP).create_decimal(F1_score)
            MCC = Context(prec=7, rounding=ROUND_HALF_UP).create_decimal(MCC)
            ACC = float(str(Decimal(ACC).quantize(Decimal('0.000'))))
            # print(ACC)
            # print(type(ACC))
            SN= float(str(Decimal(SN).quantize(Decimal('0.000'))))
            SP = float(str(Decimal(SP).quantize(Decimal('0.000'))))
            Precision = float(str(Decimal(Precision).quantize(Decimal('0.000'))))
            F1_score = float(str(Decimal(F1_score).quantize(Decimal('0.000'))))
            MCC = float(str(Decimal(MCC).quantize(Decimal('0.000'))))
            # 添加矩阵的第k行（列表嵌套列表）
            # matrix[k].append('1_3_1_3_' + str(i) +'+' +str(j))
            matrix[k].append('lab1_'+str(i)+'_400+1000_RF_new')
            # matrix[k].append('13_3_'+str(i)+'+'+str(j))
            matrix[k].append(SN)
            matrix[k].append(SP)
            matrix[k].append(ACC)
            matrix[k].append(Precision)
            matrix[k].append(F1_score)
            matrix[k].append(MCC)
            k += 1

print(matrix)
    #写入excel中
book = xlwt.Workbook(encoding='utf-8', style_compression=0)
sheet = book.add_sheet('各个指标值', cell_overwrite_ok=True)
col = ('ndim', 'SN', 'SP','ACC', 'Precision', 'F1_score', 'MCC')
for k in range(0, 7):
    sheet.write(0, k, col[k])
    #     写入每一行
for m in range(0, 1): #（0，1）
# for m in range(0,400):
    data = matrix[m]
    print(matrix[m])
        #对应每一列
    for n in range(0, 7):
        sheet.write(m+1, n, data[n])
    # savepath = 'D:/drug_disease_project/evaluation/toexcel/lab10_third.xls'
# savepath='D:/drug_disease_project/evaluation/toexcel/IBK_11_lab10_third.xls'
# savepath='D:/drug_disease_project/Lab2_detail_NS/1_1_1_3_IBK/IBK_11_lab1_third.xls'
# savepath='D:/drug_disease_project/CompareToOldEvalution/BN_lab1_800+950.xls'

# book.save(savepath)
