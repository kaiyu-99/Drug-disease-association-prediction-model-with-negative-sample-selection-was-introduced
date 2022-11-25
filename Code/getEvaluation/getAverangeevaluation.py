# 每个阈值（0，0-0.05 0-0.1 闭区间）10个csv文件400个维度对应的指标分别取平均值
import pandas as pd
from _pydecimal import Decimal, Context, ROUND_HALF_UP


# 增加第一列ndim_name
ndim_name=pd.read_csv('/RandomNS/ndim_name.csv')
print(ndim_name)
matrix = []
for y in range(400): #400
    matrix.append([])
# data1 = pd.read_csv('D:/drug_disease_project/evaluation/lab1_third.csv')
data1 = pd.read_csv('D:/drug_disease_project/evaluation/tocsvagain/13_lab1_first.csv')
data2 = pd.read_csv('D:/drug_disease_project/evaluation/tocsvagain/13_lab2_first.csv')
data3 = pd.read_csv('D:/drug_disease_project/evaluation/tocsvagain/13_lab3_first.csv')
data4 = pd.read_csv('D:/drug_disease_project/evaluation/tocsvagain/13_lab4_first.csv')
data5 = pd.read_csv('D:/drug_disease_project/evaluation/tocsvagain/13_lab5_first.csv')
data6 = pd.read_csv('D:/drug_disease_project/evaluation/tocsvagain/13_lab6_first.csv')
data7 = pd.read_csv('D:/drug_disease_project/evaluation/tocsvagain/13_lab7_first.csv')
data8 = pd.read_csv('D:/drug_disease_project/evaluation/tocsvagain/13_lab8_first.csv')
data9 = pd.read_csv('D:/drug_disease_project/evaluation/tocsvagain/13_lab9_first.csv')
data10 = pd.read_csv('D:/drug_disease_project/evaluation/tocsvagain/13_lab10_first.csv')
print(data1)
for i in range(400):#400
    ACC = (data1.iat[i, 1] + data2.iat[i, 1] + data3.iat[i, 1] + data4.iat[i, 1] + data5.iat[i, 1]
           + data6.iat[i, 1] + data7.iat[i, 1] + data8.iat[i, 1] + data9.iat[i, 1] + data10.iat[i, 1]) / 10
    print(ACC)
    SN = (data1.iat[i, 2] + data2.iat[i, 2] + data3.iat[i, 2] + data4.iat[i, 2] + data5.iat[i, 2]
          + data6.iat[i, 2] + data7.iat[i, 2] + data8.iat[i, 2] + data9.iat[i, 2] + data10.iat[i, 2]) / 10
    SP = (data1.iat[i, 3] + data2.iat[i, 3] + data3.iat[i, 3] + data4.iat[i, 3] + data5.iat[i, 3]
          + data6.iat[i, 3] + data7.iat[i, 3] + data8.iat[i, 3] + data9.iat[i, 3] + data10.iat[i, 3]) / 10
    Precision = (data1.iat[i, 4] + data2.iat[i, 4] + data3.iat[i, 4] + data4.iat[i, 4] + data5.iat[i, 4]
                 + data6.iat[i, 4] + data7.iat[i, 4] + data8.iat[i, 4] + data9.iat[i, 4] + data10.iat[i, 4]) / 10
    F1_score = (data1.iat[i, 5] + data2.iat[i, 5] + data3.iat[i, 5] + data4.iat[i, 5] + data5.iat[i, 5]
                + data6.iat[i, 5] + data7.iat[i, 5] + data8.iat[i, 5] + data9.iat[i, 5] + data10.iat[i, 5]) / 10
    MCC = (data1.iat[i, 6] + data2.iat[i, 6] + data3.iat[i, 6] + data4.iat[i, 6] + data5.iat[i, 6]
           + data6.iat[i, 6] + data7.iat[i, 6] + data8.iat[i, 6] + data9.iat[i, 6] + data10.iat[i, 6]) / 10

    ACC = Context(prec=3, rounding=ROUND_HALF_UP).create_decimal(ACC)
    # print(ACC)
    SN = Context(prec=3, rounding=ROUND_HALF_UP).create_decimal(SN)
    SP = Context(prec=3, rounding=ROUND_HALF_UP).create_decimal(SP)
    Precision = Context(prec=3, rounding=ROUND_HALF_UP).create_decimal(Precision)
    F1_score = Context(prec=3, rounding=ROUND_HALF_UP).create_decimal(F1_score)
    MCC = Context(prec=3, rounding=ROUND_HALF_UP).create_decimal(MCC)
    ACC = float(str(Decimal(ACC).quantize(Decimal('0.000'))))
    SN = float(str(Decimal(SN).quantize(Decimal('0.000'))))
    SP = float(str(Decimal(SP).quantize(Decimal('0.000'))))
    Precision = float(str(Decimal(Precision).quantize(Decimal('0.000'))))
    F1_score = float(str(Decimal(F1_score).quantize(Decimal('0.000'))))
    MCC = float(str(Decimal(MCC).quantize(Decimal('0.000'))))
    # matrix[i].append(str(i))
    matrix[i].append(ACC)
    matrix[i].append(SN)
    matrix[i].append(SP)
    matrix[i].append(Precision)
    matrix[i].append(F1_score)
    matrix[i].append(MCC)

print(matrix)
save_data = pd.DataFrame(matrix)
save_data.columns=['ACC','SN','SP','Precision','F1_score','MCC']
save_data.insert(0,'ndim_name',ndim_name)
# save_data.insert(0,'ndim_name','13_400+1000')
save_data.to_csv('D:/drug_disease_project/evaluation/13_first.csv',
                             index=None)

# print(data1.iat[1,1])
# print(data1.iat[2,1])
