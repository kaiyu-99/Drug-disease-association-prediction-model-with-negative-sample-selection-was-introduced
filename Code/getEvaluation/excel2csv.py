#30个指标的结果文件excel转化为csv文件，方便后面取指标平均值
import pandas as pd
# for j in range(12,14):
#   for i in range(1,11,1):
#
#     data=pd.read_excel('D:/drug_disease_project/evaluation/toexcel/'+str(j)+'_lab'+str(i)+'_first.xls','各个指标值')
#     print(data)
#     data.to_csv('D:/drug_disease_project/evaluation/tocsvagain/'+str(j)+'_lab'+str(i)+'_first.csv',index=None)
data=pd.read_excel('D:/drug_disease_project/CompareToOldEvalution/BN_lab1_800+950.xls','各个指标值')
print(data)
data.to_csv('D:/drug_disease_project/CompareToOldEvalution/BN_lab1_800+950.csv',index=None)