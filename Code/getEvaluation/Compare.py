import pandas as pd
#三个阈值的按precision降序找出每个阈值的最优按MCC排序tocsvagain里面其实就是最终结果
# df1=pd.read_csv('D:/drug_disease_projectevaluation\lab1_second.csv')
df1=pd.read_csv('D:/drug_disease_project/Lab2_detail_NS/1_1_1_3_IBK/IBK_11_lab1_third.csv')
# df2=pd.read_csv('D:/drug_disease_project/evaluation/second.csv')
# df3=pd.read_csv('D:/drug_disease_project/evaluation/third.csv')
df1.sort_values(by='MCC',inplace=True,ascending=False)
# df2.sort_values(by='MCC', inplace=True,ascending=False)
# df3.sort_values(by='MCC', inplace=True,ascending=False)
# print(df1)
# print(df2)
# print(df3)
df1.to_csv('D:/drug_disease_project/Lab2_detail_NS/1_1_1_3_IBK/IBK_11_lab1_third.MCC.csv',index=None)
# df2.to_csv('D:/drug_disease_project/evaluation/second_MCC.csv',index=None)
# df3.to_csv('D:/drug_disease_project/evaluation/third_MCC.csv',index=None)