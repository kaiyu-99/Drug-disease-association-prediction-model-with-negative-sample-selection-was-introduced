import pandas as pd
import re
import os


def txtTocsv():
    # 提取测试集63472*2(正、负样本)随机森林算法分类的结果txt400*10*3（3个阈值），并转换为csv
    """
    :param fpath:
    :return: 返回一个.csv文件
    """
    # for p in range(1,11,1):
    for m in range(17, 18, 1):
        for n in range(50, 1050, 50):
                # data = pd.read_table('D:/drug_disease_project/Lab2_detail_NS/1_1_1_3_IBK/11_1_3_'+str(m)+'+'+str(n)+'.txt',header=None)
                data=pd.read_table('D:/drug_disease_project/CompareToOldEvalution/'+str(m)+'_400+1000_RF_new.txt',header=None)
                # dataframe数据类型转换为array

                data = data.values
                sample_list = []
                # qi zhi(shape[0]数组的行数)
                for i in range(2, data.shape[0]):
                    list_var = []
                    # 对字符串进行切片
                    data_tep = data[i][0].split()
                    if len(data_tep) == 4:
                        for j in range(1, len(data_tep)):
                            try:
                                item = data_tep[j].split(':')[0]
                            except:
                                item = data_tep[j]
                            list_var.append(item)
                    else:
                        for j in range(1, 3):
                            try:
                                item = data_tep[j].split(':')[0]
                            except:
                                item = data_tep[j]
                            list_var.append(item)
                        list_var.append(data_tep[-1])
                    sample_list.append(list_var)
                # print(sample_list)

                save_data = pd.DataFrame(sample_list)
                # save_data.to_csv('D:/drug_disease_project/evaluation/RF_result_2and3/13_'+str(p)+'_3_'+str(m)+'+'+str(n)+'.csv',
                #                  index=None, header=None)
                # save_data.to_csv('D:/drug_disease_project/Lab2_detail_NS/1_1_1_3_IBK/11_1_3_'+str(m)+'+'+str(n)+'.csv', index=None, header=None)
                save_data.to_csv('D:/drug_disease_project/CompareToOldEvalution/'+str(m)+'_400+1000_RF_new.csv',index=None, header=None)

txtTocsv()
