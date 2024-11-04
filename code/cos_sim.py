# 求特征矩阵的余弦相似度
from numpy import *
from numpy import ndarray
import csv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


A: ndarray = zeros((4122, 64), dtype=float)  # 先创建一个全零方阵A，并且数据的类型设置为float浮点型
f = open('data/4122-kmers3.csv', "r")
lines = f.readlines()  # 把全部数据文件读到一个列表lines中
A_row = 0  # 表示矩阵的行，从0行开始
for line in lines:  # 把lines中的数据逐行读取出来
    list = line.strip('\n').split(',')  # 处理逐行数据：strip表示把头尾的'\n'去掉，split表示以空格来分割行数据，然后把处理后的行数据返回到list列表中
    #print('%d:%d' % (A_row, len(list)))
    A[A_row:] = list[0:64]  # 把处理后的数据放到方阵A中。list[0:4]表示列表的0,1,2,3列数据放到矩阵A中的A_row行
    A_row += 1  # 然后方阵A的下一行接着读
    print(A)

m1 = np.mat(A)
m1_array = np.asarray(m1)
# m1_similarity = cosine_similarity(m1)
m1_similarity = cosine_similarity(m1_array)
print(m1_similarity)

with open("data/4122-V-cos3.csv", 'w', newline='', encoding='utf-8') as f:
    csv_wirter = csv.writer(f)
    csv_wirter.writerows(m1_similarity)
