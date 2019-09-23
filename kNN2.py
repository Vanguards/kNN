from numpy import *
import operator
from numpy.ma import zeros, array
import matplotlib
import matplotlib.pyplot as plt

import kNN3

def file2matrix(fileName):
    # 数据格式如下
    # 40920   8.326976    0.953952    3
    # 14488   7.153469    1.673904    2
    file = open(fileName)
    # 读取行数
    arrayOLines = file.readlines()
    numberOfLines = len(arrayOLines)
    returnMatrix = ma.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        # 去掉回车
        line = line.strip()
        # 去掉TAB
        listFromLine = line.split('\t')
        returnMatrix[index, :] = listFromLine[0:3]
        # -1是指最后一个元素，不用int就会存成字符串而不是整型
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMatrix, classLabelVector





datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
fig = plt.figure()
# 三个1分别为子图行数列数，第几子图
ax = fig.add_subplot(111)
# 不加tolist好像也行
ax.scatter((datingDataMat[:,1]).tolist(),(datingDataMat[:,2]).tolist(),15.0*array(datingLabels),15.0*array(datingLabels))
plt.show()


normMat,ranges,minVals=kNN3.autoNorm(datingDataMat)
print(datingDataMat)
