from numpy import *
import operator
from numpy.ma import zeros, array
import matplotlib
import matplotlib.pyplot as plt

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 1], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify(inX, dataSet, labels, k):
    """
    funct:使用k-邻近算法将每组数据划分到某个类中
    :param inX:
    :param dataSet:训练样本集
    :param labels:标签向量
    :param k:用于选择最近邻居的数量
    :return:
    """
    datasetSize = dataSet.shape[0]
    diffMat = tile(inX, (datasetSize, 1)) - dataSet  #把inX行重复datasetSize次，在列上重复1次
    sqDiffMat = diffMat ** 2                         #array是数组，这里是直接将所有数字平方，不是矩阵相乘
    sqDistances = sqDiffMat.sum(axis=1)              #axis=1行和，axis=0列和
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()         #从小到大排，但是输出索引
    classCount = {}
    for i in range(k):                               #i=0,1,2
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1   #这是个字典（键值对），dict.get(key, default=None),没找到key则返回default值
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)     #operator.itemgetter(1)根据第二个域进行排序，true降序
    return sortedClassCount[0][0]


group, labels = createDataSet()
print(classify([0, 0], group, labels, k=3))


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


def autoNorm(dataSet):
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    normDataSet=zeros(shape(dataSet))
    m=dataSet.shape[0]
    normDataSet=dataSet-tile(minVals,(m,1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet,ranges,minVals


def datingClassTest():
    hoRatio=0.10
    datingDataMat,datingDataLabels=file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    m=normMat.shape[0]
    numTestVecs=int(m*hoRatio)
    errorCount=0.0
    for i in range(numTestVecs):
        classifierResult=classify(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with {},the real answer is {}".format(classifierResult,datingLabels[i]))
        if classifierResult!=datingLabels[i]:
            errorCount+=1.0
    print("the total error rate is",errorCount/float(numTestVecs))

datingClassTest()