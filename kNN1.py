from numpy import *
import operator


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
