from os import listdir
from kNN1 import classify
from numpy import *
import operator

def img2vector(filename):
    returnVect=zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])
    return returnVect

print(img2vector('testDigits/0_13.txt')[0,32:63])

def handwritingClassTest():
    hwlabels=[]
    trainingFileList=listdir('trainingDigits')
    m=len(trainingFileList)
    trainingMat=zeros((m,1024))
    for i in range(m):
        fileNameStr=trainingFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        hwlabels.append(classNumStr)
        trainingMat[i,:]=img2vector('trainingDigits/{}'.format(fileNameStr))
        # trainingMat[i, :] = img2vector('trainingDigits/%s'%fileNameStr)
    testFileList=listdir('testDigits')
    erroCount=0.0
    mTest=len(testFileList)
    for i in range(mTest):
        fileNameStr=testFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        vectorUnderTest=img2vector('testDigits/{}'.format(fileNameStr))
        classifierResult=classify(vectorUnderTest,trainingMat,hwlabels,3)
        print("the classifier came back with: %d, the real answer is: %d"%(classifierResult,classNumStr))
        if(classifierResult!=classNumStr):
            erroCount+=1.0
    print("the total number of errors is: %d"%erroCount)
    print("the total error rate is: %f" % (erroCount/float(mTest)))


handwritingClassTest()