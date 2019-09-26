from os import listdir

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

handwritingClassTest()