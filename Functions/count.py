#!/usr/bin/env python

import tensorflow as tf
import numpy as np

def setList(index, labelNum):
    l = []
    for i in range (labelNum):
        if i == index:
            l.append(1)
        else:
            l.append(0)
    return l

def findIndex(list, element):
    for i in list:
        if element == i:
            return list.index(i)
    return -1

if __name__ == '__main__':

    labelList = []
    labelNum = 194
    trX = []    # 5894x112
    trY = []    # 5894x2
    teX = []
    teY = []
    normal_cnt = 0
    disease_cnt = 0
    invalid_cnt = 0

    #train
    fy = open('../E-TABM-185.sdrf.txt','r')
    fx = open('../data_dimRed.txt','r')
    fy.readline()
    errorTrain = 0
    for i in range(0,5894):
        line_x = fx.readline().split('\t')[:-1]
        line_x = list(map(float,line_x))
        if(i%300 == 0):
            print(len(line_x))
        line_y = fy.readline().split('\t')[28:29][0]

        if line_y == '  ':
            invalid_cnt = invalid_cnt + 1
            continue
        else:
            if line_y == 'healthy' or line_y == 'normal':
                normal_cnt = normal_cnt + 1
            else:
                disease_cnt = disease_cnt + 1
            ret = findIndex(labelList, line_y)
            if ret == -1:
                labelList.append(line_y)
                line_y = setList(len(labelList)-1, labelNum)
            else:
                line_y = setList(ret, labelNum)
            
        trX.append(line_x)
        trY.append(line_y)
        if(i%300 == 0):
           print(line_y)

    #for i in range(0,590):
    #   line_x = fx.readline().split('\t')[:-1]
    #   line_x = list(map(float,line_x))
    #   teX.append(line_x)
    #    line_y = fy.readline().split('\t')[1:2][0]
    #    if line_y == 'organism_part':
    #        line_y = [1,0]
    #    else :
    #        line_y = [0,1]
    #    teY.append(line_y)

    teX,teY = trX,trY
    print(len(trX))
    print(len(trY))
    print(normal_cnt)
    print(disease_cnt)
    print(invalid_cnt)
    print(len(labelList))

    for i in range(10):
        print(labelList[i])
    
    fx.close()
    fy.close()
