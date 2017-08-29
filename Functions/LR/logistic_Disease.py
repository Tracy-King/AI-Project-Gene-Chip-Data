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

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w):
    return tf.matmul(X, w) # notice we use the same model as linear regression, this is because there is a baked in cost function which performs softmax and cross entropy

def roundRobin(l, turn):
    for i in range(int(len(l)/turn)):
        l.append(l[i])
        del l[i]

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w):
    return tf.matmul(X, w) # notice we use the same model as linear regression, this is because there is a baked in cost function which performs softmax and cross entropy


if __name__ == '__main__':

    numTrain = 5896
    numTest = 5896
    iteration = 100
    batch = 100
    robin = 10
    rate = 0.05
    errorTrain = 0
    errorTest = 0
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
    fo = open('output.txt', 'w')
    fy.readline()
    errorTrain = 0
    for i in range(0,5894):
        line_x = fx.readline().split('\t')[:-1]
        line_x = list(map(float,line_x))
        #if(i%300 == 0):
            #print(len(line_x))
        line_y = fy.readline().split('\t')[28:29][0]

        if line_y == 'normal' or line_y == 'healthy':
            line_y = [1,0]
            normal_cnt = normal_cnt + 1
        elif line_y == '  ':
            invalid_cnt = invalid_cnt + 1
            continue
        else:
            line_y = [0,1]
            disease_cnt = disease_cnt + 1
            
        trX.append(line_x)
        trY.append(line_y)
        #if(i%300 == 0):
           #print(line_y)

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

    for i in range(9):
        roundRobin(trX, robin)
        roundRobin(trY, robin)

    teX,teY = trX[0:int(len(trX)/robin)],trY[0:int(len(trY)/robin)]
    print(normal_cnt)
    print(disease_cnt)
    print(invalid_cnt)
    
    fx.close()
    fy.close()
            
    X = tf.placeholder("float", [None, 112]) # create symbolic variables
    Y = tf.placeholder("float", [None, 2])

    w = init_weights([112, 2]) # like in linear regression, we need a shared variable weight matrix for logistic regression

    py_x = model(X, w)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)) # compute mean cross entropy (softmax is applied internally)
    train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct optimizer
    predict_op = tf.argmax(py_x, 1) # at predict time, evaluate the argmax of the logistic regression

    # Launch the graph in a session
    with tf.Session() as sess:
        with tf.device("/gpu:0"):
        # you need to initialize all variables
            tf.global_variables_initializer().run()

            for i in range(iteration):
                for start, end in zip(range(int(len(trX)/robin), len(trX), batch), range(int(len(trX)/robin) + batch, len(trX)+1, batch)):
                    sess.run(train_op, feed_dict={X: trX[start:end],Y: trY[start:end]})
                accuracy = np.mean(np.argmax(teY, axis=1) ==
                             sess.run(predict_op, feed_dict={X: teX}))
                fo.write(str(accuracy) + '\n')
                print(i, accuracy)

    fo.close()

