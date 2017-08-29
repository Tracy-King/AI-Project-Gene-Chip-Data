#!/usr/bin/env python

import tensorflow as tf
import numpy as np



def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w):
    return tf.matmul(X, w) # notice we use the same model as linear regression, this is because there is a baked in cost function which performs softmax and cross entropy

def roundRobin(l, turn):
    for i in range(int(len(l)/turn)):
        l.append(l[i])
        del l[i]

if __name__ == '__main__':

    numTrain = 5896
    numTest = 5896
    iteration = 100
    rate = 0.05
    robin = 10
    batch = 100
    errorTrain = 0
    errorTest = 0
    trX = []    # 5894x112
    trY = []    # 5894x2
    teX = []
    teY = []

    organism = 0
    other = 0

    #train
    fy = open('../E-TABM-185.sdrf.txt','r')
    fx = open('../data_dimRed.txt','r')
    fo = open('output.txt', 'w')
    fy.readline()
    errorTrain = 0
    for i in range(0,5894):
        line_x = fx.readline().split('\t')[:-1]
        line_x = list(map(float,line_x))
        trX.append(line_x)
        if(i%300 == 0):
            print(len(line_x))
        line_y = fy.readline().split('\t')[1:2][0]

        if line_y == 'organism_part':
            organism = organism + 1
            line_y = [1,0]
        else :
            other = other + 1
            line_y = [0,1]
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

    for i in range(8):
        roundRobin(trX, robin)
        roundRobin(trY, robin)

    teX,teY = trX[0:int(len(trX)/robin)],trY[0:int(len(trY)/robin)]
    #print(len(trX),len(trY))
    #print(len(teX),len(teY))
    print(organism, other)
            
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

    fx.close()
    fy.close()
    fo.close()
