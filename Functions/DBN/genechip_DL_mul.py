"""
Deep Belief Network
author: Ye Hu
2016/12/20
"""
import timeit
import numpy as np
import tensorflow as tf
import input_data
from logisticRegression import LogisticRegression
from mlp import HiddenLayer
from rbm import RBM

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

def roundRobin(l, turn):
    for i in range(int(len(l)/turn)):
        l.append(l[i])
        del l[i]

accuList = []
costList = []

class DBN(object):
    """
    An implement of deep belief network
    The hidden layers are firstly pretrained by RBM, then DBN is treated as a normal
    MLP by adding a output layer.
    """
    def __init__(self, n_in=784, n_out=10, hidden_layers_sizes=[500, 500]):
        """
        :param n_in: int, the dimension of input
        :param n_out: int, the dimension of output
        :param hidden_layers_sizes: list or tuple, the hidden layer sizes
        """
        # Number of layers
        assert len(hidden_layers_sizes) > 0
        self.n_layers = len(hidden_layers_sizes)
        self.layers = []    # normal sigmoid layer
        self.rbm_layers = []   # RBM layer
        self.params = []       # keep track of params for training

        # Define the input and output
        self.x = tf.placeholder(tf.float32, shape=[None, n_in])
        self.y = tf.placeholder(tf.float32, shape=[None, n_out])

        # Contruct the layers of DBN
        for i in range(self.n_layers):
            if i == 0:
                layer_input = self.x
                input_size = n_in
            else:
                layer_input = self.layers[i-1].output
                input_size = hidden_layers_sizes[i-1]
            # Sigmoid layer
            sigmoid_layer = HiddenLayer(inpt=layer_input, n_in=input_size, n_out=hidden_layers_sizes[i],
                                    activation=tf.nn.sigmoid)
            self.layers.append(sigmoid_layer)
            # Add the parameters for finetuning
            self.params.extend(sigmoid_layer.params)
            # Create the RBM layer
            self.rbm_layers.append(RBM(inpt=layer_input, n_visiable=input_size, n_hidden=hidden_layers_sizes[i],
                                        W=sigmoid_layer.W, hbias=sigmoid_layer.b))
        # We use the LogisticRegression layer as the output layer
        self.output_layer = LogisticRegression(inpt=self.layers[-1].output, n_in=hidden_layers_sizes[-1],
                                                n_out=n_out)
        self.params.extend(self.output_layer.params)
        # The finetuning cost
        self.cost = self.output_layer.cost(self.y)
        # The accuracy
        self.accuracy = self.output_layer.accuarcy(self.y)
    
    def pretrain(self, sess, X_train, batch_size=50, pretraining_epochs=10, lr=0.1, k=1, 
                    display_step=1):
        """
        Pretrain the layers (just train the RBM layers)
        :param sess: tf.Session
        :param X_train: the input of the train set (You might modidy this function if you do not use the desgined mnist)
        :param batch_size: int
        :param lr: float
        :param k: int, use CD-k
        :param pretraining_epoch: int
        :param display_step: int
        """
        print('Starting pretraining...\n')
        start_time = timeit.default_timer()
        batch_num = int(len(X_train) / batch_size)
        print("batch_num;", batch_num)
        
        # Pretrain layer by layer
        for i in range(self.n_layers):
            l2_loss = tf.nn.l2_loss(self.rbm_layers[i].W)
            cost = self.rbm_layers[i].get_reconstruction_cost()
            if(i == 0):
                cost = cost + 0.01*l2_loss
            train_ops = self.rbm_layers[i].get_train_ops(learning_rate=lr, k=k, persistent=None)
            for epoch in range(pretraining_epochs):
                avg_cost = 0.0
                for j in range(batch_num):
                    x_batch = X_train[j*batch_size:(j+1)*batch_size]
                    # 训练
                    sess.run(train_ops, feed_dict={self.x: x_batch})
                    # 计算cost
                    avg_cost += sess.run(cost, feed_dict={self.x: x_batch,}) / batch_num
                    #print(avg_cost)
                # 输出
                if epoch % display_step == 0:
                    print("\tPretraing layer {0} Epoch {1} cost: {2}".format(i, epoch, avg_cost))

        end_time = timeit.default_timer()
        print("\nThe pretraining process ran for {0} minutes".format((end_time - start_time) / 60))
    
    def finetuning(self, sess, trX, trY, teX, teY, training_epochs=10, batch_size=100, lr=0.1,
                   display_step=1):
        """
        Finetuing the network
        """
        print("\nStart finetuning...\n")
        start_time = timeit.default_timer()
        train_op = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(
            self.cost, var_list=self.params)
        for epoch in range(training_epochs):
            avg_cost = 0.0
            batch_num = int(len(trX) / batch_size)
            for i in range(batch_num):
                x_batch, y_batch = trX[i*batch_size:(i+1)*batch_size], trY[i*batch_size:(i+1)*batch_size]
                # 训练
                sess.run(train_op, feed_dict={self.x: x_batch, self.y: y_batch})
                # 计算cost
                avg_cost += sess.run(self.cost, feed_dict=
                {self.x: x_batch, self.y: y_batch}) / batch_num
            # 输出
            if epoch % display_step == 0:
                val_acc = sess.run(self.accuracy, feed_dict={self.x: teX,
                                                       self.y: teY})
                print("\tEpoch {0} cost: {1}, validation accuacy: {2}".format(epoch, avg_cost, val_acc))
                accuList.append(val_acc)
                costList.append(avg_cost)

        end_time = timeit.default_timer()
        print("\nThe finetuning process ran for {0} minutes".format((end_time - start_time) / 60))

if __name__ == "__main__":
    labelList = []
    labelNum = 194
    numTrain = 5896
    numTest = 5896
    iteration = 20
    robin = 10
    batch = 100
    trX = []    # 5894x112
    trY = []    # 5894x2
    teX = []
    teY = []
    normal_cnt = 0
    disease_cnt = 0
    invalid_cnt = 0

    #train
    fy = open('../../E-TABM-185.sdrf.txt','r')
    fx = open('../../microarray.regular.txt','r')
    fo = open('output.txt','w')
    fy.readline()
    errorTrain = 0
    for i in range(0,5893):
        line_x = fx.readline().split('\t')[:-1]
        line_x = list(map(float,line_x))
        #if(i % 59 == 0):
         #   print(i)
         
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

        trY.append(line_y)  
        trX.append(line_x)

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

    for i in range(0):
        roundRobin(trX, robin)
        roundRobin(trY, robin)

    print(normal_cnt)
    print(disease_cnt)
    print(invalid_cnt)
    print(len(trY))
    #print(trX[-1])
    #teX, teY = trX, trY
    teX = trX[0:round(len(trX)/robin)]
    teY = trY[0:round(len(trY)/robin)]
    #teX, teY = trX, trY
    fx.close()
    fy.close()
    
    # mnist examples 1895 112 22283

    dbn = DBN(n_in=1895, n_out=labelNum, hidden_layers_sizes=[200, 200, 200])
    sess = tf.Session()
    with tf.device("/gpu:0"):
        init = tf.global_variables_initializer()
        sess.run(init)
        # set random_seed
        tf.set_random_seed(seed=1111)
        dbn.pretrain(sess, X_train=trX)
        #for i in range(iteration):
        t = round(len(trX)/robin)
        dbn.finetuning(sess, trX[round(len(trX)/robin):], trY[round(len(trX)/robin):], trX, trY)

    for item in accuList:
        fo.write(str(item) + '\n')

    fo.write('\n')
    for item in costList:
        fo.write(str(item) + '\n')
    fo.close()
