#!/usr/bin/env python
# coding: utf-8



# !pip install bayesian-optimization --user

import pandas as pd , numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from bayes_opt import BayesianOptimization
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTEENN
from sklearn.metrics import balanced_accuracy_score
from collections import Counter
#from MulticoreTSNE import MulticoreTSNE as TSNE
from IPython.display import clear_output
import collections, os
import pickle





df = pd.read_csv("../input/train.csv")
df.shape
test_df = pd.read_csv("../input/test.csv")




test_df = test_df.drop("id", axis = 1)
label = df["target"]
Input = df.drop(columns=['id', 'target'])




test_df = test_df.values.astype(np.float32)

x_data = Input.values.astype(np.float32)
y_datalabel = label
y_data = LabelEncoder().fit_transform(label)

onehot = np.zeros((y_data.shape[0], np.unique(y_data).shape[0]))
for i in range(y_data.shape[0]):
    onehot[i, y_data[i]] = 1.0
x_train, x_test, y_train, y_test, y_train_label, y_test_label = train_test_split(x_data, onehot, y_data, test_size = 0.3)




print(x_train.shape)
print(sorted(Counter(y_train_label).items()))




def focal_loss_sigmoid(labels,logits,alpha=0.25 , gamma=2):
    y_pred=tf.nn.sigmoid(logits)
    labels=tf.to_float(labels)
    L=-labels*(1-alpha)*((1-y_pred)*gamma)*tf.log(  tf.maximum(y_pred , 1e-14 )   )-      (1-labels)*alpha*(y_pred**gamma)*tf.log( tf.maximum( 1-y_pred ,  1e-14 ) ) 
    return L

def spectral_norm(w, iteration= 2 , name = None):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])
    
    u = tf.get_variable(name , [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
       
        """
       power iteration
       Usually iteration = 1 will be enough
       """
        
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm 




l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.005, scope=None)

def neural_network(num_hidden, size_layer, learning_rate , dropout_rate , beta ,
                   activation , focal_weight , reduction_node , batch_size = 20 ,
                   x_train = x_train , y_train = y_train  ):
    
    def activate(activation, first_layer, second_layer, bias):
        if activation == 0:
            activation = tf.nn.leaky_relu
#         elif activation == 1:
#             activation = tf.nn.tanh
        elif activation == 1:
            activation = tf.nn.relu
        elif activation == 2:
            activation = tf.nn.elu
        elif activation == 3:
            activation = tf.nn.relu6
        else :
            activation = tf.nn.selu
            
        layer = activation(tf.matmul(first_layer, second_layer) + bias)
        return tf.contrib.nn.alpha_dropout(layer, dropout_rate)

    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, (None, x_data.shape[1]))
    Y = tf.placeholder(tf.float32, (None, onehot.shape[1]))
    ## W = tf.Variable(tf.contrib.layers.xavier_initializer()((n_prev, n)))
    
    
    
    input_layer = tf.Variable(tf.contrib.layers.xavier_initializer()((x_data.shape[1], size_layer)))
    biased_layer = tf.Variable(tf.random_normal([size_layer], stddev = 0.1))
    output_layer = tf.Variable(tf.contrib.layers.xavier_initializer()((size_layer - reduction_node * (num_hidden - 1), onehot.shape[1])))
    #biased_output = tf.Variable(tf.random_normal([onehot.shape[1]], stddev = 0.1))
    
    
    layers, biased = [], []
    
#     for i in range(num_hidden - 1):
#         layers.append(tf.Variable(tf.contrib.layers.xavier_initializer()((size_layer, size_layer))))
#         biased.append(tf.Variable(tf.random_normal([size_layer])))
     
    for i in range(num_hidden - 1):
        size_layer2 = size_layer - reduction_node 
        layers.append( spectral_norm(tf.Variable(tf.contrib.layers.xavier_initializer()((size_layer, size_layer2))) , name = "SN" + str(i)) )        
        biased.append(tf.Variable(tf.random_normal([size_layer2])))
        size_layer = size_layer2
    
    
    
    first_l = activate(activation, X, input_layer, biased_layer)
    next_l = activate(activation, first_l, layers[0], biased[0])
    
    for i in range(1, num_hidden - 1):
        next_l = activate(activation, next_l, layers[i], biased[i])
    
    last_l = tf.matmul(next_l, output_layer) # + biased_output
    cost2 = tf.reduce_sum( focal_loss_sigmoid(logits = last_l, labels = Y , alpha = focal_weight ))
    beta_Factor = 0.99
    cost2 = (1- beta_Factor) / (1- beta_Factor**batch_size) * cost2
    # tf.nn.softmax_cross_entropy_with_logits_v2(logits = last_l, labels = Y)
      
    regularizers = tf.nn.l2_loss(input_layer) +  sum(map(lambda x: tf.nn.l2_loss(x), layers)) + tf.nn.l2_loss(output_layer)
    #+ 
    #regularizers = tf.contrib.layers.l1_l2_regularizer(0.5, 0.5)(input_layer) + \
    #sum(map(lambda x: tf.contrib.layers.l1_l2_regularizer(0.5, 0.5)(x), layers)) 
    
    #tf.contrib.layers.l1_l2_regularizer(1.0, 1.0)(input_layer) + 
    
    
    cost = cost2
    #+ beta * regularizers
    
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    last_l = tf.nn.sigmoid(last_l)
    last_l = tf.argmax( last_l, 1)
    last_y = tf.argmax(Y, 1)
    #correct_prediction = tf.equal(, )
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
    #sess = tf.InteractiveSession()
    #sess.run(tf.global_variables_initializer())
    
    config=tf.ConfigProto( log_device_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config = config)
    sess.run(tf.global_variables_initializer())
    
    
    COST, TEST_COST, ACC, TEST_ACC = [], [], [], []
    
    for i in range(100):
        train_acc, train_loss = 0, 0
        train_dt = np.concatenate((x_train , y_train) , axis = 1)
        np.random.shuffle(train_dt)
        x_train = train_dt[:,:-2]
        y_train = train_dt[:,-2:]
        for n in range(0, (x_train.shape[0] // batch_size) * batch_size, batch_size):
            _, loss , loss2 , regu2 = sess.run([optimizer, cost , cost2 , regularizers ], 
                                                 feed_dict = {X: x_train[n: n + batch_size, :], Y: y_train[n: n + batch_size, :]})
            TRUE , PRED = sess.run([last_l , last_y], feed_dict = {X: x_train[n: n + batch_size, :], Y: y_train[n: n + batch_size, :]})
            train_acc += balanced_accuracy_score(TRUE , PRED)
            train_loss += loss
            
        
        if i % 100 == 0 :
            print("Epoch : {} , Train Loss : {} , Regularizer : {} , Total Loss : {} ".format(i ,loss2 , regu2 , loss))

        
        train_loss /= (x_train.shape[0] // batch_size)
        train_acc /= (x_train.shape[0] // batch_size)
        ACC.append(train_acc)
        COST.append(train_loss)
    ## test는 학습 다하고 딱 한번만 하는 것이 맞지 않을까? 
    
    TEST_COST.append(sess.run(cost, feed_dict = {X: x_test, Y: y_test}))
    TRUE , PRED = sess.run([last_l , last_y], feed_dict = {X: x_test, Y: y_test})
    TEST_ACC.append(balanced_accuracy_score(TRUE , PRED))

    clear_output(wait=True)
    COST = np.array(COST).mean()
    ACC = np.array(ACC).mean()
    
    TEST_COST = np.array(TEST_COST).mean()
    TEST_ACC = np.array(TEST_ACC).mean()
    
    Test_TRUE = sess.run([last_l], feed_dict = { X: test_df })
    test_pred_n = collections.Counter(Test_TRUE[0])
    
    if TEST_ACC > 0.8 : 
        TEST_ACC = np.round(TEST_ACC , 3)
        with open('test_predict_{}.pkl'.format(TEST_ACC), 'wb') as f:
            pickle.dump(Test_TRUE, f)
    
    return COST, TEST_COST, ACC, TEST_ACC , test_pred_n




def generate_nn(num_hidden, size_layer, learning_rate, dropout_rate, beta, activation , focal_weight , reduction_node ):
    global accbest
    param = {
        'num_hidden' : int(np.around(num_hidden)),
        'size_layer' : int(np.around(size_layer)),
        'learning_rate' : max(min(learning_rate, 1), 0.0001),
        'dropout_rate' : max(min(dropout_rate, 0.7), 0.2),
        'beta' : max(min(beta, 0.5), 0.000001),
        'activation': int(np.around(activation)) , 
        "focal_weight" : max( min(focal_weight , 0.5) , 0.01) ,
        "reduction_node" : min( int(reduction_node) , int(np.around(size_layer) / np.around(num_hidden) )- 10  ) ,
    }
    print("\n Search parameters \n %s" % (param), file = log_file)
    
    log_file.flush()
    learning_cost, valid_cost, learning_acc, valid_acc , test_pred_n = neural_network(**param)
    print("stop after 5000 iteration with train cost %f, test cost %f, train acc %f, test acc %f" % (learning_cost, valid_cost, learning_acc, valid_acc))
    
    f = open("nn-bayesian_acc.txt",'a')
    result_ = "stop after 5000 iteration with train cost {:.3f}, test cost {:.3f}, train acc {:.3f}, test acc {:.3f} , True Test: {} \n".format(learning_cost, valid_cost, learning_acc, valid_acc , test_pred_n)
    f.write(result_)
    if (valid_acc > accbest):
        costbest = valid_acc
    return valid_acc




log_file = open('nn-bayesian.log', 'a')
accbest = 0.8
NN_BAYESIAN = BayesianOptimization(generate_nn, 
                              {'num_hidden': (3, 6),
                               'size_layer': (df.shape[1]-30 , df.shape[1] + 10 ),
                               'learning_rate': (0.001, 0.0001),
                               'dropout_rate': (0.2, 0.8),
                               'beta': (0.01, 0.001),
                               'activation': (0, 4),
                               "focal_weight" : (0.2 , 0.6),
                               "reduction_node" : (50 , 90 )
                              })

#NN_BAYESIAN.maximize(init_points = 10 , n_iter = 1 , acq="ucb", kappa= 10.0)
















