#!/usr/bin/env python
# coding: utf-8



import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import img_to_array, load_img

# A large amount of the data loading code is based on najeebkhan's kernel
# Check it out at https://www.kaggle.com/najeebkhan/leaf-classification/neural-network-through-keras
root = '../input'
np.random.seed(2016)
split_random_state = 7
split = .9


def load_numeric_training(standardize=True):
    """
    Loads the pre-extracted features for the training data
    and returns a tuple of the image ids, the data, and the labels
    """
    # Read data from the CSV file
    data = pd.read_csv(os.path.join(root, 'train.csv'))
    ID = data.pop('id')

    # Since the labels are textual, so we encode them categorically
    y = data.pop('species')
    y = LabelEncoder().fit(y).transform(y)
    # standardize the data by setting the mean to 0 and std to 1
    X = StandardScaler().fit(data).transform(data) if standardize else data.values

    return ID, X, y


def load_numeric_test(standardize=True):
    """
    Loads the pre-extracted features for the test data
    and returns a tuple of the image ids, the data
    """
    test = pd.read_csv(os.path.join(root, 'test.csv'))
    ID = test.pop('id')
    # standardize the data by setting the mean to 0 and std to 1
    test = StandardScaler().fit(test).transform(test) if standardize else test.values
    return ID, test


def resize_img(img, max_dim=96):
    """
    Resize the image to so the maximum side is of size max_dim
    Returns a new image of the right size
    """
    # Get the axis with the larger dimension
    max_ax = max((0, 1), key=lambda i: img.size[i])
    # Scale both axes so the image's largest dimension is max_dim
    scale = max_dim / float(img.size[max_ax])
    return img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))


def load_image_data(ids, max_dim=96, center=True):
    """
    Takes as input an array of image ids and loads the images as numpy
    arrays with the images resized so the longest side is max-dim length.
    If center is True, then will place the image in the center of
    the output array, otherwise it will be placed at the top-left corner.
    """
    # Initialize the output array
    # NOTE: Theano users comment line below and
    X = np.empty((len(ids), max_dim, max_dim, 1))
    # X = np.empty((len(ids), 1, max_dim, max_dim)) # uncomment this
    for i, idee in enumerate(ids):
        # Turn the image into an array
        x = resize_img(load_img(os.path.join(root, 'images', str(idee) + '.jpg'), grayscale=True), max_dim=max_dim)
        x = img_to_array(x)
        # Get the corners of the bounding box for the image
        # NOTE: Theano users comment the two lines below and
        length = x.shape[0]
        width = x.shape[1]
        # length = x.shape[1] # uncomment this
        # width = x.shape[2] # uncomment this
        if center:
            h1 = int((max_dim - length) / 2)
            h2 = h1 + length
            w1 = int((max_dim - width) / 2)
            w2 = w1 + width
        else:
            h1, w1 = 0, 0
            h2, w2 = (length, width)
        # Insert into image matrix
        # NOTE: Theano users comment line below and
        X[i, h1:h2, w1:w2, 0:1] = x
        # X[i, 0:1, h1:h2, w1:w2] = x  # uncomment this
    # Scale the array values so they are between 0 and 1
    return np.around(X / 255.0)


def load_train_data(split=split, random_state=None):
    """
    Loads the pre-extracted feature and image training data and
    splits them into training and cross-validation.
    Returns one tuple for the training data and one for the validation
    data. Each tuple is in the order pre-extracted features, images,
    and labels.
    """
    # Load the pre-extracted features
    ID, X_num_tr, y = load_numeric_training()
    # Load the image data
    X_img_tr = load_image_data(ID)
    # Split them into validation and cross-validation
    sss = StratifiedShuffleSplit(n_splits=1, train_size=split, random_state=random_state)
    train_ind, test_ind = next(sss.split(X_num_tr, y))
    X_num_val, X_img_val, y_val = X_num_tr[test_ind], X_img_tr[test_ind], y[test_ind]
    X_num_tr, X_img_tr, y_tr = X_num_tr[train_ind], X_img_tr[train_ind], y[train_ind]
    return (X_num_tr, X_img_tr, y_tr), (X_num_val, X_img_val, y_val)


def load_test_data():
    """
    Loads the pre-extracted feature and image test data.
    Returns a tuple in the order ids, pre-extracted features,
    and images.
    """
    # Load the pre-extracted features
    ID, X_num_te = load_numeric_test()
    # Load the image data
    X_img_te = load_image_data(ID)
    return ID, X_num_te, X_img_te

print('Loading the training data...')
(X_num_tr, X_img_tr, y_tr), (X_num_val, X_img_val, y_val) = load_train_data(random_state=split_random_state)
y_tr_cat = to_categorical(y_tr)
y_val_cat = to_categorical(y_val)
print('Training data loaded!')









import pandas as pd
import numpy as np
import scipy as scipy
import tensorflow as tf




def iterate_minibatches(inputs, inputs_img, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], inputs_img[excerpt], targets[excerpt]

def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    weights = weight_variable([input_dim, output_dim])
    biases = bias_variable([output_dim])
    preactivate = tf.matmul(input_tensor, weights) + biases
    activations = act(preactivate, name='activation')
    return activations

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')




df = pd.read_csv('../input/train.csv')




num_classes = 99




train_df = df
print(train_df.shape)
train_df.head()




print(train_df.species.value_counts(normalize=True))




species = train_df['species'].values




species.shape




species[1]




print(Y_train.shape)




X_train = train_df.values




X_train = scipy.delete(X_train, 0, 1) # Remove first and second columns
X_train = scipy.delete(X_train, 0, 1) 




print(X_train.shape)




X_train, X_val = X_train[:-200], X_train[-200:]
X_train, X_test = X_train[:-100], X_train[-100:]




Y_train, Y_val = Y_train[:-200], Y_train[-200:]
Y_train, Y_test = Y_train[:-100], Y_train[-100:]




sess = tf.InteractiveSession()




## Create the model
num_classes = 99
x_feature = tf.placeholder(tf.float32, [None, 192])
x_img = tf.placeholder(tf.float32, [None, 96, 96, 1])
y_ = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

net = tf.reshape(x_img, [-1, 96*96*1])
net = nn_layer(net, 96*96*1, 1024, 'layer1')
net = nn_layer(net, 1024, 512, 'layer2')
net = nn_layer(net, 512, 256, 'layer3')
net = tf.concat(1, [net, x_feature])
net = tf.nn.dropout(net, keep_prob)
net = nn_layer(net, 256+192, 512, 'layer4')
net = tf.nn.dropout(net, keep_prob)
net = nn_layer(net, 512, 1024, 'layer5')
y = nn_layer(net, 1024, num_classes, 'layer6', act=tf.identity)




cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.AdamOptimizer(
    learning_rate=1e-4, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')\
.minimize(cross_entropy)

prediction = tf.nn.softmax(y)




tf.global_variables_initializer().run()




min_loss = 100
saver = tf.train.Saver()
save_path = ''




#X_num_tr, X_img_tr, y_tr, X_num_val, X_img_val, y_val


for i in range(50):
    if i % 5 == 0:
        print(i)
    for batch in iterate_minibatches(X_num_tr, X_img_tr, y_tr_cat, 50, shuffle=True):
        batch_xs, batch_xs_img, batch_ys = batch
        sess.run(train_step, feed_dict={x_img: batch_xs_img, x_feature: batch_xs, y_: batch_ys, keep_prob: 0.5})

        # Test trained model
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("tr:{}".format(sess.run(accuracy, feed_dict={x_img: X_img_tr, 
                                                       x_feature: X_num_tr, y_: y_tr_cat, keep_prob: 1.0})))
    print("val:{}".format(sess.run(accuracy, feed_dict={x_img: X_img_val, 
                                                        x_feature: X_num_val, y_: y_val_cat, keep_prob: 1.0})))
    cur_loss = sess.run(cross_entropy, feed_dict={x_img: X_img_val, 
                                                        x_feature: X_num_val, y_: y_val_cat, keep_prob: 1.0})
    if cur_loss < min_loss:
        min_loss = cur_loss
        save_path = saver.save(sess, 'my-model')
        print("!!!NEW MIN LOSS {}. Saved at {}".format(cur_loss, save_path))

    else: 
        print("val_los:{}".format(cur_loss))

print("val:{}".format(sess.run(accuracy, feed_dict={x_img: X_img_val, 
                                                        x_feature: X_num_val, y_: y_val_cat, keep_prob: 1.0})))
    




with tf.Session() as sess2:
    # Initialize variables
    #sess.run(init)
    tf.global_variables_initializer().run()

    # Restore model weights from previously saved model
    saver.restore(sess2, save_path)
    print("Model restored from file: %s" % save_path)

    loss = sess2.run(cross_entropy, feed_dict={x_img: X_img_val, 
                                                        x_feature: X_num_val, y_: y_val_cat, keep_prob: 1.0})
    
    print("val:{}".format(loss))
    
    
    LABELS = sorted(pd.read_csv(os.path.join(root, 'train.csv')).species.unique())


    index, test, X_img_te = load_test_data()
    yPred_proba = sess2.run(prediction, feed_dict={x_img: X_img_te, 
                                                      x_feature: test, keep_prob: 1.0})

    # Converting the test predictions in a dataframe as depicted by sample submission
    yPred = pd.DataFrame(yPred_proba,index=index,columns=LABELS)

    print('Creating and writing submission...')
    #fp = open('submit.csv', 'w')
    #fp.write(yPred.to_csv())
    #print('Finished writing submission')
    # Display the submission
    yPred.tail()




loss = sess.run(cross_entropy, feed_dict={x_img: X_img_val, 
                                                        x_feature: X_num_val, y_: y_val_cat, keep_prob: 1.0})
print("val:{}".format(loss))
    




LABELS = sorted(pd.read_csv(os.path.join(root, 'train.csv')).species.unique())


index, test, X_img_te = load_test_data()
yPred_proba = sess.run(prediction, feed_dict={x_img: X_img_te, 
                                                  x_feature: test, keep_prob: 1.0})

# Converting the test predictions in a dataframe as depicted by sample submission
yPred = pd.DataFrame(yPred_proba,index=index,columns=LABELS)

print('Creating and writing submission...')
fp = open('submit.csv', 'w')
fp.write(yPred.to_csv())
print('Finished writing submission')
# Display the submission




fp = open('submit.csv', 'w')
fp.write(yPred.to_csv())




yPred.head()






