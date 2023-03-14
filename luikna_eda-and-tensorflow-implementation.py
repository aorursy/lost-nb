#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plot

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.




# Load feature vectors
df = pd.read_csv("../input/train.csv")
# Display some of the feature vectors
df




from sklearn.decomposition import PCA
pca = PCA(n_components=3)

# Obtain list of features
features = list(df)
del(features[:2]) # Delete id and species from features

# Apply PCA to our data
pca_result = pca.fit_transform(df[features].values)

# Define dataframe containing low-dimension representation of original data
pca_df = pd.DataFrame()
pca_df['species'] = df['species']
pca_df['pca-one'] = pca_result[:,0]
pca_df['pca-two'] = pca_result[:,1] 

# How much variance does this new representation retain from the original data?
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))




from ggplot import *
chart = ggplot( pca_df, aes(x='pca-one', y='pca-two', color='species') )         + geom_point(size=75,alpha=0.8)         + ggtitle("First and Second Principal Components colored by species")
chart




import time
from sklearn.manifold import TSNE

# Start t-SNE algorithm
time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=400)
tsne_results = tsne.fit_transform(df[features].values)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))




# Create dataframe for the new representation of the data
df_tsne = pd.DataFrame()
df_tsne['species'] = df['species']
df_tsne['x-tsne'] = tsne_results[:,0]
df_tsne['y-tsne'] = tsne_results[:,1]

# Create chart and display it
chart = ggplot( df_tsne, aes(x='x-tsne', y='y-tsne', color='species') )         + geom_point(size=70,alpha=0.1)         + ggtitle("tSNE dimensions colored by species")
chart




pca_50 = PCA(n_components=50)
pca_result_50 = pca_50.fit_transform(df[features].values)

print('Explained variation per principal component (PCA): {}'.format(np.sum(pca_50.explained_variance_ratio_)))




time_start = time.time()

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_pca_results = tsne.fit_transform(pca_result_50)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

# Create dataframe for the new representation of the data
df_tsne = pd.DataFrame()
df_tsne['species'] = df['species']
df_tsne['x-tsne'] = tsne_results[:,0]
df_tsne['y-tsne'] = tsne_results[:,1]

# Create chart and display it
chart = ggplot( df_tsne, aes(x='x-tsne', y='y-tsne', color='species') )         + geom_point(size=70,alpha=0.1)         + ggtitle("tSNE dimensions colored by species")
chart




# Import libraries
import pandas as pd
import numpy as np
import tensorflow as tf

# Parameters
learning_rate = 0.001
training_epochs = 200
batch_size = 100
display_step = 10

# Network Parameters
n_features = 192
n_classes = 99
neurons_layers = [n_features, 256, n_classes] # Number of features in input, hidden and output layers

# Load Train data
train_data = {}
df = pd.read_csv("../input/train.csv") # Load training set
tmp = df.as_matrix()
train_data['samples'] = tmp[:, 2:] # Obtain all the feature vectors (without ids and species)
train_data['labels'] = tmp[:, 1] # Obtain labels (i.e. species)
N_train = len(train_data['samples']) # Number of training samples

# Obtain list of features
features = list(df)
del(features[:2]) # Delete id and species from features

# Load Test data
test_data = {}
df = pd.read_csv("../input/test.csv")
tmp = df.as_matrix()
test_data['samples'] = tmp[:, 1:]
test_ids = df.pop('id')




# One hot encoding map something
enc = np.eye(n_classes)
sparse2dense = {i: enc[i] for i in range(n_classes)}

# Map class names to one-hot representation
class_names = np.unique(train_data['labels'])
class_encodings = {}
for i in range(n_classes):
    class_encodings[class_names[i]] = sparse2dense[i]




# tf Graph input
x = tf.placeholder(tf.float32, [None, n_features])
y = tf.placeholder(tf.float32, [None, n_classes])

# Model parameters (weights, biases)
weights = [
    tf.Variable(tf.random_normal([neurons_layers[k], neurons_layers[k + 1]])) for k in range(len(neurons_layers)-1)
]

biases = [
    tf.Variable(tf.random_normal([neurons_layers[k+1]])) for k in range(len(neurons_layers)-1)
]




# Function implementing a fully connected feed forward NN, with relu activations
def multilayer_perceptron(x, weights, biases):
    
    # Compute number of hidden layers
    n_hidden = len(weights)-1
    
    # Input layer to hidden layer
    out = tf.add(tf.matmul(x, weights[0]), biases[0])
    
    # Check that there are hidden layers
    if n_hidden > 0:
    
        # Iterate over all hidden layers
        for k in range(1, n_hidden+1):
            out = tf.nn.relu(out) # Apply activation fct on previous layer output
            out = tf.add(tf.matmul(out, weights[k]), biases[k]) # linear combination
               
    return out




# Construct model
y_pred = multilayer_perceptron(x, weights, biases)
p_pred = tf.nn.softmax(y_pred)

# Define cross entropy loss
cost = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y)
cost = tf.reduce_mean(cost)*batch_size

# Define optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluation of the model on a set
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))




# Create Validation and Training sets
tmp = pd.read_csv("../input/train.csv").as_matrix()

idx = np.random.choice(N_train, N_train, replace=False)
N_train2 = int(0.75*N_train)
train_data['samples'] = tmp[idx[:N_train2], 2:]
train_data['labels'] = np.array([class_encodings[t] for t in tmp[idx[:N_train2], 1]])

val_data = {}
val_data['samples'] = tmp[idx[N_train2:], 2:]
val_data['labels'] = np.array([class_encodings[t] for t in tmp[idx[N_train2:], 1]])




# Arrays containing cost for each epoch
cost_train = np.zeros([training_epochs,1])
cost_val = np.zeros([training_epochs,1])

# Initializing the variables
init = tf.global_variables_initializer()


# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(N_train2/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            idx = range(i*batch_size, (i+1)*batch_size)
            # Train samples
            batch_x = train_data['samples'][idx]
            # Train labels
            #print(idx)
            batch_y = train_data['labels'][idx]

            ## Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
            
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=",                 "{:.9f}".format(avg_cost))
            print("Accuracy on Validation nset:", accuracy.eval({x: val_data['samples']                                                                , y: val_data['labels']}))
        cost_train[epoch] = avg_cost
        cost_val[epoch] = cost.eval({x: val_data['samples'], y: val_data['labels']})
        
    print("Optimization Finished!")

    
    
    predictions = sess.run([p_pred], {x: test_data['samples']})




val_data['labels'].shape




import matplotlib.pyplot as plt

loss_train_curve, = plt.plot(cost_train, label='Training Set')
loss_val_curve, = plt.plot(cost_val, label='Validation Set')
plt.legend(handles=[loss_train_curve, loss_val_curve])




from sklearn.model_selection import KFold, cross_val_score

# Number of folds
n_folds = 4
k_fold = KFold(n_splits=n_folds, shuffle=True)




# Create Validation and Training sets
tmp = pd.read_csv("../input/train.csv").as_matrix()

train_data = {}
train_data['samples'] = tmp[:, 2:]
train_data['labels'] = np.array([class_encodings[t] for t in tmp[:, 1]])




# Arrays containing cost for each epoch
avg_cost = np.zeros([n_folds])
accuracies = np.zeros([n_folds])

# Initializing the variables
init = tf.global_variables_initializer()

# Number batches
total_batch = int((n_folds-1)*N_train/(n_folds*batch_size))

# Fold iteration counter
k = 0
training_set = {}
validation_set = {}

# Launch the graph
#with tf.Session() as sess:
#        sess.run(init)

    # Iterate over all folds
for train_indices, val_indices in k_fold.split(train_data['samples']):
    with tf.Session() as sess:
        sess.run(init)
        print(k+1,"/",n_folds, "folds")
        
        # Select validation and training sets
        training_set['samples'] = train_data['samples'][train_indices]
        training_set['labels'] = train_data['labels'][train_indices]
        validation_set['samples'] = train_data['samples'][val_indices]
        validation_set['labels'] = train_data['labels'][val_indices]
        
        # Training cycle
        for epoch in range(training_epochs):
            avg_cost[k] = 0
            # Loop over all batches
            for i in range(total_batch):
                # Obtain batches
                idx = range(i*batch_size, (i+1)*batch_size)
                batch_x = training_set['samples'][idx]
                batch_y = training_set['labels'][idx]

                ## Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
                # Compute average loss
                avg_cost[k] += c / total_batch
            
            # Display logs per epoch step
            if epoch % display_step == 0:
                acc = accuracy.eval({x: validation_set['samples'], y: validation_set['labels']})
                print("\t >> Epoch:", '%d' % (epoch+1),                       "|| Cost = {:.9f}".format(avg_cost[k]),                       "|| Accuracy on Validation Set = {:.9f}".format(acc)) 
        
        # Accuracy for this folding setting       
        accuracies[k] = accuracy.eval({x: validation_set['samples'],                                        y: validation_set['labels']})
        print("\t > Accuracy:", accuracies[k])
        k += 1
        
total_accuracy = np.mean(accuracies)
print("\n------------------------------------------")
print("Optimization Finished! Average accuracy:", total_accuracy)
print("\n------------------------------------------")




# Initializing the variables
init = tf.global_variables_initializer()

# Number batches
total_batch = int((n_folds-1)*N_train/(n_folds*batch_size))

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        # Loop over all batches
        for i in range(total_batch):
            # Obtain batches
            idx = range(i*batch_size, (i+1)*batch_size)
            batch_x = train_data['samples'][idx]
            batch_y = train_data['labels'][idx]

            ## Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
            
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("\t >> Epoch:", '%d' % (epoch+1),                   "|| Cost = {:.9f}".format(avg_cost)) 
        
    # Obtain predictions
    predictions = sess.run([p_pred], {x: test_data['samples']})




# prepare csv for submission
submission = pd.DataFrame(predictions[0], index=test_ids, columns=class_names)
submission.to_csv('submission.csv')




from PIL import Image
import glob
image_list = []
for filename in glob.glob('../input/images/*.jpg'): #assuming gif
    im=Image.open(filename)
    image_list.append(im)

