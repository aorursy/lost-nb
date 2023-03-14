#!/usr/bin/env python
# coding: utf-8



get_ipython().system('pip install hyperas')
get_ipython().system('pip install hyperopt')




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Conv1D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, GlobalAveragePooling1D
from keras.models import Model
from sklearn.model_selection import GroupKFold

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform




# Import data
X_train = pd.read_csv('../input/X_train.csv').iloc[:,3:].values.reshape(-1,128,10)
X_test  = pd.read_csv('../input/X_test.csv' ).iloc[:,3:].values.reshape(-1,128,10)
print('X_train shape:', X_train.shape, ', X_test shape:', X_test.shape)

dfy= pd.read_csv('../input/y_train.csv')
# Get groups for CV later
groups= dfy.iloc[:,1].values
Y_train=dfy.iloc[:,-1]
# Convert to one-hot for classes
num_classes = len(Y_train.unique())
Y_train = Y_train.replace(Y_train.unique(),range(num_classes))
Y_train = to_categorical(Y_train.values,num_classes)
print('Y_train shape:', Y_train.shape)




def model_FCN(input_shape=X_train.shape[1:], filters=1, kernel_size=1, s=1, units=num_classes):
    
    # Define the input placeholder as a tensor with shape input_shape.
    X_input = Input(input_shape)

    # Zero-Padding: none

    # CONV -> BN -> RELU Block
    X = Conv1D(filters, kernel_size, strides=s)(X_input)
    X = BatchNormalization(axis = 2)(X)
    X = Activation('relu')(X)
    
    # CONV -> BN -> RELU Block
    X = Conv1D(filters, kernel_size, strides=s)(X)
    X = BatchNormalization(axis = 2)(X)
    X = Activation('relu')(X)
    
    # CONV -> BN -> RELU Block
    X = Conv1D(filters, kernel_size, strides=s)(X)
    X = BatchNormalization(axis = 2)(X)
    X = Activation('relu')(X)

    # MAXPOOL - none
    
    # GAP
    X = GlobalAveragePooling1D()(X)
    
    # FLATTEN - none
    
    # FULLYCONNECTED
    X = Dense(units, activation='softmax',name='d0')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X)

    return model




def model_ResNet(input_shape=X_train.shape[1:], filters=1, kernel_size=1, s=1, units=num_classes):
    
    # Define the input placeholder as a tensor with shape input_shape.
    X_input = Input(input_shape)

    # Zero-Padding: none

    # CONV -> BN -> RELU Block
    X = Conv1D(filters, kernel_size, strides=s)(X_input)
    X = Add()([X,X_input])
    
    # CONV -> BN -> RELU Block
    X = Conv1D(filters, kernel_size, strides=s)(X)
    X = Add()([X,X_input])
    
    # CONV -> BN -> RELU Block
    X = Conv1D(filters, kernel_size, strides=s)(X)
    X = Add()([X,X_input])

    # MAXPOOL - none
    
    # GAP
    X = GlobalAveragePooling1D()(X)
    
    # FLATTEN - none
    
    # FULLYCONNECTED
    X = Dense(units, activation='softmax',name='d0')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X)

    return model




# Define model parameters
input_shape = X_train.shape[1:]
filters = 1
kernel_size = 1
s = 1
epochs = 1
batch_size = 32
folds=2
model_Name = "model_FCN"




# define 10-fold cross validation test harness
cvloss = []
cvloss_val = []
cvacc = []
cvacc_val = []
gkf = GroupKFold(n_splits=folds)

for train_idx,valid_idx in gkf.split(X_train,Y_train,groups=groups):
    # Create and compile the FCN model
    model = model_FCN(input_shape, filters, kernel_size,s,num_classes) if model_Name is "model_FCN" else                    model_ResNet(input_shape, filters, kernel_size,s,num_classes)
    model.compile(optimizer="Adam",loss="categorical_crossentropy",metrics=["accuracy"])   
    # Fit and evaluate the model
    history = model.fit(x=X_train[train_idx],y=Y_train[train_idx],epochs=epochs,                            validation_data=(X_train[valid_idx],Y_train[valid_idx]),shuffle=True,verbose=2)
    # Update score
    cvloss.append(history.history['loss'][-1])
    cvloss_val.append(history.history['val_loss'][-1])
    cvacc.append(history.history['acc'][-1])
    cvacc_val.append(history.history['val_acc'][-1])
    
    '''
    # Plot loss during training
    plt.subplot(121)
    plt.title('Loss in Fold '+str(f))
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='valid')
    plt.legend()
    
    # Plot accuracy during training
    plt.subplot(122)
    plt.title('Accuracy in Fold '+str(f))
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='valid')
    plt.legend()
    plt.show()   
    '''
    
print("Avg loss: ", np.mean(cvloss), "Avg acc: ", np.mean(cvacc))
print("Avg val_loss: ", np.mean(cvloss_val), "Avg val_acc: ", np.mean(cvacc_val))




def data():
    global X_train
    global Y_train   
    return X_train,Y_train,X_train, Y_train,




def create_model(x_train, y_train, x_test, y_test):

    model = model_FCN(filters={{choice([1,2,3,4,5])}})
    model.compile(optimizer="Adam",loss="categorical_crossentropy",metrics=["accuracy"])   
    result = model.fit(x=x_train,y=y_train,                        batch_size={{choice([64, 128])}},                        epochs={{choice([64, 128])}},                        validation_split=0.1,shuffle=True,verbose=2)

    #get the highest validation accuracy of the training epochs
    validation_acc = np.amax(result.history['val_acc']) 
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}




best_run, best_model = optim.minimize(model=create_model,
                                      data=data,
                                      algo=tpe.suggest,
                                      max_evals=5,
                                      trials=Trials(),
                                      notebook_name='2019CareerCon_Help_Navigate_Robot')
print("Evalutation of best performing model:")
print(best_model.evaluate(X_train, Y_train))
print("Best performing model chosen hyper-parameters:")
print(best_run)






