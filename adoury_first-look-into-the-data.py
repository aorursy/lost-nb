#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.




X_train = np.load('/kaggle/input/rcmemulators/X_train.dat', allow_pickle=True)
X_train.shape




for t,i in zip(['500 hPa', '700 hPa','850 hPa'],range(3)):
    plt.imshow(X_train[0,:,:,i])
    plt.colorbar()
    plt.title('Geopoential at '+t)
    plt.show()




for t,i in zip(['500 hPa', '700 hPa','850 hPa'],range(3,7)):
    plt.imshow(X_train[0,:,:,i])
    plt.colorbar()
    plt.title('Temperature at '+t)
    plt.show()




plt.imshow(X_train[0,:,:,15])
plt.colorbar()
plt.show()




temp_mpl_gcm=X_train[:,4,6,15]
plt.plot(temp_mpl_gcm[0:365])






def standardize1(data):
    mean =  data.mean(axis=(0), keepdims=True)
    sd   =  data.std(axis=(0), keepdims=True)
    ndata = (data - mean)/sd
    return (ndata)

def standardize2(data):
    maxx =  data.max(axis=(0), keepdims=True)
    minn   =  data.min(axis=(0), keepdims=True)
    ndata = (data - minn)/(maxx - minn)
    return (ndata)




X_test=np.load('/kaggle/input/rcmemulators/X_test.dat', allow_pickle=True)

Xtrain = standardize2(X_train)
Xtest  =  standardize2(X_test)




Y_train = pd.read_csv('/kaggle/input/rcmemulators/Y_train_mpl.csv')
Y_train.head()
Y_train_temp=np.asarray(Y_train.tempé)




plt.plot(Y_train_temp[0:365])




import seaborn as sn

sn.distplot(Y_train_temp)




def rmse(A,B):
    return(np.sqrt(((A-B)**2).mean(axis=(0), keepdims=True)))
rmse(Y_train_temp,temp_mpl_gcm)




import keras
import keras.models as km
import keras.layers as kl
import tensorflow as tf




model=km.Sequential()
model.add(kl.Flatten(input_shape=(11,11, 19)))
model.add(kl.Dense(256, activation='relu'))
model.add(kl.Dense(64, activation='linear'))
model.add(kl.Dense(32, activation='relu'))
model.add(kl.Dense(8, activation='linear'))
model.add(kl.Dense(1,activation='linear'))
model.compile(loss='mse', optimizer='adadelta')
model.summary()




Ytrain=np.asarray(Y_train.tempé) - temp_mpl_gcm ## I use the temperature series given by the low resolution model to standardize the data




yrs=[i for i in range(2006,2101)]
yrs2=np.repeat(yrs, 365 , axis=0)

yrs2=yrs2[np.where(yrs2%10>=3)[0]]
ech_train=np.where(yrs2%10!=6)[0]
ech_val=np.where(yrs2%10==6)[0]

X_train2 , X_val = Xtrain[ech_train,:,:,:],Xtrain[ech_val,:,:,:]    
Y_train2 , Y_val = Ytrain[ech_train],Ytrain[ech_val] 

print(X_train2.shape)





epochs = 15
batch_size = 10

history = model.fit(X_train2,Y_train2, batch_size=batch_size, validation_data=(X_val, Y_val), epochs=epochs)




previ = model.predict(Xtest)
pred = previ[:,0]+ X_test[:,4,6,15] ## we add again 
pred.shape




plt.plot(pred)




res = pd.read_csv('/kaggle/input/rcmemulators/samplesub.csv')

res.tempé = pred




res.to_csv('submission.csv', index=False)




Y_train_2d = np.load('/kaggle/input/rcmemulators/Y_train_box.dat', allow_pickle=True)
Y_train_2d.shape




plt.imshow(Y_train_2d[0,:,:])
plt.colorbar()
plt.show()




temp_2d_gcm=X_train[:,4:6,4:7,15]
plt.imshow(temp_2d_gcm[0,:,:])
plt.colorbar()




mean_box_gcm = temp_2d_gcm.mean(axis=(1,2),keepdims=True)
mean_box_gcm.shape




model2=km.Sequential()
model2.add(kl.Flatten(input_shape=(11,11, 19)))
model2.add(kl.Dense(512, activation='relu'))
model2.add(kl.Dense(256, activation='linear'))
model2.add(kl.Dense(64, activation='relu'))
model2.add(kl.Reshape((8, 8, 1)))
model2.add(kl.UpSampling2D((2,2)))
model2.add(kl.Conv2D(32, (5,5), activation='linear', padding='same'))
model2.add(kl.UpSampling2D((2,2)))
model2.add(kl.Conv2D(32, (3,3), activation='linear', padding='same'))
model2.add(kl.Conv2D(1, (3,3), activation='linear', padding='same'))
model2.compile(loss='mse', optimizer='adadelta')
model2.summary()




Y_train_2d_diff = Y_train_2d - mean_box_gcm ## We can try to predict the difference the region average in the GCM output
print(Y_train_2d_diff.shape)
Y_train2_2d , Y_val_2d = Y_train_2d_diff[ech_train,:,:,None],Y_train_2d_diff[ech_val,:,:,None] 




plt.imshow(Y_train_2d_diff[0,:,:])
plt.colorbar()




epochs = 5
batch_size = 10

history2 = model2.fit(X_train2,Y_train2_2d, batch_size=batch_size, validation_data=(X_val, Y_val_2d), epochs=epochs)




temp_2d_gcm_test=X_test[:,4:6,4:7,15]
mean_box_gcm_test = temp_2d_gcm_test.mean(axis=(1,2),keepdims=True)

pred_2d = model2.predict(Xtest) + mean_box_gcm_test[:,:,:,None]





predictions = pred_2d[:,:,:,0]
predictions.shape




plt.imshow(pred_2d[10,:,:,0],vmin=260)
plt.colorbar()
plt.show()




def create_submission(predictions):
    assert predictions.shape==(10220, 32, 32), f"Wrong shape for your prediction file : "                                      f"{predictions.shape} instead of (10220, 32, 32)" 
    
    if os.path.exists("submission.zip"):
        get_ipython().system('rm submission.zip')
    np.save("y_test", predictions)
    get_ipython().system('mv y_test.npy y_test.predict')
    get_ipython().system('zip -r submission.zip y_test_predict.zip')
    print ("Bundle submision.zip created !")
    return None

#
predictions = np.asarray(pred_2d[:,:,:,0])
create_submission(predictions) ## the output y_test.predict is created then you need to download it and zip it before submit it on codalab

