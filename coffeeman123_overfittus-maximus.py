#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.




cd '/kaggle/input/dont-overfit-ii/'




df = pd.read_csv('train.csv')
y = df['target']
x = df.drop(['target','id'],axis=1)
df.head(5)




from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1)
print(x_train.shape,x_test.shape)




# Let's first try with a very simple linear model.
from keras import models, layers
net_input = layers.Input((300,))
output = layers.Dense(1,activation='sigmoid')(net_input)
model = models.Model(net_input, output)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10,validation_data=(x_test,y_test))




# Alright, it seems to work. However, it's overfitting extremely heavily. Let's do some feature analysis to see which
# features affect the output the most.
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

bestfeatures = SelectKBest(score_func=f_regression, k=10)
fit = bestfeatures.fit(x.values,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Feature','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))  #print 10 best features




import matplotlib.pyplot as plt
top10 = featureScores.nlargest(10,'Score')
plt.bar(top10['Feature'],top10['Score'])




# Seems as if features 33 and 65 are the most important ones. Let's single those out:
values = (33,65)
x_train, x_test, y_train, y_test = train_test_split(x.values[:,tuple(values)],y,test_size=0.1)
print(x_train.shape,x_test.shape)
losses = []
for i in range(5):
    net_input = layers.Input((len(values),))
    output = layers.Dense(1,activation='sigmoid')(net_input)
    model = models.Model(net_input, output)
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
    history = model.fit(x_train, y_train, epochs=10,validation_data=(x_test,y_test),verbose=0)
    losses.extend(history.history['val_loss'][-1:])
    print("Attempt",i+1,':',history.history['val_loss'][-1:])
print("Average loss: ",sum(losses)/len(losses))

# Try playing around with the values a bit - see which combination gives you the lowest loss.




# It's pretty clear that only the five most important features help us.
# Let's see if we can enhance our model.
from keras import optimizers
from keras import callbacks
values = (33,65,217,117,91)

x_train, y_train = x.values[:,tuple(values)],y
opt = optimizers.Adam(lr=0.03)
net_input = layers.Input((len(values),))
output = layers.Dense(1,activation='sigmoid')(net_input)
model = models.Model(net_input, output)
callback = callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=2,
                              verbose=0, mode='auto',
                              restore_best_weights=True)
model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['acc'])
history = model.fit(x_train, y_train, epochs=20,validation_split=0.1,callbacks=[callback])




from keras import optimizers
x_train, y_train = x.values[:,tuple(values)],y
opt = optimizers.Adam(lr=0.03)
net_input = layers.Input((len(values),))
output = layers.Dense(1,activation='sigmoid')(net_input)
model = models.Model(net_input, output)
callback = callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=2,
                              verbose=0, mode='auto',
                              restore_best_weights=True)
model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['acc'])
history = model.fit(x_train, y_train, epochs=8)




x_val = pd.read_csv('test.csv')
x_val.head()




values = (33,65,217,117,91)
x = x_val.drop(['id'],axis=1)
ids = x_val['id']
predictions = model.predict(x.values[:,tuple(values)])




submission = ['id,target\n']
for index,prediction in enumerate(np.around(predictions)[:,0].astype(np.int)):
    txt = str(ids[index])+','+str(prediction)
    if ids[index] != 19999: txt += '\n'
    submission.append(txt)




cd ../../working




with open('submission.csv','w+') as writer:
    writer.writelines(submission)




submission = pd.read_csv('submission.csv')
submission.head(10)

