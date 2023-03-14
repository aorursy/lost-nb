#!/usr/bin/env python
# coding: utf-8



get_ipython().run_line_magic('matplotlib', 'inline')




from google.colab import files
files.upload()




get_ipython().system('pip install fastai==0.7.0')
from fastai.imports import*
from fastai.structured import *
from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from sklearn import metrics




import pandas as pd




df = pd.read_csv("train(2).csv")




df = df.drop(["ID","comp_name","website","founded_on","hq_country_code","hq_state_code","hq_region","first_funding_date","last_funding_date"],axis=1)




df.head()




import matplotlib.pyplot as plt 
import seaborn as sns




plt.figure(figsize=(10,7))
ax = sns.countplot(x="op_status", data=df)




plt.figure(figsize=(10,7))
ax = sns.countplot(x="successful_investment", data=df)




domains = df["domain"]




domains = domains.values
domains = domains.astype(str)




domains




domains.reshape((1,len(domains)))
domains = list(domains)





type(domains[0])




unique=[]




for x in domains:
  y= x.split("|")
  for z in y:
    if z not in unique:
      unique.append(z)
    else:
      continue

        




unique




dictOfWords = { i : unique[i] for i in range(0, len(unique) ) }





dictOfWords




b = dict([(value,key) for (key,value) in dictOfWords.items()])




b




df = df.values
df[:,1]=df[:,1].astype(str)




multi_label = np.zeros((len(df[:,0]),len(unique)))




for i in range(len(df[:,0])):
  x = df[i,1]
  l = x.split("|")
  for y in l:
    multi_label[i,b[y]]=1
  




multi_label[0]




df




df = np.delete(df, 1, axis=1)




df = np.hstack((multi_label,df))





new = pd.DataFrame.from_records(df)




new = new.drop([841,842],axis=1)




new




status = new[837].values




status.shape




from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()




status = labelencoder_X.fit_transform(status)









import keras




from keras.utils import to_categorical




status_o = to_categorical(status)




status_o.shape




type(status_o)




new1=new.drop([837],axis=1)
new1.shape




new1 = new1.values




print(status_o.shape)
new1.shape




new1 = np.hstack((status_o,new1))




new = pd.DataFrame.from_records(new1)




new = new[]




new[842] = new[842].fillna(new[842].mean())




new




new = new.drop([841],axis=1)




new[843] = new[843].fillna(new[843].mean())




for i in range(4):
  new[844+i] = new[844+i].fillna(new[844+i].mean())




new




new.to_csv("final.csv")




get_ipython().system('ls')




y = new.values[:,-1]
y




X=new.values[:,0:-1]




from keras.utils import to_categorical
y_o = to_categorical(y)




X=(X-X.mean())/X.std()




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_o, test_size=0.1, random_state=42)




import tensorflow as tf
import keras




from tensorflow.keras.layers import Input, Dense

input_layer = Input(shape = X.shape[1] )
hidden_layer = Dense(512, activation = 'relu',)(input_layer)
hidden_layer1 = Dense(512, activation = 'tanh',)(hidden_layer)
hidden_layer2 = Dense(512, activation = 'relu',)(hidden_layer1)
hidden_layer3 = Dense(512, activation = 'tanh',)(hidden_layer2)
output_layer = Dense(2, activation = 'softmax')(hidden_layer1)




from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
model = Model(inputs=[input_layer], outputs=[output_layer])
model.compile(optimizer=optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()




history = model.fit(X_train, y_train, validation_split = 0.1, epochs=100,batch_size=256)





from fastai.imports import *
from fastai.structured import *
from sklearn.ensemble import RandomForestClassifier
from IPython.display import display
from sklearn import metrics




new




m = RandomForestRegressor(n_jobs=-1)
m.fit(new.drop([848], axis=1), new[848])




m.score(new.drop([848], axis=1), new[848])




from google.colab import files
files.upload()




def preprocess(df,b):
    df = df.drop(["ID","comp_name","website","founded_on","hq_country_code","hq_state_code","hq_region"],axis=1)
    ''' domains = df["domain"]
    domains = domains.values
    domains = domains.astype(str)
    domains.reshape((1,len(domains)))
    domains = list(domains)
    unique=[]
    for x in domains:
        y= x.split("|")
        for z in y:
            if z not in unique:
              unique.append(z)
            else:
              continue
    dictOfWords = { i : unique[i] for i in range(0, len(unique) ) }
    b = dict([(value,key) for (key,value) in dictOfWords.items()])'''
    print(df.values.shape)
    df = df.values
    df[:,1]=df[:,1].astype(str)
    multi_label = np.zeros((len(df[:,0]),len(b)))
    print(multi_label.shape)
    for i in range(len(df[:,0])):
        x = df[i,1]
        l = x.split("|")
        for y in l:
          if i in b.keys():
            multi_label[i,b[y]]=1
    print(df.shape)
    df = np.delete(df, 1, axis=1)
    df = np.hstack((multi_label,df))
    print(df.shape)
    new = pd.DataFrame.from_records(df)
    new
    new = new.drop([841,842],axis=1)
    status = new[837].values
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder_X = LabelEncoder()
    status = labelencoder_X.fit_transform(status)
    import keras
    from keras.utils import to_categorical
    status_o = to_categorical(status)
    print(new[837])
    new1=new.drop([837],axis=1)
    new1 = new1.values
    new1 = np.hstack((status_o,new1))
    new = pd.DataFrame.from_records(new1)
    new[842] = new[842].fillna(new[842].mean())
    new = new.drop([841],axis=1)
    new[843] = new[843].fillna(new[843].mean())
    for i in range(4):
        new[844+i] = new[844+i].fillna(new[844+i].mean())

    return new


      




final_test=preprocess(test,b)




final_test




np.unique(predic,return_counts=True)




predic = m.predict(final_test)




dd = pd.DataFrame(predic)
dd.to_csv('answer.csv')

