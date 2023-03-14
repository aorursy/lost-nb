#!/usr/bin/env python
# coding: utf-8



import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')




from tqdm import tqdm
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,Dropout,MaxPooling2D,Activation,BatchNormalization,GlobalAveragePooling2D
from keras.optimizers import Adam,SGD
from sklearn.model_selection import train_test_split
from keras.callbacks import CSVLogger,ModelCheckpoint,ReduceLROnPlateau
from keras.regularizers import l2
from PIL import Image




ls ../input




trainDf=pd.read_csv("../input/train.csv")




trainDf.head()




trainDf.shape




trainDf['has_cactus'].hist()




trainDf['has_cactus'].value_counts()




def load_df(dataframe=None,batchSize=16):
    dataframe=trainDf
    if dataframe is None:
        dataframe=pd.read_csv("../input/train.csv")
        
    #The generator takes only string categorical value so converting the categorical value into str
    dataframe['has_cactus']=dataframe['has_cactus'].apply(str) 
    gen=ImageDataGenerator(rescale=1/255,horizontal_flip=True,vertical_flip=True,validation_split=0.1)
    trainGen=gen.flow_from_dataframe(dataframe,directory='../input/train/train',x_col='id',y_col='has_cactus',target_size=(32,32),
                                    class_mode='categorical',batch_size=batchSize,shuffle=True,subset='validation')
    
    testGen=gen.flow_from_dataframe(dataframe,directory='../input/train/train',x_col='id',y_col='has_cactus',target_size=(32,32),
                                    class_mode='categorical',batch_size=batchSize,shuffle=True,subset='validation')
    return trainGen,testGen




# Okay lets load the data 
trainGen,testGen=load_df(batchSize=32)




model=Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(32,32,3)))

model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))




model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.RMSprop(lr=0.0005, decay=1e-5),
              metrics=['accuracy'])
model.summary()




model.fit_generator(trainGen,steps_per_epoch=5000,epochs=3,validation_data=testGen,validation_steps=500,shuffle=True)




submission_set=pd.read_csv('../input/sample_submission.csv')




submission_set.head()




submission_set.shape




predictions=np.empty((submission_set.shape[0],))
for n in tqdm(range(submission_set.shape[0])):
    data=np.array(Image.open('../input/test/test/'+submission_set.id[n]))
    data=data.astype(np.float32)/255.
    predictions[n]=model.predict(data.reshape((1,32,32,3)))[0][1]

submission_set['has_cactus']=predictions
submission_set.to_csv('sample_submission.csv',index=False)    




Image.open("../input/test/test/000940378805c44108d287872b2f04ce.jpg")




ls ../input/test/test




submission_set.head()






