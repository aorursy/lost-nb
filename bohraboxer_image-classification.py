#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
                	
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




train=pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
test=pd.read_csv("../input/aptos2019-blindness-detection/test.csv")




train['diagnosis'].hist()
train['diagnosis'].value_counts()




import cv2
import os
import scipy
from tqdm import tqdm
from PIL import Image




# Preprocecss data
train["id_code"] = train["id_code"].apply(lambda x: x + ".png")
test["id_code"] = test["id_code"].apply(lambda x: x + ".png")
train['diagnosis'] = train['diagnosis'].astype('str')
train.head()




from keras.preprocessing.image import ImageDataGenerator








from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from keras.models import Model
from keras.applications import densenet
from keras.applications.densenet import preprocess_input




from keras import applications
base_model = applications.DenseNet201(weights=None,include_top=False)
base_model.load_weights('../input/models-pretrained-weights/densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5')




from keras import regularizers
from keras.layers import Dense,GlobalAveragePooling2D,Dropout
x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dropout(0.5)(x)
preds=Dense(5,activation='softmax')(x)




model=Model(input=base_model.input,outputs=preds)




from keras.optimizers import Adam

model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001),metrics=['accuracy'])




nb_classes=5
lbls=list(map(str,range(nb_classes)))
batch_size=32
img_size=224
nb_epochs=30




train_datagen=ImageDataGenerator(rescale=1./255,
                                featurewise_center=True,
                                featurewise_std_normalization=True,
                                zca_whitening=True,
                                rotation_range=30,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                horizontal_flip=True,
                                vertical_flip=True,
                                validation_split=0.2,
                                zoom_range=0.25)




train_generator=train_datagen.flow_from_dataframe(dataframe=train,
                                                 directory='../input/aptos2019-blindness-detection/train_images/',
                                                 x_col='id_code',
                                                 y_col='diagnosis',
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 class_mode='categorical',
                                                 classes=lbls,
                                                 target_size=(img_size,img_size),
                                                 subset='training')


valid_generator=train_datagen.flow_from_dataframe(dataframe=train,
                                                 directory='../input/aptos2019-blindness-detection/train_images/',
                                                 x_col='id_code',
                                                 y_col='diagnosis',
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 class_mode='categorical',
                                                 classes=lbls,
                                                 target_size=(img_size,img_size),
                                                 subset='validation')




from keras.callbacks import EarlyStopping, ModelCheckpoint

es=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=7)
mc=ModelCheckpoint('model_weights.h5',monitor='val_loss',save_best_only=True,mode='min',verbose=1)

history=model.fit_generator(generator=train_generator,
                           steps_per_epoch=32,
                           epochs=nb_epochs,
                           validation_data=valid_generator,
                           validation_steps=32,
                           callbacks=[es,mc])




history_df=pd.DataFrame(history.history)
history_df[['loss','val_loss']].plot()
history_df[['acc','val_acc']].plot()

    




complete_datagen = ImageDataGenerator(rescale=1./255)
complete_generator = complete_datagen.flow_from_dataframe(  
        dataframe=train,
        directory = "../input/aptos2019-blindness-detection/train_images/",
        x_col="id_code",
        target_size=(img_size, img_size),
        batch_size=1,
        shuffle=False,
        class_mode=None)

STEP_SIZE_COMPLETE = complete_generator.n//complete_generator.batch_size
train_preds = model.predict_generator(complete_generator, steps=STEP_SIZE_COMPLETE)
train_preds = [np.argmax(pred) for pred in train_preds]




from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

labels = ['0 - No DR', '1 - Mild', '2 - Moderate', '3 - Severe', '4 - Proliferative DR']
cnf_matrix = confusion_matrix(train['diagnosis'].astype('int'), train_preds)
cnf_matrix_norm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
df_cm = pd.DataFrame(cnf_matrix_norm, index=labels, columns=labels)
plt.figure(figsize=(16, 7))
sns.heatmap(df_cm, annot=True, fmt='.2f', cmap="Blues")
plt.show()




from sklearn.metrics import cohen_kappa_score


print("Train Cohen Kappa score: %.3f" % cohen_kappa_score(train_preds, train['diagnosis'].astype('int'), weights='quadratic'))




test_datagen = ImageDataGenerator(rescale=1./255)

test_datagenerator = test_datagen.flow_from_dataframe(  
        dataframe=test,
        directory = "../input/aptos2019-blindness-detection/test_images/",
        x_col="id_code",
        target_size=(img_size, img_size),
        batch_size=1,
        shuffle=False,
        class_mode=None)

test_datagenerator.reset()
STEP_SIZE_TEST = test_datagenerator.n//test_datagenerator.batch_size
preds = model.predict_generator(test_datagenerator, steps=STEP_SIZE_TEST)
predictions = [np.argmax(pred) for pred in preds]




filenames = test_datagenerator.filenames
results = pd.DataFrame({'id_code':filenames, 'diagnosis':predictions})
results['id_code'] = results['id_code'].map(lambda x: str(x)[:-4])
results.to_csv('submission.csv',index=False)
results.head(10)




f, ax = plt.subplots(figsize=(14, 8.7))
ax = sns.countplot(x="diagnosis", data=results, palette="GnBu_d")
sns.despine()
plt.show()

