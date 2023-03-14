#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import imageio

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

import os
import cv2

from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools
import shutil
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


IMAGE_SIZE = 96
IMAGE_CHANNELS = 3

SAMPLE_SIZE = 1000


# In[3]:


os.listdir('../input/histopathologic-cancer-detection/')


# In[4]:


print("Folder treningowy wynosi {} obrazów".format(len(os.listdir('../input/histopathologic-cancer-detection/train'))))
print("Folder testowy wynosi {} obrazów".format(len(os.listdir('../input/histopathologic-cancer-detection/test'))))


# In[5]:


df_data = pd.read_csv('../input/histopathologic-cancer-detection/train_labels.csv')
print("Rozmiar zbioru danych wynosi {} wierszy i".format(df_data.shape[0])," {} kolumny.".format(df_data.shape[1]))


# In[6]:


df_data['label'].value_counts()


# In[7]:


def draw_category_images(col_name,figure_cols, df, IMAGE_PATH):
    categories = (df.groupby([col_name])[col_name].nunique()).index
    f, ax = plt.subplots(nrows=len(categories),ncols=figure_cols, figsize=(4*figure_cols,4*len(categories)))
    
    for i, cat in enumerate(categories):
        sample = df[df[col_name]==cat].sample(figure_cols) 
        for j in range(0,figure_cols):
            file=IMAGE_PATH + sample.iloc[j]['id'] + '.tif'
            im=imageio.imread(file)
            ax[i, j].imshow(im, resample=True, cmap='gray')
            ax[i, j].set_title(cat, fontsize=16)  
    plt.tight_layout()
    plt.show()


# In[8]:


IMAGE_PATH = '../input/histopathologic-cancer-detection/train/'


# In[9]:


draw_category_images('label', 4, df_data, IMAGE_PATH)


# In[10]:


df_data.head()


# In[11]:


df_0 = df_data[df_data['label'] == 0].sample(SAMPLE_SIZE, random_state = 101)

df_1 = df_data[df_data['label'] == 1].sample(SAMPLE_SIZE, random_state = 101)

df_data = pd.concat([df_0, df_1],axis = 0).reset_index(drop = True)

df_data = shuffle(df_data)

df_data['label'].value_counts()


# In[12]:


df_data.head()


# In[13]:


y = df_data['label']

df_train, df_val = train_test_split(df_data, test_size = 0.10, random_state = 101, stratify = y)

print("W zbilansowanym zestawie treningowym mamy {} wierszy i ".format(df_train.shape[0])," {} kolumn.".format(df_train.shape[1]))
print("W zbilansowanym zestawie poprawności mamy {} wierszy i ".format(df_val.shape[0])," {} kolumn.".format(df_val.shape[1]))


# In[14]:


df_train['label'].value_counts()


# In[15]:


df_val['label'].value_counts()


# In[16]:


base_dir = 'base_dir'
os.mkdir(base_dir)


# In[17]:


train_dir = os.path.join(base_dir,'train_dir')
os.mkdir(train_dir)


# In[18]:


val_dir = os.path.join(base_dir,'val_dir')
os.mkdir(val_dir)


# In[19]:


no_tumor_tissue = os.path.join(train_dir, 'a_no_tumor_tissue')
os.mkdir(no_tumor_tissue)


# In[20]:


has_tumor_tissue = os.path.join(train_dir, 'b_has_tumor_tissue')
os.mkdir(has_tumor_tissue)


# In[21]:


no_tumor_tissue = os.path.join(val_dir, 'a_no_tumor_tissue')
os.mkdir(no_tumor_tissue)


# In[22]:


has_tumor_tissue = os.path.join(val_dir, 'b_has_tumor_tissue')
os.mkdir(has_tumor_tissue)


# In[23]:


os.listdir('base_dir/train_dir')


# In[24]:


os.listdir('base_dir/val_dir')


# In[25]:


df_data.set_index('id', inplace=True)


# In[26]:


train_list = list(df_train['id'])
val_list = list(df_val['id'])


# In[27]:


for image in train_list:
    # 'id' w pliku csv nie ma rozszerzenia .tif dlatego je teraz dodajemy
    fname = image + '.tif'
    
    #dodawanie etykiety do obecnego obrazka
    target = df_data.loc[image,'label']
    
    #wybieramy odpowiednią nazwę folderu, ze względu na klasę
    if target == 0:
        label = 'a_no_tumor_tissue'
    
    if target == 1:
        label = 'b_has_tumor_tissue'
    
    #ścieżka źródłowa do obrazka
    src = os.path.join('../input/histopathologic-cancer-detection/train', fname)
    
    #miejsce docelowe dla obrazka
    dst = os.path.join(train_dir, label, fname)
    
    #kopiowanie obrazka ze źródła do pliku docelowego
    shutil.copyfile(src, dst)
    
for image in val_list:
    fname = image + '.tif'
    
    target = df_data.loc[image,'label']
    
    if target == 0:
        label = 'a_no_tumor_tissue'
    
    if target == 1:
        label = 'b_has_tumor_tissue'
    
    src = os.path.join('../input/histopathologic-cancer-detection/train', fname)
    
    
    dst = os.path.join(val_dir, label, fname)
    
    shutil.copyfile(src, dst)


# In[28]:


print("W folderze treningowym mamy {} obrazów, gdzie nie ma komórek nowotworowych.".format(len(os.listdir('base_dir/train_dir/a_no_tumor_tissue'))))
print("W folderze treningowym mamy {} obrazów, na których są komórki nowotworowe.".format(len(os.listdir('base_dir/train_dir/b_has_tumor_tissue'))))


# In[29]:


print("W folderze walidacyjnym mamy {} obrazów, na których  nie ma komórek nowotworowych.".format(len(os.listdir('base_dir/val_dir/a_no_tumor_tissue'))))
print("W folderze walidacyjnym mamy {} obrazów, gdzie są komórki nowotworowe.".format(len(os.listdir('base_dir/val_dir/b_has_tumor_tissue'))))


# In[30]:


train_path = 'base_dir/train_dir'
valid_path = 'base_dir/val_dir'
test_path = '../input/test'

IMAGE_SIZE = 96
num_train_samples = len(df_train)
num_val_samples = len(df_val)
train_batch_size = 10
val_batch_size = 10

train_steps = np.ceil(num_train_samples / train_batch_size)
val_steps = np.ceil(num_val_samples / val_batch_size)


# In[31]:


datagen = ImageDataGenerator(rescale=1.0/255)

train_gen = datagen.flow_from_directory(train_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=train_batch_size,
                                        class_mode='categorical')

val_gen = datagen.flow_from_directory(valid_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=val_batch_size,
                                        class_mode='categorical')
test_gen = datagen.flow_from_directory(valid_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=1,
                                        class_mode='categorical',
                                        shuffle=False)


# In[32]:


kernel_size = (3,3)
pool_size = (2,2)
first_filters = 32
second_filters = 64
third_filters = 128


# In[33]:


model = Sequential()
model.add(Conv2D(first_filters, kernel_size, activation = 'relu', input_shape = (96, 96, 3)))
model.add(Conv2D(first_filters, kernel_size, activation = 'relu'))
model.add(Conv2D(first_filters, kernel_size, activation = 'relu'))
model.add(MaxPooling2D(pool_size = pool_size)) 
model.add(Flatten())
model.add(Dense(256, activation = "relu"))

model.add(Dense(2, activation = "softmax"))

model.summary()


# In[34]:


model.compile(Adam(0.01),
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[35]:


print(val_gen.class_indices)


# In[36]:


from keras.callbacks import EarlyStopping, ReduceLROnPlateau


# In[37]:


earlystopper = EarlyStopping(monitor = 'val_loss', patience = 2, verbose = 1, restore_best_weights = True)
reduce_l = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 1)


# In[38]:


history = model.fit_generator(train_gen, steps_per_epoch = train_steps, 
                              validation_data = val_gen, 
                              validation_steps = val_steps,
                              epochs = 10, 
                              callbacks = [reduce_l, earlystopper])


# In[39]:


val_loss, val_acc = model.evaluate_generator(test_gen, steps = len(df_val))


# In[40]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(acc)+1)


# In[41]:


print('val_loss:', val_loss)
print('val_acc:', val_acc)


# In[42]:


plt.plot(epochs,loss,'bo',label = 'Training loss')
plt.plot(epochs,val_loss,'b',label = 'Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.figure()

plt.plot(epochs,acc,'bo',label = 'Training acc')
plt.plot(epochs,val_acc,'b',label = 'Validation acc')
plt.title('Training and validation loss')
plt.legend()
plt.figure()


# In[43]:


plt.figure(1)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr_keras,tpr_keras,label = 'area = :.3f'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc = 'best')
plt.show


# In[44]:


#make a prediction

predictions = model.predict_generator(test_gen, steps = len(df_val),verbose = 1)
predictions.shape

df_preds = pd.DataFrame(predictions, columns = ['no_tumor_tissue', 'has_tumour_tissue'])
df_preds.head

#get the tru labels
y_true = test_gen.classes

#get the predicted labels as probabilities
y_pred = df_preds['has_tumour_tissue']


# In[45]:


from sklearn.metrics import roc_auc_score
roc_auc_score(y_true, y_pred)


# In[46]:


from sklearn.metrics import classification_report


# In[47]:


y_pred_binary = predictions.argmax(axis = 1)

cm_plot_labels = ['no_tumor_tissue', 'has_tumor_tissue']

report = classification_report(y_true, y_pred_binary, target_names = cm_plot_labels)

print(report)


# In[48]:


model = Sequential()
model.add(Conv2D(first_filters, kernel_size, activation = 'relu', input_shape = (96, 96, 3)))
model.add(Conv2D(first_filters, kernel_size, activation = 'relu'))
model.add(Conv2D(first_filters, kernel_size, activation = 'relu'))
model.add(MaxPooling2D(pool_size = pool_size)) 


model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
model.add(MaxPooling2D(pool_size = pool_size))


model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
model.add(MaxPooling2D(pool_size = pool_size))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))

model.add(Dense(2, activation = "softmax"))

model.summary()


# In[49]:


model.compile(Adam(0.01),
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[50]:


print(val_gen.class_indices)


# In[51]:


from keras.callbacks import EarlyStopping, ReduceLROnPlateau


# In[52]:


earlystopper = EarlyStopping(monitor = 'val_loss', patience = 2, verbose = 1, restore_best_weights = True)
reduce_l = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 1)


# In[53]:


hist = model.fit_generator(train_gen, steps_per_epoch = train_steps, 
                              validation_data = val_gen, 
                              validation_steps = val_steps,
                              epochs = 10, 
                              callbacks = [reduce_l, earlystopper])
# po pierwszej epoce z dwóch mamy accuracy = 0.5, a loss = 7.6893


# In[54]:


val_loss, val_acc = model.evaluate_generator(test_gen, steps = len(df_val))


# In[55]:


print('val_loss:', val_loss)
print('val_acc:', val_acc)


# In[56]:


acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']
epochs = range(1,len(acc)+1)


# In[57]:


plt.plot(epochs,loss,'bo',label = 'Training loss')
plt.plot(epochs,val_loss,'b',label = 'Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.figure()

plt.plot(epochs,acc,'bo',label = 'Training acc')
plt.plot(epochs,val_acc,'b',label = 'Validation acc')
plt.title('Training and validation acc')
plt.legend()
plt.figure()


# In[58]:


plt.figure(1)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr_keras,tpr_keras,label = 'area = :.3f'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc = 'best')
plt.show


# In[59]:


#make a prediction

predictions = model.predict_generator(test_gen, steps = len(df_val),verbose = 1)
predictions.shape

df_preds = pd.DataFrame(predictions, columns = ['no_tumor_tissue', 'has_tumour_tissue'])
df_preds.head

#get the tru labels
y_true = test_gen.classes

#get the predicted labels as probabilities
y_pred = df_preds['has_tumour_tissue']


# In[60]:


roc_auc_score(y_true, y_pred)


# In[61]:


from sklearn.metrics import classification_report


# In[62]:


y_pred_bianry = predictions.argmax(axis = 1)

report = classification_report(y_true, y_pred_binary, target_names = cm_plot_labels)

print(report)


# In[63]:


dropout_conv = 0.3
dropout_dense = 0.3


# In[64]:


model = Sequential()
model.add(Conv2D(first_filters, kernel_size, activation = 'relu', input_shape = (96, 96, 3)))
model.add(Conv2D(first_filters, kernel_size, activation = 'relu'))
model.add(Conv2D(first_filters, kernel_size, activation = 'relu'))
model.add(MaxPooling2D(pool_size = pool_size)) 
model.add(Dropout(dropout_conv))

model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(dropout_dense))
model.add(Dense(2, activation = "softmax"))

model.summary()


# In[65]:


model.compile(Adam(0.01),
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[66]:


print(val_gen.class_indices)


# In[67]:


earlystopper = EarlyStopping(monitor = 'val_loss', patience = 2, verbose = 1, restore_best_weights = True)
reduce_l = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 1)


# In[68]:


hist = model.fit_generator(train_gen, steps_per_epoch = train_steps, 
                              validation_data = val_gen, 
                              validation_steps = val_steps,
                              epochs = 2, 
                              callbacks = [reduce_l, earlystopper])


# In[69]:


from sklearn.metrics import roc_auc_score


# In[70]:


#make a prediction

predictions = model.predict_generator(test_gen, steps = len(df_val),verbose = 1)
predictions.shape

df_preds = pd.DataFrame(predictions, columns = ['no_tumor_tissue', 'has_tumour_tissue'])
df_preds.head

#get the tru labels
y_true = test_gen.classes

#get the predicted labels as probabilities
y_pred = df_preds['has_tumour_tissue']


# In[71]:


roc_auc_score(y_true, y_pred)


# In[72]:


val_loss, val_acc = model.evaluate_generator(test_gen, steps = len(df_val))


# In[73]:


acc = hist.hist['acc']
val_acc = hist.hist['val_acc']
loss = hist.hist['loss']
val_loss = hist.hist['val_loss']
epochs = range(1,len(acc)+1)


# In[74]:


plt.plot(epochs,loss,'bo',label = 'Training loss')
plt.plot(epochs,val_loss,'b',label = 'Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.figure()

plt.plot(epochs,acc,'bo',label = 'Training acc')
plt.plot(epochs,val_acc,'b',label = 'Validation acc')
plt.title('Training and validation loss')
plt.legend()
plt.figure()


# In[75]:


plt.figure(0)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr_keras,tpr_keras,label = 'area = :.3f'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc = 'best')
plt.show


# In[76]:


plt.figure(1)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr_keras,tpr_keras,label = 'area = :.3f'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc = 'best')
plt.show


# In[77]:


from sklearn.metrics import classification_report


# In[78]:


y_pred_bianry = predictions.argmax(axis = 1)

report = classification_report(y_true, y_pred_binary, target_names = cm_plot_labels)

print(report)


# In[ ]:





# In[ ]:




