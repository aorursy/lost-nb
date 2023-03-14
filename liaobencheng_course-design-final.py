#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.patches as patches
import random
from sklearn.utils import shuffle
from tqdm import tqdm
from tqdm import tqdm_notebook

from numpy.random import seed
seed(101)
from tensorflow import set_random_seed
set_random_seed(101)

import pandas as pd
import numpy as np

from shutil import copyfile, move
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import initializers

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import BatchNormalization

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


data = pd.read_csv('/kaggle/input/train_labels.csv')
train_path = '/kaggle/input/train/'
test_path = '/kaggle/input/test/'
# 为flow from dataframe做准备
data["filename"] = [item.id+".tif" for idx, item in data.iterrows()]
data["class"] = ["b_has tumor" if item.label==1 else "a_no tumor" for idx, item in data.iterrows()]
# 保证每次重新跑时用于快速验证的10000个数据是相同的
baseline_data = data[:10000]
print(data['label'].value_counts())
print(data.head())

# data = data.sample(10000, random_state = 101)


# In[3]:


def readImage(path):
    bgr_img = cv2.imread(path)
    b,g,r = cv2.split(bgr_img)
    rgb_img = cv2.merge([r,g,b])
    return rgb_img

shuffled_data = shuffle(data)
# shuffled_data = shuffle(baseline_data)
# path = os.path.join(train_path, idx)

dark_th = 10      # 黑色图片
bright_th = 245   # 白色图片
too_dark_idx = []
too_bright_idx = []

x_tot = np.zeros(3)
x2_tot = np.zeros(3)
counted_ones = 0
for i, idx in tqdm_notebook(enumerate(shuffled_data['filename']), 'computing statistics...(220025 it total)'):
# for i, idx in tqdm_notebook(enumerate(shuffled_data['filename']), 'computing statistics...(10000 it total)'):
    path = os.path.join(train_path, idx)
    imagearray = readImage(path).reshape(-1,3)
#     imagearray = readImage(path + '.tif')
    # is this too dark
    if(imagearray.max() < dark_th):
        too_dark_idx.append(idx)
        continue # do not include in statistics
    # is this too bright
    if(imagearray.min() > bright_th):
        too_bright_idx.append(idx)
        continue # do not include in statistics
    x_tot += imagearray.mean(axis=0)
    x2_tot += (imagearray**2).mean(axis=0)
    counted_ones += 1
    


# In[4]:


channel_avr = (x_tot/counted_ones)/255
channel_std = (np.sqrt(x2_tot/counted_ones - channel_avr**2))/255
print(channel_avr,channel_std)


# In[5]:


print('有{0}张黑色图片'.format(len(too_dark_idx)))
print('以及{0}张白色图片'.format(len(too_bright_idx)))
print('黑色图片:')
print(too_dark_idx)
print('白色图片:')
print(too_bright_idx)


# In[6]:


# from sklearn.model_selection import train_test_split
# train_df = baseline_data

# #If removing outliers, uncomment the four lines below
# print('Before removing outliers we had {0} training samples.'.format(len(train_df)))

# for i in too_dark_idx:
#     train_df =  train_df[train_df['filename'] != i]
    
# for j in too_bright_idx:
#     train_df =  train_df[train_df['filename'] != j]

# print('After removing outliers we have {0} training samples.'.format(len(train_df)))

# train_names = train_df.id.values
# train_labels = np.asarray(train_df['label'].values)

# # split, this function returns more than we need as we only need the validation indexes for fastai
# df_train, df_val= train_test_split(train_df, test_size=0.1, stratify=train_labels, random_state=101)


# In[7]:


from sklearn.model_selection import train_test_split
train_df = data

#If removing outliers, uncomment the four lines below
print('Before removing outliers we had {0} training samples.'.format(len(train_df)))
# train_df[(~train_df['id'].isin(too_dark_idx))]
# train_df = train_df.drop(labels=too_dark_idx, axis=0)
# train_df = train_df.drop(labels=too_bright_idx, axis=0)
# train_df = train_df[train_df.id != too_dark_idx]
for i in too_dark_idx:
    train_df =  train_df[train_df['filename'] != i]
    
for j in too_bright_idx:
    train_df =  train_df[train_df['filename'] != j]

print('After removing outliers we have {0} training samples.'.format(len(train_df)))
train_df = train_df.reset_index(drop=True)
train_names = train_df.id.values
train_labels = np.asarray(train_df['label'].values)

# split, this function returns more than we need as we only need the validation indexes for fastai
df_train, df_val= train_test_split(train_df, test_size=0.1, stratify=train_labels, random_state=101)


# In[8]:


df_train.head()


# In[9]:


num_train_samples = len(df_train)
num_val_samples = len(df_val)
train_batch_size = 64
val_batch_size = 64


train_steps = np.ceil(num_train_samples / train_batch_size)
val_steps = np.ceil(num_val_samples / val_batch_size)

target_size = (96,96)

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=40,
#     zoom_range=0.2, 
#     width_shift_range=0.1,
#     height_shift_range=0.1
)

# train_datagen = ImageDataGenerator(
#         rescale=1./255
# )

train_generator = train_datagen.flow_from_dataframe(
    dataframe = df_train,
    x_col='filename',
    y_col='class',
    directory='../input/train/',
    target_size=target_size,
    batch_size=train_batch_size,
    shuffle=True,
    class_mode='binary')


val_datagen = ImageDataGenerator(rescale=1. / 255)
val_generator = val_datagen.flow_from_dataframe(
    dataframe = df_val,
    x_col='filename',
    y_col='class',
    directory='../input/train/',
    target_size=target_size,
    shuffle=False,
    batch_size=val_batch_size,
    class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = val_datagen.flow_from_dataframe(
    dataframe = df_val,
    x_col='filename',
    y_col='class',
    directory='../input/train/',
    target_size=target_size,
    shuffle=False,
    batch_size=val_batch_size,
    class_mode='binary')


# In[10]:


test_generator.class_indices


# In[11]:


def plot_random_samples(generator):
    generator_size = len(generator)
    index=random.randint(0,generator_size-1)
    image,label = generator.__getitem__(index)

    sample_number = 10
    fig = plt.figure(figsize = (20,sample_number))
    for i in range(0,sample_number):
        ax = fig.add_subplot(2, 5, i+1)
        ax.imshow(image[i])
        if label[i]==0:
            ax.set_title("has tumor")
        elif label[i]==1:
            ax.set_title("no tumor")
    plt.tight_layout()
    plt.show()


# In[12]:


plot_random_samples(val_generator)


# In[13]:


# kernel_size = (3,3)
# pool_size= (2,2)
# first_filters = 32
# second_filters = 64
# third_filters = 128

# dropout_conv = 0.3
# dropout_dense = 0.3


# CNN3_model = Sequential()
# CNN3_model.add(Conv2D(first_filters, kernel_size, activation = 'relu', input_shape = (96, 96, 3)))
# CNN3_model.add(MaxPooling2D(pool_size = pool_size)) 
# CNN3_model.add(Dropout(dropout_conv))

# CNN3_model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
# CNN3_model.add(MaxPooling2D(pool_size = pool_size))
# CNN3_model.add(Dropout(dropout_conv))

# CNN3_model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
# CNN3_model.add(MaxPooling2D(pool_size = pool_size))
# CNN3_model.add(Dropout(dropout_conv))

# CNN3_model.add(Flatten())
# CNN3_model.add(Dense(256, activation = "relu"))
# CNN3_model.add(Dropout(dropout_dense))
# CNN3_model.add(Dense(1, activation = "sigmoid"))

# CNN3_model.summary()


# In[14]:





# In[14]:


from keras import regularizers

IMAGE_SIZE = 96
kernel_size = (3,3)
pool_size= (2,2)
pool_size1 = (1,1)
first_filters = 32
second_filters = 64
third_filters = 128

dropout_conv = 0.3
dropout_dense = 0.3


CNN9_model = Sequential()
CNN9_model.add(Conv2D(first_filters, kernel_size, activation = 'relu', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3),
                 kernel_initializer='lecun_normal'))
CNN9_model.add(Conv2D(first_filters, kernel_size, activation = 'relu',
                 kernel_initializer='lecun_normal'))
CNN9_model.add(Conv2D(first_filters, kernel_size, activation = 'relu',
                 kernel_initializer='lecun_normal'))
CNN9_model.add(MaxPooling2D(pool_size = pool_size)) 
CNN9_model.add(Dropout(dropout_conv))

CNN9_model.add(Conv2D(second_filters, kernel_size, activation ='relu',
                 kernel_initializer='lecun_normal'))
CNN9_model.add(Conv2D(second_filters, kernel_size, activation ='relu',
                 kernel_initializer='lecun_normal'))
CNN9_model.add(Conv2D(second_filters, kernel_size, activation ='relu',
                 kernel_initializer='lecun_normal'))
CNN9_model.add(MaxPooling2D(pool_size = pool_size))
CNN9_model.add(Dropout(dropout_conv))

CNN9_model.add(Conv2D(third_filters, kernel_size, activation ='relu',
                 kernel_initializer='lecun_normal'))
CNN9_model.add(Conv2D(third_filters, kernel_size, activation ='relu',
                 kernel_initializer='lecun_normal'))
CNN9_model.add(Conv2D(third_filters, kernel_size, activation ='relu',
                 kernel_initializer='lecun_normal'))
CNN9_model.add(MaxPooling2D(pool_size = pool_size))
CNN9_model.add(Dropout(dropout_conv))

CNN9_model.add(Flatten())
CNN9_model.add(Dense(256, activation = "relu",
                 kernel_initializer='lecun_normal'))
CNN9_model.add(Dropout(dropout_dense))
CNN9_model.add(Dense(1, activation = "sigmoid", activity_regularizer=regularizers.l1(0.001),
                 kernel_initializer='lecun_normal'))

CNN9_model.summary()


# In[15]:


# os.listdir('/kaggle/working')


# In[16]:


# os.remove('/kaggle/working/CNN3_model.h5')


# In[17]:


# filepath = "CNN3_model.h5"

# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
#                              save_best_only=True, mode='max')

# reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2, 
#                                    verbose=1, mode='max', min_lr=0.00001)
                              
                              
# callbacks_list = [checkpoint, reduce_lr]

# CNN3_history = CNN3_model.fit_generator(train_generator, steps_per_epoch=train_steps, 
#                     validation_data=val_generator,
#                     validation_steps=val_steps,
#                     epochs=30, verbose=1,
#                    callbacks=callbacks_list)


# In[18]:


# CNN3_model.load_weights('CNN3_model.h5')

# val_loss, val_acc = \
# CNN3_model.evaluate_generator(test_generator, 
#                         steps=len(df_val))

# print('val_loss:', val_loss)
# print('val_acc:', val_acc)


# In[19]:


# os.listdir('/kaggle/working')


# In[20]:


CNN9_model.compile(Adam(lr=0.0001), loss='binary_crossentropy', 
              metrics=['accuracy'])

# sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
# CNN9_model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])


# In[21]:


os.listdir('/kaggle/working')


# In[22]:


os.remove('/kaggle/working/CNN9_model.h5')


# In[23]:


filepath = "CNN9_model.h5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_best_only=True, mode='max')

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2, 
                                   verbose=1, mode='max', min_lr=0.00001)
                              
                              
callbacks_list = [checkpoint, reduce_lr]

CNN9_history = CNN9_model.fit_generator(train_generator, steps_per_epoch=train_steps, 
                    validation_data=val_generator,
                    validation_steps=val_steps,
                    epochs=10, verbose=1,
                   callbacks=callbacks_list)


# In[24]:


os.listdir('/kaggle/working')


# In[25]:


CNN9_model.load_weights('CNN9_model.h5')

val_loss, val_acc = CNN9_model.evaluate_generator(test_generator, 
                        steps=len(df_val))

print('val_loss:', val_loss)
print('val_acc:', val_acc)


# In[26]:


plt.plot(CNN9_history.history['acc'])
plt.plot(CNN9_history.history['val_acc'])
plt.title('Accuracy over epochs')
plt.ylabel('Acc')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='best')
plt.show()


# In[27]:


plt.plot(CNN9_history.history['loss'])
plt.plot(CNN9_history.history['val_loss'])
plt.title('Loss over epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='best')
plt.show()


# In[28]:





# In[28]:


predictions = CNN9_model.predict_generator(test_generator, steps=len(df_val), verbose=1)
predictions.shape


# In[29]:


# 查看不同类的索引
test_generator.class_indices


# In[30]:


df_preds = pd.DataFrame(predictions, columns=['b_has tumor'])
df_preds.head()


# In[31]:


test_generator.classes


# In[32]:


y_true = test_generator.classes
y_pred = df_preds['b_has tumor']


# In[33]:


from sklearn.metrics import roc_curve, auc
# 概率
probs = np.exp(y_pred[:])
# 计算ROC曲线
fpr, tpr, thresholds = roc_curve(y_true, probs, pos_label=1)

# 计算ROC面积
roc_auc = auc(fpr, tpr)
print('ROC area is {0}'.format(roc_auc))


# In[34]:


# from sklearn.metrics import roc_auc_score

# roc_auc_score(y_true, y_pred)


# In[35]:


plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")


# In[36]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    画出混淆矩阵
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# In[37]:


predictions = predictions.flatten()


# In[38]:


index = 0

for i in range(len(predictions)):
    if predictions[i]>=0.5:
        predictions[i]=1
    else:
        predictions[i]=0


# In[39]:


predictions = predictions.astype(int)


# In[40]:


test_labels = test_generator.classes


# In[41]:


test_labels = np.array(test_labels)


# In[42]:


cm = confusion_matrix(test_labels, predictions)


# In[43]:


test_generator.class_indices


# In[44]:


# 定义类别的索引
cm_plot_labels = ['a_no tumor', 'b_has tumor']

plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')


# In[45]:


from sklearn.metrics import classification_report

# Generate a classification report

# For this to work we need y_pred as binary labels not as probabilities
y_pred_binary = predictions

report = classification_report(y_true, y_pred_binary, target_names=cm_plot_labels)

print(report)


# In[46]:


os.listdir('/kaggle')


# In[47]:


src="../input/test"

test_folder="../test_folder"
dst = test_folder+"/test"
os.mkdir(test_folder)
os.mkdir(dst)

file_list =  os.listdir(src)
with tqdm(total=len(file_list)) as pbar:
    for filename in file_list:
        pbar.update(1)
        copyfile(src+"/"+filename,dst+"/"+filename)
        
test_datagen = ImageDataGenerator(
    rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    directory=test_folder,
    target_size=target_size,
    batch_size=1,
    shuffle=False,
    class_mode='binary'
)


# In[48]:


pred=CNN9_model.predict_generator(test_generator,verbose=1)


# In[49]:


csv_file = open("sample_submission.csv","w")
csv_file.write("id,label\n")
for filename, prediction in zip(test_generator.filenames,pred):
    name = filename.split("/")[1].replace(".tif","")
    csv_file.write(str(name)+","+str(prediction[0])+"\n")
csv_file.close()

