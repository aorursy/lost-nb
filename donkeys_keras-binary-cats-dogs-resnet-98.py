#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import math
import PIL
from PIL import ImageOps
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Activation, Dropout, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, applications
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras import backend as K 
from sklearn.preprocessing import LabelEncoder

from tqdm.auto import tqdm
tqdm.pandas()


# In[2]:


get_ipython().system('ls ../input')


# In[3]:


ls ../input/dogs-vs-cats/test1


# In[4]:


#!ls ../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5


# In[5]:


train_dir = "../input/dogs-vs-cats/train/train"
file_list = os.listdir(train_dir)
DOG = "dog"
CAT = "cat"
TRAIN_TOTAL = len(file_list)
labels = []
df_train = pd.DataFrame()


# In[6]:


get_ipython().run_cell_magic('time', '', 'idx = 0\nimg_sizes = []\nwidths = np.zeros(TRAIN_TOTAL, dtype=int)\nheights = np.zeros(TRAIN_TOTAL, dtype=int)\naspect_ratios = np.zeros(TRAIN_TOTAL) #defaults to type float\nfor filename in file_list:\n    if "cat" in filename.lower():\n        labels.append(CAT)\n    else:\n        labels.append(DOG)\n    img = PIL.Image.open(f"{train_dir}/{filename}")\n    img_size = img.size\n    img_sizes.append(img_size)\n    widths[idx] = img_size[0]\n    heights[idx] = img_size[1]\n    aspect_ratios[idx] = img_size[0]/img_size[1]\n    img.close()\n    idx += 1')


# In[7]:


df_train["filename"] = file_list
df_train["cat_or_dog"] = labels
label_encoder = LabelEncoder()
df_train["cd_label"] = label_encoder.fit_transform(df_train["cat_or_dog"])
df_train["size"] = img_sizes
df_train["width"] = widths
df_train["height"] = heights
df_train["aspect_ratio"] = aspect_ratios
df_train.head()


# In[8]:


df_train["aspect_ratio"].max()


# In[9]:


df_train["aspect_ratio"].min()


# In[10]:


max_idx = df_train["aspect_ratio"].values.argmax()
max_idx


# In[11]:


df_train.iloc[max_idx]


# In[12]:


filename = df_train.iloc[max_idx]["filename"]
img = PIL.Image.open(f"{train_dir}/{filename}")


# In[13]:


### The Broadest Image in the Set


# In[14]:


plt.imshow(img)


# In[15]:


img.close()


# In[16]:


df_sorted = df_train.sort_values(by="aspect_ratio")


# In[17]:


def plot_first_9(df_to_plot):
    plt.figure(figsize=[30,30])
    for x in range(9):
        filename = df_to_plot.iloc[x].filename
        img = PIL.Image.open(f"{train_dir}/{filename}")
        print(filename)
        plt.subplot(3, 3, x+1)
        plt.imshow(img)
        title_str = filename+" "+str(df_to_plot.iloc[x].aspect_ratio)
        plt.title(title_str)


# In[18]:


plot_first_9(df_sorted)


# In[19]:


df_sorted = df_train.sort_values(by="aspect_ratio", ascending=False)


# In[20]:


plot_first_9(df_sorted)


# In[21]:


df_sorted.drop(df_sorted.index[:3], inplace=True)


# In[22]:


plot_first_9(df_sorted)


# In[23]:


df_train = df_sorted


# In[24]:


df_train.dtypes


# In[25]:


#This batch size seemed to work without memory issues
batch_size = 32
#299 is the input size for some of the pre-trained networks. I think ResNet50 is actually 224x224 but I left this as 299 anyway.
img_size = 299 #TODO: 224
#I will try a few variations of training my model on top of ResNet, 5 seems to be enough to get results but leave some time to try the variants.
epochs = 7


# In[26]:


from keras.applications.resnet50 import preprocess_input

def create_generators(validation_perc, shuffle=False, horizontal_flip=False, 
                      zoom_range=0, w_shift=0, h_shift=0, rotation_range=0, shear_range=0,
                     fill_zeros=False, preprocess_func=None):
    #the "nearest" mode copies image pixels on borders when shifting/rotation/etc to cover empty space
    fill_mode = "nearest"
    if fill_zeros:
        #with constant mode, we fill created empty space with zeros
        fill_mode = "constant"
        
    #rescale changes pixels from 1-255 integers to 0-1 floats suitable for neural nets
    rescale = 1./255
    if preprocess_func is not None:
        #https://stackoverflow.com/questions/48677128/what-is-the-right-way-to-preprocess-images-in-keras-while-fine-tuning-pre-traine
        #no need to rescale if using Keras in-built ResNet50 preprocess_func: https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py#L157
        rescale = None

    train_datagen=ImageDataGenerator(
        rescale = rescale, 
        validation_split = validation_perc, #0.25, #subset for validation. seems to be subset='validation' in flow_from_dataframe
        horizontal_flip = horizontal_flip,
        zoom_range = zoom_range,
        width_shift_range = w_shift,
        height_shift_range=h_shift,
        rotation_range=rotation_range,
        shear_range=shear_range,
        fill_mode=fill_mode,
        cval=0,#this is the color value to fill with when "constant" mode used. 0=black
        preprocessing_function=preprocess_func
    )

    #Keras has this two-part process of defining generators. 
    #First the generic properties above, then the actual generators with filenames and all.
    train_generator=train_datagen.flow_from_dataframe(
        dataframe=df_train,
        directory=train_dir,
        x_col="filename", #the name of column containing image filename in dataframe
        y_col="cat_or_dog", #the y-col in dataframe
        batch_size=batch_size, 
        shuffle=shuffle,
        class_mode="binary", #categorical if multiple. then y_col can be list or tuple also 
        #classes=lbls, #list of ouput classes. if not provided, inferred from data
        target_size=(img_size,img_size),
        subset='training') #the subset of data from the ImageDataGenerator definition above. The validation_split seems to produce these 2 values.

    valid_generator=train_datagen.flow_from_dataframe(
        dataframe=df_train,
        directory=train_dir,
        x_col="filename",
        y_col="cat_or_dog",
        batch_size=batch_size,
        shuffle=shuffle,
        class_mode="binary",
        #classes=lbls,
        target_size=(img_size,img_size), #gave strange error about tuple cannot be interpreted as integer
        subset='validation') #the subset of data from the ImageDataGenerator definition above. The validation_split seems to produce these 2 values.

    return train_generator, valid_generator, train_datagen


# In[27]:





# In[27]:


train_generator, valid_generator, train_datagen = create_generators(0, False, False, 0, 0, 0)


# In[28]:


train_generator.class_indices


# In[29]:


class_map = {v: k for k, v in train_generator.class_indices.items()}


# In[30]:


import matplotlib

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)


# In[31]:


def plot_batch_9():
    train_generator.reset()
    # configure batch size and retrieve one batch of images
    plt.clf() #clears matplotlib data and axes
    #for batch in train_generator:
    plt.figure(figsize=[30,30])
    batch = next(train_generator)
    for x in range(0,9):
    #    print(train_generator.filenames[x])
        plt.subplot(3, 3, x+1)
        plt.imshow(batch[0][x], interpolation='nearest')
        item_label = batch[1][x]
        item_label = class_map[int(item_label)]
        plt.title(item_label)

    plt.show()


# In[32]:


plot_batch_9()


# In[33]:


def show_img(idx):
    filename = df_train.iloc[idx]["filename"]
    img = PIL.Image.open(f"{train_dir}/{filename}")
    plt.imshow(img)
    img.close()


# In[34]:


show_img(2)


# In[35]:


show_img(1)


# In[36]:


train_generator, valid_generator, train_datagen = create_generators(validation_perc = 0, 
                                                                    shuffle = False, 
                                                                    horizontal_flip = True, 
                                                                    zoom_range = 0, 
                                                                    w_shift = 0, 
                                                                    h_shift = 0)


# In[37]:


plot_batch_9()


# In[38]:


train_generator, valid_generator, train_datagen = create_generators(validation_perc = 0, 
                                                                    shuffle = False, 
                                                                    horizontal_flip = True, 
                                                                    zoom_range = 0, 
                                                                    w_shift = 0.2, 
                                                                    h_shift = 0)


# In[39]:


plot_batch_9()


# In[40]:


train_generator, valid_generator, train_datagen = create_generators(validation_perc = 0, 
                                                                    shuffle = False, 
                                                                    horizontal_flip = True, 
                                                                    zoom_range = 0.2, 
                                                                    w_shift = 0.2, 
                                                                    h_shift = 0.2)


# In[41]:


plot_batch_9()


# In[42]:


train_generator, valid_generator, train_datagen = create_generators(validation_perc = 0, 
                                                                    shuffle = False, 
                                                                    horizontal_flip = True, 
                                                                    zoom_range = 0.2, 
                                                                    w_shift = 0.2, 
                                                                    h_shift = 0.2,
                                                                   fill_zeros = True)


# In[43]:


plot_batch_9()


# In[44]:


train_generator, valid_generator, train_datagen = create_generators(validation_perc = 0, 
                                                                    shuffle = False, 
                                                                    horizontal_flip = False, 
                                                                    zoom_range = 0, 
                                                                    w_shift = 0, 
                                                                    h_shift = 0,
                                                                    fill_zeros = True,
                                                                   rotation_range=20)


# In[45]:


plot_batch_9()


# In[46]:


train_generator, valid_generator, train_datagen = create_generators(validation_perc = 0, 
                                                                    shuffle = False, 
                                                                    horizontal_flip = False, 
                                                                    zoom_range = 0, 
                                                                    w_shift = 0, 
                                                                    h_shift = 0,
                                                                    fill_zeros = True,
                                                                   shear_range=20)


# In[47]:


plot_batch_9()


# In[48]:


train_generator, valid_generator, train_datagen = create_generators(validation_perc = 0, 
                                                                    shuffle = False, 
                                                                    horizontal_flip = False, 
                                                                    zoom_range = 0, 
                                                                    w_shift = 0, 
                                                                    h_shift = 0,
                                                                    fill_zeros = True,
                                                                   shear_range=90)


# In[49]:


plot_batch_9()


# In[50]:


from keras.applications import resnet50

train_generator, valid_generator, train_datagen = create_generators(validation_perc = 0.2, 
                                                                    shuffle = True, 
                                                                    horizontal_flip = True, 
                                                                    zoom_range = 0.2, 
                                                                    w_shift = 0.2, 
                                                                    h_shift = 0.2,
                                                                    fill_zeros = True,
                                                                    preprocess_func = resnet50.preprocess_input,
                                                                   shear_range=10)


# In[51]:


plot_batch_9()


# In[52]:


df_train.head()


# In[53]:





# In[53]:


#the total number of images we have:
train_size = len(train_generator.filenames)
#train_steps is how many steps per epoch Keras runs the genrator. One step is batch_size*images
train_steps = train_size/batch_size
#use 2* number of images to get more augmentations in. some do, some dont. up to you
train_steps = int(2*train_steps)
#same for the validation set
valid_size = len(valid_generator.filenames)
valid_steps = valid_size/batch_size
valid_steps = int(2*valid_steps) 


# In[54]:


from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import (Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D,
                          BatchNormalization, Input, Conv2D, GlobalAveragePooling2D)
from keras.applications.resnet50 import ResNet50

def create_model(trainable_layer_count):
    input_tensor = Input(shape=(img_size, img_size, 3))
    base_model = ResNet50(include_top=False,
                          #the weights value can apparently also be a file path..
                   weights=None, #loading weights from dataset, avoiding need for internet conn
                   input_tensor=input_tensor)
    base_model.load_weights('../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
    if trainable_layer_count == "all":
        #the full pre-trained model is fine-tuned in this case
        for layer in base_model.layers:
            layer.trainable = True
    else:
        #if not all should be trainable, first set them all as non-trainable (fixed)
        for layer in base_model.layers:
            layer.trainable = False
        #and finally set the last N layers as trainable
        #idea is to re-use higher level features and fine-tune the finer details
        for layer in base_model.layers[-trainable_layer_count:]:
            layer.trainable = True
    print("base model has {} layers".format(len(base_model.layers)))
    #here on it is the fully custom classification on top of pre-trained layers above
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu', kernel_regularizer=l2(5e-4))(x)
    x = Dropout(0.5)(x)
    #doing binary prediction, so just 1 neuron is enough
    final_output = Dense(1, activation='sigmoid', name='final_output')(x)
    model = Model(input_tensor, final_output)
    
    return model


# In[55]:


# create callbacks list
from keras.callbacks import (ModelCheckpoint, LearningRateScheduler,
                             EarlyStopping, ReduceLROnPlateau,CSVLogger)
                             
from sklearn.model_selection import train_test_split


checkpoint = ModelCheckpoint('../working/Resnet50_best.h5', monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)
reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,
                                   verbose=1, mode='auto', epsilon=0.0001)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=7)

csv_logger = CSVLogger(filename='../working/training_log.csv',
                       separator=',',
                       append=True)

callbacks_list = [checkpoint, csv_logger, early]
# callbacks_list = [checkpoint, csv_logger, reduceLROnPlat]


# In[56]:


model = create_model("all")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[57]:


fit_history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_steps,
        epochs = epochs,
        validation_data=valid_generator,
        validation_steps=valid_steps,
        callbacks=callbacks_list,
    verbose = 1
)
#this would load the best scoring weights from above for prediction
model.load_weights("../working/Resnet50_best.h5")


# In[58]:


fit_history.history


# In[59]:


pd.DataFrame(fit_history.history).head(20)


# In[60]:


def plot_loss_and_accuracy(fit_history):
    plt.clf()
    plt.plot(fit_history.history['acc'])
    plt.plot(fit_history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.clf()
    # summarize history for loss
    plt.plot(fit_history.history['loss'])
    plt.plot(fit_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


# In[61]:


plot_loss_and_accuracy(fit_history)


# In[62]:


model = create_model(0)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[63]:


train_generator.reset()
valid_generator.reset()
fit_history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_steps,
        epochs = epochs,
        validation_data=valid_generator,
        validation_steps=valid_steps,
        callbacks=callbacks_list,
    verbose = 1
)
model.load_weights("../working/Resnet50_best.h5")


# In[64]:


pd.DataFrame(fit_history.history).head(20)


# In[65]:


plot_loss_and_accuracy(fit_history)


# In[66]:





# In[66]:


model = create_model(5)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[67]:


train_generator.reset()
valid_generator.reset()
fit_history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_steps,
        epochs = epochs,
        validation_data=valid_generator,
        validation_steps=valid_steps,
        callbacks=callbacks_list,
    verbose = 1
)
model.load_weights("../working/Resnet50_best.h5")


# In[68]:


pd.DataFrame(fit_history.history).head(20)


# In[69]:


plot_loss_and_accuracy(fit_history)


# In[70]:


valid_generator.reset()
df_valid = pd.DataFrame()


# In[71]:


np.set_printoptions(suppress=True)
diffs = []
predictions = []
cat_or_dog = []
cd_labels = []
for filename in tqdm(valid_generator.filenames):
    img = PIL.Image.open(f'{train_dir}/{filename}')
    resized = img.resize((img_size, img_size))
    np_img = np.array(resized)
    if "cat" in filename.lower():
        reference = 0 #cat
        cat_or_dog.append(CAT)
    else:
        reference = 1 #dog
        cat_or_dog.append(DOG)
    cd_labels.append(reference)
    score_predict = model.predict(preprocess_input(np_img[np.newaxis]))
#    print(reference)
#    print(score_predict[0][0])
    diffs.append(abs(reference-score_predict[0][0]))
    predictions.append(score_predict)


# In[72]:


max(diffs)


# In[73]:


df_valid["filename"] = valid_generator.filenames
df_valid["cat_or_dog"] = cat_or_dog
df_valid["cd_label"] = cd_labels
df_valid["diff"] = diffs
df_valid["prediction"] = predictions


# In[74]:


df_valid.sort_values(by="diff", ascending=False).head()


# In[75]:


def show_diff_imgs(n):
    sorted_diffs = df_valid.sort_values(by="diff", ascending=False)
    x = 0
    rows = int(math.ceil(n/3))
    height = rows*10
    plt.figure(figsize=[30,height])
    for index, row in sorted_diffs.iterrows():
        filename = row["filename"]
        cat_or_dog = row["cat_or_dog"]
        cd_label = row["cd_label"]
        diff = row["diff"]
        prediction = row["prediction"]
        #print(prediction)
        pred_str = "{:.2f}".format(prediction[0][0])
        img = PIL.Image.open(f"{train_dir}/{filename}")
        print(filename+" "+cat_or_dog+" "+str(diff))
        plt.subplot(3, rows, x+1)
        plt.imshow(img)
        title_str = f"{cat_or_dog}: {cd_label} vs {pred_str}"
        plt.title(title_str)        
        img.close()
        x += 1
        if x > n:
            break


# In[76]:


show_diff_imgs(10)


# In[77]:


get_ipython().system('ls ../input')


# In[78]:


test_dir = "../input/dogs-vs-cats/test1/test1"
test_filenames = os.listdir(test_dir)
test_df = pd.DataFrame({
    'filename': test_filenames
})
nb_samples = test_df.shape[0]


# In[79]:


np.set_printoptions(suppress=True)
predictions = []
for filename in tqdm(test_filenames):
    img = PIL.Image.open(f'{test_dir}/{filename}')
    resized = img.resize((img_size, img_size))
    np_img = np.array(resized)
    np_img = resnet50.preprocess_input(np_img)
    score_predict = model.predict(np_img[np.newaxis])
    predictions.append(score_predict)


# In[80]:


#1=dog,0=cat
threshold = 0.5
test_df['probability'] = predictions
test_df['category'] = np.where(test_df['probability'] > threshold, 1,0)


# In[81]:


test_df.head()


# In[82]:


filename = test_df.iloc[1]["filename"]
img = PIL.Image.open(f'{test_dir}/{filename}')
plt.imshow(img)


# In[83]:


submission_df = test_df.copy()
submission_df['id'] = submission_df['filename'].str.split('.').str[0]
submission_df['label'] = submission_df['category']
submission_df.drop(['filename', 'category'], axis=1, inplace=True)
submission_df.to_csv('submission.csv', index=False)

