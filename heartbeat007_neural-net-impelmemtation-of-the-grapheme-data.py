#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras


# In[2]:


from keras.applications import DenseNet121
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, optimizers
from sklearn.model_selection import train_test_split


# In[3]:


image_parent_url = '/kaggle/input/bengaliai/256_train/256/'
metadata_url     = '/kaggle/input/bengaliai-cv19/train.csv'


# In[4]:


def load_metadata(url):
    ## for loading all the image data select the dir
    train = pd.read_csv(url)
    return train


# In[5]:


## importing the metadata of the image
train = load_metadata(metadata_url)
train.head()


# In[6]:


## adding corresponding image path to the metadata
## 3gb imag parquet image data create memory allocation problem
## so we are usin ghr png version
train['filename'] = train.image_id.apply(lambda filename: image_parent_url + filename + '.png')


# In[7]:


print(train.head()['grapheme'][2])
img1 = cv2.imread(train.head()['filename'][2])
plt.imshow(img1)


# In[8]:


print(train.head()['grapheme'][3])
img = cv2.imread(train.head()['filename'][3])
plt.imshow(img)
## so all the image is mapped perfectly


# In[9]:


img.shape ## wso we can see that the image is rgb 
### it is ont necessary in this typ of image
### have to convert to gray scale


# In[10]:


## there is more space and random size with random padding 
## we need to change the padding in a fixed size and we need to do some image 
## processing so changing the padding
def get_pad_width(im, new_shape, is_rgb=True):
    ## reduicing the padding
    ## subract and then make this half od the shape
    pad_diff = new_shape - im.shape[0], new_shape - im.shape[1]
    t, b = math.floor(pad_diff[0]/2), math.ceil(pad_diff[0]/2)
    l, r = math.floor(pad_diff[1]/2), math.ceil(pad_diff[1]/2)
    if is_rgb:
        ## if 3 dim then make onde dim 0
        pad_width = ((t,b), (l,r), (0, 0))
    else:
        ## if not settinh up the same
        pad_width = ((t,b), (l,r))
    return pad_width


# In[11]:


## testing
get_pad_width(img,220)


# In[12]:


## the image have unnecessary space even the padding is reduced 
## and position in a different way
## need to get the area of only then image
## the reference should be the biggest image 
## in the data set
## and we consider the image as a square


# In[13]:


def test_image(img, thresh=220, maxval=255, square=True):
    ## frayscale conversion
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
    ### invert the image
    ### now color is not necessary in this image
    retval, thresh_gray = cv2.threshold(gray, thresh=thresh, maxval=maxval, type=cv2.THRESH_BINARY_INV)
    ## finding the countour position in the image
    contours, hierarchy = cv2.findContours(thresh_gray,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    
    ## set with the initial image
    # Find object with the biggest bounding box
    mx = (0,0,0,0)      # biggest bounding box so far
    mx_area = 0
    for cont in contours:
        ## get the co ordinates
        x,y,w,h = cv2.boundingRect(cont)
        ## calculate the area
        area = w*h
        ## if the area is max
        if area > mx_area:
            ## then set the area
            mx = x,y,w,h
            mx_area = area
    x,y,w,h = mx
    
    ## then find the subset of the image
    crop = img[y:y+h, x:x+w]
    
    ## all image are square 
    ## so it will not cause a problem
    if square:
        pad_width = get_pad_width(crop, max(crop.shape))
        crop = np.pad(crop, pad_width=pad_width, mode='constant', constant_values=255)
    
    return crop


# In[14]:


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# In[15]:


plt.imshow(gray,cmap="gray")


# In[16]:


retval, thresh_gray = cv2.threshold(gray, thresh=220, maxval=225, type=cv2.THRESH_BINARY_INV)


# In[17]:


plt.imshow(thresh_gray,cmap="binary")


# In[18]:


contours, hierarchy = cv2.findContours(thresh_gray,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


# In[19]:


mx = (0,0,0,0)      # biggest bounding box so far
mx_area = 0
for cont in contours:
    ## get the co ordinates
    x,y,w,h = cv2.boundingRect(cont)
    ## calculate the area
    area = w*h
    ## if the area is max
    if area > mx_area:
        ## then set the area
        mx = x,y,w,h
        mx_area = area
x,y,w,h = mx

## then find the subset of the image
crop = img[y:y+h, x:x+w]


# In[20]:


plt.imshow(crop)


# In[21]:


pad_width = get_pad_width(crop, max(crop.shape))
crop = np.pad(crop, pad_width=pad_width, mode='constant', constant_values=255)


# In[22]:


plt.imshow(crop)


# In[23]:


plt.imshow(test_image(img))


# In[24]:


plt.imshow(test_image(img)/255)


# In[25]:



plt.imshow(test_image(img1))


# In[26]:


def crop_object(img, thresh=220, maxval=255, square=True):
    ## frayscale conversion
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
    ### invert the image
    ### now color is not necessary in this image
    retval, thresh_gray = cv2.threshold(gray, thresh=thresh, maxval=maxval, type=cv2.THRESH_BINARY_INV)
    ## finding the countour position in the image
    contours, hierarchy = cv2.findContours(thresh_gray,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    
    ## set with the initial image
    # Find object with the biggest bounding box
    mx = (0,0,0,0)      # biggest bounding box so far
    mx_area = 0
    for cont in contours:
        ## get the co ordinates
        x,y,w,h = cv2.boundingRect(cont)
        ## calculate the area
        area = w*h
        ## if the area is max
        if area > mx_area:
            ## then set the area
            mx = x,y,w,h
            mx_area = area
    x,y,w,h = mx
    
    ## then find the subset of the image
    crop = img[y:y+h, x:x+w]
    ## all image are square 
    ## so it will not cause a problem
    if square:
        pad_width = get_pad_width(crop, max(crop.shape))
        crop = np.pad(crop, pad_width=pad_width, mode='constant', constant_values=255)
    
    return crop


# In[27]:


## image shuffling
## and resizing
## converting
## and giving batch for neural net
## this gives a ran

def data_generator(filenames, y, batch_size=64, shape=(128, 128, 1), random_state=2019):
    y = y.copy()
    np.random.seed(random_state)
    indices = np.arange(len(filenames))
    
    while True:
        np.random.shuffle(indices)
        
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            size = len(batch_idx)
            
            batch_files = filenames[batch_idx]
            X_batch = np.zeros((size, *shape))
            y_batch = y[batch_idx]
            
            for i, file in enumerate(batch_files):
                img = cv2.imread(file)
                img = crop_object(img, thresh=220)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, shape[:2])
                ## replacing 
                X_batch[i, :, :, 0] = img / 255.
            ## this will create a generator
            ## return but not ending loop
            yield X_batch, [y_batch[:, i] for i in range(y_batch.shape[1])]


# In[28]:


def build_model(densenet):
    x_in = layers.Input(shape=(128, 128, 1))
    x = layers.Conv2D(3, (3, 3), padding='same')(x_in)
    x = densenet(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    
    ## there are three prediction head
    ## thats why using model api
    ## not sequentiual
    out_grapheme = layers.Dense(168, activation='softmax', name='grapheme')(x)
    out_vowel = layers.Dense(11, activation='softmax', name='vowel')(x)
    out_consonant = layers.Dense(7, activation='softmax', name='consonant')(x)
    
    model = Model(inputs=x_in, outputs=[out_grapheme, out_vowel, out_consonant])
    
    model.compile(
        optimizers.Adam(lr=0.0001), 
        metrics=['accuracy'], 
        loss='sparse_categorical_crossentropy'
    )
    
    return model


# In[29]:



densenet = DenseNet121(include_top=False, input_shape=(128, 128, 3))


# In[30]:


model = build_model(densenet)
model.summary()


# In[31]:


train_files, valid_files, y_train, y_valid = train_test_split(
    train.filename.values, 
    train[['grapheme_root','vowel_diacritic', 'consonant_diacritic']].values, 
    test_size=0.25, 
    random_state=2019
)


# In[32]:


batch_size = 128

train_gen = data_generator(train_files, y_train)
valid_gen = data_generator(valid_files, y_valid)

train_steps = round(len(train_files) / batch_size) + 1
valid_steps = round(len(valid_files) / batch_size) + 1


# In[33]:


## do not run this in your computer
## if you dont have any GPU
## keras model will save it untill the early stopping hit
callbacks = [keras.callbacks.ModelCheckpoint('model.h5', save_best_only=True)]

train_history = model.fit_generator(
    train_gen,
    steps_per_epoch=train_steps,
    epochs=20,
    validation_data=valid_gen,
    validation_steps=valid_steps,
    callbacks=callbacks
)


# In[34]:


plt.plot(train_history.history['val_grapheme_loss'])
plt.plot(train_history.history['val_vowel_loss'])
plt.plot(train_history.history['val_consonant_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()


# In[35]:


plt.plot(train_history.history['val_grapheme_accuracy'])

plt.plot(train_history.history['val_vowel_accuracy'])
plt.plot(train_history.history['val_consonant_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')


# In[36]:


def build_model1(densenet):
    x_in = layers.Input(shape=(128, 128, 1))
    x = layers.Conv2D(3, (3, 3), padding='same')(x_in)
    x = densenet(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    
    ## there are three prediction head
    ## thats why using model api
    ## not sequentiual
    out_grapheme = layers.Dense(168, activation='softmax', name='grapheme')(x)
    out_vowel = layers.Dense(11, activation='softmax', name='vowel')(x)
    out_consonant = layers.Dense(7, activation='softmax', name='consonant')(x)
    
    model = Model(inputs=x_in, outputs=[out_grapheme, out_vowel, out_consonant])
    
    model.compile(
        optimizers.SGD(lr=0.0001), 
        metrics=['accuracy'], 
        loss='sparse_categorical_crossentropy'
    )
    
    return model


# In[37]:


model1 = build_model1(densenet)
model1.summary()


# In[ ]:





# In[38]:


def build_model2(densenet):
    x_in = layers.Input(shape=(128, 128, 1))
    x = layers.Conv2D(3, (3, 3), padding='same')(x_in)
    x = densenet(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    
    ## there are three prediction head
    ## thats why using model api
    ## not sequentiual
    out_grapheme = layers.Dense(168, activation='softmax', name='grapheme')(x)
    out_vowel = layers.Dense(11, activation='softmax', name='vowel')(x)
    out_consonant = layers.Dense(7, activation='softmax', name='consonant')(x)
    
    model = Model(inputs=x_in, outputs=[out_grapheme, out_vowel, out_consonant])
    
    model.compile(
        optimizers.RMSprop(lr=0.0001), 
        metrics=['accuracy'], 
        loss='sparse_categorical_crossentropy'
    )
    
    return model


# In[39]:


model2 = build_model2(densenet)
model2.summary()


# In[40]:


train_files, valid_files, y_train, y_valid = train_test_split(
    train.filename.values, 
    train[['grapheme_root','vowel_diacritic', 'consonant_diacritic']].values, 
    test_size=0.25, 
    random_state=2019
)
batch_size = 128

train_gen = data_generator(train_files, y_train)
valid_gen = data_generator(valid_files, y_valid)

train_steps = round(len(train_files) / batch_size) + 1
valid_steps = round(len(valid_files) / batch_size) + 1


callbacks = [keras.callbacks.ModelCheckpoint('model.h5', save_best_only=True)]

train_history = model1.fit_generator(
    train_gen,
    steps_per_epoch=train_steps,
    epochs=20,
    validation_data=valid_gen,
    validation_steps=valid_steps,
    callbacks=callbacks
)


# In[41]:


plt.plot(train_history.history['val_grapheme_accuracy'])

plt.plot(train_history.history['val_vowel_accuracy'])
plt.plot(train_history.history['val_consonant_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')


# In[42]:


plt.plot(train_history.history['val_grapheme_loss'])
plt.plot(train_history.history['val_vowel_loss'])
plt.plot(train_history.history['val_consonant_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()


# In[43]:


del model
del model1


# In[44]:


train_files, valid_files, y_train, y_valid = train_test_split(
    train.filename.values, 
    train[['grapheme_root','vowel_diacritic', 'consonant_diacritic']].values, 
    test_size=0.25, 
    random_state=2019
)
batch_size = 128

train_gen = data_generator(train_files, y_train)
valid_gen = data_generator(valid_files, y_valid)

train_steps = round(len(train_files) / batch_size) + 1
valid_steps = round(len(valid_files) / batch_size) + 1


callbacks = [keras.callbacks.ModelCheckpoint('model.h5', save_best_only=True)]

train_history = model2.fit_generator(
    train_gen,
    steps_per_epoch=train_steps,
    epochs=20,
    validation_data=valid_gen,
    validation_steps=valid_steps,
    callbacks=callbacks
)


# In[45]:


plt.plot(train_history.history['val_grapheme_accuracy'])

plt.plot(train_history.history['val_vowel_accuracy'])
plt.plot(train_history.history['val_consonant_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')


# In[46]:


plt.plot(train_history.history['val_grapheme_loss'])
plt.plot(train_history.history['val_vowel_loss'])
plt.plot(train_history.history['val_consonant_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

