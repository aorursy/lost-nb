#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf

get_ipython().run_line_magic('matplotlib', 'inline')

# Any results you write to the current directory are saved as output.


# In[2]:


os.listdir('../input/')


# In[3]:


df = pd.read_csv('../input/train.csv')
df.head(10)


# In[4]:


os.listdir('../input/test/test')


# In[5]:


os.listdir('../input/train/train')


# In[6]:


data_dir = '../input/train/train/'
filename = df['id'][0]
path = os.path.join(data_dir, filename)
path

test_dir = '../input/test/test/'


# In[7]:


image_pil = Image.open(path)
image_pil


# In[8]:


image = np.array(image_pil)
plt.imshow(image)
plt.show()


# In[9]:


has_cactus = df['has_cactus'][0]
has_cactus


# In[10]:


image = np.array(image_pil)
plt.title(has_cactus)
plt.imshow(image)
plt.show()


# In[11]:


np.mean(df['has_cactus']) # cactus가 포함될 비율


# In[12]:


np.sum(df['has_cactus']), len(df['has_cactus'])


# In[13]:


image.shape


# In[14]:


np.min(image), np.max(image)


# In[15]:



#def get_label(path):
#    if df['id'] == path.split('/')[-1]
    

def get_data(pathtuple):
    path, label = pathtuple
    image_pil = Image.open(path)
    image = np.array(image_pil)
    image = image/255.0
    label=tf.keras.utils.to_categorical(label,2)

    #return image, label
    return image.astype(np.float32), label.astype(np.float32)

    
    


# In[16]:


get_ipython().run_line_magic('pinfo', 'tf.keras.utils.to_categorical')


# In[17]:


heights = []
widths = []
train_arr = []
test_arr = []

train_filenames = []
index = 0

for filename in df['id']:
    path = os.path.join(data_dir, filename)
    train_filenames.append((path, df['has_cactus'][index]))
    index = index + 1
    
#trainimage, trainlabel=get_data(train_filenames[0])
#plt.imshow(trainimage)
#print(trainlabel)
    
for testfilename in os.listdir(test_dir):
    #print(testfilename)
    path = os.path.join(test_dir, testfilename)
    image_pil = Image.open(path)
    image = np.array(image_pil)
    image = image/255.0
    test_arr.append(image.astype(np.float32))
    
test_data = np.array(test_arr)


# In[18]:


# batch dataset

#batch_paths = train_filenames[:8]

def make_batch(batch_paths):

    batch_images = []
    batch_labels = []

    for pathtuple in batch_paths:
        path, label = pathtuple
        image, label = get_data(pathtuple)
        batch_images.append(image)
        batch_labels.append(label)

    batch_images = np.array(batch_images)
    batch_labels = np.array(batch_labels)

    return batch_images, batch_labels

#images, labels = make_batch(batch_paths)
#images.shape, labels.shape 


# In[19]:


batch_size = 32


# In[20]:


def data_gen(data_paths, is_training=True):
    global_step = 0
    steps_per_epoch = len(data_paths) // batch_size
    while True:
        step = global_step % steps_per_epoch
        if step == 0:
            np.random.shuffle(data_paths)
        images, labels = make_batch(data_paths[step*batch_size:(step+1)*batch_size])
        global_step += 1
        yield images, labels


# In[21]:


#generator = data_gen(train_paths)
generator = data_gen(train_filenames)

for i, (img, lbl) in enumerate(generator):
    if i < 5:
        plt.title(i + lbl[0])
        plt.imshow(img[0])
        plt.show()
        #print(lbl)
    else:
        break
        


# In[22]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import time

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import clear_output

import tensorflow as tf
from tensorflow.keras import layers
tf.enable_eager_execution()

os.environ["CUDA_VISIBLE_DEVICES"]="0"


# In[23]:


batch_size = 32
num_epochs = 20
learning_rate = 0.001

num_classes = 2
input_shape = (32, 32, 3)


# In[24]:


# VGG16
inputs = layers.Input(input_shape)
net = layers.Conv2D(64, (3, 3), padding='same')(inputs)
net = layers.Conv2D(64, (3, 3), padding='same')(net)
net = layers.Conv2D(64, (3, 3), padding='same')(net)
net = layers.BatchNormalization()(net)
net = layers.Activation('relu')(net)
net = layers.MaxPooling2D(pool_size=(2, 2))(net)

net = layers.Conv2D(128, (3, 3), padding='same')(net)
net = layers.Conv2D(128, (3, 3), padding='same')(net)
net = layers.Conv2D(128, (3, 3), padding='same')(net)
net = layers.BatchNormalization()(net)
net = layers.Activation('relu')(net)
net = layers.MaxPooling2D(pool_size=(2, 2))(net)
net = layers.Dropout(0.25)(net)

net = layers.Conv2D(256, (3, 3), padding='same')(net)
net = layers.Conv2D(256, (3, 3), padding='same')(net)
net = layers.Conv2D(256, (3, 3), padding='same')(net)
net = layers.BatchNormalization()(net)
net = layers.Activation('relu')(net)
net = layers.MaxPooling2D(pool_size=(2, 2))(net)
net = layers.Dropout(0.25)(net)

net = layers.Conv2D(512, (3, 3), padding='same')(net)
net = layers.Conv2D(512, (3, 3), padding='same')(net)
net = layers.Conv2D(512, (3, 3), padding='same')(net)
net = layers.BatchNormalization()(net)
net = layers.Activation('relu')(net)
net = layers.MaxPooling2D(pool_size=(2, 2))(net)
net = layers.Dropout(0.25)(net)

net = layers.Conv2D(512, (3, 3), padding='same')(net)
net = layers.Conv2D(512, (3, 3), padding='same')(net)
net = layers.Conv2D(512, (3, 3), padding='same')(net)
net = layers.BatchNormalization()(net)
net = layers.Activation('relu')(net)
net = layers.MaxPooling2D(pool_size=(2, 2))(net)
net = layers.Dropout(0.25)(net)

# net = layers.GlobalAveragePooling2D()(net)
net = layers.Flatten()(net)
net = layers.Dense(512)(net)
net = layers.Activation('relu')(net)
net = layers.Dropout(0.5)(net)
net = layers.Dense(num_classes)(net)
net = layers.Activation('softmax')(net)

model = tf.keras.Model(inputs=inputs, outputs=net)


# In[25]:


model.compile(loss='categorical_crossentropy', 
             optimizer=tf.keras.optimizers.Adam(learning_rate),
             metrics=['accuracy'])


# In[26]:


#callbacks = [
#    tf.keras.callbacks.TensorBoard(log_dir=os.path.join(os.getcwd(),'logs'))
#]


# In[27]:


get_ipython().run_line_magic('pinfo', 'model.fit_generator')


# In[28]:


steps_per_epoch = len(train_filenames) // batch_size

history=model.fit_generator(generator=data_gen(train_filenames),
                    steps_per_epoch=steps_per_epoch,
                    epochs=num_epochs,
                    verbose=1)
                    #callbacks=callbacks)


# In[29]:


print(history.history.keys())


# In[30]:


# summarize history for accuracy
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()


# In[ ]:





# In[31]:


test_predictions = []

for i in range(test_data.shape[0]):
#for i in range(1):
    predictions = model.predict(np.expand_dims(test_data[i],0))
    test_predictions.append(np.argmax(tf.squeeze(predictions)))


# In[32]:


#print(test_predictions[0])
#len(test_predictions)


# In[33]:


test_filenames=os.listdir(test_dir)

test_df = pd.DataFrame({'id': test_filenames, 'has_cactus': test_predictions },columns=['id','has_cactus'])

test_df.to_csv('test_submission.csv', index=False)


# In[34]:


#test_batches_per_epoch = len(test_paths) // batch_size
#model.evaluate_generator(data_gen(test_paths, False),
#                        steps = test_batches_per_epoch,
#                        verbose=1)


# In[35]:


#model.predict(np.expand_dims(image,0))


# In[36]:


#model.save(model.name+'_cactus.h5')

#load model
#model = tf.keras.models.load_model(model.name + '_cactus.h5')


# In[ ]:





# In[ ]:





# In[37]:


#!tensorboard --logdir=./logs


# In[38]:



#t = [('a',1),('b',2),('c',3)]
#np.random.shuffle(t)
#print(t)


# In[39]:


#(df['id']==path.split('/')[-1])==1


# In[40]:


#from tqdm import tqdm_notebook


# In[ ]:





# In[ ]:





# In[ ]:




