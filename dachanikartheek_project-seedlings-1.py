#!/usr/bin/env python
# coding: utf-8

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

# Any results you write to the current directory are saved as output.


# In[2]:


os.listdir('../input/train')


# In[3]:


import tensorflow as tf


# In[ ]:





# In[4]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')


# In[5]:


train_generator=train_datagen.flow_from_directory("../input/train/",batch_size=20,target_size=(256,256),
                                                  class_mode='categorical')


# In[6]:


from tensorflow.keras import layers
from tensorflow.keras import Model


# In[7]:


from tensorflow.keras.applications.inception_v3 import InceptionV3


# In[8]:


pre_trained_model=InceptionV3(input_shape=(256, 256, 3),
                             include_top= False,
                             weights= 'imagenet')


# In[9]:


for layer in pre_trained_model.layers:
    layer.trainable=False


# In[10]:


last_layer=pre_trained_model.get_layer('mixed7')
print('last layer output shape: ',last_layer.output_shape)
last_output=last_layer.output


# In[11]:


from tensorflow.keras.optimizers import RMSprop

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)                  
# Add a final sigmoid layer for classification
x = layers.Dense  (12, activation='softmax')(x)           

model = Model( pre_trained_model.input, x) 

model.compile(optimizer = RMSprop(lr=0.0001), 
              loss = 'categorical_crossentropy', 
              metrics = ['acc'])


# In[12]:


history=model.fit_generator(train_generator,steps_per_epoch=238,epochs=15)


# In[13]:


##model.save('modelp.h5')


# In[14]:


import matplotlib.pyplot as plt
acc = history.history['acc']
loss = history.history['loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.legend(loc=0)
plt.figure()


# In[15]:


import numpy as np
from PIL import Image
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt 
from PIL import Image 
import seaborn as sns
import pandas as pd 
import numpy as np 
import os 


# In[16]:


from PIL import Image
import numpy as np
from skimage import transform
def load(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (256, 256, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image
image = load('../input/test/00d090cde.png')
label=np.argmax(model.predict(image))


# In[17]:


if label==0:
    label_name='black grass'
elif label ==1:
    label_name='charlock'
elif label==2:
    label_name='cleavers'
elif label==3:
    label_name='chickweed'
elif label ==4:
    label_name='wheat'
elif label ==5:
    label_name='fat hen'
elif label ==6:
    label_name='silky bent'
elif label ==7:
    label_name='maize'
elif label ==8:
    label_name='scentless mayweed'
elif label ==9:
    label_name='shepherd purse'
elif label ==10:
    label_name='cranes bills'
elif label ==11:
    label_name='sugarbeet'    
     


# In[18]:


label_name

