#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the libraries

import json
import math
import os

import cv2
from PIL import Image
import numpy as np
from keras import layers
from keras.layers import Flatten
from keras.applications import DenseNet121
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
import scipy
import tensorflow as tf
from tqdm import tqdm
import glob

IMG_SIZE = 224
NUM_CLASSES = 5
SEED = 77
TRAIN_NUM = -1


# In[2]:


import os
print(os.listdir("../input/efficientnet/efficientnet-master/efficientnet-master/efficientnet"))
import sys
sys.path.append(os.path.abspath('../input/efficientnet/efficientnet-master/efficientnet-master/'))
from efficientnet import EfficientNetB5


# In[3]:


import os
print(os.listdir("../input"))


# In[4]:


# Loading the dataframe

train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv') 
test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')
print(train_df.shape)
print(test_df.shape)
train_df.head()


# In[5]:


# Seeing the number of labels we have for different classes in our distribution

train_df['diagnosis'].value_counts()


# In[6]:


# Plotting the distribution of different labels we have

size_NoDR = 0
size_Mild = 0
size_Moderate = 0
size_Severe = 0
size_ProliferativeDR = 0
for i in range(len(train_df)):
    if train_df['diagnosis'][i] == 0:
        size_NoDR = size_NoDR + 1
    if train_df['diagnosis'][i] == 1:
        size_Mild = size_Mild + 1
    if train_df['diagnosis'][i] == 2:
        size_Moderate = size_Moderate + 1
    if train_df['diagnosis'][i] == 3:
        size_Severe = size_Severe + 1 
    if train_df['diagnosis'][i] == 4:
        size_ProliferativeDR = size_ProliferativeDR + 1
explode = [0.1, 0, 0, 0, 0]
labels = 'No DR', 'Mild','Moderate','Severe','Proliferative DR'        
sizes = [size_NoDR, size_Mild, size_Moderate, size_Severe, size_ProliferativeDR]        
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode = explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# In[7]:


# Let us display some sample images to see how our input images are

def display_samples(df, columns = 4, rows = 3):
    fig = plt.figure(figsize = (5 * columns, 4 * rows))

    for i in range(columns * rows):
        image_path = df.loc[i,'id_code']
        image_id = df.loc[i,'diagnosis']
        img = cv2.imread(f'../input/aptos2019-blindness-detection/train_images/{image_path}.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig.add_subplot(rows, columns, i+1)
        plt.title(image_id)
        plt.imshow(img)
    
    plt.tight_layout()

display_samples(train_df)


# In[8]:


train_y = train_df['diagnosis']
train_y.shape


# In[9]:


def crop_image_from_gray(img, tol = 7):
    if img.ndim == 2:
        mask = img>tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1 = img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2 = img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3 = img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img


# In[10]:


def circle_crop_v2(img):

        #img = cv2.imread(img)
        #img = crop_image_from_gray(img)

        height, width, depth = img.shape
        largest_side = np.max((height, width))
        img = cv2.resize(img, (largest_side, largest_side))

        height, width, depth = img.shape

        x = int(width / 2)
        y = int(height / 2)
        r = np.amin((x, y))

        circle_img = np.zeros((height, width), np.uint8)
        cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
        img = cv2.bitwise_and(img, img, mask=circle_img)
        #img = crop_image_from_gray(img)

        return img


# In[11]:


def ben_color2(image_path, sigmaX = 10, scale = 270):
   #image = cv2.imread(image_path)   
   img = cv2.imread(image_path)
   #gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
   #clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(10,10))
   #gray_img = clahe.apply(gray_img)
   #image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
   bgr = cv2.imread(image_path)

   lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
   
   lab_planes = cv2.split(lab)

   clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(10,10))

   lab_planes[0] = clahe.apply(lab_planes[0])

   lab = cv2.merge(lab_planes)

   image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

   x = image[image.shape[0]//2 ,: ,:].sum(1)
   r = (x > x.mean()/10).sum()//2
   s = scale * 1.0/ r
   image = cv2.resize(image,(0,0), fx = s, fy = s)
   #image = crop_image_from_gray(image)
   #image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
   image = cv2.addWeighted (image,4, cv2.GaussianBlur(image, (0,0) ,sigmaX), -4, 128)
   image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
   image = circle_crop_v2(image)
   #image = cv2.fastNlMeansDenoisingColored(image,None,20,10,7,21)
   return image


# In[12]:


# Defining the function to resize the images

def preprocess_image(image_path, desired_size = 300):
    image = ben_color2(image_path,sigmaX= 10)
    return image


# In[13]:


# Let us display some processed sample images to see how our input images are

def display_samples(df, columns = 4, rows = 3):
    fig = plt.figure(figsize = (5 * columns, 4 * rows))

    for i in range(columns * rows):
        image_path = df.loc[i,'id_code']
        image_id = df.loc[i,'diagnosis']
        img = cv2.imread(f'../input/aptos2019-blindness-detection/train_images/{image_path}.png')
        #img = load_ben_color(f'../input/aptos2019-blindness-detection/train_images/{image_path}.png', 10)
        img = ben_color2(f'../input/aptos2019-blindness-detection/train_images/{image_path}.png', 10)
       
        fig.add_subplot(rows, columns, i + 1)
        plt.title(image_id)
        plt.imshow(img)
    
    plt.tight_layout()

display_samples(train_df)


# In[14]:


#### Processing the training images

N = train_df.shape[0]
x_train = np.empty((N, 224, 224, 3), dtype = np.uint8)

for i, image_id in enumerate(tqdm(train_df['id_code'])):
    x_train[i, :, :, :] = preprocess_image(
        f'../input/aptos2019-blindness-detection/train_images/{image_id}.png'
    )


# In[15]:


# Resizing the test images

N = test_df.shape[0]
x_test = np.empty((N, 224, 224, 3), dtype = np.uint8)

for i, image_id in enumerate(tqdm(test_df['id_code'])):
    x_test[i, :, :, :] = preprocess_image(
        f'../input/aptos2019-blindness-detection/test_images/{image_id}.png'
    )


# In[16]:


# Forming the target labels

y_train = pd.get_dummies(train_df['diagnosis']).values

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)


# In[17]:


# Converting our target labels into multi-labels

y_train_multi = np.empty(y_train.shape, dtype = y_train.dtype)
y_train_multi[:, 4] = y_train[:, 4]

for i in range(3, -1, -1):
    y_train_multi[:, i] = np.logical_or(y_train[:, i], y_train_multi[:, i+1])

print("Original y_train:", y_train.sum(axis = 0))
print("Multilabel version:", y_train_multi.sum(axis = 0))


# In[18]:


from sklearn.utils import class_weight
wts = class_weight.compute_class_weight('balanced', np.unique(train_y),train_y)


# In[19]:


# Splitting our data into training and cross-validations sets

x_train_NN, x_val_NN, y_train_NN, y_val_NN = train_test_split(x_train, y_train_multi, test_size = 0.15, random_state = 77) 


# In[20]:


# Defining the data generator function

BATCH_SIZE = 32

def create_datagen():
            return ImageDataGenerator(
               zoom_range = 0.15,  # set range for random zoom
               rotation_range = 360,
               # set mode for filling points outside the input boundaries
               fill_mode = 'constant',
               cval = 0, 
               horizontal_flip = True,  # randomly flip images
               vertical_flip = True,  # randomly flip images
               #rescale = 1 / 256
              )


# In[21]:


# Using original generator

data_generator = create_datagen().flow(x_train_NN, y_train_NN, batch_size = BATCH_SIZE, seed = 77)


# In[22]:



import keras
class RAdam(keras.optimizers.Optimizer):
    """RAdam optimizer.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        weight_decay: float >= 0. Weight decay for each param.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".
        total_steps: int >= 0. Total number of training steps. Enable warmup by setting a positive value.
        warmup_proportion: 0 < warmup_proportion < 1. The proportion of increasing steps.
        min_lr: float >= 0. Minimum learning rate after warmup.
    # References
        - [Adam - A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980v8)
        - [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ)
        - [On The Variance Of The Adaptive Learning Rate And Beyond](https://arxiv.org/pdf/1908.03265v1.pdf)
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., weight_decay=0., amsgrad=False,
                 total_steps=0, warmup_proportion=0.1, min_lr=0., **kwargs):
        super(RAdam, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
            self.weight_decay = K.variable(weight_decay, name='weight_decay')
            self.total_steps = K.variable(total_steps, name='total_steps')
            self.warmup_proportion = K.variable(warmup_proportion, name='warmup_proportion')
            self.min_lr = K.variable(lr, name='min_lr')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.initial_weight_decay = weight_decay
        self.initial_total_steps = total_steps
        self.amsgrad = amsgrad

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr

        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations, K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1

        if self.initial_total_steps > 0:
            warmup_steps = self.total_steps * self.warmup_proportion
            decay_steps = self.total_steps - warmup_steps
            lr = K.switch(
                t <= warmup_steps,
                lr * (t / warmup_steps),
                lr * (1.0 - K.minimum(t, decay_steps) / decay_steps),
            )

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name='m_' + str(i)) for (i, p) in enumerate(params)]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name='v_' + str(i)) for (i, p) in enumerate(params)]

        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name='vhat_' + str(i)) for (i, p) in enumerate(params)]
        else:
            vhats = [K.zeros(1, name='vhat_' + str(i)) for i in range(len(params))]

        self.weights = [self.iterations] + ms + vs + vhats

        beta_1_t = K.pow(self.beta_1, t)
        beta_2_t = K.pow(self.beta_2, t)

        sma_inf = 2.0 / (1.0 - self.beta_2) - 1.0
        sma_t = sma_inf - 2.0 * t * beta_2_t / (1.0 - beta_2_t)

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)

            m_corr_t = m_t / (1.0 - beta_1_t)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                v_corr_t = K.sqrt(vhat_t / (1.0 - beta_2_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                v_corr_t = K.sqrt(v_t / (1.0 - beta_2_t) + self.epsilon)

            r_t = K.sqrt((sma_t - 4.0) / (sma_inf - 4.0) *
                         (sma_t - 2.0) / (sma_inf - 2.0) *
                         sma_inf / sma_t)

            p_t = K.switch(sma_t > 5, r_t * m_corr_t / v_corr_t, m_corr_t)

            if self.initial_weight_decay > 0:
                p_t += self.weight_decay * p

            p_t = p - lr * p_t

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {
            'lr': float(K.get_value(self.lr)),
            'beta_1': float(K.get_value(self.beta_1)),
            'beta_2': float(K.get_value(self.beta_2)),
            'decay': float(K.get_value(self.decay)),
            'weight_decay': float(K.get_value(self.weight_decay)),
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad,
            'total_steps': float(K.get_value(self.total_steps)),
            'warmup_proportion': float(K.get_value(self.warmup_proportion)),
            'min_lr': float(K.get_value(self.min_lr)),
        }
        base_config = super(RAdam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# In[23]:


from __future__ import absolute_import
import random
import numpy as np
from keras.layers import *
import tensorflow as tf

class FractionalPooling2D(Layer):
	def __init__(self, pool_ratio = None, pseudo_random = True, overlap = False, name ='FractionPooling2D', **kwargs):
		self.pool_ratio = pool_ratio
		self.input_spec = [InputSpec(ndim=4)]
		self.pseudo_random = pseudo_random
		self.overlap = overlap
		self.name = name
		super(FractionalPooling2D, self).__init__(**kwargs)
		
	def call(self, input):
		[batch_tensor,row_pooling,col_pooling] = tf.nn.fractional_max_pool(input, pooling_ratio = self.pool_ratio, pseudo_random = self.pseudo_random, overlapping = self.overlap, seed2 = 0, seed = 0)
		return(batch_tensor)
		
	def compute_output_shape(self, input_shape):
	
		if(K.image_dim_ordering() == 'channels_last' or K.image_dim_ordering() == 'tf'):
			if(input_shape[0] != None):
				batch_size = int(input_shape[0]/self.pool_ratio[0])
			else:
				batch_size = input_shape[0]
			width = int(input_shape[1]/self.pool_ratio[1])
			height = int(input_shape[2]/self.pool_ratio[2])
			channels = int(input_shape[3]/self.pool_ratio[3])
			return(batch_size, width, height, channels)
			
		elif(K.image_dim_ordering() == 'channels_first' or K.image_dim_ordering() == 'th'):
			if(input_shape[0] != None):
				batch_size = int(input_shape[0]/self.pool_ratio[0])
			else:
				batch_size = input_shape[0]
			channels = int(input_shape[1]/self.pool_ratio[1])
			width = int(input_shape[2]/self.pool_ratio[2])
			height = int(input_shape[3]/self.pool_ratio[3])
			return(batch_size, channels, width, height)
		
	def get_config(self):
		config = {'pooling_ratio': self.pool_ratio, 'pseudo_random': self.pseudo_random, 'overlap': self.overlap, 'name':self.name}
		base_config = super(FractionalPooling2D, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
		
	def build(self, input_shape):
		self.input_spec = [InputSpec(shape=input_shape)]


# In[24]:


from efficientnet import EfficientNetB5

effnet = EfficientNetB5(
    weights= None, 
    include_top=False,
    input_shape=(224,224,3)
)


# In[25]:


def build_model2():
    model = Sequential()
    model.add(effnet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(2048, activation = 'relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1024, activation = 'relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(512, activation = 'relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(256, activation = 'relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(128, activation = 'relu'))
    model.add(layers.Dense(5, activation = 'sigmoid'))
    effnet.load_weights('../input/efficientnet-keras-weights-b0b5/efficientnet-b5_imagenet_1000_notop.h5')
    model.compile(
        loss = 'binary_crossentropy',
        optimizer = RAdam(lr=0.0005),
        metrics = ['accuracy']
    )
    
    return model
model2 = build_model2()


# In[26]:


model2.summary()


# In[27]:


# Creating the Metrics class

class Metrics2(Callback):
    def on_train_begin(self, logs = {}):
        self.val_kappas = []

    def on_epoch_end(self, epoch, logs = {}):
        X_val, y_val = self.validation_data[:2]
        y_val = y_val.sum(axis = 1) - 1
        
        y_pred = self.model.predict(X_val) > 0.5
        y_pred = y_pred.astype(int).sum(axis = 1) - 1

        _val_kappa = cohen_kappa_score(
            y_val,
            y_pred, 
            weights = 'quadratic'
        )

        self.val_kappas.append(_val_kappa)

        print(f"val_kappa: {_val_kappa:.4f}")
        
        if _val_kappa == max(self.val_kappas):
            print("Validation Kappa has improved. Saving model.")
            self.model.save('model2.h5')

        return


# In[28]:


# Training our model

kappa_metrics2 = Metrics2()

history2 = model2.fit_generator(
    data_generator,
    steps_per_epoch = x_train_NN.shape[0] / BATCH_SIZE,
    epochs = 25,
    validation_data = (x_val_NN, y_val_NN),
    callbacks = [kappa_metrics2],
    class_weight = wts,
    validation_steps = x_val_NN.shape[0] / BATCH_SIZE
)


# In[29]:


model2.load_weights('model2.h5')


# In[30]:


# Plotting the graph to show our training and cross-validation loss

with open('history2.json', 'w') as f:
    json.dump(history2.history, f)

history_df = pd.DataFrame(history2.history)
history_df[['loss', 'val_loss']].plot()
history_df[['acc', 'val_acc']].plot()


# In[31]:


predict = model2.predict(x_val_NN)


# In[32]:


# Plotting the trend for our kappa values

plt.plot(kappa_metrics2.val_kappas)


# In[33]:


model2.load_weights('model2.h5')
#y_val_pred_lol = model2.predict(x_val_NN)

def compute_score_inv(threshold):
    y1 = predict > 0.5
    y1 = y1.astype(int).sum(axis=1) - 1
    y2 = y_val_NN.sum(axis=1) - 1
    score = cohen_kappa_score(y1, y2, weights='quadratic')
    
    return 1 - score

simplex = scipy.optimize.minimize(
    compute_score_inv, 0.5, method='nelder-mead'
)

best_threshold = simplex['x'][0]


# In[34]:


y1 = predict > 0.5
y1 = y1.astype(int).sum(axis=1) - 1
y2 = y_val_NN.sum(axis=1) - 1
score = cohen_kappa_score(y1, y2, weights='quadratic')
print(score)


# In[35]:


# Finding the predictions:

y_test = model2.predict(x_test) > 0.5
y_test = y_test.astype(int).sum(axis=1) - 1

test_df['diagnosis'] = y_test
test_df.to_csv('submission.csv',index=False)

