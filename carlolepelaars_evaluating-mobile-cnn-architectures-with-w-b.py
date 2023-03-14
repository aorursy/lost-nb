#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Upgrade tensorflow and efficientnet and W&B
get_ipython().system('pip install --upgrade tensorflow')
get_ipython().system('pip install --upgrade efficientnet')
get_ipython().system('pip install --upgrade wandb')


# In[2]:


# Obfuscated WANDB API Key
from kaggle_secrets import UserSecretsClient
WANDB_KEY = UserSecretsClient().get_secret("WANDB_API_KEY")


# In[3]:


import os
import sys
import glob
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Flatten, ReLU
from tensorflow.keras.applications import MobileNet, MobileNetV2, NASNetMobile
from efficientnet.tfkeras import EfficientNetB0

# Path variables
BASE_PATH = "/kaggle/input/plant-pathology-2020-fgvc7/"
TRAIN_PATH = BASE_PATH + "train.csv"
TEST_PATH = BASE_PATH + "test.csv"
SUB_PATH = BASE_PATH + "sample_submission.csv"
IMG_PATH = BASE_PATH + "images/"

# Set seed for reproducability
seed = 1234
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

# Surpress scientific notation
np.set_printoptions(suppress=True)

# Global Variables
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 100
LABELS = ['healthy', 'multiple_diseases', 'rust', 'scab']
N_CLASSES = len(LABELS)


# In[4]:


# Import custom GhostNet architecture made by sunnyyeah
sys.path.append("../input/ghostnet/GhostNet-Keras-master/")
from ghostNet import GhostNet


# In[5]:


# Initialize Weights and Biases
import wandb
from wandb.keras import WandbCallback
wandb.login(key=WANDB_KEY);


# In[6]:


# Load annotations and labels
img_dir = glob.glob(f"{IMG_PATH}*.jpg")
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sub = pd.read_csv(SUB_PATH)

train['filename'] = IMG_PATH + train['image_id'] + ".jpg"
test['filename'] = IMG_PATH + test['image_id'] + ".jpg"


# In[7]:


# Split into train and validation
X_train, X_val, y_train, y_val = train_test_split(train['filename'], train[LABELS].values.astype(np.float32), test_size=0.15, random_state=seed) 
X_test = test['filename']


# In[8]:


def decode_image(filename, label=None):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    
    if label is None:
        return image
    else:
        return image, label
    
def decode_image_2(filename, label=None):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.mobilenet.preprocess_input(image)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    
    if label is None:
        return image
    else:
        return image, label


# In[9]:


def data_augment(image, label=None):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    if label is None:
        return image
    else:
        return image, label


# In[10]:


def build_model(backbone: tf.keras.Model, n_classes: int = 4) -> tf.keras.Model:
    """ 
    Initialize model with custom backbone 
    
    :param backbone: A tf.keras.Model object used as a backbone for the network
    :return: A tf.keras.Model object
    
    """
    model = tf.keras.Sequential()
    model.add(backbone)
    model.add(Flatten())
    model.add(Dense(256, activation=None))
    model.add(ReLU(max_value=6)) # ReLU6
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam',
                  metrics=['acc'])
    return model


# In[11]:


def build_lrfn(lr_start=0.00001, lr_max=0.00005, 
               lr_min=0.00001, lr_rampup_epochs=5, 
               lr_sustain_epochs=0, lr_exp_decay=.8):
    """ 
    Learning rate scheduler with warm-up and exponential decay 
    """
    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) *                 lr_exp_decay**(epoch - lr_rampup_epochs                                - lr_sustain_epochs) + lr_min
        return lr
    return lrfn


# In[12]:


AUTO = tf.data.experimental.AUTOTUNE

model_dict = {"GhostNet": GhostNet,
              "NasNetMobile": NASNetMobile,
              "Mobilenet": MobileNet, 
              "MobileNetV2": MobileNetV2,
              "EfficientNetB0": EfficientNetB0}


# In[13]:


metrics = []
for name, net in model_dict.items():
    wandb.init(project="mobile_architectures", name=name, 
                   notes="Mobile architectures review", reinit=True)
    # Slight changes for GhostNet and EfficientNet
    if name == "GhostNet":
        backbone = GhostNet((IMG_SIZE, IMG_SIZE, 3), 4, include_top=False).build(plot=False)
        lrfn = build_lrfn(lr_start=0.005, lr_max=0.01, lr_min=0.001)
    else: 
        if "EfficientNet" in name:
            weights = 'noisy-student'
        else:
            weights = 'imagenet'
            
        # Get backbone and learning rate scheduler
        backbone = net(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), weights=weights, pooling='avg')
        lrfn = build_lrfn()
    model = build_model(backbone)
    
    # Tensorflow wrapper for the learning rate schedule function
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)
    
    # Set image decoder
    if "MobileNet" in name or name == "GhostNet":
        image_decoder = decode_image_2
    else:
        image_decoder = decode_image
        
    # Initialize datasets
    train_dataset = (tf.data.Dataset
                     .from_tensor_slices((X_train, y_train))
                     .map(image_decoder, num_parallel_calls=AUTO)
                     .map(data_augment, num_parallel_calls=AUTO)
                     .shuffle(512)
                     .repeat()
                     .batch(BATCH_SIZE)
                     .prefetch(AUTO))

    valid_dataset = (tf.data.Dataset
                     .from_tensor_slices((X_val, y_val))
                     .map(image_decoder, num_parallel_calls=AUTO)
                     .batch(BATCH_SIZE)
                     .cache()
                     .prefetch(AUTO))
        
    # Train model
    hist = model.fit(train_dataset, validation_data=valid_dataset,
                     steps_per_epoch=y_train.shape[0] // BATCH_SIZE,
                     callbacks=[lr_schedule, WandbCallback(save_model=False)], 
                     epochs=EPOCHS, verbose=1)

    # Inference speed benchmarking on validation data
    start_time = time.time()
    y_pred = model.predict(valid_dataset)
    end_time = time.time()
    inf_speed = (end_time - start_time) / len(X_val)

    # Evaluate validation accuracy and log metrics
    val_acc = accuracy_score(y_val.argmax(axis=1), y_pred.argmax(axis=1))
    results = {"Validation Accuracy": val_acc, "Inference Speed": inf_speed, "Parameters": model.count_params()}
    metrics.append((name, results['Validation Accuracy'], results['Inference Speed'], results['Parameters']))
    wandb.log(results)
    wandb.join()


# In[14]:


eval_df = pd.DataFrame(metrics, columns=['Name', 'Validation Accuracy', 'Inference Speed', 'Parameters'])
eval_df


# In[15]:


test_dataset = (tf.data.Dataset
                .from_tensor_slices(X_test)
                .map(decode_image, num_parallel_calls=AUTO)
                .batch(BATCH_SIZE))

# Make final predictions using the EfficientNetB0 architecture
sub.loc[:, LABELS] = model.predict(test_dataset, verbose=1)
sub.to_csv('submission.csv', index=False)
sub.head()

