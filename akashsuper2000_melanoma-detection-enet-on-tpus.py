#!/usr/bin/env python
# coding: utf-8



get_ipython().system('pip install -q efficientnet')

import os
import re
import numpy as np
import pandas as pd
import random
import math
import time
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold
import tensorflow as tf
from kaggle_datasets import KaggleDatasets
import efficientnet.tfkeras as efn
import dill
from tensorflow.keras import backend as K

import warnings
warnings.filterwarnings('ignore')




# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)




# For tf.dataset
AUTO = tf.data.experimental.AUTOTUNE

# Configuration
EPOCHS = 25
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
AUG_BATCH = BATCH_SIZE

# Image size
IMAGE_SIZE = [256, 256]

# Seed
SEED = 123

# Learning rate
LR = 0.0005

# Data access
if(IMAGE_SIZE[0] == 256):
    print('Using size 256')
    GCS_PATH = KaggleDatasets().get_gcs_path('melanoma-256x256')
elif(IMAGE_SIZE[0] == 384):
    print('Using size 384')
    GCS_PATH = KaggleDatasets().get_gcs_path('melanoma-384x384')
elif(IMAGE_SIZE[0] == 512):
    print('Using size 512')
    GCS_PATH = KaggleDatasets().get_gcs_path('melanoma-512x512')
elif(IMAGE_SIZE[0] == 768):
    print('Using size 768')
    GCS_PATH = KaggleDatasets().get_gcs_path('melanoma-768x768')
elif(IMAGE_SIZE[0] == 1024):
    print('Using size 1024')
    GCS_PATH = KaggleDatasets().get_gcs_path('siim-isic-melanoma-classification')
    
# training filenames directory
TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/train*.tfrec')
# test filenames directory
TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/test*.tfrec')

if(IMAGE_SIZE[0] == 1024):
    # training filenames directory
    TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/tfrecords/train*.tfrec')
    # test filenames directory
    TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/tfrecords/test*.tfrec')

# submission file
SUB = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')




def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    # returns 3x3 transformmatrix which transforms indicies
        
    # CONVERT DEGREES TO RADIANS
    rotation = math.pi * rotation / 180.
    shear = math.pi * shear / 180.
    
    # ROTATION MATRIX
    c1 = tf.math.cos(rotation)
    s1 = tf.math.sin(rotation)
    one = tf.constant([1],dtype='float32')
    zero = tf.constant([0],dtype='float32')
    rotation_matrix = tf.reshape( tf.concat([c1,s1,zero, -s1,c1,zero, zero,zero,one],axis=0),[3,3] )
        
    # SHEAR MATRIX
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)
    shear_matrix = tf.reshape( tf.concat([one,s2,zero, zero,c2,zero, zero,zero,one],axis=0),[3,3] )    
    
    # ZOOM MATRIX
    zoom_matrix = tf.reshape( tf.concat([one/height_zoom,zero,zero, zero,one/width_zoom,zero, zero,zero,one],axis=0),[3,3] )
    
    # SHIFT MATRIX
    shift_matrix = tf.reshape( tf.concat([one,zero,height_shift, zero,one,width_shift, zero,zero,one],axis=0),[3,3] )
    
    return K.dot(K.dot(rotation_matrix, shear_matrix), K.dot(zoom_matrix, shift_matrix))

def transform(image, label):
    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    # output - image randomly rotated, sheared, zoomed, and shifted
    DIM = IMAGE_SIZE[0]
    XDIM = DIM%2 #fix for size 331
    
    tmp = random.uniform(0, 1)
    if 0 < tmp <= 0.1:
        rot = 15.0 * tf.random.normal([1],dtype='float32')
    elif 0.1 < tmp <= 0.2:
        rot = 30.0 * tf.random.normal([1],dtype='float32')
    elif 0.2 < tmp <= 0.3:
        rot = 45.0 * tf.random.normal([1],dtype='float32')
    elif 0.3 < tmp <= 0.4:
        rot = 60.0 * tf.random.normal([1],dtype='float32')
    elif 0.4 < tmp <= 0.5:
        rot = 75.0 * tf.random.normal([1],dtype='float32')
    elif 0.5 < tmp <= 0.6:
        rot = 90.0 * tf.random.normal([1],dtype='float32')
    elif 0.6 < tmp <= 0.7:
        rot = 110.0 * tf.random.normal([1],dtype='float32')
    elif 0.7 < tmp <= 0.8:
        rot = 130.0 * tf.random.normal([1],dtype='float32')
    elif 0.8 < tmp <= 0.9:
        rot = 150.0 * tf.random.normal([1],dtype='float32')
    elif 0.9 < tmp <= 1.0:
        rot = 180.0 * tf.random.normal([1],dtype='float32')
        
    shr = 5. * tf.random.normal([1],dtype='float32') 
    h_zoom = 1.0 + tf.random.normal([1],dtype='float32')/10.
    w_zoom = 1.0 + tf.random.normal([1],dtype='float32')/10.
    h_shift = 16. * tf.random.normal([1],dtype='float32') 
    w_shift = 16. * tf.random.normal([1],dtype='float32') 
  
    # GET TRANSFORMATION MATRIX
    m = get_mat(rot,shr,h_zoom,w_zoom,h_shift,w_shift) 

    # LIST DESTINATION PIXEL INDICES
    x = tf.repeat( tf.range(DIM//2,-DIM//2,-1), DIM )
    y = tf.tile( tf.range(-DIM//2,DIM//2),[DIM] )
    z = tf.ones([DIM*DIM],dtype='int32')
    idx = tf.stack( [x,y,z] )
    
    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = K.dot(m,tf.cast(idx,dtype='float32'))
    idx2 = K.cast(idx2,dtype='int32')
    idx2 = K.clip(idx2,-DIM//2+XDIM+1,DIM//2)
    
    # FIND ORIGIN PIXEL VALUES           
    idx3 = tf.stack( [DIM//2-idx2[0,], DIM//2-1+idx2[1,]] )
    d = tf.gather_nd(image['inp1'],tf.transpose(idx3))
        
    return {'inp1': tf.reshape(d,[DIM,DIM,3]), 'inp2': image['inp2']}, label

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)

# function to decode our images (normalize and reshape)
def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    # convert image to floats in [0, 1] range
    image = tf.cast(image, tf.float32) / 255.0 
    # explicit size needed for TPU
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image

# this function parse our images and also get the target variable
def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        # tf.string means bytestring
        "image": tf.io.FixedLenFeature([], tf.string), 
        # shape [] means single element
        "target": tf.io.FixedLenFeature([], tf.int64),
        # meta features
        "age_approx": tf.io.FixedLenFeature([], tf.int64),
        "sex": tf.io.FixedLenFeature([], tf.int64),
        "anatom_site_general_challenge": tf.io.FixedLenFeature([], tf.int64)
        
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    label = tf.cast(example['target'], tf.float32)
    # meta features
    data = {}
    data['age_approx'] = tf.cast(example['age_approx'], tf.int32)
    data['sex'] = tf.cast(example['sex'], tf.int32)
    data['anatom_site_general_challenge'] = tf.cast(tf.one_hot(example['anatom_site_general_challenge'], 7), tf.int32)
    # returns a dataset of (image, label, data)
    return image, label, data

# this function parse our image and also get our image_name (id) to perform predictions
def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        # tf.string means bytestring
        "image": tf.io.FixedLenFeature([], tf.string), 
        # shape [] means single element
        "image_name": tf.io.FixedLenFeature([], tf.string),
        # meta features
        "age_approx": tf.io.FixedLenFeature([], tf.int64),
        "sex": tf.io.FixedLenFeature([], tf.int64),
        "anatom_site_general_challenge": tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    image_name = example['image_name']
    # meta features
    data = {}
    data['age_approx'] = tf.cast(example['age_approx'], tf.int32)
    data['sex'] = tf.cast(example['sex'], tf.int32)
    data['anatom_site_general_challenge'] = tf.cast(tf.one_hot(example['anatom_site_general_challenge'], 7), tf.int32)
    # returns a dataset of (image, key, data)
    return image, image_name, data
    
def load_dataset(filenames, labeled = True, ordered = False):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # Diregarding data order. Order does not matter since we will be shuffling the data anyway
    
    ignore_order = tf.data.Options()
    if not ordered:
        # disable order, increase speed
        ignore_order.experimental_deterministic = False 
        
    # automatically interleaves reads from multiple files
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads = AUTO)
    # use data as soon as it streams in, rather than in its original order
    dataset = dataset.with_options(ignore_order)
    # returns a dataset of (image, label) pairs if labeled = True or (image, id) pair if labeld = False
    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls = AUTO) 
    return dataset

# function for training and validation dataset
def setup_input1(image, label, data):
    
    # get anatom site general challenge vectors
    anatom = [tf.cast(data['anatom_site_general_challenge'][i], dtype = tf.float32) for i in range(7)]
    
    tab_data = [tf.cast(data[tfeat], dtype = tf.float32) for tfeat in ['age_approx', 'sex']]
    
    tabular = tf.stack(tab_data + anatom)
    
    return {'inp1': image, 'inp2':  tabular}, label

# function for the test set
def setup_input2(image, image_name, data):
    
    # get anatom site general challenge vectors
    anatom = [tf.cast(data['anatom_site_general_challenge'][i], dtype = tf.float32) for i in range(7)]
    
    tab_data = [tf.cast(data[tfeat], dtype = tf.float32) for tfeat in ['age_approx', 'sex']]
    
    tabular = tf.stack(tab_data + anatom)
    
    return {'inp1': image, 'inp2':  tabular}, image_name

# function for the validation (image name)
def setup_input3(image, image_name, target, data):
    
    # get anatom site general challenge vectors
    anatom = [tf.cast(data['anatom_site_general_challenge'][i], dtype = tf.float32) for i in range(7)]
    
    tab_data = [tf.cast(data[tfeat], dtype = tf.float32) for tfeat in ['age_approx', 'sex']]
    
    tabular = tf.stack(tab_data + anatom)
    
    return {'inp1': image, 'inp2':  tabular}, image_name, target

def data_augment(data, label):
    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement 
    # in the next function (below), this happens essentially for free on TPU. 
    # Data pipeline code is executed on the "CPU" part
    # of the TPU while the TPU itself is computing gradients.
    data['inp1'] = tf.image.random_flip_left_right(data['inp1'])
    data['inp1'] = tf.image.random_flip_up_down(data['inp1'])
    data['inp1'] = tf.image.random_hue(data['inp1'], 0.01)
    data['inp1'] = tf.image.random_saturation(data['inp1'], 0.8, 1.2)
    data['inp1'] = tf.image.random_contrast(data['inp1'], 0.8, 1.2)
    data['inp1'] = tf.image.random_brightness(data['inp1'], 0.1)
    #image = tf.image.random_saturation(image, 0, 2)
    
    return data, label

def get_training_dataset(filenames, labeled = True, ordered = False):
    dataset = load_dataset(filenames, labeled = labeled, ordered = ordered)
    dataset = dataset.map(setup_input1, num_parallel_calls = AUTO)
    dataset = dataset.map(data_augment, num_parallel_calls = AUTO)
    dataset = dataset.map(transform, num_parallel_calls = AUTO)
    # the training dataset must repeat for several epochs
    dataset = dataset.repeat() 
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    # prefetch next batch while training (autotune prefetch buffer size)
    dataset = dataset.prefetch(AUTO)
    return dataset

def get_validation_dataset(filenames, labeled = True, ordered = True):
    dataset = load_dataset(filenames, labeled = labeled, ordered = ordered)
    dataset = dataset.map(setup_input1, num_parallel_calls = AUTO)
    dataset = dataset.batch(BATCH_SIZE)
    # using gpu, not enought memory to use cache
    # dataset = dataset.cache()
    # prefetch next batch while training (autotune prefetch buffer size)
    dataset = dataset.prefetch(AUTO) 
    return dataset

def get_test_dataset(filenames, labeled = False, ordered = True):
    dataset = load_dataset(filenames, labeled = labeled, ordered = ordered)
    dataset = dataset.map(setup_input2, num_parallel_calls = AUTO)
    dataset = dataset.batch(BATCH_SIZE)
    # prefetch next batch while training (autotune prefetch buffer size)
    dataset = dataset.prefetch(AUTO) 
    return dataset

# function to count how many photos we have in
def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

# this function parse our images and also get the target variable
def read_tfrecord_full(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), 
        "image_name": tf.io.FixedLenFeature([], tf.string), 
        "target": tf.io.FixedLenFeature([], tf.int64), 
        # meta features
        "age_approx": tf.io.FixedLenFeature([], tf.int64),
        "sex": tf.io.FixedLenFeature([], tf.int64),
        "anatom_site_general_challenge": tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    image_name = example['image_name']
    target = tf.cast(example['target'], tf.float32)
    # meta features
    data = {}
    data['age_approx'] = tf.cast(example['age_approx'], tf.int32)
    data['sex'] = tf.cast(example['sex'], tf.int32)
    data['anatom_site_general_challenge'] = tf.cast(tf.one_hot(example['anatom_site_general_challenge'], 7), tf.int32)
    return image, image_name, target, data

def load_dataset_full(filenames):        
    # automatically interleaves reads from multiple files
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads = AUTO)
    # returns a dataset of (image_name, target)
    dataset = dataset.map(read_tfrecord_full, num_parallel_calls = AUTO) 
    return dataset

def get_data_full(filenames):
    dataset = load_dataset_full(filenames)
    dataset = dataset.map(setup_input3, num_parallel_calls = AUTO)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    return dataset



NUM_TRAINING_IMAGES = int(count_data_items(TRAINING_FILENAMES) * 0.8)
# use validation data for training
NUM_VALIDATION_IMAGES = int(count_data_items(TRAINING_FILENAMES) * 0.2)
NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

print('Dataset: {} training images, {} validation images, {} unlabeled test images'.format(NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES, NUM_TEST_IMAGES))




get_ipython().run_cell_magic('time', '', '\ndef binary_focal_loss(gamma=2., alpha=.25):\n    """\n    Binary form of focal loss.\n      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)\n      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.\n    References:\n        https://arxiv.org/pdf/1708.02002.pdf\n    Usage:\n     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)\n    """\n    def binary_focal_loss_fixed(y_true, y_pred):\n        """\n        :param y_true: A tensor of the same shape as `y_pred`\n        :param y_pred:  A tensor resulting from a sigmoid\n        :return: Output tensor.\n        """\n        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))\n        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))\n\n        epsilon = K.epsilon()\n        # clip to prevent NaN\'s and Inf\'s\n        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)\n        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)\n\n        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \\\n               -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))\n\n    return binary_focal_loss_fixed\n\ndef get_model():\n    \n    with strategy.scope():\n        \n        inp1 = tf.keras.layers.Input(shape = (*IMAGE_SIZE, 3), name = \'inp1\')\n        inp2 = tf.keras.layers.Input(shape = (9), name = \'inp2\')\n        \n        efnetb6 = efn.EfficientNetB6(weights = \'imagenet\', include_top = False)\n        x6 = efnetb6(inp1)\n        x6 = tf.keras.layers.GlobalAveragePooling2D()(x6)\n        \n        x1 = tf.keras.layers.Dense(32)(inp2)\n        x1 = tf.keras.layers.BatchNormalization()(x1)\n        x1 = tf.keras.layers.Activation(\'relu\')(x1)\n        \n        concat = tf.keras.layers.concatenate([x6, x1])\n        concat = tf.keras.layers.Dense(256, activation = \'relu\')(concat)\n        concat = tf.keras.layers.BatchNormalization()(concat)\n        concat = tf.keras.layers.Dropout(0.2)(concat)\n        concat = tf.keras.layers.Dense(64, activation = \'relu\')(concat)\n        concat = tf.keras.layers.BatchNormalization()(concat)\n        concat = tf.keras.layers.Dropout(0.2)(concat)\n        output = tf.keras.layers.Dense(1, activation = \'sigmoid\')(concat)\n\n        model = tf.keras.models.Model(inputs = [inp1, inp2], outputs = [output])\n\n        opt = tf.keras.optimizers.Adam(learning_rate = LR)\n\n        model.compile(\n            optimizer = opt,\n#             loss = [binary_focal_loss(gamma = 2.0, alpha = 0.80)]\n            loss = [tf.keras.losses.BinaryCrossentropy(label_smoothing = 0.05)],\n            metrics = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()]\n        )\n        \n        print(model.summary())\n\n        return model\n    \ndef train_and_predict(SUB, folds = 5):\n    \n    models = []\n    oof_image_name = []\n    oof_target = []\n    oof_prediction = []\n    \n    # seed everything\n    seed_everything(SEED)\n\n    kfold = KFold(folds, shuffle = True, random_state = SEED)\n    \n    for fold, (trn_ind, val_ind) in enumerate(kfold.split(TRAINING_FILENAMES)):\n        \n        tic = time.time()\n        \n        print(\'\\n\\n\')\n        print(\'-\'*50)\n        print(f\'Training fold {fold}\')\n        print(\'-\'*50)\n        print(\'\\n\')\n        \n        train_dataset = get_training_dataset([TRAINING_FILENAMES[x] for x in trn_ind], labeled = True, ordered = False)\n        val_dataset = get_validation_dataset([TRAINING_FILENAMES[x] for x in val_ind], labeled = True, ordered = True)\n        K.clear_session()\n        model = get_model()\n        \n        # using early stopping using val loss\n        early_stopping = tf.keras.callbacks.EarlyStopping(monitor = \'val_auc\', mode = \'max\', patience = 10, \n                                                      verbose = 1, min_delta = 0.0001, restore_best_weights = True)\n        \n        # lr scheduler\n        cb_lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor = \'val_auc\', factor = 0.5, patience = 2, verbose = 1, min_delta = 0.0001, mode = \'max\')\n        \n        history = model.fit(train_dataset, \n                            steps_per_epoch = STEPS_PER_EPOCH,\n                            epochs = EPOCHS,\n                            callbacks = [early_stopping, cb_lr_schedule],\n                            validation_data = val_dataset,\n                            verbose = 2)\n        \n        models.append(model)\n\n        model.save_weights(f\'ENet_fold_{fold}.h5\')\n        print(\'Weights saved!\')\n\n        # want to predict the validation set and save them for stacking\n        number_of_files = count_data_items([TRAINING_FILENAMES[x] for x in val_ind])\n        dataset = get_data_full([TRAINING_FILENAMES[x] for x in val_ind])\n        \n        # get the image name\n        image_name = dataset.map(lambda image, image_name, target: image_name).unbatch()\n        image_name = next(iter(image_name.batch(number_of_files))).numpy().astype(\'U\')\n        \n        # get the real target\n        target = dataset.map(lambda image, image_name, target: target).unbatch()\n        target = next(iter(target.batch(number_of_files))).numpy()\n        \n        # predict the validation set\n        image = dataset.map(lambda image, image_name, target: image)\n        probabilities = model.predict(image)\n        \n        oof_image_name.extend(list(image_name))\n        oof_target.extend(list(target))\n        oof_prediction.extend(list(np.concatenate(probabilities)))\n\n        toc = time.time()\n        \n        print(\'\\n\')\n        print(\'Time taken:\', toc-tic)\n        print(\'\\n\')\n    \n    print(\'-\'*50)\n    \n    # save oof predictions\n    oof_df = pd.DataFrame({\'image_name\': oof_image_name, \'target\': oof_target, \'predictions\': oof_prediction})\n    oof_df.to_csv(\'oof.csv\', index = False)\n        \n    # since we are splitting the dataset and iterating separately on images and ids, order matters.\n    test_ds = get_test_dataset(TEST_FILENAMES, labeled = False, ordered = True)\n    test_images_ds = test_ds.map(lambda image, image_name: image)\n    \n    print(\'Computing predictions...\')\n    \n    probabilities = np.average([np.concatenate(models[i].predict(test_images_ds)) for i in range(folds)], axis = 0)\n    print(\'Generating submission file...\')\n    \n    test_ids_ds = test_ds.map(lambda image, image_name: image_name).unbatch()\n    # all in one batch\n    test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype(\'U\') # all in one batch\n    \n    pred_df = pd.DataFrame({\'image_name\': test_ids, \'target\': probabilities})\n    \n    SUB.drop(\'target\', inplace = True, axis = 1)\n    SUB = SUB.merge(pred_df, on = \'image_name\')\n    SUB.to_csv(f\'submission.csv\', index = False)\n    \n    return oof_target, oof_prediction\n    \noof_target, oof_prediction = train_and_predict(SUB)')




# calculate our out of folds roc auc score
roc_auc = metrics.roc_auc_score(oof_target, oof_prediction)
print('Our out of folds roc auc score is: ', roc_auc)

