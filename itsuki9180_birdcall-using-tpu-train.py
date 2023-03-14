#!/usr/bin/env python
# coding: utf-8



import pandas as pd, numpy as np, gc
from kaggle_datasets import KaggleDatasets
import tensorflow as tf, re, math
import tensorflow.keras.backend as K
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt




DEVICE = "TPU" # "TPU" or "GPU"
IMG_SIZE = [128, 313]
FOLDS = 5
BATCH_SIZE = 512
EPOCHS = [50]*FOLDS
AUG_BATCH = BATCH_SIZE




if DEVICE == "TPU":
    print("connecting to TPU...")
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())
    except ValueError:
        print("Could not connect to TPU")
        tpu = None

    if tpu:
        try:
            print("initializing  TPU ...")
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.experimental.TPUStrategy(tpu)
            print("TPU initialized")
        except _:
            print("failed to initialize TPU")
    else:
        DEVICE = "GPU"

if DEVICE != "TPU":
    print("Using default strategy for CPU and single GPU")
    strategy = tf.distribute.get_strategy()

if DEVICE == "GPU":
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    

AUTO     = tf.data.experimental.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync
print(f'REPLICAS: {REPLICAS}')




GCS_PATH = [None]*FOLDS; GCS_PATH2 = [None]*FOLDS; GCS_PATH3 = [None]*FOLDS; GCS_PATH4 = [None]*FOLDS; GCS_PATH5 = [None]*FOLDS
for i in range(FOLDS):
    GCS_PATH[i] = KaggleDatasets().get_gcs_path('birdsongab')
    GCS_PATH2[i] = KaggleDatasets().get_gcs_path('birdsongcf')
    GCS_PATH3[i] = KaggleDatasets().get_gcs_path('birdsonggm')
    GCS_PATH4[i] = KaggleDatasets().get_gcs_path('birdsongnr')
    GCS_PATH5[i] = KaggleDatasets().get_gcs_path('birdsongsy')

files_train = np.sort(np.array(tf.io.gfile.glob(GCS_PATH[0] + '/*.tfrec')))




def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [128, 313, 3]) # explicit size needed for TPU
    return image

def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "target": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    label = tf.cast(example['target'], tf.int32)
    return image, label # returns a dataset of (image, label) pairs

def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "id": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element
        # class is missing, this competitions's challenge is to predict flower classes for the test dataset
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    idnum = example['id']
    return image, idnum # returns a dataset of image(s)

def load_dataset(filenames, labeled = True, ordered = False):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # Diregarding data order. Order does not matter since we will be shuffling the data anyway
    
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed
        
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads = AUTO) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # use data as soon as it streams in, rather than in its original order
    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls = AUTO) # returns a dataset of (image, label) pairs if labeled = True or (image, id) pair if labeld = False
    return dataset

def data_augment(image, label):
    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),
    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part
    # of the TPU while the TPU itself is computing gradients.
    #image = tf.image.random_flip_left_right(image)
    return image, label   

def get_training_dataset(dataset, do_aug=True):
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.batch(AUG_BATCH)
    if do_aug: dataset = dataset.map(transform, num_parallel_calls=AUTO) # note we put AFTER batching
    dataset = dataset.unbatch()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_validation_dataset(dataset, do_onehot=True):
    dataset = dataset.batch(BATCH_SIZE)
    if do_onehot: dataset = dataset.map(onehot, num_parallel_calls=AUTO) # we must use one hot like augmented train data
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_test_dataset(ordered=False):
    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

NUM_TRAINING_IMAGES = int( count_data_items(files_train) * (FOLDS-1.)/FOLDS )
NUM_VALIDATION_IMAGES = int( count_data_items(files_train) * (1./FOLDS) )

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

#print('Dataset: {} training images, {} validation images, {} unlabeled test images'.format(NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES, NUM_TEST_IMAGES))




def onehot(image,label):
    CLASSES = 264
    return image,tf.one_hot(label,CLASSES)




def mixup(image, label, PROBABILITY = 1.0):
    # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
    # output - a batch of images with mixup applied
    #DIM = IMAGE_SIZE[0]
    CLASSES = 264
    
    imgs = []; labs = []
    for j in range(AUG_BATCH):
        # DO MIXUP WITH PROBABILITY DEFINED ABOVE
        P = tf.cast( tf.random.uniform([],0,1)<=PROBABILITY, tf.float32)
        # CHOOSE RANDOM
        k = tf.cast( tf.random.uniform([],0,AUG_BATCH),tf.int32)
        a = tf.random.uniform([],0,1)*P # this is beta dist with alpha=1.0
        b = 1-a
        #a = tf.sqrt(a)
        #b = tf.sqrt(b)
        # MAKE MIXUP IMAGE
        img1 = image[j,]
        img2 = image[k,]
        imgs.append(b*img1 + a*img2)
        # MAKE CUTMIX LABEL
        if len(label.shape)==1:
            lab1 = tf.one_hot(label[j],CLASSES)
            lab2 = tf.one_hot(label[k],CLASSES)
        else:
            lab1 = label[j,]
            lab2 = label[k,]
        
        
        labs.append(tf.math.minimum(b*lab1 + a*lab2,1))
            
    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
    image2 = tf.reshape(tf.stack(imgs),(AUG_BATCH,128,313,3))
    label2 = tf.reshape(tf.stack(labs),(AUG_BATCH,CLASSES))
    return image2,label2




def augmentation(img):
    image = tf.image.flip_left_right(img)
    return img




def transform(image,label):
    # THIS FUNCTION APPLIES BOTH CUTMIX AND MIXUP
    CLASSES = 264
    SWITCH = 0.5
    CUTMIX_PROB = 0.666
    MIXUP_PROB = 0.
    # FOR SWITCH PERCENT OF TIME WE DO CUTMIX AND (1-SWITCH) WE DO MIXUP
    #image2, label2 = cutmix(image, label, CUTMIX_PROB)
    image3, label3 = mixup(image, label, MIXUP_PROB)
    imgs = []; labs = []
    for j in range(AUG_BATCH):
        imgs.append(augmentation(image[j,]))
        labs.append(label3[j,])
    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
    image4 = tf.reshape(tf.stack(imgs),(AUG_BATCH,128,313,3))
    label4 = tf.reshape(tf.stack(labs),(AUG_BATCH,CLASSES))
    return image4,label4





row = 2; col = 2;
row = min(row,AUG_BATCH//col)
all_elements = get_training_dataset(load_dataset(files_train),do_aug=False).unbatch()
augmented_element = all_elements.repeat().batch(AUG_BATCH).map(transform)

for (img,label) in augmented_element:
    print(label)
    plt.figure(figsize=(15,int(15*row/col)))
    for j in range(row*col):
        plt.subplot(row,col,j+1)
        plt.axis('off')
        plt.imshow(img[j,])
    plt.show()
    break




from keras.callbacks import Callback
def total_acc(y_true, y_pred):
    pred = K.cast(K.greater_equal(y_pred, 0.5), "float")
    flag = K.cast(K.equal(y_true, pred), "float")
    return K.prod(flag, axis=-1)

def binary_acc(y_true, y_pred):
    pred = K.cast(K.greater_equal(y_pred, 0.5), "float")
    flag = K.cast(K.equal(y_true, pred), "float")
    return K.mean(flag, axis=-1)

class F1Callback(Callback):
    def __init__(self):
        self.f1s = []

    def on_epoch_end(self, epoch, logs):
        eps = np.finfo(np.float32).eps
        recall = logs["val_true_positives"] / (logs["val_possible_positives"] + eps)
        precision = logs["val_true_positives"] / (logs["val_predicted_positives"] + eps)
        f1 = 2*precision*recall / (precision+recall+eps)
        print("f1_val (from log) =", f1)
        self.f1s.append(f1)

def true_positives(y_true, y_pred):
    return K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

def possible_positives(y_true, y_pred):
    return K.sum(K.round(K.clip(y_true, 0, 1)))

def predicted_positives(y_true, y_pred):
    return K.sum(K.round(K.clip(y_pred, 0, 1)))

def F1(y_true, y_pred):
    TPFN = possible_positives(y_true, y_pred)
    TPFP = predicted_positives(y_true, y_pred)
    TP = true_positives(y_true, y_pred)
    return  (TP * 2) / (TPFN + TPFP + K.epsilon())




f1cb = F1Callback()




def binary_loss(y_true, y_pred):
    bce = K.binary_crossentropy(y_true, y_pred)
    return K.sum(bce, axis=-1)




from tensorflow.keras import backend as K

import dill


def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))                -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return binary_focal_loss_fixed


def categorical_focal_loss(gamma=2., alpha=.25):
    """
    Softmax version of focal loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """
        epsilon = K.epsilon()
        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred+epsilon, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Sum the losses in mini_batch
        return K.sum(loss, axis=1)

    return categorical_focal_loss_fixed




# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=invalid-name
"""EfficientNet models for Keras.

Reference paper:
  - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks]
    (https://arxiv.org/abs/1905.11946) (ICML 2019)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math
import os

from tensorflow.python.keras import backend
from tensorflow.python.keras import layers
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.util.tf_export import keras_export


BASE_WEIGHTS_PATH = 'https://storage.googleapis.com/keras-applications/'

WEIGHTS_HASHES = {
    'b0': ('902e53a9f72be733fc0bcb005b3ebbac',
           '50bc09e76180e00e4465e1a485ddc09d'),
    'b1': ('1d254153d4ab51201f1646940f018540',
           '74c4e6b3e1f6a1eea24c589628592432'),
    'b2': ('b15cce36ff4dcbd00b6dd88e7857a6ad',
           '111f8e2ac8aa800a7a99e3239f7bfb39'),
    'b3': ('ffd1fdc53d0ce67064dc6a9c7960ede0',
           'af6d107764bb5b1abb91932881670226'),
    'b4': ('18c95ad55216b8f92d7e70b3a046e2fc',
           'ebc24e6d6c33eaebbd558eafbeedf1ba'),
    'b5': ('ace28f2a6363774853a83a0b21b9421a',
           '38879255a25d3c92d5e44e04ae6cec6f'),
    'b6': ('165f6e37dce68623721b423839de8be5',
           '9ecce42647a20130c1f39a5d4cb75743'),
    'b7': ('8c03f828fec3ef71311cd463b6759d99',
           'cbcfe4450ddf6f3ad90b1b398090fe4a'),
}

DEFAULT_BLOCKS_ARGS = [{
    'kernel_size': 3,
    'repeats': 1,
    'filters_in': 32,
    'filters_out': 16,
    'expand_ratio': 1,
    'id_skip': True,
    'strides': 1,
    'se_ratio': 0.25
}, {
    'kernel_size': 3,
    'repeats': 2,
    'filters_in': 16,
    'filters_out': 24,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 5,
    'repeats': 2,
    'filters_in': 24,
    'filters_out': 40,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 3,
    'repeats': 3,
    'filters_in': 40,
    'filters_out': 80,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 5,
    'repeats': 3,
    'filters_in': 80,
    'filters_out': 112,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 1,
    'se_ratio': 0.25
}, {
    'kernel_size': 5,
    'repeats': 4,
    'filters_in': 112,
    'filters_out': 192,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 3,
    'repeats': 1,
    'filters_in': 192,
    'filters_out': 320,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 1,
    'se_ratio': 0.25
}]

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'truncated_normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}


def EfficientNet(
    width_coefficient,
    depth_coefficient,
    default_size,
    dropout_rate=0.2,
    drop_connect_rate=0.2,
    depth_divisor=8,
    activation='swish',
    blocks_args='default',
    model_name='efficientnet',
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation='softmax',
):
  """Instantiates the EfficientNet architecture using given scaling coefficients.

  Optionally loads weights pre-trained on ImageNet.
  Note that the data format convention used by the model is
  the one specified in your Keras config at `~/.keras/keras.json`.

  Arguments:
    width_coefficient: float, scaling coefficient for network width.
    depth_coefficient: float, scaling coefficient for network depth.
    default_size: integer, default input image size.
    dropout_rate: float, dropout rate before final classifier layer.
    drop_connect_rate: float, dropout rate at skip connections.
    depth_divisor: integer, a unit of network width.
    activation: activation function.
    blocks_args: list of dicts, parameters to construct block modules.
    model_name: string, model name.
    include_top: whether to include the fully-connected
        layer at the top of the network.
    weights: one of `None` (random initialization),
          'imagenet' (pre-training on ImageNet),
          or the path to the weights file to be loaded.
    input_tensor: optional Keras tensor
        (i.e. output of `layers.Input()`)
        to use as image input for the model.
    input_shape: optional shape tuple, only to be specified
        if `include_top` is False.
        It should have exactly 3 inputs channels.
    pooling: optional pooling mode for feature extraction
        when `include_top` is `False`.
        - `None` means that the output of the model will be
            the 4D tensor output of the
            last convolutional layer.
        - `avg` means that global average pooling
            will be applied to the output of the
            last convolutional layer, and thus
            the output of the model will be a 2D tensor.
        - `max` means that global max pooling will
            be applied.
    classes: optional number of classes to classify images
        into, only to be specified if `include_top` is True, and
        if no `weights` argument is specified.
    classifier_activation: A `str` or callable. The activation function to use
        on the "top" layer. Ignored unless `include_top=True`. Set
        `classifier_activation=None` to return the logits of the "top" layer.

  Returns:
    A `keras.Model` instance.

  Raises:
    ValueError: in case of invalid argument for `weights`,
      or invalid input shape.
    ValueError: if `classifier_activation` is not `softmax` or `None` when
      using a pretrained top layer.
  """
  if blocks_args == 'default':
    blocks_args = DEFAULT_BLOCKS_ARGS

  if not (weights in {'imagenet', None} or os.path.exists(weights)):
    raise ValueError('The `weights` argument should be either '
                     '`None` (random initialization), `imagenet` '
                     '(pre-training on ImageNet), '
                     'or the path to the weights file to be loaded.')

  if weights == 'imagenet' and include_top and classes != 1000:
    raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                     ' as true, `classes` should be 1000')

  # Determine proper input shape
  input_shape = imagenet_utils.obtain_input_shape(
      input_shape,
      default_size=default_size,
      min_size=32,
      data_format=backend.image_data_format(),
      require_flatten=include_top,
      weights=weights)

  if input_tensor is None:
    img_input = layers.Input(shape=input_shape)
  else:
    if not backend.is_keras_tensor(input_tensor):
      img_input = layers.Input(tensor=input_tensor, shape=input_shape)
    else:
      img_input = input_tensor

  bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

  def round_filters(filters, divisor=depth_divisor):
    """Round number of filters based on depth multiplier."""
    filters *= width_coefficient
    new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
      new_filters += divisor
    return int(new_filters)

  def round_repeats(repeats):
    """Round number of repeats based on depth multiplier."""
    return int(math.ceil(depth_coefficient * repeats))

  # Build stem
  x = img_input
  #x = layers.Rescaling(1. / 255.)(x)
  x = layers.Normalization(axis=bn_axis)(x)

  x = layers.ZeroPadding2D(
      padding=imagenet_utils.correct_pad(x, 3),
      name='stem_conv_pad')(x)
  x = layers.Conv2D(
      round_filters(32),
      3,
      strides=2,
      padding='valid',
      use_bias=False,
      kernel_initializer=CONV_KERNEL_INITIALIZER,
      name='stem_conv')(x)
  x = layers.BatchNormalization(axis=bn_axis, name='stem_bn')(x)
  x = layers.Activation(activation, name='stem_activation')(x)

  # Build blocks
  blocks_args = copy.deepcopy(blocks_args)

  b = 0
  blocks = float(sum(args['repeats'] for args in blocks_args))
  for (i, args) in enumerate(blocks_args):
    assert args['repeats'] > 0
    # Update block input and output filters based on depth multiplier.
    args['filters_in'] = round_filters(args['filters_in'])
    args['filters_out'] = round_filters(args['filters_out'])

    for j in range(round_repeats(args.pop('repeats'))):
      # The first block needs to take care of stride and filter size increase.
      if j > 0:
        args['strides'] = 1
        args['filters_in'] = args['filters_out']
      x = block(
          x,
          activation,
          drop_connect_rate * b / blocks,
          name='block{}{}_'.format(i + 1, chr(j + 97)),
          **args)
      b += 1

  # Build top
  x = layers.Conv2D(
      round_filters(1280),
      1,
      padding='same',
      use_bias=False,
      kernel_initializer=CONV_KERNEL_INITIALIZER,
      name='top_conv')(x)
  x = layers.BatchNormalization(axis=bn_axis, name='top_bn')(x)
  x = layers.Activation(activation, name='top_activation')(x)
  if include_top:
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    if dropout_rate > 0:
      x = layers.Dropout(dropout_rate, name='top_dropout')(x)
    imagenet_utils.validate_activation(classifier_activation, weights)
    x = layers.Dense(
        classes,
        activation=classifier_activation,
        kernel_initializer=DENSE_KERNEL_INITIALIZER,
        name='predictions')(x)
  else:
    if pooling == 'avg':
      x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    elif pooling == 'max':
      x = layers.GlobalMaxPooling2D(name='max_pool')(x)

  # Ensure that the model takes into account
  # any potential predecessors of `input_tensor`.
  if input_tensor is not None:
    inputs = layer_utils.get_source_inputs(input_tensor)
  else:
    inputs = img_input

  # Create model.
  model = training.Model(inputs, x, name=model_name)

  # Load weights.
  if weights == 'imagenet':
    if include_top:
      file_suffix = '.h5'
      file_hash = WEIGHTS_HASHES[model_name[-2:]][0]
    else:
      file_suffix = '_notop.h5'
      file_hash = WEIGHTS_HASHES[model_name[-2:]][1]
    file_name = model_name + file_suffix
    weights_path = data_utils.get_file(
        file_name,
        BASE_WEIGHTS_PATH + file_name,
        cache_subdir='models',
        file_hash=file_hash)
    model.load_weights(weights_path)
  elif weights is not None:
    model.load_weights(weights)
  return model


def block(inputs,
          activation='swish',
          drop_rate=0.,
          name='',
          filters_in=32,
          filters_out=16,
          kernel_size=3,
          strides=1,
          expand_ratio=1,
          se_ratio=0.,
          id_skip=True):
  """An inverted residual block.

  Arguments:
      inputs: input tensor.
      activation: activation function.
      drop_rate: float between 0 and 1, fraction of the input units to drop.
      name: string, block label.
      filters_in: integer, the number of input filters.
      filters_out: integer, the number of output filters.
      kernel_size: integer, the dimension of the convolution window.
      strides: integer, the stride of the convolution.
      expand_ratio: integer, scaling coefficient for the input filters.
      se_ratio: float between 0 and 1, fraction to squeeze the input filters.
      id_skip: boolean.

  Returns:
      output tensor for the block.
  """
  bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

  # Expansion phase
  filters = filters_in * expand_ratio
  if expand_ratio != 1:
    x = layers.Conv2D(
        filters,
        1,
        padding='same',
        use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name=name + 'expand_conv')(
            inputs)
    x = layers.BatchNormalization(axis=bn_axis, name=name + 'expand_bn')(x)
    x = layers.Activation(activation, name=name + 'expand_activation')(x)
  else:
    x = inputs

  # Depthwise Convolution
  if strides == 2:
    x = layers.ZeroPadding2D(
        padding=imagenet_utils.correct_pad(x, kernel_size),
        name=name + 'dwconv_pad')(x)
    conv_pad = 'valid'
  else:
    conv_pad = 'same'
  x = layers.DepthwiseConv2D(
      kernel_size,
      strides=strides,
      padding=conv_pad,
      use_bias=False,
      depthwise_initializer=CONV_KERNEL_INITIALIZER,
      name=name + 'dwconv')(x)
  x = layers.BatchNormalization(axis=bn_axis, name=name + 'bn')(x)
  x = layers.Activation(activation, name=name + 'activation')(x)

  # Squeeze and Excitation phase
  if 0 < se_ratio <= 1:
    filters_se = max(1, int(filters_in * se_ratio))
    se = layers.GlobalAveragePooling2D(name=name + 'se_squeeze')(x)
    se = layers.Reshape((1, 1, filters), name=name + 'se_reshape')(se)
    se = layers.Conv2D(
        filters_se,
        1,
        padding='same',
        activation=activation,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name=name + 'se_reduce')(
            se)
    se = layers.Conv2D(
        filters,
        1,
        padding='same',
        activation='sigmoid',
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name=name + 'se_expand')(se)
    x = layers.multiply([x, se], name=name + 'se_excite')

  # Output phase
  x = layers.Conv2D(
      filters_out,
      1,
      padding='same',
      use_bias=False,
      kernel_initializer=CONV_KERNEL_INITIALIZER,
      name=name + 'project_conv')(x)
  x = layers.BatchNormalization(axis=bn_axis, name=name + 'project_bn')(x)
  if id_skip and strides == 1 and filters_in == filters_out:
    if drop_rate > 0:
      x = layers.Dropout(
          drop_rate, noise_shape=(None, 1, 1, 1), name=name + 'drop')(x)
    x = layers.add([x, inputs], name=name + 'add')
  return x


@keras_export('keras.applications.efficientnet.EfficientNetB0',
              'keras.applications.EfficientNetB0')
def EfficientNetB0(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):
  return EfficientNet(
      1.0,
      1.0,
      224,
      0.2,
      model_name='efficientnetb0',
      include_top=include_top,
      weights=weights,
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      **kwargs)


@keras_export('keras.applications.efficientnet.EfficientNetB1',
              'keras.applications.EfficientNetB1')
def EfficientNetB1(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):
  return EfficientNet(
      1.0,
      1.1,
      240,
      0.2,
      model_name='efficientnetb1',
      include_top=include_top,
      weights=weights,
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      **kwargs)


@keras_export('keras.applications.efficientnet.EfficientNetB2',
              'keras.applications.EfficientNetB2')
def EfficientNetB2(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):
  return EfficientNet(
      1.1,
      1.2,
      260,
      0.3,
      model_name='efficientnetb2',
      include_top=include_top,
      weights=weights,
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      **kwargs)


@keras_export('keras.applications.efficientnet.EfficientNetB3',
              'keras.applications.EfficientNetB3')
def EfficientNetB3(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):
  return EfficientNet(
      1.2,
      1.4,
      300,
      0.3,
      model_name='efficientnetb3',
      include_top=include_top,
      weights=weights,
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      **kwargs)


@keras_export('keras.applications.efficientnet.EfficientNetB4',
              'keras.applications.EfficientNetB4')
def EfficientNetB4(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):
  return EfficientNet(
      1.4,
      1.8,
      380,
      0.4,
      model_name='efficientnetb4',
      include_top=include_top,
      weights=weights,
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      **kwargs)


@keras_export('keras.applications.efficientnet.EfficientNetB5',
              'keras.applications.EfficientNetB5')
def EfficientNetB5(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):
  return EfficientNet(
      1.6,
      2.2,
      456,
      0.4,
      model_name='efficientnetb5',
      include_top=include_top,
      weights=weights,
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      **kwargs)


@keras_export('keras.applications.efficientnet.EfficientNetB6',
              'keras.applications.EfficientNetB6')
def EfficientNetB6(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):
  return EfficientNet(
      1.8,
      2.6,
      528,
      0.5,
      model_name='efficientnetb6',
      include_top=include_top,
      weights=weights,
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      **kwargs)


@keras_export('keras.applications.efficientnet.EfficientNetB7',
              'keras.applications.EfficientNetB7')
def EfficientNetB7(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):
  return EfficientNet(
      2.0,
      3.1,
      600,
      0.5,
      model_name='efficientnetb7',
      include_top=include_top,
      weights=weights,
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      **kwargs)


@keras_export('keras.applications.efficientnet.preprocess_input')
def preprocess_input(x, data_format=None):  # pylint: disable=unused-argument
  return x


@keras_export('keras.applications.efficientnet.decode_predictions')
def decode_predictions(preds, top=5):
  """Decodes the prediction result from the model.

  Arguments
    preds: Numpy tensor encoding a batch of predictions.
    top: Integer, how many top-guesses to return.

  Returns
    A list of lists of top class prediction tuples
    `(class_name, class_description, score)`.
    One list of tuples per sample in batch input.

  Raises
    ValueError: In case of invalid shape of the `preds` array (must be 2D).
  """
  return imagenet_utils.decode_predictions(preds, top=top)




def build_model():
    inp = tf.keras.layers.Input(shape=(128,313,3))
    #base = tf.keras.applications.Xception(include_top=False, weights='imagenet')
    base = EfficientNetB3(include_top=False,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=[128,313,3])
    x = base(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(264,activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inp,outputs=x)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt,loss=binary_focal_loss(),metrics=[total_acc, F1,true_positives,possible_positives,predicted_positives])
    return model




def get_lr_callback(batch_size=8):
    lr_start   = 0.0005
    lr_max     = 0.001
    lr_min     = 0.00001
    lr_ramp_ep = 5
    lr_sus_ep  = 5
    lr_decay   = 0.9
   
    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
            
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
            
        else:
            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
            
        return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
    return lr_callback




skf = KFold(n_splits=FOLDS,shuffle=True,random_state=12)
for fold,(idxT,idxV) in enumerate(skf.split(np.arange(5))):
    if fold==(FOLDS-1):
        idxTT = idxT; idxVV = idxV
        print('### Using fold',fold,'for experiments')
    print('Fold',fold,'has TRAIN:',idxT,'VALID:',idxV)




for fold in range(FOLDS):
    if fold>0:
        break;
    # REPEAT SAME FOLD OVER AND OVER
    idxT = idxTT
    idxV = idxVV
    
    # DISPLAY FOLD INFO
    if DEVICE=='TPU':
        if tpu: tf.tpu.experimental.initialize_tpu_system(tpu)
    print('#'*25); print('#### EXPERIMENT',fold+1)

    
    # CREATE TRAIN AND VALIDATION SUBSETS
    files_train = tf.io.gfile.glob([GCS_PATH[fold] + '/train-*-%.2i*.tfrec'%x for x in idxT])
    files_train += tf.io.gfile.glob([GCS_PATH2[fold] + '/train-*-%.2i*.tfrec'%x for x in idxT])
    files_train += tf.io.gfile.glob([GCS_PATH3[fold] + '/train-*-%.2i*.tfrec'%x for x in idxT])
    files_train += tf.io.gfile.glob([GCS_PATH4[fold] + '/train-*-%.2i*.tfrec'%x for x in idxT])
    files_train += tf.io.gfile.glob([GCS_PATH5[fold] + '/train-*-%.2i*.tfrec'%x for x in idxT])
    print('#### all trains',len(files_train))
        
    files_valid = tf.io.gfile.glob([GCS_PATH[fold] + '/train-*-%.2i*.tfrec'%x for x in idxV])
    files_valid += tf.io.gfile.glob([GCS_PATH2[fold] + '/train-*-%.2i*.tfrec'%x for x in idxV])
    files_valid += tf.io.gfile.glob([GCS_PATH3[fold] + '/train-*-%.2i*.tfrec'%x for x in idxV])
    files_valid += tf.io.gfile.glob([GCS_PATH4[fold] + '/train-*-%.2i*.tfrec'%x for x in idxV])
    files_valid += tf.io.gfile.glob([GCS_PATH5[fold] + '/train-*-%.2i*.tfrec'%x for x in idxV])
    print('#### all valids',len(files_valid))
    
    NUM_TRAINING_IMAGES = int( count_data_items(files_train))
    NUM_VALIDATION_IMAGES = int( count_data_items(files_valid) )
    STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
    print('Dataset: {} training images, {} validation images,'.format(NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES))
    
    # BUILD MODEL
    K.clear_session()
    with strategy.scope():
        model = build_model()
        
    # SAVE BEST MODEL EACH FOLD
    sv = tf.keras.callbacks.ModelCheckpoint(
        'fold-%i.h5'%fold, monitor='val_loss', verbose=0, save_best_only=True,
        save_weights_only=True, mode='min', save_freq='epoch')
    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto',
        baseline=None, restore_best_weights=False
    )
    # TRAIN
    train_dataset = load_dataset(list(pd.DataFrame({'TRAINING_FILENAMES': files_train}).loc[:]['TRAINING_FILENAMES']), labeled = True)
    val_dataset = load_dataset(list(pd.DataFrame({'VALIDATION_FILENAMES': files_valid}).loc[:]['VALIDATION_FILENAMES']), labeled = True, ordered = True)
    print(model.summary())
    print('Training...')
    history = model.fit(
            get_training_dataset(train_dataset), 
            steps_per_epoch = STEPS_PER_EPOCH,
            epochs = EPOCHS[fold],
            callbacks = [sv,get_lr_callback(BATCH_SIZE),f1cb],
            validation_data = get_validation_dataset(val_dataset),
            verbose=2
        )
    model.save_weights('fold-%if.h5'%fold)
        
    del model; z = gc.collect()











