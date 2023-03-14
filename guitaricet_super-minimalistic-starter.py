#!/usr/bin/env python
# coding: utf-8



import re

import numpy as np

import tensorflow as tf
from tensorflow.keras import layers

import matplotlib.pyplot as plt
from kaggle_datasets import KaggleDatasets  # required for TPU dataloading


AUTO = tf.data.experimental.AUTOTUNE
tf.__version__




# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())

    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except ValueError:
    # default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)




IMAGE_SIZE = [192, 192]
BATCH_SIZE = 32 * strategy.num_replicas_in_sync

CLASSES = ['pink primrose',    'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea',     'wild geranium',     'tiger lily',           'moon orchid',              'bird of paradise', 'monkshood',        'globe thistle',         # 00 - 09
           'snapdragon',       "colt's foot",               'king protea',      'spear thistle', 'yellow iris',       'globe-flower',         'purple coneflower',        'peruvian lily',    'balloon flower',   'giant white arum lily', # 10 - 19
           'fire lily',        'pincushion flower',         'fritillary',       'red ginger',    'grape hyacinth',    'corn poppy',           'prince of wales feathers', 'stemless gentian', 'artichoke',        'sweet william',         # 20 - 29
           'carnation',        'garden phlox',              'love in the mist', 'cosmos',        'alpine sea holly',  'ruby-lipped cattleya', 'cape flower',              'great masterwort', 'siam tulip',       'lenten rose',           # 30 - 39
           'barberton daisy',  'daffodil',                  'sword lily',       'poinsettia',    'bolero deep blue',  'wallflower',           'marigold',                 'buttercup',        'daisy',            'common dandelion',      # 40 - 49
           'petunia',          'wild pansy',                'primula',          'sunflower',     'lilac hibiscus',    'bishop of llandaff',   'gaura',                    'geranium',         'orange dahlia',    'pink-yellow dahlia',    # 50 - 59
           'cautleya spicata', 'japanese anemone',          'black-eyed susan', 'silverbush',    'californian poppy', 'osteospermum',         'spring crocus',            'iris',             'windflower',       'tree poppy',            # 60 - 69
           'gazania',          'azalea',                    'water lily',       'rose',          'thorn apple',       'morning glory',        'passion flower',           'lotus',            'toad lily',        'anthurium',             # 70 - 79
           'frangipani',       'clematis',                  'hibiscus',         'columbine',     'desert-rose',       'tree mallow',          'magnolia',                 'cyclamen ',        'watercress',       'canna lily',            # 80 - 89
           'hippeastrum ',     'bee balm',                  'pink quill',       'foxglove',      'bougainvillea',     'camellia',             'mallow',                   'mexican petunia',  'bromelia',         'blanket flower',        # 90 - 99
           'trumpet creeper',  'blackberry lily',           'common tulip',     'wild rose']                                                                                                                                               # 100 - 102
N_CLASSES = len(CLASSES)




def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU
    return image

def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    label = tf.cast(example['class'], tf.int32)
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

def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)




GCS_DS_PATH = KaggleDatasets().get_gcs_path()

# python glob won't wors as you're using GCP buckets (required for TPU)
train_fnames = tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-192x192/train/*.tfrec')
valid_fnames = tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-192x192/val/*.tfrec')
test_fnames = tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-192x192/test/*.tfrec')

n_train = count_data_items(train_fnames)

train_ds = tf.data.TFRecordDataset(filenames=train_fnames)    .map(read_labeled_tfrecord, num_parallel_calls=AUTO)    .repeat()    .shuffle(buffer_size=2048)

valid_ds = tf.data.TFRecordDataset(filenames=valid_fnames)    .map(read_labeled_tfrecord, num_parallel_calls=AUTO)    .cache()




show_n = 3
plt.figure(figsize=(15, 5))

for i, (image, label) in enumerate(train_ds.take(show_n)):
    plt.subplot(1, show_n, i+1)
    plt.imshow(image.numpy())
    plt.title(CLASSES[label])




with strategy.scope():  # device specification (TPU/GPU/CPU)
    body = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, input_shape=[*IMAGE_SIZE, 3])
#     body.trainable = False  # either finetune pretrained model or use it as a feature extractor

    model = tf.keras.Sequential([
        body,
        layers.GlobalAveragePooling2D(),
        layers.Dense(N_CLASSES, activation='softmax')
    ])




batch, label = next(iter(train_ds.batch(2)))

out = model(batch)
out.shape




model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy']
)
model.summary()




steps_per_epoch = n_train // BATCH_SIZE + int(n_train % BATCH_SIZE > 0)

train_dl = train_ds.batch(BATCH_SIZE).prefetch(AUTO)
valid_dl = valid_ds.batch(BATCH_SIZE).prefetch(AUTO)

history = model.fit(train_dl,
    validation_data=valid_dl,
    steps_per_epoch=steps_per_epoch,
    epochs=5)




plt.figure(figsize=(15,7))
plt.plot(history.history['sparse_categorical_accuracy'], label='train')
plt.plot(history.history['val_sparse_categorical_accuracy'], label='valid')
plt.title('Accuracy')
plt.legend()




get_image_lambda = lambda image, id_: image  # Autograph asked to create lambda as a standalone statement
get_id_lambda = lambda image, id_: id_

test_ds = tf.data.TFRecordDataset(filenames=test_fnames)    .map(read_unlabeled_tfrecord, num_parallel_calls=AUTO)    .batch(BATCH_SIZE)    .prefetch(AUTO)
n_test = count_data_items(test_fnames)

print('Computing predictions...')

probabilities = model.predict(test_ds.map(get_image_lambda))
predictions = np.argmax(probabilities, axis=-1)

print('Generating submission.csv file...')
test_ids_ds = test_ds.map(get_id_lambda).unbatch()
test_ids = next(iter(test_ids_ds.batch(n_test))).numpy().astype('U') # all in one batch
np.savetxt('submission.csv', np.rec.fromarrays([test_ids, predictions]), fmt=['%s', '%d'], delimiter=',', header='id,label', comments='')
get_ipython().system('head submission.csv')






