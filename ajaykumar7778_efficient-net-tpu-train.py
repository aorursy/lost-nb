#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd, numpy as np, gc
from kaggle_datasets import KaggleDatasets
import tensorflow as tf, re, math
import tensorflow.keras.backend as K
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


# In[2]:


DEVICE = "TPU" # "TPU" or "GPU"
IMG_SIZE = [128, 313]
FOLDS = 5
BATCH_SIZE = 512
EPOCHS = [50]*FOLDS
AUG_BATCH = BATCH_SIZE


# In[3]:


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


# In[4]:


GCS_PATH = [None]*FOLDS; GCS_PATH2 = [None]*FOLDS; GCS_PATH3 = [None]*FOLDS; GCS_PATH4 = [None]*FOLDS; GCS_PATH5 = [None]*FOLDS
for i in range(FOLDS):
    GCS_PATH[i] = KaggleDatasets().get_gcs_path('birdsongab')
    GCS_PATH2[i] = KaggleDatasets().get_gcs_path('birdsongcf')
    GCS_PATH3[i] = KaggleDatasets().get_gcs_path('birdsonggm')
    GCS_PATH4[i] = KaggleDatasets().get_gcs_path('birdsongnr')
    GCS_PATH5[i] = KaggleDatasets().get_gcs_path('birdsongsy')

files_train = np.sort(np.array(tf.io.gfile.glob(GCS_PATH[0] + '/*.tfrec')))


# In[5]:


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


# In[6]:


def onehot(image,label):
    CLASSES = 264
    return image,tf.one_hot(label,CLASSES)


# In[7]:


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


# In[8]:


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
        imgs.append(image3[j,])
        labs.append(label3[j,])
    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
    image4 = tf.reshape(tf.stack(imgs),(AUG_BATCH,128,313,3))
    label4 = tf.reshape(tf.stack(labs),(AUG_BATCH,CLASSES))
    return image4,label4


# In[9]:


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


# In[10]:


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


# In[11]:


f1cb = F1Callback()


# In[12]:


def binary_loss(y_true, y_pred):
    bce = K.binary_crossentropy(y_true, y_pred)
    return K.sum(bce, axis=-1)


# In[13]:


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


# In[14]:


# !pip install -q efficientnet


# In[15]:


# import efficientnet.keras as efn 
# from tensorflow import keras
# model = efn.EfficientNetB0(weights='imagenet')  # or weights='noisy-student'


# In[ ]:





# In[16]:


# #!pip uninstall tensorflow
# !pip install tensorflow==1.15.2

# #!pip uninstall Keras
# !pip install Keras==2.3.1

# #!pip uninstall Keras-Applications
# !pip install Keras-Applications==1.0.8

# #!pip uninstall Keras-Preprocessing
# !pip install Keras-Preprocessing==1.1.0


# In[17]:


# # import efficientnet.tfkeras as efn
# import tensorflow.keras.layers as L

# def build_model():
#     #inp = tf.keras.layers.Input(shape=(128,313,3))
    
#     model = tf.keras.Sequential([
#         #tf.keras.applications.EfficientNetB5(
#         tf.keras.applications.DenseNet121(
            
#             input_shape=(*IMG_SIZE, 3),
#             weights='imagenet',
#             #weights='noisy-student',
#             include_top=False
#         ),
#         L.GlobalAveragePooling2D(),
#         L.Dense(1024, activation = 'relu'), 
#         L.Dropout(0.3), 
#         L.Dense(512, activation= 'relu'), 
#         L.Dropout(0.2), 
#         L.Dense(512, activation='relu'), 
        
#         L.Dropout(0.1), 
#         L.Dense(264, activation='sigmoid')
#     ])
#     #model = tf.keras.Model(inputs=inp,outputs=x)
#     opt = tf.keras.optimizers.Adam(learning_rate=0.001)
#     model.compile(optimizer=opt,loss=binary_focal_loss(),metrics=[total_acc, true_positives, possible_positives, predicted_positives])
#     return model

# # model = build_model()
# # model.summary()


# In[ ]:





# In[18]:




def build_model():
    inp = tf.keras.layers.Input(shape=(128,313,3))
    base = tf.keras.applications.DenseNet121(include_top=False, weights='imagenet')
    x = base(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024,activation='swish')(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(512,activation='swish')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(384,activation='swish')(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(264,activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inp,outputs=x)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt,loss=binary_focal_loss(),metrics=[total_acc, true_positives, possible_positives, predicted_positives])
    return model


# In[19]:


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


# In[20]:


skf = KFold(n_splits=FOLDS,shuffle=True,random_state=12)
for fold,(idxT,idxV) in enumerate(skf.split(np.arange(5))):
    if fold==(FOLDS-1):
        idxTT = idxT; idxVV = idxV
        print('### Using fold',fold,'for experiments')
    print('Fold',fold,'has TRAIN:',idxT,'VALID:',idxV)


# In[21]:


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
        monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto',
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
            callbacks = [sv,get_lr_callback(BATCH_SIZE),f1cb,es],
            validation_data = get_validation_dataset(val_dataset),
            verbose=1
        )
    model.save_weights('fold-%if.h5'%fold)
        
    del model; z = gc.collect()


# In[ ]:





# In[ ]:





# In[ ]:




