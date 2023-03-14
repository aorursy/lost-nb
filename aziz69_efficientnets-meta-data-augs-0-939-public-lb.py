#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install -q efficientnet')


# In[ ]:


get_ipython().system('pip install image-classifiers==1.0.0b1')
from classification_models.tfkeras import Classifiers


# In[ ]:


get_ipython().system('pip install keras-swa')
from swa.tfkeras import SWA


# In[ ]:


import pandas as pd
import numpy as np
import os , math , re , random
import cv2
import matplotlib.pyplot as plt

from kaggle_datasets import KaggleDatasets

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report
from sklearn.utils import shuffle


import tensorflow as tf
import tensorflow.keras.layers as L

from tensorflow.keras.callbacks import EarlyStopping,LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Model,Sequential


import efficientnet.tfkeras as efn
from tensorflow.keras.applications import DenseNet121, DenseNet201
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import MobileNet , MobileNetV2
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras import optimizers

from tensorflow.keras import backend as K

get_ipython().system("pip install tensorflow-addons=='0.9.1'")
import tensorflow_addons as tfa


# In[ ]:


# for reproducible results :
def seed_everything(seed=13):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_KERAS'] = '1'
    random.seed(seed)
    
seed_everything(1234)


# In[ ]:


try :
    tpu=tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on :',tpu.master())
except ValueError :
    tpu = None

if tpu :    
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else :
    strategy = tf.distribute.get_strategy()
    
print('Replicas :',strategy.num_replicas_in_sync)  


# In[ ]:


AUTO  = tf.data.experimental.AUTOTUNE

GCS_DS_PATH2 = KaggleDatasets().get_gcs_path('siim-isic-melanoma-classification')
GCS_DS_PATH1 = KaggleDatasets().get_gcs_path('512x512-melanoma-tfrecords-70k-images')
EPOCHS = 20
BATCH_SIZE = 8 * strategy.num_replicas_in_sync 
img_size = 468 # 468 for effnet b5 , b1 , b3
SEED =  1234
nb_classes = 1


# In[ ]:


train = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')
test = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')
sub = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')

train_paths = train.image_name.apply(lambda x : GCS_DS_PATH1 + '/train/' +x + '.jpg').values
test_paths = test.image_name.apply(lambda x : GCS_DS_PATH1 + '/test/' +x + '.jpg').values


# In[ ]:


print(train['sex'].value_counts())
print('*'*100)
print(train['diagnosis'].value_counts())
print('*'*100)
print(train['anatom_site_general_challenge'].value_counts())
print(train['benign_malignant'].value_counts())
print(train['target'].value_counts())


# In[ ]:


train.head()


# In[ ]:


train.patient_id.duplicated().any()


# In[ ]:


dup_patients_test = test[test.patient_id.duplicated() == True]
unique_patient_test_ids = list(set(dup_patients_test['patient_id']))
patient = test[test['patient_id'] == unique_patient_test_ids[2]] 
patient = patient.reset_index(drop=True)
images = []

for i in range(len(patient)) :
    img = '../input/siim-isic-melanoma-classification/jpeg/test/'+patient['image_name'][i]+'.jpg'
    n = cv2.imread(img)
    n = cv2.cvtColor(n, cv2.COLOR_BGR2GRAY)
    images.append(n)

    
plt.figure(figsize=(10,10))
for i in range(len(images)) :
    plt.subplot(8,8,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images[i])


# In[ ]:


dup_patients = train[train.patient_id.duplicated() == True ]
dup_patients


# In[ ]:


unique_patient_ids = set(dup_patients['patient_id'])
len(unique_patient_ids)


# In[ ]:


mislabeled_patient_ids = []
for patient_id in unique_patient_ids :    
    class_count = len(dup_patients[dup_patients['patient_id' ] == patient_id].benign_malignant.value_counts())
    if class_count > 1 :
        mislabeled_patient_ids.append(patient_id)
        print(patient_id ,' is mislabeled :')
        print(dup_patients[dup_patients['patient_id' ] == patient_id].benign_malignant.value_counts())


# In[ ]:


len(mislabeled_patient_ids)


# In[ ]:


dup_patients.loc[dup_patients['patient_id'].isin(mislabeled_patient_ids)]


# In[ ]:


dup_patients.loc[dup_patients['patient_id'].isin(mislabeled_patient_ids)].target.value_counts()


# In[ ]:


train_pos = train[train['target'] == 1]
train_neg_ids = []
for i in range(1200) :
    if train['target'][i] == 0 :
        train_neg_ids.append(i)
train_neg = train.iloc[train_neg_ids]   
train_balanced = pd.concat([train_pos,train_neg])
train_balanced = train_balanced.sample(frac=1).reset_index(drop=True)


# In[ ]:


train_balanced_paths = train_balanced.image_name.apply(lambda x : GCS_DS_PATH + '/jpeg/train/' + x + '.jpg').values
train_balanced_labels = train_balanced.loc[:,'target'].values

from sklearn.model_selection import train_test_split
X_train_paths, X_valid_paths, Y_train, Y_valid = train_test_split(train_balanced_paths,train_balanced_labels, test_size=0.15, random_state=42)


# In[ ]:


bool_random_brightness = False # 0.902 no improvement
bool_random_contrast =False  # doesn't improve
bool_random_hue = False
bool_random_saturation = False

gridmask_rate = 0.4 #improve 0.4
cutmix_rate =  0.4   #improve 0.4
mixup_rate = 0 #doesn't improve
rotation = False
random_blackout = False
crop_size = 0
transforms = False
micro_aug = False
hair_aug = True # hair aug can be good


# In[ ]:


# batch
def cutmix(image, label, PROBABILITY = cutmix_rate):
    # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
    # output - a batch of images with cutmix applied
    
    DIM = img_size    
    imgs = []; labs = []
    
    for j in range(BATCH_SIZE):
        
        #random_uniform( shape, minval=0, maxval=None)        
        # DO CUTMIX WITH PROBABILITY DEFINED ABOVE
        P = tf.cast(tf.random.uniform([], 0, 1) <= PROBABILITY, tf.int32)
        
        # CHOOSE RANDOM IMAGE TO CUTMIX WITH
        k = tf.cast(tf.random.uniform([], 0, BATCH_SIZE), tf.int32)
        
        # CHOOSE RANDOM LOCATION
        x = tf.cast(tf.random.uniform([], 0, DIM), tf.int32)
        y = tf.cast(tf.random.uniform([], 0, DIM), tf.int32)
        
        # Beta(1, 1)
        b = tf.random.uniform([], 0, 1) # this is beta dist with alpha=1.0
        

        WIDTH = tf.cast(DIM * tf.math.sqrt(1-b),tf.int32) * P
        ya = tf.math.maximum(0,y-WIDTH//2)
        yb = tf.math.minimum(DIM,y+WIDTH//2)
        xa = tf.math.maximum(0,x-WIDTH//2)
        xb = tf.math.minimum(DIM,x+WIDTH//2)
        
        # MAKE CUTMIX IMAGE
        one = image[j,ya:yb,0:xa,:]
        two = image[k,ya:yb,xa:xb,:]
        three = image[j,ya:yb,xb:DIM,:]        
        #ya:yb
        middle = tf.concat([one,two,three],axis=1)

        img = tf.concat([image[j,0:ya,:,:],middle,image[j,yb:DIM,:,:]],axis=0)
        imgs.append(img)
        
        # MAKE CUTMIX LABEL
        a = tf.cast(WIDTH*WIDTH/DIM/DIM,tf.float32)
        lab1 = label[j,]
        lab2 = label[k,]
        labs.append((1-a)*lab1 + a*lab2)

    image2 = tf.reshape(tf.stack(imgs),(BATCH_SIZE,DIM,DIM,3))
    label2 = tf.reshape(tf.stack(labs),(BATCH_SIZE, nb_classes))
    return image2,label2


# In[ ]:


def mixup(image, label, PROBABILITY = mixup_rate):
    # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
    # output - a batch of images with mixup applied
    DIM = img_size
    
    imgs = []; labs = []
    for j in range(BATCH_SIZE):
        
        # CHOOSE RANDOM
        k = tf.cast( tf.random.uniform([],0,BATCH_SIZE),tf.int32)
        a = tf.random.uniform([],0,1) # this is beta dist with alpha=1.0

        #mixup
        P = tf.cast(tf.random.uniform([], 0, 1) <= PROBABILITY, tf.int32)
        if P==1:
            a=0.
        
        # MAKE MIXUP IMAGE
        img1 = image[j,]
        img2 = image[k,]
        imgs.append((1-a)*img1 + a*img2)
        
        # MAKE CUTMIX LABEL
        lab1 = label[j,]
        lab2 = label[k,]
        labs.append((1-a)*lab1 + a*lab2)
            
    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
    image2 = tf.reshape(tf.stack(imgs),(BATCH_SIZE,DIM,DIM,3))
    label2 = tf.reshape(tf.stack(labs),(BATCH_SIZE,nb_classes))
    return image2,label2


# In[ ]:


def transform(image, inv_mat, image_shape):
    h, w, c = image_shape
    cx, cy = w//2, h//2
    new_xs = tf.repeat( tf.range(-cx, cx, 1), h)
    new_ys = tf.tile( tf.range(-cy, cy, 1), [w])
    new_zs = tf.ones([h*w], dtype=tf.int32)
    old_coords = tf.matmul(inv_mat, tf.cast(tf.stack([new_xs, new_ys, new_zs]), tf.float32))
    old_coords_x, old_coords_y = tf.round(old_coords[0, :] + w//2), tf.round(old_coords[1, :] + h//2)
    clip_mask_x = tf.logical_or(old_coords_x<0, old_coords_x>w-1)
    clip_mask_y = tf.logical_or(old_coords_y<0, old_coords_y>h-1)
    clip_mask = tf.logical_or(clip_mask_x, clip_mask_y)
    old_coords_x = tf.boolean_mask(old_coords_x, tf.logical_not(clip_mask))
    old_coords_y = tf.boolean_mask(old_coords_y, tf.logical_not(clip_mask))
    new_coords_x = tf.boolean_mask(new_xs+cx, tf.logical_not(clip_mask))
    new_coords_y = tf.boolean_mask(new_ys+cy, tf.logical_not(clip_mask))
    old_coords = tf.cast(tf.stack([old_coords_y, old_coords_x]), tf.int32)
    new_coords = tf.cast(tf.stack([new_coords_y, new_coords_x]), tf.int64)
    rotated_image_values = tf.gather_nd(image, tf.transpose(old_coords))
    rotated_image_channel = list()
    for i in range(c):
        vals = rotated_image_values[:,i]
        sparse_channel = tf.SparseTensor(tf.transpose(new_coords), vals, [h, w])
        rotated_image_channel.append(tf.sparse.to_dense(sparse_channel, default_value=0, validate_indices=False))
    return tf.transpose(tf.stack(rotated_image_channel), [1,2,0])


# In[ ]:


def random_rotate(image, angle, image_shape):
    def get_rotation_mat_inv(angle):
        # transform to radian
        angle = math.pi * angle / 180
        cos_val = tf.math.cos(angle)
        sin_val = tf.math.sin(angle)
        one = tf.constant([1], tf.float32)
        zero = tf.constant([0], tf.float32)
        rot_mat_inv = tf.concat([cos_val, sin_val, zero, -sin_val, cos_val, zero, zero, zero, one], axis=0)
        rot_mat_inv = tf.reshape(rot_mat_inv, [3,3])
        return rot_mat_inv
    angle = float(angle) * tf.random.normal([1],dtype='float32')
    rot_mat_inv = get_rotation_mat_inv(angle)
    return transform(image, rot_mat_inv, image_shape)


# In[ ]:


def GridMask(image_height, image_width, d1, d2, rotate_angle=1, ratio=0.5):
    h, w = image_height, image_width
    hh = int(np.ceil(np.sqrt(h*h+w*w)))
    hh = hh+1 if hh%2==1 else hh
    d = tf.random.uniform(shape=[], minval=d1, maxval=d2, dtype=tf.int32)
    l = tf.cast(tf.cast(d,tf.float32)*ratio+0.5, tf.int32)

    st_h = tf.random.uniform(shape=[], minval=0, maxval=d, dtype=tf.int32)
    st_w = tf.random.uniform(shape=[], minval=0, maxval=d, dtype=tf.int32)

    y_ranges = tf.range(-1 * d + st_h, -1 * d + st_h + l)
    x_ranges = tf.range(-1 * d + st_w, -1 * d + st_w + l)

    for i in range(0, hh//d+1):
        s1 = i * d + st_h
        s2 = i * d + st_w
        y_ranges = tf.concat([y_ranges, tf.range(s1,s1+l)], axis=0)
        x_ranges = tf.concat([x_ranges, tf.range(s2,s2+l)], axis=0)

    x_clip_mask = tf.logical_or(x_ranges < 0 , x_ranges > hh-1)
    y_clip_mask = tf.logical_or(y_ranges < 0 , y_ranges > hh-1)
    clip_mask = tf.logical_or(x_clip_mask, y_clip_mask)

    x_ranges = tf.boolean_mask(x_ranges, tf.logical_not(clip_mask))
    y_ranges = tf.boolean_mask(y_ranges, tf.logical_not(clip_mask))

    hh_ranges = tf.tile(tf.range(0,hh), [tf.cast(tf.reduce_sum(tf.ones_like(x_ranges)), tf.int32)])
    x_ranges = tf.repeat(x_ranges, hh)
    y_ranges = tf.repeat(y_ranges, hh)

    y_hh_indices = tf.transpose(tf.stack([y_ranges, hh_ranges]))
    x_hh_indices = tf.transpose(tf.stack([hh_ranges, x_ranges]))

    y_mask_sparse = tf.SparseTensor(tf.cast(y_hh_indices, tf.int64),  tf.zeros_like(y_ranges), [hh, hh])
    y_mask = tf.sparse.to_dense(y_mask_sparse, 1, False)

    x_mask_sparse = tf.SparseTensor(tf.cast(x_hh_indices, tf.int64), tf.zeros_like(x_ranges), [hh, hh])
    x_mask = tf.sparse.to_dense(x_mask_sparse, 1, False)

    mask = tf.expand_dims( tf.clip_by_value(x_mask + y_mask, 0, 1), axis=-1)

    mask = random_rotate(mask, rotate_angle, [hh, hh, 1])
    mask = tf.image.crop_to_bounding_box(mask, (hh-h)//2, (hh-w)//2, image_height, image_width)

    return mask


# In[ ]:


def apply_grid_mask(image, image_shape, PROBABILITY = gridmask_rate):
    AugParams = {
        'd1' : 100,
        'd2': 160,
        'rotate' : 45,
        'ratio' : 0.3
    }
    
        
    mask = GridMask(image_shape[0], image_shape[1], AugParams['d1'], AugParams['d2'], AugParams['rotate'], AugParams['ratio'])
    if image_shape[-1] == 3:
        mask = tf.concat([mask, mask, mask], axis=-1)
        mask = tf.cast(mask,tf.float32)
        P = tf.cast(tf.random.uniform([], 0, 1) <= PROBABILITY, tf.int32)
    if P==1:
        return image*mask
    else:
        return image

def gridmask(img_batch, label_batch):
    return apply_grid_mask(img_batch, (img_size,img_size, 3)), label_batch


# In[ ]:


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

cfg = dict(
transform_prob    =  1.0,
rot               = 180.0,
shr               =   2.0,
hzoom             =   8.0,
wzoom             =   8.0,
hshift            =   8.0,
wshift            =   8.0,
)

def transform_shear_rot(image,cfg):
    
    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    # output - image randomly rotated, sheared, zoomed, and shifted
    DIM = img_size
    XDIM = DIM%2 #fix for size 331
    
    rot = cfg['rot'] * tf.random.normal([1],dtype='float32')
    shr = cfg['shr'] * tf.random.normal([1],dtype='float32') 
    h_zoom = 1.0 + tf.random.normal([1],dtype='float32')/cfg['hzoom']
    w_zoom = 1.0 + tf.random.normal([1],dtype='float32')/cfg['wzoom']
    h_shift = cfg['hshift'] * tf.random.normal([1],dtype='float32') 
    w_shift = cfg['wshift'] * tf.random.normal([1],dtype='float32') 

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
    d = tf.gather_nd(image,tf.transpose(idx3))
        
    return tf.reshape(d,[DIM,DIM,3])


# In[ ]:


def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU
    image = tf.image.resize(image, [img_size,img_size])
    return image

def decode_image2(filename,label=None, image_size=(img_size,img_size)) :
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits,channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image,image_size)
    if label == None :
        return image
    else :
        return image, label


def data_augment(image, label=None,seed = 2020):
    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),
    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part
    # of the TPU while the TPU itself is computing gradients.
    image = tf.image.random_flip_left_right(image,seed=seed)
    image = tf.image.random_flip_up_down(image,seed=seed)
    #image = tf.keras.preprocessing.image.random_rotation(image,45) 
    #if transforms :
    #    image = transform_shear_rot(image,cfg)
    if crop_size :   
        image = tf.image.random_crop(image, size=[crop_size, crop_size, 3], seed=seed)
    if bool_random_brightness:
        image = tf.image.random_brightness(image,0.2)
    if bool_random_contrast:
        image = tf.image.random_contrast(image,0.6,1.4)
    if bool_random_hue:
        image = tf.image.random_hue(image,0.07)
    if bool_random_saturation:
        image = tf.image.random_saturation(image,0.5,1.5)
    if random_blackout :
        image= transform_random_blackout(image)
        
    if label == None :
        return image
    else :
        return image,label 


# In[ ]:


def create_train_data(train_paths,train_labels) :
    
    
    train_dataset=(tf.data.Dataset
    .from_tensor_slices((train_paths,train_labels.astype(np.float32)))
    .map(decode_image2,num_parallel_calls = AUTO)
    .map(data_augment,num_parallel_calls = AUTO)
    .repeat()
    .shuffle(512)
    .batch(BATCH_SIZE)
    .prefetch(AUTO))
    
    if cutmix_rate :
        train_dataset = train_dataset.map(cutmix,num_parallel_calls = AUTO) 
    if mixup_rate : 
        train_dataset = train_dataset.map(mixup, num_parallel_calls = AUTO)
    if gridmask_rate:
        train_dataset =train_dataset.map(gridmask, num_parallel_calls=AUTO)    
     
    return train_dataset 

def create_validation_data(valid_paths,valid_labels) :
    valid_data = (
        tf.data.Dataset
        .from_tensor_slices((valid_paths,valid_labels))
        .map(decode_image2, num_parallel_calls = AUTO)
        #.map(data_augment, num_parallel_calls= AUTO)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
        
    ) 
    return valid_data


# In[ ]:


#TRAINING_FILENAMES = tf.io.gfile.glob(GCS_DS_PATH2 + '/tfrecords/train*')
#TEST_FILENAMES = tf.io.gfile.glob(GCS_DS_PATH2 + '/tfrecords/test*')

TRAINING_FILENAMES = tf.io.gfile.glob(GCS_DS_PATH1 + '/train*')
#TEST_FILENAMES = tf.io.gfile.glob('../input/siim-isic-melanoma-classification/tfrecords/test*')
TEST_FILENAMES = tf.io.gfile.glob(GCS_DS_PATH1 + '/test*')

#TRAINING_FILENAMES = tf.io.gfile.glob('../input/512x512-melanoma-tfrecords-70k-images/train*')
#TEST_FILENAMES = tf.io.gfile.glob('../input/512x512-melanoma-tfrecords-70k-images/test*')

IMAGE_SIZE = [512, 512] 


# In[ ]:


#Read train tf Records :
def read_labeled_tfrecord(example):
    
    
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "target": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element
    }
 
    
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    label = tf.cast(example['target'], tf.float32)
    return image, label

#Read test tf Records :
def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "image_name": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element
        # class is missing, this competitions's challenge is to predict flower classes for the test dataset
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    idnum = example['image_name']
    return image, idnum # returns a dataset of image(s)

def load_dataset(filenames, labeled=True, ordered=False):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.

    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)
    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
    return dataset

   
    
def get_training_dataset():
    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)# prefetch next batch while training (autotune prefetch buffer size)
    
    if cutmix_rate :
        dataset = dataset.map(cutmix,num_parallel_calls = AUTO) 
    if mixup_rate : 
        dataset = dataset.map(mixup, num_parallel_calls = AUTO)
    if rotation :
        dataset = dataset.map(transform_rotation, num_parallel_calls = AUTO)
   # if blackout :
   #     dataset = dataset.map(transform_random_blackout, num_parallel_calls = AUTO)
    if gridmask_rate:
        dataset = dataset.map(gridmask, num_parallel_calls=AUTO) 
    
    
    return dataset   

def get_test_dataset(ordered=False,aug=False):
    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    
    if aug == True :        
        
        dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
        
    return dataset


# In[ ]:


def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
print('Dataset: {} training images, {} unlabeled test images'.format(NUM_TRAINING_IMAGES, NUM_TEST_IMAGES))


# In[ ]:


lr_start = 0.00001

lr_max = 0.00001 * strategy.num_replicas_in_sync
lr_min = 0.00001 
lr_rampup_epochs = 8
lr_sustain_epochs = 3
lr_exp_decay = .6


def lrfn(epoch) :
    if epoch < lr_rampup_epochs :
        lr = lr_start + (lr_max-lr_min) / lr_rampup_epochs * epoch
    elif epoch < lr_rampup_epochs + lr_sustain_epochs :
        lr = lr_max
    else :
        lr = lr_min + (lr_max - lr_min) * lr_exp_decay**(epoch - lr_sustain_epochs - lr_rampup_epochs)
    return lr

lr_callback = LearningRateScheduler(lrfn, verbose = True)

rng = [i for i in range(EPOCHS)]
y = [lrfn(x) for x in rng]

from matplotlib import pyplot as plt

plt.plot(rng,y)
print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))


# In[ ]:


reduce_lr =  ReduceLROnPlateau(monitor = "val_loss", factor = 0.5, patience = 10,
  verbose = 0, mode = "auto", epsilon = 1e-04, cooldown = 0,
  min_lr = 1e-5)


# In[ ]:


es = EarlyStopping(monitor = "val_loss" , verbose = 1 , mode = 'min' , patience = 50 )


# In[ ]:


mc = ModelCheckpoint('best_model.h5', monitor = 'loss' , mode = 'min', verbose = 1 , save_best_only = True)


# In[ ]:


checkpoint_path = "best_model.h5"

swa_mc = tfa.callbacks.AverageModelCheckpoint(filepath=checkpoint_path, save_weights_only=False,
                                                   update_weights=True)


"""start_epoch = 10
swa_mc = SWA(start_epoch=start_epoch, 
          lr_schedule='cyclic', 
          swa_lr=1e-6,
          swa_lr2=1e-5,
          swa_freq=3,
        #batch_size = BATCH_SIZE,
          verbose=1)"""


# In[ ]:


focal_loss = True
label_smoothing = 0
SWA = False


# In[ ]:


def get_model_generalized(name,trainable_layers=20,opt='adam',lr=0.001):
    if name == 'EfficientNetB7' :
        base_model = efn.EfficientNetB7(weights='imagenet',
                                        include_top = False,
                                        input_shape=(img_size,img_size,3)
                                       )
    elif name == 'EfficientNetB5' :
        base_model = efn.EfficientNetB5(weights='imagenet',
                                        include_top = False,
                                        input_shape=(img_size,img_size,3)
                                       )
    elif name == 'EfficientNetB3' :
        base_model = efn.EfficientNetB3(weights='imagenet',
                                        include_top = False,
                                        input_shape=(img_size,img_size,3)
                                       )
    elif name == 'EfficientNetB1' :
        base_model = efn.EfficientNetB1(weights='imagenet',
                                        include_top = False,
                                        input_shape=(img_size,img_size,3)
                                       )    
    elif name == 'DenseNet' :
        base_model = DenseNet201(weights='imagenet',include_top=False,input_shape=(img_size,img_size,3))
    elif name == 'MobileNet' :
        base_model = MobileNet(weights = 'imagenet', include_top=False,input_shape=(img_size,img_size,3))
    elif name == 'Inception' :
        base_model = InceptionV3(weights = 'imagenet',include_top=False,input_shape=(img_size,img_size,3))
    elif name == 'ResNet' :
        base_model = ResNet50(weights = 'imagenet',include_top=False,input_shape=(img_size,img_size,3))
    elif name == 'Incepresnet' :
        base_model = InceptionResNetV2(weights = 'imagenet',include_top=False,input_shape=(img_size,img_size,3))
    elif name == 'SEResNet50' :
        seresnet50, _ = Classifiers.get('seresnet50')
        base_model =  seresnet50(weights = 'imagenet', include_top = False, input_shape = (img_size,img_size,3))
    elif name == 'SEResNext50' :
        seresnext50 , _ = Classifiers.get('seresnext50')
        base_model = seresnext50(weights = 'imagenet', include_top = False,input_shape = (img_size,img_size,3))
    elif name == 'NasNetLarge' :
        nasnet , _ = Classifiers.get('nasnetlarge')
        base_model = nasnet(weights= 'imagenet', include_top = False , input_shape = (img_size,img_size,3))    
            
    base_model.trainable = True
    for layer in base_model.layers[:-trainable_layers] :
        layer.trainable = True
    layer = base_model.output
    layer = L.GlobalAveragePooling2D()(layer)
    layer = L.Dense(1024,activation='relu')(layer)
    layer = L.Dropout(0.4)(layer)
    predictions = L.Dense(nb_classes,activation='sigmoid')(layer)
   # predictions = tf.cast(predictions,tf.float32)
    model = Model(inputs = base_model.input, outputs=predictions)
    
    if focal_loss : 
        loss= tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.AUTO)
    if label_smoothing :
        loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing)
    else :
        loss = 'binary_crossentropy'
    
    
    if opt == 'RMSprop' :
        opt = optimizers.RMSprop(learning_rate = lr)
    elif SWA == True :
        opt = tf.keras.optimizers.Adam(lr=1e-5) # roll back
        opt = tfa.optimizers.SWA(opt)
    else :
        opt =  tf.keras.optimizers.Adam(lr=1e-5) # roll back

        
    
    model.compile(optimizer=opt,loss=loss,metrics=['accuracy',tf.keras.metrics.AUC()])  
    return model


# In[ ]:


from sklearn.utils import class_weight 
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(train.target),
                                                 train.target)


# In[ ]:


class_weights = { 0 :  0.50897302 , 1 : 28.36130137 }


# In[ ]:


classweights =[item for k in class_weights for item in (k, class_weights[k])]


# In[ ]:


with strategy.scope() :
    model = get_model_generalized('EfficientNetB1')

history = model.fit(
    get_training_dataset(),
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    callbacks = [],
    #class_weight = classweights,
)


# In[ ]:


with strategy.scope() :
    model = get_model_generalized('EfficientNetB7')

history = model.fit(
    train_balanced_data,
    steps_per_epoch= len(Y_train) // BATCH_SIZE,
    epochs=50,
    callbacks = [],
    #class_weight = classweights,
    validation_data = valid_balanced_data
)


# In[ ]:


with strategy.scope() :
    model = get_model_generalized('EfficientNetB3')

history = model.fit(
    get_training_dataset(),
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    callbacks = [],
    #class_weight = classweights
)


# In[ ]:


with strategy.scope() :
    model = get_model_generalized('EfficientNetB5')

history = model.fit(
    get_training_dataset(),
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    #callbacks = [swa_mc],
   # batch_size = BATCH_SIZE
    #class_weight = classweights
)


# In[ ]:


with strategy.scope() :
    model = get_model_generalized('EfficientNetB5')

history = model.fit(
    train_balanced_data,
    steps_per_epoch= len(Y_train) // BATCH_SIZE,
    epochs=50,
    callbacks = [],
    #class_weight = classweights,
    validation_data = valid_balanced_data
)


# In[ ]:


with strategy.scope() :
    model = get_model_generalized('ResNet')

history = model.fit(
    get_training_dataset(),
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS-10,
    callbacks = [],
    #class_weight = classweights
)


# In[ ]:


with strategy.scope() :
    model = get_model_generalized('SEResNet50')

history = model.fit(
    get_training_dataset(),
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS-10,
    callbacks = [],
    #class_weight = classweights
)


# In[ ]:


with strategy.scope() :
    model = get_model_generalized('SEResNext50')

history = model.fit(
    get_training_dataset(),
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS-5,
    callbacks = [],
    #class_weight = classweights
)


# In[ ]:





# In[ ]:


with strategy.scope() :
    model = get_model_generalized('DenseNet')

history = model.fit(
    get_training_dataset(),
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=12,
    callbacks = [],
    #class_weight = classweights
)


# In[ ]:


with strategy.scope() :
    model = get_model_generalized('MobileNet')

history = model.fit(
    get_training_dataset(),
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    callbacks = [],
    #class_weight = classweights
)


# In[ ]:


with strategy.scope() :
    model = get_model_generalized('NasNetLarge')

history = model.fit(
    get_training_dataset(),
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=15,
    callbacks = [],
    #class_weight = classweights
)


# In[ ]:


del model
import gc 
gc.collect()


# In[ ]:


label_smoothing = 0
def LabelSmoothing(encodings , alpha):
    K = encodings.shape[1]
    y_ls = (1 - alpha) * encodings + alpha / K
    return y_ls


# In[ ]:


label_smoothing = 0
test_ds = get_test_dataset(ordered=True)
#model.load_weights('best_model.h5')
print('Computing predictions...')
test_images_ds = test_ds.map(lambda image, idnum: image)
probabilities = model.predict(test_images_ds)
if label_smoothing :
    probabilities = LabelSmoothing(probabilities,label_smoothing)
probabilities =probabilities.flatten()
print(probabilities)
print('Generating submission.csv file...')
test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U')
pred_df = pd.DataFrame({'image_name': test_ids, 'target': probabilities})
del sub['target']
sub = sub.merge(pred_df, on='image_name')
sub.to_csv('densenet_468img_size_12epochs_fl_0.4cutmix_0.4gridmask_20trainable.csv', index=False)
sub.head()


# In[ ]:


sub[sub['target'] > 0.5]


# In[ ]:


import gc
del model
gc.collect()


# In[ ]:





# In[ ]:


sub[sub['target']  >  0.8] 


# In[ ]:


sub[sub['target'] < 0.01]


# In[ ]:


#model.load_weights('best_model.h5')
TTA_NUM = 3
probabilities = []
for i in range(TTA_NUM):
    print(f'TTA Number: {i}\n')
    test_ds = get_test_dataset(ordered=True,aug=True)
    test_images_ds = test_ds.map(lambda image, idnum: image)
    probabilities.append(model.predict(test_images_ds).flatten())


# In[ ]:


tab = np.zeros((len(probabilities[1]),1))
for i in range(0,len(probabilities[1])) :
    for j in range(0,TTA_NUM) :
        tab[i] = tab[i] + probabilities[j][i]
tab = tab / TTA_NUM              
test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U')
pred_df = pd.DataFrame({'image_name': test_ids, 'target': np.squeeze(tab)})
del sub['target']
sub = sub.merge(pred_df, on='image_name')
sub.to_csv('e3_468_20epochs_fl_20trainable_3tta_heavy_augs.csv', index=False)
sub.head()


# In[ ]:


# batch
def cutmix_v2(data, label, PROBABILITY = cutmix_rate):
    # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
    # output - a batch of images with cutmix applied
    
    DIM = img_size    
    imgs = []; labs = []
    
    for j in range(BATCH_SIZE):
        
        #random_uniform( shape, minval=0, maxval=None)        
        # DO CUTMIX WITH PROBABILITY DEFINED ABOVE
        P = tf.cast(tf.random.uniform([], 0, 1) <= PROBABILITY, tf.int32)
        
        # CHOOSE RANDOM IMAGE TO CUTMIX WITH
        k = tf.cast(tf.random.uniform([], 0, BATCH_SIZE), tf.int32)
        
        # CHOOSE RANDOM LOCATION
        x = tf.cast(tf.random.uniform([], 0, DIM), tf.int32)
        y = tf.cast(tf.random.uniform([], 0, DIM), tf.int32)
        
        # Beta(1, 1)
        b = tf.random.uniform([], 0, 1) # this is beta dist with alpha=1.0
        

        WIDTH = tf.cast(DIM * tf.math.sqrt(1-b),tf.int32) * P
        ya = tf.math.maximum(0,y-WIDTH//2)
        yb = tf.math.minimum(DIM,y+WIDTH//2)
        xa = tf.math.maximum(0,x-WIDTH//2)
        xb = tf.math.minimum(DIM,x+WIDTH//2)
        
        # MAKE CUTMIX IMAGE
        one = data['image_data'][j,ya:yb,0:xa,:]
        two = data['image_data'][k,ya:yb,xa:xb,:]
        three = data['image_data'][j,ya:yb,xb:DIM,:]        
        #ya:yb
        middle = tf.concat([one,two,three],axis=1)

        img = tf.concat([data['image_data'][j,0:ya,:,:],middle,data['image_data'][j,yb:DIM,:,:]],axis=0)
        imgs.append(img)
        
        # MAKE CUTMIX LABEL
        a = tf.cast(WIDTH*WIDTH/DIM/DIM,tf.float32)
        lab1 = label[j,]
        lab2 = label[k,]
        labs.append((1-a)*lab1 + a*lab2)

    image2 = tf.reshape(tf.stack(imgs),(BATCH_SIZE,DIM,DIM,3))
    label2 = tf.reshape(tf.stack(labs),(BATCH_SIZE, nb_classes))
    
    data['image_data'] = image2
    
    return data,label2


# In[ ]:


def gridmask_v2(data, label_batch):
    return apply_grid_mask_v2(data, (img_size,img_size, 3)), label_batch


# In[ ]:


def apply_grid_mask_v2(data, image_shape, PROBABILITY = gridmask_rate):
    AugParams = {
        'd1' : 100,
        'd2': 160,
        'rotate' : 45,
        'ratio' : 0.3
    }
    
        
    mask = GridMask(image_shape[0], image_shape[1], AugParams['d1'], AugParams['d2'], AugParams['rotate'], AugParams['ratio'])
    if image_shape[-1] == 3:
        mask = tf.concat([mask, mask, mask], axis=-1)
        mask = tf.cast(mask,tf.float32)
        P = tf.cast(tf.random.uniform([], 0, 1) <= PROBABILITY, tf.int32)
    if P==1:
        data['image_data'] = data['image_data']  * mask
        return data
    else:
        return data


# In[ ]:


#Read train tf Records :
def read_labeled_tfrecord(example):
    
    
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "patient_id" : tf.io.FixedLenFeature([],tf.int64) ,
         "sex": tf.io.FixedLenFeature([],tf.int64),
      "age_approx": tf.io.FixedLenFeature([],tf.int64),
      "anatom_site_general_challenge": tf.io.FixedLenFeature([],tf.int64),
     # "source": tf.io.FixedLenFeature([],tf.int64),
        "target": tf.io.FixedLenFeature([], tf.int64), # shape [] means single element  
    }
 
    
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    
    data = {}
    
    data['patient_id'] = tf.cast(example['patient_id'],tf.float32)
    data['sex'] = tf.cast(example['patient_id'],tf.float32)
    data['age_approx'] = tf.cast(example['age_approx'],tf.float32)
    #data['anatom_site_general_challenge'] = tf.cast(tf.one_hot(example['anatom_site_general_challenge'], 7), tf.int32)
    
    data['anatom_site_general_challenge'] = tf.cast(example['anatom_site_general_challenge'],tf.float32)
    #data['source'] = tf.cast(example['source'],tf.float32)
    
    
    label = tf.cast(example['target'], tf.float32)
    return image, label , data

#Read test tf Records :
def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
      "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
      "image_name": tf.io.FixedLenFeature([], tf.string), # shape [] means single element
      "patient_id" : tf.io.FixedLenFeature([],tf.int64) ,
      "sex": tf.io.FixedLenFeature([],tf.int64),
      "age_approx": tf.io.FixedLenFeature([],tf.int64),
      "anatom_site_general_challenge": tf.io.FixedLenFeature([],tf.int64),
      #"source": tf.io.FixedLenFeature([],tf.int64),

    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    img_name = example['image_name']
    
    data = {}
    
    data['patient_id'] = tf.cast(example['patient_id'],tf.float32)
    data['sex'] = tf.cast(example['patient_id'],tf.float32)
    data['age_approx'] = tf.cast(example['age_approx'],tf.float32)
    
    #data['anatom_site_general_challenge'] = tf.cast(tf.one_hot(example['anatom_site_general_challenge'], 7), tf.int32)
    
    
    
    data['anatom_site_general_challenge'] = tf.cast(example['anatom_site_general_challenge'],tf.float32)
    #data['source'] = tf.cast(example['source'],tf.float32)
    
    return image, img_name , data # returns a dataset of image(s)

def load_dataset(filenames, labeled=True, ordered=False):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.

    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)
    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
    
    return dataset


def train_setup(image,label,data) :
    #anatom = [tf.cast(data['anatom_site_general_challenge'][i], dtype = tf.float32) for i in range(7)]
    tab_data=[tf.cast(data[tfeat], dtype=tf.float32) for tfeat in ['sex','age_approx','patient_id','anatom_site_general_challenge']]#,'source','anatom_site_general_challenge',
    tabular=tf.stack(tab_data) #+anatom
    
    return {'image_data' : image , 'meta_data' : tabular} , label

def test_setup(image,image_name,data) :
    #anatom = [tf.cast(data['anatom_site_general_challenge'][i], dtype = tf.float32) for i in range(7)]
    tab_data=[tf.cast(data[tfeat], dtype=tf.float32) for tfeat in ['sex','age_approx','patient_id','anatom_site_general_challenge']] #,'source','anatom_site_general_challenge',
    tabular=tf.stack(tab_data) #+anatom
    return {'image_data' : image , 'meta_data' : tabular } , image_name


# In[ ]:


def data_augment(data, label=None,seed = 2020):
    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),
    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part
    # of the TPU while the TPU itself is computing gradients.
    
    data['image_data'] = tf.image.random_flip_left_right(data['image_data'],seed=seed)
    data['image_data'] = tf.image.random_flip_up_down(data['image_data'],seed=seed)
   
    #data['image_data'] = tf.keras.preprocessing.image.random_rotation(data['image_data'],45) 
    #if transforms :
    #    data['image_data'] = transform_shear_rot(data['image_data'],cfg)
    
  
     
    if crop_size :   
        data['image_data'] = tf.image.random_crop(data['image_data'], size=[crop_size, crop_size, 3], seed=seed)
    if bool_random_brightness:
        data['image_data'] = tf.image.random_brightness(data['image_data'],0.2)
    if bool_random_contrast:
        data['image_data'] = tf.image.random_contrast(data['image_data'],0.6,1.4)
    if bool_random_hue:
        data['image_data'] = tf.image.random_hue(data['image_data'],0.07)
    if bool_random_saturation:
        data['image_data'] = tf.image.random_saturation(data['image_data'],0.5,1.5)
    if random_blackout :
        data['image_data'] = transform_random_blackout(data['image_data'])
        
    if label == None :
        return data
    else :
        return data,label 


# In[ ]:


'''def crop_microscope(img_to_crop):
    pad_y = img_to_crop.shape[0]//200 
    pad_x = img_to_crop.shape[1]//200
    img = img_to_crop[pad_y:-pad_y, pad_y:-pad_y,:]
    
#cropping 0.5% from every side, because some microscope images
#have frames along the edges so cv2.boundingRect crops by frame, 
#but not by needed part of the image.
    
    
    gray = cv2.cvtColor(np.float32(img),cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,50,255,cv2.THRESH_BINARY) 
    x,y,w,h = cv2.boundingRect(thresh) #getting crop points
    
#since we cropped borders we need to uncrop it back 
    if y!=0: 
        y = y+pad_y
    if h == thresh.shape[0]:
        h = h+pad_y
    if x !=0:
        x = x +pad_x
    if w == thresh.shape[1]:
        w = w + pad_x
    h = h+pad_y
    w = w + pad_x
    img = img_to_crop[y:y+h,x:x+w]
    return img


def crop_circle(image) :
    out = tf.py_function(crop_microscope, [image], tf.float32)
    return  out'''
    
    
class Microscope:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, data , label):
        for k in range(BATCH_SIZE) :
            if random.random() < self.p:
                circle = cv2.circle((np.ones(data['image_data'][k].shape) * 255).astype(np.uint8),
                            (data['image_data'][k].shape[0]//2, data['image_data'][k][1]//2),
                            random.randint(data['image_data'][k].shape[0]//2 - 3, data['image_data'][k].shape[0]//2 + 15),
                            (0, 0, 0),
                            -1)

                mask = circle - 255
                data['image_data'][k] = np.multiply(data['image_data'][k], mask)

        return data , label

    def __repr__(self):
        return f'{self.__class__.__name__}(p={self.p})'
    
micro = Microscope()   

class Microscope_test:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, data , image_name):
        if random.random() < self.p:
            circle = cv2.circle((np.ones(data['image_data'].shape) * 255).astype(np.uint8),
                        (data['image_data'].shape[0]//2, data['image_data'][1]//2),
                        random.randint(data['image_data'].shape[0]//2 - 3, data['image_data'].shape[0]//2 + 15),
                        (0, 0, 0),
                        -1)

            mask = circle - 255
            data['image_data'] = np.multiply(data['image_data'], mask)

        return data , image_name

    def __repr__(self):
        return f'{self.__class__.__name__}(p={self.p})'



class AdvancedHairAugmentation:
    def __init__(self, hairs: int = 4, hairs_folder: str = ""):
        self.hairs = hairs
        self.hairs_folder = hairs_folder

    def __call__(self, data,label):
        for k in range(BATCH_SIZE) :
            n_hairs = random.randint(0, self.hairs)

            if not n_hairs:
                return data , label

            height, width, _ = data['image_data'][k].shape  # target image width and height
            hair_images = [im for im in os.listdir(self.hairs_folder) if 'png' in im]

            for _ in range(n_hairs):
                hair = cv2.imread(os.path.join(self.hairs_folder, random.choice(hair_images)))
                hair = cv2.flip(hair, random.choice([-1, 0, 1]))
                hair = cv2.rotate(hair, random.choice([0, 1, 2]))

                h_height, h_width, _ = hair.shape  # hair image width and height
                roi_ho = random.randint(0, data['image_data'][k].shape[0] - hair.shape[0])
                roi_wo = random.randint(0, data['image_data'][k].shape[1] - hair.shape[1])
                roi = data['image_data'][k][roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width]

                img2gray = cv2.cvtColor(np.float32(hair), cv2.COLOR_BGR2GRAY)
                ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
                mask_inv = cv2.bitwise_not(mask)
                img_bg = cv2.bitwise_and(np.float32(roi),np.float32(roi) , mask=mask_inv)#
                hair_fg = cv2.bitwise_and(np.float32(hair),np.float32(hair), mask=mask) #

                dst = cv2.add(img_bg, hair_fg)
                data['image_data'][k][roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width] = dst

        return data , label
    
hair = AdvancedHairAugmentation(hairs_folder = '../input/melanoma-hairs')


# In[ ]:


def get_training_dataset():
    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)
    dataset = dataset.map(train_setup , num_parallel_calls = AUTO)
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)# prefetch next batch while training (autotune prefetch buffer size)
    
    if hair_aug :
        dataset = dataset.map(hair, num_parallel_calls = AUTO)
    if micro_aug :
        dataset = dataset.map(micro,num_parallel_calls = AUTO)
    if cutmix_rate :
        dataset = dataset.map(cutmix_v2,num_parallel_calls = AUTO) 
    if mixup_rate : 
        dataset = dataset.map(mixup, num_parallel_calls = AUTO)
    if rotation :
        dataset = dataset.map(transform_rotation, num_parallel_calls = AUTO)
   # if blackout :
   #     dataset = dataset.map(transform_random_blackout, num_parallel_calls = AUTO)
    if gridmask_rate:
        dataset = dataset.map(gridmask_v2, num_parallel_calls=AUTO)  
    
    return dataset   

def get_test_dataset(ordered=False,aug=False):
    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    dataset = dataset.map(test_setup , num_parallel_calls = AUTO)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    
    if aug == True :        
        dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
        #dataset = dataset.map(micro, num_parallel_calls = AUTO)
    return dataset


# In[ ]:


focal_loss = True
label_smoothing = 0
SWA = False


# In[ ]:


def roc_auc_loss(
    labels,
    logits,
    weights=1.0,
    surrogate_type='xent',
    scope=None):
  """Computes ROC AUC loss.
  The area under the ROC curve is the probability p that a randomly chosen
  positive example will be scored higher than a randomly chosen negative
  example. This loss approximates 1-p by using a surrogate (either hinge loss or
  cross entropy) for the indicator function. Specifically, the loss is:
    sum_i sum_j w_i*w_j*loss(logit_i - logit_j)
  where i ranges over the positive datapoints, j ranges over the negative
  datapoints, logit_k denotes the logit (or score) of the k-th datapoint, and
  loss is either the hinge or log loss given a positive label.
  Args:
    labels: A `Tensor` of shape [batch_size] or [batch_size, num_labels].
    logits: A `Tensor` with the same shape and dtype as `labels`.
    weights: Coefficients for the loss. Must be a scalar or `Tensor` of shape
      [batch_size] or [batch_size, num_labels].
    surrogate_type: Either 'xent' or 'hinge', specifying which upper bound
      should be used for the indicator function.
    scope: Optional scope for `name_scope`.
  Returns:
    loss: A `Tensor` of the same shape as `logits` with the component-wise loss.
    other_outputs: An empty dictionary, for consistency.
  Raises:
    ValueError: If `surrogate_type` is not `xent` or `hinge`.
  """
  with tf.name_scope(scope, 'roc_auc', [labels, logits, weights]):
    # Convert inputs to tensors and standardize dtypes.
    labels, logits, weights, original_shape = _prepare_labels_logits_weights(
        labels, logits, weights)

    # Create tensors of pairwise differences for logits and labels, and
    # pairwise products of weights. These have shape
    # [batch_size, batch_size, num_labels].
    logits_difference = tf.expand_dims(logits, 0) - tf.expand_dims(logits, 1)
    labels_difference = tf.expand_dims(labels, 0) - tf.expand_dims(labels, 1)
    weights_product = tf.expand_dims(weights, 0) * tf.expand_dims(weights, 1)

    signed_logits_difference = labels_difference * logits_difference
    raw_loss = util.weighted_surrogate_loss(
        labels=tf.ones_like(signed_logits_difference),
        logits=signed_logits_difference,
        surrogate_type=surrogate_type)
    weighted_loss = weights_product * raw_loss

    # Zero out entries of the loss where labels_difference zero (so loss is only
    # computed on pairs with different labels).
    loss = tf.reduce_mean(tf.abs(labels_difference) * weighted_loss, 0) * 0.5
    loss = tf.reshape(loss, original_shape)
    return loss, {}


def custom_loss_wrapper() :
    def roc_auc_loss(
    labels,
    logits,
    weights=1.0,
    surrogate_type='xent',
    scope=None):
        
      """Computes ROC AUC loss.
      The area under the ROC curve is the probability p that a randomly chosen
      positive example will be scored higher than a randomly chosen negative
      example. This loss approximates 1-p by using a surrogate (either hinge loss or
      cross entropy) for the indicator function. Specifically, the loss is:
        sum_i sum_j w_i*w_j*loss(logit_i - logit_j)
      where i ranges over the positive datapoints, j ranges over the negative
      datapoints, logit_k denotes the logit (or score) of the k-th datapoint, and
      loss is either the hinge or log loss given a positive label.
      Args:
        labels: A `Tensor` of shape [batch_size] or [batch_size, num_labels].
        logits: A `Tensor` with the same shape and dtype as `labels`.
        weights: Coefficients for the loss. Must be a scalar or `Tensor` of shape
          [batch_size] or [batch_size, num_labels].
        surrogate_type: Either 'xent' or 'hinge', specifying which upper bound
          should be used for the indicator function.
        scope: Optional scope for `name_scope`.
      Returns:
        loss: A `Tensor` of the same shape as `logits` with the component-wise loss.
        other_outputs: An empty dictionary, for consistency.
      Raises:
        ValueError: If `surrogate_type` is not `xent` or `hinge`.
      """
      with tf.name_scope(scope, 'roc_auc', [labels, logits, weights]):
        # Convert inputs to tensors and standardize dtypes.
        labels, logits, weights, original_shape = _prepare_labels_logits_weights(
            labels, logits, weights)

        # Create tensors of pairwise differences for logits and labels, and
        # pairwise products of weights. These have shape
        # [batch_size, batch_size, num_labels].
        logits_difference = tf.expand_dims(logits, 0) - tf.expand_dims(logits, 1)
        labels_difference = tf.expand_dims(labels, 0) - tf.expand_dims(labels, 1)
        weights_product = tf.expand_dims(weights, 0) * tf.expand_dims(weights, 1)

        signed_logits_difference = labels_difference * logits_difference
        raw_loss = util.weighted_surrogate_loss(
            labels=tf.ones_like(signed_logits_difference),
            logits=signed_logits_difference,
            surrogate_type=surrogate_type)
        weighted_loss = weights_product * raw_loss

        # Zero out entries of the loss where labels_difference zero (so loss is only
        # computed on pairs with different labels).
        loss = tf.reduce_mean(tf.abs(labels_difference) * weighted_loss, 0) * 0.5
        loss = tf.reshape(loss, original_shape)
        return loss, {}
    return roc_auc_loss    


# In[ ]:


def get_model_generalized_v2(name,trainable_layers=20,opt='adam',lr=0.001):
   if name == 'EfficientNetB2' :
       base_model = efn.EfficientNetB2(weights='imagenet',
                                       include_top = False,
                                      )
   elif name == 'EfficientNetB5' :
       base_model = efn.EfficientNetB5(weights='imagenet',
                                       include_top = False,
                                      )
   elif name == 'EfficientNetB3' :
       base_model = efn.EfficientNetB3(weights='imagenet',
                                       include_top = False,
                                      )
   elif name == 'EfficientNetB1' :
       base_model = efn.EfficientNetB1(weights='imagenet',
                                       include_top = False,
                                      )    
   elif name == 'DenseNet' :
       base_model = DenseNet201(weights='imagenet',include_top=False)
   elif name == 'MobileNet' :
       base_model = MobileNet(weights = 'imagenet', include_top=False)
   elif name == 'Inception' :
       base_model = InceptionV3(weights = 'imagenet',include_top=False)
   elif name == 'ResNet' :
       base_model = ResNet50(weights = 'imagenet',include_top=False)
   elif name == 'Incepresnet' :
       base_model = InceptionResNetV2(weights = 'imagenet',include_top=False)
   elif name == 'SEResNet50' :
       seresnet50, _ = Classifiers.get('seresnet50')
       base_model =  seresnet50(weights = 'imagenet', include_top = False)
   elif name == 'SEResNext50' :
       seresnext50 , _ = Classifiers.get('seresnext50')
       base_model = seresnext50(weights = 'imagenet', include_top = False)
   elif name == 'NasNetLarge' :
       nasnet , _ = Classifiers.get('nasnetlarge')
       base_model = nasnet(weights= 'imagenet', include_top = False )    
           
   base_model.trainable = True
   for layer in base_model.layers[:-trainable_layers] :
       layer.trainable = True
       
   inp1 = L.Input(shape = (img_size,img_size, 3), name = 'image_data')
   inp2 = L.Input(shape = (4), name = 'meta_data')  
   
   
   layer1 = base_model(inp1)
   layer1 = L.GlobalAveragePooling2D()(layer1)
   layer1 = L.Dense(1024,activation='relu')(layer1)
   layer1 = L.Dropout(0.4)(layer1)
   layer1 = L.Dense(512,activation='relu')(layer1)
   layer1 = L.Dropout(0.4)(layer1)#0.4
   layer1 = L.Dense(256,activation='relu')(layer1)
   layer1 = L.Dropout(0.4)(layer1) #0.4
   
   #
   layer2 = L.Dense(128,activation = 'relu')(inp2)
   layer2 = L.Dropout(0.4)(layer2)
   layer2 = L.Dense(64, activation = 'relu')(layer2)
   layer2 = L.BatchNormalization()(layer2)
   layer2 = L.Dropout(0.3)(layer2) #0.3
   layer2 = L.Dense(32, activation = 'relu')(layer2)
   layer2 = L.BatchNormalization()(layer2)
   layer2 = L.Dropout(0.3)(layer2) #0.3
   #layer2 = L.Dense(16, activation = 'relu')(layer2)
   #layer2 = L.BatchNormalization()(layer2)
   #layer2 = L.Dropout(0.2)(layer2)
   
   concat = L.concatenate([layer1, layer2])
   
   concat = L.Dense(512,activation = 'relu')(concat) #512 is best 0.933
   concat = L.Dropout(0.4)(concat)
   
   #concat = L.Dense(256,activation = 'relu')(concat) #testing
   #concat = L.Dropout(0.3)(concat)    
   
   predictions = L.Dense(nb_classes,activation='sigmoid')(concat)
  # predictions = tf.cast(predictions,tf.float32)
   model = Model(inputs = [inp1,inp2], outputs=[predictions])
   
   if focal_loss : 
       loss= tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.AUTO)
   if label_smoothing :
       loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing)
   else :
       loss = 'binary_crossentropy'
   
   
   if opt == 'RMSprop' :
       opt = optimizers.RMSprop(learning_rate = lr)
   elif SWA == True :
       opt = tf.keras.optimizers.Adam(lr=1e-5) # roll back
       opt = tfa.optimizers.SWA(opt)
   else :
       opt =  tf.keras.optimizers.Adam(lr=1e-5) # roll back

       
   model.compile(optimizer=opt,loss=loss,metrics=['accuracy',tf.keras.metrics.AUC()])  
   return model


# In[ ]:


with strategy.scope() :
    model = get_model_generalized_v2('EfficientNetB2')

    history = model.fit(
        get_training_dataset(),
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        callbacks = [],
        #class_weight = classweights
)


# In[ ]:


label_smoothing = 0
test_ds = get_test_dataset(ordered=True)
#model.load_weights('best_model.h5')
print('Computing predictions...')
test_images_ds = test_ds.map(lambda image, idnum: image)
probabilities = model.predict(test_images_ds)
if label_smoothing :
    probabilities = LabelSmoothing(probabilities,label_smoothing)
probabilities =probabilities.flatten()
print(probabilities)
print('Generating submission.csv file...')
test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U')
pred_df = pd.DataFrame({'image_name': test_ids, 'target': probabilities})
del sub['target']
sub = sub.merge(pred_df, on='image_name')
sub.to_csv('E4_468img_size_20epochs_fl_v2_1024.4_512.4_256.4cnn128.4_64.3_32.3meta_cutmix_grid_512both_hair.csv', index=False)
sub.head()


# In[ ]:


#model.load_weights('best_model.h5')
TTA_NUM = 5
probabilities = []
for i in range(TTA_NUM):
    print(f'TTA Number: {i}\n')
    test_ds = get_test_dataset(ordered=True,aug=True)
    test_images_ds = test_ds.map(lambda image, idnum: image)
    probabilities.append(model.predict(test_images_ds).flatten())
tab = np.zeros((len(probabilities[1]),1))
for i in range(0,len(probabilities[1])) :
    for j in range(0,TTA_NUM) :
        tab[i] = tab[i] + probabilities[j][i]
tab = tab / TTA_NUM              
test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U')
pred_df = pd.DataFrame({'image_name': test_ids, 'target': np.squeeze(tab)})
del sub['target']
sub = sub.merge(pred_df, on='image_name')
sub.to_csv('e4_468_20epochs_fl_20trainable_5ttav2.csv', index=False)
sub.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




