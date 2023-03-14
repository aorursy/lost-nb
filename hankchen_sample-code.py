#!/usr/bin/env python
# coding: utf-8



get_ipython().run_line_magic('matplotlib', 'inline')
import os
import glob
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial

import pydicom as dicom
import nibabel as nib




# TODO: modify the following paths
train_image_folder = "../input/train-images/image/"
train_label_folder = "../input/train-labels/label/"
test_image_folder = "../input/test-images/image/"

train_list = os.listdir(train_image_folder)

# Ignore this data
if 'hmvsa0loxh3ek2y8rzmcyb6zrrh9mwyp' in train_list:
    train_list.remove('hmvsa0loxh3ek2y8rzmcyb6zrrh9mwyp')
    
print('Train data:', len(train_list))




def load_dicom_volume(src_dir, suffix='*.dcm'):
    """Load DICOM volume and get meta data.
    """
    encode_name = src_dir.split('/')[-1]
    # Read dicom files from the source directory
    # Sort the dicom slices in their respective order by slice location
    dicom_scans = [dicom.read_file(sp)                    for sp in glob.glob(os.path.join(src_dir, suffix))]
    # dicom_scans.sort(key=lambda s: float(s.SliceLocation))
    dicom_scans.sort(key=lambda s: float(s[(0x0020, 0x0032)][2]))

    # Convert to int16, should be possible as values should always be low enough
    # Volume image is in z, y, x order
    volume_image = np.stack([ds.pixel_array                              for ds in dicom_scans]).astype(np.int16)
    return encode_name, volume_image

def load_label(label_fpath, transpose=False):
    encode_name = label_fpath[-39: -7]
    label_data = nib.load(label_fpath)
    label_array = label_data.get_fdata()
    if transpose:
        label_array = np.transpose(label_array, axes=(2, 1, 0))
    return encode_name, label_array




train_image_npz_folder = './npz/train_images/'

if not os.path.exists(train_image_npz_folder):
    os.makedirs(train_image_npz_folder)




for encode in tqdm.tqdm(train_list):
    _, volume_image = load_dicom_volume(os.path.join(train_image_folder, encode))
    npz_folder = os.path.join(train_image_npz_folder, encode)
    if not os.path.exists(npz_folder):
        os.mkdir(npz_folder) 
        
    num_slice = volume_image.shape[0]
    for _z in range(0, num_slice):
        npz_path = os.path.join(npz_folder, "%03d.npz"%(_z))
        np.savez_compressed(npz_path, image=volume_image[_z])
        
    del volume_image




get_ipython().system("ls './npz/train_images/'")




train_label_npz_folder = './npz/train_labels/'

if not os.path.exists(train_label_npz_folder):
    os.makedirs(train_label_npz_folder)




for encode in tqdm.tqdm(train_list):
    _, label_array = load_label(os.path.join(train_label_folder, encode + '.nii.gz'), transpose=True)
    npz_folder = os.path.join(train_label_npz_folder, encode)
    if not os.path.exists(npz_folder):
        os.mkdir(npz_folder) 
        
    num_slice = label_array.shape[0]
    for _z in range(0, num_slice):
        npz_path = os.path.join(npz_folder, "%03d.npz"%(_z))
        np.savez_compressed(npz_path, label=label_array[_z])
        
    del label_array




from keras.models import Model, load_model
from keras import layers as klayers
from keras.optimizers import Adam
from keras import utils as kutils
from keras import backend as K
from keras.callbacks import ModelCheckpoint

# Make sure keras running on GPU
K.tensorflow_backend._get_available_gpus()




map_image_list = sorted(glob.glob(os.path.join(train_image_npz_folder, '*/*.npz')))
map_label_list = sorted(glob.glob(os.path.join(train_label_npz_folder, '*/*.npz')))

map_df = pd.DataFrame(data={'image': map_image_list, 'label': map_label_list})
map_df.head()




class LungSliceModelGenerator(kutils.Sequence):
    'Generates data for Keras'
    def __init__(self, mapping_df, batch_size, shuffle=True):
        'Initialization'
        self.mapping_df = mapping_df
        self.data_num   = mapping_df.shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.data_num / self.batch_size))


    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch_mapping_df =             self.mapping_df.iloc[index*self.batch_size: (index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(batch_mapping_df)
        return X, y


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            self.mapping_df = self.mapping_df.sample(frac=1).reset_index(drop=True)
            
    def __data_generation(self, batch_mapping_df):
        'Generates data containing batch_size samples' 
        # Initialization
        X = np.zeros((self.batch_size, 512, 512, 1))
        y = np.zeros((self.batch_size, 512, 512, 1))

        # Generate data
        cnt = 0
        for i, row in batch_mapping_df.iterrows():
            X[cnt, :, :, 0] = np.load(row['image'])['image']
            y[cnt, :, :, 0] = np.load(row['label'])['label']
            cnt += 1
        return X, y




batch_size = 16
slice_generator = LungSliceModelGenerator(map_df, batch_size=batch_size)




def _dice_coefficient(threshold = 0.3):
    def hard_dice_coefficient(y_true, y_pred, smooth=1.0):
        y_true_f = K.flatten(K.cast(y_true > threshold, dtype=float))
        y_pred_f = K.flatten(K.cast(y_pred > threshold, dtype=float))
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return hard_dice_coefficient

def dice_coefficient_loss(y_true, y_pred):
    return 1 - _dice_coefficient()(y_true, y_pred)




def unet(pretrained_weights=None, input_size=[512, 512, 1], depth=3, init_filter=8, 
         filter_size=3, padding='same', pool_size=[2, 2], strides=[2, 2]):
    
    inputs = klayers.Input(input_size)
    
    current_layer = inputs
    encoding_layers = []
    
    # Encoder path
    for d in range(depth + 1):
        num_filters = init_filter * 2 ** d
        
        conv = klayers.Conv2D(num_filters, filter_size, padding=padding, kernel_initializer='he_normal')(current_layer)
        conv = klayers.BatchNormalization()(conv)
        conv = klayers.Activation('relu')(conv)
        conv = klayers.Conv2D(num_filters * 2, filter_size, padding=padding, kernel_initializer='he_normal')(conv)
        conv = klayers.BatchNormalization()(conv)
        conv = klayers.Activation('relu')(conv)
        encoding_layers.append(conv)
    
        pool = klayers.MaxPooling2D(pool_size=pool_size)(conv)
        
        if d == depth:
            # Bridge
            current_layer = conv
        else:
            current_layer = pool

        
    # Decoder path
    for d in range(depth, 0, -1):
        num_filters = init_filter * 2 ** d
        up = klayers.Deconvolution2D(num_filters * 2, pool_size, strides=strides)(current_layer)

        crop_layer = encoding_layers[d - 1]
        # Calculate two layers shape
        up_shape = np.array(up._keras_shape[1:-1])
        conv_shape = np.array(crop_layer._keras_shape[1:-1])

        # Calculate crop size of left and right
        crop_left = (conv_shape - up_shape) // 2

        crop_right = (conv_shape - up_shape) // 2 + (conv_shape - up_shape) % 2
        crop_sizes = tuple(zip(crop_left, crop_right))

        crop = klayers.Cropping2D(cropping=crop_sizes)(crop_layer)

        # Concatenate
        up = klayers.Concatenate(axis=-1)([crop, up])
        conv = klayers.Conv2D(num_filters, filter_size, padding=padding, kernel_initializer='he_normal')(up)
        conv = klayers.BatchNormalization()(conv)
        conv = klayers.Activation('relu')(conv)
        conv = klayers.Conv2D(num_filters, filter_size, padding=padding, kernel_initializer='he_normal')(conv)
        conv = klayers.BatchNormalization()(conv)
        conv = klayers.Activation('relu')(conv)
        
        current_layer = conv
    
    
    outputs = klayers.Conv2D(1, 1, padding=padding, kernel_initializer='he_normal')(current_layer)
    outputs = klayers.Activation('sigmoid')(outputs)
    model = Model(inputs=inputs, outputs=outputs)

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model




model = unet(depth=3)
model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=[_dice_coefficient(0.5)])
model.summary()




model_folder = os.path.join('./model', 'sample-code')

if not os.path.exists(model_folder):
    os.makedirs(model_folder)

callbacks = []
callbacks.append(ModelCheckpoint(os.path.join(model_folder, 'model-{epoch:03d}.h5'), 
                                 save_best_only=False, 
                                 period=5))




history = model.fit_generator(slice_generator,
                              epochs=15,
                              verbose=1, 
                              callbacks=callbacks)




def retrieve_pred_str(src_dir, model, threshold=0.4):
    encode_name = src_dir.split('/')[-1]
    
    _, test_volume = load_dicom_volume(src_dir, suffix='*.dcm')
    
    pred_label = model.predict(np.expand_dims(test_volume, axis=-1))
    pred_label = np.transpose(pred_label[:, :, :, 0], axes=(2, 1, 0))
    
    pred_label = (pred_label > threshold).astype(np.int)

    label_flatten = pred_label.flatten()

    label_flatten_idx = np.where(label_flatten == 1)[0]

    label_str = ''
    
    if label_flatten_idx.size > 0:
        prev_idx = label_flatten_idx[0]
        idx_start = label_flatten_idx[0]
        cnt = 1
        for _idx in label_flatten_idx[1:]:
            if _idx == prev_idx+1:
                cnt += 1
            else:
                label_str += str(idx_start) + ' ' + str(cnt) + ' '

                cnt = 1
                idx_start = _idx
            prev_idx = _idx

        label_str = label_str.rstrip(' ')
    return (encode_name, label_str)




sample_submission = np.genfromtxt('../input/sample_submission.csv', 
                                  delimiter=',', 
                                  dtype='str',
                                  skip_header = 1)




test_encode_list = sample_submission[:, 0]




pred_pair_list = []

for encode_name in tqdm.tqdm(test_encode_list, total=len(test_encode_list)):
    (encode, label_str) = retrieve_pred_str(os.path.join(test_image_folder, encode_name), model, threshold=0.4)
    pred_pair_list.append((encode, label_str))




solution_path = './sample-code_pred.csv'
with open(solution_path, 'w') as f:
    f.write('encode,pixel_value\n')
    for _pair in pred_pair_list:
        encode = _pair[0]
        label_str = _pair[1]
        f.write(encode + ',' + label_str + '\n')




get_ipython().system('rm -r ./npz')

