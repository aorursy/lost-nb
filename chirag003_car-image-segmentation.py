#!/usr/bin/env python
# coding: utf-8



import os, keras
import cv2
import glob
import PIL
from PIL import Image
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
import keras.backend as K

from sklearn.model_selection import train_test_split
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator, img_to_array, array_to_img




#Set some directories
trainHQ_zip_path = '/kaggle/input/carvana-image-masking-challenge/train_hq.zip'
masks_zip_path = '/kaggle/input/carvana-image-masking-challenge/train_masks.zip'




import zipfile
#Extract train images.
with zipfile.ZipFile(trainHQ_zip_path,'r') as zip_ref:
    zip_ref.extractall('/kaggle/working')
#Extract train masks/labels.
with zipfile.ZipFile(masks_zip_path,'r') as zip_ref:
    zip_ref.extractall('/kaggle/working')
data_size = len(os.listdir('/kaggle/working/train_hq'))
print('Number of train images: ', len(os.listdir('/kaggle/working/train_hq')))
print('Number of train masks: ', len(os.listdir('/kaggle/working/train_masks')))




#Display ids for images and masks.
car_ids = sorted(os.listdir('/kaggle/working/train_hq'))
mask_ids = sorted(os.listdir('/kaggle/working/train_masks'))
#Generate some random index.
rnd_ind = list(np.random.choice(data_size,8))
for i in rnd_ind:
    print("Car image id: '{}' -- Corressponding Mask id '{}'".format(car_ids[i], mask_ids[i]))




#Pick the 1553th car&mask ids from ids lists.
n = 1553
car_id = car_ids[n]
mask_id = mask_ids[n]
#Load car&mask images using thier ids.
car = load_img('/kaggle/working/train_hq/' + car_id)
mask = load_img('/kaggle/working/train_masks/' + mask_id)
print("Image Size: ", car.size)
print("Mask Size: ", mask.size)
#Plot them.
fig, ax = plt.subplots(1, 2, figsize=(20,20))
fig.subplots_adjust(hspace=.1, wspace=.01)
ax[0].imshow(car)
ax[0].axis('off')
ax[0].title.set_text('Car Image')
ax[1].imshow(mask)
ax[1].axis('off')
ax[1].title.set_text('Car Mask')




main_image = cv2.imread('/kaggle/working/train_hq/' + car_id)
main_image = load_img('/kaggle/working/train_hq/' + car_id)
main_image = np.asarray(main_image)




main_image.shape




plt.imshow(main_image)




mask_image = Image.open('/kaggle/working/train_masks/' + mask_id)
#mask_image = load_img('/kaggle/working/train_masks/' + mask_id)




mask_image = np.asarray(mask_image)




plt.imshow(mask_image)




img_masked = cv2.bitwise_and(main_image, main_image, mask=mask_image)




plt.imshow(img_masked)









#Randomly split car&mask ids list to training and validation lists.
#X is car image ids list, y is mask image ids list.
X_train_ids, X_val_ids, y_train_ids, y_val_ids= train_test_split(car_ids, mask_ids,
                                                                 test_size=.2, train_size=.8,
                                                                 random_state=42)
X_train_size = len(X_train_ids)
X_val_size = len(X_val_ids)
print('Training images size: ', X_train_size)
print('Validation images size: ', X_val_size)




#Input size could be 128 or 256 or 512 or 1024.
input_size = [128, 128, 3]
def data_generator(images_path, masks_path, image_ids, mask_ids, batch_size, img_size=input_size):
    '''
    images_path/masks_path: Images/Masks folder directory.
    images_ids/mask_ids: Ids for '.jpg' images/masks.
    img_size: Generated imgs/masks size.
    
    returns: batch of randomly-selected car&mask images value-scaled (0 -> 1). 
    '''
    data_size = len(image_ids)
    while True:
        #Choose random indice for later picking.
        rnd_ind = np.random.choice(np.arange(data_size),batch_size)
        imgs = []
        masks = []
        for i in rnd_ind:
            #Pick a random id for car&mask images.
            img_id, mask_id = image_ids[i], mask_ids[i]
            #Load/resize images.
            img = load_img(images_path + img_id, target_size=img_size) 
            mask = load_img(masks_path + mask_id, target_size=img_size[:-1], color_mode = 'grayscale')
            #Add to the batch data.
            imgs.append(img_to_array(img))
            masks.append(img_to_array(mask).reshape(img_size[:-1] + [1]))
        yield np.array(imgs, dtype=np.float16) / 255., np.array(masks, dtype=np.float16) / 255.




#Try out the generator, generate data samples from the validation set.
gen = data_generator('/kaggle/working/train_hq/', '/kaggle/working/train_masks/',
                    X_val_ids, y_val_ids, batch_size=32)

imgs, masks = next(gen)
print('Images batch shape: ', imgs.shape)
print('Masks batch shape: ', masks.shape)




#Plot output samples of the generator.
fig, ax = plt.subplots(2, 4, figsize=(15,7))
fig.subplots_adjust(hspace=.1, wspace=.05)
car_samples, mask_samples = imgs[:4].astype(np.float32), masks[:4][:,:,:,0].astype(np.float32)
for i, (car, mask) in enumerate(zip(car_samples, mask_samples)):
    ax[0, i].imshow(car)
    ax[0, i].axis('off')
    ax[0, i].title.set_text('Car Image')
    
    ax[1, i].imshow(mask, cmap='gray')
    ax[1, i].axis('off')
    ax[1, i].title.set_text('Car Mask')
plt.show()




def dice_coef(y_true, y_pred):
    '''
    Metric
    '''
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    '''
    Loss function
    '''
    loss = 1 - dice_coef(y_true, y_pred)
    return loss


def bce_dice_loss(y_true, y_pred):
    '''
    Mixed crossentropy and dice loss.
    '''
    loss = keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss




def get_unet_128(input_shape=(128, 128, 3),
                 num_classes=1):
    inputs = Input(shape=input_shape)
    # 128

    down1 = Conv2D(64, (3, 3), padding='same')(inputs)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32

    down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16

    down4 = Conv2D(512, (3, 3), padding='same')(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8

    center = Conv2D(1024, (3, 3), padding='same')(down4_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # center

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    # 16

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    # 32

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    # 64

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    # 128

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up1)

    model = Model(inputs=inputs, outputs=classify)

    model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coef])

    return model

uNet = get_unet_128()




#Prepare callbacks
LR_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=10, factor=.2, min_lr=.00001)
EarlyStop_callback = keras.callbacks.EarlyStopping(monitor='val_loss',patience=10, restore_best_weights=True)




#Perpare data generators.
batch_size = 32
train_gen = data_generator('/kaggle/working/train_hq/', '/kaggle/working/train_masks/',
                           X_train_ids, y_train_ids, batch_size=batch_size)
val_gen = data_generator('/kaggle/working/train_hq/', '/kaggle/working/train_masks/',
                           X_val_ids, y_val_ids, batch_size=batch_size)




history = uNet.fit_generator(train_gen, steps_per_epoch=int(X_train_size/batch_size),
                             epochs=21, validation_data=val_gen,
                             validation_steps=int(X_val_size/batch_size),
                             callbacks=[LR_callback, EarlyStop_callback])




uNet.save('unet_main1.h5')




uNet = tf.keras.models.load_model('../input/sir-unet/sir_unet.h5', custom_objects={'bce_dice_loss': bce_dice_loss, 'dice_coef':dice_coef})
#load_model(modelPath, custom_objects={'mean_squared_abs_error': mean_squared_abs_error})




# Plot the loss and accuracy curves for training and validation
fig, ax = plt.subplots(2,1, figsize=(15,7))
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['dice_coef'], color='b', label="Training dice loss")
ax[1].plot(history.history['val_dice_coef'], color='r',label="Validation dice loss")
legend = ax[1].legend(loc='best', shadow=True)




#Perdict some imgs.
pred_masks = uNet.predict(imgs)




fig, ax = plt.subplots(32, 3, figsize=(20,150))
fig.subplots_adjust(hspace=.1, wspace=.05)
for i in range(32):
    ax[i, 0].imshow(imgs[i].astype(np.float32))
    ax[i, 0].axis('off')
    ax[i, 0].title.set_text('Car')
    
    ax[i, 1].imshow(masks[i,:,:,0].astype(np.float32), cmap='gray')
    ax[i, 1].axis('off')
    ax[i, 1].title.set_text('Real Mask')
    
    ax[i, 2].imshow(pred_masks[i,:,:,0], cmap='gray')
    ax[i, 2].axis('off')
    ax[i, 2].title.set_text('Predicted Mask')
plt.show()




import shutil # for removing the directory
shutil.rmtree('/kaggle/working/train_hq')
shutil.rmtree('/kaggle/working/train_masks')




# Prediction




input_image = imgs[0].astype('float32')
input_image = np.asarray(input_image)




plt.imshow(input_image)




plt.imshow(masks[0, :, :, 0].astype('float32'))




pred_mask_image = np.reshape(pred_masks[0], (128, 128))
pred_mask_image = pred_mask_image > 0.5
pred_mask_image = np.asarray(Image.fromarray(pred_mask_image, 'L'))




plt.imshow(pred_mask_image)




pred_masked_image = cv2.bitwise_and(input_image, input_image, mask=pred_mask_image)




plt.imshow(pred_masked_image)









# Prediction




images = [cv2.imread(image) for image in glob.glob('../input/inputcar/input car/*.*')]
#images = [cv2.imread(file) for file in glob.glob('path/to/files/*.jpg')]




images2 = images.copy()




images2[0].shape




#size_images = [cv2.imread(image) for image in glob.glob('../input/inputcar/input car/*.*')]




#image = PIL.Image.open("../input/inputcar/input car/0010-000222-before.jpg")
#image to open

#width, height = image.size




names = glob.glob('../input/inputcar/input car/*.*')




names[0]




names[0].split('/')[-1].split('.')[0]




len(images)




images = []
img = load_img('../input/dataset/dataset/AgktOpMQ.jpeg', target_size=(128, 128))
images.append(img_to_array(img))
img = load_img('../input/dataset/dataset/aQzTDcFQ.jpeg', target_size=(128, 128))
images.append(img_to_array(img))
img = load_img('../input/dataset/dataset/eBbAEfLA.jpeg', target_size=(128, 128))
images.append(img_to_array(img))
img = load_img('../input/dataset/dataset/o-Oh9z6Q.jpeg', target_size=(128, 128))
images.append(img_to_array(img))
img = load_img('../input/dataset/dataset/p3S_zKbA.jpeg', target_size=(128, 128))
images.append(img_to_array(img))
images = np.array(images, dtype=np.float32) / 255.




for i in range(len(images)):
    images[i] = cv2.resize(images[i], (128, 128))
    #images[i] = images[i]/255.
    #images[i] = np.asarray(images[i])
    images[i] = np.array(images[i]) / 255.




images = np.array(images)




images.shape




pred_masks = uNet.predict(images)




plt.imshow(pred_masks[0, :, :, 0])




plt.imshow(images2[0])




#pred_mask_image = np.reshape(pred_masks[0], (3200, 2400))
pred_mask_image = cv2.resize(pred_masks[0], (3200, 2400))

max_output_value = 100
neighborhood_size = 50
subtract_from_mean = 2
'''
binarized_images = [cv2.adaptiveThreshold(image, max_output_value,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY,
                                        neighborhood_size,
                                        subtract_from_mean) for image in gray_images]
'''
# pred_mask_image = cv2.adaptiveThreshold(pred_mask_image, max_output_value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, neighborhood_size, subtract_from_mean)
pred_mask_image = pred_mask_image > 0.5
pred_mask_image = np.asarray(Image.fromarray(pred_mask_image, 'L'))




plt.imshow(pred_mask_image)




pred_mask_image = cv2.adaptiveThreshold(pred_mask_image, max_output_value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, neighborhood_size, subtract_from_mean)




main_image = images2[0]




plt.imshow(main_image)




pred_masked_image = cv2.bitwise_and(main_image, main_image, mask=pred_mask_image)




plt.imshow(pred_masked_image)




pred_masked_image = cv2.resize(pred_masked_image, (3200, 2400))




matplotlib.pyplot.imsave('abc.jpeg', pred_masked_image)




for i in range(5, len(pred_masks)):
    #pred_mask_image = np.reshape(pred_masks[i], (128, 128))
    pred_mask_image = cv2.resize(pred_masks[i], (3200, 2400))
    pred_mask_image = pred_mask_image > 0.5
    pred_mask_image = np.asarray(Image.fromarray(pred_mask_image, 'L'))
    
    main_image = images2[i]
    
    pred_masked_image = cv2.bitwise_and(main_image, main_image, mask=pred_mask_image)
    
    pred_masked_image = cv2.resize(pred_masked_image, (3200, 2400))
    
    matplotlib.pyplot.imsave(names[i].split('/')[-1].split('.')[0] +'.jpeg', pred_masked_image)
    print(i)









'''
Thank You
                                            
Regards,
Chirag Verma
'''































