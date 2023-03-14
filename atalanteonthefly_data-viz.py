#!/usr/bin/env python
# coding: utf-8



from glob import glob
from PIL import Image

import matplotlib.pylab as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
import cv2

DATA_DIR = '../input/'




train_masks_csv = pd.read_csv(DATA_DIR + 'train_masks.csv')
sample_submission_csv = pd.read_csv(DATA_DIR + 'sample_submission.csv')

NUMBER_POSES = 16
train_number_images = train_masks_csv.shape[0]
train_number_cars = train_number_images/NUMBER_POSES

test_number_images = sample_submission_csv.shape[0]
test_number_cars = test_number_images/NUMBER_POSES

print("\nThe total number of images in the training set is : %s." %  (train_number_images,))
print("The total number of images in the test set is : %s." %  (test_number_images,))
print("\nGiven that we have 16 different poses for a car we can deduce that we have %s different cars in the training set and %s different cars in the test set.." %  (train_number_cars, test_number_cars))




fig = plt.figure(figsize=(30, 30))
fig.suptitle('\nThe different poses for the same model of a car', fontsize=50)

car_poses_list = glob(DATA_DIR + 'train/293a0fa72e5b***.jpg')
car_poses_list.sort()

for i, car_pose_path in zip(range(1,17), car_poses_list):
    
    bgr_img = cv2.imread(car_pose_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    
    plt.subplot(4,4,i)
    plt.imshow(img)

plt.tight_layout()
plt.show()




fig = plt.figure(figsize=(30, 30))
fig.suptitle('\nDifferent model of car in the training set', fontsize=50)

car_list = glob(DATA_DIR + 'train/*.jpg')
car_list.sort()
car_list = np.random.choice(car_list, 16)

for i, car_path in zip(range(1,17), car_list):

    bgr_img = cv2.imread(car_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    plt.subplot(4, 4, i)
    plt.imshow(img)
    
plt.tight_layout()
plt.show()




bgr_original_img = cv2.imread(DATA_DIR + 'train/6cc98271f4dd_15.jpg')
original_img = cv2.cvtColor(bgr_original_img, cv2.COLOR_BGR2RGB)
mask_img = Image.open(DATA_DIR + 'train_masks/6cc98271f4dd_15_mask.gif')
mask_img = np.asarray(mask_img)
no_background_img = cv2.bitwise_and(original_img, original_img, mask=mask_img)

fig = plt.figure(figsize=(20, 20))
ax1 = plt.subplot(131)
ax1.set_title('Original training image')
plt.imshow(original_img)
ax2 = plt.subplot(132)
ax2.set_title('training mask')
plt.imshow(mask_img)
ax3 = plt.subplot(133)
ax2.set_title('image with no background')
plt.imshow(no_background_img)

plt.tight_layout()
plt.show()




metadata_csv = pd.read_csv('data/metadata.csv')

fig = plt.figure(figsize=(40, 60))

plt.subplot(311)
ax1 = sns.countplot(x="year", data=metadata_csv)
ax1.set_title('Year of Construction Distribution', fontsize=30)
ax1.yaxis.label.set_size(30)
ax1.xaxis.label.set_size(30)
ax1.tick_params(labelsize=30)

plt.subplot(312)
ax2 = sns.countplot(y="make", data=metadata_csv)
ax2.set_title('Manufacturer\'s Brand Distribution', fontsize=30)
ax2.yaxis.label.set_size(30)
ax2.xaxis.label.set_size(30)
ax2.tick_params(labelsize=30)

plt.subplot(313)
models = metadata_csv['model']
less_frequent_models = list(models.value_counts().index.values[10:])
for model in less_frequent_models:
    indices = pd.Index(models).get_loc(model)
    models = models.drop(models.index[indices])
pd_models = pd.DataFrame(data={'model': models})
ax3 = sns.countplot(x='model', data=pd_models)
ax3.set_title('Car Most Frequent Model Distribution', fontsize=30)
ax3.yaxis.label.set_size(30)
ax3.xaxis.label.set_size(30)
ax3.tick_params(labelsize=30)

plt.show()

