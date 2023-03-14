#!/usr/bin/env python
# coding: utf-8



import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import OneHotEncoder
from keras.applications.vgg19 import VGG19
from keras import optimizers
import tensorflow as tf


win_size = (48, 48)


def single_cell_block_hog_descriptor(cell_size):
    if isinstance(cell_size, int):
        cell_size = (cell_size, cell_size)
    block_size = cell_size
    block_stride = tuple(np.array(cell_size) // 2)
    nbins = 9
    return cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)


def hog_descriptor_compute_shaped(hog, image):
    return hog.compute(image).reshape(
        *(np.array(hog.winSize) - np.array(hog.blockSize)) // np.array(hog.blockStride) + 1,
        *np.array(hog.blockSize) // np.array(hog.cellSize),
        hog.nbins,
    ).transpose((1, 0, 2, 3, 4))

root_path = '/kaggle/input/tl-signs-hse-itmo-2020-winter/'

labeled_csv = pd.read_csv(root_path + "train.csv")
hog_descriptor = single_cell_block_hog_descriptor(12)


def image_features(image):
    return hog_descriptor.compute(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)).flatten()
    return np.concatenate([
        hog_descriptor.compute(color).flatten()
        for color in cv2.cvtColor(image, cv2.COLOR_RGB2LAB).transpose((2, 0, 1))
    ])




labeled_X = np.array([
    image_features(cv2.imread(f"{root_path}/train/train/{filename}"))
    for filename in labeled_csv['filename']
])




from sklearn.model_selection import cross_val_score

cross_val_score(svm.SVC(C=5), labeled_X, labeled_csv['class_number'], cv=5)




from sklearn.model_selection import cross_val_score

cross_val_score(svm.SVC(C=10), labeled_X, labeled_csv['class_number'], cv=5)




from sklearn.model_selection import cross_val_score

cross_val_score(svm.SVC(C=20), labeled_X, labeled_csv['class_number'], cv=5)




from sklearn.model_selection import cross_val_score

cross_val_score(svm.SVC(C=5), labeled_X, labeled_csv['class_number'], cv=3)




from sklearn.model_selection import cross_val_score

cross_val_score(svm.SVC(C=10), labeled_X, labeled_csv['class_number'], cv=3)




from sklearn.model_selection import cross_val_score

cross_val_score(svm.SVC(C=20), labeled_X, labeled_csv['class_number'], cv=3)




clf = svm.SVC(C=10)
clf.fit(labeled_X, labeled_csv['class_number'])
clf.score(labeled_X, labeled_csv['class_number'])




sample_submission_csv = pd.read_csv(root_path + "sample_submission.csv")
sample_submission_csv['class_number'] = clf.predict(np.array([
    image_features(cv2.imread(f"{root_path}/test/test/{filename}"))
    for filename in sample_submission_csv['filename']
]))
sample_submission_csv.to_csv('/kaggle/working/sample_submission.csv', index=False)




model = Sequential([
    Conv2D(128, (3, 3), activation='relu', input_shape=(48, 48, 3)),
    MaxPooling2D(2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(labeled_csv['class_number'].max() + 1, activation='sigmoid'),
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
    metrics=['accuracy'],
)

with tf.device('/GPU:0'):
    model.fit_generator(ImageDataGenerator().flow_from_dataframe(
        dataframe=labeled_csv,
        directory=f"{root_path}train/train",  # this is the target directory
        x_col="filename",
        y_col="class_number",
        target_size=(48, 48),
        class_mode='raw',
    ), epochs=50)




model = VGG19(
    include_top=False,
    input_shape=(48, 48, 3),
    classes=labeled_csv['class_number'].max() + 1,
)
output = Dense(labeled_csv['class_number'].max() + 1, activation='softmax')(Dense(1024, activation='relu')(Flatten()(model.outputs[0])))
model = Model(inputs=model.inputs, outputs=output)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
    metrics=['accuracy'],
)

train_filename, test_filename, train_class_number, test_class_number    = train_test_split(labeled_csv['filename'], labeled_csv['class_number'])

def image_flow(filename, class_number):
    return ImageDataGenerator().flow_from_dataframe(
        dataframe=pd.DataFrame({'filename': filename, 'class_number': class_number }),
        directory=f"{root_path}train/train",
        x_col="filename",
        y_col="class_number",
        target_size=(48, 48),
        class_mode='raw',
    )

import tensorflow as tf

with tf.device('/GPU:0'):
    model.fit_generator(
        image_flow(train_filename, train_class_number),
        epochs=50,
        validation_data=image_flow(test_filename, test_class_number),
    )




model = VGG19(
    include_top=False,
    input_shape=(48, 48, 3),
    classes=labeled_csv['class_number'].max() + 1,
)
output = Dense(labeled_csv['class_number'].max() + 1, activation='softmax')(Dense(1024, activation='relu')(Flatten()(model.outputs[0])))
model = Model(inputs=model.inputs, outputs=output)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
    metrics=['accuracy'],
)

def image_flow(filename, class_number):
    return ImageDataGenerator().flow_from_dataframe(
        dataframe=pd.DataFrame({'filename': filename, 'class_number': class_number }),
        directory=f"{root_path}train/train",
        x_col="filename",
        y_col="class_number",
        target_size=(48, 48),
        class_mode='raw',
    )

with tf.device('/GPU:0'):
    model.fit_generator(
        image_flow(labeled_csv['filename'], labeled_csv['class_number']),
        epochs=35,
    )




sample_submission_csv = pd.read_csv(root_path + "sample_submission.csv")
sample_submission_csv['class_number'] = model.predict_generator(ImageDataGenerator().flow_from_dataframe(
    dataframe=sample_submission_csv,
    directory=f"{root_path}test/test",
    x_col="filename",
    target_size=(48, 48),
    class_mode=None,
    shuffle=None,
), workers=0).argmax(axis=1)
sample_submission_csv.to_csv('/kaggle/working/sample_submission.csv', index=False)

