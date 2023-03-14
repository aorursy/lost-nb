#!/usr/bin/env python
# coding: utf-8



import keras




from keras.models import Sequential




from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense




classifier = Sequential()




classifier.add(Convolution2D(32,3,3, input_shape = (64,64,3),activation = ('relu')))




classifier.add(MaxPooling2D(2,2))




classifier.add(Flatten())




classifier.add(Dense(output_dim = 128,activation='relu'))




classifier.add(Dense(output_dim = 1,activation = 'sigmoid'))




classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])




from keras.preprocessing.image import ImageDataGenerator




train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)




test_datagen=ImageDataGenerator(rescale=1./255)




train=train_datagen.flow_from_directory('Users\sathish gade\Desktop\data\cats and dogs\train',target_size=(64,64),batch_size=32,class_mode='binary')




os.path.abspath("\Users\sathish gade\Desktop\data\catsanddogs\train")




train=train_datagen.flow_from_directory("../input/dogs-vs-cats-redux-kernels-edition/train.zip")




train_datagen = ImageDataGenerator(
                                       rescale=1./255,
                shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800)

