#!/usr/bin/env python
# coding: utf-8



get_ipython().system('pip install -U -q kaggle --force')




# !pip uninstall tensorflow
# !pip install tensorflow==2.0.0




import tensorflow as tf
tf.test.gpu_device_name()




from google.colab import files
f=files.upload()




get_ipython().system('mkdir -p ~/.kaggle')




get_ipython().system('cp kaggle.json ~/.kaggle/')




get_ipython().system('chmod 600 /root/.kaggle/kaggle.json')




get_ipython().system('kaggle competitions download -c nnfl-cnn-lab2')




get_ipython().run_cell_magic('bash', '', 'cd /content\nunzip nnfl-cnn-lab2.zip')




import tensorflow as tf




import numpy as np
import pandas as pd 
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os

print(os.listdir("../content"))




FAST_RUN = False
IMAGE_WIDTH=224
IMAGE_HEIGHT=224
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3
batch_size=16




name_frame = os.listdir("../content/upload/train_images/train_images")  
label_frame = pd.read_csv('../content/upload/train_set.csv', usecols=range(1,2)) 
df = pd.read_csv('../content/upload/train_set.csv', dtype=str)




df.head()








df.tail()




df['label'].value_counts().plot.bar()




# # 0.88732 - 16 epochs

# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

# model = Sequential()

# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Conv2D(64, (5, 5), activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Conv2D(128, (9, 9), activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Flatten())
# model.add(Dense(64, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))
# model.add(Dense(32, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))
# model.add(Dense(6, activation='sigmoid'))

# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# model.summary()




# # 0.91198 - 21 epoch

# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

# model = Sequential()

# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(3, 3)))
# model.add(Dropout(0.1))

# model.add(Conv2D(64, (5, 5), activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))

# model.add(Conv2D(128, (7, 7), activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Conv2D(256, (9, 9), activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.3))

# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.4))
# model.add(Dense(512, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))
# model.add(Dense(6, activation='sigmoid'))

# model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# model.summary()




# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
# from tensorflow.keras.optimizers import Nadam

# model = Sequential()

# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
# # model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(3, 3)))
# model.add(Dropout(0.1))

# model.add(Conv2D(64, (5, 5), activation='relu'))
# # model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))

# model.add(Conv2D(128, (7, 7), activation='relu'))
# # model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Conv2D(256, (9, 9), activation='relu'))
# # model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.3))

# model.add(Flatten())
# model.add(Dense(784, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.4))
# model.add(Dense(784, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))
# model.add(Dense(6, activation='sigmoid'))

# model.compile(loss='sparse_categorical_crossentropy', optimizer=Nadam(lr=0.005), metrics=['accuracy'])

# model.summary()




# # 0.918 Acc - 10 epochs - batch 20

# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
# from tensorflow.keras.optimizers import Nadam
# from tensorflow.keras.applications.inception_v3 import InceptionV3

# model = InceptionV3(
#     include_top = True,
#     weights = None,
#     input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS),
#     pooling = 'max',
#     classes = 6
# )

# model.compile(loss='sparse_categorical_crossentropy', optimizer=Nadam(), metrics=['accuracy'])

# # model.summary()




# # 0.915 Acc

# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
# from tensorflow.keras.optimizers import Nadam
# from tensorflow.keras.applications.densenet import DenseNet201

# model = DenseNet201(
#     include_top = True,
#     weights = None,
#     input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS),
#     pooling = 'max',
#     classes = 6
# )

# model.compile(loss='sparse_categorical_crossentropy', optimizer=Nadam(), metrics=['accuracy'])

# # model.summary()




from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.applications.resnet_v2 import ResNet50V2

model = ResNet50V2(
    include_top = True,
    weights = None,
    input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS),
    pooling = 'max',
    classes = 6
)

model.compile(loss='sparse_categorical_crossentropy', optimizer=Nadam(), metrics=['accuracy'])

# model.summary()




from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# from tensorboardcolab import TensorBoardColab, TensorBoardColabCallback




earlystop = EarlyStopping(patience=10)




learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.25, 
                                            min_lr=0.00001)




mchk = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')




# tbc = TensorBoardColab()




callbacks = [earlystop, learning_rate_reduction, mchk]




df['label'].head()




# df["label"] = df["label"].replace({0: 'zero', 1: 'one', 2: 'two', 3:'three', 4:'four', 5: 'five'}) 




train_df, validate_df = train_test_split(df, test_size=0.2, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)




train_df['label'].value_counts().plot.bar()




validate_df['label'].value_counts().plot.bar()




total_train = train_df.shape[0]
total_validate = validate_df.shape[0]




print(total_train)
print(total_validate)





train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "../content/upload/train_images/train_images/", 
    x_col='image_name',
    y_col='label',
    target_size=IMAGE_SIZE,
    class_mode='sparse',
    batch_size=batch_size
)




validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "../content/upload/train_images/train_images/", 
    x_col='image_name',
    y_col='label',
    target_size=IMAGE_SIZE,
    class_mode='sparse',
    batch_size=batch_size
)




epochs=3 if FAST_RUN else 15
history = model.fit(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks
)




model.save_weights("model_wts.h5")
model.save("model.h5")




from keras.models import load_model
 
# load model
model = load_model('model.h5')




fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(1, epochs, 1))
ax1.set_yticks(np.arange(0, 1, 0.1))

ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(1, epochs, 1))

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()




test_filenames = os.listdir("../content/upload/test_images/test_images/")
test_df = pd.DataFrame({
    'filename': test_filenames
})
nb_samples = test_df.shape[0]
print(nb_samples)




test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df, 
    "../content/upload/test_images/test_images/", 
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    shuffle=False
)




predict = model.predict(test_generator, steps=np.ceil(nb_samples/batch_size))




test_df['category'] = np.argmax(predict, axis=-1)




label_map = dict((v,k) for k,v in train_generator.class_indices.items())
test_df['category'] = test_df['category'].replace(label_map)




test_df.head()




test_df['category'].value_counts().plot.bar()




submission_df = test_df.copy()
submission_df['image_name'] = submission_df['filename']
submission_df['label'] = submission_df['category']
submission_df.drop(['filename', 'category'], axis=1, inplace=True)
submission_df.to_csv('submission.csv', index=False)




from google.colab import drive
drive.mount('/content/drive')




get_ipython().system('mkdir /content/drive/My\\ Drive/NNFL_lab/Lab2/sub11')
get_ipython().system('cp /content/submission.csv /content/drive/My\\ Drive/NNFL_lab/Lab2/sub11')
get_ipython().system('cp /content/model.h5 /content/drive/My\\ Drive/NNFL_lab/Lab2/sub11')
get_ipython().system('cp /content/model_wts.h5 /content/drive/My\\ Drive/NNFL_lab/Lab2/sub11')






