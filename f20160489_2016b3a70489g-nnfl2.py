import tensorflow.keras.layers as Layers
import tensorflow.keras.activations as Actications
import tensorflow.keras.models as Models
import tensorflow.keras.optimizers as Optimizer
import tensorflow.keras.metrics as Metrics
import tensorflow.keras.utils as Utils
from keras.utils.vis_utils import model_to_dot
import os
import matplotlib.pyplot as plot
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix as CM
from random import randint
from IPython.display import SVG
import matplotlib.gridspec as gridspec

from google.colab import drive
drive.mount('/content/gdrive')

from google.colab import files
f=files.upload()

!pip install -U -q kaggle --force

!ls

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/

!chmod 600 /root/.kaggle/kaggle.json

!kaggle competitions download -c nnfl-cnn-lab2

%%bash
cd /content
unzip nnfl-cnn-lab2.zip

import numpy as np
import pandas as pd 
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
print(os.listdir("../content"))

df=pd.read_csv("../content/upload/train_set.csv")

df.head()

df['label'].value_counts().plot.bar()


image = load_img("../content/upload/train_images/train_images/0.jpg")
plt.imshow(image)

np.asarray(image).shape

FAST_RUN = False
IMAGE_WIDTH= 150
IMAGE_HEIGHT= 150
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3

model = Models.Sequential()

model.add(Layers.Conv2D(200,kernel_size=(3,3),activation='relu',input_shape=(IMAGE_WIDTH,IMAGE_HEIGHT,3)))
model.add(Layers.Conv2D(180,kernel_size=(3,3),activation='relu',padding='same'))
model.add(Layers.MaxPool2D(5,5))
model.add(Layers.Conv2D(180,kernel_size=(3,3),activation='relu'))
model.add(Layers.Conv2D(140,kernel_size=(3,3),activation='relu'))
model.add(Layers.Conv2D(100,kernel_size=(3,3),activation='relu'))
model.add(Layers.Conv2D(50,kernel_size=(3,3),activation='relu'))
model.add(Layers.MaxPool2D(5,5))
model.add(Layers.Flatten())
model.add(Layers.Dense(180,activation='relu'))
model.add(Layers.Dense(100,activation='relu'))
model.add(Layers.Dense(50,activation='relu'))
model.add(Layers.Dropout(rate=0.5))
model.add(Layers.Dense(6,activation='softmax'))

model.compile(optimizer=Optimizer.Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()
#SVG(model_to_dot(model).create(prog='dot', format='svg'))
#Utils.plot_model(model,to_file='model.png',show_shapes=True)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

earlystop = EarlyStopping(patience=10)

learning_rate_reduction = ReduceLROnPlateau(monitor='accuracy', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

callbacks = [earlystop, learning_rate_reduction]

df['label'].head()

train_df, validate_df = train_test_split(df, test_size=0.1, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

validate_df.shape

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size=64

train_df['label'].value_counts().plot.bar()

print(total_train)
print(total_validate)

filenames = os.listdir("../content/upload/train_images/train_images")

df['label']=df['label'].astype('str')
train_df['label']=train_df['label'].astype('str')
validate_df['label']=validate_df['label'].astype('str')

import cv2

im = cv2.imread('../content/upload/train_images/train_images/0.jpg')

print(type(im))
# <class 'numpy.ndarray'>

print(im.shape)
print(type(im.shape))

train_datagen = ImageDataGenerator(
    #rotation_range= 5,
    rescale=1./255,
    #shear_range=0.1,
    #zoom_range=0.4,
    horizontal_flip=True,
    #width_shift_range=0.2,
    #height_shift_range=0.2,
    #vertical_flip=True
)

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "../content/upload/train_images/train_images", 
    x_col='image_name',
    y_col='label',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "../content/upload/train_images/train_images", 
    x_col='image_name',
    y_col='label',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)

#!pip3 install git+https://github.com/keras-team/keras.git -U

#from tensorflow.keras.callbacks import EarlyStopping
epochs=3 if FAST_RUN else 30
history = model.fit_generator(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks
)

model.save_weights("model1.h5")
files.download("model1.h5")

test_df=pd.read_csv("../content/upload/sample_submission.csv")

test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df, 
    "../content/upload/test_images/test_images", 
    x_col='image_name',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    shuffle=False
)

nb_samples = test_df.shape[0]

predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))


test_df['label'] = np.argmax(predict, axis=-1)

label_map = dict((v,k) for k,v in train_generator.class_indices.items())
test_df['label'] = test_df['label'].replace(label_map)

test_df['label'].value_counts().plot.bar()

submission_df = test_df.copy()
submission_df['image_name'] = submission_df['image_name']
submission_df['label'] = submission_df['label']


submission_df.to_csv("submission3.csv",index=False)

from google.colab import files
files.download("submission3.csv") 








