import numpy as np
import pandas as pd 
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os

from google.colab import drive
drive.mount('/content/gdrive')


from google.colab import files
f=files.upload()

!pip install -U -q kaggle --force

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

df.tail()

df['label'].value_counts().plot.bar()


image = load_img("../content/upload/train_images/train_images/100.jpg")
plt.imshow(image)

FAST_RUN = False
IMAGE_WIDTH=150
IMAGE_HEIGHT=150
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3

!pip show tensorflow





from tensorflow.keras.models import Sequential
#from tensorflow.keras import optimizer as Optimizer


import tensorflow.keras.optimizers as Optimizer
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

model = Sequential()


model.add(Conv2D(200,kernel_size=(3,3),activation='relu',input_shape=(150,150,3)))
model.add(Conv2D(180,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(5,5))
model.add(Conv2D(180,kernel_size=(3,3),activation='relu'))
model.add(Conv2D(140,kernel_size=(3,3),activation='relu'))
model.add(Conv2D(100,kernel_size=(3,3),activation='relu'))
model.add(Conv2D(50,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(5,5))
model.add(Flatten())
model.add(Dense(180,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(6,activation='softmax'))

model.compile(optimizer=Optimizer.Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()



from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

earlystop = EarlyStopping(patience=10)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

callbacks = [earlystop, learning_rate_reduction]


df['label'].head()

train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size=15

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
   rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
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

#from tensorflow.keras.callbacks import EarlyStopping
epochs=3 if FAST_RUN else 35
history = model.fit_generator(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks
)

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


submission_df.head()

submission_df.to_csv("submissionugh.csv",index=False)

from google.colab import files
files.download("submissionugh.csv")

model.save_weights("model10.h5")
files.download("model10.h5")




