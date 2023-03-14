#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import tensorflow as tf
# # tf.InteractiveSession()
# tf.random.set_seed(0)


# In[2]:


get_ipython().system('pip install -U -q kaggle --force')


# In[3]:


from google.colab import files
f=files.upload()


# In[4]:


get_ipython().system('mkdir -p ~/.kaggle')


# In[5]:


get_ipython().system('cp kaggle.json ~/.kaggle/')


# In[6]:


get_ipython().system('chmod 600 /root/.kaggle/kaggle.json')


# In[7]:


get_ipython().system('kaggle competitions download -c nnfl-cnn-lab2')


# In[8]:


get_ipython().run_cell_magic('bash', '', 'cd /content\nunzip nnfl-cnn-lab2.zip')


# In[9]:


import numpy as np
import pandas as pd 
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
print(os.listdir("../content"))


# In[10]:


FAST_RUN = False
IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3


# In[11]:


# !unzip ../content/train_images


# In[12]:


# !unzip ../content/test1.zip


# In[13]:


filenames = os.listdir("/content/upload/train_images/train_images")
# categories = []
# for filename in filenames:
#     category = filename.split('.')[0]
#     if category == 'dog':
#         categories.append(1)
#     else:
#         categories.append(0)

df = pd.DataFrame(pd.read_csv('/content/upload/train_set.csv'))


# In[14]:


df.head()


# In[15]:


df.tail()


# In[16]:


df['label'].value_counts().plot.bar()


# In[17]:


sample = random.choice(filenames)
# sample='0.jpg'
image = load_img("/content/upload/train_images/train_images/"+sample)
plt.imshow(image)


# In[18]:


from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from tensorflow.python.keras.layers import Input, Activation, Dense, Conv2D, Reshape, concatenate, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D
# model = Sequential()

# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Flatten())
# model.add(Dense(64, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))
# model.add(Dense(6, activation='softmax')) # 2 because we have cat and dog classes

# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# model.summary()


# In[19]:


bnmomemtum=0.9

def block(x, squeeze, expand):
  y  = Conv2D(filters=squeeze, kernel_size=3, activation='relu', padding='same')(x)
  y  = BatchNormalization(momentum=bnmomemtum)(y)
  y1 = Conv2D(filters=expand//2, kernel_size=5, activation='relu', padding='same')(y)
  y1 = BatchNormalization(momentum=bnmomemtum)(y1)
  y3 = Conv2D(filters=expand//2, kernel_size=5, activation='relu', padding='same')(y)
  y3 = BatchNormalization(momentum=bnmomemtum)(y3)
  return concatenate([y1, y3])

def block_module(squeeze, expand):
  return lambda x: block(x, squeeze, expand)


# In[20]:


from tensorflow.python.keras.models import Model

x = Input(shape=[IMAGE_WIDTH, IMAGE_HEIGHT, 3])
y = BatchNormalization(center=True, scale=False)(x)
y = Activation('relu')(y)
y = Conv2D(kernel_size=5, filters=16, padding='same', use_bias=True, activation='relu')(x)
y = BatchNormalization(momentum=bnmomemtum)(y)

y = block_module(16, 32)(y)
y = MaxPooling2D(pool_size=2)(y)
# y  = Dropout(0.1, seed=0)(y) 

y = block_module(32, 64)(y)
y = MaxPooling2D(pool_size=2)(y)
# y  = Dropout(0.1, seed=0)(y) 

y = block_module(64, 128)(y)
y = MaxPooling2D(pool_size=2)(y)
# y  = Dropout(0.1, seed=0)(y) 

y = block_module(128, 64)(y)
y = MaxPooling2D(pool_size=2)(y)
# y  = Dropout(0.1, seed=0)(y) 

y = block_module(64, 32)(y)
# y = MaxPooling2D(pool_size=2)(y)
# y  = Dropout(0.1, seed=0)(y) 

# y = block_module(32, 16)(y)

y = GlobalAveragePooling2D()(y)
y = Flatten()(y)
y = Dense(64, activation='relu')(y)
y = BatchNormalization()(y)
y = Dense(6, activation='softmax')(y)
model = Model(x, y)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[21]:


model.summary()


# In[22]:


from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# In[23]:


earlystop = EarlyStopping(patience=10)


# In[24]:


learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


# In[25]:


callbacks = [learning_rate_reduction]


# In[26]:


df['label'].head()


# In[27]:


df["label"] = df["label"].replace({0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5'}) 


# In[28]:


train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)


# In[29]:


train_df['label'].value_counts().plot.bar()


# In[30]:


validate_df['label'].value_counts().plot.bar()


# In[31]:


total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size=16


# In[32]:


print(total_train)
print(total_validate)


# In[33]:


train_datagen = ImageDataGenerator(
    rotation_range=10,
    rescale=1/255,
    horizontal_flip=True,
    # width_shift_range=0.05,
    # height_shift_range=0.05,
    brightness_range=[0.6, 1.0]

    # width_shift_range= 0.2, height_shift_range= 0.2,
    # rotation_range= 90, rescale = 1/255,
    # horizontal_flip= True, vertical_flip=True
)

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "/content/upload/train_images/train_images", 
    x_col='image_name',
    y_col='label',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)


# In[34]:


validation_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "/content/upload/train_images/train_images", 
    x_col='image_name',
    y_col='label',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)


# In[35]:


example_df = train_df.sample(n=1).reset_index(drop=True)
example_generator = train_datagen.flow_from_dataframe(
    example_df, 
    "/content/upload/train_images/train_images", 
    x_col='image_name',
    y_col='label',
    target_size=IMAGE_SIZE,
    class_mode='categorical'
)


# In[36]:


plt.figure(figsize=(12, 12))
for i in range(0, 15):
    plt.subplot(5, 3, i+1)
    for X_batch, Y_batch in example_generator:
        image = X_batch[0]
        plt.title(Y_batch[0])
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()


# In[37]:


epochs=3 if FAST_RUN else 32
history = model.fit_generator(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks
)


# In[38]:


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


# In[39]:


test_filenames = os.listdir("/content/upload/test_images/test_images")
test_df = pd.DataFrame({
    'image_name': test_filenames
})
nb_samples = test_df.shape[0]


# In[40]:


test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df, 
    "/content/upload/test_images/test_images", 
    x_col='image_name',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    shuffle=False
)


# In[41]:


predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))


# In[42]:


test_df['label'] = np.argmax(predict, axis=-1)


# In[43]:


label_map = dict((v,k) for k,v in train_generator.class_indices.items())
test_df['label'] = test_df['label'].replace(label_map)


# In[44]:


test_df['label'] = test_df['label'].replace({ '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5 })


# In[45]:


test_df['label'].value_counts().plot.bar()


# In[46]:


sample_test = test_df.head(18)
sample_test.head()
plt.figure(figsize=(12, 24))
for index, row in sample_test.iterrows():
    filename = row['image_name']
    category = row['label']
    img = load_img("/content/upload/test_images/test_images/"+filename, target_size=IMAGE_SIZE)
    plt.subplot(6, 3, index+1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category) + ')' )
plt.tight_layout()
plt.show()


# In[47]:


submission_df = test_df.copy()
# submission_df['id'] = submission_df['filename'].str.split('.').str[0]
# submission_df['label'] = submission_df['category']
# submission_df.drop(['filename', 'category'], axis=1, inplace=True)
# submission_df.to_csv('submission.csv', index=False)
submission_df.sort_values(by=['image_name'], axis=0, ascending=True)


# In[ ]:





# In[48]:


from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):
  csv = df.to_csv(index=False)
  b64 = base64.b64encode(csv.encode())
  payload = b64.decode()
  html = '<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'
  html = html.format(payload=payload,title=title,filename=filename)
  return HTML(html)

create_download_link(submission_df)


# In[ ]:





# In[ ]:





# In[49]:


model.save_weights("model4.h5")


# In[ ]:




