#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ignore warnings :
import warnings
warnings.filterwarnings('ignore')


# In[2]:


from keras import backend as K

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# In[3]:


main_path = "/kaggle/input/taller1ann-usm/"

train_path = main_path + 'train_images/'
train_labels_path = main_path + 'train_labels.csv'
valid_path = main_path + 'test_images/'
sample_submission_path = main_path + 'sample_submission.csv'
submission_last = 'submission_last.csv'


# In[4]:


df = pd.read_csv(train_labels_path)
bd = dict(df.Expected.value_counts())
class_names = list(bd.keys())
num_class = len(class_names)


# In[5]:


img = plt.imread(train_path + df.iloc[0]['Id']+'.jpg')
shape_img = img.shape


# In[6]:


df_q = pd.read_csv(train_labels_path)
df_q.columns = ['image', 'label']
dict_classes_img = df_q.groupby(['label'])['image'].apply(list).to_dict()

fig, axs = plt.subplots(num_class, 5, figsize=(64,64), dpi=120)
i=0
for label, imgs in dict_classes_img.items():
  for j in range(5):
    imgplot = plt.imread(train_path + imgs[j] + '.jpg')
    axs[i, j].imshow(imgplot)
    axs[i, j].set_title(label, fontsize=20)
    axs[i, j].set_yticklabels([])
    axs[i, j].set_xticklabels([])
  i+=1
plt.tight_layout()
plt.show()


# In[7]:


labels = pd.read_csv(train_labels_path).set_index('Id')
labels = labels.to_dict()
labels = labels[list(labels.keys())[0]]

df = pd.DataFrame(list(labels.items()))
df.columns = ['images', 'labels']
df = df.astype({'labels': str})
df['images'] = df['images'].map(lambda s: s+'.jpg')


# In[8]:


from sklearn.model_selection import train_test_split

train_df, validation_df = train_test_split(df, test_size=0.2)

print(f'train_df: {train_df.shape}')
print(f'validation_df: {validation_df.shape}')


# In[9]:


from keras.preprocessing.image import ImageDataGenerator
from keras.applications import inception_resnet_v2

datagen = ImageDataGenerator(
    horizontal_flip=True, 
    validation_split=0.2,
    brightness_range = [0.5,2], 
    zoom_range=0.2,
    preprocessing_function=inception_resnet_v2.preprocess_input)


# In[10]:


train_generator = datagen.flow_from_dataframe(
       dataframe=train_df,
       directory=train_path,
       x_col="images",
       y_col="labels",
       target_size=shape_img[:-1],
       batch_size=32,
       class_mode='categorical')

validation_generator = datagen.flow_from_dataframe(
       dataframe=validation_df,
       directory=train_path,
       x_col="images",
       y_col="labels",
       target_size=shape_img[:-1],
       batch_size=32,
       class_mode='categorical')


# In[11]:


from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from keras.models import Model

base_model = inception_resnet_v2.InceptionResNetV2(weights='imagenet', include_top=False, input_shape=shape_img)

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation = 'elu')(x)
predictions = Dense(len(class_names), activation='softmax')(x)

model_resnet_v2_inception = Model(inputs=base_model.input, outputs = predictions)


# In[12]:


from keras.optimizers import Adam,SGD
model_resnet_v2_inception.compile(loss="categorical_crossentropy", optimizer=SGD(learning_rate=0.001, momentum=0.9), metrics=[f1,'accuracy'])


# In[13]:


from keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stop = EarlyStopping(monitor='val_loss', patience=8, verbose=1, min_delta=1e-4)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, min_delta=1e-4)
callbacks_list = [early_stop, reduce_lr]


# In[14]:


from keras.utils.vis_utils import plot_model
plot_model(model_resnet_v2_inception, show_shapes=True, show_layer_names=True)


# In[15]:


STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size

model_history = model_resnet_v2_inception.fit_generator(
                train_generator,
                steps_per_epoch=STEP_SIZE_TRAIN,
                epochs=50,
                validation_data=validation_generator,
                validation_steps=STEP_SIZE_VALID,
                callbacks=callbacks_list)


# In[16]:


from matplotlib import pyplot as plt

plt.rcdefaults()
fig, axs = plt.subplots(1,3, figsize=(20,5), dpi=120)

metrics = [('accuracy','val_accuracy'), ('loss','val_loss'), ('f1','val_f1')]

for ite, pac in enumerate(metrics):
  axs[ite].plot(model_history.history[pac[0]])
  axs[ite].plot(model_history.history[pac[1]])
  axs[ite].set_xlabel('epoch')
  axs[ite].set_ylabel(pac[0])
  axs[ite].set_title('model %s' % pac[0])
  axs[ite].legend(['train', 'test'], loc='upper left')

plt.show()


# In[24]:


sample_submission_path


# In[31]:


df_test = pd.read_csv(sample_submission_path).set_index('Id').to_dict()
df_test = df_test[list(df_test.keys())[0]]
df_test = pd.DataFrame(list(df_test.items()))
df_test.columns = ['images', 'expected']
df_test['images'] = df_test['images'].map(lambda s: s+'.jpg')

test_generator = datagen.flow_from_dataframe(
        dataframe=df_test,
        directory=valid_path,
        x_col="images",
        y_col="expected",
        target_size=shape_img[:-1],
        batch_size=1,
        class_mode='categorical')

filenames = [filename for filename in test_generator.filenames]
nb_samples = len(filenames)


# In[32]:


predictions = model_resnet_v2_inception.predict_generator(test_generator,steps = nb_samples, verbose=1)


# In[33]:


y_pred_labels = np.argmax(predictions, axis = 1)


labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions_label = [labels[k] for k in y_pred_labels]
filenames_label = [s[12:-4] for s in filenames]


submission = pd.DataFrame({'Id':filenames_label,'Expected':predictions_label})
submission.to_csv(submission_last, index=False, header=True)

