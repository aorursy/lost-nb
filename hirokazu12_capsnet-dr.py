#!/usr/bin/env python
# coding: utf-8

# In[1]:


# submission score = 0.000って何？


# In[2]:


get_ipython().system('pip install keras==2.2.4')
import numpy as np
import os
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks
from keras.utils.vis_utils import plot_model
import keras.backend as K
import tensorflow as tf
from keras import initializers, layers, models
from keras.utils import to_categorical


# In[3]:


class Length(layers.Layer):
    def call(self, inputs, **kwargs): 
        return K.sqrt(K.sum(K.square(inputs), -1)) 
 
    def compute_output_shape(self, input_shape):
        return input_shape[:-1]


# In[4]:


class Mask(layers.Layer):
    def call(self, inputs, **kwargs):
        if type(inputs) is list: 
            assert len(inputs) == 2
            inputs, mask = inputs
        else:
            x = inputs
 
            x = (x - K.max(x, 1, True)) / K.epsilon() + 1
            mask = K.clip(x, 0, 1)
 
        inputs_masked = K.batch_dot(inputs, mask, [1, 1])
        return inputs_masked
 
    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:
            return tuple([None, input_shape[0][-1]])
        else:
            return tuple([None, input_shape[-1]])


# In[5]:


def squash(vectors, axis=-1):
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm)
    return scale * vectors


# In[6]:


class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsule, dim_vector, num_routing=3,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_vector = dim_vector
        self.num_routing = num_routing
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
 
    def build(self, input_shape):
        #assert len(input_shape) >= 3,
        self.input_num_capsule = input_shape[1]
        self.input_dim_vector = input_shape[2]
 
        self.W = self.add_weight(shape=[self.input_num_capsule, self.num_capsule, self.input_dim_vector, self.dim_vector],
                                 initializer=self.kernel_initializer,
                                 name='W')
 
        self.bias = self.add_weight(shape=[1, self.input_num_capsule, self.num_capsule, 1, 1],
                                    initializer=self.bias_initializer,
                                    name='bias',
                                    trainable=False)
        self.built = True
 
    def call(self, inputs, training=None):
        inputs_expand = K.expand_dims(K.expand_dims(inputs, 2), 2)
 
        inputs_tiled = K.tile(inputs_expand, [1, 1, self.num_capsule, 1, 1])
 
        inputs_hat = tf.scan(lambda ac, x: K.batch_dot(x, self.W, [3, 2]),
                             elems=inputs_tiled,
                             initializer=K.zeros([self.input_num_capsule, self.num_capsule, 1, self.dim_vector]))
 
        assert self.num_routing > 0, 'The num_routing should be > 0.'
        for i in range(self.num_routing):
            c = tf.nn.softmax(self.bias, dim=2)
 
            outputs = squash(K.sum(c * inputs_hat, 1, keepdims=True))
 
 
            if i != self.num_routing - 1:
                self.bias += K.sum(inputs_hat * outputs, -1, keepdims=True)
        return K.reshape(outputs, [-1, self.num_capsule, self.dim_vector])
 
    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_vector])


# In[7]:


def PrimaryCap(inputs, dim_vector, n_channels, kernel_size, strides, padding):
    output = layers.Conv2D(filters=dim_vector*n_channels, kernel_size=kernel_size, strides=strides, padding=padding)(inputs)
    outputs = layers.Reshape(target_shape=[-1, dim_vector])(output)
    return layers.Lambda(squash)(outputs)


# In[8]:


from keras import backend as K
def CapsNet(input_shape, n_class, num_routing):
    x = layers.Input(shape=input_shape)
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)
    primarycaps = PrimaryCap(conv1, dim_vector=8, n_channels=32, kernel_size=9, strides=2, padding='valid')
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_vector=16, num_routing=num_routing, name='digitcaps')(primarycaps)
    out_caps = Length(name='out_caps')(digitcaps)
 
    y = layers.Input(shape=(n_class,))
    masked = Mask()([digitcaps, y])
    x_recon = layers.Dense(512, activation='relu')(masked)
    x_recon = layers.Dense(1024, activation='relu')(x_recon)
    x_recon = layers.Dense(width*breadth*3, activation='sigmoid')(x_recon)
    x_recon = layers.Reshape(target_shape=[width, breadth, 3], name='out_recon')(x_recon)
 
    return models.Model([x, y], [out_caps, x_recon])


# In[9]:


def margin_loss(y_true, y_pred):
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) +         0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))
 
    return K.mean(K.sum(L, 1))


# In[10]:


width, breadth = 32, 32


# In[11]:


model = CapsNet(input_shape=[width, breadth, 3],
                n_class=5,
                num_routing=3)
model.summary()
try:
    plot_model(model, to_file='model.png', show_shapes=True)
except Exception as e:
    print('No fancy plot {}'.format(e))


# In[12]:


# make data, train 16000, test 400 images
# 1_left なら patientID = 1の人の左目の写真ということらしい


# In[13]:


trainCSV = pd.read_csv('../input/diabetic-retinopathy-detection/trainLabels.csv')
trainCSV['PatientId'] = trainCSV['image'].map(lambda x: x.split('_')[0])
trainCSV['imagePath'] = trainCSV['image'].map(lambda x: os.path.join('../input/diabetic-retinopathy-detection/','{}.jpeg'.format(x)))
trainCSV['exists'] = trainCSV['imagePath'].map(os.path.exists)
trainCSV['leftorright'] = trainCSV['image'].map(lambda x: 'left' if x.split('_')[-1]=='left' else 'right')
trainCSV['label'] = trainCSV['level'].map(lambda x: to_categorical(x, 5))
trainCSV.dropna(inplace = True)
trainCSV = trainCSV[trainCSV['exists']]


# In[14]:


trainCSV.head()


# In[15]:


from PIL import Image
import time
import sys


# In[16]:


trainCSV.shape


# In[17]:


def transformImagetoArray(imagePathsList, width=480, breadth=480):
    startTime = time.time()
    imagesArrayList = []
    for imagePath in imagePathsList:
        image = np.array(Image.open(imagePath).resize((width, breadth)), np.float).reshape(width, breadth, 3)
        imagesArrayList.append(image)
    print ("needed_time_for_makingArrays: {}".format(time.time()-startTime) + "[sec]")
    imagesArray = np.asarray(imagesArrayList)
    print('imagesArray: {} MB'.format(str(sys.getsizeof(imagesArray) / (10**6))))
    return imagesArray


# In[18]:


x = transformImagetoArray(list(trainCSV['imagePath']), width=width, breadth=breadth)
y = np.asarray(list(trainCSV['label']))


# In[19]:


from sklearn.model_selection import train_test_split

#data = np.load('kaggle_retina_datasets.small.npz')
#x, y = data['x'], data['y']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# In[20]:


#x_train = x_train.reshape(1600, 480, 480, 3).astype('float32') / 255.
#x_test = x_test.reshape(400, 480, 480, 3).astype('float32') / 255.
#y_train = to_categorical(y_train.astype('float32'))
#y_test = to_categorical(y_test.astype('float32'))


# In[21]:


import gc
del x
gc.collect()


# In[22]:


print('x train: %s' % str(x_train.shape))
print('x test: %s' % str(x_test.shape))
print('y train: %s' % str(y_train.shape))
print('y test: %s' % str(y_test.shape))


# In[23]:


def train(model, data, epoch_size_frac=1.0, epochs=100, batch_size=64):
 
    (x_train, y_train), (x_test, y_test) = data
    
    log = callbacks.CSVLogger('log.csv')
    checkpoint = callbacks.ModelCheckpoint('weights-{epoch:02d}val_loss-{val_loss}.h5',
                                           save_best_only=True, save_weights_only=False, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: 0.001 * np.exp(-epoch / 10.))
    # add early stopping
    early_stopping = callbacks.EarlyStopping(monitor = 'val_loss', min_delta=0, patience = 5, verbose = 1)
    model.compile(optimizer='adam',
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., 0.0005],
                  metrics={'out_caps': 'accuracy'})
 
    # -----------------------------------Begin: Training with data augmentation -----------------------------------#
    def train_generator(x, y, batch_size, shift_fraction=0.):
        train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           height_shift_range=shift_fraction)
        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield ([x_batch, y_batch], [y_batch, x_batch])
    # change max_queue_size from default 10 to 2
    model.fit_generator(generator=train_generator(x_train, y_train, batch_size, 0.1),
                        max_queue_size=2,
                        steps_per_epoch=int(epoch_size_frac*y_train.shape[0] / batch_size),
                        epochs=epochs,
                        validation_data=[[x_test, y_test], [y_test, x_test]],
                        callbacks=[log, checkpoint, lr_decay, early_stopping])
    # -----------------------------------End: Training with data augmentation -----------------------------------#
 
    model.save('trained_model.h5')
    print('Trained model saved to \'trained_model.h5\'')
 
    return model


# In[24]:


train(model=model, data=((x_train, y_train), (x_test[:60], y_test[:60])), 
      epoch_size_frac = 1)


# In[25]:


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(np.sqrt(num))
    height = int(np.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] =             img[:, :, 0]
    return image
 
def test(model, data):
    x_test, y_test = data
    y_pred, x_recon = model.predict([x_test, y_test], batch_size=100)
    print('-'*50)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])
 
    import matplotlib.pyplot as plt
    from PIL import Image
 
    img = combine_images(np.concatenate([x_test[:50],x_recon[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save("real_and_recon.png")
    print()
    print('Reconstructed images are saved to ./real_and_recon.png')
    print('-'*50)
    plt.imshow(plt.imread("real_and_recon.png", ))
    plt.show()


# In[26]:


test(model=model, data=(x_test[:100], y_test[:100]))


# In[27]:


del x_train, x_test, y_train, y_test
gc.collect()


# In[28]:


# DISK 容量的にkernel上でsubmissionを作るのは無理そう


# In[29]:


# なんかデータセットまとめてくれてる人いた


# In[30]:


len(os.listdir('../input/resized-2015-2019-blindness-detection-images/resized test 15'))


# In[31]:


def predictandSaveSubmission():
    imagePaths = os.listdir('../input/resized-2015-2019-blindness-detection-images/resized test 15')
    predictionList = []
    for imagePath in imagePaths:
        tmp = [int(imagePath.split('.')[0].split('_')[0]), imagePath.split('.')[0].split('_')[1], imagePath.split('.')[0]]
        imagePath = '../input/resized-2015-2019-blindness-detection-images/resized test 15/' + imagePath
        imageArray = np.array([np.array(Image.open(imagePath).resize((width, breadth)), np.float).reshape(width, breadth, 3)])
        y_pred, _ = model.predict_on_batch([imageArray, np.zeros((1, 5))])
        #tmp += [int(np.argmax(y_pred))]
        tmp += [1]
        predictionList.append(tmp)
    predictionList.sort(key=lambda x: (x[0], x[1]))
    predictionDict = {}
    predictionDict['image'] = [i[2] for i in predictionList]
    predictionDict['level'] = [i[3] for i in predictionList]
    
    df_submission = pd.DataFrame(predictionDict)
    df_submission.to_csv("submission.csv",index=False)


# In[32]:


predictandSaveSubmission()

