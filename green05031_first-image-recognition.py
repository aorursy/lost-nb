%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h5py


from sklearn.model_selection import train_test_split
from keras import initializers
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D, Dense, Dropout, BatchNormalization, Input, Flatten, Activation, ZeroPadding2D, Lambda, GlobalAveragePooling2D, concatenate
from keras.layers.merge import Concatenate, add
from keras.optimizers import Adam, SGD, Adamax
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from skimage.restoration import denoise_tv_chambolle, wiener
from skimage.filters import gaussian, sobel, frangi, median, laplace

from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance


def load_train_data():
    data = pd.read_json('../input/train.json')       
    return data

def load_test_data():
    data = pd.read_json('../input/test.json')
        
    return data

def denoise(x, weight, multichannel):
    return np.asarray([denoise_tv_chambolle(item, weight=weight, multichannel=multichannel)for item in x])

#def denoise(x, multichannel):
 #   return np.asarray([denoise_bilateral(item, multichannel=multichannel) for item in x])

def gaussian_blur(x, sigma, multichannel):
    return np.asarray([gaussian(item, sigma=sigma, multichannel=multichannel) for item in x])

def color_composite(data):
    rgb_arrays = []
    for i, row in data.iterrows():
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 / band_2
        
        r = (band_1 + abs(band_1.min())) / np.max((band_1 + abs(band_1.min())))
        g = (band_2 + abs(band_2.min())) / np.max((band_2 + abs(band_2.min())))
        b = (band_3 + abs(band_3.min())) / np.max((band_3 + abs(band_3.min())))
        
        rgb = np.dstack((r, g, b))
        rgb_arrays.append(rgb)
    return np.array(rgb_arrays)

from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance

def lee_filter(img, size):
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(img)

    img_weights = img_variance**2 / (img_variance**2 + overall_variance**2)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output

train_df = load_train_data()
test_df = load_test_data()

label = np.array(train_df["is_iceberg"])
train_angle = train_df['inc_angle']
train_angle = train_angle.replace('na',np.nan)
train_angle = train_angle.astype(float).fillna(np.mean(train_angle))
test_angle = test_df['inc_angle']
test_angle = test_angle.replace('na',np.nan)
test_angle = test_angle.astype(float).fillna(0)
print('Done')

def create_dataset(data, labeled, weight_gray=0.05, weight_rgb=0.05):
    band_1, band_2, images = data['band_1'].values, data['band_2'], color_composite(data)
    to_arr = lambda x: np.asarray([np.asarray(item) for item in x])
    band_1 = to_arr(band_1)
    band_2 = to_arr(band_2)
    band_3 = (band_1 / band_2)
    
    to_rgb = lambda x: np.asarray([(item + abs(item.min())) / np.max((item + abs(item.min()))) for item in x])
    band_1 = to_rgb(band_1)
    band_2 = to_rgb(band_2)
    band_3 = to_rgb(band_3)
    
    
    gray_reshape = lambda x: np.asarray([item.reshape(75, 75) for item in x])
  
    band_1 = gray_reshape(band_1)
    band_2 = gray_reshape(band_2)
    band_3 = gray_reshape(band_3)
    print('denoising')
    
    #band_1 = denoise(band_1, 0.1, False)
    #band_2 = denoise(band_2, 0.1, False)
    #band_3 = denoise(band_3, 0.1, False)

        
    #images = denoise(images, weight_rgb, True)
    #print('blur')
    
    #band_1 = gaussian(band_1, 0.2, False)
    #band_2 = gaussian(band_2, 0.2, False)
    #band_3 = gaussian(band_3, 0.2, False)
    
    #images = gaussian_blur(images, 0.2, True)
    
    tf_reshape = lambda x: np.asarray([item.reshape(75, 75, 1) for item in x])
    band_1 = tf_reshape(band_1)
    band_2 = tf_reshape(band_2)
    band_3 = tf_reshape(band_3)
    img_reshape = lambda x: np.asarray([item.reshape(75,75,3) for item in x])
    images = img_reshape(images)
    
    band = np.concatenate([band_1, band_2, band_3], axis=3)
    
    x_angle = data.inc_angle
    x_angle = x_angle.replace('na',np.nan)
    x_angle = x_angle.astype(float).fillna(np.mean(x_angle))
    x_angle = np.array(x_angle)
    if labeled:
        y = np.array(data['is_iceberg'])
    else:
        y = None
    
    print('Done')
    return y, x_angle, band, images

y, x_angle, x_band, x_img = create_dataset(train_df, True)

np.amax(x_band)

fig = plt.figure(200, figsize=(15,15))
random_indicies = np.random.choice(range(len(x_band)), 9, False)
subset = x_band[random_indicies]
for i in range(9):
    ax = fig.add_subplot(3,3,i+1)
    ax.imshow(subset[i])
plt.show()

fig = plt.figure(200, figsize=(15, 15))
#random_indicies = np.random.choice(range(len(x_img)), 9, False)
subset = x_img[random_indicies]
for i in range(9):
    ax = fig.add_subplot(3, 3, i + 1)
    ax.imshow(subset[i])
plt.show()

from keras.utils.np_utils import to_categorical
x_train, x_val, \
x_angle_train, x_angle_val, \
y_train, y_val = train_test_split(x_img,x_angle, y, random_state=666, test_size=0.25)

print('img', x_train.shape, y_train.shape)
print('angle', x_angle_train.shape,y_val.shape)


adamax01 = Adamax(lr=0.0003)

def model2(kernel_size=3, filters_1=32, filters_2=32, filters_3=64, filters_4=128,
               optimizers=adamax01,init='lecun_normal',relu_type='selu'):
    band_input = Input(shape=(75, 75, 3))

    cnn = BatchNormalization()(band_input)
    for i in range(1):
        cnn = Conv2D(filters=filters_1, kernel_size=(kernel_size,kernel_size),
                     kernel_initializer=init)(cnn)
        cnn = Activation(relu_type)(cnn)
        cnn = MaxPooling2D((2,2), strides=(2,2))(cnn)
    for i in range(1):
        cnn = Conv2D(filters=filters_2, kernel_size=(kernel_size,kernel_size),
                     kernel_initializer=init)(cnn)
        cnn = Activation(relu_type)(cnn)
        cnn = MaxPooling2D((2,2), strides=(2,2))(cnn)
    for i in range(1):
        cnn = Conv2D(filters=filters_3, kernel_size=(kernel_size,kernel_size),
                     kernel_initializer=init)(cnn)
        cnn = Activation(relu_type)(cnn)
        cnn = MaxPooling2D((2,2), strides=(2,2))(cnn)
    for i in range(1):
        cnn = Conv2D(filters=filters_4, kernel_size=(kernel_size,kernel_size),
                     kernel_initializer=init)(cnn)
        cnn = Activation(relu_type)(cnn)
        cnn = MaxPooling2D((2,2), strides=(2,2))(cnn)
    
    
    cnn = Flatten()(cnn)
    cnn = Dense(512,activation= relu_type)(cnn)
    cnn = Dense(1, activation = 'sigmoid')(cnn)

    simple_cnn = Model(inputs=[band_input],outputs=cnn)

    simple_cnn.compile(optimizer=optimizers, loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return simple_cnn
m = model2()
m.summary()

filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

gen = ImageDataGenerator(horizontal_flip = True,
                         vertical_flip = True,
                         width_shift_range = 0,
                         height_shift_range = 0,
                         zoom_range = 0.2,
                         rotation_range = 20)
def gen_flow_for_one_inputs(X1, y):
    genX1 = gen.flow(X1,y,  batch_size=64,seed=1)
    while True:
        X1i = genX1.next()
        
        yield X1i[0],  X1i[1]
    
gen_flow = gen_flow_for_one_inputs(x_train, y_train)
gen_val = gen_flow_for_one_inputs(x_val, y_val)

#model = simple_cnn()
model = model2()
ss = model.fit_generator(gen_flow, validation_data=(x_val,y_val),
                         steps_per_epoch=len(x_train) / 64, 
                         epochs=130,
                         callbacks = callbacks_list,
                         verbose = 1)


plt.plot(ss.history['loss'])
plt.plot(ss.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='best')

model.load_weights(filepath=filepath)
score = model.evaluate(x_train, y_train,verbose=1)
print('Train loss:', score[0])
print('Train accuracy:', score[1])
test_score = model.evaluate(x_val, y_val,verbose=1)
print('Test loss:', test_score[0])
print('Test accuracy:', test_score[1])


model.load_weights(filepath=filepath)
#train_predictions = m.predict([x_band,x_angle])

#pre_label= np.round(train_predictions)
#pre_label = np.array(pre_label).flatten()
#pre_label = pre_label.astype('int64')
#exc_label = label
#corr_index = np.where((pre_label == exc_label))[0]
#incorr_index = np.where((pre_label != exc_label))[0]

#from sklearn.metrics import confusion_matrix
#plt.subplots(figsize=(12,12))
#g = sns.heatmap(confusion_matrix(exc_label,pre_label),annot=True, fmt='2.0f')
#g.set_xlabel('predicted labels')
#g.set_ylabel('true labels')

q, test_angle, test_band, test_img = create_dataset(test_df, False)

test_pred = model.predict([test_img])
test_pred = test_pred.reshape(-1)
print('Done')

img_indicies = np.random.choice(range(len(test_img)), len(x_train)*1, False)
test_sub = test_img[img_indicies]
test_pred_sub = test_pred[img_indicies]

#X = np.vstack((x_train, test_sub))
Y = np.concatenate((y_train, test_pred_sub))
print('Done')

 mm = model.fit_generator(gen_flow, validation_data=(x_val, y_val),
                          steps_per_epoch=len(X) / 32, 
                          epochs=10,
                          callbacks = callbacks_list,
                          verbose = 1,
                          shuffle=True)

model.load_weights(filepath=filepath)
test_score = model.evaluate(x_val, y_val,verbose=1)
print('Test loss:', test_score[0])
print('Test accuracy:', test_score[1])

plt.plot(mm.history['loss'])
plt.plot(mm.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='best')

model.load_weights(filepath=filepath)
test_predictions = model.predict([test_img])


pred_df = test_df[['id']].copy()
pred_df['is_iceberg'] = test_predictions
pred_df.to_csv('predictions.csv', index = False)
pred_df.head(3)


