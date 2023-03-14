#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from scipy import signal
import numpy as np  
import pandas as pd 
import numpy as np 
import pandas as pd 
from scipy.io import wavfile
import os
import glob
import pickle
from sklearn.model_selection import train_test_split 
import librosa as lbr
import IPython.display as ipd
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
import os
import librosa.display
print(os.listdir("../input"))


# In[2]:


INPUT_FOLDER = "../input/"
train_files = glob.glob("../input/train_curated/*.wav")
test_files=glob.glob("../input/test/*.wav")
print(os.listdir(INPUT_FOLDER))


# In[3]:


TEST = INPUT_FOLDER + "sample_submission.csv"
test = pd.read_csv(TEST)


# In[4]:


train_curated = pd.read_csv("../input/train_curated.csv")
train_curated['is_curated'] = True
train_noisy = pd.read_csv('../input/train_noisy.csv')
train_noisy['is_curated'] = False
train = pd.concat([train_curated, train_noisy], axis=0)
del train_noisy


# In[5]:





# In[5]:


print("Number of train examples=", train.shape[0], "  Number of classes=", len(set(train.labels)))
print("Number of test examples=", test.shape[0], "  Number of classes=", len(set(test.columns[1:])))


# In[6]:


#get only the lables that are in the testing file
#train is for one lable per class data, train_curated is the multilabel dataset
# train = train[train.labels.isin(test.columns[1:])]
# print(len(train))
# category_group = train.groupby(['labels']).count()['fname']
# category_group.columns = ['counts']


# In[7]:


print('Minimum samples per category = ', min(train.labels.value_counts()))
print('Maximum samples per category = ', max(train.labels.value_counts()))


# In[8]:





# In[8]:


train['n_label'] = train.labels.str.split(',').apply(lambda x: len(x))
print('curated\n',train.query('is_curated == True').n_label.value_counts())
print('noisy\n',train.query('is_curated == False').n_label.value_counts())


# In[9]:


#chacking the multilables
#[label.split(',') for i, label in enumerate(train['labels']) if len(label.split(',')) >=2] 


# In[10]:


#get target names from test 
target_names = test.columns[1:]
target_names.shape


# In[11]:


num_targets = len(target_names)

src_dict = {target_names[i]:i for i in range(num_targets)}
src_dict_inv = {i:target_names[i] for i in range(num_targets)}


# In[12]:


def one_hot(labels, src_dict):
    ar = np.zeros([len(labels), len(src_dict)])
    invalid=['77b925c2.wav','f76181c4.wav', '6a1f682a.wav', 'c7db12aa.wav', '7752cc8a.wav','1d44b0bd.wav']
    for i, label in enumerate(labels): 
        if label not in invalid:
            label_list = label.split(',')
            for la in label_list:
                ar[i, src_dict[la]] = 1
    return ar


# In[13]:


import IPython.display as ipd  # To play sound in the notebook
track=train[train.is_curated==True].fname.sample(1).values[0]
path = '../input/train_curated/{}'.format(track)   
label=train[train.fname==track].labels.values[0]
print(label)
ipd.Audio(path)


# In[14]:


track_n=train[(train.is_curated==False)&(train.labels ==label)].sample(1).fname.values[0]
path_n = '../input/train_noisy/{}'.format(track_n)   
print(train[train.fname==track_n].labels.values[0])
print(train[train.fname==track_n].fname.values[0])
ipd.Audio(path_n)


# In[15]:


audio, sample_rate=lbr.load(path,sr=44100)
n_fft = int(0.03 * sample_rate) #25ms window length
hop_length =  n_fft//2
N_MELS = 128 #frequency bins
#X = lbr.stft(audio[0], n_fft=n_fft, hop_length=hop_length)
S=lbr.feature.melspectrogram(audio,n_fft=n_fft, hop_length=hop_length,n_mels=N_MELS )
S = lbr.amplitude_to_db(abs(S))
#S=np.log(X)
plt.figure(figsize=(15, 5))
lbr.display.specshow(S, sr=44100, hop_length=hop_length, x_axis='time',cmap='magma')
plt.colorbar(format='%+2.0f dB')


# In[16]:


audio, sample_rate=lbr.load(path_n,sr=44100)
n_fft = int(0.03 * sample_rate) #25ms window length
hop_length =  n_fft//2
N_MELS = 128 #frequency bins
#X = lbr.stft(audio[0], n_fft=n_fft, hop_length=hop_length)
S=lbr.feature.melspectrogram(audio,n_fft=n_fft, hop_length=hop_length,n_mels=N_MELS )
S = lbr.amplitude_to_db(abs(S))
#S=np.log(X)
plt.figure(figsize=(15, 5))
lbr.display.specshow(S, sr=44100, hop_length=hop_length, x_axis='time',cmap='magma')
plt.colorbar(format='%+2.0f dB')


# In[17]:


track = audio[0:int(1 * sample_rate)] #5 secs of audio
plt.plot(track)


# In[18]:


from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.layers import Input, Dense, Dropout, Activation,          Convolution1D, MaxPooling1D, BatchNormalization, Flatten,GlobalAveragePooling1D,Convolution2D,MaxPooling2D
import scipy
from keras import losses
from keras import backend as K
from keras.activations import relu, softmax
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
from keras.utils import Sequence
import shutil


# In[19]:


class Config(object):
    def __init__(self,
                 sampling_rate=44100, audio_duration=2, #audio duration: specify length of the track in sec
                 n_classes=target_names,
                 use_mfcc=True, n_folds=1, learning_rate=0.0001, 
                 max_epochs=30, n_mfcc=64):
        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.n_classes = n_classes
        self.use_mfcc = use_mfcc
        self.n_mfcc = n_mfcc
        self.n_folds = n_folds
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.win_len= int(0.02 * sample_rate) #ms window length

        self.audio_length = self.sampling_rate * self.audio_duration
        if self.use_mfcc:
           # self.dim = (self.n_mfcc, 1 + int(np.floor(self.audio_length/self.win_len)*2))
            self.dim = (self.n_mfcc,401)
        else:
            self.dim = (self.audio_length)


# In[20]:


#lwrap implementation for keras
def tf_one_sample_positive_class_precisions(y_true, y_pred) :
    num_samples,num_classes = y_pred.shape
    
    # find true labels
    pos_class_indices = tf.where(y_true > 0) 
    
    # put rank on each element
    retrieved_classes = tf.nn.top_k(y_pred, k=num_classes).indices
    sample_range = tf.zeros(shape=tf.shape(tf.transpose(y_pred)), dtype=tf.int32)
    sample_range = tf.add(sample_range, tf.range(tf.shape(y_pred)[0], delta=1))
    sample_range = tf.transpose(sample_range)
    sample_range = tf.reshape(sample_range, (-1,num_classes*tf.shape(y_pred)[0]))
    retrieved_classes = tf.reshape(retrieved_classes, (-1,num_classes*tf.shape(y_pred)[0]))
    retrieved_class_map = tf.concat((sample_range, retrieved_classes), axis=0)
    retrieved_class_map = tf.transpose(retrieved_class_map)
    retrieved_class_map = tf.reshape(retrieved_class_map, (tf.shape(y_pred)[0], num_classes, 2))
    
    class_range = tf.zeros(shape=tf.shape(y_pred), dtype=tf.int32)
    class_range = tf.add(class_range, tf.range(num_classes, delta=1))
    
    class_rankings = tf.scatter_nd(retrieved_class_map,
                                          class_range,
                                          tf.shape(y_pred))
    
    #pick_up ranks
    num_correct_until_correct = tf.gather_nd(class_rankings, pos_class_indices)

    # add one for division for "presicion_at_hits"
    num_correct_until_correct_one = tf.add(num_correct_until_correct, 1) 
    num_correct_until_correct_one = tf.cast(num_correct_until_correct_one, tf.float32)
    
    # generate tensor [num_sample, predict_rank], 
    # top-N predicted elements have flag, N is the number of positive for each sample.
    sample_label = pos_class_indices[:, 0]   
    sample_label = tf.reshape(sample_label, (-1, 1))
    sample_label = tf.cast(sample_label, tf.int32)
    
    num_correct_until_correct = tf.reshape(num_correct_until_correct, (-1, 1))
    retrieved_class_true_position = tf.concat((sample_label, 
                                               num_correct_until_correct), axis=1)
    retrieved_pos = tf.ones(shape=tf.shape(retrieved_class_true_position)[0], dtype=tf.int32)
    retrieved_class_true = tf.scatter_nd(retrieved_class_true_position, 
                                         retrieved_pos, 
                                         tf.shape(y_pred))
    # cumulate predict_rank
    retrieved_cumulative_hits = tf.cumsum(retrieved_class_true, axis=1)

    # find positive position
    pos_ret_indices = tf.where(retrieved_class_true > 0)

    # find cumulative hits
    correct_rank = tf.gather_nd(retrieved_cumulative_hits, pos_ret_indices)  
    correct_rank = tf.cast(correct_rank, tf.float32)

    # compute presicion
    precision_at_hits = tf.truediv(correct_rank, num_correct_until_correct_one)
    return pos_class_indices, precision_at_hits

def tf_lwlrap(y_true, y_pred):
    num_samples,num_classes = y_pred.shape
    
    pos_class_indices, precision_at_hits = (tf_one_sample_positive_class_precisions(y_true, y_pred))
    pos_flgs = tf.cast(y_true > 0, tf.int32)
    labels_per_class = tf.reduce_sum(pos_flgs, axis=0)
    weight_per_class = tf.truediv(tf.cast(labels_per_class, tf.float32),
                                  tf.cast(tf.reduce_sum(labels_per_class), tf.float32))
    sum_precisions_by_classes = tf.zeros(shape=(num_classes), dtype=tf.float32)  
    class_label = pos_class_indices[:,1]
    sum_precisions_by_classes = tf.unsorted_segment_sum(precision_at_hits,
                                                        class_label,
                                                       num_classes)
    labels_per_class = tf.cast(labels_per_class, tf.float32)
    labels_per_class = tf.add(labels_per_class, 1e-7)
    per_class_lwlrap = tf.truediv(sum_precisions_by_classes,
                                  tf.cast(labels_per_class, tf.float32))
    out = tf.cast(tf.tensordot(per_class_lwlrap, weight_per_class, axes=1), dtype=tf.float32)
    return out


# In[21]:


def audio_norm(data):
#     max_data = np.max(data)
#     min_data = np.min(data)
#     data = (data-min_data)/(max_data-min_data+1e-6)
#     return data - 0.5
    data = ( data - np.mean(data) ) / np.std(data)
    data /= np.max(data)
    return data


# In[22]:





# In[22]:


def build_model(config):
    
    nclass = len(config.n_classes)
    input_length = config.audio_length
    input_shape = (64,401,1)
    print(input_shape)
    rate=0.2
    model_input = Input(input_shape, name='input')
    layer = model_input
    layer = Convolution2D(32, (3,3) ,activation=tf.nn.leaky_relu,name='convolution_1' ,padding='same',strides=(2,2))(layer)
   # layer = BatchNormalization(momentum=0.9)(layer) #momentum=0.9
    layer=MaxPooling2D(pool_size=(2, 2), strides=(2,2),padding='same')(layer)
    layer = Dropout(rate)(layer)
    
    layer = Convolution2D(64, (3,3) ,activation=tf.nn.leaky_relu,name='convolution_2' , padding='same',strides=(2,2))(layer)
   # layer = BatchNormalization(momentum=0.9)(layer) #momentum=0.9
    layer=MaxPooling2D(pool_size=(2, 2), strides=(2,2),padding='same')(layer)
    layer = Dropout(rate)(layer)
    
    layer = Convolution2D(64, (3,3) ,activation=tf.nn.leaky_relu,name='convolution_3' , padding='same')(layer)
   # layer = BatchNormalization(momentum=0.9)(layer) #momentum=0.9
    layer=MaxPooling2D(pool_size=(2, 2), strides=(2,2),padding='same')(layer)
    layer = Dropout(rate)(layer)
    
    layer = Convolution2D(128, (3,3) ,activation=tf.nn.leaky_relu,name='convolution_4' , padding='same')(layer)
  #  layer = BatchNormalization(momentum=0.9)(layer) #momentum=0.9
    layer=MaxPooling2D(pool_size=(2, 2), strides=(2,2),padding='same')(layer)
    layer = Dropout(rate)(layer)
    
    layer= Flatten()(layer)
    layer = Dense(256)(layer)
    layer = Dropout(rate)(layer)
    layer = Dense(nclass)(layer)
    
    output = Activation('softmax', name='Final_output')(layer)
    model = Model(model_input, output)
   # opt = Adam(lr=config.learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[tf_lwlrap])
    return model

  


# In[23]:


sample_rate


# In[24]:


# trim silent part
def trim_silent(data):
    data_tr = librosa.effects.trim(data)[0]
    return data_tr


# In[25]:


def prepare_data(df, config, data_dir,downsample=False):
    input_length = config.audio_length
    WINDOW_SIZE = int(0.03 * config.sampling_rate)
    hop_length =  n_fft//4
    N_MELS = 64#frequency bins

    MEL_KWARGS = {
        'n_fft': WINDOW_SIZE,
        'hop_length': hop_length,
        'n_mels': N_MELS 
    }
    X = np.empty(shape=(df.shape[0], config.dim[0], config.dim[1],1))
    
    invalid=['77b925c2.wav','f76181c4.wav', '6a1f682a.wav', 'c7db12aa.wav', '7752cc8a.wav','1d44b0bd.wav']
    for i, fname in enumerate(df.index):
            
            if fname not in invalid:
                file_path = data_dir + fname
                data, _ = librosa.core.load(file_path, sr=44100, res_type="kaiser_fast")
                if len(audio)/44100>=0.5:
                    trim_silent(data)
                   # print('data_shape: ',data.shape)
                    # Random offset / Padding
                    if len(data) > input_length:
                        max_offset = len(data) - input_length
                        offset = np.random.randint(max_offset)
                        data = data[offset:(input_length+offset)]
                    else:
                        if input_length > len(data):
                            max_offset = input_length - len(data)
                            offset = np.random.randint(max_offset)
                        else:
                            offset = 0
                        #pad with zeros
                        data = np.pad(data, (offset, input_length - len(data) - offset), "constant")
                    #print('before spec: ',data.shape)
                   # data = librosa.feature.mfcc(data, sr=config.sampling_rate, n_mfcc=config.n_mfcc,**MEL_KWARGS).T
                    if downsample==True:
                        data = librosa.feature.melspectrogram(lbr.resample(data, 44100, 44100/2),**MEL_KWARGS)
                    else:
                        data = librosa.feature.melspectrogram(data,**MEL_KWARGS)
                    data = lbr.core.power_to_db(data)
                    #print('after padding')
                    #print(data.shape)
                    X[i,] = data[:,:,np.newaxis]
            else:
                    print(fname)
    return X
    


# In[26]:


train.set_index("fname", inplace=True)
test.set_index('fname',inplace=True)


# In[27]:


downsample=False
if downsample==True:
    config = Config(sampling_rate=22050, audio_duration=3, n_folds=1, learning_rate=0.001, use_mfcc=True, n_mfcc=64,max_epochs=40)
else:
    config = Config(sampling_rate=44100, audio_duration=3, n_folds=1, learning_rate=0.001, use_mfcc=True, n_mfcc=64,max_epochs=40)
X_train=prepare_data(train[train.is_curated==True],config,'../input/train_curated/',downsample)
X_test=prepare_data(test,config,'../input/test/',downsample)


# In[28]:


y = one_hot(train_curated['labels'], src_dict)


# In[29]:


y_noisy=one_hot(train[train.is_curated==False].labels, src_dict)


# In[30]:





# In[30]:


X_test.shape


# In[31]:


X = audio_norm(X_train)
X_test = audio_norm(X_test)


# In[32]:


y_noisy.shape


# In[33]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=10)


# In[34]:


PREDICTION_FOLDER = "predictions_1d_conv"
if not os.path.exists(PREDICTION_FOLDER):
    os.mkdir(PREDICTION_FOLDER)
if os.path.exists('logs/' + PREDICTION_FOLDER):
    shutil.rmtree('logs/' + PREDICTION_FOLDER)
    
K.clear_session()
checkpoint = ModelCheckpoint('best.h5', monitor='val_tf_lwlrap', verbose=1, save_best_only=True)
early = EarlyStopping(monitor="val_loss", mode="min", patience=10)
tb = TensorBoard(log_dir='./logs/' + PREDICTION_FOLDER + '/fold', write_graph=True)
callbacks_list = [checkpoint, tb,early]
model = build_model(config)
history = model.fit(X_train,y_train,validation_data=(X_val,y_val),callbacks=callbacks_list,batch_size=128, epochs=40,shuffle=True)
#model.load_weights('best.h5')
#history = model.fit(X,y,callbacks=callbacks_list,batch_size=256, epochs=40,shuffle=True)

# # Save train predictions
# predictions = model.predict(X_train, batch_size=64, verbose=1)
# np.save(PREDICTION_FOLDER + "/train_predictions.npy", predictions)

# # Save test predictions
# predictions = model.predict(X_test, batch_size=64, verbose=1)
# np.save(PREDICTION_FOLDER + "/test_predictions.npy", predictions)

# # Make a submission file
# top_3 = np.array(config.n_classes)[np.argsort(-predictions, axis=1)[:, :3]]
# predicted_labels = [' '.join(list(x)) for x in top_3]
# test['label'] = predicted_labels
# test[['label']].to_csv(PREDICTION_FOLDER + "/predictions.csv") 


# In[35]:


for fname in train[train.is_curated==False].index
    
    


# In[36]:





# In[36]:


model.evaluate(X_val[20:21],y_val[20:21],batch_size=1)


# In[37]:


#prepare noisy data
X_noisy=prepare_data(train[train.is_curated==False],config,'../input/train_noisy/',downsample)


# In[38]:


results=[]
for i in range(y_noisy.shape[0]):
    r=model.evaluate(X_noisy[i:i+1],y_noisy[i:i+1],batch_size=1)
    results.append(r[1])


# In[39]:


results=np.array(results)


# In[40]:


print(min(results))
print(max(results))


# In[41]:


indexes=[]
for i,j in enumerate(results):
    if(j>=0.1):
        indexes.append(i)
        


# In[42]:


d=np.vstack((X_train,X_noisy[indexes]))
y2=np.vstack((y_train,y_noisy[indexes]))


# In[43]:


PREDICTION_FOLDER = "predictions_1d_conv"
if not os.path.exists(PREDICTION_FOLDER):
    os.mkdir(PREDICTION_FOLDER)
if os.path.exists('logs/' + PREDICTION_FOLDER):
    shutil.rmtree('logs/' + PREDICTION_FOLDER)
    
model.load_weights('best.h5')
checkpoint = ModelCheckpoint('best.h5', monitor='val_loss', verbose=1, save_best_only=True)
early = EarlyStopping(monitor="val_loss", mode="min", patience=10)
tb = TensorBoard(log_dir='./logs/' + PREDICTION_FOLDER + '/fold', write_graph=True)
callbacks_list = [checkpoint, tb,early]
model = build_model(config)
history = model.fit(d,y2,validation_data=(X_val,y_val),callbacks=callbacks_list,batch_size=128, epochs=50,shuffle=True)


# In[44]:


submission= model.predict(X_test, verbose=1)


# In[45]:


# Output all random to see a baseline
sample_sub = pd.read_csv('../input/sample_submission.csv')
sample_sub.iloc[:,1:] = submission
sample_sub.to_csv('submission.csv', index=False)

