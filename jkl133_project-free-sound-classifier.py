#!/usr/bin/env python
# coding: utf-8



batch_size = 32
BASE_FILTER_COUNT = 16
max_steps = 10
SAMPLING_RATE = 8000
input_length = SAMPLING_RATE*2




import numpy as np 
np.random.seed(1001)  # 특정 숫자를 지정하여 난수 설정
import os  # 경로 설정 시 사용
import shutil  # 파일 및 디렉터리 작업을 수행하는 데 사용할 모듈의 이름

import IPython
import matplotlib
import matplotlib.pyplot as plt  # 데이터 시각화를 위한 라이브러리
import pandas as pd  # 데이터 분석을 위한 라이브러리 설정
import seaborn as sns # matplotlib을 기반으로 하는 데이터 시각화 라이브러리
from tqdm import tqdm_notebook  # 반복문 진행상태를 확인할 수 있는 라이브러리
from sklearn.model_selection import StratifiedKFold

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.style.use('ggplot')




import IPython.display as ipd  # IPython의 display와 관련된 Public API: To play sound in the notebook
import wave  # wav 파일을 읽을 수 있는 모듈
from scipy.io import wavfile  #초당 샘플 수의 샘플링 속도와 파일에서 읽은 모든 데이터가 있는 numpy 배열을 반환
SAMPLE_RATE = 44100

import seaborn as sns 
color = sns.color_palette()
import plotly.offline as py  # 데이터 시각화 라이브러리
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline  # 오프라인에서 차트 작업하기
offline.init_notebook_mode()
import plotly.tools as tls
from plotly.offline import *

import numpy as np
from scipy.fftpack import fft  # 소리 특징(주파수) 추출을 위한 Discrete Fourier transforms 패키지 
from scipy import signal  # Signal processing 패키지
import librosa  # 파이썬 음악 분석 라이브러리




def audio_norm(data):
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data-min_data)/(max_data-min_data+0.0001)
    return data-0.5

def load_audio_file(file_path, input_length=input_length):
    data = librosa.core.load(file_path, sr=SAMPLING_RATE)[0] #sr=16000
    if len(data)>input_length:
        max_offset = len(data)-input_length
        offset = np.random.randint(max_offset)
        data = data[offset:(input_length+offset)]
    else:
        if input_length > len(data):
            max_offset = input_length - len(data)
            offset = np.random.randint(max_offset)
        else: 
            offset = 0
        data = np.pad(data, (offset, input_length - len(data) - offset), "constant")
    data = audio_norm(data)
    return data




import glob  # python에서 파일 리스트를 사용할 때 이용. 특정 이름의 파일을 찾고 파일 리스트를 받아서 처리할 때 유용
train_files = glob.glob("../input/audio_train/audio_train/*.wav")
test_files =glob.glob("../input/audio_test/audio_test/*.wav")
train = pd.read_csv("../input/train.csv")   
print(len(train_files), 'training', len(test_files), 'testing')
train.groupby(['label']).size().plot.bar()  # label을 기준으로 그룹화
train.sample(3)  # 3개만 제시 




train_audio_path = '../input/audio_train/audio_train'
filename = '/e6949d46.wav' 
sample_rate, samples = wavfile.read(str(train_audio_path) + filename)




print(samples)




print("Size of training data", train.shape)




train.head()




#제출용이 아니므로 submission.head()를 따로 출력하지 않음




def clean_filename(fname, string):
    file_name = fname.split('/')[1]
    if file_name[:2] == '__':
        file_name = string + file_name 
    return file_name

def load_wav_file(name, path):
    _, b = wavfile.read(path + name)
    assert _ == SAMPLE_RATE
    return b




train_data = pd.DataFrame({'file_name' : train['fname'], 'target' : train ['label']})
train_data['time_series'] = train_data['file_name'].apply(load_wav_file, 
                                                          path='../input/audio_train/audio_train/')
train_data['nframes'] = train_data['time_series'].apply(len)

# series: numpy array 하나를 표현하는데 데이터 프레임(행렬)의 차원에서 보면 하나의 칼럼에 해당하는 값들의 모음.
# dataframe: 여러 series들이 모여 하나의 매트릭스를 구성                                                         




train_data.head()




print ("Size of trainig data after some preprocessing : ", train_data.shape)




# missing data in trainig data set
total = train_data.isnull().sum().sort_values(ascending = False)
percent = (train_data.isnull().sum()/train_data.isnull().count()).sort_values(ascending = False)
missing_train_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_train_data.head()




temp = train['manually_verified'].value_counts()
labels = temp.index
sizes = (temp / temp.sum())*100
trace = go.Pie(labels=labels, values=sizes, hoverinfo='label+percent')
layout = go.Layout(title='Manually verification of labels(0 - No, 1 - Yes)')
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)




plt.figure(figsize=(12,8))
sns.distplot(train_data.nframes.values, bins=50, kde=False)
plt.xlabel('nframes', fontsize=12)
plt.title("Histogram of #frames")
plt.show()




plt.figure(figsize=(17,8))
boxplot = sns.boxplot(x="target", y="nframes", data=train_data)
boxplot.set(xlabel='', ylabel='')
plt.title('Distribution of audio frames, per label', fontsize=17)
plt.xticks(rotation=80, fontsize=17)
plt.yticks(fontsize=17)
plt.xlabel('Label name')
plt.ylabel('nframes')
plt.show()




print ("Total number of labels in trainig data : ",len(train_data['target'].
                                                      value_counts()))
print("Labels are : ", train_data['target'].unique())
plt.figure(figsize=(15,8))
audio_type = train_data['target'].value_counts().head(30)
sns.barplot(audio_type.values, audio_type.index)
for i, v in enumerate(audio_type.values):
    plt.text(0.8,i,v,color='k',fontsize=12)
plt.xticks(rotation='vertical')
plt.xlabel('Frequency')
plt.ylabel('Label Name')
plt.title("Top 30 labels woth their frequencies on traing data")
plt.show()




print("Number of traing examples=", train.shape[0], "  NUmber of classes=", len(train.label.unique()))




print(train.label.unique())




category_group = train.groupby(['label', 'manually_verified']).count()
plot = category_group.unstack().reindex(category_group.unstack().sum(axis=1).sort_values().index).plot(kind='bar', stacked=True, title="NUmber of Audio Sample per category", figsize=(16,10))
plot.set_xlabel("Category")
plot.set_ylabel("Number of Samples");




print('Minimum samples per category = ', min(train.label.value_counts()))
print('Maximum samples per category = ', max(train.label.value_counts()))




import IPython.display as ipd # To play sound in the notebook
fname = '../input/audio_train/audio_train/e6949d46.wav'
train_path = '../input/audio_train/audio_train'
test_path = '../input/audio_test/audio_test'
ipd.Audio(fname)




import wave
wav = wave.open(fname)
print("Sampling (fname) rate = ", wav.getframerate())
print("Total samples (frames) = ", wav.getnframes())
print("Duration = ", wav.getnframes()/wav.getframerate())




# Using scipy
# scipy.io.wavfile.read(filename, mmap=False) %rate, data 추출
from scipy.io import wavfile
rate, data = wavfile.read(fname)
print("Sampling (frame) rate = ", rate)
print("Total samples (frames) = ", data.shape)
print(data)




plt.plot(data, '-', );




plt.figure(figsize=(16,4))
plt.plot(data[:500], '-'); plt.plot(data[:500], '-');




# Setup variables
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Dropout, MaxPool1D, Flatten
input_length = 44100*10 
n_classes = train['label'].unique().shape[0]




#Create model
model = Sequential()
model.add(Conv1D(filters=4, kernel_size=16, activation='relu', padding='same', 
                 input_shape=(input_length,1)))
model.add(MaxPool1D(pool_size=5))
model.add(Dropout(rate=0.1))
model.add(Conv1D(filters=6, kernel_size=16, activation='relu', padding='same'))
model.add(MaxPool1D(pool_size=5))
model.add(Dropout(rate=0.1))
model.add(Conv1D(filters=9, kernel_size=16, activation='relu', padding='same'))
model.add(MaxPool1D(pool_size=5))
model.add(Dropout(rate=0.1))
model.add(Conv1D(filters=14, kernel_size=16, activation='relu', padding='same'))
model.add(MaxPool1D(pool_size=5))
model.add(Dropout(rate=0.1))
model.add(Conv1D(filters=21, kernel_size=16, activation='relu', padding='same'))
model.add(MaxPool1D(pool_size=5))
model.add(Dropout(rate=0.1))
model.add(Conv1D(filters=31, kernel_size=16, activation='relu', padding='same'))
model.add(MaxPool1D(pool_size=5))
model.add(Dropout(rate=0.1))
model.add(Conv1D(filters=46, kernel_size=16, activation='relu', padding='same'))
model.add(MaxPool1D(pool_size=5))
model.add(Dropout(rate=0.1))
model.add(Flatten())
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=n_classes, activation='softmax'))

# Complie model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
model.summary()




from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

get_ipython().run_line_magic('matplotlib', 'inline')

SVG(model_to_dot(model, show_shapes=True).create(prog='dot',format='svg'))




from sklearn.preprocessing import LabelEncoder
from scipy.io import wavfile
#fname, label, verified = train.sample(1).value[0]
#train.sample(1)
# train.sample(1).values[0]




# Map files to label
file_label_dict = {fname:label for fname, label in train[['fname', 'label']].values}

example_file = '6a446a35.wav'
print('File Label "{}":\n{}'.format(example_file, file_label_dict[example_file]))

# Create vector encoded labels
LabelEncoder = {}
for i, label in enumerate(train['label'].unique()): #리스트에서 유일한 값 찾기
    label_array = np.zeros(n_classes)
    label_array[i] = 1
    LabelEncoder[label] = label_array
    
example_label = 'Cello'
print('\nEncoded Label "{}":\n{}'.format(example_label, LabelEncoder[example_label]))

# Remap predictions to label
prediction_to_label = {np.argmax(array):label for label, array in LabelEncoder.items()}




# Define batch generator to yield random data batches
def batchGenerator(files, batch_size):
    # Generate infinite random batches
    while True:
        # Get random files
        batch_files = np.random.choice(files, batch_size, replace = False)
        
        # Get labels and data
        batch_label = []
        batch_data = []
        # Combine batch
        for file in batch_files:
            # Get label and data
            label = file_label_dict[file]
            rate, data = wavfile.read(train+file) # 알라라
            # Trim data to get uniform length
            data_uniform_length = np.zeros(input_length)
            minimum = min(input_length, data.shape[0])
            data_uniform_length[:minimum] = data[:minimum]
            # Encode label
            encoded_label = labelEncoder[label]
            # Create label and data batch
            batch_label.append(encoded_label)
            batch_data = np.array(batch_data).reshape(-1, input_length, 1)
            
            # Batch normalization
            minimum, maximum = batch_data.min().astype(float), batch_data.max().astype(float)
            batch_data = (batch_data - minimum) / (maximum - minimum)
            
            # Yield batches for training
            yield batch_data, batch_label




import warnings
warnings.filterwarnings('ignore')




# Create random maxk to split files in train and validation set
train_val_split_mask = np.zeros(train.shape[0], dtype=bool)
train_val_split_mask[:8500] = True
np.random.shuffle(train_val_split_mask)

# Get train and validation files
train_files = train['fname'][train_val_split_mask] #stratifiedkfold cross val 로 했는데 이렇게 써도 되나요?
val_files = train['fname'][~train_val_split_mask]

# Specify train and validation generators
batch_size = 50
train_generator = batchGenerator(train_files, batch_size=batch_size)
val_generator = batchGenerator(val_files, batch_size=50)




import warnings
warnings.filterwarnings('ignore')
history = model.fit_generator(generator=train_generator, 
                              validation_data=val_generator, 
                              validation_steps=10, 
                              use_multiprocessing=True, 
                              epochs=10, 
                              steps_per_epoch=train.shape[0]//batch_size)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

