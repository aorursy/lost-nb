#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import matplotlib.pyplot as plt
import json
import datetime as dt
import seaborn as sns
import cv2
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.




# Keras Package Load
import tensorflow as tf
import keras
from keras.layers import Conv2D, GlobalAveragePooling2D, MaxPooling2D, GlobalAveragePooling1D, Conv1D, MaxPooling1D
from keras.layers import Dense, Dropout, Flatten, Activation, Input, BatchNormalization
from keras.layers import CuDNNLSTM as LSTM
from keras.layers.merge import concatenate
from keras.metrics import categorical_crossentropy, top_k_categorical_accuracy, categorical_accuracy
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,TensorBoard, LearningRateScheduler
from keras.optimizers import Adam,Adagrad,RMSprop,SGD
from keras.applications import Xception, MobileNet, MobileNetV2, xception, mobilenet


from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.metrics import top_k_categorical_accuracy

#supplement package load
from tqdm import tqdm_notebook as tqdm
from ast import literal_eval
import glob
from multiprocessing import Pool
from functools import partial
from itertools import repeat
from itertools import product




SH_DIR = '../input/shufflecsvs/shuffle-csvs/shuffle-csvs/'
INPUT_DIR = '../input/quickdraw-doodle-recognition/'




SEED = 42
np.random.seed(seed=SEED)
tf.set_random_seed(seed=SEED)




def apk(actual, predicted, k=3):
    """
    Source: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    """
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    if not actual:
        return 0.0
    return score / min(len(actual), k)

def mapk(actual, predicted, k=3):
    """
    Source: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

def preds2catids(predictions):
    return pd.DataFrame(np.argsort(-predictions, axis=1)[:, :3], columns=['a', 'b', 'c'])

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)




def f2cat(filename: str) -> str:
    return filename.split('.')[0]

def list_all_categories():
    files = os.listdir(os.path.join(INPUT_DIR, 'train_simplified'))
    return sorted([f2cat(f) for f in files], key=str.lower)




mobile_BASE_SIZE = 256
mobile_NCSVS = 100
mobile_NCATS = 340

mobile_STEPS = 800
mobile_EPOCHS = 70
mobile_SIZE = 64
mobile_BATCHSIZE = 128
# drop_rate = 0.5


lstm_batch_size = 128
lstm_STROKE_COUNT = 196

BATCHSIZE = 128




def draw_cv2(raw_strokes, size=256, lw=6, time_color=True):
    img = np.zeros((mobile_BASE_SIZE, mobile_BASE_SIZE), np.uint8)
    for t, stroke in enumerate(raw_strokes):
        for i in range(len(stroke[0]) - 1):
            color = 255 - min(t, 10) * 13 if time_color else 255
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
    if size != mobile_BASE_SIZE:
        return cv2.resize(img, (size, size))
    else:
        return img




def image_generator_xd(size, batchsize, ks, lw=6, time_color=True):
    while True:
        for k in np.random.permutation(ks):
            filename = os.path.join(SH_DIR, 'train_k{}.csv.gz'.format(k))
            for df in pd.read_csv(filename, chunksize=batchsize):
                df['drawing'] = df['drawing'].apply(json.loads)
                x = np.zeros((len(df), size, size, 1))
                for i, raw_strokes in enumerate(df.drawing.values):
                    x[i, :, :, 0] = draw_cv2(raw_strokes, size=size, lw=lw,
                                             time_color=time_color)
                x = mobilenet.preprocess_input(x).astype(np.float32)
                y = keras.utils.to_categorical(df.y, num_classes=NCATS)
                yield x, y




def df_to_image_array_xd(df, size, lw=6, time_color=True):
    df['drawing'] = df['drawing'].apply(json.loads)
    x = np.zeros((len(df), size, size, 1))
    for i, raw_strokes in enumerate(df.drawing.values):
        x[i, :, :, 0] = draw_cv2(raw_strokes, size=size, lw=lw, time_color=time_color)
    x = mobilenet.preprocess_input(x).astype(np.float32)
    return x




def df_to_sequence_array(df, time_size):
    df['drawing'] = df['drawing'].apply(json.loads)
    x = np.zeros((len(df), time_size, 3))
    for i, raw_strokes in enumerate(df.drawing.values):
        x[i, : ,: ] = _stack_it(raw_strokes, time_size)
    return x




def _stack_it(raw_strokes, time_size):
    """preprocess the string and make 
    a standard Nx3 stroke vector"""
    
#     stroke_vec = literal_eval(raw_strokes) # string->list
    
    # unwrap the list
    in_strokes = [(xi,yi,i)  
     for i,(x,y) in enumerate(raw_strokes) 
     for xi,yi in zip(x,y)]
    c_strokes = np.stack(in_strokes)
    # replace stroke id with 1 for continue, 2 for new
    c_strokes[:,2] = [1]+np.diff(c_strokes[:,2]).tolist()
    c_strokes[:,2] += 1 # since 0 is no stroke
    # pad the strokes with zeros
    return pad_sequences(c_strokes.swapaxes(0, 1), 
                         maxlen=time_size, 
                         padding='post').swapaxes(0, 1)




def read_batch(samples=5, 
               start_row=0,
               max_rows = 1000):
    """
    load and process the csv files
    this function is horribly inefficient but simple
    """
    out_df_list = []
    for c_path in ALL_TRAIN_PATHS:
        c_df = pd.read_csv(c_path, nrows=max_rows, skiprows=start_row)
        c_df.columns=COL_NAMES
        out_df_list += [c_df.sample(samples)[['drawing', 'word']]]
    full_df = pd.concat(out_df_list)
    full_df['drawing'] = full_df['drawing'].        map(_stack_it)
    
    return full_df




def image_generator_xd(size,time_size, batchsize, ks, lw=6, time_color=True):
    while True:
        for k in np.random.permutation(ks):
            filename = os.path.join(SH_DIR, 'train_k{}.csv.gz'.format(k))
            for df in pd.read_csv(filename, chunksize=batchsize):
                df['drawing'] = df['drawing'].apply(json.loads)
                x = np.zeros((len(df), size, size, 1))
                x_lstm = np.zeros((len(df), time_size, 3))
                for i, raw_strokes in enumerate(df.drawing.values):
#                     print(raw_strokes)
                    x[i, :, :, 0] = draw_cv2(raw_strokes, size=size, lw=lw,
                                             time_color=time_color)
#                     print(_stack_it(raw_strokes, time_size))
                    x_lstm[i, :, :] = _stack_it(raw_strokes, time_size)
                x = mobilenet.preprocess_input(x).astype(np.float32)
                y = keras.utils.to_categorical(df.y, num_classes=mobile_NCATS)
                yield [x, x_lstm],  y




train_datagen = image_generator_xd(size = mobile_SIZE, time_size=lstm_STROKE_COUNT, batchsize=mobile_BASE_SIZE, ks= range(mobile_NCSVS-1))
x, y =next(train_datagen)




print("CNN shape{}       \nLSTM shape {}".format(x[0].shape,x[1].shape))




valid_df = pd.read_csv(os.path.join(SH_DIR, 'train_k{}.csv.gz'.format(mobile_NCSVS-1)), nrows = 34000)




iterable_1 = product(np.array_split(valid_df, 100), [mobile_SIZE])
iterable_2 = product(np.array_split(valid_df,  100), [lstm_STROKE_COUNT])




# Virtual Core count
NJOBS = get_ipython().getoutput('grep -c processor /proc/cpuinfo')
NJOBS = int(NJOBS[0])
print(NJOBS)




with Pool(processes=NJOBS) as p:
    x_valid  = p.starmap(df_to_image_array_xd, iterable_1)
#     x_lstm_valid = p.starmap(_stack_it, iterable_2)


with Pool(processes=NJOBS) as p:
    x_lstm_valid = p.starmap(df_to_sequence_array, iterable_2)




x_valid = np.vstack(np.array(x_valid))
x_lstm_valid = np.vstack(np.array(x_lstm_valid))
y_valid = keras.utils.to_categorical(valid_df.y, num_classes=mobile_NCATS)

print(
    'mobile_valid size :{} \
    \nRNN valid size : {} \
    \ny_valid size : {}'.format(x_valid.shape, x_lstm_valid.shape,  y_valid.shape))




def mobiel_net(size):
    base_model = MobileNet(weights = None, input_shape=(size, size,1), include_top=False ,alpha=1.)
    x= base_model.output
    x= GlobalAveragePooling2D()(x)
#     model = Model(inputs = base_model.input, outputs = base_model.outputs, name = 'mobile_net')
    
    return x, base_model.input




def lstm_model(time_size ,f_size ):
    stroke_input = Input(shape=(time_size, f_size), name = 'lstm_input')
#     batch_norm = BatchNormalization()(stroke_input)
#     conv1 = Conv1D(48, (5,), activation='relu')(batch_norm)
#     drop1 = Dropout(0.3)(conv1)
#     conv2 = Conv1D(64, (5,), activation='relu')(drop1)
#     drop2 = Dropout(0.3)(conv2)
#     conv3 = Conv1D(96, (3,), activation='relu')(drop2)
#     drop3 = Dropout(0.3)(conv3)
#     lstm1 = LSTM(128, return_sequences=True)(drop3)
#     drop4 = Dropout(0.3)(lstm1)
#     lstm2 = LSTM(128, return_sequences=False)(drop4)
#     return lstm2, stroke_input
    
    batch_norm = BatchNormalization()(stroke_input)
    conv1 = Conv1D(48, (5,), activation='relu')(batch_norm)
#     drop1 = Dropout(0.3)(conv1)
    conv2 = Conv1D(64, (5,), activation='relu')(conv1)
#     drop2 = Dropout(0.3)(conv2)
    conv3 = Conv1D(96, (3,), activation='relu')(conv2)
#     drop3 = Dropout(0.3)(conv3)
    
    """ BATCH norm test _4"""
    max_pool = MaxPooling1D(pool_size  = 4)(conv3)
    """ BATCH norm test _4"""
    
    lstm1 = LSTM(128, return_sequences=True)(max_pool)
#     drop4 = Dropout(0.3)(lstm1)

#     """ BATCH norm test _3"""
#     batch_norm_2 = BatchNormalization()(lstm1)
#     """BATCH norm test _3"""

    lstm2 = LSTM(128, return_sequences=False)(lstm1)
    """ BATCH norm test _4"""
    batch_norm_2 = BatchNormalization()(lstm2)
    """BATCH norm test _4"""
    
    return batch_norm_2, stroke_input




mobile_model, mobile_input = mobiel_net(mobile_SIZE)
sequence_model, sequence_input = lstm_model(lstm_STROKE_COUNT, 3)




print(mobile_model.shape,      sequence_model.shape)




merge = concatenate([mobile_model, sequence_model])




""" BATCH norm test _4"""
hidden_1 = Dense(512, activation='relu')(merge)
merge_batch_norm = BatchNormalization()(hidden_1)
"""BATCH norm test _4"""
# merge_drop1 = Dropout(0.3)(merge)
# merge_hidden1 = Dense(512)(merge_drop1)
# merge_drop2 = Dropout(0.3)(merge_hidden1)
output = Dense(mobile_NCATS, activation = 'softmax')(merge_batch_norm)
merge_model = Model(inputs = [mobile_input, sequence_input ], output = output, name = 'merge_model_fin_70')




merge_model.name




merge_model.summary()




get_ipython().system(' mkdir ../working/weights')
get_ipython().system(' mkdir ../working/logs')




def directory_check(path):
    if not os.path.isdir(path):
        os.mkdir(path)




def compile_and_train(model, num_epochs, BATCHSIZE, OPTIMIZER):
    model.compile(optimizer=OPTIMIZER(lr=0.002), loss='categorical_crossentropy',
              metrics=[categorical_crossentropy, categorical_accuracy, top_3_accuracy])
    directory_path = '../working/weights/{}/'.format(model.name)
    log_path = '../working/logs/{}/'.format(model.name)
    directory_check(directory_path)
    directory_check(log_path)
    
    file_name = model.name + '.{epoch:02d}-{loss:.2f}.h5'
    filepath = os.path.join(directory_path, file_name)
#     filepath = 'weights/' + model.name + '.{epoch:02d}-{loss:.2f}.h5'
    callbacks = [
        ReduceLROnPlateau(monitor='val_top_3_accuracy', factor = 0.75, patience=3, min_delta=0.001, mode='max', min_lr=1e-5, verbose=1),
        ModelCheckpoint(filepath, monitor='val_top_3_accuracy', mode= 'max', save_best_only=True, save_weights_only=True),
        TensorBoard(log_dir=log_path, histogram_freq=0, batch_size=BATCHSIZE)
        
]
#     
    
    hist = model.fit_generator(
        train_datagen, steps_per_epoch = mobile_STEPS, epochs = num_epochs, verbose = 1,
        validation_data = ([x_valid,x_lstm_valid], y_valid), 
        callbacks = callbacks
)
#     weight_files = glob.glob(os.path.join(os.getcwd(), '{}*'.format(directory_path)))
    weight_files = glob.glob('{}*'.format(directory_path))
    weight_file = max(weight_files, key = os.path.getctime)
#     for file in weight_files:
#         if file == weight_file:
#             pass
#         else:
            
#             os.remove(file)
#     hists.append(hist)
    return hist, weight_file




model_his, model_weight = compile_and_train(merge_model, 60 , BATCHSIZE, Adam)




history_df = pd.concat([pd.DataFrame(model_his.history)],sort = True)




fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(16, 10))
axs[0].plot(history_df.val_categorical_accuracy, lw = 3, label ='val_acc')
axs[0].plot(history_df.categorical_accuracy, lw = 3, label ='train_acc')
axs[0].set_ylabel('acc')
axs[0].set_xlabel('epoch')
axs[0].grid()
axs[0].legend(loc=0)

axs[1].plot(history_df.val_categorical_crossentropy, lw = 3, label ='val_loss')
axs[1].plot(history_df.categorical_crossentropy, lw = 3, label ='train_loss')
axs[1].set_ylabel('loss')
axs[1].set_xlabel('epoch')
axs[1].grid()
axs[1].legend(loc=0)
fig.savefig('merge_model_fin_70.png', dpi = 300)
plt.show()




valid_predictions = merge_model.predict([x_valid, x_lstm_valid], batch_size=BATCHSIZE, verbose=1)
map3 = mapk(valid_df[['y']].values, preds2catids(valid_predictions).values)
print('MAP3 : {:.3f}'.format(map3))




test  = pd.read_csv(os.path.join(INPUT_DIR, 'test_simplified.csv'))
test.head()




x_test = df_to_image_array_xd(test.copy(), mobile_SIZE)
x_lstm_test = df_to_sequence_array(test.copy() , lstm_STROKE_COUNT)

print(test.shape, x_test.shape)
print(test.shape, x_lstm_test.shape)




test_predictions = merge_model.predict([x_test,x_lstm_test], batch_size=BATCHSIZE, verbose=1)
top3 = preds2catids(test_predictions)
top3.head()




cats = list_all_categories()
id2cat = {k : cat.replace(' ', '_')for k, cat in enumerate(cats)}
top3cats = top3.replace(id2cat)
top3cats.head()




test['word'] = top3cats['a'] + ' ' + top3cats['b'] + ' ' + top3cats['c']
submission = test[['key_id', 'word']]
submission.to_csv('{}-submission-fin.csv'.format(merge_model.name), index=False)






