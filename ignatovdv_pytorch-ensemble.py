#!/usr/bin/env python
# coding: utf-8



import time
import torch
import torch.nn as nn
import torch.optim as optim

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm_notebook

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')




dtypes = {
        'event_count' : 'uint16',
        'event_code' : 'uint16',
        'game_time' : 'uint32',
        'title' : 'category',
        'type' : 'category',
        'world' : 'category',
        'event_id' : 'category',
        'game_session' : 'category',
        'installation_id' : 'category'        
        }




df_train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv', dtype = dtypes )
df_test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv', dtype = dtypes )
spec = pd.read_csv( '/kaggle/input/data-science-bowl-2019/specs.csv', usecols = ['event_id'] )




def lower_str_columns(df):
  col = ['title', 'type', 'world']

  for col in col:
    df[col] = df[col].str.lower().astype('category')




lower_str_columns(df_train)
lower_str_columns(df_test)




def one_hot_encoding(df, spec = spec):
  
    print('Add features...')
    
    df['game_time'] = df['game_time'] / 1000
    df['game_time'] = df['game_time'].astype('uint32')
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df['incorrect_game_attempt'] = np.where( (df['event_data'].str.contains('"correct":false')                                           & (df['type']=='game')), 1, 0 ).astype('uint16')
    
    df['correct_game_attempt'] = np.where( (df['event_data'].str.contains('"correct":true')                                         & (df['type']=='game')), 1, 0 ).astype('uint16')
    
    df['day'] = df['timestamp'].dt.day_name().str.lower().astype('category')

    df['phase_of_day'] = np.where( df['timestamp'].dt.hour.isin(range(6, 11)), 'morning', np.where( df['timestamp'].dt.hour.isin(range(11, 17)), 'day', np.where( df['timestamp'].dt.hour.isin(range(17, 23)), 'evening', 'night' ) ) )

    df['phase_of_day'] = df['phase_of_day'].astype('category')  

    print('One hot encoding 1 out of 6')
    
    df_world = pd.pivot_table(data = df.loc[ : , ['installation_id', 'game_session', 'world'] ].drop_duplicates(),                             index = ['installation_id','game_session'],                             columns = ['world'],                             aggfunc = len,                             fill_value = 0).add_prefix('world_').reset_index()
    
    col = df_world.select_dtypes('int64').columns.tolist()
    df_world[col] = df_world[col].astype('uint8')  
 
    print('One hot encoding 2 out of 6')
    
    df_type = pd.pivot_table(data = df.loc[ : , ['installation_id', 'game_session', 'type']].drop_duplicates(),                          index = ['installation_id','game_session'],                           columns = ['type'],                           fill_value = 0,                           aggfunc = len).add_prefix('type_').reset_index()

    col = df_type.select_dtypes('int64').columns.tolist()
    df_type[col] = df_type[col].astype('uint8')

    sparse_matrix = pd.merge(df_world, df_type, on = ['installation_id','game_session'], how = 'right')

    print('One hot encoding 3 out of 6')
    
    df_title = pd.pivot_table(data = df.loc[:, ['installation_id', 'game_session', 'title'] ].drop_duplicates(),                             index = ['installation_id','game_session'],                             columns = ['title'],                             fill_value = 0,                             aggfunc = len).add_prefix('title_').reset_index()

    col = df_title.select_dtypes('int64').columns.tolist()
    df_title[col] = df_title[col].astype('uint8')


    sparse_matrix = pd.merge(sparse_matrix, df_title, on = ['installation_id','game_session'], how = 'right')

    print('One hot encoding 4 out of 6')
    
    df_days = pd.pivot_table(data = df.loc[: , ['installation_id', 'game_session', 'day'] ].drop_duplicates(),                           index = ['installation_id','game_session'],                           columns = ['day'],                           fill_value = 0,                           aggfunc = len).add_prefix('day_').reset_index()

    col = df_days.select_dtypes('int64').columns.tolist()
    df_days[col] = df_days[col].astype('uint8')

    sparse_matrix = pd.merge(sparse_matrix, df_days, on = ['installation_id','game_session'], how = 'right')

    print('One hot encoding 5 out of 6')
    
    df_time = pd.pivot_table(data = df.loc[ : , ['installation_id', 'game_session', 'phase_of_day'] ].drop_duplicates(),                           index = ['installation_id', 'game_session'],                           columns = ['phase_of_day'],                           fill_value = 0,                           aggfunc = len).add_prefix('phase_').reset_index()

    col = df_time.select_dtypes('int64').columns.tolist()
    df_time[col] = df_time[col].astype('uint8')

    sparse_matrix = pd.merge(sparse_matrix, df_time, on = ['installation_id','game_session'], how = 'right')

    print('One hot encoding 6 out of 6')
    
    df_event_code = pd.pivot_table(data = df.loc[ : , ['installation_id', 'game_session', 'event_code'] ],                                 index = ['installation_id', 'game_session'],                                 columns = ['event_code'],                                 fill_value = 0,                                 aggfunc = len).add_prefix('code_').reset_index()

    col = df_event_code.select_dtypes('int64').columns.tolist()
    df_event_code[col] = df_event_code[col].astype('uint16')

    sparse_matrix = pd.merge(sparse_matrix, df_event_code, on = ['installation_id','game_session'], how = 'right')

    print('Сomputing game attempts...')
    
    col = ['installation_id','game_session']
    df[col] = df[col].astype('object')

    df_game_attempt = df.groupby(['installation_id','game_session'])['incorrect_game_attempt','correct_game_attempt'].sum().reset_index()

    col = df_game_attempt.select_dtypes(['object']).columns.tolist()
    df_game_attempt[col] = df_game_attempt[col].astype('category')
    
    sparse_matrix = pd.merge(sparse_matrix, df_game_attempt, on = ['installation_id','game_session'], how = 'right')
    
    print('Сomputing time session...')
    
    df_gametime = df.groupby(['installation_id','game_session'])['game_time','timestamp','event_count'].max().reset_index()
    
    col = ['installation_id','game_session']
    df[col] = df[col].astype('category')

    col = df_gametime.select_dtypes(['object']).columns.tolist()
    df_gametime[col] = df_gametime[col].astype('category')

    sparse_matrix = pd.merge(sparse_matrix, df_gametime, on = ['installation_id','game_session'], how = 'right')

    print('One hot encoding of event id...')

    spec['encode_event_id'] = np.arange(len(spec))

    z = dict( zip ( spec['event_id'], spec['encode_event_id'] ) )

    df['event_id'] = df['event_id'].map(z)

    df_event_id = pd.pivot_table(data = df.loc[:, ['installation_id','game_session','event_id']],                               aggfunc = len,                               columns = ['event_id'],                               index = ['installation_id','game_session'],                               fill_value = 0).add_prefix('event_id_').reset_index()

    col = df_event_id.select_dtypes('int64').columns.tolist()
    df_event_id[col] = df_event_id[col].astype('uint16')
    
    sparse_matrix = pd.merge(sparse_matrix, df_event_id, on = ['installation_id','game_session'], how = 'left')

    return sparse_matrix




get_ipython().run_cell_magic('time', '', 'sparse_matrix = one_hot_encoding(df_train, spec = spec)')




get_ipython().run_cell_magic('time', '', 'sparse_matrix_test = one_hot_encoding(df_test, spec = spec)')




def del_missing_columns(sparse_matrix, sparse_matrix_test):
  
  no_columns = set(sparse_matrix.columns.values) - set(sparse_matrix_test.columns.values)
  sparse_matrix.drop(no_columns, axis='columns', inplace=True)




del_missing_columns(sparse_matrix, sparse_matrix_test)




def calculate_accuracy(df, sparse_matrix):

    df['incorrect_attempt'] = np.where( (df['event_data'].str.contains('"correct":false') )                                                     & ( ( (df['title'] != "bird measurer (assessment)") & (df['event_code']==4100) ) | ( (df['title'] == "bird measurer (assessment)") & (df['event_code']==4110) ) ), 1, 0 ).astype('uint32')

    df['correct_attempt'] = np.where( (df['event_data'].str.contains('"correct":true') )                                                   & ( ( (df['title'] != "bird measurer (assessment)") & (df['event_code']==4100) ) | ( (df['title'] == "bird Measurer (assessment)") & (df['event_code']==4110) ) ), 1, 0).astype('uint32')

    assessment_title = ['bird measurer (assessment)', 'mushroom sorter (assessment)', 'cauldron filler (assessment)', 'chest sorter (assessment)', 'cart balancer (assessment)']

    col = ['installation_id', 'title', 'game_session']
    df[col] = df[col].astype('object')

    df_train_acc = df[ df['title'].isin(assessment_title)]    .groupby(['installation_id','title','game_session'])['incorrect_attempt','correct_attempt']    .sum().reset_index()
    
    col = ['installation_id','title','game_session']
    df[col] = df[col].astype('category')
    
    col = df_train_acc.select_dtypes(['object']).columns.tolist()
    df_train_acc[col] = df_train_acc[col].astype('category')
    
    df_train_acc['total_attempts'] = df_train_acc.apply(lambda x: x['incorrect_attempt'] + x['correct_attempt'], axis=1).astype('uint32')

    df_train_acc['accuracy'] = np.where(df_train_acc['total_attempts'] > 0, np.around( (df_train_acc['correct_attempt'] / df_train_acc['total_attempts']), 1), 0).astype('float16')

    df_train_acc['accuracy_group'] = np.where(df_train_acc['accuracy']==1, 3, np.where(df_train_acc['accuracy']==.5, 2, np.where(df_train_acc['accuracy']==0, 0, 1))).astype('uint8')

    df_final = pd.merge(df_train_acc, sparse_matrix, on = ['installation_id','game_session'], how = 'right' )

    col = df_final.select_dtypes('category').columns.values.tolist()
    df_final[col] = df_final[col].astype('object')
    
    df_final = df_final.fillna(value=0)

    convert_dict = { 'incorrect_attempt': 'uint32', 'correct_attempt': 'uint32', 'total_attempts': 'uint32', 'accuracy_group': 'uint8' }
    df_final = df_final.astype(convert_dict)
    
    col = df_final.select_dtypes('object').columns.values.tolist()
    df_final[col] = df_final[col].astype('category')
    
    del df_final['title']

    return df_final




get_ipython().run_cell_magic('time', '', 'df_count_acc_train = calculate_accuracy(df_train, sparse_matrix)')




get_ipython().run_cell_magic('time', '', 'df_count_acc_test = calculate_accuracy(df_test, sparse_matrix_test)')




def dataset_history(df, is_train = True):
  
    id_session_attempt = df[df['total_attempts'] != 0]['game_session'].unique()

    df = df.sort_values(['installation_id', 'timestamp'])

    col_all = list(df.columns)

    df['num_session'] = np.ones(len(df)).astype('uint32')

    col = df.select_dtypes(['category']).columns.tolist()
    df[col] = df[col].astype('object')

    col_uin8 = df.select_dtypes(['uint8']).columns.tolist()
    col_uin16 = df.select_dtypes(['uint16']).columns.tolist()
    col_uin32 = df.select_dtypes(['uint32']).columns.tolist()
    col_encode = col_uin8 + col_uin16 + col_uin32
    df[col_encode] = df[col_encode].astype('int32')

    print('Rolling sum num_session...')

    num_session_groups = df.groupby('installation_id') 

    num_session = num_session_groups.rolling(len(df), on = 'game_session', min_periods = 0)['num_session'].sum().astype('uint32').reset_index()

    col = df.select_dtypes(['object']).columns.tolist()
    df[col] = df[col].astype('category')

    df[col_uin8] = df[col_uin8].astype('uint8')
    df[col_uin16] = df[col_uin16].astype('uint16')
    df[col_uin32] = df[col_uin32].astype('uint32')

    num_session['installation_id'] = num_session['installation_id'].astype('category')

    df = pd.merge(df.loc[:, col_all], num_session, on = ['installation_id', 'game_session'], how = 'outer')

    '''Сохраняем опыт предыдущих сессий'''

    col_sum = list(df.select_dtypes(['float16', 'uint8', 'uint16', 'uint32']).columns)
    no_history = ['num_session', 'accuracy', 'accuracy_group']
    for column in no_history:
        col_sum.remove(column)

    col = df.select_dtypes(['category']).columns.tolist()
    df[col] = df[col].astype('object')

    df[col_encode] = df[col_encode].astype('int32')
    
    print('Rolling sum history...')

    rolling_sum_group = df.groupby('installation_id') 


    rolling_sum = rolling_sum_group.rolling(len(df), on = 'num_session', min_periods = 0)[col_sum].sum().astype('uint32').reset_index()
    
    rolling_sum['installation_id'] = rolling_sum['installation_id'].astype('category')
    rolling_sum['num_session'] = rolling_sum['num_session'].astype('uint32')
    
    df = pd.merge(df.loc[:, ['installation_id', 'timestamp', 'game_session', 'num_session', 'accuracy', 'accuracy_group'] ], rolling_sum, on = ['installation_id', 'num_session'], how = 'right')


    if is_train:

        df = df[df['game_session'].isin(id_session_attempt)]

        df.drop(['timestamp', 'installation_id', 'game_session', 'incorrect_attempt', 'correct_attempt', 'total_attempts', 'accuracy'], axis='columns', inplace=True)

    else: 

        df['installation_id'] = df['installation_id'].astype('object')
        
        df = df.sort_values(['installation_id','timestamp']).groupby(['installation_id'], as_index=False).last()

        df['installation_id'] = df['installation_id'].astype('category')
        
        df.drop(['timestamp', 'game_session', 'incorrect_attempt', 'correct_attempt', 'total_attempts', 'accuracy'], axis='columns', inplace=True)

        print('Есть значения Nan?:', df.isnull().values.any())

    return df




get_ipython().run_cell_magic('time', '', 'df_final = dataset_history(df_count_acc_train, is_train = True)')




get_ipython().run_cell_magic('time', '', 'df_final_test = dataset_history(df_count_acc_test, is_train = False)')




def dataset(data, is_train = True):
  
  if is_train:
    X = data.loc[:, data.columns != 'accuracy_group']
  else:
    X = data.loc[:, ((data.columns != 'accuracy_group') & (data.columns != 'installation_id') ) ]
  
  X = normalize(X, axis=0)
  y = data['accuracy_group'].values 

  if is_train:
  
    train_stack = np.hstack((X, y[:, np.newaxis]))

    train, val = train_test_split(train_stack, test_size = 0.25, stratify = train_stack[:, -1], random_state = 42)

    X_train = torch.FloatTensor(train[:, :-1])
    y_train = torch.LongTensor(train[:, -1])
    X_val = torch.FloatTensor(val[:, :-1])
    y_val = torch.LongTensor(val[:, -1])

    data = {'train': X_train, 'val': X_val}
    labels = {'train': y_train, 'val': y_val}
  
  else:
    data = torch.FloatTensor(X)
    labels = torch.LongTensor(y)
  
  return data, labels




def batch_generator(X, y, batch_size, shuffle = True):
    
  if shuffle:
    np.random.seed(42)
    perm = np.random.permutation(len(X))
    
    for j in range(0, len(X), batch_size):
      idx = perm[j : j + batch_size]
      yield X[idx], y[idx]

  else:
    
    for j in range(0, len(X), batch_size):
      yield X[j : j + batch_size], y[j : j + batch_size]




def initialize_model(model_name, num_classes, num_features):

    model = None
    input_size = 0
    torch.manual_seed(42)
    np.random.seed(42)

    if model_name == 'fc3':

        """ fc: 3, layer_1: 475, layer_2: 238
        """
        
        D_in, H1, H2, D_out  = num_features, 475, 238, num_classes

        model = nn.Sequential( 
                              nn.Linear(D_in, H1),
                              nn.BatchNorm1d(H1),
                              nn.ReLU(),
                              nn.Linear(H1, H2),
                              nn.BatchNorm1d(H2),
                              nn.ReLU(),
                              nn.Linear(H2, D_out),
                             )

    elif model_name == 'fc3do':

        """ fc: 3, layer_1: 952, layer_2: 476, dropout: 0.5
        """
        
        D_in, H1, H2, D_out  = num_features, 952, 476, num_classes

        model = nn.Sequential( 
                              nn.Linear(D_in, H1),
                              nn.BatchNorm1d(H1),
                              nn.ReLU(),
                              nn.Dropout(p = 0.5),
                              nn.Linear(H1, H2),
                              nn.BatchNorm1d(H2),
                              nn.ReLU(),
                              nn.Dropout(p = 0.5),
                              nn.Linear(H2, D_out),
                             )
    
    elif model_name == "fc2":
        """ fc: 2, layer_1: 1024, dropout: 0.5
        """
        
        D_in, H1, D_out  = num_features, 1024, num_classes

        model = nn.Sequential( 
                              nn.Linear(D_in, H1),
                              nn.BatchNorm1d(H1),
                              nn.ReLU(),
                              nn.Dropout(p = 0.5),
                              nn.Linear(H1, D_out),
                             )
    
    elif model_name == 'fc3b':
        """ fc: 3, layer_1: 4096, layer_2: 2048, dropout: 0.5
        """
        
        D_in, H1, H2, D_out  = num_features, 4096, 2048, num_classes

        model = nn.Sequential( 
                              nn.Linear(D_in, H1),
                              nn.BatchNorm1d(H1),
                              nn.ReLU(),
                              nn.Dropout(p = 0.5),
                              nn.Linear(H1, H2),
                              nn.BatchNorm1d(H2),
                              nn.ReLU(),
                              nn.Dropout(p = 0.5),
                              nn.Linear(H2, D_out),
                             )

    else:
        print("Invalid model name, exiting...")
        exit()

    return model




def train_model(model, data, labels, criterion, optimizer, shuffle = True, num_epochs = 25, batch_size = 100):
    since = time.time()

    val_acc_history = []
    val_loss_history = []
    train_acc_history = []
    train_loss_history = []
    lr_find_lr = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in tqdm_notebook(range(num_epochs)):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()   

            running_loss = 0.0
            running_corrects = 0

            for inputs, label in batch_generator(data[phase], labels[phase], batch_size, shuffle):
                inputs = inputs
                label = label

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                  outputs = model(inputs)
                  loss = criterion(outputs, label)

                  _, preds = torch.max(outputs, 1)

                if phase == 'train':  
                  loss.backward()
                  optimizer.step()
                  scheduler.step()
                  lr_step = optimizer.state_dict()["param_groups"][0]["lr"]
                  lr_find_lr.append(lr_step)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == label.data)

            epoch_loss = running_loss / len(data[phase])
            epoch_acc = running_corrects.double() / len(data[phase])

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
            else:
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    history_val = {'loss': val_loss_history, 'acc': val_acc_history}
    history_train = {'loss': train_loss_history, 'acc': train_acc_history}

    model.load_state_dict(best_model_wts)

    return model, history_val, history_train, time_elapsed, lr_find_lr 




def visualization(train, val, is_loss = True):
  
  if is_loss:
    plt.figure(figsize=(17,10))
    plt.plot(train, label = 'Training loss')
    plt.plot(val, label = 'Val loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
  
  else:
    plt.figure(figsize=(17,10))
    plt.plot(train, label = 'Training acc')
    plt.plot(val, label = 'Val acc')
    plt.title('Training and validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()
    plt.show()




data, labels = dataset(df_final, is_train = True)
data_test, labels_test = dataset(df_final_test, is_train = False)




model_name = 'fc3'
num_classes = 4
num_features = 475
batch_size = 512




model = initialize_model(model_name, num_classes, num_features)
print(model)




base_lr = 0.00001
max_lr = 0.003

num_epochs = 90

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.000013, momentum=0.9, nesterov = True)

step_size = 2 * math.ceil( len(data['train']) / batch_size )

scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr = base_lr, max_lr = max_lr, step_size_up=step_size, mode='exp_range', gamma=0.9994, scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)




val_loss = []
val_acc = []
train_loss = []
train_acc = []
lr_cycle = []




fc3, history_val, history_train, time_elapsed, lr_find_lr = train_model(model, data, labels, criterion, optimizer, shuffle = True, num_epochs = num_epochs, batch_size = batch_size)

val_loss += history_val['loss']
val_acc += history_val['acc']
train_loss += history_train['loss']
train_acc += history_train['acc']
lr_cycle += lr_find_lr




plt.figure(figsize=(17,10))
plt.plot(lr_cycle);




visualization(train_acc, val_acc, is_loss = False)




visualization(train_loss, val_loss, is_loss = True)




model_name = 'fc3do'
num_classes = 4
num_features = 475
batch_size = 512




model = initialize_model(model_name, num_classes, num_features)
print(model)




base_lr = 0.0001
max_lr = 0.009

num_epochs = 72

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.000013, momentum=0.9, nesterov = True)

step_size = 4 * math.ceil( len(data['train']) / batch_size )

scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr = base_lr, max_lr = max_lr, step_size_up=step_size, mode='exp_range', gamma=0.9999, scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)




val_loss = []
val_acc = []
train_loss = []
train_acc = []
lr_cycle = []




fc3do, history_val, history_train, time_elapsed, lr_find_lr = train_model(model, data, labels, criterion, optimizer, shuffle = True, num_epochs = num_epochs, batch_size = batch_size)

val_loss += history_val['loss']
val_acc += history_val['acc']
train_loss += history_train['loss']
train_acc += history_train['acc']
lr_cycle += lr_find_lr




plt.figure(figsize=(17,10))
plt.plot(lr_cycle);




visualization(train_acc, val_acc, is_loss = False)




visualization(train_loss, val_loss, is_loss = True)




model_name = 'fc2'
num_classes = 4
num_features = 475
batch_size = 512




model = initialize_model(model_name, num_classes, num_features)
print(model)




base_lr = 0.00025
max_lr = 0.018

num_epochs = 90

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.000013, momentum=0.9, nesterov = True)

step_size = 3 * math.ceil( len(data['train']) / batch_size )

scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr = base_lr, max_lr = max_lr, step_size_up=step_size, mode='exp_range', gamma=0.9993, scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)




val_loss = []
val_acc = []
train_loss = []
train_acc = []
lr_cycle = []




fc2, history_val, history_train, time_elapsed, lr_find_lr = train_model(model, data, labels, criterion, optimizer, shuffle = True, num_epochs = num_epochs, batch_size = batch_size)

val_loss += history_val['loss']
val_acc += history_val['acc']
train_loss += history_train['loss']
train_acc += history_train['acc']
lr_cycle += lr_find_lr




plt.figure(figsize=(17,10))
plt.plot(lr_cycle);




visualization(train_acc, val_acc, is_loss = False)




visualization(train_loss, val_loss, is_loss = True)




model_name = 'fc3b'
num_classes = 4
num_features = 475
batch_size = 512




model = initialize_model(model_name, num_classes, num_features)
print(model)




base_lr = 0.0004
max_lr = 0.025

num_epochs = 10

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.000013, momentum=0.9, nesterov = True)

step_size = 2 * math.ceil( len(data['train']) / batch_size )

scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr = base_lr, max_lr = max_lr, step_size_up=step_size, mode='exp_range', gamma=0.999, scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)




val_loss = []
val_acc = []
train_loss = []
train_acc = []
lr_cycle = []




fc3b, history_val, history_train, time_elapsed, lr_find_lr = train_model(model, data, labels, criterion, optimizer, shuffle = True, num_epochs = num_epochs, batch_size = batch_size)

val_loss += history_val['loss']
val_acc += history_val['acc']
train_loss += history_train['loss']
train_acc += history_train['acc']
lr_cycle += lr_find_lr




plt.figure(figsize=(17,10))
plt.plot(lr_cycle);




visualization(train_acc, val_acc, is_loss = False)




visualization(train_loss, val_loss, is_loss = True)




def predict(model, data, labels, shuffle = False):
    with torch.no_grad():
        logits = []
    
        for inputs, label in batch_generator(data, labels, batch_size, shuffle):
            inputs = inputs
            model.eval()
            outputs = model(inputs).cpu()
            logits.append(outputs)
            
    probs = nn.functional.softmax(torch.cat(logits), dim=1).numpy()
    return probs




fc3_pred = predict(fc3, data_test, labels_test, shuffle = False)




fc3do_pred = predict(fc3do, data_test, labels_test, shuffle = False)




fc2_pred = predict(fc2, data_test, labels_test, shuffle = False)




fc3b_pred = predict(fc3b, data_test, labels_test, shuffle = False)




fc3_class = np.argmax(fc3_pred, axis=1) 




fc3do_class = np.argmax(fc3do_pred, axis=1) 




fc2_class = np.argmax(fc2_pred, axis=1) 




fc3b_class = np.argmax(fc3b_pred, axis=1) 




matrix_class = np.hstack(( fc3_class[:, np.newaxis], fc3do_class[:, np.newaxis], fc2_class[:, np.newaxis], fc3b_class[:, np.newaxis] ))




matrix_class




fc3_prob = np.max(fc3_pred, axis=1)




fc3do_prob = np.max(fc3do_pred, axis=1)




fc2_prob = np.max(fc2_pred, axis=1)




fc3b_prob = np.max(fc3b_pred, axis=1)




matrix_prob = np.hstack((fc3_prob[:, np.newaxis], fc3do_prob[:, np.newaxis], fc2_prob[:, np.newaxis], fc3b_prob[:, np.newaxis] ))




matrix_prob




vector_pred = []

for i in range(matrix_class.shape[0]):
  candidates, count_vote = np.unique(matrix_class[i], return_counts=True, axis=0)
  candid_numvote = np.hstack((candidates[:, np.newaxis], count_vote[:, np.newaxis]))
  if len(np.unique(count_vote)) == 1:
    ind = np.argmax(matrix_prob[i])
    vector_pred.append(int(matrix_class[i][ind]))
  else:
    max_num_vote = np.max(candid_numvote[:, 1])
    candit_max_vote = candid_numvote[candid_numvote[:, 1] == max_num_vote]
    if len(candit_max_vote[:, 0]) == 1:
      vector_pred.append( int(candit_max_vote[:, 0]) )
    else: 
      indx = [ np.where(matrix_class[0] == candit_max_vote[:, 0][i]) for i in range(len(candit_max_vote[:, 0])) ]
      prob_cand = np.array( [ matrix_prob[i][indx[i]].sum() for i in range(len(indx)) ] )
      matrix_choise = np.hstack((candit_max_vote, prob_cand[:, np.newaxis]))
      if len(np.unique(matrix_choise[:, 2])) == 1:
        vector_pred.append( int(matrix_choise[:, 0][0]) )
      else:
        indx_max_prob = np.argmax(matrix_choise[:, 2])
        vector_pred.append( int(matrix_choise[:, 0][indx_max_prob]) )




submission = pd.DataFrame({'installation_id': df_final_test['installation_id'].values, 'accuracy_group': vector_pred})
submission.to_csv('submission.csv', index = False)




submission['accuracy_group'].value_counts()




submission
















