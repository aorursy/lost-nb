#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from string import punctuation
import datetime
from keras.layers import Dense,Input,Flatten,concatenate,Dropout,Lambda
from keras.models import Model
import keras.backend as K
import re
from keras.losses import binary_crossentropy
from  keras.callbacks import EarlyStopping,ModelCheckpoint
import codecs

from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from sklearn.metrics import f1_score
from keras import optimizers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from keras.layers import Dense,Input,Flatten,concatenate,Dropout,Lambda,BatchNormalization
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from  keras.callbacks import EarlyStopping,ModelCheckpoint
import datetime
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import time

warnings.filterwarnings("ignore")
pd.set_option('max_columns', None)


# In[2]:


path1 = '/kaggle/input/nfl-big-data-bowl-2020/train.csv'
path2 = '../input/nfl-big-data-bowl-2020/train.csv'
if os.path.exists(path1):
    df = pd.read_csv(path1)
    TRAIN_OFFLINE = False

else:
    df = pd.read_csv(path2)
    TRAIN_OFFLINE = True
df.info()


# In[3]:


plt.figure(figsize=(10, 5))
sns.distplot(df['Yards'], kde=True)
plt.axvline(df['Yards'].mean(), 0, 1, c='r')


# In[4]:


df.head(22)


# In[5]:


def process_unique_features(df, fillna=-999):
    
    def clean_weather(txt):
        ans = 1
        if pd.isna(txt):
            return 0
        if 'partly' in txt:
            ans*=0.5
        if 'climate controlled' in txt or 'indoor' in txt:
            return ans*3
        if 'sunny' in txt or 'sun' in txt:
            return ans*2
        if 'clear' in txt:
            return ans
        if 'cloudy' in txt:
            return -ans
        if 'rain' in txt or 'rainy' in txt:
            return -2*ans
        if 'snow' in txt:
            return -3*ans
        return 0
    
    def orientation_to_cat(x):
        x = np.clip(x, 0, 360 - 1)
        try:
            return str(int(x/15))
        except:
            return "nan"
        
    def strtofloat(x):
        try:
            return float(x)
        except:
            return fillna

    
    def get_num_offense(row):
        splits = row.split(', ')
        cnt = 0
        for s in splits:
            cnt += int(s[0])
        return cnt

    #unique features
    add_new_feas = []
    df = df.copy()
    
    df['Temperature'] = df['Temperature'].fillna(60).astype(np.float)
    df['GameWeather'] = df['GameWeather'].apply(clean_weather).astype(np.float)
    df['TimeHandoff'] = df['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
    
    df["Orientation_ob"] = df["Orientation"].apply(lambda x : orientation_to_cat(x)).astype("object")
    df["Dir_ob"] = df["Dir"].apply(lambda x : orientation_to_cat(x)).astype("object")
    df["Orientation_sin"] = df["Orientation"].apply(lambda x : np.sin(x/360 * 2 * np.pi))
    df["Orientation_cos"] = df["Orientation"].apply(lambda x : np.cos(x/360 * 2 * np.pi))
    df["Dir_sin"] = df["Dir"].apply(lambda x : np.sin(x/360 * 2 * np.pi))
    df["Dir_cos"] = df["Dir"].apply(lambda x : np.cos(x/360 * 2 * np.pi))
    df['Position'] = df['Position'].apply(lambda x: 1 if x in ['RB', 'TE', 'GB', 'WR'] else 0)
    ## diff Score
    df["diffScoreBeforePlay"] = df["HomeScoreBeforePlay"] - df["VisitorScoreBeforePlay"]
    df['PlayerHeight_dense'] = df['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))
    df['PlayerBMI'] = 703*(df['PlayerWeight']/(df['PlayerHeight_dense'])**2)
    df['PlayerBirthDate'] = df['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))
    seconds_in_year = 60*60*24*365
    df['PlayerAge'] = df.apply(lambda row: (row['TimeHandoff']-row['PlayerBirthDate']).total_seconds()/seconds_in_year, axis=1)
    formation_dict = {'ACE': 3, 'EMPTY': 9, 'I_FORM': 4, 'JUMBO': 1, 'PISTOL': 7, 'SHOTGUN': 8, 'SINGLEBACK': 5, 'WILDCAT': 6, 'unknown': 2}
    df['OffenseFormation'] = df['OffenseFormation'].fillna('unknown').map(formation_dict)
    df['kg']=df["PlayerWeight"] * 0.45359237 / 9.8
    # the momentum is just mass (in kg) X speed in m/s (so convert from yards/sec to mps)
    df['True_Momentum']=df['kg'] * df['S'] * 0.9144 
    df['Force_Newtons']=df['kg'] * df['A'] * 0.9144
    df['JerseyNumber'] = df['JerseyNumber'].fillna(fillna)    
    df['DefendersInTheBox'] = df['DefendersInTheBox'].apply(lambda x: 2.0 if x == 1.0 else x)
    # Create the DL-LB combos
    df['DL_LB'] = df['DefensePersonnel']                     .str[:10]                     .str.replace(' DL, ','-')                     .str.replace(' LB','') # Clean up and convert to DL-LB combo
    df['SUB_DL_BL'] = df['DL_LB'].apply(lambda x: int(x.split('-')[0]) - int(x.split('-')[1]))
    df['num_OffensePersonnel'] = df['OffensePersonnel'].apply(get_num_offense)
    
    #df['JerseyNumber'] = df['JerseyNumber'] // 25
    ## WindSpeed
    '''
    df['WindSpeed'] = df['WindSpeed'].astype(str)
    df['WindSpeed_ob'] = df['WindSpeed'].apply(lambda x: x.lower().replace('mph', '').strip() if not pd.isna(x) else x)
    df['WindSpeed_ob'] = df['WindSpeed_ob'].apply(lambda x: (int(x.split('-')[0])+int(x.split('-')[1]))/2 if not pd.isna(x) and '-' in x else x)
    df['WindSpeed_ob'] = df['WindSpeed_ob'].apply(lambda x: (int(x.split()[0])+int(x.split()[-1]))/2 if not pd.isna(x) and type(x)!=float and 'gusts up to' in x else x)
    df['WindSpeed_dense'] = df['WindSpeed_ob'].apply(strtofloat)
    '''
    
    #add_new_feas.append("TimeDelta_snap")
    #add_new_feas.append("TimeDelta")
    #add_new_feas.append("Orientation_ob")
    #add_new_feas.append("Dir_ob")
    add_new_feas.append("Dir_sin")
    add_new_feas.append("Dir_cos")
    #add_new_feas.append("Orientation_sin")
    #add_new_feas.append("Orientation_cos")
    add_new_feas.append("diffScoreBeforePlay")
    add_new_feas.append('PlayerHeight_dense')
    #add_new_feas.append('WindSpeed_dense')
    add_new_feas.append('SUB_DL_BL')
    #add_new_feas.append('num_OffensePersonnel')
    add_new_feas.append('True_Momentum')
    #add_new_feas.append('Force_Newtons')
    #add_new_feas.append('PlayerBMI')
    
    basic_feas =['GameId','PlayId',
                 'X','Y',
                 'S','A',
                 'Dis',
                 #'PlayDirection',
                 'Orientation',
                 'PlayerWeight',
                 'PlayerAge',
                 'Dir',
                 #'Temperature', 'GameWeather',
                 'YardLine',
                 'OffenseFormation',
                 #'Quarter',
                 #'Position',
                 'Down',
                 'Distance',
                 'DefendersInTheBox',
                 'JerseyNumber',
                 ]    
    
    static_features = df[df['NflId'] == df['NflIdRusher']]                        [add_new_feas+basic_feas].drop_duplicates()

    static_features.fillna(fillna,inplace=True)
    return static_features


# In[6]:


def defense_features(df):
    rusher = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','Team','X','Y']]
    rusher.columns = ['GameId','PlayId','RusherTeam','RusherX','RusherY']

    defense = pd.merge(df,rusher,on=['GameId','PlayId'],how='inner')
    defense = defense[defense['Team'] != defense['RusherTeam']][['GameId','PlayId','X','Y','RusherX','RusherY']]
    defense['def_dist_to_back'] = defense[['X','Y','RusherX','RusherY']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)
    
    defense = defense.groupby(['GameId','PlayId'])                     .agg({'def_dist_to_back':['min','max','mean','std']})                     .reset_index()
    defense.columns = ['GameId','PlayId',
                       'def_min_dist','def_max_dist',
                       'def_mean_dist','def_std_dist']
    
    return defense

def features_relative_to_back(df, carriers):
    player_distance = df[['GameId','PlayId','NflId','X','Y']]
    player_distance = pd.merge(player_distance, carriers, on=['GameId','PlayId'], how='inner')
    player_distance = player_distance[player_distance['NflId'] != player_distance['NflIdRusher']]
    player_distance['dist_to_back'] = player_distance[['X','Y','back_X','back_Y']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)

    player_distance = player_distance.groupby(['GameId','PlayId','back_from_scrimmage','back_oriented_down_field','back_moving_down_field'])                                     .agg({'dist_to_back':['min','max','mean','std']})                                     .reset_index()
    player_distance.columns = ['GameId','PlayId','back_from_scrimmage','back_oriented_down_field','back_moving_down_field',
                               'min_dist',
                               'max_dist',
                               'mean_dist','std_dist']

    return player_distance

def update_orientation(df, fix_ori=False, ori_dir_sub=False):
    df['X'] = df[['X','PlayDirection']].apply(lambda x: new_X(x[0],x[1]), axis=1)
    df['Orientation'] = df[['Orientation','PlayDirection']].apply(lambda x: new_orientation(x[0],x[1]), axis=1)
    df['Dir'] = df[['Dir','PlayDirection']].apply(lambda x: new_orientation(x[0],x[1]), axis=1)
    # Add 90 to Orientation for 2017 season only
    if fix_ori:
        df.loc[df['Season'] == 2017, 'Orientation'] = np.mod(90 + df.loc[df['Season'] ==2017, 'Orientation'], 360)
        df.drop(['Season'], axis=1, inplace=True)
    if ori_dir_sub:
        df['Ori_Dir_Sub'] = abs(df['Orientation'] - df['Dir'])
    return df

def back_features(df):
    carriers = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','NflIdRusher','X','Y','Orientation','Dir','YardLine']]
    carriers['back_from_scrimmage'] = carriers['YardLine'] - carriers['X']
    carriers['back_oriented_down_field'] = carriers['Orientation'].apply(lambda x: back_direction(x))
    carriers['back_moving_down_field'] = carriers['Dir'].apply(lambda x: back_direction(x))
    carriers = carriers.rename(columns={'X':'back_X',
                                        'Y':'back_Y'})
    carriers = carriers[['GameId','PlayId','NflIdRusher','back_X','back_Y','back_from_scrimmage','back_oriented_down_field','back_moving_down_field']]

    return carriers

def new_line(rush_team, field_position, yardline):
    if rush_team == field_position:
        # offense starting at X = 0 plus the 10 yard endzone plus the line of scrimmage
        return 10.0 + yardline
    else:
        # half the field plus the yards between midfield and the line of scrimmage
        return 60.0 + (50 - yardline)
    
def new_X(x_coordinate, play_direction):
    if play_direction == 'left':
        return 120.0 - x_coordinate
    else:
        return x_coordinate

def combine_features(relative_to_back, defense, personal, dist=True):
    if dist:
        df = relative_to_back
    else:
        df = pd.merge(relative_to_back,defense,on=['GameId','PlayId'],how='inner')
    df = pd.merge(df,personal,on=['GameId','PlayId'],how='inner')
    new_columns = list(df.columns)
    new_columns.remove('GameId')
    new_columns.remove('PlayId')
    return df, new_columns

def new_orientation(angle, play_direction):
    if play_direction == 'left':
        new_angle = 360.0 - angle
        if new_angle == 360.0:
            new_angle = 0.0
        return new_angle
    else:
        return angle

def euclidean_distance(x1,y1,x2,y2):
    x_diff = (x1-x2)**2
    y_diff = (y1-y2)**2

    return np.sqrt(x_diff + y_diff)

def back_direction(orientation):
    if orientation > 180.0:
        return 1
    else:
        return 0


# In[7]:


def process_player_acc(df):
    rusher = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','Team','A']]
    rusher.columns = ['GameId','PlayId','RusherTeam','RusherA']

    acc = pd.merge(df,rusher,on=['GameId','PlayId'],how='inner')
    #상대팀의  A
    acc = acc[['GameId','PlayId','A','RusherA']]
    #상대 팀과 rusher의 A 차이
    acc['Sub_A'] = acc[['A','RusherA']].apply(lambda x: abs(x[0]-x[1]), axis=1)
    #상태 팀과 rusher의 거리 -> min, max, mean, std
    acc = acc.groupby(['GameId','PlayId'])                     .agg({'Sub_A':['mean']})                     .reset_index()
    acc.columns = ['GameId','PlayId','sub_mean_A']
    
    return acc

def process_yardline(df):
     
    df = df.copy()
    new_yardline = df[df['NflId'] == df['NflIdRusher']]
    new_yardline['YardLine'] = new_yardline[['PossessionTeam','FieldPosition', 'YardLine']]                                                            .apply(lambda x: new_line(x[0],x[1],x[2]), axis=1)
    new_yardline = new_yardline[['GameId','PlayId','YardLine']]
    df = df.drop('YardLine', axis=1)
    df = pd.merge(df, new_yardline, on=['GameId','PlayId'], how='inner')
    return df

def rusher_features(df):
        
    rusher = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','Dir', 'S', 'A', 'X', 'Y']]
    rusher.columns = ['GameId','PlayId', 'RusherDir', 'RusherS', 'RusherA', 'RusherX', 'RusherY']

    radian_angle = (90 - rusher['RusherDir']) * np.pi / 180.0
    v_horizontal = np.abs(rusher['RusherS'] * np.cos(radian_angle))
    v_vertical = np.abs(rusher['RusherS'] * np.sin(radian_angle)) 
    
    rusher['v_horizontal'] = v_horizontal
    rusher['v_vertical'] = v_vertical

    rusher.columns = ['GameId','PlayId', 'RusherDir', 'RusherS','RusherA','RusherX', 'RusherY','v_horizontal', 'v_vertical']

    return rusher

def ori_dir_sub_features(df):
    ori_rusher = df[df['NflId'] == df['NflIdRusher']][['GameId', 'PlayId', 'Ori_Dir_Sub', 'Team']]
    ori_rusher.columns = ['GameId', 'PlayId', 'Rusher_Ori_Dir_Sub', 'RusherTeam']
    ori_df = pd.merge(df, ori_rusher, on=['GameId', 'PlayId'], how='inner')
    ori_away = ori_df[ori_df['RusherTeam'] != ori_df['Team']][['GameId', 'PlayId', 'Ori_Dir_Sub', 'Rusher_Ori_Dir_Sub']]
    ori_away['Ori_Dir_Sub_rusher'] = abs(ori_away['Rusher_Ori_Dir_Sub'] - ori_away['Ori_Dir_Sub'])
    ori_away['Ori_Dir_Sub_rusher'] = ori_away['Ori_Dir_Sub_rusher'].fillna(0).astype(np.int16)
    ori_away = ori_away.groupby(['GameId', 'PlayId'])['Ori_Dir_Sub_rusher'].mean().reset_index()
    ori_away.columns = ['GameId', 'PlayId', 'Ori_Dir_Sub_rusher']
    ori_rusher = pd.merge(ori_rusher, ori_away,
                          on=['GameId', 'PlayId'], how='inner')
    ori_rusher = ori_rusher[['GameId', 'PlayId', 'Ori_Dir_Sub_rusher', 'Rusher_Ori_Dir_Sub']]
    return ori_rusher
 
def radian_features(df):
    rusher = df[df['NflId']==df['NflIdRusher']][['GameId', 'PlayId', 'X', 'Y', 'Team']]
    rusher.columns = ['GameId', 'PlayId', 'rusherX', 'rusherY', 'RusherTeam']
    tmp_df = pd.merge(df, rusher, on=['GameId', 'PlayId'], how='left')
    tmp_df = tmp_df[tmp_df['RusherTeam'] != tmp_df['Team']][['GameId', 'PlayId', 'X', 'Y', 'rusherX', 'rusherY']]
    tmp_df['radian'] = abs(tmp_df['Y'] - tmp_df['rusherY']) / abs(tmp_df['X'] - tmp_df['rusherX'])
    radian_df = tmp_df.groupby(['GameId', 'PlayId']).aggregate({'radian': ['std', 'mean']})
    radian_df.columns = ['radian_std', 'radian_mean']
    radian_df = radian_df.replace([np.inf, -np.inf], np.nan).fillna(-999)
    return radian_df

def std_features(df):
    ##X position test
    tmp = df[df['NflId'] == df['NflIdRusher']][['GameId', 'PlayId', 'X', 'Team']]
    tmp.columns = ['GameId', 'PlayId', 't_rusher_x', 'RusherTeam']
    tmp_merge = pd.merge(df[['GameId', 'PlayId', 'X', 'Team']], tmp, on=['GameId', 'PlayId'], how='inner')
    tmp_merge = tmp_merge[tmp_merge['Team'] != tmp_merge['RusherTeam']]
    tmp_merge['dist_x'] = abs(tmp_merge['X'] - tmp_merge['t_rusher_x'])
    x_tmp = tmp_merge.groupby(['GameId', 'PlayId']).aggregate({'dist_x': ['std']})
    x_tmp.columns = ['dist_x_std']
    tmp['dist_x_std'] = x_tmp['dist_x_std'].values
    #y_tmp = tmp_merge.groupby(['GameId', 'PlayId']).aggregate({'dist_y': ['std']})
    #y_tmp.columns = ['dist_y_std']
    #tmp['dist_y_std'] = y_tmp['dist_y_std'].values
    return tmp[['GameId', 'PlayId', 'dist_x_std']]


# In[8]:


def create_feature(df, fix_ori,
                   ori_dir_sub,
                   process_acc,
                   process_ori_sub,
                   process_radian,
                   process_xy_std,
                   fillna=-999):
    def combine_df(df1, df2):
        return pd.merge(df1, df2, on=['GameId', 'PlayId'], how='inner')
    
    df = df.copy()
    df = process_yardline(df)
    df = update_orientation(df, fix_ori, ori_dir_sub)
    
    static_feats = process_unique_features(df)
    back_feats = back_features(df) #by rusher
    rel_back = features_relative_to_back(df, back_feats)
    def_feats = defense_features(df)
    rush_feats = rusher_features(df)
    
    table = df[df['NflId'] == df['NflIdRusher']][['GameId', 'PlayId']]
    table = combine_df(table, rel_back)
    table = combine_df(table, static_feats)
    table = combine_df(table, def_feats)
    table = combine_df(table, rush_feats)
    
    if process_acc:
        acc_feats = process_player_acc(df)
        table = combine_df(table, acc_feats)
    
    if process_ori_sub:
        ori_feats = ori_dir_sub_features(df)
        table = combine_df(table, ori_feats)
    
    if process_xy_std:
        std_feats = std_features(df)
        table = combine_df(table, std_feats)
        
    if process_radian:
        radian_df = radian_features(df)
        table = combine_df(table, radian_df)
    
    rusher = df[df['NflId'] == df['NflIdRusher']][['GameId', 'PlayId', 'Team']]
    rusher.columns = ['GameId', 'PlayId', 'RusherTeam']
    tmp = pd.merge(df[['GameId', 'PlayId', 'Team', 'X', 'YardLine']], rusher, on=['GameId', 'PlayId'], how='left')
    tmp['back_scrim'] = tmp['YardLine'] - tmp['X']
    tmp_away = tmp[tmp['Team'] != tmp['RusherTeam']].groupby(['GameId', 'PlayId'])['back_scrim'].mean().reset_index()
    tmp_away.columns = ['GameId', 'PlayId', 'back_scrim_away']
    table = pd.merge(table, tmp_away, on=['GameId', 'PlayId'], how='left')
    tmp_home = tmp[tmp['Team'] == tmp['RusherTeam']].groupby(['GameId', 'PlayId'])['back_scrim'].mean().reset_index()
    tmp_home.columns = ['GameId', 'PlayId', 'back_scrim_home']
    table = pd.merge(table, tmp_home, on=['GameId', 'PlayId'], how='left')
    
    table.drop(['GameId','PlayId', 'X', 'Y','A', 'S'], axis=1, inplace=True)
    return table


# In[9]:


def get_crps(y_pred, y_valid):
    y_valid = np.clip(np.cumsum(y_valid, axis=1), 0, 1)
    y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
    val_s = ((y_valid - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * y_valid.shape[0])
    crps = np.round(val_s, 6)
    return crps


# In[10]:


class CRPSCallback(Callback):
    
    def __init__(self,validation, predict_batch_size=20, include_on_batch=False):
        super(CRPSCallback, self).__init__()
        self.validation = validation
        self.predict_batch_size = predict_batch_size
        self.include_on_batch = include_on_batch
        
        print('validation shape',len(self.validation))

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_train_begin(self, logs={}):
        if not ('CRPS_score_val' in self.params['metrics']):
            self.params['metrics'].append('CRPS_score_val')

    def on_batch_end(self, batch, logs={}):
        if (self.include_on_batch):
            logs['CRPS_score_val'] = float('-inf')

    def on_epoch_end(self, epoch, logs={}):
        logs['CRPS_score_val'] = float('-inf')
            
        if (self.validation):
            X_valid, y_valid = self.validation[0], self.validation[1]
            y_pred = self.model.predict(X_valid)
            y_true = np.clip(np.cumsum(y_valid, axis=1), 0, 1)
            y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
            val_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * X_valid.shape[0])
            val_s = np.round(val_s, 6)
            logs['CRPS_score_val'] = val_s
    


# In[11]:


def get_nn_model(x_tr,y_tr,x_val,y_val,step):
    inp = Input(shape = (x_tr.shape[1],))
    x = Dense(512, input_dim=x_tr.shape[1], activation='relu')(inp)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    out = Dense(199, activation='softmax')(x)
    model = Model(inp,out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[])
    
    es = EarlyStopping(monitor='val_loss', 
                       mode='min',
                       restore_best_weights=True, 
                       verbose=1, 
                       patience=30)

    mc = ModelCheckpoint('best_model_{}.h5'.format(step),monitor='CRPS_score_val',mode='min',
                                   save_best_only=True, verbose=1, save_weights_only=True)
    
    bsz = 1024
    steps = x_tr.shape[0]/bsz
    


    hist = model.fit(x_tr, y_tr,callbacks=[CRPSCallback(validation = (x_val,y_val)),es,mc], epochs=250, batch_size=bsz,verbose=1)
    model.load_weights('best_model_{}.h5'.format(step))
    y_pred = model.predict(x_val)
    crps = get_crps(y_pred, y_val)

    return model, crps, hist


# In[12]:


def train(X, yards, yards_label, step=2, fold=5):
    crpses = []
    models = []
    s_time = time.time()
    hists = []
    fold = fold
    y = get_y(yards)
    for i in range(step):
        kfold = StratifiedKFold(n_splits=fold, random_state = 42+i, shuffle = True)
        for k_fold, (tr_inds, val_inds) in enumerate(kfold.split(yards_label,yards_label)):
            tr_x, tr_y = X[tr_inds],y[tr_inds]
            val_x, val_y = X[val_inds],y[val_inds]
            model,crps, hist = get_nn_model(tr_x,tr_y,val_x,val_y,i*fold+(k_fold+1))
            models.append(model)
            crpses.append(crps)
            hists.append(hist)
            print("the %d fold crps is %f"%((i*(k_fold+1))+(k_fold+1),crps))
            
    print("mean crps is %f"%np.mean(crpses))
    
    return models, crpses, hists


# In[13]:


def get_y(yards):
    y = np.zeros((yards.shape[0], 199))
    for idx, target in enumerate(list(yards)):
        y[idx][99 + target] = 1
    return y


# In[14]:


def predict(x_te, models):
    model_num = len(models)
    for k,m in enumerate(models):
        if k==0:
            y_pred = m.predict(x_te,batch_size=1024)
        else:
            y_pred+=m.predict(x_te,batch_size=1024)
    y_pred = y_pred / model_num
    
    return y_pred


# In[15]:


fillna = -999
fix_ori = True
ori_dir_sub = True
process_ori_sub = True
process_acc =  True
process_radian = False
process_xy_std = True
input_df = create_feature(df, 
                          fix_ori=fix_ori,
                          ori_dir_sub=ori_dir_sub,
                          process_acc=process_acc,
                          process_ori_sub=process_ori_sub,
                          process_radian=process_radian,
                          process_xy_std=process_xy_std,
                          fillna=fillna)
print('[*]input_df create')
print(input_df.shape)


# In[16]:


input_df.head()


# In[17]:


scaler = StandardScaler()
X = scaler.fit_transform(input_df)
yards = np.array([df['Yards'][i] for i in range(0,df.shape[0],22)])
df['Yards_Label'] = df['Yards'] // 10  # use stratifiedkfold
yards_label = np.array([df['Yards_Label'][i] for i in range(0,df.shape[0],22)])
models, crpses, hist = train(X, yards, yards_label=yards_label, step=2, fold=5)


# In[18]:


# x y std info
print(X.shape)
print('crps mean: ', np.mean(crpses))
# befor fix(columns 43) 0.1260
loss_sum = 0
for i in range(10):
    loss_sum += hist[i].history['loss'][hist[i].history['CRPS_score_val'].index(min(hist[i].history['CRPS_score_val']))]
print('loss mean: ', loss_sum/10)


# In[19]:


if  TRAIN_OFFLINE==True:
    basetable = create_feature(df.iloc[:22, :],
                               fix_ori=fix_ori,
                              ori_dir_sub=ori_dir_sub,
                              process_acc=process_acc,
                              process_ori_sub=process_ori_sub,
                              process_radian=process_radian,
                              process_xy_std=process_xy_std,
                              fillna=fillna)
    
    scaled_basetable = scaler.transform(basetable)
    y_pred = predict(scaled_basetable, models=models)
    y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1).tolist()[0]
    print(len(y_pred))
    print(y_pred)

elif  TRAIN_OFFLINE==False:
    from kaggle.competitions import nflrush
    import tqdm
    env = nflrush.make_env()
    iter_test = env.iter_test()

    for (test_df, sample_prediction_df) in tqdm.tqdm(iter_test):
        basetable = create_feature(test_df,
                                   fix_ori=fix_ori,
                                  ori_dir_sub=ori_dir_sub,
                                  process_acc=process_acc,
                                  process_ori_sub=process_ori_sub,
                                  process_radian=process_radian,
                                  process_xy_std=process_xy_std,
                                  fillna=fillna)
        
        scaled_basetable = scaler.transform(basetable)
        y_pred = predict(scaled_basetable, models=models)
        y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1).tolist()[0]

        preds_df = pd.DataFrame(data=[y_pred], columns=sample_prediction_df.columns)
        env.predict(preds_df)

    env.write_submission_file()
print('done')


# In[ ]:




