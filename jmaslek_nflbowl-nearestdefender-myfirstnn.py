#!/usr/bin/env python
# coding: utf-8



from kaggle.competitions import nflrush

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.patches as patches
import seaborn as sns; sns.set(color_codes=True)
import datetime
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from scipy.stats import norm
from keras import backend as K
import tensorflow as tf
import tqdm
import time
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler

df_train=pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv',low_memory=False)
df = df_train




def personnelmap(val):
    if 'OL' not in val:
        return val+', 5 OL'
    else:
        return val

def OffensePersonnelSplit(x):
    dic = {'DB' : 0, 'DL' : 0, 'LB' : 0, 'OL' : 0, 'QB' : 0, 'RB' : 0, 'TE' : 0, 'WR' : 0}
    for xx in x.split(","):
        xxs = xx.split(" ")
        dic[xxs[-1]] = int(xxs[-2])
    return dic

def DefensePersonnelSplit(x):
    dic = {'DB' : 0, 'DL' : 0, 'LB' : 0, 'OL' : 0}
    for xx in x.split(","):
        xxs = xx.split(" ")
        dic[xxs[-1]] = int(xxs[-2])
    return dic




df['OffensePersonnel'] = df['OffensePersonnel'].apply(personnelmap)
print(df.columns)




get_ipython().run_line_magic('time', '')
def weather1(txt):
    txt = str(txt).lower()
    if pd.isna(txt):
        return "none"
    if "indoor" in txt:
        return "indoor"
    if "rain" in txt:
        return "rain"
    if "snow" in txt:
        return "snow"
    else:
        return "other"
#For Later
def norm(x):
    return (x - train_stats['mean']) / (train_stats['std'])

# Function for height to inches.  Already in X-Y -> h=12X+Y
def height_to_inch(x):
    return 12 * int(x.split('-')[0]) + int(x.split('-')[1])

df['PlayerHeight'] = df['PlayerHeight'].apply(height_to_inch)

#Remap Possession Teams for consistancy 
map_abbr = {'ARI': 'ARZ', 'BAL': 'BLT', 'CLE': 'CLV', 'HOU': 'HST'}
for abb in df['PossessionTeam'].unique():
    map_abbr[abb] = abb

#Dummy Variables for Snow, Rain, Indoors, Turf
    
df['GameWeather'] = df['GameWeather'].apply(weather1)
df = pd.concat([df.drop(['GameWeather'], axis=1), pd.get_dummies(df['GameWeather'], prefix='Weather')], axis=1)

Turf = {'Field Turf':'Artificial', 'A-Turf Titan':'Artificial', 'Grass':'Natural', 'UBU Sports Speed S5-M':'Artificial', 'Artificial':'Artificial', 'DD GrassMaster':'Artificial', 'Natural Grass':'Natural', 'UBU Speed Series-S5-M':'Artificial', 'FieldTurf':'Artificial', 'FieldTurf 360':'Artificial', 'Natural grass':'Natural', 'grass':'Natural', 'Natural':'Natural', 'Artifical':'Artificial', 'FieldTurf360':'Artificial', 'Naturall Grass':'Natural', 'Field turf':'Artificial', 'SISGrass':'Artificial', 'Twenty-Four/Seven Turf':'Artificial', 'natural grass':'Natural'} 
df['Turf'] = df['Turf'].map(Turf)

df = pd.concat([df.drop(['Turf'], axis=1), pd.get_dummies(df['Turf'], prefix='Turf')], axis=1)




df['PossessionTeam'] = df['PossessionTeam'].map(map_abbr)
df['HomeTeamAbbr'] = df['HomeTeamAbbr'].map(map_abbr)
df['VisitorTeamAbbr'] = df['VisitorTeamAbbr'].map(map_abbr)
df['X_std'] = df.apply(lambda x: x.X if x.PlayDirection == 'right' else 120 - x.X, axis=1)
df['Y_std'] = df.apply(lambda x: x.Y if x.PlayDirection == 'right' else 53.3 - x.Y, axis=1)
df['Orientation_std'] = df.apply(lambda x: x.Orientation     if x.PlayDirection == 'right'     else x.Orientation + 180, axis=1)
df['YardLine_std'] = df.apply(lambda x: x.YardLine + 10     if (x.PlayDirection == 'right') & (x.FieldPosition == x.PossessionTeam)        | (x.PlayDirection == 'left') & (x.FieldPosition == x.PossessionTeam)     else 60 + (50 - x.YardLine), axis=1)
df['FieldPosition_std'] = df.apply(lambda x: 'left'     if x.FieldPosition ==        x.PossessionTeam     else 'right', axis=1)

df['OffDef'] = df.apply(lambda x: "Off" if ((x.Team == 'home') & (x.PossessionTeam == x.HomeTeamAbbr)) |                                            ((x.Team == 'away') & (x.PossessionTeam == x.VisitorTeamAbbr))                                         else "Def", axis=1)

df.drop(['X', 'Y', 'Orientation', 'YardLine', 'FieldPosition'], axis=1, inplace=True)
df["Rusher"]=df["NflId"]==df["NflIdRusher"]
df.drop(columns=["WindDirection","WindSpeed","PlayerCollegeName"],inplace=True)

df['TimeSnap'] = df['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ'))
df['TimeHandoff'] = df['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ'))

df['TimeToHandoff'] = df.apply(lambda row: (row['TimeHandoff']-row['TimeSnap']).total_seconds(),axis=1)

df = pd.concat([df.drop(['OffenseFormation'], axis=1), pd.get_dummies(df['OffenseFormation'], prefix='Formation')], axis=1)
df["DistanceToRusher"] = (df.X_std - df.X_std[df.groupby('PlayId')['Rusher'].transform('idxmax')].reset_index(drop=True))**2                         +(df.Y_std - df.Y_std[df.groupby('PlayId')['Rusher'].transform('idxmax')].reset_index(drop=True))**2
df["DistanceToRusher"] = np.sqrt(df.DistanceToRusher)
df["Def_DistanceToRusher"] = df.apply(lambda x:x.DistanceToRusher if x.OffDef=='Def' else np.nan,axis=1)


# New DF that has rusher only and the position of the closest defender
df_rusher = pd.merge(df.loc[df['Rusher'].groupby(df['PlayId']).idxmax()],         df.loc[df['Def_DistanceToRusher'].groupby(df['PlayId']).idxmin(), ['PlayId','Def_DistanceToRusher']],         on='PlayId',
         suffixes=['','_closest']
        )

df_rusher.drop(columns=['Def_DistanceToRusher','Rusher','OffDef','Humidity'], inplace=True)
df_rusher.drop(columns=['PlayId','GameId','TimeHandoff','TimeSnap','DistanceToRusher'],inplace=True)

df_rusher['Direction_Rad'] = (90 - df_rusher.loc[:,'Dir']) * np.pi / 180.0

df_rusher['RusherVx'] = np.abs(df_rusher.loc[:,'S']) * np.cos(df_rusher.Direction_Rad)
df_rusher['RusherVy'] = np.abs(df_rusher.loc[:,'S']) * np.sin(df_rusher.Direction_Rad)

df_rusher['TimeToClosestDefender'] = df_rusher.loc[:,'Def_DistanceToRusher_closest']/(df_rusher.loc[:,'S']+0.0001)

df_rusher.drop(columns='Direction_Rad')

print(list(df_rusher.columns))




import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

df_rushertemp = df_rusher.dropna()

df_rushertemp['Yards1'] = df_rushertemp.Yards + 99
df_rushertemp.drop(columns='Yards',inplace=True)
print(df_rushertemp.columns)

train_dataset = df_rushertemp
train_dataset.drop(columns = ['GameClock','DefensePersonnel','OffensePersonnel','HomeTeamAbbr','VisitorTeamAbbr','Stadium','Team','StadiumType','Location','PossessionTeam','DisplayName','FieldPosition_std','NflId','JerseyNumber','PlayDirection','PlayerBirthDate','Position'],inplace = True)

train_stats = train_dataset.describe()
train_stats.pop('Yards1')
train_stats = train_stats.transpose()

train_labels = train_dataset.pop('Yards1')


ss = StandardScaler()

#normed_train_data = norm(train_dataset)
scaled_df = ss.fit_transform(train_dataset)
normed_train_data = pd.DataFrame(scaled_df, columns=train_dataset.columns)
dummy_col = normed_train_data.columns




def build_model():
  model = keras.Sequential([
    layers.Input(shape=(len(train_dataset.keys()),)),
    layers.Dense(800, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(.1), 
    layers.Dense(400,activation='relu'),
    layers.LeakyReLU(alpha = .15),
    layers.Dropout(.25),
    layers.BatchNormalization(),      
    layers.Dense(400),
    layers.PReLU(), 
    layers.Dense(200, activation='relu'),
    layers.Dropout(.15), 
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(.1),
    layers.BatchNormalization(),
    layers.Dense(199,activation='softmax')  
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)
  #optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.2, nesterov=False)
  #optimizer = tf.keras.optimizers.Adam(learning_rate=0.005, decay = 1e-4)
    
  model.compile(loss='sparse_categorical_crossentropy' ,
                optimizer=optimizer,
                metrics=['mae','accuracy'])
  return model


model = build_model()
model.summary()




rkf = RepeatedKFold(n_splits=5, n_repeats=5)

x1 = time.time()
X_train = normed_train_data
Y_train = train_labels
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

EPOCHS = 100


for tr_idx, vl_idx in rkf.split(X_train, Y_train):
    x_tr, y_tr = X_train.iloc[tr_idx], Y_train.iloc[tr_idx]
    x_vl, y_vl = X_train.iloc[vl_idx], Y_train.iloc[vl_idx]
    history = model.fit(x_tr, y_tr, epochs=EPOCHS,
                    validation_data=[x_vl, y_vl], verbose=1, callbacks=[early_stop])
    
x2 = time.time()
print(x2-x1)




def calc_crps(y_pred_cdfs, actuals):
     stops = np.arange(-99, 100)
     unit_steps = stops >= actuals.reshape(-1, 1)
     crps = np.mean((y_pred_cdfs - unit_steps)**2)
     return crps




ypred = model.predict(X_train) 
calc_crps(np.cumsum(ypred,axis=1),np.array(Y_train))/199




def make_pred(df,sample,env,model):
    df = df.reset_index()
    for abb in df['PossessionTeam'].unique():
        map_abbr[abb] = abb
    
    df['PlayerHeight'] = df['PlayerHeight'].apply(height_to_inch)

    df['GameWeather'] = df['GameWeather'].apply(weather1)
    df = pd.concat([df.drop(['GameWeather'], axis=1), pd.get_dummies(df['GameWeather'], prefix='Weather')], axis=1)
    df['Turf'] = df['Turf'].map(Turf)
    df = pd.concat([df.drop(['Turf'], axis=1), pd.get_dummies(df['Turf'], prefix='Turf')], axis=1)
    df['PossessionTeam'] = df['PossessionTeam'].map(map_abbr)
    df['HomeTeamAbbr'] = df['HomeTeamAbbr'].map(map_abbr)
    df['VisitorTeamAbbr'] = df['VisitorTeamAbbr'].map(map_abbr)
    df['X_std'] = df.apply(lambda x: x.X if x.PlayDirection == 'right' else 120 - x.X, axis=1)
    df['Y_std'] = df.apply(lambda x: x.Y if x.PlayDirection == 'right' else 53.3 - x.Y, axis=1)
    df['Orientation_std'] = df.apply(lambda x: x.Orientation         if x.PlayDirection == 'right'         else x.Orientation + 180, axis=1)
    df['YardLine_std'] = df.apply(lambda x: x.YardLine + 10         if (x.PlayDirection == 'right') & (x.FieldPosition == x.PossessionTeam)            | (x.PlayDirection == 'left') & (x.FieldPosition == x.PossessionTeam)         else 60 + (50 - x.YardLine), axis=1)
    df['FieldPosition_std'] = df.apply(lambda x: 'left'         if x.FieldPosition == x.PossessionTeam         else 'right', axis=1)

    df['OffDef'] = df.apply(lambda x: "Off" if ((x.Team == 'home') & (x.PossessionTeam == x.HomeTeamAbbr)) |                                            ((x.Team == 'away') & (x.PossessionTeam == x.VisitorTeamAbbr))                                             else "Def", axis=1)

    df.drop(['X', 'Y', 'Orientation', 'YardLine', 'FieldPosition',"WindDirection","WindSpeed","PlayerCollegeName"], axis=1, inplace=True)
    
    df["Rusher"]=df["NflId"]==df["NflIdRusher"]

    df['TimeSnap'] = df['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ'))
    df['TimeHandoff'] = df['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ'))

    df['TimeToHandoff'] = df.apply(lambda row: (row['TimeHandoff']-row['TimeSnap']).total_seconds(),axis=1)

    df = pd.concat([df.drop(['OffenseFormation'], axis=1), pd.get_dummies(df['OffenseFormation'], prefix='Formation')], axis=1)
    df["DistanceToRusher"] = (df.X_std - df.X_std[df.groupby('PlayId')['Rusher'].transform('idxmax')].reset_index(drop=True))**2                             +(df.Y_std - df.Y_std[df.groupby('PlayId')['Rusher'].transform('idxmax')].reset_index(drop=True))**2
    df["DistanceToRusher"] = np.sqrt(df.DistanceToRusher)
    df["Def_DistanceToRusher"] = df.apply(lambda x:x.DistanceToRusher if x.OffDef=='Def' else np.nan,axis=1)

    df_rusher = pd.DataFrame(data=[df.loc[df['Rusher'].idxmax()]])
    df_rusher["Def_DistanceToRusher"] = np.min(df.loc[:,'Def_DistanceToRusher'])
    df_rusher.drop(columns=['Rusher','OffDef','Humidity'], inplace=True)
    df_rusher.drop(columns=['PlayId','GameId','TimeHandoff','TimeSnap'],inplace=True)
    df_rusher['Direction_Rad'] = (90 - df_rusher.loc[:,'Dir']) * np.pi / 180.0

    df_rusher['RusherVx'] = np.abs(df_rusher.loc[:,'S']) * np.cos(df_rusher.Direction_Rad)
    df_rusher['RusherVy'] = np.abs(df_rusher.loc[:,'S']) * np.sin(df_rusher.Direction_Rad)
    df_rusher.drop(columns='Direction_Rad')
    df_rusher['TimeToClosestDefender'] = df_rusher.loc[:,'Def_DistanceToRusher']/(df_rusher.loc[:,'S']+0.0001)

    train_dataset = df_rusher
    train_dataset.drop(columns = ['GameClock','DefensePersonnel','OffensePersonnel','HomeTeamAbbr','VisitorTeamAbbr','Stadium','Team','StadiumType','Location','PossessionTeam','DisplayName','FieldPosition_std','NflId','JerseyNumber','PlayDirection','PlayerBirthDate','Position','DistanceToRusher'],inplace = True)
    missing_cols = set( dummy_col ) - set( train_dataset.columns )-set('Yards')
    
    for c in missing_cols:
        train_dataset[c] = 0
    
    train_dataset = train_dataset[dummy_col]    
    
    scaled_df = ss.fit_transform(train_dataset)
    normed_train_data = pd.DataFrame(scaled_df, columns=train_dataset.columns)
    
    if np.array(normed_train_data.Temperature.isna())==True:
        normed_train_data.loc[:,'Temperature'] = 0
        
    y_pred = model.predict(normed_train_data)
    y_pred = np.cumsum(y_pred)
    env.predict(pd.DataFrame(data=[y_pred.clip(0,1)],columns=sample.columns))
    return y_pred




env=nflrush.make_env()
testing = []
for test, sample in tqdm.tqdm(env.iter_test()):
    make_pred(test, sample, env, model)
    
env.write_submission_file()    

