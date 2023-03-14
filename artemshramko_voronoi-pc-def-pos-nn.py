#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
TRAIN_ABLE_FALSE=True
if TRAIN_ABLE_FALSE:
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import numpy as np
import pandas as pd
import sklearn.metrics as mtr
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from keras.layers import Dense,Input,Flatten,concatenate,Dropout,Lambda,BatchNormalization
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from  keras.callbacks import EarlyStopping,ModelCheckpoint
import datetime

import math
import scipy
from random import choice
from scipy.spatial.distance import euclidean
from scipy.special import expit
from tqdm import tqdm

from scipy.spatial import Voronoi, voronoi_plot_2d

TRAIN_OFFLINE = False


pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 150)


# In[2]:


import numpy as np
import pandas as pd
import sklearn.metrics as mtr
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
import time

from keras.layers import Dense,Input,Flatten,concatenate,Dropout,Lambda,BatchNormalization,LeakyReLU,PReLU,ELU,ThresholdedReLU,Concatenate
from keras.models import Model
import keras.backend as K
from  keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.optimizers import Adam
import re
from keras.losses import binary_crossentropy
from  keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
import codecs
from keras.utils import to_categorical
from sklearn.metrics import f1_score

import datetime
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')

TRAIN_OFFLINE = False

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 150)


# In[3]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[4]:


# train = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', dtype={'WindSpeed': 'object'})
if TRAIN_OFFLINE:
    train = pd.read_csv('../input/train.csv', dtype={'WindSpeed': 'object'})
else:
    train = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', dtype={'WindSpeed': 'object'})
    


# In[5]:


outcomes = train[['GameId','PlayId','Yards']].drop_duplicates()


# In[6]:



train.head()


# In[7]:


list(train.columns)


# In[ ]:





# In[ ]:





# In[8]:


def positional(df):
    
    
    # Offense
    offensive_line = ["G","C","T", "OL"] #calculate their force, distance of runner to centroid, st.d.
    running_back = ["RB","FB","HB","TB","WB"] # running back, oposing
    wide_receiver = ["SE","FL","SL"] #run fast

    # Defense
    defensive_line = ["DL","DT","NT","DE"] #force
    lineback = ["LB","OLB","ILB","MLB"] # force, speed
    defensive_back = ["DB","SS","FS","CB","NB"]

    def field_position(x):
        if x in offensive_line:
            return "OL"
        if x in running_back:
            return "RB"
        if x in wide_receiver:
            return "WR"
        if x in defensive_line:
            return "DL"
        if x in lineback:
            return "LB"
        if x in defensive_back:
            return "DB"

    df["pos_group"] = df["Position"].apply(field_position)
    
    df["Force"] = df[["A","PlayerWeight"]].apply(lambda x: max(x[0]*x[1],x[1]),axis=1)
    
    # Rusher feats

    rusher = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','Dir_std', 'S', 'A', 'X_std', 'Y_std', "Force","Voronoi"]]
                                                 # "IsBallCarrier", "IsOnOffense"]]
    rusher.columns = ['GameId','PlayId', 'RusherDir', 'RusherS', 'RusherA', 'RusherX', 'RusherY',"RusherForce","RusherVoronoi"]

    df = df.merge(rusher,on=['GameId','PlayId'], how='inner')

    init_cols = df.columns
    
    # Offense

    df["OL_force"] = df[df["pos_group"]=="OL"].groupby(["GameId","PlayId"])["Force"].transform("mean")
    df["OL_Voronoi"] = df[df["pos_group"]=="OL"].groupby(["GameId","PlayId"])["Voronoi"].transform("sum")
    df["OL_X"] = df[df["pos_group"]=="OL"].groupby(["GameId","PlayId"])["X_std"].transform("mean")
    df["OL_Y"] = df[df["pos_group"]=="OL"].groupby(["GameId","PlayId"])["Y_std"].transform("mean")

    df["RB_force"] = df[df["pos_group"]=="RB"].groupby(["GameId","PlayId"])["Force"].transform("mean")
    df["RB_Voronoi"] = df[df["pos_group"]=="RB"].groupby(["GameId","PlayId"])["Voronoi"].transform("sum")
    df["RB_X"] = df[df["pos_group"]=="RB"].groupby(["GameId","PlayId"])["X_std"].transform("mean")
    df["RB_Y"] = df[df["pos_group"]=="RB"].groupby(["GameId","PlayId"])["Y_std"].transform("mean")


    #df["WR_force"] = df[df["pos_group"]=="WR"].groupby(["GameId","PlayId"])["Force"].transform("mean")
    #df["WR_Voronoi"] = df[df["pos_group"]=="WR"].groupby(["GameId","PlayId"])["Voronoi"].transform("mean")
    #df["WR_speed"] = df[df["pos_group"]=="WR"].groupby(["GameId","PlayId"])["S"].transform("mean")

    # Defense

    df["DL_force"] = df[df["pos_group"]=="DL"].groupby(["GameId","PlayId"])["Force"].transform("mean")
    df["DL_Voronoi"] = df[df["pos_group"]=="DL"].groupby(["GameId","PlayId"])["Voronoi"].transform("sum")
    df["DL_X"] = df[df["pos_group"]=="DL"].groupby(["GameId","PlayId"])["X_std"].transform("mean")
    df["DL_Y"] = df[df["pos_group"]=="DL"].groupby(["GameId","PlayId"])["Y_std"].transform("mean")

    df["LB_force"] = df[df["pos_group"]=="LB"].groupby(["GameId","PlayId"])["Force"].transform("mean")
    df["LB_Voronoi"] = df[df["pos_group"]=="LB"].groupby(["GameId","PlayId"])["Voronoi"].transform("sum")
    df["LB_X"] = df[df["pos_group"]=="LB"].groupby(["GameId","PlayId"])["X_std"].transform("mean")
    df["LB_Y"] = df[df["pos_group"]=="LB"].groupby(["GameId","PlayId"])["Y_std"].transform("mean")
    #df["LB_Y_stdiv"] = df[df["pos_group"]=="LB"].groupby(["GameId","PlayId"])["Y_std"].transform("std")
    #df["LB_X_stdiv"] = df[df["pos_group"]=="LB"].groupby(["GameId","PlayId"])["Y_std"].transform("std")

    #df["LB_Y_spread"] = df[df["pos_group"]=="LB"].groupby(["GameId","PlayId"])["Y_std"].transform("max") -\
    #                    df[df["pos_group"]=="LB"].groupby(["GameId","PlayId"])["Y_std"].transform("min")
    #df["LB_X_spread"] = df[df["pos_group"]=="LB"].groupby(["GameId","PlayId"])["X_std"].transform("max") -\
    #                df[df["pos_group"]=="LB"].groupby(["GameId","PlayId"])["X_std"].transform("min")

    df["DB_force"] = df[df["pos_group"]=="DB"].groupby(["GameId","PlayId"])["Force"].transform("mean")
    df["DB_Voronoi"] = df[df["pos_group"]=="DB"].groupby(["GameId","PlayId"])["Voronoi"].transform("sum")
    df["DB_speed"] = df[df["pos_group"]=="DB"].groupby(["GameId","PlayId"])["S"].transform("mean")
    df["DB_X"] = df[df["pos_group"]=="DB"].groupby(["GameId","PlayId"])["X_std"].transform("mean")
    df["DB_Y"] = df[df["pos_group"]=="DB"].groupby(["GameId","PlayId"])["Y_std"].transform("mean")
   # df["DB_Y_stdiv"] = df[df["pos_group"]=="DB"].groupby(["GameId","PlayId"])["Y_std"].transform("std")
   # df["DB_X_stdiv"] = df[df["pos_group"]=="DB"].groupby(["GameId","PlayId"])["X_std"].transform("std")
   # df["DB_Y_spread"] = df[df["pos_group"]=="DB"].groupby(["GameId","PlayId"])["Y_std"].transform("max") - \
   #                     df[df["pos_group"]=="DB"].groupby(["GameId","PlayId"])["Y_std"].transform("min")
   # df["DB_X_spread"] = df[df["pos_group"]=="DB"].groupby(["GameId","PlayId"])["X_std"].transform("max") - \
   #                     df[df["pos_group"]=="DB"].groupby(["GameId","PlayId"])["X_std"].transform("min")

    pos_cols = ["OL_force", "OL_Voronoi", "OL_X", "OL_Y",
                "RB_force", "RB_Voronoi", "RB_X", "RB_Y",
                #"WR_force", #"WR_speed", "WR_Voronoi", 
                "DL_force", "DL_Voronoi", "DL_X", "DL_Y", 
                "LB_force", "LB_Voronoi", "LB_X", "LB_Y", #"LB_Y_spread", "LB_Y_stdiv", "LB_X_stdiv", "LB_X_spread",
                "DB_force", "DB_speed", "DB_Voronoi","DB_X", "DB_Y"] #"DB_Y_stdiv", "DB_Y_spread","DB_X_spread", "DB_X_stdiv"]

    for col in pos_cols:
        df[col] = df.groupby(["GameId","PlayId"])[col].transform(lambda grp: grp.fillna(np.mean(grp)))

    df["OL_vs_DL_Force"] = df["OL_force"] - df["DL_force"]
    #df["OL_vs_DL_Voronoi"] = df["OL_Voronoi"] - df["DL_Voronoi"]

    df["RB_vs_LB_Force"] = df["RB_force"] - df["LB_force"]
    df["RB_vs_LB_Voronoi"] =  df["RB_Voronoi"] - df["LB_Voronoi"] 
    
    df["Rusher_vs_LB_Force"] = df["RusherForce"] - df["LB_force"]
    df["Rusher_vs_LB_Voronoi"] = df["RusherVoronoi"] - df["LB_Voronoi"]

    df["Rusher_vs_DL_Force"] = df["RusherForce"] - df["DL_force"]
    df["Rusher_vs_DL_Voronoi"] = df["RusherVoronoi"] - df["DL_Voronoi"]
    
    
    #df["WR_vs_DB_Force"] = df["WR_force"] - df["DB_force"]
    #df["WR_vs_DB_Voronoi"] = df["WR_Voronoi"] - df["DB_Voronoi"]
    #df["WR_vs_DB_speed"] = df["WR_speed"] - df["DB_speed"]
    
    # Euclidean distances

    #df["dist_to_DL"] = df[["DL_X", "DL_Y", "RusherX","RusherY"]].apply(lambda x: euclidean( (x[0], x[1]), (x[2], x[3]) ), axis=1)
    df["dist_to_DL"] = ((df["RusherX"]-df["DL_X"])**2+(df["RusherY"]-df["DL_Y"])**2)**0.5
    
    #df["dist_to_LB"] = df[["LB_X", "LB_Y", "RusherX","RusherY"]].apply(lambda x: euclidean( (x[0], x[1]), (x[2], x[3]) ), axis=1)
    df["dist_to_LB"] = ((df["RusherX"]-df["LB_X"])**2+(df["RusherY"]-df["LB_Y"])**2)**0.5

    #df["dist_RB_to_LB"] = df[["LB_X", "LB_Y", "RB_X","RB_Y"]].apply(lambda x: euclidean( (x[0], x[1]), (x[2], x[3]) ), axis=1)
    df["dist_RB_to_LB"] = ((df["LB_X"]-df["RB_X"])**2+(df["LB_Y"]-df["RB_Y"])**2)**0.5

    #df["dist_to_DB"] = df[["DB_X", "DB_Y", "RusherX","RusherY"]].apply(lambda x: euclidean( (x[0], x[1]), (x[2], x[3]) ), axis=1)
    #df["dist_to_RB"] = ((df["RusherX"]-df["RB_X"])**2+(df["RusherY"]-df["RB_Y"])**2)**0.5
    
    #df["dist_to_OL"] = df[["OL_X", "OL_Y", "RusherX","RusherY"]].apply(lambda x: euclidean( (x[0], x[1]), (x[2], x[3]) ), axis=1)
    #df["dist_to_OL"] = ((df["RusherX"]-df["OL_X"])**2+(df["RusherY"]-df["OL_Y"])**2)**0.5

    #df["dist_to_RB"] = df[["RB_X", "RB_Y", "RusherX","RusherY"]].apply(lambda x: euclidean( (x[0], x[1]), (x[2], x[3]) ), axis=1)
    #df["dist_to_DB"] = ((df["RusherX"]-df["DB_X"])**2+(df["RusherY"]-df["DB_Y"])**2)**0.5
    
    drop_cols = [ "OL_X", "OL_Y","RB_X", "RB_Y","LB_X", "LB_Y","DL_X", "DL_Y","DB_X", "DB_Y"]

    force_cols = [col for col in pos_cols if "force" in col]
    #Xs = [col for col in pos_cols if "X" in col]
    #Ys = [col for col in pos_cols if "Y" in col]
    
    df.drop(force_cols, axis=1, inplace = True)
    df.drop(drop_cols, axis=1, inplace = True)
    #df.drop(Xs, axis=1, inplace=True)
    #df.drop(Ys, axis=1, inplace=True)

    new_cols = [col for col in df.columns.values if col not in init_cols]
    
    return df, new_cols


# In[9]:


"""def get_play_control(df):
    
    #cols = list(df.columns)
    
    rusher = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','Dir_std', 'S', 'A', 'X_std', 'Y_std']]
                                                 # "IsBallCarrier", "IsOnOffense"]]
    rusher.columns = ['GameId','PlayId', 'RusherDir', 'RusherS', 'RusherA', 'RusherX', 'RusherY']

    df = df.merge(rusher,on=['GameId','PlayId'], how='inner')


    df["theta"] = df["Orientation_std"].apply(math.radians)
    df["dist_to_ball"] = df[['X_std', 'Y_std', 'RusherX', 'RusherY']].apply(lambda x: euclidean( (x[0], x[1]), (x[2], x[3]) ), axis=1)
    df["RADIUS"] = df["dist_to_ball"].apply(radius_calc)
    df["S_ratio"] = df["S"].apply(lambda x: (x/13)**2)
    
    df["influence"] = df[["RADIUS", "S_ratio", "theta", "RusherX", "RusherY", "S", "X_std", "Y_std" ]].apply(lambda x: influence(*x), axis=1)
    play_control = df[df["IsOnOffense"]].groupby(['GameId','PlayId'])["influence"].sum() - df[~df["IsOnOffense"]].groupby(['GameId','PlayId'])["influence"].sum()
    df = pd.merge(df, play_control.rename("play_control"),on=['GameId','PlayId'],how='inner')    
    #cols.append("influence")
    #cols.append("play_control")
    
    #df.drop(rusher.columns, axis=1, inplace=True)
    #df.drop(["theta", "dist_to_ball", "RADIUS", "S_ratio"], axis=1, inplace=True)
    
    return df[['GameId','PlayId', "influence", "play_control"]]"""


# In[10]:


def new_def_feats(df):
    
    """
    'secondsLeftInHalf',
    'numericFormation',
    "Defense_X_stdiv", 
    "Defense_Y_stdiv", 
    "Defense_Y_spread", 
    "Defense_X_spread", 
    "min_time_to_tackle",
    "mean_time_to_tackle",
    "dist_to_offense_centroid",
    "dist_to_defense_centroid"
    """
    
    def translate_game_clock(row):
        raw_game_clock = row['GameClock']
        quarter = row['Quarter']
        minutes, seconds_raw = raw_game_clock.partition(':')[::2]

        seconds = seconds_raw.partition(':')[0]

        total_seconds_left_in_quarter = int(seconds) + (int(minutes) * 60)

        if quarter == 3 or quarter == 1:
            return total_seconds_left_in_quarter + 900
        elif quarter == 4 or quarter == 2:
            return total_seconds_left_in_quarter
    
    df['secondsLeftInHalf'] = df.apply(translate_game_clock, axis=1)
    
    df['OffenseFormation'] = df['OffenseFormation'].map(lambda f : 'EMPTY' if pd.isna(f) else f)

    def formation(row):
        form = row['OffenseFormation'].strip()
        if form == 'SHOTGUN':
            return 0
        elif form == 'SINGLEBACK':
            return 1
        elif form == 'EMPTY':
            return 2
        elif form == 'I_FORM':
            return 3
        elif form == 'PISTOL':
            return 4
        elif form == 'JUMBO':
            return 5
        elif form == 'WILDCAT':
            return 6
        elif form=='ACE':
            return 7
        else:
            return -1

    get_ipython().run_line_magic('time', "df['numericFormation'] = df.apply(formation, axis=1)")
    
    # X and Y st.d.
    df["Defense_X_stdiv"] = df[~df["IsOnOffense"]==True].groupby(["GameId","PlayId"])["X_std"].transform("std")
    df["Defense_Y_stdiv"] = df[~df["IsOnOffense"]==True].groupby(["GameId","PlayId"])["Y_std"].transform("std")

    # Y spread
    df["Defense_Y_max"] = df[~df["IsOnOffense"]==True].groupby(["GameId","PlayId"])["Y_std"].transform("max") 
    df["Defense_Y_min"] = df[~df["IsOnOffense"]==True].groupby(["GameId","PlayId"])["Y_std"].transform("min")
    df["Defense_Y_spread"] = df["Defense_Y_max"] - df["Defense_Y_min"]

    # X spread
    df["Defense_X_max"] = df[~df["IsOnOffense"]==True].groupby(["GameId","PlayId"])["X_std"].transform("max") 
    df["Defense_X_min"] = df[~df["IsOnOffense"]==True].groupby(["GameId","PlayId"])["X_std"].transform("min")
    df["Defense_X_spread"] = df["Defense_X_max"] - df["Defense_X_min"]
        
    # time to tackle
    df["time_to_tackle"] = df["dist_to_ball"]/df["S"].apply(lambda x: max(x,1))  
    df["min_time_to_tackle"] = df[~df["IsOnOffense"]==True].groupby(["GameId","PlayId"])["time_to_tackle"].transform("min")
    df["mean_time_to_tackle"] = df[~df["IsOnOffense"]==True].groupby(["GameId","PlayId"])["time_to_tackle"].transform("mean")
    
    #centroids
    df["Offense_X"] = df[df["IsOnOffense"]==True].groupby(["GameId","PlayId"])["X_std"].transform("mean")
    df["Offense_Y"] = df[df["IsOnOffense"]==True].groupby(["GameId","PlayId"])["X_std"].transform("mean")
    
    df["Defense_X"] = df[~df["IsOnOffense"]==True].groupby(["GameId","PlayId"])["X_std"].transform("mean")
    df["Defense_Y"] = df[~df["IsOnOffense"]==True].groupby(["GameId","PlayId"])["X_std"].transform("mean")
    
    for col in ["Defense_X_stdiv", "Defense_Y_stdiv", "Defense_Y_spread", "Defense_X_spread", 
                "min_time_to_tackle","mean_time_to_tackle",
               "Offense_X","Offense_Y","Defense_X","Defense_Y"]:
        df[col] = df.groupby(["GameId","PlayId"])[col].transform(lambda grp: grp.fillna(np.mean(grp)))
        
    #df["dist_to_offense_centroid"] = df[["Offense_X", "Offense_Y", "RusherX","RusherY"]].apply(lambda x: euclidean( (x[0], x[1]), (x[2], x[3]) ), axis=1)
    df["dist_to_offense_centroid"] = ((df["RusherX"]-df["Offense_X"])**2+(df["RusherY"]-df["Offense_Y"])**2)**0.5

    #df["dist_to_defense_centroid"] = df[["Defense_X", "Defense_Y", "RusherX","RusherY"]].apply(lambda x: euclidean( (x[0], x[1]), (x[2], x[3]) ), axis=1)
    df["dist_to_defense_centroid"] = ((df["RusherX"]-df["Defense_X"])**2+(df["RusherY"]-df["Defense_Y"])**2)**0.5

    
    return df


# In[ ]:





# In[ ]:





# In[11]:


def get_play_control(df):
    
    # it makes sense to use unadjusted X, Y and Orientation for calculating play control
    
    rusher = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','Orientation', 'S', 'A', 'X', 'Y']]
                                                 # "IsBallCarrier", "IsOnOffense"]]
    rusher.columns = ['GameId','PlayId', 'RusherDir', 'RusherS', 'RusherA', 'RusherX', 'RusherY']

    df = df.merge(rusher,on=['GameId','PlayId'], how='inner')


    df["theta"] = df["Orientation"].apply(math.radians)
    #df["dist_to_ball"] = df[['X', 'Y', 'RusherX', 'RusherY']].apply(lambda x: euclidean( (x[0], x[1]), (x[2], x[3]) ), axis=1)
    df["dist_to_ball"] = ((df["RusherX"]-df["X"])**2+(df["RusherY"]-df["Y"])**2)**0.5
    df["RADIUS"] = df["dist_to_ball"].apply(radius_calc)
    df["S_ratio"] = df["S"].apply(lambda x: (x/13)**2)
    
    df["influence"] = df[["RADIUS", "S_ratio", "theta", "RusherX", "RusherY", "S", "X", "Y" ]].apply(lambda x: influence(*x), axis=1)
    #play_control = df[df["IsOnOffense"]].groupby(['GameId','PlayId'])["influence"].sum() - df[~df["IsOnOffense"]].groupby(['GameId','PlayId'])["influence"].sum()
    #play_control.set_index(df["PlayId"])
    #df = pd.merge(df, play_control.rename("play_control"),on=['PlayId'],how='inner')    
    
    df["OffenceControl"] = df[df["IsOnOffense"]].groupby(['GameId','PlayId'])["influence"].transform("sum") 
    df["DefenceControl"] = df[~df["IsOnOffense"]].groupby(['GameId','PlayId'])["influence"].transform("sum")
    
    df["OffenceControl"]= df.groupby(['GameId','PlayId'])["OffenceControl"].fillna(method="ffill")
    df["OffenceControl"]= df.groupby(['GameId','PlayId'])["OffenceControl"].fillna(method="bfill")
    df["DefenceControl"]= df.groupby(['GameId','PlayId'])["DefenceControl"].fillna(method="ffill")
    df["DefenceControl"]= df.groupby(['GameId','PlayId'])["DefenceControl"].fillna(method="bfill")
    
    df["play_control"] = df["OffenceControl"] - df["DefenceControl"]
    
    df = new_def_feats(df)
    
    df.drop(rusher.columns, axis=1, inplace=True)
    df.drop(["theta", "dist_to_ball", "RADIUS", "S_ratio", "OffenceControl", "DefenceControl"], axis=1, inplace=True)
    
    return df


    #cols.append("influence")
    #cols.append("play_control")
    
    #
    #df.drop(["theta", "dist_to_ball", "RADIUS", "S_ratio"], axis=1, inplace=True)
    
    #return df


# In[12]:


@np.vectorize    
def influence(RADIUS, S_ratio, theta, RusherX, RusherY, speed, X_std, Y_std ):

    player_coords = np.array([X_std, Y_std])
    point = np.array([RusherX, RusherY])
    
    S_matrix = np.matrix([[RADIUS * (1 + S_ratio), 0], [0, RADIUS * (1 - S_ratio)]])
    R_matrix = np.matrix([[np.cos(theta), - np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    COV_matrix = np.dot(np.dot(np.dot(R_matrix, S_matrix), S_matrix), np.linalg.inv(R_matrix))
    
    norm_fact = (1 / 2 * np.pi) * (1 / np.sqrt(np.linalg.det(COV_matrix)))    
    mu_play = player_coords + speed * np.array([np.cos(theta), np.sin(theta)]) / 2
    
    intermed_scalar_player = np.dot(np.dot((player_coords - mu_play),
                                    np.linalg.inv(COV_matrix)),
                             np.transpose((player_coords - mu_play)))
    player_influence = norm_fact * np.exp(- 0.5 * intermed_scalar_player[0, 0])
    
    intermed_scalar_point = np.dot(np.dot((point - mu_play), 
                                    np.linalg.inv(COV_matrix)), 
                             np.transpose((point - mu_play)))
    point_influence = norm_fact * np.exp(- 0.5 * intermed_scalar_point[0, 0])

    return point_influence / player_influence


# In[13]:


def radius_calc(dist_to_ball):
    ''' I know this function is a bit awkward but there is not the exact formula in the paper,
    so I try to find something polynomial resembling
    Please consider this function as a parameter rather than fixed
    I'm sure experts in NFL could find a way better curve for this'''
    return 4 + 6 * (dist_to_ball >= 15) + (dist_to_ball ** 3) / 560 * (dist_to_ball < 15)


# In[14]:


def strtoseconds(txt):
    txt = txt.split(':')
    ans = int(txt[0])*60 + int(txt[1]) + int(txt[2])/60
    return ans

def strtofloat(x):
    try:
        return float(x)
    except:
        return -1

def map_weather(txt):
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

def orientation_to_cat(x):
    x = np.clip(x, 0, 360 - 1)
    try:
        return str(int(x/15))
    except:
        return "nan"
def preprocess(train):
    ## GameClock
    train['GameClock_sec'] = train['GameClock'].apply(strtoseconds)
    train["GameClock_minute"] = train["GameClock"].apply(lambda x : x.split(":")[0]).astype("object")

    ## Height
    train['PlayerHeight_dense'] = train['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))

    ## Time
    train['TimeHandoff'] = train['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
    train['TimeSnap'] = train['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))

    train['TimeDelta'] = train.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)
    train['PlayerBirthDate'] = train['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))

    ## Age
    seconds_in_year = 60*60*24*365.25
    train['PlayerAge'] = train.apply(lambda row: (row['TimeHandoff']-row['PlayerBirthDate']).total_seconds()/seconds_in_year, axis=1)
    train["PlayerAge_ob"] = train['PlayerAge'].astype(np.int).astype("object")

    ## WindSpeed
    #train['WindSpeed_ob'] = train['WindSpeed'].apply(lambda x: x.lower().replace('mph', '').strip() if not pd.isna(x) else x)
    #train['WindSpeed_ob'] = train['WindSpeed_ob'].apply(lambda x: (int(x.split('-')[0])+int(x.split('-')[1]))/2 if not pd.isna(x) and '-' in x else x)
    #train['WindSpeed_ob'] = train['WindSpeed_ob'].apply(lambda x: (int(x.split()[0])+int(x.split()[-1]))/2 if not pd.isna(x) and type(x)!=float and 'gusts up to' in x else x)
    #train['WindSpeed_dense'] = train['WindSpeed_ob'].apply(strtofloat)

    ## Weather
    train['GameWeather_process'] = train['GameWeather'].str.lower()
    train['GameWeather_process'] = train['GameWeather_process'].apply(lambda x: "indoor" if not pd.isna(x) and "indoor" in x else x)
    train['GameWeather_process'] = train['GameWeather_process'].apply(lambda x: x.replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party', 'partly') if not pd.isna(x) else x)
    train['GameWeather_process'] = train['GameWeather_process'].apply(lambda x: x.replace('clear and sunny', 'sunny and clear') if not pd.isna(x) else x)
    train['GameWeather_process'] = train['GameWeather_process'].apply(lambda x: x.replace('skies', '').replace("mostly", "").strip() if not pd.isna(x) else x)
    train['GameWeather_dense'] = train['GameWeather_process'].apply(map_weather)

    ## Rusher
    train['IsRusher'] = (train['NflId'] == train['NflIdRusher'])
    train['IsRusher_ob'] = (train['NflId'] == train['NflIdRusher']).astype("object")
    temp = train[train["IsRusher"]][["Team", "PlayId"]].rename(columns={"Team":"RusherTeam"})
    train = train.merge(temp, on = "PlayId")
    train["IsRusherTeam"] = train["Team"] == train["RusherTeam"]

    ## dense -> categorical
    train["Quarter_ob"] = train["Quarter"].astype("object")
    train["Down_ob"] = train["Down"].astype("object")
    train["JerseyNumber_ob"] = train["JerseyNumber"].astype("object")
    train["YardLine_ob"] = train["YardLine"].astype("object")
    # train["DefendersInTheBox_ob"] = train["DefendersInTheBox"].astype("object")
    # train["Week_ob"] = train["Week"].astype("object")
    # train["TimeDelta_ob"] = train["TimeDelta"].astype("object")


    ## Orientation and Dir
    train["Orientation_ob"] = train["Orientation"].apply(lambda x : orientation_to_cat(x)).astype("object")
    train["Dir_ob"] = train["Dir"].apply(lambda x : orientation_to_cat(x)).astype("object")

    train["Orientation_sin"] = train["Orientation"].apply(lambda x : np.sin(x/360 * 2 * np.pi))
    train["Orientation_cos"] = train["Orientation"].apply(lambda x : np.cos(x/360 * 2 * np.pi))
    train["Dir_sin"] = train["Dir"].apply(lambda x : np.sin(x/360 * 2 * np.pi))
    train["Dir_cos"] = train["Dir"].apply(lambda x : np.cos(x/360 * 2 * np.pi))

    ## diff Score
    train["diffScoreBeforePlay"] = train["HomeScoreBeforePlay"] - train["VisitorScoreBeforePlay"]
    train["diffScoreBeforePlay_binary_ob"] = (train["HomeScoreBeforePlay"] > train["VisitorScoreBeforePlay"]).astype("object")

    ## Turf
    Turf = {'Field Turf':'Artificial', 'A-Turf Titan':'Artificial', 'Grass':'Natural', 'UBU Sports Speed S5-M':'Artificial', 'Artificial':'Artificial', 'DD GrassMaster':'Artificial', 'Natural Grass':'Natural', 'UBU Speed Series-S5-M':'Artificial', 'FieldTurf':'Artificial', 'FieldTurf 360':'Artificial', 'Natural grass':'Natural', 'grass':'Natural', 'Natural':'Natural', 'Artifical':'Artificial', 'FieldTurf360':'Artificial', 'Naturall Grass':'Natural', 'Field turf':'Artificial', 'SISGrass':'Artificial', 'Twenty-Four/Seven Turf':'Artificial', 'natural grass':'Natural'} 
    train['Turf'] = train['Turf'].map(Turf)

    ## OffensePersonnel
    temp = train["OffensePersonnel"].iloc[np.arange(0, len(train), 22)].apply(lambda x : pd.Series(OffensePersonnelSplit(x)))
    temp.columns = ["Offense" + c for c in temp.columns]
    temp["PlayId"] = train["PlayId"].iloc[np.arange(0, len(train), 22)]
    train = train.merge(temp, on = "PlayId")

    ## DefensePersonnel
    temp = train["DefensePersonnel"].iloc[np.arange(0, len(train), 22)].apply(lambda x : pd.Series(DefensePersonnelSplit(x)))
    temp.columns = ["Defense" + c for c in temp.columns]
    temp["PlayId"] = train["PlayId"].iloc[np.arange(0, len(train), 22)]
    train = train.merge(temp, on = "PlayId")

    ## sort
#     train = train.sort_values(by = ['X']).sort_values(by = ['Dis']).sort_values(by=['PlayId', 'Team', 'IsRusher']).reset_index(drop = True)
    train = train.sort_values(by = ['X']).sort_values(by = ['Dis']).sort_values(by=['PlayId', 'IsRusherTeam', 'IsRusher']).reset_index(drop = True)
    return train


# In[15]:


def clean_features(train):
    
    """
    New clean features:
    
    WindDirection --> WindDirection_std     #in dergees
    
    MatchDay
    
    Hight Feet
    
    try:
        train.loc[train['Season'] == 2017, 'Orientation'] = np.mod(90 + train.loc[train['Season'] == 2017, 'Orientation'], 360)
        print("...    | Season")
    except:
        pass
    """
    #########################################################################################################
    
    # Team names
    
    #########################################################################################################
    try:
    
        train.loc[train.VisitorTeamAbbr == "ARI", 'VisitorTeamAbbr'] = "ARZ"
        train.loc[train.HomeTeamAbbr == "ARI", 'HomeTeamAbbr'] = "ARZ"

        train.loc[train.VisitorTeamAbbr == "BAL", 'VisitorTeamAbbr'] = "BLT"
        train.loc[train.HomeTeamAbbr == "BAL", 'HomeTeamAbbr'] = "BLT"

        train.loc[train.VisitorTeamAbbr == "CLE", 'VisitorTeamAbbr'] = "CLV"
        train.loc[train.HomeTeamAbbr == "CLE", 'HomeTeamAbbr'] = "CLV"

        train.loc[train.VisitorTeamAbbr == "HOU", 'VisitorTeamAbbr'] = "HST"
        train.loc[train.HomeTeamAbbr == "HOU", 'HomeTeamAbbr'] = "HST"
    except:
        pass
    
    #########################################################################################################
    
    # Wind Direction
    
    #########################################################################################################
    
    print("...    | Wind Direction")
    
    try:
    
        train['WindDirection_std'] = train['WindDirection'].fillna(0) #.apply(lambda x: x.lower())
    except:
        train['WindDirection_std'] = 0
        train['WindDirection'] = "unknown"
    
    north = ['N','From S','North']
    south = ['S','From N','South','s']
    west = ['W','From E','West']
    east = ['E','From W','from W','EAST','East']
    
    north_east = ['FROM SW','NE','NORTH EAST','North East','NorthEast','Northeast','From SW']
    north_west = ['E','NW','NORTHWEST', 'Northwest']
    south_east = ['SE','SOUTHEAST','Southeast']
    south_west = ['SW','SOUTHWEST','SouthWest','Southwest']
    
    nne = ["NNE", "From SSW", "N-NE", "North Northeast"]
    ene = ["East North East", 'ENE','From WSW']
    ese = ['East Southeast', 'ESE','From WNW']
    sse = [ 'From NNW', 'South Southeast','SSE']
    ssw = [ 'From NNE', 'South Southwest','SSW']
    wsw = [ 'From ENE', 'W-SW','West-Southwest','WSW']
    wnw = [ 'From ESE', 'W-NW','WNW','West Northwest']
    nnw = [ 'From SSE','North/Northwest','NNW']
    
    def clean_wind_dir(x):
        if x in north:
            return 0
        elif x in south:
            return 180
        elif x in west:
            return 270
        elif x in east:
            return 90
        elif x in north_east:
            return 45
        elif x in north_west:
            return 315
        elif x in south_east:
            return 135
        elif x in south_west:
            return 225
        elif x in nne:
            return 20
        elif x in ene:
            return 70
        elif x in sse:
            return 110
        elif x in ssw:
            return 200
        elif x in wsw:
            return 250
        elif x in wnw:
            return 290
        elif x in nnw:
            return 340
        else:
            return 0
        
    train['WindDirection_std'] = train['WindDirection_std'].apply(clean_wind_dir)
        
    """
    no_wind = ['clear','Calm']
    #nan = ['1','8','13']
    nan = [i for i in train['WindDirection'].fillna("0") if i.isdigit()]

    train['WindDirection_std'] = train['WindDirection_std'].replace(north,0)         #'north'
    train['WindDirection_std'] = train['WindDirection_std'].replace(south,180)       #'south'
    train['WindDirection_std'] = train['WindDirection_std'].replace(west, 270)       #'west'
    train['WindDirection_std'] = train['WindDirection_std'].replace(east, 90)        #'east'
    
    train['WindDirection_std'] = train['WindDirection_std'].replace(north_east,45)   #'north_east'
    train['WindDirection_std'] = train['WindDirection_std'].replace(north_west, 315) #'north_west'
    train['WindDirection_std'] = train['WindDirection_std'].replace(south_east, 135) #'south_east'
    train['WindDirection_std'] = train['WindDirection_std'].replace(south_west, 225) #'south_west'
    
    train['WindDirection_std'] = train['WindDirection_std'].replace(nan,0)
    #train['WindDirection_std'] = train['WindDirection'].fillna("unknown")
    
    train['WindDirection_std'] = train['WindDirection_std'].replace(nne, 20)         # NNE
    train['WindDirection_std'] = train['WindDirection_std'].replace(ene, 70)         # ENE
    train['WindDirection_std'] = train['WindDirection_std'].replace(ese, 110)        # ESE
    train['WindDirection_std'] = train['WindDirection_std'].replace(sse, 160)        # SSE
    train['WindDirection_std'] = train['WindDirection_std'].replace(ssw, 200)        # SSW
    train['WindDirection_std'] = train['WindDirection_std'].replace(wsw, 250)        # WSW
    train['WindDirection_std'] = train['WindDirection_std'].replace(wnw, 290)        # WNW
    train['WindDirection_std'] = train['WindDirection_std'].replace(nnw, 340)        # NNW    
    
    train['WindDirection_std'] = train['WindDirection_std'].replace(no_wind,0)
    #train['WindDirection_std'] = train['WindDirection'].replace("unknown",0)

    def winddir(x):
        if type(x) == str:
            return 0
    train['WindDirection_std'] = train['WindDirection_std'].apply(winddir)
    
    # rotating for right-to-left plays
    """
 
    #########################################################################################################
    
    # Wind Speed
    
    #########################################################################################################
    print("...    | Wind Speed")

    
    
    def give_me_WindSpeed(txt):
        txt = str(txt).lower().replace('mph', '').strip()
        if '-' in txt:
            txt = (int(txt.split('-')[0]) + int(txt.split('-')[1])) / 2
        try:
            return float(txt)
        except:
            return -1
        
    train['WindSpeed'] = train['WindSpeed'].apply(give_me_WindSpeed)
    
    #########################################################################################################
    
    # Weather
    
    #########################################################################################################
    print("...    | Weather")

    
    def give_me_GameWeather(txt):
        txt = str(txt).lower().replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party', 'partly').replace('clear and sunny', 'sunny and clear')
        txt = txt.replace('skies', '').replace("mostly", "").strip()
        if "indoor" in txt:
            txt = "indoor"
        ans = 1
        if pd.isna(txt):
            return 0
        if 'partly' in txt:
            ans*=0.5
        if 'climate controlled' in txt or 'indoor' in txt:
            return ans*5
        if 'sunny' in txt or 'sun' in txt:
            return ans*3
        if 'clear' in txt:
            return ans
        if 'cloudy' in txt:
            return -ans
        if 'rain' in txt or 'rainy' in txt:
            return -3*ans
        if 'snow' in txt:
            return -5*ans
        return 0

    train['GameWeather_std'] = train['GameWeather'].apply(give_me_GameWeather)
    
    rain = ['Rainy', 'Rain Chance 40%', 'Showers','Cloudy with periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.',
        'Scattered Showers', 'Cloudy, Rain', 'Rain shower', 'Light Rain', 'Rain']

    overcast = ['Cloudy, light snow accumulating 1-3"', 'Party Cloudy', 'Cloudy, chance of rain',
                'Coudy', 'Cloudy, 50% change of rain', 'Rain likely, temps in low 40s.',
                'Cloudy and cold', 'Cloudy, fog started developing in 2nd quarter',
                'Partly Clouidy', '30% Chance of Rain', 'Mostly Coudy', 'Cloudy and Cool',
                'cloudy', 'Partly cloudy', 'Overcast', 'Hazy', 'Mostly cloudy', 'Mostly Cloudy',
                'Partly Cloudy', 'Cloudy']

    clear = ['Partly clear', 'Sunny and clear', 'Sun & clouds', 'Clear and Sunny',
            'Sunny and cold', 'Sunny Skies', 'Clear and Cool', 'Clear and sunny',
            'Sunny, highs to upper 80s', 'Mostly Sunny Skies', 'Cold',
            'Clear and warm', 'Sunny and warm', 'Clear and cold', 'Mostly sunny',
            'T: 51; H: 55; W: NW 10 mph', 'Clear Skies', 'Clear skies', 'Partly sunny',
            'Fair', 'Partly Sunny', 'Mostly Sunny', 'Clear', 'Sunny', 'Sunny, Windy']

    snow  = ['Heavy lake effect snow', 'Snow']

    none  = ['N/A Indoor', 'Indoors', 'Indoor', 'N/A (Indoors)', 'Controlled Climate']
    
    def clean_weather(x):
        if x in rain:
            return "rain"
        elif x in overcast:
            return "overcast"
        elif x in clear:
            return "clear"
        elif x in snow:
            return "snow"
        else:
            return "other"
    
    train['GameWeather'] = train['GameWeather'].apply(clean_weather)
    #train['GameWeather'] = train['GameWeather'].replace(rain,'rain')
    #train['GameWeather'] = train['GameWeather'].replace(overcast,'overcast')
    #train['GameWeather'] = train['GameWeather'].replace(clear,'clear')
    #train['GameWeather'] = train['GameWeather'].replace(snow,'snow')
    #train['GameWeather'] = train['GameWeather'].replace(none,'none')    
    
    #########################################################################################################
    
    # Turf
    
    #########################################################################################################   
    print("...    | Turf")
    
    
    def clean_turf(x):
        if "art" in x:
            return 0
        if "grass" in x:
            return 1
        if "natural" in x:
            return 1
        if "turf" in x:
            return 0
        else:
            return 0
        
    train["Turf_std"] = train["Turf"].apply(lambda x: clean_turf(x.lower()))
    
    #########################################################################################################
    
    # Stadium
    
    #########################################################################################################      
    print("...    | Stadium")
    
    
    stadium_replace = {
    "Twickenham Stadium":"Twickenham",
    "Los Angeles Memorial Coliesum":"Los Angeles Memorial Coliseum",
    "FirstEnergy Stadium":"FirstEnergyStadium",
    "FirstEnergy":"FirstEnergyStadium",
    "First Energy Stadium":"FirstEnergyStadium",
    "M&T Bank Stadium":"M & T Bank Stadium",
    "M&T Stadium": "M & T Bank Stadium",
    "Broncos Stadium at Mile High":"Broncos Stadium At Mile High",
    "Mercedes-Benz Dome":"Mercedes-Benz Stadium",
    "Mercedes-Benz Superdome": "Mercedes-Benz Stadium",
    "MetLife Stadium":"Metlife Stadium",
    "MetLife":"Metlife Stadium",
    }

    for key, value in stadium_replace.items():
        train["Stadium"] = train["Stadium"].replace(key, value)
        
        
    #########################################################################################################
    
    # Stadium Type
    
    #########################################################################################################           
    
    outdoor       = [
    'Outdoor', 'Outdoors', 'Cloudy', 'Heinz Field', 
    'Outdor', 'Ourdoor', 'Outside', 'Outddors', 
    'Outdoor Retr Roof-Open', 'Oudoor', 'Bowl'
    ]
    indoor_closed = [
    'Indoors', 'Indoor', 'Indoor, Roof Closed', 'Indoor, Roof Closed', 
    'Retractable Roof', 'Retr. Roof-Closed', 'Retr. Roof - Closed', 'Retr. Roof Closed',
    ]
    indoor_open   = ['Indoor, Open Roof', 'Open', 'Retr. Roof-Open', 'Retr. Roof - Open']
    dome_closed   = ['Dome', 'Domed, closed', 'Closed Dome', 'Domed', 'Dome, closed']
    dome_open     = ['Domed, Open', 'Domed, open']
    
    train['StadiumType'] = train['StadiumType'].replace(outdoor, "outdoor")
    train['StadiumType'] = train['StadiumType'].replace(indoor_closed, 'indoor closed')
    train['StadiumType'] = train['StadiumType'].replace(indoor_open, 'indoor open')
    train['StadiumType'] = train['StadiumType'].replace(dome_closed, 'dome closed')
    train['StadiumType'] = train['StadiumType'].replace(dome_open, 'dome open')
    
    #########################################################################################################
    
    # Time
    
    #########################################################################################################      
    
    def timesnap2day(x):
        days = x.split("-")
        return 365 * int(days[0]) + 30 * int(days[1]) + int(days[2][:2])
    
    train['MatchDay'] = train['TimeSnap'].apply(timesnap2day)    
    
    
    print("...    | Time")

    def strtoseconds(txt):
        txt = txt.split(':')
        return int(txt[0])*60 + int(txt[1]) + int(txt[2])/60
      
    train['GameClock_std'] = train['GameClock'].apply(strtoseconds)    
    
    

#     train['Birth_year'] = train['PlayerBirthDate'].apply(lambda x: int(x.split('/')[2]))

    #########################################################################################################
    
    # Height
    
    #########################################################################################################    
    train['Height_feet'] = train['PlayerHeight'].apply(lambda x: int(x.split('-')[0]))
    #train['PlayerHeight'] = train['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))
    
    
    
    # Return
    return train


# In[16]:


def remove_NAs(train):
    
    def na_categorical(x):
        try:
            return x.fillna(x.mode()[0])
        except:
            return x.fillna(method="ffill")

    train["OffenseFormation"] = train.OffenseFormation.transform(lambda x: na_categorical(x))
    
    
    train["StadiumType"] = train.groupby("Stadium")["StadiumType"].transform(lambda x: na_categorical(x))    
    
    train["DefendersInTheBox"] = train.groupby("DefensePersonnel")["DefendersInTheBox"].transform(lambda x: x.fillna(x.median()))
    
    try:    
        # Weather, Temperature
        train["GameWeather_std"] = train.groupby("Week")["GameWeather_std"].transform(lambda x: x.median())
    except:
        train["GameWeather_std"] = train["GameWeather_std"].transform(lambda x: x.fillna(x.median()))

    try:
        train["Temperature"] = train.groupby("Week")["Temperature"].transform(lambda x: x.fillna(x.mean()))    
    except:
        train['Temperature'] = train['Temperature'].fillna(train['Temperature'].median(), inplace=True)
    
    try:
        train["WindSpeed"] = train.groupby("Week")["WindSpeed"].transform(lambda x: x.fillna(x.mean()))
    except:
        train["WindSpeed"] = train["WindSpeed"].transform(lambda x: x.fillna(x.mean()))
        
    def wind_speed_closed(x):
        if "closed" in x[0]:
            return 0
        
    #train["WindSpeed"] = train[["StadiumType","WindSpeed"]].fillna("").apply(wind_speed_closed)
    train["StadiumClosed"] = train["StadiumType"].fillna("").apply(lambda x: "closed" in x)
    #train[train["StadiumClosed"]==1, "WindSpeed"] = 0

    # making sure
    train['Temperature'] = train['Temperature'].fillna(75)
    train['Humidity'] = train['Humidity'].fillna(65)
    train["WindSpeed"] = train["WindSpeed"].fillna(10)     
    
    
    
    return train


# In[17]:



def fix_sides(train):
  # https://www.kaggle.com/cpmpml/initial-wrangling-voronoi-areas-in-python
  
  """
  New features:
  
  TeamOnOffence:        home / away
  ToLeft :              binary
  IsBallCarrier:        binary
  
  YardLine_std
  X_std
  Y_std
  Orientation_std
  Dir_std
  """

  train['ToLeft'] = train.PlayDirection == "left"
  train['IsBallCarrier'] = train.NflId == train.NflIdRusher
  
  # fix team names

  
  train.loc[train.VisitorTeamAbbr == "ARI", 'VisitorTeamAbbr'] = "ARZ"
  train.loc[train.HomeTeamAbbr == "ARI", 'HomeTeamAbbr'] = "ARZ"

  train.loc[train.VisitorTeamAbbr == "BAL", 'VisitorTeamAbbr'] = "BLT"
  train.loc[train.HomeTeamAbbr == "BAL", 'HomeTeamAbbr'] = "BLT"

  train.loc[train.VisitorTeamAbbr == "CLE", 'VisitorTeamAbbr'] = "CLV"
  train.loc[train.HomeTeamAbbr == "CLE", 'HomeTeamAbbr'] = "CLV"

  train.loc[train.VisitorTeamAbbr == "HOU", 'VisitorTeamAbbr'] = "HST"
  train.loc[train.HomeTeamAbbr == "HOU", 'HomeTeamAbbr'] = "HST"

  
  '''
  Our ultimate goal will be to ensure that the offensive team (PossessionTeam) is moving left to right, 
  even if in the raw data, the offense is moving right to left.
  '''
  
  train['TeamOnOffense'] = "home"
  train.loc[train.PossessionTeam != train.HomeTeamAbbr, 'TeamOnOffense'] = "away"
  train['IsOnOffense'] = train.Team == train.TeamOnOffense # Is player on offense?
  train['YardLine_std'] = 100 - train.YardLine
  train.loc[train.FieldPosition.fillna('') == train.PossessionTeam,  
            'YardLine_std'
           ] = train.loc[train.FieldPosition.fillna('') == train.PossessionTeam,  
            'YardLine']
  train['X_std'] = train.X
  train.loc[train.ToLeft, 'X_std'] = 120 - train.loc[train.ToLeft, 'X'] 
  train['Y_std'] = train.Y
  train.loc[train.ToLeft, 'Y_std'] = 160/3 - train.loc[train.ToLeft, 'Y'] 
  train['Orientation_std'] = train.Orientation
  train.loc[train.ToLeft, 'Orientation_std'] = np.mod(180 + train.loc[train.ToLeft, 'Orientation_std'], 360)
  train['Dir_std'] = train.Dir
  train.loc[train.ToLeft, 'Dir_std'] = np.mod(180 + train.loc[train.ToLeft, 'Dir_std'], 360)
  train.loc[train.ToLeft, 'WindDirection_std'] = np.mod(180 + train.loc[train.ToLeft, 'WindDirection_std'], 360)

  
  return train


# In[18]:


def voronoi_feats(train):
    
    def shoelace(corners):
        n = len(corners) # of corners
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += corners[i][0] * corners[j][1]
            area -= corners[j][0] * corners[i][1]
        area = abs(area) / 2.0
        return area
    
    def get_voronoi(play_id, train):
        df = train[train.PlayId == play_id]
        xy = df[['X_std', 'Y_std']].values
        n_points = xy.shape[0]
        xy1 = xy.copy()
        xy1[:,1] = - xy[:,1]
        xy2 = xy.copy()
        xy2[:,1] = 320/3 - xy[:,1]
        xy3 = xy.copy()
        xy3[:,0] = 20 - xy[:,0]
        xy4 = xy.copy()
        xy4[:,0] = 220 - xy[:,0]
        xy = np.concatenate((xy, xy1, xy2, xy3, xy4), axis=0)
        offense = df.IsOnOffense.values
        vor = Voronoi(xy)
        areas = []
        #voronoi_plot_2d(vor, ax=ax, show_points=False, show_vertices=False)
        for r in range(n_points):
            region = vor.regions[vor.point_region[r]]
            if not -1 in region:
                polygon = [vor.vertices[i] for i in region]
                polygon_area = shoelace(polygon)
                areas.append(polygon_area)
        #areas = [int(a) for a in areas]
        fractions = [a/sum(areas) for a in areas]
        return areas, fractions
    
    plays = train.PlayId.unique()
    train["Voronoi"] = 0
    train["Voronoi_fraction"] = 0
    for play in plays:
        area, frac = get_voronoi(play, train)
        n = len(train[train.PlayId == play])
        while len(area)<n:
            area.append(0)
            frac.append(0)
        train.loc[train.PlayId == play,"Voronoi"] = area[:n]
        train.loc[train.PlayId == play,"Voronoi_fraction"] = frac[:n]
        
    return train


# In[ ]:





# In[ ]:





# In[19]:


def feature_engineering(train):
    
    """
    New features:
    
    OffenceHome
    OffenceLead
    YardsToGo
    BMI
    isRB
    Speed_and_Wind_dot
    DefendersInTheBox_vs_Distance
    DL
    LB
    DB
    RB
    TE
    WR
    Morning
    Afternoon
    Evening
    Hours
    YardsFromOwnGoal
    Snowing
    U
    W
    Uo
    Wo
    """
    
    # How many points is the ofence team ahead

    train['OffenseTeam'] = train['PossessionTeam']
    train['OffenseHome'] = train[['OffenseTeam','HomeTeamAbbr']].apply(lambda x: 1 if x[0] == x[1] else 0, axis = 1)
    train['DefenseTeam'] = train[['OffenseHome','HomeTeamAbbr','VisitorTeamAbbr']].apply(lambda x: x[2] if x[0] == 1 else x[1], axis = 1)
    train['OffenseLead'] = train[['OffenseHome','HomeScoreBeforePlay','VisitorScoreBeforePlay']].apply(lambda x: x[1]-x[2] if x[0] == 1 else x[2]-x[1], axis = 1)
    
    
    # Yards to go
    train['YardsToGo'] = train[['FieldPosition','OffenseTeam','YardLine']].apply(             lambda x: (50-x['YardLine'])+50 if x['OffenseTeam']==x['FieldPosition'] else x['YardLine'],1)
    
    # BMI
    train["BMI"] = train["Height_feet"]/train["PlayerWeight"]
    
    # Speed and WindSpeed vector dot product
    #train["Speed_and_Wind_dot"] = train[["S",'WindSpeed',"Dir_std","WindDirection_std"]].apply(lambda x: x[0]*x[1]*math.cos(x[2]-x[3]))
    def dotpr(x):
        return x[0]*x[1]*np.cos(x[2]-x[3])
    train["Speed_and_Wind_dot"] = train[["S",'WindSpeed',"Dir_std","WindDirection_std"]].apply(dotpr, axis=1)
    
    train["isRB"] = train["Position"].apply(lambda x: x == "RB")
    
    # dummified Offence Formation
    #off_form = train['OffenseFormation'].unique()
    #train = pd.concat([train.drop(['OffenseFormation'], axis=1), pd.get_dummies(train['OffenseFormation'], prefix='Formation')], axis=1)
    #dummy_col = train.columns
    
    train['DefendersInTheBox_vs_Distance'] = train['DefendersInTheBox'] / train['Distance']
    
    """
    # Defence and offence personnel
    train['DL'] = 0
    train['LB'] = 0
    train['DB'] = 0
    
    arr = [[int(s[0]) for s in t.split(", ")] for t in train["OffensePersonnel"]]
    train['RB'] = 0
    train['TE'] = 0
    train['WR'] = 0

    arr = [[int(s[0]) for s in t.split(", ")] for t in train["DefensePersonnel"]]
    train['DL'] = [a[0] for a in arr]
    train['LB'] = [a[1] for a in arr]
    train['DB'] = [a[2] for a in arr]
    
    arr = [[int(s[0]) for s in t.split(", ")] for t in train["OffensePersonnel"]]
    train['RB'] = [a[0] for a in arr]
    train['TE'] = [a[1] for a in arr]
    train['WR'] = [a[2] for a in arr]
    """
    # Time
    train['Morning'] = train['GameClock'].apply(lambda x : 1 if (int(x[0:2]) >=0 and int(x[0:2]) <12) else 0)
    train['Afternoon'] = train['GameClock'].apply(lambda x : 1 if (int(x[0:2]) <18 and int(x[0:2]) >=12) else 0)
    train['Evening'] = train['GameClock'].apply(lambda x : 1 if (int(x[0:2]) >= 18 and int(x[0:2]) < 24) else 0)
    train['Hours'] = train['GameClock'].apply(lambda x : int(x[0:2]))  
    
    # is this correct?
    train.loc[train.FieldPosition == train.PossessionTeam,'YardsFromOwnGoal'] = train.loc[train.FieldPosition == train.PossessionTeam,'YardLine']
    train.loc[train.FieldPosition != train.PossessionTeam,'YardsFromOwnGoal'] = 50 - train.loc[train.FieldPosition != train.PossessionTeam,'YardLine']
    
    train["Snowing"] = train["GameWeather"].apply(lambda x: x == "snow")
    
    
    train['U'] = train['S'] * 1.75 * np.cos(train['Dir_std'])+ train['X_std']
    train['W'] = train['S'] * 1.75 * np.sin(train['Dir_std']) + train['Y_std']
    
    # orientation vector
    train['Uo'] = train['S'] * 1.25 * np.cos(train['Orientation_std'])+ train['X_std']
    train['Wo'] = train['S'] * 1.25 * np.sin(train['Orientation_std']) + train['Y_std']
    
    train = train.drop(['OffenseTeam',"DefenseTeam"], axis=1)
    
    
    def projection_features(df):
        rad = 2 * np.pi * (90 - df[['Orientation_std']]) / 360
        v_0 = df['S'].values * np.cos(rad).values.reshape(-1)
        v_1 = np.sin(rad).values.reshape(-1)

        a_0 = df['A'].values * np.cos(rad).values.reshape(-1)
        a_1 = np.sin(rad)

        df['v_0'] = v_0
        df['v_1'] = v_1
        df['a_0'] = a_0
        df['a_1'] = a_1
        
        return df
    
    train = projection_features(train)
    
    def euclidean_distance(x1,y1,x2,y2):
        x_diff = (x1-x2)**2
        y_diff = (y1-y2)**2

        return np.sqrt(x_diff + y_diff)
    
    
    return train


# In[20]:


def create_features(df, deploy=False):
    def new_X(x_coordinate, play_direction):
        if play_direction == 'left':
            return 120.0 - x_coordinate
        else:
            return x_coordinate

    def new_line(rush_team, field_position, yardline):
        if rush_team == field_position:
            # offense starting at X = 0 plus the 10 yard endzone plus the line of scrimmage
            return 10.0 + yardline
        else:
            # half the field plus the yards between midfield and the line of scrimmage
            return 60.0 + (50 - yardline)

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

    def update_yardline(df):
        new_yardline = df[df['NflId'] == df['NflIdRusher']]
        new_yardline['YardLine'] = new_yardline[['PossessionTeam','FieldPosition','YardLine']].apply(lambda x: new_line(x[0],x[1],x[2]), axis=1)
        new_yardline = new_yardline[['GameId','PlayId','YardLine']]

        return new_yardline

    def update_orientation(df, yardline):
        df['X'] = df[['X','PlayDirection']].apply(lambda x: new_X(x[0],x[1]), axis=1)
        df['Orientation'] = df[['Orientation','PlayDirection']].apply(lambda x: new_orientation(x[0],x[1]), axis=1)
        df['Dir'] = df[['Dir','PlayDirection']].apply(lambda x: new_orientation(x[0],x[1]), axis=1)

        df = df.drop('YardLine', axis=1)
        df = pd.merge(df, yardline, on=['GameId','PlayId'], how='inner')

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

    def features_relative_to_back(df, carriers):
        player_distance = df[['GameId','PlayId','NflId','X','Y']]
        player_distance = pd.merge(player_distance, carriers, on=['GameId','PlayId'], how='inner')
        player_distance = player_distance[player_distance['NflId'] != player_distance['NflIdRusher']]
        player_distance['dist_to_back'] = player_distance[['X','Y','back_X','back_Y']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)

        player_distance = player_distance.groupby(['GameId','PlayId','back_from_scrimmage','back_oriented_down_field','back_moving_down_field'])                                         .agg({'dist_to_back':['min','max','mean','std']})                                         .reset_index()
        player_distance.columns = ['GameId','PlayId','back_from_scrimmage','back_oriented_down_field','back_moving_down_field',
                                   'min_dist','max_dist','mean_dist','std_dist']

        return player_distance

    def defense_features(df):
        rusher = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','Team','X','Y']]
        rusher.columns = ['GameId','PlayId','RusherTeam','RusherX','RusherY']

        defense = pd.merge(df,rusher,on=['GameId','PlayId'],how='inner')
        defense = defense[defense['Team'] != defense['RusherTeam']][['GameId','PlayId','X','Y','RusherX','RusherY']]
        defense['def_dist_to_back'] = defense[['X','Y','RusherX','RusherY']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)

        defense = defense.groupby(['GameId','PlayId'])                         .agg({'def_dist_to_back':['min','max','mean','std']})                         .reset_index()
        defense.columns = ['GameId','PlayId','def_min_dist','def_max_dist','def_mean_dist','def_std_dist']

        return defense

    def static_features(df, p_cols):
        
        
        add_new_feas = []

        ## Height
        df['PlayerHeight_dense'] = df['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))
        
        add_new_feas.append('PlayerHeight_dense')

        ## Time
        df['TimeHandoff'] = df['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
        df['TimeSnap'] = df['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))

        df['TimeDelta'] = df.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)
        df['PlayerBirthDate'] =df['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))

        ## Age
        seconds_in_year = 60*60*24*365.25
        df['PlayerAge'] = df.apply(lambda row: (row['TimeHandoff']-row['PlayerBirthDate']).total_seconds()/seconds_in_year, axis=1)
        add_new_feas.append('PlayerAge')

        ## WindSpeed
        #df['WindSpeed_ob'] = df['WindSpeed'].apply(lambda x: x.lower().replace('mph', '').strip() if not pd.isna(x) else x)
        #df['WindSpeed_ob'] = df['WindSpeed_ob'].apply(lambda x: (int(x.split('-')[0])+int(x.split('-')[1]))/2 if not pd.isna(x) and '-' in x else x)
        #df['WindSpeed_ob'] = df['WindSpeed_ob'].apply(lambda x: (int(x.split()[0])+int(x.split()[-1]))/2 if not pd.isna(x) and type(x)!=float and 'gusts up to' in x else x)
        #df['WindSpeed_dense'] = df['WindSpeed_ob'].apply(strtofloat)
        #add_new_feas.append('WindSpeed_dense')

        ## Weather
        df['GameWeather_process'] = df['GameWeather'].str.lower()
        df['GameWeather_process'] = df['GameWeather_process'].apply(lambda x: "indoor" if not pd.isna(x) and "indoor" in x else x)
        df['GameWeather_process'] = df['GameWeather_process'].apply(lambda x: x.replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party', 'partly') if not pd.isna(x) else x)
        df['GameWeather_process'] = df['GameWeather_process'].apply(lambda x: x.replace('clear and sunny', 'sunny and clear') if not pd.isna(x) else x)
        df['GameWeather_process'] = df['GameWeather_process'].apply(lambda x: x.replace('skies', '').replace("mostly", "").strip() if not pd.isna(x) else x)
        df['GameWeather_dense'] = df['GameWeather_process'].apply(map_weather)
        add_new_feas.append('GameWeather_dense')
#         ## Rusher
#         train['IsRusher'] = (train['NflId'] == train['NflIdRusher'])
#         train['IsRusher_ob'] = (train['NflId'] == train['NflIdRusher']).astype("object")
#         temp = train[train["IsRusher"]][["Team", "PlayId"]].rename(columns={"Team":"RusherTeam"})
#         train = train.merge(temp, on = "PlayId")
#         train["IsRusherTeam"] = train["Team"] == train["RusherTeam"]

        ## dense -> categorical
#         train["Quarter_ob"] = train["Quarter"].astype("object")
#         train["Down_ob"] = train["Down"].astype("object")
#         train["JerseyNumber_ob"] = train["JerseyNumber"].astype("object")
#         train["YardLine_ob"] = train["YardLine"].astype("object")
        # train["DefendersInTheBox_ob"] = train["DefendersInTheBox"].astype("object")
        # train["Week_ob"] = train["Week"].astype("object")
        # train["TimeDelta_ob"] = train["TimeDelta"].astype("object")


        ## Orientation and Dir
        df["Orientation_ob"] = df["Orientation"].apply(lambda x : orientation_to_cat(x)).astype("object")
        df["Dir_ob"] = df["Dir"].apply(lambda x : orientation_to_cat(x)).astype("object")

        df["Orientation_sin"] = df["Orientation"].apply(lambda x : np.sin(x/360 * 2 * np.pi))
        df["Orientation_cos"] = df["Orientation"].apply(lambda x : np.cos(x/360 * 2 * np.pi))
        add_new_feas.append("Orientation_cos")
        add_new_feas.append("Orientation_sin")
        
        
        df["Dir_sin"] = df["Dir"].apply(lambda x : np.sin(x/360 * 2 * np.pi))
        df["Dir_cos"] = df["Dir"].apply(lambda x : np.cos(x/360 * 2 * np.pi))
        add_new_feas.append("Dir_sin")
        add_new_feas.append("Dir_cos")
        
        df["WindDir_sin"] = df["WindDirection_std"].apply(lambda x : np.sin(x/360 * 2 * np.pi))
        df["WindDir_cos"] = df["WindDirection_std"].apply(lambda x : np.cos(x/360 * 2 * np.pi))
        add_new_feas.append("WindDir_sin")
        add_new_feas.append("WindDir_cos")

        # Player force: F=m*a
        
        df["Force"] = df["PlayerWeight"]*df["A"]

        ## diff Score
        df["diffScoreBeforePlay"] = df["HomeScoreBeforePlay"] - df["VisitorScoreBeforePlay"]
        add_new_feas.append("diffScoreBeforePlay")
        
        # features from Artem
        add_new_feas.append("TimeDelta")
        
        fe = ['v_0', 'v_1', 
            #'a_0', 'a_1', 
            #"YardsToGo", 
              "BMI", 
            "Force",
            #"Snowing", 
              #"RB", "DL", "LB", "DB", "TE", "WR", 
              #"U", "W", "Uo", "Wo", 
            "Speed_and_Wind_dot",
            "Voronoi", "Voronoi_fraction",
            #"IsOnOffense"]
               "play_control"]
        
        def_f = [    'secondsLeftInHalf',
    'numericFormation',
    "Defense_X_stdiv", 
    "Defense_Y_stdiv", 
    "Defense_Y_spread", 
    "Defense_X_spread", 
    "min_time_to_tackle",
    "mean_time_to_tackle",
    "dist_to_offense_centroid",
    "dist_to_defense_centroid"]
    
        static_features = df[df['NflId'] == df['NflIdRusher']][add_new_feas+fe+p_cols + def_f +['GameId','PlayId','X','Y','S','A','Dis','Orientation','Dir',
                                                            'YardLine','Quarter','Down','Distance','DefendersInTheBox']].drop_duplicates()
#         static_features['DefendersInTheBox'] = static_features['DefendersInTheBox'].fillna(np.mean(static_features['DefendersInTheBox']))
        static_features.fillna(-999,inplace=True)
#         for i in add_new_feas:
#             static_features[i] = static_features[i].fillna(np.mean(static_features[i]))
            

        return static_features

    def split_personnel(s):
        splits = s.split(',')
        for i in range(len(splits)):
            splits[i] = splits[i].strip()

        return splits

    def defense_formation(l):
        dl = 0
        lb = 0
        db = 0
        other = 0

        for position in l:
            sub_string = position.split(' ')
            if sub_string[1] == 'DL':
                dl += int(sub_string[0])
            elif sub_string[1] in ['LB','OL']:
                lb += int(sub_string[0])
            else:
                db += int(sub_string[0])

        counts = (dl,lb,db,other)

        return counts

    def offense_formation(l):
        qb = 0
        rb = 0
        wr = 0
        te = 0
        ol = 0

        sub_total = 0
        qb_listed = False
        for position in l:
            sub_string = position.split(' ')
            pos = sub_string[1]
            cnt = int(sub_string[0])

            if pos == 'QB':
                qb += cnt
                sub_total += cnt
                qb_listed = True
            # Assuming LB is a line backer lined up as full back
            elif pos in ['RB','LB']:
                rb += cnt
                sub_total += cnt
            # Assuming DB is a defensive back and lined up as WR
            elif pos in ['WR','DB']:
                wr += cnt
                sub_total += cnt
            elif pos == 'TE':
                te += cnt
                sub_total += cnt
            # Assuming DL is a defensive lineman lined up as an additional line man
            else:
                ol += cnt
                sub_total += cnt

        # If not all 11 players were noted at given positions we need to make some assumptions
        # I will assume if a QB is not listed then there was 1 QB on the play
        # If a QB is listed then I'm going to assume the rest of the positions are at OL
        # This might be flawed but it looks like RB, TE and WR are always listed in the personnel
        if sub_total < 11:
            diff = 11 - sub_total
            if not qb_listed:
                qb += 1
                diff -= 1
            ol += diff

        counts = (qb,rb,wr,te,ol)

        return counts
    
    def personnel_features(df):
        personnel = df[['GameId','PlayId','OffensePersonnel','DefensePersonnel']].drop_duplicates()
        personnel['DefensePersonnel'] = personnel['DefensePersonnel'].apply(lambda x: split_personnel(x))
        personnel['DefensePersonnel'] = personnel['DefensePersonnel'].apply(lambda x: defense_formation(x))
        personnel['num_DL'] = personnel['DefensePersonnel'].apply(lambda x: x[0])
        personnel['num_LB'] = personnel['DefensePersonnel'].apply(lambda x: x[1])
        personnel['num_DB'] = personnel['DefensePersonnel'].apply(lambda x: x[2])

        personnel['OffensePersonnel'] = personnel['OffensePersonnel'].apply(lambda x: split_personnel(x))
        personnel['OffensePersonnel'] = personnel['OffensePersonnel'].apply(lambda x: offense_formation(x))
        personnel['num_QB'] = personnel['OffensePersonnel'].apply(lambda x: x[0])
        personnel['num_RB'] = personnel['OffensePersonnel'].apply(lambda x: x[1])
        personnel['num_WR'] = personnel['OffensePersonnel'].apply(lambda x: x[2])
        personnel['num_TE'] = personnel['OffensePersonnel'].apply(lambda x: x[3])
        personnel['num_OL'] = personnel['OffensePersonnel'].apply(lambda x: x[4])

        # Let's create some features to specify if the OL is covered
        personnel['OL_diff'] = personnel['num_OL'] - personnel['num_DL']
        personnel['OL_TE_diff'] = (personnel['num_OL'] + personnel['num_TE']) - personnel['num_DL']
        # Let's create a feature to specify if the defense is preventing the run
        # Let's just assume 7 or more DL and LB is run prevention
        personnel['run_def'] = (personnel['num_DL'] + personnel['num_LB'] > 6).astype(int)

        personnel.drop(['OffensePersonnel','DefensePersonnel'], axis=1, inplace=True)
        
        return personnel
    
    def rusher_features(df):
        
        rusher = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','Dir', 'S', 'A', 'X', 'Y']]
        rusher.columns = ['GameId','PlayId', 'RusherDir', 'RusherS', 'RusherA', 'RusherX', 'RusherY']
        
       
        radian_angle = (90 - rusher['RusherDir']) * np.pi / 180.0
        v_horizontal = np.abs(rusher['RusherS'] * np.cos(radian_angle))
        v_vertical = np.abs(rusher['RusherS'] * np.sin(radian_angle)) 
        
        #rusher["dist_to_rusher"] = df
       
        rusher['v_horizontal'] = v_horizontal
        rusher['v_vertical'] = v_vertical
        
        
        rusher.columns = ['GameId','PlayId', 'RusherDir', 
                          'RusherS',
                          'RusherA','RusherX', 'RusherY','v_horizontal', 'v_vertical']
        
        
        return rusher
    
    def dist_to_rusher(x):
        a = (x[0],x[1])
        b = (x[2],x[3])
        return euclidean(a,b)

    def combine_features(relative_to_back, defense,rushing, static, deploy=deploy):
        df = pd.merge(relative_to_back,defense,on=['GameId','PlayId'],how='inner')
        df = pd.merge(df,rushing,on=['GameId','PlayId'],how='inner')
        df = pd.merge(df,static,on=['GameId','PlayId'],how='inner')
        #df = pd.merge(df,pc,how='left')


        if not deploy:
            df = pd.merge(df, outcomes, on=['GameId','PlayId'], how='inner')

        return df
    
    #print("Cleaning features...")
    df = clean_features(df)
    
    #print("Removing NAs...")
    #df = remove_NAs(df)
    
    #print("Feature Engineering...")
    df = fix_sides(df)
    df = feature_engineering(df)
    
    #print("Calculating Voronoi Areas")
    df = voronoi_feats(df)
    
    #print("Calculating Play Control")
    pc = get_play_control(df)    
    
    #df["influence"] = pc["influence"].values
    df["play_control"] = pc["play_control"].values
    
    for col in [    'secondsLeftInHalf',
    'numericFormation',
    "Defense_X_stdiv", 
    "Defense_Y_stdiv", 
    "Defense_Y_spread", 
    "Defense_X_spread", 
    "min_time_to_tackle",
    "mean_time_to_tackle",
    "dist_to_offense_centroid",
    "dist_to_defense_centroid"]:
        df[col] = pc[col].values
    
    # positional features
    pf, p_cols = positional(df)
    
    for col in p_cols:
        df[col] = pf[col].values
    
    #print("Yardline features...")
    yardline = update_yardline(df)
    #print("Fixing orientation...")
    df = update_orientation(df, yardline)
    #print("Backline features...")
    back_feats = back_features(df)
    #print("Features relative to backline...")
    rel_back = features_relative_to_back(df, back_feats)
    #print("Defence features...")
    def_feats = defense_features(df)
    #print("Static features...")
    static_feats = static_features(df, p_cols)
    #print("Rusher features...")
    rush_feats = rusher_features(df)
    #print("Personnel features...")
    personnel = personnel_features(df)
    #basetable = combine_features(rel_back, def_feats, static_feats, deploy=deploy)
    basetable = combine_features(rel_back, def_feats,rush_feats,static_feats, deploy=deploy)
    
    # some more rusher based features
    #basetable["distance_to_rusher"] = 0
    
    #basetable["distance_to_rusher"] = basetable[["X","Y","RusherX","RusherY"]].apply(lambda x: dist_to_rusher(x), axis=1)
    basetable["s_delta_0"] = basetable["v_0"]-basetable['v_horizontal']
    basetable["s_delta_1"] = basetable["v_1"]-basetable['v_vertical']
    
    #basetable["influence"] = pc["influence"].values
    #basetable["control"] = pc["play_control"].values
    
    basetable.drop(["v_0","v_1","X","Y","S","A", "Force"],axis=1, inplace=True)
    
    #print("Complete.")
    
    return basetable


# In[21]:


get_ipython().run_line_magic('time', 'train_basetable = create_features(train, False)')


# In[22]:


train_basetable.head()


# In[23]:


X = train_basetable.copy()
yards = X.Yards

y = np.zeros((yards.shape[0], 199))
for idx, target in enumerate(list(yards)):
    y[idx][99 + target] = 1

X.drop(['GameId','PlayId','Yards'], axis=1, inplace=True)


# In[24]:


sns.set(rc={'figure.figsize':(25, 25)})
corr = X.corr()
plt.figure() 
ax = sns.heatmap(corr, linewidths=.5, annot=True, cmap="YlGnBu", fmt='.1g')
plt.show()


# In[25]:


# Drop highly correlated features (37->28)
columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.99:
            if columns[j]:
                columns[j] = False

feature_columns = X.columns.values
drop_columns = X.columns[columns == False].values
print(feature_columns)
print(drop_columns)


# In[26]:


scaler = StandardScaler()
X = scaler.fit_transform(X[feature_columns])


# In[27]:


"""scaler = StandardScaler()
X = scaler.fit_transform(X)"""


# In[28]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=12345)


# In[29]:


print(X_train.shape, X_val.shape)
print(y_train.shape, y_val.shape)


# In[30]:


def crps_score(y_prediction, y_valid, shape=X.shape[0]):
    y_true = np.clip(np.cumsum(y_valid, axis=1), 0, 1)
    y_pred = np.clip(np.cumsum(y_prediction, axis=1), 0, 1)
    val_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * shape)
    crps = np.round(val_s, 6)
    
    return crps


# In[31]:


from keras.layers import Dense,Input,Flatten,concatenate,Dropout,Lambda, LeakyReLU
from keras.models import Model
import keras.backend as K
import re
from keras.losses import binary_crossentropy
from  keras.callbacks import EarlyStopping,ModelCheckpoint
import codecs

from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from sklearn.metrics import f1_score




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
    


# In[32]:


def get_model(x_tr,y_tr,x_val,y_val):
    inp = Input(shape = (x_tr.shape[1],))
    x = Dense(1024, input_dim=X.shape[1])(inp)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    
    out = Dense(199, activation='softmax')(x)
    model = Model(inp,out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[])
    #add lookahead
#     lookahead = Lookahead(k=5, alpha=0.5) # Initialize Lookahead
#     lookahead.inject(model) # add into model

    
    es = EarlyStopping(monitor='CRPS_score_val', 
                       mode='min',
                       restore_best_weights=True, 
                       verbose=1, 
                       patience=20)

    mc = ModelCheckpoint('best_model.h5',monitor='CRPS_score_val',mode='min',
                                   save_best_only=True, verbose=1, save_weights_only=True)
    
    bsz = 1024
    steps = x_tr.shape[0]/bsz
    


    model.fit(x_tr, y_tr,callbacks=[CRPSCallback(validation = (x_val,y_val)),es,mc], epochs=100, batch_size=bsz,verbose=1)
    model.load_weights("best_model.h5")
    
    y_pred = model.predict(x_val)
    y_valid = y_val
    y_true = np.clip(np.cumsum(y_valid, axis=1), 0, 1)
    y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
    val_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * x_val.shape[0])
    crps = np.round(val_s, 6)

    return model,crps


# In[33]:


"""def get_rf(x_tr, y_tr, x_val, y_val, shape):
    model = RandomForestRegressor(bootstrap=False, max_features=0.3, min_samples_leaf=15, 
                                  min_samples_split=8, n_estimators=50, n_jobs=-1, random_state=42)
    model.fit(x_tr, y_tr)
    
    y_pred = model.predict(x_val)
    y_valid = y_val
    crps = crps_score(y_pred, y_valid, shape=shape)
    
    return model, crps"""


# In[34]:


from sklearn.model_selection import train_test_split, KFold
import time

losses = []
models = []
crps_csv = []

s_time = time.time()


for k in range(2):
    kfold = KFold(5, random_state = 42 + k, shuffle = True)
    for k_fold, (tr_inds, val_inds) in enumerate(kfold.split(yards)):
        print("-----------")
        print("-----------")
        tr_x,tr_y = X[tr_inds],y[tr_inds]
        val_x,val_y = X[val_inds],y[val_inds]
        model,crps = get_model(tr_x,tr_y,val_x,val_y)
        models.append(model)
        print("the %d fold crps is %f"%((k_fold+1),crps))
        crps_csv.append(crps)
 
print("mean crps is %f"%np.mean(crps_csv))


def predict(x_te):
    model_num = len(models)
    for k,m in enumerate(models):
        if k==0:
            y_pred = m.predict(x_te,batch_size=1024)
        else:
            y_pred+=m.predict(x_te,batch_size=1024)
            
    y_pred = y_pred / model_num
    
    return y_pred
            
        


# In[35]:


"""loop = 2
fold = 5

oof_nn = np.zeros([loop, y.shape[0], y.shape[1]])
oof_rf = np.zeros([loop, y.shape[0], y.shape[1]])

models_nn = []
crps_csv_nn = []
models_rf = []
crps_csv_rf = []

feature_importance = np.zeros([loop, fold, X.shape[1]])

s_time = time.time()

for k in range(loop):
    kfold = KFold(fold, random_state = 42 + k, shuffle = True)
    for k_fold, (tr_inds, val_inds) in enumerate(kfold.split(yards)):
        print("-----------")
        print(f'Loop {k+1}/{loop}' + f' Fold {k_fold+1}/{fold}')
        print("-----------")
        tr_x, tr_y = X[tr_inds], y[tr_inds]
        val_x, val_y = X[val_inds], y[val_inds]
        
        # Train NN
        nn, crps_nn = get_nn(tr_x, tr_y, val_x, val_y, shape=val_x.shape[0])
        models_nn.append(nn)
        print("the %d fold crps (NN) is %f"%((k_fold+1), crps_nn))
        crps_csv_nn.append(crps_nn)
        
        # Train RF
        rf, crps_rf = get_rf(tr_x, tr_y, val_x, val_y, shape=val_x.shape[0])
        models_rf.append(rf)
        print("the %d fold crps (RF) is %f"%((k_fold+1), crps_rf))
        crps_csv_rf.append(crps_rf)
        
        # Feature Importance
        feature_importance[k, k_fold, :] = rf.feature_importances_
        
        #Predict OOF
        oof_nn[k, val_inds, :] = nn.predict(val_x)
        oof_rf[k, val_inds, :] = rf.predict(val_x)"""


# In[36]:


def get_model(x_tr,y_tr,x_val,y_val):
    inp = Input(shape = (x_tr.shape[1],))
    x = Dense(1024, input_dim=X.shape[1])(inp)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    
    out = Dense(199, activation='softmax')(x)
    model = Model(inp,out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[])
    #add lookahead
#     lookahead = Lookahead(k=5, alpha=0.5) # Initialize Lookahead
#     lookahead.inject(model) # add into model

    
    es = EarlyStopping(monitor='CRPS_score_val', 
                       mode='min',
                       restore_best_weights=True, 
                       verbose=1, 
                       patience=10)

    mc = ModelCheckpoint('best_model1.h5',monitor='CRPS_score_val',mode='min',
                                   save_best_only=True, verbose=1, save_weights_only=True)
    
    bsz = 1024
    steps = x_tr.shape[0]/bsz
    


    model.fit(x_tr, y_tr,callbacks=[CRPSCallback(validation = (x_val,y_val)),es,mc], epochs=100, batch_size=bsz,verbose=1)
    model.load_weights("best_model1.h5")
    
    y_pred = model.predict(x_val)
    y_valid = y_val
    y_true = np.clip(np.cumsum(y_valid, axis=1), 0, 1)
    y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
    val_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * x_val.shape[0])
    crps = np.round(val_s, 6)

    return model,crps


# In[37]:


from sklearn.model_selection import train_test_split, KFold
import time

s_time = time.time()


for k in range(1):
    kfold = KFold(5, random_state = 42 + k, shuffle = True)
    for k_fold, (tr_inds, val_inds) in enumerate(kfold.split(yards)):
        print("-----------")
        print("-----------")
        tr_x,tr_y = X[tr_inds],y[tr_inds]
        val_x,val_y = X[val_inds],y[val_inds]
        model,crps = get_model(tr_x,tr_y,val_x,val_y)
        models.append(model)
        print("the %d fold crps is %f"%((k_fold+1),crps))
        crps_csv.append(crps)
 
print("mean crps is %f"%np.mean(crps_csv))


def predict(x_te):
    model_num = len(models)
    for k,m in enumerate(models):
        if k==0:
            y_pred = m.predict(x_te,batch_size=1024)
        else:
            y_pred+=m.predict(x_te,batch_size=1024)
            
    y_pred = y_pred / model_num
    
    return y_pred
            
        


# In[38]:


def get_model(x_tr,y_tr,x_val,y_val):
    inp = Input(shape = (x_tr.shape[1],))
    x = Dense(512, input_dim=X.shape[1])(inp)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    
    out = Dense(199, activation='softmax')(x)
    model = Model(inp,out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[])
    #add lookahead
#     lookahead = Lookahead(k=5, alpha=0.5) # Initialize Lookahead
#     lookahead.inject(model) # add into model

    
    es = EarlyStopping(monitor='CRPS_score_val', 
                       mode='min',
                       restore_best_weights=True, 
                       verbose=1, 
                       patience=10)

    mc = ModelCheckpoint('best_model2.h5',monitor='CRPS_score_val',mode='min',
                                   save_best_only=True, verbose=1, save_weights_only=True)
    
    bsz = 1024
    steps = x_tr.shape[0]/bsz
    


    model.fit(x_tr, y_tr,callbacks=[CRPSCallback(validation = (x_val,y_val)),es,mc], epochs=100, batch_size=bsz,verbose=1)
    model.load_weights("best_model2.h5")
    
    y_pred = model.predict(x_val)
    y_valid = y_val
    y_true = np.clip(np.cumsum(y_valid, axis=1), 0, 1)
    y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
    val_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * x_val.shape[0])
    crps = np.round(val_s, 6)

    return model,crps


# In[39]:


from sklearn.model_selection import train_test_split, KFold
import time

s_time = time.time()


for k in range(1):
    kfold = KFold(5, random_state = 42 + k, shuffle = True)
    for k_fold, (tr_inds, val_inds) in enumerate(kfold.split(yards)):
        print("-----------")
        print("-----------")
        tr_x,tr_y = X[tr_inds],y[tr_inds]
        val_x,val_y = X[val_inds],y[val_inds]
        model,crps = get_model(tr_x,tr_y,val_x,val_y)
        models.append(model)
        print("the %d fold crps is %f"%((k_fold+1),crps))
        crps_csv.append(crps)
 
print("mean crps is %f"%np.mean(crps_csv))


def predict(x_te):
    model_num = len(models)
    for k,m in enumerate(models):
        if k==0:
            y_pred = m.predict(x_te,batch_size=1024)
        else:
            y_pred+=m.predict(x_te,batch_size=1024)
            
    y_pred = y_pred / model_num
    
    return y_pred
            
        


# In[40]:


"""print("mean crps is %f"%np.mean(crps_csv))"""


# In[41]:


"""crps_oof_nn = []
crps_oof_rf = []

for k in range(loop):
    crps_oof_nn.append(crps_score(oof_nn[k,...], y))
    crps_oof_rf.append(crps_score(oof_rf[k,...], y))"""


# In[42]:


"""feature_importances = pd.DataFrame(np.mean(feature_importance, axis=0).T, columns=[[f'fold_{fold_n}' for fold_n in range(fold)]])
feature_importances['feature'] = feature_columns
feature_importances['average'] = feature_importances[[f'fold_{fold_n}' for fold_n in range(fold)]].mean(axis=1)
feature_importances.sort_values(by=('average',), ascending=False).head(10)"""


# In[43]:


"""feature_importance_flatten = pd.DataFrame()
for i in range(len(feature_importances.columns)-2):
    col = ['feature', feature_importances.columns.values[i][0]]
    feature_importance_flatten = pd.concat([feature_importance_flatten, feature_importances[col].rename(columns={f'fold_{i}': 'importance'})], axis=0)

plt.figure(figsize=(16, 16))
sns.barplot(data=feature_importance_flatten.sort_values(by=('importance',), ascending=False), x=('importance',), y=('feature',))
plt.title(f'Feature Importances over {loop} loops and {fold} folds')
plt.show()"""


# In[44]:


"""def weight_opt(oof_nn, oof_rf, y_true):
    weight_nn = np.inf
    best_crps = np.inf
    
    for i in np.arange(0, 1.01, 0.05):
        crps_blend = np.zeros(oof_nn.shape[0])
        for k in range(oof_nn.shape[0]):
            crps_blend[k] = crps_score(i * oof_nn[k,...] + (1-i) * oof_rf[k,...], y_true)
        if np.mean(crps_blend) < best_crps:
            best_crps = np.mean(crps_blend)
            weight_nn = round(i, 2)
            
        print(str(round(i, 2)) + ' : mean crps (Blend) is ', round(np.mean(crps_blend), 6))
        
    print('-'*36)
    print('Best weight for NN: ', weight_nn)
    print('Best weight for RF: ', round(1-weight_nn, 2))
    print('Best mean crps (Blend): ', round(best_crps, 6))
    
    return weight_nn, round(1-weight_nn, 2)"""


# In[45]:


"""weight_nn, weight_rf = weight_opt(oof_nn, oof_rf, y)"""


# In[46]:


"""def predict(x_te, models_nn, models_rf, weight_nn, weight_rf):
    model_num_nn = len(models_nn)
    model_num_rf = len(models_rf)
    for k,m in enumerate(models_nn):
        if k==0:
            y_pred_nn = m.predict(x_te, batch_size=1024)
            y_pred_rf = models_rf[k].predict(x_te)
        else:
            y_pred_nn += m.predict(x_te, batch_size=1024)
            y_pred_rf += models_rf[k].predict(x_te)
            
    y_pred_nn = y_pred_nn / model_num_nn
    y_pred_rf = y_pred_rf / model_num_rf
    
    return weight_nn * y_pred_nn + weight_rf * y_pred_rf"""


# In[47]:


"""%%time
if  TRAIN_OFFLINE==False:
    from kaggle.competitions import nflrush
    env = nflrush.make_env()
    iter_test = env.iter_test()

    for (test_df, sample_prediction_df) in iter_test:
        basetable = create_features(test_df, deploy=True)
        basetable.drop(['GameId','PlayId'], axis=1, inplace=True)
        
        scaled_basetable = scaler.transform(basetable[feature_columns])

        y_pred = predict(scaled_basetable, models_nn, models_rf, weight_nn, weight_rf)
        y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1).tolist()[0]

        preds_df = pd.DataFrame(data=[y_pred], columns=sample_prediction_df.columns)
        env.predict(preds_df)

    env.write_submission_file()"""


# In[48]:


get_ipython().run_cell_magic('time', '', "if  TRAIN_OFFLINE==False:\n    from kaggle.competitions import nflrush\n    env = nflrush.make_env()\n    iter_test = env.iter_test()\n\n    for (test_df, sample_prediction_df) in iter_test:\n        basetable = create_features(test_df, deploy=True)\n        basetable.drop(['GameId','PlayId'], axis=1, inplace=True)\n        scaled_basetable = scaler.transform(basetable)\n\n        y_pred = predict(scaled_basetable)\n        y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1).tolist()[0]\n\n        preds_df = pd.DataFrame(data=[y_pred], columns=sample_prediction_df.columns)\n        env.predict(preds_df)\n\n    env.write_submission_file()")

