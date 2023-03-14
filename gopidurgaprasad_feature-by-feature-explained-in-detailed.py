#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from string import punctuation
import datetime
import re




from IPython.display import Image
import os
get_ipython().system('ls ../input/')




train_df = pd.read_csv("../input/nfl-big-data-bowl-2020/train.csv", low_memory=False)
train_df.head()




# https://www.kaggle.com/kabure/extensive-eda-and-modeling-xgb-hyperopt

def resumetable(df):
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values    
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values

    for name in summary['Name'].value_counts().index:
        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 

    return summary

## Function to reduce the DF size
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df




resumetable(train_df)[:]




## Reducting memory
train_df = reduce_mem_usage(train_df)




train_df.shape




print("Total number of games : ",train_df.GameId.nunique())
print("Total number of HandOff's : ",train_df.PlayId.nunique())
print("Total number of uniq players : ", train_df.NflId.nunique())




### Checking Train data sorted by PlayId and Team.

# https://www.kaggle.com/hukuda222/nfl-simple-model-using-lightgbm

ok = True
for i in range(0,509762,22):
    p=train_df["PlayId"][i]
    for j in range(1,22):
        if(p!=train_df["PlayId"][i+j]):
            ok=False
            break
print("train data is sorted by PlayId." if ok else "train data is not sorted by PlayId.")
ok = True
for i in range(0,509762,11):
    p=train_df["Team"][i]
    for j in range(1,11):
        if(p!=train_df["Team"][i+j]):
            ok=False
            break
            
print("train data is sorted by Team." if ok else "train data is not sorted by Team.")




print("Total number of games : ",train_df.GameId.nunique())
print("Average number of HandOff's in every game : ", train_df.GameId.value_counts().mean())
print("Max number of HandOff's in one game : ", train_df.GameId.value_counts().max())
print("Min number of HandOff's in one game : ", train_df.GameId.value_counts().min())




print("Total number of HandOff's : ",train_df.PlayId.nunique())
print("Every HandOff have ", int(train_df.PlayId.value_counts().mean()), "Players Data")




playId_groupby = train_df.groupby("PlayId")




print("Total number of Teams :", train_df.Team.value_counts())
print("Every PlayId have ", playId_groupby["Team"].value_counts().max() , "players from each category")




plt.figure()
sns.countplot(train_df["Team"])
plt.title("Away and Home team countplot")
plt.show()




print("Total number of positions of X : ", train_df.X.shape[0])
print("Total number of positions of Y : ", train_df.Y.shape[0])
print("*"*50)
print("max of X : ", train_df.X.max())
print("max of Y : ", train_df.Y.max())
print("*"*50)
print("min of X : ", train_df.X.min())
print("min of Y : ", train_df.Y.min())
print("*"*50)
print("mean of X : ", train_df.X.values.mean())
print("mean of Y : ", train_df.Y.values.mean())




plt.figure(figsize=(16,6))
plt.subplot(121)
sns.distplot(train_df.X)
plt.vlines(train_df.X.values.mean(), plt.ylim()[0], plt.ylim()[1], color='r', linestyles='--');
plt.text(train_df.X.values.mean()-8, plt.ylim()[1]-0.001, "Mean of X", size=15, color='r')
plt.title("X axis Distribution")
plt.subplot(122)
sns.distplot(train_df.Y)
plt.vlines(train_df.Y.values.mean(), plt.ylim()[0], plt.ylim()[1], color='r', linestyles='--');
plt.title("Y axis Distribution")
plt.text(train_df.Y.values.mean()-8, plt.ylim()[1]-0.003, "Mean of Y", size=15, color='r')




plt.figure(figsize=(16,12))
sns.scatterplot(train_df["X"], train_df["Y"])
plt.xlabel('X axis', fontsize=12)
plt.ylabel('Y axis', fontsize=12)
plt.title("Players positions", fontsize=20)
plt.show()




# https://www.kaggle.com/robikscube/nfl-big-data-bowl-plotting-player-position

def create_football_field(linenumbers=True,
                          endzones=True,
                          highlight_line=False,
                          highlight_line_number=50,
                          highlighted_name='Line of Scrimmage',
                          fifty_is_los=False,
                          figsize=(12*2, 6.33*2)):
    """
    Function that plots the football field for viewing plays.
    Allows for showing or hiding endzones.
    """
    rect = patches.Rectangle((0, 0), 120, 53.3, linewidth=0.1,
                             edgecolor='r', facecolor='darkgreen', zorder=0)

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.add_patch(rect)

    plt.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
              80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
             [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
              53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
             color='white')
    if fifty_is_los:
        plt.plot([60, 60], [0, 53.3], color='gold')
        plt.text(62, 50, '<- Player Yardline at Snap', color='gold')
    # Endzones
    if endzones:
        ez1 = patches.Rectangle((0, 0), 10, 53.3,
                                linewidth=0.1,
                                edgecolor='r',
                                facecolor='blue',
                                alpha=0.2,
                                zorder=0)
        ez2 = patches.Rectangle((110, 0), 120, 53.3,
                                linewidth=0.1,
                                edgecolor='r',
                                facecolor='blue',
                                alpha=0.2,
                                zorder=0)
        ax.add_patch(ez1)
        ax.add_patch(ez2)
    plt.xlim(0, 120)
    plt.ylim(-5, 58.3)
    plt.axis('off')
    if linenumbers:
        for x in range(20, 110, 10):
            numb = x
            if x > 50:
                numb = 120 - x
            plt.text(x, 5, str(numb - 10),
                     horizontalalignment='center',
                     fontsize=20,  # fontname='Arial',
                     color='white')
            plt.text(x - 0.95, 53.3 - 5, str(numb - 10),
                     horizontalalignment='center',
                     fontsize=20,  # fontname='Arial',
                     color='white', rotation=180)
    if endzones:
        hash_range = range(11, 110)
    else:
        hash_range = range(1, 120)

    for x in hash_range:
        ax.plot([x, x], [0.4, 0.7], color='white')
        ax.plot([x, x], [53.0, 52.5], color='white')
        ax.plot([x, x], [22.91, 23.57], color='white')
        ax.plot([x, x], [29.73, 30.39], color='white')

    if highlight_line:
        hl = highlight_line_number + 10
        plt.plot([hl, hl], [0, 53.3], color='yellow')
        plt.text(hl + 2, 50, '<- {}'.format(highlighted_name),
                 color='yellow')
    return fig, ax

create_football_field()
plt.show()




# https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-nfl

import math
def get_dx_dy(angle, dist):
    cartesianAngleRadians = (450-angle)*math.pi/180.0
    dx = dist * math.cos(cartesianAngleRadians)
    dy = dist * math.sin(cartesianAngleRadians)
    return dx, dy

play_id = 20181007011551
fig, ax = create_football_field()
train_df.query("PlayId == @play_id and Team == 'away'")     .plot(x='X', y='Y', kind='scatter', ax=ax, color='orange', s=50, legend='Away')
train_df.query("PlayId == @play_id and Team == 'home'")     .plot(x='X', y='Y', kind='scatter', ax=ax, color='blue', s=50, legend='Home')
train_df.query("PlayId == @play_id and NflIdRusher == NflId")     .plot(x='X', y='Y', kind='scatter', ax=ax, color='red', s=100, legend='Rusher')
rusher_row = train_df.query("PlayId == @play_id and NflIdRusher == NflId")
yards_covered = rusher_row["Yards"].values[0]

x = rusher_row["X"].values[0]
y = rusher_row["Y"].values[0]
rusher_dir = rusher_row["Dir"].values[0]
rusher_speed = rusher_row["S"].values[0]
dx, dy = get_dx_dy(rusher_dir, rusher_speed)

ax.arrow(x, y, dx, dy, length_includes_head=True, width=0.3)
plt.title(f'Play # {play_id} and yard distance is {yards_covered}', fontsize=20)
plt.legend()
plt.show()




print("Total number of  S : ", train_df.S.shape[0])
print("Total number of  A : ", train_df.A.shape[0])
print("*"*50)
print("max of S : ", train_df.S.max())
print("max of A : ", train_df.A.max())
print("*"*50)
print("min of S : ", train_df.S.min())
print("min of A : ", train_df.A.min())
print("*"*50)
print("mean of S : ", train_df.S.values.mean())
print("mean of A : ", train_df.A.values.mean())




plt.figure(figsize=(16,6))
plt.subplot(121)
sns.distplot(train_df.S)
plt.vlines(train_df.S.values.mean(), plt.ylim()[0], plt.ylim()[1], color='r', linestyles='--');
plt.text(train_df.S.values.mean(), plt.ylim()[1]-0.01, "Mean of S", size=15, color='r')
plt.title("Speed('S') Distribution")
plt.subplot(122)
sns.distplot(train_df.A)
plt.vlines(train_df.A.values.mean(), plt.ylim()[0], plt.ylim()[1], color='r', linestyles='--');
plt.text(train_df.A.values.mean(), plt.ylim()[1]-0.02, "Mean of A", size=15, color='r')
plt.title("Acceleration('A') Distribution")
plt.show()




print("Total number of  Dis : ", train_df.Dis.shape[0])
print("*"*50)
print("max of Dis : ", train_df.Dis.max())
print("*"*50)
print("min of Dis : ", train_df.Dis.min())
print("*"*50)
print("mean of Dis : ", train_df.Dis.values.mean())




plt.figure(figsize=(16,6))
sns.distplot(train_df.Dis)
plt.vlines(train_df.Dis.values.mean(), plt.ylim()[0], plt.ylim()[1], color='r', linestyles='--');
plt.text(train_df.Dis.values.mean(), plt.ylim()[1]-0.01, "Mean of Dis", size=15, color='r')
plt.title("Distance(Dis) distribution")
plt.show()




print("Total number of  Orientation : ", train_df.Orientation.shape[0])
print("*"*50)
print("max of Orientation : ", train_df.Orientation.max())
print("*"*50)
print("min of Orientation : ", train_df.Orientation.min())
print("*"*50)
print("Number of missing values : ", train_df.Orientation.isna().sum())




drop_na_Orientation = train_df.Orientation.dropna()




plt.figure(figsize=(16,6))
sns.distplot(drop_na_Orientation)
plt.vlines(drop_na_Orientation.values.mean(), plt.ylim()[0], plt.ylim()[1], color='r', linestyles='--');
plt.text(drop_na_Orientation.values.mean(), plt.ylim()[1]-0.0002, "Mean of Orienataion", size=15, color='r')
plt.title("Orientation distribution")
plt.show()




print("Total number of  Dir : ", train_df.Dir.shape[0])
print("*"*50)
print("max of Dir : ", train_df.Dir.max())
print("*"*50)
print("min of Dir : ", train_df.Dir.min())
print("*"*50)
print("Number of missing values : ", train_df.Dir.isna().sum())




drop_na_Dir = train_df.Dir.dropna()




plt.figure(figsize=(16,6))
sns.distplot(drop_na_Dir)
plt.vlines(drop_na_Dir.values.mean(), plt.ylim()[0], plt.ylim()[1], color='r', linestyles='--');
plt.text(drop_na_Dir.values.mean(), plt.ylim()[1]-0.0002, "Mean of Dir", size=15, color='r')
plt.title("Direction(Dir) distribution")
plt.show()




print("Total number unique players : ", train_df.NflId.nunique())
print("*"*50)
print("max number of times PlayId player id is : ", train_df.NflId.value_counts().index[0] , "number of HandOffs is : ", train_df.NflId.value_counts().values[0])
print("*"*50)
print("min number of times PlayId player id is : ", train_df.NflId.value_counts().index[-1] , "number of HandOffs is : ", train_df.NflId.value_counts().values[-1])
print("*"*50)
print("Number of missing values : ", train_df.NflId.isna().sum())




print("Total number unique player names : ", train_df.DisplayName.nunique())
print("*"*50)
print("max of number of times PlayId player name is : ", train_df.DisplayName.value_counts().index[0] , "number of HandOffs is : ", train_df.DisplayName.value_counts().values[0])
print("*"*50)
print("min of number of times PlayId player name is : ", train_df.DisplayName.value_counts().index[-1] , "number of HandOffs is : ", train_df.DisplayName.value_counts().values[-1])
print("*"*50)
print("Number of missing values : ", train_df.DisplayName.isna().sum())




print("Total number unique player numbers : ", train_df.JerseyNumber.nunique())
print("*"*50)
print("max of number of times PlayId player number is : ", train_df.JerseyNumber.value_counts().index[0] , "number of HandOffs is : ", train_df.JerseyNumber.value_counts().values[0])
print("*"*50)
print("min of number of times PlayId player number is : ", train_df.JerseyNumber.value_counts().index[-1] , "number of HandOffs is : ", train_df.JerseyNumber.value_counts().values[-1])
print("*"*50)
print("Number of missing values : ", train_df.JerseyNumber.isna().sum())




print("Total number unique seasons : ", train_df.Season.nunique())

print("Those are : \n", train_df.Season.value_counts())




train_dff = train_df[::22]
print("Total number of  YardLine : ", train_dff.YardLine.shape[0])
print("*"*50)
print("max of YardLine : ", train_dff.YardLine.max())
print("*"*50)
print("min of YardLine : ", train_dff.YardLine.min())
print("*"*50)
print("Number of missing values : ", train_dff.YardLine.isna().sum())




plt.figure()
sns.distplot(train_dff.YardLine)
plt.title("Distribution of YardLine")
plt.show()




Quarter = train_df.Quarter[: : 22]




sns.countplot(Quarter)




GameClock = train_df.GameClock[::22]




GameClock.value_counts()[:5]




train_df.loc[train_df['PossessionTeam'] == 'ARZ', 'PossessionTeam'] = 'ARI'
train_df.loc[train_df['PossessionTeam'] == 'BLT', 'PossessionTeam'] = 'BAL'
train_df.loc[train_df['PossessionTeam'] == 'CLV', 'PossessionTeam'] = 'CLE'
train_df.loc[train_df['PossessionTeam'] == 'HST', 'PossessionTeam'] = 'HOU'




PossessionTeam = train_df.PossessionTeam[::22]




plt.figure(figsize=(15,10))
sns.countplot(y=PossessionTeam)
plt.title("PossessionTeam countplot")
plt.show()




train_dff = train_df[::22]
print("Total number of  Downs : ", train_dff.Down.shape[0])
print("*"*50)
print("max of HandOffs done on Down : ", train_dff.Down.value_counts().keys()[0], " are : ",train_dff.Down.value_counts().values[0] )
print("*"*50)
print("min of HandOffs done on Down : ", train_dff.Down.value_counts().keys()[-1], " are : ",train_dff.Down.value_counts().values[-1] )
print("*"*50)
print("Number of missing values : ", train_dff.Down.isna().sum())




plt.figure()
sns.countplot(x=train_dff.Down)
plt.title("Down countplot")
plt.show()




train_dff = train_df[::22]
print("Total number of  Distances : ", train_dff.Distance.shape[0])
print("*"*50)
print("max of Distance : ", train_dff.Distance.max())
print("*"*50)
print("min of Distance : ", train_dff.Distance.min())
print("*"*50)
print("Number of missing values : ", train_dff.YardLine.isna().sum())




plt.figure(figsize=(15,10))
sns.countplot(y=train_dff.Distance)
plt.title("PossessionTeam countplot")
plt.show()




FieldPosition = train_df.FieldPosition[::22]




plt.figure(figsize=(15,10))
sns.countplot(y=FieldPosition)
plt.title("FieldPosition countplot")
plt.show()




HomeScoreBeforePlay = train_df["HomeScoreBeforePlay"][::22]




print("max of HomeScoreBeforePlay : ", HomeScoreBeforePlay.max())
print("*"*50)
print("min of HomeScoreBeforePlay : ", HomeScoreBeforePlay.min())
print("*"*50)
print("Number of missing values : ", HomeScoreBeforePlay.isna().sum())




plt.figure()
sns.distplot(HomeScoreBeforePlay)
plt.title("Distribution of HomeScoreBeforePlay")
plt.show()




VisitorScoreBeforePlay = train_df["VisitorScoreBeforePlay"][::22]

print("max of HomeScoreBeforePlay : ", VisitorScoreBeforePlay.max())
print("*"*50)
print("min of HomeScoreBeforePlay : ", VisitorScoreBeforePlay.min())
print("*"*50)
print("Number of missing values : ", VisitorScoreBeforePlay.isna().sum())




plt.figure()
sns.distplot(VisitorScoreBeforePlay)
plt.title("Distribution of VisitorScoreBeforePlay")
plt.show()




NflIdRusher = train_df.NflIdRusher[::22]
print("Total number of unique NflIdRyshers : ", NflIdRusher.nunique())
print("*"*50)
print("max times HandOff for NflIdRusher ID is : ", NflIdRusher.value_counts().keys()[0] , "Number of times is :", NflIdRusher.value_counts().values[0])
print("*"*50)
print("min times HandOff for NflIdRusher ID is : ", NflIdRusher.value_counts().keys()[-1] , "Number of times is :", NflIdRusher.value_counts().values[-1])
print("*"*50)
print("Number of missing values : ", NflIdRusher.isna().sum())




OffenseFormation = train_df.OffenseFormation[::22] 




print("Number of missing values : ", OffenseFormation.isna().sum())




plt.figure()
sns.countplot(y=OffenseFormation)
plt.title("OffenseFormation countplot")
plt.show()




import tqdm as tqdm




OffensePersonnel = train_df.OffensePersonnel[::22]




unique_OffensePersonnel = []
for row in OffensePersonnel:
    result = ''.join([i for i in row.replace(',','') if not i.isdigit()]).strip()
    for per in result.split():
        if per not in unique_OffensePersonnel:
            unique_OffensePersonnel.append(per)
unique_OffensePersonnel




train_df.OffensePersonnel[:5]




OffensePersonnel_df = pd.DataFrame(0 ,columns=unique_OffensePersonnel, index=train_df.index)
for ind,personnel in enumerate(train_df.OffensePersonnel):
    pos = personnel.split(',')
    for i in pos:
        col = i[-2:]
        OffensePersonnel_df.loc[ind][col] = int(i[-4])




OffensePersonnel_df = OffensePersonnel_df.add_prefix("offense_")




OffensePersonnel_df.head()




train_df = pd.merge(train_df , OffensePersonnel_df, how="left", left_index=True, right_index=True)




DefendersInTheBox = train_df.DefendersInTheBox[::22]

print("Total number of  DefendersInTheBox : ", DefendersInTheBox.shape[0])
print("*"*50)
print("max of Dir : ", DefendersInTheBox.max())
print("*"*50)
print("min of Dir : ", DefendersInTheBox.min())
print("*"*50)
print("Number of missing values : ", DefendersInTheBox.isna().sum())




plt.figure()
sns.distplot(DefendersInTheBox.dropna())
plt.title("Distribution of DefendersInTheBox")
plt.show()




plt.figure()
sns.countplot(y=DefendersInTheBox.dropna())
plt.title("DefendersInTheBox countplot")
plt.show()




DefensePersonnel = train_df.DefensePersonnel[::22]




unique_DefensePersonnel = []
for row in DefensePersonnel:
    result = ''.join([i for i in row.replace(',','') if not i.isdigit()]).strip()
    for per in result.split():
        if per not in unique_DefensePersonnel:
            unique_DefensePersonnel.append(per)
unique_DefensePersonnel




train_df.DefensePersonnel[:5]




DefensePersonnel_df = pd.DataFrame(0 ,columns=unique_DefensePersonnel, index=train_df.index)
for ind,personnel in enumerate(train_df.DefensePersonnel):
    pos = personnel.split(',')
    for i in pos:
        col = i[-2:]
        DefensePersonnel_df.loc[ind][col] = int(i[-4])




DefensePersonnel_df = DefensePersonnel_df.add_prefix("defense_")




train_df = pd.merge(train_df , DefensePersonnel_df, how="left", left_index=True, right_index=True)




train_df.PlayDirection.value_counts()




train_df["TimeHandoff"] = train_df["TimeHandoff"].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))




train_df.TimeHandoff[::22][:5]




train_df['TimeSnap'] = train_df['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))




train_df.TimeSnap[::22][:5]




Yards = train_df.Yards[::22]




print("max Yards : ", Yards.max())
print("*"*50)
print("min Yards : ", Yards.min())
print("*"*50)
print("Number of missing values : ", Yards.isna().sum())




plt.figure()
sns.distplot(Yards)
plt.title("Distribution of Yards")
plt.show()




train_df.PlayerHeight[:5]




# https://www.kaggle.com/bgmello/neural-networks-feature-engineering-for-the-win

train_df['PlayerHeight'] = train_df['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))




print("max PlayerHeight : ", train_df.PlayerHeight.max())
print("*"*50)
print("min PlayerHeight : ", train_df.PlayerHeight.min())
print("*"*50)
print("Number of missing values : ", train_df.PlayerHeight.isna().sum())




plt.figure()
sns.distplot(train_df.PlayerHeight)
plt.title("Distribution of player height")
plt.show()




print("max PlayerWeight : ", train_df.PlayerWeight.max())
print("*"*50)
print("min PlayerWeight : ", train_df.PlayerWeight.min())
print("*"*50)
print("Number of missing values : ", train_df.PlayerWeight.isna().sum())




plt.figure()
sns.distplot(train_df.PlayerWeight)
plt.title("Distribution of player weight")
plt.show()




train_df["PlayerBirthDate"] = train_df["PlayerBirthDate"].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))




train_df.PlayerBirthDate[:5]




print("Total number of unique PlayerCollegeName : ", train_df.PlayerCollegeName.nunique())
print("*"*50)
print("max number of players from : ", train_df.PlayerCollegeName.value_counts().keys()[0] , " and Number of players :", train_df.PlayerCollegeName.value_counts().values[0])
print("*"*50)
print("min number of players from : ", train_df.PlayerCollegeName.value_counts().keys()[-1] , " and Number of players :", train_df.PlayerCollegeName.value_counts().values[-1])
print("*"*50)
print("Number of missing values : ", train_df.PlayerCollegeName.isna().sum())




HomeTeamAbbr = train_df.HomeTeamAbbr[::22]
plt.figure(figsize=(15,10))
sns.countplot(y=HomeTeamAbbr)
plt.title("HomeTeamAbbr countplot")
plt.show()




VisitorTeamAbbr = train_df.VisitorTeamAbbr[::22]
plt.figure(figsize=(15,10))
sns.countplot(y=VisitorTeamAbbr)
plt.title("VisitorTeamAbbr countplot")
plt.show()




Week = train_df.Week[::22]




plt.figure(figsize=(15,10))
sns.countplot(y=Week)
plt.title("Week countplot")
plt.show()




Stadium = train_df.Stadium[::22].str.lower()




print("Total number of unique Stadiums : ", Stadium.nunique())
print("*"*50)
print("max number of plays in Stadium is : ", Stadium.value_counts().keys()[0] , " and Number of played :", Stadium.value_counts().values[0])
print("*"*50)
print("min number of plays in Stadium is : ", Stadium.value_counts().keys()[-1] , " and Number of played :", Stadium.value_counts().values[-1])
print("*"*50)
print("Number of missing values : ", Stadium.isna().sum())




Location = train_df.Location[::22].str.lower()




print("Total number of unique Location : ", Location.nunique())
print("*"*50)
print("max number of plays in Location is : ", Location.value_counts().keys()[0] , " and Number of played :", Location.value_counts().values[0])
print("*"*50)
print("min number of plays in Location is : ", Location.value_counts().keys()[-1] , " and Number of played :", Location.value_counts().values[-1])
print("*"*50)
print("Number of missing values : ", Location.isna().sum())




train_df.StadiumType[::22].value_counts()




def clean_StadiumType(txt):
    if pd.isna(txt):
        return np.nan
    txt = txt.lower()
    txt = ''.join([c for c in txt if c not in punctuation])
    txt = re.sub(' +', ' ', txt)
    txt = txt.replace('outside', 'outdoor')
    txt = txt.replace('outdor', 'outdoor')
    txt = txt.replace('outddors', 'outdoor')
    txt = txt.replace('outdoors', 'outdoor')
    txt = txt.replace('oudoor', 'outdoor')
    txt = txt.replace('indoors', 'indoor')
    txt = txt.replace('ourdoor', 'outdoor')
    txt = txt.replace('retractable', 'rtr.')
    txt = txt.replace('dome','domed')
    txt = txt.replace('retr roofclosed', 'retr roof closed')
    txt = txt.replace('retr roofopen', 'retr roof open')
    txt = txt.replace('domeddd', 'domed')
    txt = txt.replace('domedd closed', 'domed closed')
    txt = txt.replace('closed domed', 'domed closed')
    txt = txt.replace('domed closedd', 'domed closed')
    txt = txt.replace('domedd', 'domed')
    return txt




train_df['StadiumType'] = train_df['StadiumType'].apply(clean_StadiumType)




train_df.Turf.value_counts()




#from https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/112681#latest-649087
grass_labels = ['grass', 'natural grass', 'natural', 'naturall grass']
train_df['Grass'] = np.where(train_df.Turf.str.lower().isin(grass_labels), 1, 0)




train_df.GameWeather.unique()




train_df['GameWeather'] = train_df['GameWeather'].str.lower()
indoor = "indoor"
train_df['GameWeather'] = train_df['GameWeather'].apply(lambda x: indoor if not pd.isna(x) and indoor in x else x)
train_df['GameWeather'] = train_df['GameWeather'].apply(lambda x: x.replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party', 'partly') if not pd.isna(x) else x)
train_df['GameWeather'] = train_df['GameWeather'].apply(lambda x: x.replace('clear and sunny', 'sunny and clear') if not pd.isna(x) else x)
train_df['GameWeather'] = train_df['GameWeather'].apply(lambda x: x.replace('skies', '').replace("mostly", "").strip() if not pd.isna(x) else x)




train_df['GameWeather'].unique()




from collections import Counter
weather_count = Counter()
for weather in train_df['GameWeather']:
    if pd.isna(weather):
        continue
    for word in weather.split():
        weather_count[word]+=1
        
weather_count.most_common()[:15]




Temperature = train_df.Temperature[::22]

print("max of Temperature : ", Temperature.max())
print("*"*50)
print("min of Temperature : ", Temperature.min())
print("*"*50)
print("Number of missing values : ", Temperature.isna().sum())




plt.figure()
sns.distplot(Temperature.dropna())
plt.title("Distribution of Temperature")
plt.show()




Humidity = train_df.Humidity[::22]

print("max of Humidity : ", Humidity.max())
print("*"*50)
print("min of Humidity : ", Humidity.min())
print("*"*50)
print("Number of missing values : ", Humidity.isna().sum())




plt.figure()
sns.distplot(Humidity.dropna())
plt.title("Distribution of Humidity")
plt.show()




train_df['WindSpeed'].value_counts()




train_df['WindSpeed'] = train_df['WindSpeed'].apply(lambda x: x.lower().replace('mph', '').strip() if not pd.isna(x) else x)




#let's replace the ones that has x-y by (x+y)/2
# and also the ones with x gusts up to y
train_df['WindSpeed'] = train_df['WindSpeed'].apply(lambda x: (int(x.split('-')[0])+int(x.split('-')[1]))/2 if not pd.isna(x) and '-' in x else x)
train_df['WindSpeed'] = train_df['WindSpeed'].apply(lambda x: (int(x.split()[0])+int(x.split()[-1]))/2 if not pd.isna(x) and type(x)!=float and 'gusts up to' in x else x)




def str_to_float(txt):
    try:
        return float(txt)
    except:
        return -1




train_df['WindSpeed'] = train_df['WindSpeed'].apply(str_to_float)




train_df.WindSpeed.value_counts()




train_df['WindDirection'].value_counts()




def clean_WindDirection(txt):
    if pd.isna(txt):
        return np.nan
    txt = txt.lower()
    txt = ''.join([c for c in txt if c not in punctuation])
    txt = txt.replace('from', '')
    txt = txt.replace(' ', '')
    txt = txt.replace('north', 'n')
    txt = txt.replace('south', 's')
    txt = txt.replace('west', 'w')
    txt = txt.replace('east', 'e')
    return txt




train_df['WindDirection'] = train_df['WindDirection'].apply(clean_WindDirection)




train_df['WindDirection'].value_counts()




def transform_WindDirection(txt):
    if pd.isna(txt):
        return np.nan
    
    if txt=='n':
        return 0
    if txt=='nne' or txt=='nen':
        return 1/8
    if txt=='ne':
        return 2/8
    if txt=='ene' or txt=='nee':
        return 3/8
    if txt=='e':
        return 4/8
    if txt=='ese' or txt=='see':
        return 5/8
    if txt=='se':
        return 6/8
    if txt=='ses' or txt=='sse':
        return 7/8
    if txt=='s':
        return 8/8
    if txt=='ssw' or txt=='sws':
        return 9/8
    if txt=='sw':
        return 10/8
    if txt=='sww' or txt=='wsw':
        return 11/8
    if txt=='w':
        return 12/8
    if txt=='wnw' or txt=='nww':
        return 13/8
    if txt=='nw':
        return 14/8
    if txt=='nwn' or txt=='nnw':
        return 15/8
    return np.nan




train_df['WindDirection'] = train_df['WindDirection'].apply(transform_WindDirection)






