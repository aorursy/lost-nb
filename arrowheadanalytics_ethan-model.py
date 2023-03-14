#!/usr/bin/env python
# coding: utf-8

# In[1]:


#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import time
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold

from keras.layers import Dense,Input,Flatten,concatenate,Dropout,Lambda,BatchNormalization
from keras.models import Model
import keras.backend as K
import re
from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
import codecs
from keras.utils import to_categorical
from sklearn.metrics import f1_score


# In[2]:


#Set display options for our dataframes
pd.set_option('display.max_columns', 25)
pd.set_option('display.max_rows', 100)
#Silence the warning for chained assignments
pd.set_option('mode.chained_assignment', None)


# In[3]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', dtype={'WindSpeed': 'object'})

#Get our outcome variables
outcomes = train_df[['GameId','PlayId','Yards']].drop_duplicates()

#Look at number of plays in our sample
print(len(outcomes))


# In[4]:


#Get how many defenders are in the box - I found that binning the outliers (<5 defenders and >9 defenders) worked well
train_df['Simple_Box'] = train_df['DefendersInTheBox']
train_df.loc[(train_df['DefendersInTheBox'].isnull()),'Simple_Box'] = 6
train_df.loc[(train_df['DefendersInTheBox']<=5),'Simple_Box'] = 5
train_df.loc[(train_df['DefendersInTheBox']>=9),'Simple_Box'] = 9
train_df['Simple_Box'] = train_df['Simple_Box'].astype(str)

#Create our "predictions"
yds = np.linspace(-99,99,199)
box = train_df['Simple_Box'].unique()
box_cdf = {}
box_pdf = {}
for n in box:
    df = train_df.loc[(train_df['Simple_Box']==n)]
    data = df['Yards']
    kde1 = stats.gaussian_kde(data)
    predict_pdf = kde1(yds)
    box_pdf[n] = predict_pdf
    predict_cdf = np.cumsum(predict_pdf)
    predict_cdf = np.clip(predict_cdf,0,1)
    #This last line gives us a dictionary where the keys are the number of defenders in the box,
    #and the values are a 199 length array (the probability of each yard gained)
    #This allows us to just check the testing data for how many defenders in the box, 
    box_cdf[n] = predict_cdf


# In[5]:


plt.style.use('seaborn-talk')
plt.style.use('seaborn-darkgrid')
for box in sorted(box_pdf):
    plt.plot(yds,box_cdf[box],label=box)
    

plt.xlim(-10,30)
plt.legend(title='PDF by Defenders in the Box')
plt.ylabel('Cumulative Probability')
plt.xlabel('Expected Yards Gained')


# In[6]:


#Repeat the same process, this time binning by both defenders in the box AND yards to go

#These numbers seemed to be the most predictive.
train_df.loc[(train_df['Distance']<=8),'binned_yards'] = 'med'
train_df.loc[(train_df['Distance']<=3),'binned_yards'] = 'short'
# train_df.loc[(train_df['Distance']<=1),'binned_yards'] = 'really short'
train_df.loc[(train_df['Distance']>8),'binned_yards'] = 'long'

train_df['box_yards'] = train_df['Simple_Box'] + ' ' + train_df['binned_yards']

box = train_df['box_yards'].unique()
box_cdf = {}
box_pdf = {}
for n in box:
    df = train_df.loc[(train_df['box_yards']==n)]
    print('There are ' + str(len(df)/22) + ' plays for ' + n)
    data = df['Yards']
    kde1 = stats.gaussian_kde(data)
    predict_pdf = kde1(yds)
    box_pdf[n] = predict_pdf
    predict_cdf = np.cumsum(predict_pdf)
    predict_cdf = np.clip(predict_cdf,0,1)
    box_cdf[n] = predict_cdf


# In[7]:


plt.style.use('seaborn-talk')
plt.style.use('seaborn-darkgrid')
for box in sorted(box_pdf):
    plt.plot(yds,box_cdf[box],label=box)
    

plt.xlim(-10,30)
plt.legend(title='Defenders in the Box + Yards to Go')
plt.ylabel('Cumulative Probability')
plt.xlabel('Expected Yards Gained')


# In[8]:


#Function to create the features we want to create for each play - and we won't use our outcomes for when we are testing, 
#hence the need for this boolean
def create_features(df, test=False):
    
    
    #Function to standardize the play-by-play coordinates
    def standardize(df):
        
        #Create a binary variable to use for play direction
        df['ToLeft'] = df.PlayDirection == "left"
        #Binary variable for who is the ball carrier - important for our predictions
        df['IsBallCarrier'] = df.NflId == df.NflIdRusher
        
        #Fix some inconsistencies in team abbreviations betweeten columns
        df.loc[df.VisitorTeamAbbr == "ARI", 'VisitorTeamAbbr'] = "ARZ"
        df.loc[df.HomeTeamAbbr == "ARI", 'HomeTeamAbbr'] = "ARZ"

        df.loc[df.VisitorTeamAbbr == "BAL", 'VisitorTeamAbbr'] = "BLT"
        df.loc[df.HomeTeamAbbr == "BAL", 'HomeTeamAbbr'] = "BLT"

        df.loc[df.VisitorTeamAbbr == "CLE", 'VisitorTeamAbbr'] = "CLV"
        df.loc[df.HomeTeamAbbr == "CLE", 'HomeTeamAbbr'] = "CLV"

        df.loc[df.VisitorTeamAbbr == "HOU", 'VisitorTeamAbbr'] = "HST"
        df.loc[df.HomeTeamAbbr == "HOU", 'HomeTeamAbbr'] = "HST"
        
        #Figure out if home or away team is on offense
        df['TeamOnOffense'] = "home"
        df.loc[df.PossessionTeam != df.HomeTeamAbbr, 'TeamOnOffense'] = "away"
        #Binary variable for each player to see if they are on offense
        df['IsOnOffense'] = df.Team == df.TeamOnOffense # Is player on offense?
        #Create a standardized yardline - from 1-99, instead of 1-50 and 1-50 again (on other side of the field)
        df['YardLine_std'] = 100 - df.YardLine
        df.loc[df.FieldPosition.fillna('') == df.PossessionTeam,  
                  'YardLine_std'
                 ] = df.loc[df.FieldPosition.fillna('') == df.PossessionTeam,  
                  'YardLine']
        
        df['YardLine'] = df['YardLine_std']
        #Create standardized coordinates to each coordinate represents a distance away from the team's own goal,
        #and so that all plays are moving the same direction (positive yardage is going towards the opponent's goal)
        df['X_std'] = df.X
        df.loc[df.ToLeft, 'X_std'] = 120 - df.loc[df.ToLeft, 'X'] 
        df['Y_std'] = df.Y
        df.loc[df.ToLeft, 'Y_std'] = 160/3 - df.loc[df.ToLeft, 'Y']
        df['X'] = df['X_std'] - 10
        df['Y'] = df['Y_std']
        
        #Derive some standardized directions. 
        
        #Standardize the direction of player's similar to what we did above for coordinates
        #Reasoning for this is explained here:https://www.kaggle.com/statsbymichaellopez/nfl-tracking-wrangling-voronoi-and-sonars
        #But basically here we have 180 degrees to be moving backward 
        #and 0 degrees to be moving directly downfield towards the offense
        
        #We are also converting to radians here.
        df['Dir_rad'] = np.mod(90 - df.Dir, 360) * np.pi/180.0
        df['Dir_std'] = df.Dir_rad
        df.loc[df.ToLeft, 'Dir_std'] = np.mod(np.pi + df.loc[df.ToLeft, 'Dir_rad'], 2*np.pi)
        df['Dir_rad'] = df['Dir_std']
        
        #Simple trig to get horizontal and vertical components of movement
        df["Dir_y"] = np.sin(df["Dir_rad"])
        df["Dir_x"] = np.cos(df["Dir_rad"])
        #Get horizontal and vertical components of velocity
        df['V_y'] = np.absolute(df['Dir_y'] * df['S'])
        df['V_x'] = np.absolute(df['Dir_x'] * df['S'])
  
        #Since we recopied the column values with their standardized versions, we don't need these anymore
        df.drop(columns=['X_std','Y_std',
                        'Dir_std','YardLine_std'],inplace=True)
        return df

    #Features specific to the ball carrier
    def back_features(df):
        #Get ballcarrier features - how far back they are from the line of scrimmage, and their X and Y position
        carriers = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','NflIdRusher','X','Y','Orientation','Dir','YardLine','Position']]
        carriers['back_from_scrimmage'] = carriers['YardLine'] - carriers['X']
        carriers = carriers.rename(columns={'X':'back_X',
                                            'Y':'back_Y',
                                           'Position':'RusherPosition'})
        carriers = carriers[['GameId','PlayId','NflIdRusher','RusherPosition','back_X','back_Y','back_from_scrimmage']]
        
        return carriers
    
    #Features relative to the ball carrier
    def features_relative_to_back(df, carriers):
        #Get positions of the rest of the offense and defense relative to the running back for every play
        player_distance = df[['GameId','PlayId','NflId','X','Y']]
        player_distance = pd.merge(player_distance, carriers, on=['GameId','PlayId'], how='inner')
        player_distance = player_distance[player_distance['NflId'] != player_distance['NflIdRusher']]
        
        player_distance['X_dist_to_back'] = (player_distance['X'] - player_distance['back_X'])
        player_distance['Y_dist_to_back'] = np.absolute(player_distance['Y'] - player_distance['back_Y'])
        player_distance['dist_to_back'] = np.sqrt(player_distance['X_dist_to_back']**2 + player_distance['Y_dist_to_back']**2)
        #Get some aggregated features of the rest of the players on this play
        player_distance = player_distance.groupby(['GameId','PlayId','back_from_scrimmage'])                                         .agg({'dist_to_back':['min','max','mean','std'],
                                              'X_dist_to_back':['min','max','mean','std'],
                                              'Y_dist_to_back':['min','max','mean','std']})\
                                         .reset_index()
        player_distance.columns = ['GameId','PlayId','back_from_scrimmage',
                                   'min_dist','max_dist','mean_dist','std_dist',
                                  'X_min_dist','X_max_dist','X_mean_dist','X_std_dist',
                                  'Y_min_dist','Y_max_dist','Y_mean_dist','Y_std_dist']

        return player_distance

    def defense_features(df):
        #Get rusher features
        rusher = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','Team','X','Y','S']]
        rusher.columns = ['GameId','PlayId','RusherTeam','RusherX','RusherY','RusherS']

        #Merge defense with rusher features
        defense = pd.merge(df,rusher,on=['GameId','PlayId'],how='inner')
        defense = defense[defense['Team'] != defense['RusherTeam']][['GameId','PlayId','X','Y',
                                                                     'S','A',
                                                                     'Dir','Dis',
                                                                     'PlayerWeight',
                                                                     'Orientation',
                                                                     'V_x','V_y',
                                                                     'RusherX','RusherY','RusherS']]
        #Calculate dfense features
#         defense['def_dist_to_back'] = defense[['X','Y','RusherX','RusherY']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)
        defense['X_def_dist_to_back'] = (defense['X'] - defense['RusherX'])
        defense['Y_def_dist_to_back'] = np.absolute((defense['Y'] - defense['RusherY']))
        defense['def_dist_to_back'] = np.sqrt(defense['X_def_dist_to_back']**2 + defense['Y_def_dist_to_back']**2)
        
        #Calculate the minimum time to tackle a ball carrier, based on how far away the closest defender is
        defense['def_time_to_tackle'] = defense['def_dist_to_back'] / (defense['S'] + defense['RusherS'])
        
        
        #Get 8 closest defenders
        defense.sort_values(by=['GameId','PlayId','def_dist_to_back'],inplace=True)
        defense.reset_index(inplace=True)
        
        #Defender 1
        defense['defender1'] = 0
        defense['defender1'].loc[np.arange(0, len(defense), 11)] = 1
        defender1 = defense[defense['defender1'] == 1][['GameId','PlayId',
                                         'X_def_dist_to_back','Y_def_dist_to_back',
                                                        'def_dist_to_back',
                                         'V_x','V_y','A']]
        defender1 = defender1.rename(columns={'X_def_dist_to_back':'def1_X_dist',
                                              'Y_def_dist_to_back':'def1_Y_dist',
                                              'def_dist_to_back':'def1_dist',
                                              'V_x':'def1_V_x',
                                              'V_y':'def1_V_y',
                                              'A':'def1_A',
                                              'Orientation':'def1_orientation',
                                              'Dir':'def1_Dir',
                                              'Dis':'def1_Dis'
                                             })
        
        #Defender 2
        defense['defender2'] = 0
        defense['defender2'].loc[np.arange(1, len(defense), 11)] = 1
        defender2 = defense[defense['defender2'] == 1][['GameId','PlayId',
                                         'X_def_dist_to_back','Y_def_dist_to_back',
                                                        'def_dist_to_back',
                                         'V_x','V_y','A']]
        defender2 = defender2.rename(columns={'X_def_dist_to_back':'def2_X_dist',
                                              'Y_def_dist_to_back':'def2_Y_dist',
                                              'def_dist_to_back':'def2_dist',
                                              'V_x':'def2_V_x',
                                              'V_y':'def2_V_y',
                                              'A':'def2_A',
                                              'Orientation':'def2_orientation',
                                              'Dir':'def2_Dir',
                                              'Dis':'def2_Dis'
                                             })
        #Defender 3
        defense['defender3'] = 0
        defense['defender3'].loc[np.arange(2, len(defense), 11)] = 1
        defender3 = defense[defense['defender3'] == 1][['GameId','PlayId',
                                         'X_def_dist_to_back','Y_def_dist_to_back',
                                                        'def_dist_to_back',
                                         'V_x','V_y','A']]
        defender3 = defender3.rename(columns={'X_def_dist_to_back':'def3_X_dist',
                                              'Y_def_dist_to_back':'def3_Y_dist',
                                              'def_dist_to_back':'def3_dist',
                                              'V_x':'def3_V_x',
                                              'V_y':'def3_V_y',
                                              'A':'def3_A',
                                              'Orientation':'def3_orientation',
                                              'Dir':'def3_Dir',
                                              'Dis':'def3_Dis'
                                             })
        
        #Defender 4
        defense['defender4'] = 0
        defense['defender4'].loc[np.arange(3, len(defense), 11)] = 1
        defender4 = defense[defense['defender4'] == 1][['GameId','PlayId',
                                         'X_def_dist_to_back','Y_def_dist_to_back',
                                                        'def_dist_to_back',
                                         'V_x','V_y','A']]
        defender4 = defender4.rename(columns={'X_def_dist_to_back':'def4_X_dist',
                                              'Y_def_dist_to_back':'def4_Y_dist',
                                              'def_dist_to_back':'def4_dist',
                                              'V_x':'def4_V_x',
                                              'V_y':'def4_V_y',
                                              'A':'def4_A',
                                              'Orientation':'def4_orientation',
                                              'Dir':'def4_Dir',
                                              'Dis':'def4_Dis'
                                             })
        
        #Defender 5
        defense['defender5'] = 0
        defense['defender5'].loc[np.arange(4, len(defense), 11)] = 1
        defender5 = defense[defense['defender5'] == 1][['GameId','PlayId',
                                         'X_def_dist_to_back','Y_def_dist_to_back',
                                                        'def_dist_to_back',
                                         'V_x','V_y','A']]
        defender5 = defender5.rename(columns={'X_def_dist_to_back':'def5_X_dist',
                                              'Y_def_dist_to_back':'def5_Y_dist',
                                              'def_dist_to_back':'def5_dist',
                                              'V_x':'def5_V_x',
                                              'V_y':'def5_V_y',
                                              'A':'def5_A',
                                              'Orientation':'def5_orientation',
                                              'Dir':'def5_Dir',
                                              'Dis':'def5_Dis'
                                             })
        
        #Defender 6
        defense['defender6'] = 0
        defense['defender6'].loc[np.arange(5, len(defense), 11)] = 1
        defender6 = defense[defense['defender6'] == 1][['GameId','PlayId',
                                         'X_def_dist_to_back','Y_def_dist_to_back',
                                                        'def_dist_to_back',
                                         'V_x','V_y','A']]
        defender6 = defender6.rename(columns={'X_def_dist_to_back':'def6_X_dist',
                                              'Y_def_dist_to_back':'def6_Y_dist',
                                              'def_dist_to_back':'def6_dist',
                                              'V_x':'def6_V_x',
                                              'V_y':'def6_V_y',
                                              'A':'def6_A',
                                              'Orientation':'def6_orientation',
                                              'Dir':'def6_Dir',
                                              'Dis':'def6_Dis'
                                             })
        #Defender 7
        defense['defender7'] = 0
        defense['defender7'].loc[np.arange(6, len(defense), 11)] = 1
        defender7 = defense[defense['defender7'] == 1][['GameId','PlayId',
                                         'X_def_dist_to_back','Y_def_dist_to_back',
                                                        'def_dist_to_back',
                                         'V_x','V_y','A']]
        defender7 = defender7.rename(columns={'X_def_dist_to_back':'def7_X_dist',
                                              'Y_def_dist_to_back':'def7_Y_dist',
                                              'def_dist_to_back':'def7_dist',
                                              'V_x':'def7_V_x',
                                              'V_y':'def7_V_y',
                                              'A':'def7_A',
                                              'Orientation':'def7_orientation',
                                              'Dir':'def7_Dir',
                                              'Dis':'def7_Dis'
                                             })
        
        #Defender 8
        defense['defender8'] = 0
        defense['defender8'].loc[np.arange(7, len(defense), 11)] = 1
        defender8 = defense[defense['defender8'] == 1][['GameId','PlayId',
                                         'X_def_dist_to_back','Y_def_dist_to_back',
                                                        'def_dist_to_back',
                                         'V_x','V_y','A']]
        defender8 = defender8.rename(columns={'X_def_dist_to_back':'def8_X_dist',
                                              'Y_def_dist_to_back':'def8_Y_dist',
                                              'def_dist_to_back':'def8_dist',
                                              'V_x':'def8_V_x',
                                              'V_y':'def8_V_y',
                                              'A':'def8_A',
                                              'Orientation':'def8_orientation',
                                              'Dir':'def8_Dir',
                                              'Dis':'def8_Dis'
                                             })
    
        #Perform Aggregate calculations
        defense = defense.groupby(['GameId','PlayId'])                         .agg({'def_dist_to_back':['max','mean','std'],
                              'def_time_to_tackle':['min','max','mean','std'],
                              'X_def_dist_to_back':['min','max','mean','std'],
                              'Y_def_dist_to_back':['min','max','mean','std'],
                              'V_x':['min','max','mean'],
                              'V_y':['min','max','mean']})\
                         .reset_index()
        
        #Rename Columns
        defense.columns = ['GameId','PlayId','def_max_dist','def_mean_dist','def_std_dist',
                           'def_min_ttt','def_max_ttt','def_mean_ttt','def_std_ttt',
                          'X_def_min_dist','X_def_max_dist','X_def_mean_dist','X_def_std_dist',
                          'Y_def_min_dist','Y_def_max_dist','Y_def_mean_dist','Y_def_std_dist',
                          'X_def_min_vel','X_def_max_vel','X_def_mean_vel',
                          'Y_def_min_vel','Y_def_max_vel','Y_def_mean_vel']
    
        #Some sensor values are way off - resulting in a player having a very high time to tackle (they may have registered a very small speed)
        #To correct for this, we clip the time to tackles at reasonable values
        defense['def_max_ttt'].clip(lower=0,upper=25,inplace=True)
        defense['def_min_ttt'].clip(lower=0,upper=5,inplace=True)
        defense['def_std_ttt'].clip(lower=0,upper=11,inplace=True)
        defense['def_mean_ttt'].clip(lower=0,upper=11,inplace=True)
        
        #Merge with individual defender df
        #This puts our defensive back features in order by distance to the rusher - in other words,
        #For every play we know the time to tackle of the 1st through 8th closest defender
        
        #any more than that seemed to add noise
        defense = pd.merge(defense,defender1,on=['GameId','PlayId'],how='inner')
        defense = pd.merge(defense,defender2,on=['GameId','PlayId'],how='inner')
        defense = pd.merge(defense,defender3,on=['GameId','PlayId'],how='inner')
        defense = pd.merge(defense,defender4,on=['GameId','PlayId'],how='inner')
        defense = pd.merge(defense,defender5,on=['GameId','PlayId'],how='inner')
        defense = pd.merge(defense,defender6,on=['GameId','PlayId'],how='inner')
        defense = pd.merge(defense,defender7,on=['GameId','PlayId'],how='inner')
        defense = pd.merge(defense,defender8,on=['GameId','PlayId'],how='inner')
        
        return defense

    #Do basically the same thing we did for the defense, but for offense
    def offense_features(df):
        rusher = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','Team','X','Y','Position']]
        rusher.columns = ['GameId','PlayId','RusherTeam','RusherX','RusherY','RusherPosition']

        offense = pd.merge(df,rusher,on=['GameId','PlayId'],how='inner')
       #TRY - Just to Min Gap, Max Gap, Average Gap -- since different number of gaps
        offense = offense.loc[(offense['Team'] == offense['RusherTeam']) & 
                              (offense['NflId']!=offense['NflIdRusher']) & 
                              (offense['PlayerWeight']>=240)][['GameId','PlayId','X','Y','S',
                                                               'V_x','V_y','Dir','RusherX',
                                                               'A',
                                                               'RusherY','Position',
                                                               'RusherPosition','PlayerWeight']]

        offense['X_off_dist_to_back'] = (offense['X'] - offense['RusherX'])
        offense['Y_off_dist_to_back'] = np.absolute((offense['Y'] - offense['RusherY']))
        offense['off_dist_to_back'] = np.sqrt(offense['X_off_dist_to_back']**2 + offense['Y_off_dist_to_back']**2)
        offense = offense.groupby(['GameId','PlayId'])                        .agg({'off_dist_to_back':['min','max','mean','std'],
                             'V_x':['min','max','mean'],
                             'V_y':['min','max','mean']})\
                        .reset_index()
        offense.columns = ['GameId','PlayId',
                           'off_min_dist','off_max_dist',
                           'off_mean_dist','off_std_dist',
                          'X_off_min_vel','X_off_max_vel','X_off_mean_vel',
                          'Y_off_min_vel','Y_off_max_vel','Y_off_mean_vel']
#         offense = pd.merge(offense,df3,on=['GameId','PlayId'],how='inner')

        return offense
    
    def play_features(df):
        
        
        add_new_feas = []

        ## Height
        df['PlayerHeight_dense'] = df['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))
        
        add_new_feas.append('PlayerHeight_dense')


        #Get the score differential before the play
        df["diffScoreBeforePlay"] = df["HomeScoreBeforePlay"] - df["VisitorScoreBeforePlay"]
        df.loc[(df['TeamOnOffense']=='away'), 'diffScoreBeforePlay'] = -1*df.diffScoreBeforePlay
        add_new_feas.append("diffScoreBeforePlay")
        
        #Get all of our "play" features
        play_features = df[df['NflId'] == df['NflIdRusher']][add_new_feas+['GameId','PlayId','X','Y','S','A','Dis',
                                                                             'Orientation','Dir_rad','Dir_y','Dir_x',
                                                                             'V_y','V_x',
                                                                             'YardLine','Quarter',
                                                                             'Down','Distance',
                                                                             'DefendersInTheBox']].drop_duplicates()

        #Fill our missing values to prevent errors
        play_features.fillna(-999,inplace=True)

            

        return play_features


    #Create a function to merge all of these engineered features into one dataframe
    def combine_features(relative_to_back, offense, defense, play, test=test):
        df = pd.merge(relative_to_back,defense,on=['GameId','PlayId'],how='inner')
        df = pd.merge(df,offense,on=['GameId','PlayId'],how='inner')
        df = pd.merge(df,play,on=['GameId','PlayId'],how='inner')

        if not test:
            df = pd.merge(df, outcomes, on=['GameId','PlayId'], how='inner')

        return df
    
    
    #Standardize features
    df = standardize(df)
    #Get RB features
    back_feats = back_features(df)
    rel_back = features_relative_to_back(df, back_feats)
    #defensive players
    def_feats = defense_features(df)
    #offensive players
    off_feats = offense_features(df)
    #play features
    play_feats = play_features(df)
    #Get our table
    feature_table = combine_features(rel_back, off_feats, def_feats, play_feats, test=test)
    #Plays that use fewer than 4 or more than 11 defenders in the box are anamolies and add noise
    feature_table['DefendersInTheBox'].clip(lower=4,upper=11,inplace=True)
    #I've found that dropping these columns improves my predictions - 
    #hey just add noise to the model, and are likely accounted for with another combination of features
    feature_table.drop(columns=['Down','Quarter','X_def_max_dist',
                            'Y_def_max_dist','Y_def_std_dist',
                            'Y_max_dist','Y_mean_dist',
                            'Y_std_dist','std_dist'],inplace=True)
    return feature_table


# In[9]:


get_ipython().run_line_magic('time', 'train_feats = create_features(train_df, False)')

#May have some infinities from dividing by something close to 0 - make these nan
train_feats.replace([np.inf, -np.inf], np.nan, inplace=True)
#remove any rows with nans we have to prevent the model not running
train_feats = train_feats.dropna()


# In[10]:


#Get a list of all our features we engineered
list(train_feats)


# In[11]:


#Show the first few plays
train_feats.head(10)


# In[12]:


#Get our explanatory variables
X = train_feats.copy()
#Sort them by yards gained
X.sort_values(by='Yards',inplace=True)
yards = np.array(X.Yards)


#We can't have a 99 yard run here, or our next step won't work (will be out of index)
# - it's not important to have an exact 99 yard run, as this is an incredibly rare event (just 1 in our data)
# and no different in practicality than a 98 yard run
yards = np.clip(yards,-14,98)

#Assign probablity values to the run
y = np.zeros((yards.shape[0], 199))
for idx, target in enumerate(list(yards)):
    y[idx][98 + target] = 0.1
    y[idx][99 + target] = 0.8
    y[idx][100 + target] = 0.1

X.drop(['GameId','PlayId','Yards'], axis=1, inplace=True)


# In[13]:


#Standarize our variables. This helps our model train quicker
scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[14]:


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


# In[15]:


def get_model(x_tr,y_tr,x_val,y_val):
    inp = Input(shape = (x_tr.shape[1],))

    x = Dense(512, input_dim=X.shape[1], activation='relu')(inp)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    
    out = Dense(199, activation='softmax')(x)
    model = Model(inp,out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[])

    
    es = EarlyStopping(monitor='CRPS_score_val', 
                       mode='min',
                       restore_best_weights=True, 
                       verbose=1, 
                       patience=10)

    mc = ModelCheckpoint('best_model.h5',monitor='CRPS_score_val',mode='min',
                                   save_best_only=True, verbose=1, save_weights_only=True)
    
    bsz = 32
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


# In[16]:


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
        print(datetime.datetime.now())
        crps_csv.append(crps)
    print("mean crps is %f"%np.mean(crps_csv))
print("Final mean crps is %f"%np.mean(crps_csv))


def predict(x_te):
    model_num = len(models)
    for k,m in enumerate(models):
        if k==0:
#             y_pred = m.predict(x_te,batch_size=1024)
            y_pred = m.predict(x_te,batch_size=32)
        else:
            y_pred+=m.predict(x_te,batch_size=32)
#             y_pred+=m.predict(x_te,batch_size=32)
            
    y_pred = y_pred / model_num
    
    return y_pred


# In[17]:


print("mean crps is %f"%np.mean(crps_csv))


# In[18]:


from kaggle.competitions import nflrush
env = nflrush.make_env()
iter_test = env.iter_test()

for (test_df, sample_prediction_df) in iter_test:
    basetable = create_features(test_df, deploy=True)
    basetable['YardsFromOtherGoal'] = 100 - (basetable['YardLine'])
    yds_max = np.ceil(basetable['YardsFromOtherGoal'].iloc[0])
    yds_max = int(yds_max)
    yds = np.ceil(basetable['back_from_scrimmage'].iloc[0])
    yds = int(yds)
    basetable.drop(['GameId','PlayId','YardsFromOtherGoal'], axis=1, inplace=True)
    scaled_basetable = scaler.transform(basetable)

    y_pred = predict(scaled_basetable)
    y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1).tolist()[0]
    y_pred = np.array(y_pred)
    y_pred[0:(99-(yds_min))] = 0
    y_pred[0:(99-(yds+3))] = 0
    y_pred[(99+yds_max):] = 1
    preds_df = pd.DataFrame(data=[y_pred], columns=sample_prediction_df.columns)
    env.predict(preds_df)

env.write_submission_file()


# In[19]:


from kaggle.competitions import nflrush


# In[20]:


env = nflrush.make_env()
iter_test = env.iter_test()

