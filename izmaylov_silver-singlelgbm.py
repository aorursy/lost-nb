#!/usr/bin/env python
# coding: utf-8

# In[1]:


# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import json

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from xgboost import plot_importance
from catboost import CatBoostRegressor
from matplotlib import pyplot
import shap

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from time import time
from tqdm import tqdm_notebook as tqdm
from collections import Counter
from scipy import stats
import lightgbm as lgb
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
import json
pd.set_option('display.max_columns', 1000)

import datetime

import random
random.seed(1029)
np.random.seed(1029)

import copy
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm import tqdm_notebook
from tqdm import tqdm

import gc
from collections import Counter

from typing import Dict


# In[2]:


get_ipython().run_cell_magic('time', '', "LOCAL = False\n\ndef read_data(LOCAL):\n    if LOCAL:\n        PATH  = 'data/'\n    else:\n        PATH = '/kaggle/input/data-science-bowl-2019/'\n\n    print('Reading train.csv file....')\n    train = pd.read_csv(PATH + 'train.csv')\n    print('Training.csv file have {} rows and {} columns'.format(train.shape[0], train.shape[1]))\n\n    print('Reading test.csv file....')\n    test = pd.read_csv(PATH + 'test.csv')\n    print('Test.csv file have {} rows and {} columns'.format(test.shape[0], test.shape[1]))\n\n    print('Reading train_labels.csv file....')\n    train_labels = pd.read_csv(PATH + 'train_labels.csv')\n    print('Train_labels.csv file have {} rows and {} columns'.format(train_labels.shape[0], train_labels.shape[1]))\n\n    print('Reading specs.csv file....')\n    specs = pd.read_csv(PATH + 'specs.csv')\n    print('Specs.csv file have {} rows and {} columns'.format(specs.shape[0], specs.shape[1]))\n\n    print('Reading sample_submission.csv file....')\n    sample_submission = pd.read_csv(PATH + 'sample_submission.csv')\n    print('Sample_submission.csv file have {} rows and {} columns'.format(sample_submission.shape[0], sample_submission.shape[1]))\n    return train, test, train_labels, specs, sample_submission\n\n# read data\ntrain, test, train_labels, specs, sample_submission = read_data(LOCAL)")


# In[3]:


def encode_title(train, test, train_labels):
    # encode title
    train['title_event_code'] = sorted(list(map(lambda x, y: str(x) + '_' + str(y), train['title'], train['event_code'])))
    test['title_event_code'] = sorted(list(map(lambda x, y: str(x) + '_' + str(y), test['title'], test['event_code'])))
    all_title_event_code = sorted(list(set(train["title_event_code"].unique()).union(test["title_event_code"].unique())))
    # make a list with all the unique 'titles' from the train and test set
    list_of_user_activities = sorted(list(set(train['title'].unique()).union(set(test['title'].unique()))))
    # make a list with all the unique 'event_code' from the train and test set
    list_of_event_code = sorted(list(set(train['event_code'].unique()).union(set(test['event_code'].unique()))))
    list_of_event_id = sorted(list(set(train['event_id'].unique()).union(set(test['event_id'].unique()))))
    # make a list with all the unique worlds from the train and test set
    list_of_worlds = sorted(list(set(train['world'].unique()).union(set(test['world'].unique()))))
    # create a dictionary numerating the titles
    activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))
    activities_labels = dict(zip(np.arange(len(list_of_user_activities)), list_of_user_activities))
    activities_world = dict(zip(list_of_worlds, np.arange(len(list_of_worlds))))
    assess_titles = sorted(list(set(train[train['type'] == 'Assessment']['title'].value_counts().index).union(set(test[test['type'] == 'Assessment']['title'].value_counts().index))))
    # replace the text titles with the number titles from the dict
    train['title'] = train['title'].map(activities_map)
    test['title'] = test['title'].map(activities_map)
    train['world'] = train['world'].map(activities_world)
    test['world'] = test['world'].map(activities_world)
    train_labels['title'] = train_labels['title'].map(activities_map)
    win_code = dict(zip(activities_map.values(), (4100*np.ones(len(activities_map))).astype('int')))
    # then, it set one element, the 'Bird Measurer (Assessment)' as 4110, 10 more than the rest
    win_code[activities_map['Bird Measurer (Assessment)']] = 4110
    # convert text into datetime
    train['timestamp'] = pd.to_datetime(train['timestamp'])
    test['timestamp'] = pd.to_datetime(test['timestamp'])
    
    
    return train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code
# get usefull dict with maping encode
train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code = encode_title(train, test, train_labels)


# In[4]:


def get_data(user_sample, test_set=False):


    users_sample_features = []
    users_sample_featureNames = []
    all_assessments = []
    games_activities = [23,12,26,32,6,18,19,36,1,2,17]
    ass_activities = [8,10,9,4,30]
    activity_activities = [20,5,7,11,21,22,35,42]
    world_activity = {0: [12, 23, 8, 26, 32, 10],
     1: [36, 19, 6, 5, 18, 9], 
     3: [2, 30, 1, 17, 4]}
    all_name = {36: 15,
         19: 3,2: 3,
         30: 1,
         1: 3,
         17: 9, 6: 2,
         5: 8,
         4: 2,
         12: 3,18: 4,9: 2,
         23: 3,8: 1,
         26: 3,32: 8,10: 1}
    cliptitles = ['Balancing Act','Crystal Caves - Level 1', 'Crystal Caves - Level 2', 'Crystal Caves - Level 3','Honey Cake' ,'Lifting Heavy Things',
             'Heavy, Heavier, Heaviest', 'Magma Peak - Level 1', 'Magma Peak - Level 2','Slop Problem','Welcome to Lost Lagoon!',
             'Rulers','Costume Box', '12 Monkeys','Ordering Spheres',"Pirate's Tale",'Treasure Map',
              'Tree Top City - Level 1', 'Tree Top City - Level 2', 'Tree Top City - Level 3']
    eventid_D = ['9c5ef70c',
                 '2ec694de', '86c924c4', 'abc5811c', 'c6971acf', '3bf1cf26', '5be391b5', '3ddc79c3', '30df3273', '90d848e0', '28ed704e', 'd06f75b5', '26fd2d99',
                 '2b9272f4', '6088b756', '45d01abe', 'a8a78786', 'e5c9df6f', '5859dfb6', '9b23e8ee', '9d29771f', 'ecc36b7f', '67aa2ada', 'b7dc8128', 
                 '13f56524', 'e4f1efe6', 'c2baf0bd', '1beb320a', '3a4be871', '5b49460a', 'ec138c1c', '88d4a5be', 'f54238ee', 'e080a381', '5a848010',
                 'a8876db3', 'e5734469', '6f4bd64e', '9ed8f6da', '9e6b7fb5', 'd38c2fd7', 'cb1178ad', '7961e599', '6f4adc4b', 'eb2c19cd', 'c74f40cd', 
                 '0ce40006', '857f21c0', '3edf6747', '1c178d24', 'fbaf3456', '29a42aea', '1375ccb7', '731c0cbe', 'cf7638f3', '0d18d96c', '99ea62f3',
                 '38074c54', 'e57dd7af', '3b2048ee', 'c7fe2a55', '16dffff1', 'd2278a3b', '92687c59', 'a1192f43', '15eb4a7d', '37937459', '3afde5dd',
                 '3bb91dda', '7423acbc', '4b5efe37', '65abac75', '3bb91ced', 'c7f7f0e1', '8d748b58', '6f8106d9', 'a8cc6fec', '46cd75b4', '90ea0bac', 
                 '71e712d8', 'f56e0afc', '4e5fc6f5', '1cf54632', '9b01374f', '83c6c409', '53c6e11a', 'f6947f54', '29bdd9ba', '14de4c5d', '070a5291', 
                 'c1cac9a2', '8ac7cce4', '08fd73f3', 'd3268efa', '9b4001e4', '8f094001', 'f5b8c21a', '15f99afc', '0086365d', 'e4d32835', '19967db1', 
                 '5de79a6a', 'd3640339', '6077cc36', '895865f3', '7d5c30a2', 'd2659ab4', '3afb49e6', 'bfc77bd6', '2c4e6db0', '2a512369', '85d1b0de', 
                 '0413e89d', '99abe2bb', '1575e76c', 'db02c830', 'a76029ee', '46b50ba8', 'ad148f58', '55115cbd', 'd9c005dd', 'a592d54e', 'e04fb33d', 
                 '392e14df', '9de5e594', 'c51d8688', '08ff79ad', '31973d56', '26a5a3dd', '47f43a44', '44cb4907', 'e720d930', 'ecaab346', '6aeafed4', 
                 'dcb1663e', '47efca07', '48349b14', 'c277e121', '17113b36', '04df9b66', '51311d7a', '9554a50b', '763fc34e', 'ab4ec3a4', 'c54cf6c5',
                 '7525289a', '9ce586dd', 'e7561dd2', 'b5053438', 'b74258a0', '4a09ace1', 'df4fe8b6', 'd88ca108', '86ba578b', 'b2e5b0f1', '28a4eb9a',
                 '16667cc5', '63f13dd7', 'b012cd7f', '7040c096', '87d743c1', 'd45ed6a1', '1996c610', '01ca3a3c', '58a0de5c', '85de926c', '7d093bf9', 
                 'a5e9da97', '119b5b02', '93edfe2e', '29f54413', 'e3ff61fb', '5f5b2617', 'dcaede90', 'ac92046e', '05ad839b', '6cf7d25c', 'cc5087a3', 
                 '06372577', '1af8be29', 'daac11b0', '5dc079d8', '73757a5e', '3bfd1a65', '003cd2ee', 'fd20ea40', '5290eab1', 'ad2fc29c', '804ee27f', 
                 '7fd1ac25', '8d84fa81', '1b54d27f', 'f32856e4', '37c53127', '6c930e6e', '6043a2b4', '3dfd4aa4', 'ca11f653', '6f445b57', '7cf1bc53', 
                 '250513af', '1340b8d7', '47026d5f', '611485c5', 'c189aaf2', '56817e2b', '33505eae', 'c7128948', '15ba1109', '17ca3959', '89aace00', 
                 '36fa3ebe', '222660ff', '4074bac2', 'f93fc684', '6d90d394', 'e7e44842', 'b120f2ac', 'dcb55a27', 'd88e8f25', '77c76bc5', 'a6d66e51', 
                 'ecc6157f', '155f62a4', 'ea296733', 'd51b1749', '160654fd', 'cb6010f8', '3393b68b', '1f19558b', '2b058fe3', '25fa8af4', '3323d7e9', 
                 'e64e2cfd', 'e37a2b78', '756e5507', '28520915', '2a444e03', '4d911100', 'f71c4741', '4901243f', '77ead60d']
    eventcode_D = [4050,2050,4235, 4230]
    list_of_event_idD = list(set(list_of_event_id) - set(eventid_D))
    list_of_event_codeD = list(set(list_of_event_code) - set(eventcode_D))
    for idx , (i, session) in enumerate(user_sample.groupby('game_session', sort=False)):
        features = {}
        session_features = []
        user_activities_count = {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}
        event_code_count: Dict[str, int] = {ev: 0 for ev in list_of_event_code}
        corincorids = []
        event_id_count: Dict[str, int] = {eve: 0 for eve in list_of_event_id}
        title_count: Dict[str, int] = {eve: 0 for eve in activities_labels.values()} 
        last_accuracy_title = {'acc_' + title: -1 for title in assess_titles}
        all_name_dict_features = {}
        for x in all_name.items():
            all_name_dict_features['activity_' + str(x[0])] = np.nan
        session_type = session['type'].iloc[0]
        session_title = session['title'].iloc[0]
        session_title_text = activities_labels[session_title]
        session_hour = session.timestamp.iloc[0].time().hour
        session_weekday = session.timestamp.iloc[0].weekday()
        session_world = session['world'].iloc[0]
        timestamp = session['timestamp'].iloc[0].timestamp()
        installation_id = session['installation_id'].iloc[0]
        true_attempts = 0
        false_attempts = 0
        misses_stats = {}
        for tA in games_activities:
            misses_stats[str(tA) + '_num_of_misses'] = 0
        for tA in ass_activities:
            misses_stats[str(tA) + '_num_of_misses'] = 0
        for tA in activity_activities:
            misses_stats[str(tA) + '_num_of_misses'] = 0
        duration_stats = {}
        for tA in games_activities:
            duration_stats[str(tA) + '_duration'] = np.nan
        for tA in ass_activities:
            duration_stats[str(tA) + '_duration'] = np.nan
        for tA in activity_activities:
            duration_stats[str(tA) + '_duration'] = np.nan
        ass_accuracy = {}
        for tA in ass_activities:
            ass_accuracy[str(tA) + '_true_attempts'] = 0
            ass_accuracy[str(tA) + '_false_attempts'] = 0
            ass_accuracy[str(tA) + '_accuracy_in_Ass'] = np.nan
            ass_accuracy[str(tA) + '_true_attempts_dur'] = np.nan
            ass_accuracy[str(tA) + '_false_attempts_dur'] = np.nan
        avg_good_round_stats = {}
        for tA in games_activities:
            avg_good_round_stats[str(tA) + '_num_goodrounds'] = 0
            avg_good_round_stats[str(tA) + '_duration_goodrounds_mean'] = np.nan
            avg_good_round_stats[str(tA) + '_duration_goodrounds_median'] = np.nan
            avg_good_round_stats[str(tA) + '_misses_goodrounds_sum'] = np.nan
            avg_good_round_stats[str(tA) + '_misses_goodrounds_median'] = np.nan
            avg_good_round_stats[str(tA) + '_max_round'] = np.nan
        for tA in ass_activities:
            avg_good_round_stats[str(tA) + '_num_goodrounds'] = 0
            avg_good_round_stats[str(tA) + '_duration_goodrounds_mean'] = np.nan
            avg_good_round_stats[str(tA) + '_duration_goodrounds_median'] = np.nan
            avg_good_round_stats[str(tA) + '_misses_goodrounds_sum'] = np.nan
            avg_good_round_stats[str(tA) + '_misses_goodrounds_median'] = np.nan
            avg_good_round_stats[str(tA) + '_max_round'] = np.nan
        for tA in activity_activities:
            avg_good_round_stats[str(tA) + '_num_goodrounds'] = 0
            avg_good_round_stats[str(tA) + '_duration_goodrounds_mean'] = np.nan
            avg_good_round_stats[str(tA) + '_duration_goodrounds_median'] = np.nan
            avg_good_round_stats[str(tA) + '_misses_goodrounds_sum'] = np.nan
            avg_good_round_stats[str(tA) + '_misses_goodrounds_median'] = np.nan
            avg_good_round_stats[str(tA) + '_max_round'] = np.nan
        cor_incor_stats = {}
        for tA in games_activities:
            cor_incor_stats[str(tA) + '_num_attempts'] = np.nan
            cor_incor_stats[str(tA) + '_share_cor'] = np.nan
            cor_incor_stats[str(tA) + '_first_goodattemp_'] = np.nan
        for tA in ass_activities:
            cor_incor_stats[str(tA) + '_num_attempts'] = np.nan
            cor_incor_stats[str(tA) + '_share_cor'] = np.nan
            cor_incor_stats[str(tA) + '_first_goodattemp_'] = np.nan
        if idx == 0:
            users_sample_featureNames.append('timestamp')
            users_sample_featureNames.extend(list(user_activities_count.keys()))
            users_sample_featureNames.extend(list(event_code_count.keys()))
            users_sample_featureNames.extend(list(event_id_count.keys()))
            users_sample_featureNames.extend(list(title_count.keys()))
            users_sample_featureNames.extend(list(last_accuracy_title.keys()))
            users_sample_featureNames.append('true_attempts')
            users_sample_featureNames.append('false_attempts')
            users_sample_featureNames.append('session_worlds')
            users_sample_featureNames.append('session_hour')
            users_sample_featureNames.append('session_weekday')
            users_sample_featureNames.extend(list(duration_stats.keys())) 
            users_sample_featureNames.extend(list(avg_good_round_stats.keys())) 
            users_sample_featureNames.extend(list(cor_incor_stats.keys()))
            users_sample_featureNames.extend(list(misses_stats.keys()))
            users_sample_featureNames.extend(list(ass_accuracy.keys()))
            users_sample_featureNames.extend(list(all_name_dict_features.keys()))
            users_sample_featureNamesD = {k:idx for idx,k in enumerate(users_sample_featureNames)}
        if (session_type == 'Assessment'):
            all_attempts = session[session.event_code == win_code[session_title]]
            true_attempts = all_attempts['event_data'].str.contains('true').sum()
            false_attempts = all_attempts['event_data'].str.contains('false').sum()
            accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0
            if accuracy == 0:
                accuracy_group = 0
            elif accuracy == 1:
                accuracy_group = 3
            elif accuracy == 0.5:
                accuracy_group = 2
            else:
                accuracy_group = 1
            features['accuracy_group'] = accuracy_group
            features['session_world'] = session_world
            features['installation_id'] = installation_id
            features['session_title'] = session_title
            
            #датафрейм c историей 
            prev_time_periods_min = [60*60*24*3, 60*60*24*100]
            start_assessment_time =  session['timestamp'].iloc[0].timestamp()
            for t in prev_time_periods_min:
                start_of_df = start_assessment_time - t
                df_ = np.array([])
                for idx2, T in enumerate(users_sample_features):
                    if T[0] >= start_of_df:
                        df_ = np.array(users_sample_features[idx2:])
                        break
                got_history = 1 if df_.shape[0] > 0 else 0
                features['got_history_' + str(t)] = got_history        

                if got_history:
                    previous_time_duration = (session['timestamp'].iloc[0].timestamp() - df_[0][0])

                if t in [60*60*24*100]:
                    if got_history:
#                         features['hour_of_first_session_start'] = df_[:,users_sample_featureNamesD['session_hour']][0]
#                         features['weekday_of_first_session_start'] = df_[:,users_sample_featureNamesD['session_weekday']][0]
                        features['time_passed_from_session_started'] = (session['timestamp'].iloc[0].timestamp() - users_sample_features[0][0])/60

                    else:
#                         features['hour_of_first_session_start'] = -100
#                         features['weekday_of_first_session_start'] = -100
                        features['time_passed_from_session_started'] = 0
                if got_history:
                    features['median_hour_session_start_' + str(t)] = np.nanmedian(df_[:,users_sample_featureNamesD['session_hour']])
                    features['std_hour_session_start_' + str(t)] = np.nanstd(df_[:,users_sample_featureNamesD['session_hour']])
                    features['DIFF_median_hour_session_start_' + str(t)] = features['median_hour_session_start_' + str(t)] - session_hour
                    features['ABS_DIFF_median_hour_session_start_' + str(t)] = np.abs(features['median_hour_session_start_' + str(t)] - session_hour)
                else:        
                    features['median_hour_session_start_' + str(t)] = -100
                    features['std_hour_session_start_' + str(t)] = -100
                    features['DIFF_median_hour_session_start_' + str(t)] = -100
                    features['ABS_DIFF_median_hour_session_start_' + str(t)] = -100
                if got_history:
                    features['clip_count_' + str(t)] = df_[:,users_sample_featureNamesD['Clip']].sum() 
                    features['activity_count_' + str(t)] = df_[:,users_sample_featureNamesD['Activity']].sum() 
                    features['ass_count_' + str(t)] = df_[:,users_sample_featureNamesD['Assessment']].sum() 
                    features['game_count_' + str(t)] = df_[:,users_sample_featureNamesD['Game']].sum() 

                    features['FR_clip_count_' + str(t)] = df_[:,users_sample_featureNamesD['Clip']].sum() /previous_time_duration
                    features['FR_activity_count_' + str(t)] = df_[:,users_sample_featureNamesD['Activity']].sum()/previous_time_duration 
                    features['FR_ass_count_' + str(t)] = df_[:,users_sample_featureNamesD['Assessment']].sum() /previous_time_duration
                    features['FR_game_count_' + str(t)] = df_[:,users_sample_featureNamesD['Game']].sum() /previous_time_duration

                    

                    features['share_of_games_' + str(t)] = features['game_count_' + str(t)]/ (features['game_count_' + str(t)] + features['ass_count_' + str(t)] +                                                features['activity_count_' + str(t)] +  features['clip_count_' + str(t)])
                    features['share_of_activities_' + str(t)] = features['activity_count_' + str(t)]/ (features['game_count_' + str(t)] + features['ass_count_' + str(t)] +                                                features['activity_count_' + str(t)] +  features['clip_count_' + str(t)]) 
                    features['share_of_clips_' + str(t)] = features['clip_count_' + str(t)]/ (features['game_count_' + str(t)] + features['ass_count_' + str(t)] +                                                features['activity_count_' + str(t)] +  features['clip_count_' + str(t)]) 
                else:
                    features['clip_count_' + str(t)] = 0
                    features['activity_count_' + str(t)] =  0
                    features['ass_count_' + str(t)] = 0
                    features['game_count_' + str(t)] = 0
                    features['FR_clip_count_' + str(t)] = -100
                    features['FR_activity_count_' + str(t)] = -100
                    features['FR_ass_count_' + str(t)] = -100
                    features['FR_game_count_' + str(t)] = -100

                    features['share_of_games_' + str(t)] = -1
                    features['share_of_activities_' + str(t)] = -1
                    features['share_of_clips_' + str(t)] = -1
                if got_history:
                    for f in last_accuracy_title.keys():
                        temp_D = df_[:,users_sample_featureNamesD[f]]
                        if len(temp_D[temp_D >= 0]) > 0:
                            temp_ = temp_D[temp_D >= 0][-1]
                        else:
                            temp_ = -100
                        features[str(f) + '_' + str(t)] = temp_
                else:
                    for f in last_accuracy_title.keys():
                        features[str(f) + '_' + str(t)] = -100
                #################################################################################
                if t in [60*60*24*3,60*60*24*100]:            
                    #Счетчики счетчиков
                    for f in event_code_count.keys():
                        if f in eventcode_D:
                            continue
                        temp_ = df_[:,users_sample_featureNamesD[f]].sum() if got_history else 0
                        features[str(f) + '_' + str(t)] = temp_
                    for f in event_id_count.keys():
                        if f in eventid_D:
                            continue
                        # удаялем излишние коды                    
                        temp_ = df_[:,users_sample_featureNamesD[f]].sum() if got_history else 0
                        features[str(f) + '_' + str(t)] = temp_
                    for f in title_count.keys():
                        if f in cliptitles:
                            continue
                        else:
                            temp_ = df_[:,users_sample_featureNamesD[f]].sum() if got_history else 0
                            features[str(f) + '_' + str(t)] = temp_
                #################################################################################
                features['accumulated_correct_attempts_' + str(t)] = df_[:,users_sample_featureNamesD['true_attempts']].sum() if got_history else 0
                features['accumulated_uncorrect_attempts_' + str(t)] = df_[:,users_sample_featureNamesD['false_attempts']].sum() if got_history else 0
                features['accumulated_share_coruncor_attempts_' + str(t)] = features['accumulated_correct_attempts_' + str(t)]/                (features['accumulated_uncorrect_attempts_' + str(t)] + features['accumulated_correct_attempts_' + str(t)]) if                (features['accumulated_uncorrect_attempts_' + str(t)] + features['accumulated_correct_attempts_' + str(t)]) > 0 else -1
                features['accumulated_accuracy_' + str(t)] = features['accumulated_correct_attempts_' + str(t)]/                (features['accumulated_correct_attempts_' + str(t)] + features['accumulated_uncorrect_attempts_' + str(t)]) if                 features['accumulated_correct_attempts_' + str(t)] + features['accumulated_uncorrect_attempts_' + str(t)] > 0 else -1
                for f in ass_accuracy.keys():
                    features[str(f) + '_mean_' + str(t)] = np.nanmean(df_[:,users_sample_featureNamesD[f]]) if got_history else -1
                    features[str(f) + '_median_' + str(t)] = np.nanmedian(df_[:,users_sample_featureNamesD[f]]) if got_history else -1
                    if f in ['8_true_attempts', '8_false_attempts',  '10_true_attempts', '10_false_attempts',                              '9_true_attempts', '9_false_attempts', '4_true_attempts',                             '4_false_attempts', '30_true_attempts', '30_false_attempts']:
                        features[str(f) + '_sum_' + str(t)] = np.nansum(df_[:,users_sample_featureNamesD[f]]) if got_history else 0
                true_list = ['8_true_attempts', '10_true_attempts', '9_true_attempts', '4_true_attempts','30_true_attempts']
                false_list = ['8_false_attempts', '10_false_attempts', '9_false_attempts', '4_false_attempts','30_false_attempts']
                for nL in range(5):
                    true_list_f = features[true_list[nL] + '_sum_' + str(t)]
                    false_list_f = features[false_list[nL] + '_sum_' + str(t)]
                    if (true_list_f+false_list_f) != 0:
                        features['accumulated_quality_by_ass_' + true_list[nL]+ '_'+ str(t)] = true_list_f/(true_list_f+false_list_f)
                    else:
                        features['accumulated_quality_by_ass_' + true_list[nL]+ '_'+ str(t)]  = np.nan
                for f in avg_good_round_stats.keys():
                    features[str(f) + '_mean_' + str(t)] = np.nanmean(df_[:,users_sample_featureNamesD[f]]) if got_history else -1
                    features[str(f) + '_median_' + str(t)] = np.nanmedian(df_[:,users_sample_featureNamesD[f]]) if got_history else -1
                    features[str(f) + '_sum_' + str(t)] = np.nansum(df_[:,users_sample_featureNamesD[f]]) if got_history else -1
                    features[str(f) + '_max_' + str(t)] = np.nanmax(df_[:,users_sample_featureNamesD[f]]) if got_history else -1
                for f in cor_incor_stats.keys():
                    features[str(f) + '_mean_' + str(t)] = np.nanmean(df_[:,users_sample_featureNamesD[f]]) if got_history else -1
                    features[str(f) + '_median_' + str(t)] = np.nanmedian(df_[:,users_sample_featureNamesD[f]]) if got_history else -1
                    features[str(f) + '_sum_' + str(t)] = np.nansum(df_[:,users_sample_featureNamesD[f]])  if got_history else -1
                for f in misses_stats.keys():
                    features[str(f) + '_total_misses_'  +str(t)] = np.nansum(list(df_[:,users_sample_featureNamesD[f]])) if got_history else -1
                for f in duration_stats.keys():
                    features[str(f) + '_mean_'  +str(t)] = np.nanmean(list(df_[:,users_sample_featureNamesD[f]])) if got_history else -1
                    features[str(f) + '_sum_'  +str(t)] = np.nansum(list(df_[:,users_sample_featureNamesD[f]])) if got_history else -1
                    features[str(f) + '_median_'  +str(t)] = np.nanmedian(list(df_[:,users_sample_featureNamesD[f]])) if got_history else -1
                    features[str(f) + '_90p_'  +str(t)] = np.nanpercentile(list(df_[:,users_sample_featureNamesD[f]]),90) if got_history else -1
                if got_history:
                    G_W = df_[(df_[:,users_sample_featureNamesD['Game']] == 1) & (df_[:,users_sample_featureNamesD['session_worlds']] == session_world)]
                    features['num_of_games_in_world_' + str(t)] = len(G_W) if len(G_W)>0 else 0
                else:
                    features['num_of_games_in_world_' + str(t)] = 0
                for f in all_name_dict_features.keys():
                    features[str(f) + '_median_' + str(t)] = np.nanmedian(df_[:,users_sample_featureNamesD[f]]) if got_history else -1
                    features[str(f) + '_max_' + str(t)] = np.nanmax(df_[:,users_sample_featureNamesD[f]])  if got_history else -1
            last_accuracy_title['acc_' + session_title_text] = accuracy
            if test_set:
                all_assessments.append(features)
            elif true_attempts+false_attempts > 0:
                all_assessments.append(features)
            del features
        #################################################################################################################
        #################################################################################################################
        #################################################################################################################
        def update_counters(counter: dict, col: str):
                num_of_session_count = Counter(session[col])
                for k in num_of_session_count.keys():
                    x = k
                    if col == 'title':
                        x = activities_labels[k]
                    counter[x] += num_of_session_count[k]
                return counter
        #Начало сбора фичей для каждой отдельной сессии
        #Счётчики
        user_activities_count[session_type] = 1
        event_code_count = update_counters(event_code_count, "event_code")
        event_id_count = update_counters(event_id_count, "event_id")
        title_count = update_counters(title_count, 'title')
        duration_D = (session.iloc[-1, 2] - session.iloc[0, 2]).seconds + 1 
    #     2030 - корректно выполнил раунд
    #     3020 - ошибка в действии
    #     3021 - правильное действие
        #Промежуточная статистика в завершенных играх, активностях и ассесментах
        good_end_of_round = session[session.event_code == 2030]
        cors = session[session.event_code == 3021]
        incors = session[session.event_code == 3020]
        misses = session[session.event_code == 4070]
        if (session_title in ass_activities):
            tA = session_title
            all_attempts = session[session.event_code == win_code[session_title]]
            true_attempts = all_attempts['event_data'].str.contains('true').sum()
            false_attempts = all_attempts['event_data'].str.contains('false').sum()
            accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else np.nan
            ass_accuracy[str(tA) + '_true_attempts'] = true_attempts
            ass_accuracy[str(tA) + '_false_attempts'] = false_attempts
            ass_accuracy[str(tA) + '_accuracy_in_Ass'] = accuracy
            ass_accuracy[str(tA) + '_true_attempts_dur'] = true_attempts/duration_D
            ass_accuracy[str(tA) + '_false_attempts_dur'] = false_attempts/duration_D
        if misses.shape[0] > 0:
            if session_title in games_activities:
                tA = session_title
                misses_stats[str(tA) + '_num_of_misses'] = misses.shape[0]
        if session_title in ass_activities:
                tA = session_title
                misses_stats[str(tA) + '_num_of_misses'] = misses.shape[0]
        if session_title in activity_activities:
                tA = session_title
                misses_stats[str(tA) + '_num_of_misses'] = misses.shape[0]
        if good_end_of_round.shape[0] > 0:
            duration_of_good_rounds = good_end_of_round['event_data'].apply(lambda x: json.loads(x)['duration'])
            if (session_title in ass_activities) or (session_title in games_activities) or  (session_title in activity_activities):
                tA = session_title
                avg_good_round_stats[str(tA) + '_num_goodrounds'] = good_end_of_round.shape[0]
                avg_good_round_stats[str(tA) + '_duration_goodrounds_mean'] = np.mean(duration_of_good_rounds)
                avg_good_round_stats[str(tA) + '_duration_goodrounds_median'] = np.median(duration_of_good_rounds)
                try:
                    num_of_round = good_end_of_round['event_data'].apply(lambda x: json.loads(x)['round'])
                    avg_good_round_stats[str(tA) + '_max_round'] = np.max(num_of_round)
                except:
                    try:
                        num_of_round = good_end_of_round['event_data'].apply(lambda x: json.loads(x)['round_number'])
                        avg_good_round_stats[str(tA) + '_max_round'] = np.max(num_of_round)
                    except:
                        avg_good_round_stats[str(tA) + '_max_round'] = np.nan
                try:            
                    misses_of_good_rounds = good_end_of_round['event_data'].apply(lambda x: json.loads(x)['misses'])
                    avg_good_round_stats[str(tA) + '_misses_goodrounds_sum'] = np.sum(misses_of_good_rounds)
                    avg_good_round_stats[str(tA) + '_misses_goodrounds_median'] = np.median(misses_of_good_rounds)
                except:
                    avg_good_round_stats[str(tA) + '_misses_goodrounds_sum'] = np.nan
                    avg_good_round_stats[str(tA) + '_misses_goodrounds_median'] = np.nan
        #Статистика по правльным и неправильным дейтсвиям
        if cors.shape[0] + incors.shape[0] > 0:
            if (session_title in ass_activities) or (session_title in games_activities):
                tA = session_title
                cor_incor_stats[str(tA) + '_num_attempts'] = cors.shape[0] + incors.shape[0]
                cor_incor_stats[str(tA) + '_share_cor'] = cors.shape[0]/(cor_incor_stats[str(tA) + '_num_attempts'])
                if (incors.shape[0] == 0) :
                    cor_incor_stats[str(tA) + '_first_goodattemp_'] = 1
                elif (cors.shape[0] > 0) and (cors['timestamp'].iloc[0] < incors['timestamp'].iloc[0]):
                    cor_incor_stats[str(tA) + '_first_goodattemp_'] = 1
                else:
                    cor_incor_stats[str(tA) + '_first_goodattemp_'] = 0
        if session_title in games_activities:
            tA = session_title
            duration_stats[str(tA) + '_duration'] = duration_D
        if session_title in ass_activities:
            tA = session_title
            duration_stats[str(tA) + '_duration'] = duration_D
        if session_title in activity_activities:
            tA = session_title
            duration_stats[str(tA) + '_duration'] = duration_D
        #как хорошо справляется игрок с играми, порходит, играет
        if (session_title in ass_activities) or (session_title in games_activities):
            tA = session_title
            all_name_dict_features['activity_' + str(tA)] = 0
            if good_end_of_round.shape[0] > 0 :
                try:
                    rounds_ = good_end_of_round['event_data'].apply(lambda x: json.loads(x)['round'])
                    max_round = np.max(rounds_)
                    all_name_dict_features['activity_' + str(tA)] = max_round/all_name[tA]
                except:
                    try:
                        rounds_ = good_end_of_round['event_data'].apply(lambda x: json.loads(x)['round_number'])
                        max_round = np.max(rounds_)
                        all_name_dict_features['activity_' + str(tA)] = max_round/all_name[tA]
                    except:
                        all_name_dict_features['activity_' + str(tA)] = np.nan
        session_features.append(timestamp)
        session_features.extend(list(user_activities_count.values()))
        session_features.extend(list(event_code_count.values()))
        session_features.extend(list(event_id_count.values()))
        session_features.extend(list(title_count.values()))
        session_features.extend(list(last_accuracy_title.values()))
        session_features.append(true_attempts)
        session_features.append(false_attempts)
        session_features.append(session_world)
        session_features.append(session_hour)
        session_features.append(session_weekday)
        session_features.extend(list(duration_stats.values()))
        session_features.extend(list(avg_good_round_stats.values()))
        session_features.extend(list(cor_incor_stats.values()))  
        session_features.extend(list(misses_stats.values()))
        session_features.extend(list(ass_accuracy.values()))
        session_features.extend(list(all_name_dict_features.values()))
        users_sample_features.append(session_features)   
        del session_features
    
                            
    # if it't the test_set, only the last assessment must be predicted, the previous are scraped
    if test_set:
        return all_assessments[-1]
    # in the train_set, all assessments goes to the dataset
    return all_assessments


# In[5]:


asessment_activity_dict = {
    
    # 'Mushroom Sorter (Assessment)'
    30: {'game' : [2],
         'activity': [ 22]},
    
    # 'Bird Measurer (Assessment)'
    4: {'game': [17],
        'activity': [7]},
    
    # 'Cauldron Filler (Assessment)'
    9: {'game': [19],
        'activity': [42]},
    
    # 'Cart Balancer (Assessment)'
    8: {'game': [23],
        'activity': [11]},
    
    # 'Chest Sorter (Assessment)'
    10: {'game': [32],
        'activity': [20]},
}

def calculate_features_per_row(row):
    
    ass = int(row['session_title'])
    
    ## asessments
    row['relevant_asessment_share_cor_median_8640000'] = row[str(ass) + '_share_cor_median_8640000']
    row['relevant_asessment_accuracy_in_Ass_median_8640000'] = row[str(ass) + '_accuracy_in_Ass_median_8640000']
    
    # num_goodrounds
    features_game = [str(game) + '_num_goodrounds_mean_8640000'  for game in asessment_activity_dict[ass]['game']]
    features_act = [str(act) + '_num_goodrounds_mean_8640000'  for act in asessment_activity_dict[ass]['activity']]
    row['relevant_game_num_goodrounds_8640000__sum'] = np.sum(row[features_game])
    row['relevant_activity_num_goodrounds_8640000__sum'] = np.sum(row[features_act])
    
    features_game = [str(game) + '_num_goodrounds_mean_259200'  for game in asessment_activity_dict[ass]['game']]
    row['relevant_game_num_goodrounds_259200__sum'] = np.sum(row[features_game])
    
    # duration_goodrounds_median
    features_game = [str(game) + '_duration_goodrounds_median_mean_8640000'  for game in asessment_activity_dict[ass]['game']]
    features_act = [str(act) + '_duration_goodrounds_median_mean_8640000'  for act in asessment_activity_dict[ass]['activity']]
    row['relevant_game_duration_goodrounds_median_8640000__mean'] = np.mean(row[features_game])
    row['relevant_activity_duration_goodrounds_median_8640000__mean'] = np.mean(row[features_act])
    
    # misses_goodrounds_median
    features_game = [str(game) + '_misses_goodrounds_median_mean_8640000'  for game in asessment_activity_dict[ass]['game']]
    features_act = [str(act) + '_misses_goodrounds_median_mean_8640000'  for act in asessment_activity_dict[ass]['activity']]
    row['relevant_game_misses_goodrounds_median_8640000__mean'] = np.mean(row[features_game])
    row['relevant_activity_misses_goodrounds_median_8640000__mean'] = np.mean(row[features_act])
    
    # max_round
    features_game = [str(game) + '_max_round_mean_8640000'  for game in asessment_activity_dict[ass]['game']]
    features_act = [str(act) + '_max_round_mean_8640000'  for act in asessment_activity_dict[ass]['activity']]
    row['relevant_game_max_round_8640000__mean'] = np.mean(row[features_game])
    row['relevant_activity_max_round_8640000__mean'] = np.mean(row[features_act])

    # share_cor
    features_game = [str(game) + '_share_cor_mean_8640000'  for game in asessment_activity_dict[ass]['game']]
    row['relevant_game_share_cor_8640000__mean'] = np.mean(row[features_game])
    
    features_game = [str(game) + '_share_cor_mean_259200'  for game in asessment_activity_dict[ass]['game']]
    row['relevant_game_share_cor_259200__mean'] = np.mean(row[features_game])
    
    # num_attempts
    features_game = [str(game) + '_num_attempts_mean_8640000'  for game in asessment_activity_dict[ass]['game']]
    row['relevant_game_num_attempts_8640000__mean'] = np.mean(row[features_game])
    row['relevant_game_num_attempts_8640000__sum'] = np.sum(row[features_game])
    
    features_game = [str(game) + '_num_attempts_mean_259200'  for game in asessment_activity_dict[ass]['game']]
    row['relevant_game_num_attempts_259200__sum'] = np.sum(row[features_game])
    
    return row

def add_features(df):
    
    df['relevant_asessment_share_cor_median_8640000'] = -100
    df['relevant_asessment_accuracy_in_Ass_median_8640000'] = -100
    
    df['relevant_game_num_goodrounds_8640000__sum'] = -100
    df['relevant_activity_num_goodrounds_8640000__sum'] = -100
    
    df['relevant_game_duration_goodrounds_median_8640000__mean'] = -100
    df['relevant_activity_duration_goodrounds_median_8640000__mean'] = -100
    df['relevant_game_misses_goodrounds_median_8640000__mean'] = -100
    df['relevant_activity_misses_goodrounds_median_8640000__mean'] = -100
    df['relevant_activity_max_round_8640000__mean'] = -100
    df['relevant_game_share_cor_8640000__mean'] = -100
    df['relevant_game_num_attempts_8640000__mean'] = -100
    df['relevant_game_num_attempts_8640000__sum'] = -100
    
    
    df['relevant_game_num_attempts_259200__sum'] = -100
    df['relevant_game_num_goodrounds_259200__sum'] = -100
    df['relevant_game_share_cor_259200__mean'] = -100

    df = df.apply(lambda row: calculate_features_per_row(row), axis=1)
    
    return df


# In[6]:


# categoricals = ['session_title','hour','weekday']
categoricals = ['session_title','session_world']
def get_train(train):
    compiled_train = []
    for i, (ins_id, user_sample) in tqdm(enumerate(train.groupby('installation_id', sort = False)), total = 17000):
        compiled_train += get_data(user_sample)
    reduce_train = pd.DataFrame(compiled_train)
    return reduce_train

def get_test(test):
    compiled_test = []
    for ins_id, user_sample in tqdm(test.groupby('installation_id', sort = False), total = 1000):
        test_data = get_data(user_sample, test_set = True)
        compiled_test.append(test_data)
    reduce_test = pd.DataFrame(compiled_test)
    return reduce_test


# In[7]:


reduce_test = get_test(test)
del test
gc.collect()


# In[8]:


reduce_test = add_features(reduce_test)
gc.collect()


# In[9]:


reduce_train = get_train(train)
del train
gc.collect()


# In[10]:


reduce_train = add_features(reduce_train)
gc.collect()


# In[11]:


# counts = [list_of_event_code, list_of_event_id, activities_labels]
# time_periods = [60*60*24*3, 60*60*24*100]
# columns_to_clip = []
# for t in time_periods:
#     columns_to_clip.extend(list(map(lambda x: str(x) + '_' + str(t), counts[0])))
#     columns_to_clip.extend(list(map(lambda x: str(x) + '_' + str(t), counts[1])))
#     columns_to_clip.extend(list(map(lambda x: str(x) + '_' + str(t), counts[2].values())))
# columns_to_clip = [col for col in columns_to_clip if col in reduce_train.columns] + [col for col in reduce_train.columns if 'count' in col]
# values_to_clip_to = reduce_train[columns_to_clip].quantile(0.75)
# reduce_train[columns_to_clip] = reduce_train[columns_to_clip].clip(upper=values_to_clip_to, axis=1)
# reduce_test[columns_to_clip] = reduce_test[columns_to_clip].clip(upper=values_to_clip_to, axis=1)


# In[12]:


import pickle
with open('reduce_train.pickle', 'wb') as f:
    pickle.dump(reduce_train, f)
with open('reduce_test.pickle', 'wb') as f:
    pickle.dump(reduce_test, f)


# In[13]:


features = reduce_train.loc[(reduce_train.sum(axis=1) != 0), (reduce_train.sum(axis=0) != 0)].columns # delete useless columns
features = [x for x in features if x not in ['accuracy_group', 'installation_id']] + ['acc_' + title for title in assess_titles]


# In[14]:


to_exclude = ['10_duration_goodrounds_mean_sum_259200',
'10_duration_goodrounds_median_mean_259200',
'10_duration_goodrounds_median_median_259200',
'10_duration_goodrounds_median_sum_259200',
'10_false_attempts_median_259200',
'10_false_attempts_median_8640000',
'10_false_attempts_sum_259200',
'10_first_goodattemp__median_259200',
'10_first_goodattemp__median_8640000',
'10_first_goodattemp__sum_259200',
'10_first_goodattemp__sum_8640000',
'10_max_round_max_259200',
'10_max_round_max_8640000',
'10_max_round_mean_259200',
'10_max_round_mean_8640000',
'10_max_round_median_259200',
'10_max_round_median_8640000',
'10_max_round_sum_259200',
'10_max_round_sum_8640000',
'10_misses_goodrounds_median_max_259200',
'10_misses_goodrounds_median_mean_259200',
'10_misses_goodrounds_median_median_259200',
'10_misses_goodrounds_median_median_8640000',
'10_misses_goodrounds_median_sum_259200',
'10_misses_goodrounds_median_sum_8640000',
'10_misses_goodrounds_sum_median_259200',
'10_misses_goodrounds_sum_sum_259200',
'10_misses_goodrounds_sum_sum_8640000',
'10_num_goodrounds_max_259200',
'10_num_goodrounds_max_8640000',
'10_num_goodrounds_median_259200',
'10_num_goodrounds_median_8640000',
'10_num_goodrounds_sum_259200',
'10_num_goodrounds_sum_8640000',
'10_true_attempts_median_259200',
'10_true_attempts_median_8640000',
'10_true_attempts_sum_259200',
'10_true_attempts_sum_8640000',
'11_duration_goodrounds_mean_max_259200',
'11_duration_goodrounds_mean_max_8640000',
'11_duration_goodrounds_mean_mean_259200',
'11_duration_goodrounds_mean_mean_8640000',
'11_duration_goodrounds_mean_median_259200',
'11_duration_goodrounds_mean_median_8640000',
'11_duration_goodrounds_mean_sum_259200',
'11_duration_goodrounds_mean_sum_8640000',
'11_duration_goodrounds_median_max_259200',
'11_duration_goodrounds_median_max_8640000',
'11_duration_goodrounds_median_mean_259200',
'11_duration_goodrounds_median_mean_8640000',
'11_duration_goodrounds_median_median_259200',
'11_duration_goodrounds_median_median_8640000',
'11_duration_goodrounds_median_sum_259200',
'11_duration_goodrounds_median_sum_8640000',
'11_max_round_max_259200',
'11_max_round_max_8640000',
'11_max_round_mean_259200',
'11_max_round_mean_8640000',
'11_max_round_median_259200',
'11_max_round_median_8640000',
'11_max_round_sum_259200',
'11_max_round_sum_8640000',
'11_misses_goodrounds_median_max_259200',
'11_misses_goodrounds_median_max_8640000',
'11_misses_goodrounds_median_mean_259200',
'11_misses_goodrounds_median_mean_8640000',
'11_misses_goodrounds_median_median_259200',
'11_misses_goodrounds_median_median_8640000',
'11_misses_goodrounds_median_sum_259200',
'11_misses_goodrounds_median_sum_8640000',
'11_misses_goodrounds_sum_max_259200',
'11_misses_goodrounds_sum_max_8640000',
'11_misses_goodrounds_sum_mean_259200',
'11_misses_goodrounds_sum_mean_8640000',
'11_misses_goodrounds_sum_median_259200',
'11_misses_goodrounds_sum_median_8640000',
'11_misses_goodrounds_sum_sum_259200',
'11_misses_goodrounds_sum_sum_8640000',
'11_num_goodrounds_max_259200',
'11_num_goodrounds_max_8640000',
'11_num_goodrounds_mean_259200',
'11_num_goodrounds_mean_8640000',
'11_num_goodrounds_median_259200',
'11_num_goodrounds_median_8640000',
'11_num_goodrounds_sum_259200',
'11_num_goodrounds_sum_8640000',
'12_first_goodattemp__sum_259200',
'12_max_round_sum_259200',
'12_num_goodrounds_median_259200',
'12_num_goodrounds_median_8640000',
'17_first_goodattemp__median_259200',
'17_first_goodattemp__median_8640000',
'17_first_goodattemp__sum_259200',
'17_first_goodattemp__sum_8640000',
'17_max_round_median_259200',
'17_max_round_sum_259200',
'17_max_round_sum_8640000',
'17_misses_goodrounds_median_median_259200',
'17_misses_goodrounds_median_median_8640000',
'17_misses_goodrounds_median_sum_259200',
'17_misses_goodrounds_median_sum_8640000',
'17_misses_goodrounds_sum_sum_259200',
'17_num_goodrounds_max_259200',
'17_num_goodrounds_median_259200',
'17_num_goodrounds_median_8640000',
'17_num_goodrounds_sum_259200',
'17_num_goodrounds_sum_8640000',
'18_first_goodattemp__mean_259200',
'18_first_goodattemp__mean_8640000',
'18_first_goodattemp__median_259200',
'18_first_goodattemp__median_8640000',
'18_first_goodattemp__sum_259200',
'18_first_goodattemp__sum_8640000',
'18_max_round_sum_259200',
'18_max_round_sum_8640000',
'18_misses_goodrounds_median_max_259200',
'18_misses_goodrounds_median_sum_259200',
'18_misses_goodrounds_median_sum_8640000',
'18_misses_goodrounds_sum_sum_259200',
'18_num_goodrounds_median_259200',
'18_num_goodrounds_median_8640000',
'18_num_goodrounds_sum_259200',
'19_first_goodattemp__median_259200',
'19_first_goodattemp__sum_259200',
'19_first_goodattemp__sum_8640000',
'19_max_round_sum_259200',
'19_misses_goodrounds_median_max_259200',
'19_misses_goodrounds_median_median_259200',
'19_misses_goodrounds_median_sum_259200',
'19_misses_goodrounds_median_sum_8640000',
'19_misses_goodrounds_sum_median_259200',
'19_misses_goodrounds_sum_sum_259200',
'19_num_goodrounds_median_259200',
'19_num_goodrounds_median_8640000',
'19_num_goodrounds_sum_259200',
'1_duration_goodrounds_mean_sum_259200',
'1_duration_goodrounds_median_sum_259200',
'1_first_goodattemp__mean_259200',
'1_first_goodattemp__median_259200',
'1_first_goodattemp__sum_259200',
'1_first_goodattemp__sum_8640000',
'1_max_round_sum_259200',
'1_max_round_sum_8640000',
'1_misses_goodrounds_median_mean_259200',
'1_misses_goodrounds_median_median_259200',
'1_misses_goodrounds_median_median_8640000',
'1_misses_goodrounds_median_sum_259200',
'1_misses_goodrounds_median_sum_8640000',
'1_misses_goodrounds_sum_sum_259200',
'1_misses_goodrounds_sum_sum_8640000',
'1_num_attempts_median_259200',
'1_num_attempts_sum_259200',
'1_num_goodrounds_max_259200',
'1_num_goodrounds_max_8640000',
'1_num_goodrounds_median_259200',
'1_num_goodrounds_median_8640000',
'1_num_goodrounds_sum_259200',
'1_num_goodrounds_sum_8640000',
'20_duration_goodrounds_mean_max_259200',
'20_duration_goodrounds_mean_max_8640000',
'20_duration_goodrounds_mean_mean_259200',
'20_duration_goodrounds_mean_mean_8640000',
'20_duration_goodrounds_mean_median_259200',
'20_duration_goodrounds_mean_median_8640000',
'20_duration_goodrounds_mean_sum_259200',
'20_duration_goodrounds_mean_sum_8640000',
'20_duration_goodrounds_median_max_259200',
'20_duration_goodrounds_median_max_8640000',
'20_duration_goodrounds_median_mean_259200',
'20_duration_goodrounds_median_mean_8640000',
'20_duration_goodrounds_median_median_259200',
'20_duration_goodrounds_median_median_8640000',
'20_duration_goodrounds_median_sum_259200',
'20_duration_goodrounds_median_sum_8640000',
'20_max_round_max_259200',
'20_max_round_max_8640000',
'20_max_round_mean_259200',
'20_max_round_mean_8640000',
'20_max_round_median_259200',
'20_max_round_median_8640000',
'20_max_round_sum_259200',
'20_max_round_sum_8640000',
'20_misses_goodrounds_median_max_259200',
'20_misses_goodrounds_median_max_8640000',
'20_misses_goodrounds_median_mean_259200',
'20_misses_goodrounds_median_mean_8640000',
'20_misses_goodrounds_median_median_259200',
'20_misses_goodrounds_median_median_8640000',
'20_misses_goodrounds_median_sum_259200',
'20_misses_goodrounds_median_sum_8640000',
'20_misses_goodrounds_sum_max_259200',
'20_misses_goodrounds_sum_max_8640000',
'20_misses_goodrounds_sum_mean_259200',
'20_misses_goodrounds_sum_mean_8640000',
'20_misses_goodrounds_sum_median_259200',
'20_misses_goodrounds_sum_median_8640000',
'20_misses_goodrounds_sum_sum_259200',
'20_misses_goodrounds_sum_sum_8640000',
'20_num_goodrounds_max_259200',
'20_num_goodrounds_max_8640000',
'20_num_goodrounds_mean_259200',
'20_num_goodrounds_mean_8640000',
'20_num_goodrounds_median_259200',
'20_num_goodrounds_median_8640000',
'20_num_goodrounds_sum_259200',
'20_num_goodrounds_sum_8640000',
'21_duration_goodrounds_mean_max_259200',
'21_duration_goodrounds_mean_max_8640000',
'21_duration_goodrounds_mean_mean_259200',
'21_duration_goodrounds_mean_mean_8640000',
'21_duration_goodrounds_mean_median_259200',
'21_duration_goodrounds_mean_median_8640000',
'21_duration_goodrounds_mean_sum_259200',
'21_duration_goodrounds_mean_sum_8640000',
'21_duration_goodrounds_median_max_259200',
'21_duration_goodrounds_median_max_8640000',
'21_duration_goodrounds_median_mean_259200',
'21_duration_goodrounds_median_mean_8640000',
'21_duration_goodrounds_median_median_259200',
'21_duration_goodrounds_median_median_8640000',
'21_duration_goodrounds_median_sum_259200',
'21_duration_goodrounds_median_sum_8640000',
'21_max_round_max_259200',
'21_max_round_max_8640000',
'21_max_round_mean_259200',
'21_max_round_mean_8640000',
'21_max_round_median_259200',
'21_max_round_median_8640000',
'21_max_round_sum_259200',
'21_max_round_sum_8640000',
'21_misses_goodrounds_median_max_259200',
'21_misses_goodrounds_median_max_8640000',
'21_misses_goodrounds_median_mean_259200',
'21_misses_goodrounds_median_mean_8640000',
'21_misses_goodrounds_median_median_259200',
'21_misses_goodrounds_median_median_8640000',
'21_misses_goodrounds_median_sum_259200',
'21_misses_goodrounds_median_sum_8640000',
'21_misses_goodrounds_sum_max_259200',
'21_misses_goodrounds_sum_max_8640000',
'21_misses_goodrounds_sum_mean_259200',
'21_misses_goodrounds_sum_mean_8640000',
'21_misses_goodrounds_sum_median_259200',
'21_misses_goodrounds_sum_median_8640000',
'21_misses_goodrounds_sum_sum_259200',
'21_misses_goodrounds_sum_sum_8640000',
'21_num_goodrounds_max_259200',
'21_num_goodrounds_max_8640000',
'21_num_goodrounds_mean_259200',
'21_num_goodrounds_mean_8640000',
'21_num_goodrounds_median_259200',
'21_num_goodrounds_median_8640000',
'21_num_goodrounds_sum_259200',
'21_num_goodrounds_sum_8640000',
'22_duration_goodrounds_mean_max_259200',
'22_duration_goodrounds_mean_max_8640000',
'22_duration_goodrounds_mean_mean_259200',
'22_duration_goodrounds_mean_mean_8640000',
'22_duration_goodrounds_mean_median_259200',
'22_duration_goodrounds_mean_median_8640000',
'22_duration_goodrounds_mean_sum_259200',
'22_duration_goodrounds_mean_sum_8640000',
'22_duration_goodrounds_median_max_259200',
'22_duration_goodrounds_median_max_8640000',
'22_duration_goodrounds_median_mean_259200',
'22_duration_goodrounds_median_mean_8640000',
'22_duration_goodrounds_median_median_259200',
'22_duration_goodrounds_median_median_8640000',
'22_duration_goodrounds_median_sum_259200',
'22_duration_goodrounds_median_sum_8640000',
'22_max_round_max_259200',
'22_max_round_max_8640000',
'22_max_round_mean_259200',
'22_max_round_mean_8640000',
'22_max_round_median_259200',
'22_max_round_median_8640000',
'22_max_round_sum_259200',
'22_max_round_sum_8640000',
'22_misses_goodrounds_median_max_259200',
'22_misses_goodrounds_median_max_8640000',
'22_misses_goodrounds_median_mean_259200',
'22_misses_goodrounds_median_mean_8640000',
'22_misses_goodrounds_median_median_259200',
'22_misses_goodrounds_median_median_8640000',
'22_misses_goodrounds_median_sum_259200',
'22_misses_goodrounds_median_sum_8640000',
'22_misses_goodrounds_sum_max_259200',
'22_misses_goodrounds_sum_max_8640000',
'22_misses_goodrounds_sum_mean_259200',
'22_misses_goodrounds_sum_mean_8640000',
'22_misses_goodrounds_sum_median_259200',
'22_misses_goodrounds_sum_median_8640000',
'22_misses_goodrounds_sum_sum_259200',
'22_misses_goodrounds_sum_sum_8640000',
'22_num_goodrounds_max_259200',
'22_num_goodrounds_max_8640000',
'22_num_goodrounds_mean_259200',
'22_num_goodrounds_mean_8640000',
'22_num_goodrounds_median_259200',
'22_num_goodrounds_median_8640000',
'22_num_goodrounds_sum_259200',
'22_num_goodrounds_sum_8640000',
'23_first_goodattemp__median_259200',
'23_first_goodattemp__median_8640000',
'23_first_goodattemp__sum_259200',
'23_first_goodattemp__sum_8640000',
'23_max_round_sum_259200',
'23_max_round_sum_8640000',
'23_misses_goodrounds_median_sum_259200',
'23_num_goodrounds_max_259200',
'23_num_goodrounds_median_259200',
'23_num_goodrounds_median_8640000',
'23_num_goodrounds_sum_259200',
'26_first_goodattemp__median_259200',
'26_first_goodattemp__sum_259200',
'26_first_goodattemp__sum_8640000',
'26_max_round_max_259200',
'26_max_round_sum_259200',
'26_max_round_sum_8640000',
'26_misses_goodrounds_median_max_259200',
'26_misses_goodrounds_median_median_259200',
'26_misses_goodrounds_median_sum_259200',
'26_misses_goodrounds_median_sum_8640000',
'26_misses_goodrounds_sum_sum_259200',
'26_num_attempts_sum_259200',
'26_num_goodrounds_median_259200',
'26_num_goodrounds_median_8640000',
'26_num_goodrounds_sum_259200',
'2_first_goodattemp__median_259200',
'2_first_goodattemp__sum_259200',
'2_first_goodattemp__sum_8640000',
'2_max_round_sum_259200',
'2_misses_goodrounds_median_sum_259200',
'2_num_goodrounds_median_259200',
'2_num_goodrounds_median_8640000',
'2_num_goodrounds_sum_259200',
'30_false_attempts_median_259200',
'30_false_attempts_median_8640000',
'30_false_attempts_sum_259200',
'30_first_goodattemp__mean_259200',
'30_first_goodattemp__median_259200',
'30_first_goodattemp__median_8640000',
'30_first_goodattemp__sum_259200',
'30_first_goodattemp__sum_8640000',
'30_max_round_max_259200',
'30_max_round_max_8640000',
'30_max_round_mean_259200',
'30_max_round_mean_8640000',
'30_max_round_median_259200',
'30_max_round_median_8640000',
'30_max_round_sum_259200',
'30_max_round_sum_8640000',
'30_misses_goodrounds_median_max_259200',
'30_misses_goodrounds_median_sum_259200',
'30_misses_goodrounds_median_sum_8640000',
'30_misses_goodrounds_sum_sum_259200',
'30_misses_goodrounds_sum_sum_8640000',
'30_num_goodrounds_max_259200',
'30_num_goodrounds_max_8640000',
'30_num_goodrounds_median_259200',
'30_num_goodrounds_median_8640000',
'30_num_goodrounds_sum_259200',
'30_num_goodrounds_sum_8640000',
'30_true_attempts_median_259200',
'30_true_attempts_median_8640000',
'30_true_attempts_sum_259200',
'30_true_attempts_sum_8640000',
'32_first_goodattemp__median_259200',
'32_first_goodattemp__sum_259200',
'32_first_goodattemp__sum_8640000',
'32_max_round_max_259200',
'32_max_round_sum_259200',
'32_max_round_sum_8640000',
'32_misses_goodrounds_median_max_259200',
'32_misses_goodrounds_median_sum_259200',
'32_misses_goodrounds_median_sum_8640000',
'32_misses_goodrounds_sum_sum_259200',
'32_num_goodrounds_max_259200',
'32_num_goodrounds_median_259200',
'32_num_goodrounds_median_8640000',
'35_duration_goodrounds_mean_max_259200',
'35_duration_goodrounds_mean_max_8640000',
'35_duration_goodrounds_mean_mean_259200',
'35_duration_goodrounds_mean_mean_8640000',
'35_duration_goodrounds_mean_median_259200',
'35_duration_goodrounds_mean_median_8640000',
'35_duration_goodrounds_mean_sum_259200',
'35_duration_goodrounds_mean_sum_8640000',
'35_duration_goodrounds_median_max_259200',
'35_duration_goodrounds_median_max_8640000',
'35_duration_goodrounds_median_mean_259200',
'35_duration_goodrounds_median_mean_8640000',
'35_duration_goodrounds_median_median_259200',
'35_duration_goodrounds_median_median_8640000',
'35_duration_goodrounds_median_sum_259200',
'35_duration_goodrounds_median_sum_8640000',
'35_max_round_max_259200',
'35_max_round_max_8640000',
'35_max_round_mean_259200',
'35_max_round_mean_8640000',
'35_max_round_median_259200',
'35_max_round_median_8640000',
'35_max_round_sum_259200',
'35_max_round_sum_8640000',
'35_misses_goodrounds_median_max_259200',
'35_misses_goodrounds_median_max_8640000',
'35_misses_goodrounds_median_mean_259200',
'35_misses_goodrounds_median_mean_8640000',
'35_misses_goodrounds_median_median_259200',
'35_misses_goodrounds_median_median_8640000',
'35_misses_goodrounds_median_sum_259200',
'35_misses_goodrounds_median_sum_8640000',
'35_misses_goodrounds_sum_max_259200',
'35_misses_goodrounds_sum_max_8640000',
'35_misses_goodrounds_sum_mean_259200',
'35_misses_goodrounds_sum_mean_8640000',
'35_misses_goodrounds_sum_median_259200',
'35_misses_goodrounds_sum_median_8640000',
'35_misses_goodrounds_sum_sum_259200',
'35_misses_goodrounds_sum_sum_8640000',
'35_num_goodrounds_max_259200',
'35_num_goodrounds_max_8640000',
'35_num_goodrounds_mean_259200',
'35_num_goodrounds_mean_8640000',
'35_num_goodrounds_median_259200',
'35_num_goodrounds_median_8640000',
'35_num_goodrounds_sum_259200',
'35_num_goodrounds_sum_8640000',
'36_first_goodattemp__sum_259200',
'36_max_round_sum_259200',
'36_misses_goodrounds_median_max_259200',
'36_misses_goodrounds_median_max_8640000',
'36_misses_goodrounds_median_mean_259200',
'36_misses_goodrounds_median_median_259200',
'36_misses_goodrounds_median_median_8640000',
'36_misses_goodrounds_median_sum_259200',
'36_misses_goodrounds_median_sum_8640000',
'36_misses_goodrounds_sum_sum_259200',
'36_num_goodrounds_median_259200',
'36_num_goodrounds_median_8640000',
'4080_259200',
'4080_8640000',
'42_duration_goodrounds_mean_max_259200',
'42_duration_goodrounds_mean_max_8640000',
'42_duration_goodrounds_mean_mean_259200',
'42_duration_goodrounds_mean_mean_8640000',
'42_duration_goodrounds_mean_median_259200',
'42_duration_goodrounds_mean_median_8640000',
'42_duration_goodrounds_mean_sum_259200',
'42_duration_goodrounds_mean_sum_8640000',
'42_duration_goodrounds_median_max_259200',
'42_duration_goodrounds_median_max_8640000',
'42_duration_goodrounds_median_mean_259200',
'42_duration_goodrounds_median_mean_8640000',
'42_duration_goodrounds_median_median_259200',
'42_duration_goodrounds_median_median_8640000',
'42_duration_goodrounds_median_sum_259200',
'42_duration_goodrounds_median_sum_8640000',
'42_max_round_max_259200',
'42_max_round_max_8640000',
'42_max_round_mean_259200',
'42_max_round_mean_8640000',
'42_max_round_median_259200',
'42_max_round_median_8640000',
'42_max_round_sum_259200',
'42_max_round_sum_8640000',
'42_misses_goodrounds_median_max_259200',
'42_misses_goodrounds_median_max_8640000',
'42_misses_goodrounds_median_mean_259200',
'42_misses_goodrounds_median_mean_8640000',
'42_misses_goodrounds_median_median_259200',
'42_misses_goodrounds_median_median_8640000',
'42_misses_goodrounds_median_sum_259200',
'42_misses_goodrounds_median_sum_8640000',
'42_misses_goodrounds_sum_max_259200',
'42_misses_goodrounds_sum_max_8640000',
'42_misses_goodrounds_sum_mean_259200',
'42_misses_goodrounds_sum_mean_8640000',
'42_misses_goodrounds_sum_median_259200',
'42_misses_goodrounds_sum_median_8640000',
'42_misses_goodrounds_sum_sum_259200',
'42_misses_goodrounds_sum_sum_8640000',
'42_num_goodrounds_max_259200',
'42_num_goodrounds_max_8640000',
'42_num_goodrounds_mean_259200',
'42_num_goodrounds_mean_8640000',
'42_num_goodrounds_median_259200',
'42_num_goodrounds_median_8640000',
'42_num_goodrounds_sum_259200',
'42_num_goodrounds_sum_8640000',
'4_false_attempts_median_259200',
'4_false_attempts_median_8640000',
'4_false_attempts_sum_259200',
'4_first_goodattemp__mean_259200',
'4_first_goodattemp__median_259200',
'4_first_goodattemp__median_8640000',
'4_first_goodattemp__sum_259200',
'4_first_goodattemp__sum_8640000',
'4_max_round_max_259200',
'4_max_round_max_8640000',
'4_max_round_mean_259200',
'4_max_round_mean_8640000',
'4_max_round_median_259200',
'4_max_round_median_8640000',
'4_max_round_sum_259200',
'4_max_round_sum_8640000',
'4_misses_goodrounds_median_sum_259200',
'4_misses_goodrounds_median_sum_8640000',
'4_misses_goodrounds_sum_median_259200',
'4_misses_goodrounds_sum_sum_8640000',
'4_num_goodrounds_max_259200',
'4_num_goodrounds_max_8640000',
'4_num_goodrounds_median_259200',
'4_num_goodrounds_median_8640000',
'4_true_attempts_median_259200',
'4_true_attempts_median_8640000',
'4_true_attempts_sum_259200',
'4_true_attempts_sum_8640000',
'5_max_round_sum_259200',
'5_misses_goodrounds_median_max_259200',
'5_misses_goodrounds_median_max_8640000',
'5_misses_goodrounds_median_mean_259200',
'5_misses_goodrounds_median_mean_8640000',
'5_misses_goodrounds_median_median_259200',
'5_misses_goodrounds_median_median_8640000',
'5_misses_goodrounds_median_sum_259200',
'5_misses_goodrounds_median_sum_8640000',
'5_misses_goodrounds_sum_max_259200',
'5_misses_goodrounds_sum_max_8640000',
'5_misses_goodrounds_sum_mean_259200',
'5_misses_goodrounds_sum_mean_8640000',
'5_misses_goodrounds_sum_median_259200',
'5_misses_goodrounds_sum_median_8640000',
'5_misses_goodrounds_sum_sum_259200',
'5_misses_goodrounds_sum_sum_8640000',
'5_num_goodrounds_median_259200',
'5_num_goodrounds_median_8640000',
'5_num_goodrounds_sum_259200',
'6_first_goodattemp__median_259200',
'6_first_goodattemp__median_8640000',
'6_first_goodattemp__sum_259200',
'6_first_goodattemp__sum_8640000',
'6_max_round_median_259200',
'6_max_round_sum_259200',
'6_max_round_sum_8640000',
'6_misses_goodrounds_median_max_259200',
'6_misses_goodrounds_median_median_259200',
'6_misses_goodrounds_median_sum_259200',
'6_misses_goodrounds_sum_max_259200',
'6_misses_goodrounds_sum_median_259200',
'6_misses_goodrounds_sum_sum_259200',
'6_num_goodrounds_max_259200',
'6_num_goodrounds_max_8640000',
'6_num_goodrounds_median_259200',
'6_num_goodrounds_median_8640000',
'6_num_goodrounds_sum_259200',
'7_duration_goodrounds_mean_max_259200',
'7_duration_goodrounds_mean_max_8640000',
'7_duration_goodrounds_mean_mean_259200',
'7_duration_goodrounds_mean_mean_8640000',
'7_duration_goodrounds_mean_median_259200',
'7_duration_goodrounds_mean_median_8640000',
'7_duration_goodrounds_mean_sum_259200',
'7_duration_goodrounds_mean_sum_8640000',
'7_duration_goodrounds_median_max_259200',
'7_duration_goodrounds_median_max_8640000',
'7_duration_goodrounds_median_mean_259200',
'7_duration_goodrounds_median_mean_8640000',
'7_duration_goodrounds_median_median_259200',
'7_duration_goodrounds_median_median_8640000',
'7_duration_goodrounds_median_sum_259200',
'7_duration_goodrounds_median_sum_8640000',
'7_max_round_max_259200',
'7_max_round_max_8640000',
'7_max_round_mean_259200',
'7_max_round_mean_8640000',
'7_max_round_median_259200',
'7_max_round_median_8640000',
'7_max_round_sum_259200',
'7_max_round_sum_8640000',
'7_misses_goodrounds_median_max_259200',
'7_misses_goodrounds_median_max_8640000',
'7_misses_goodrounds_median_mean_259200',
'7_misses_goodrounds_median_mean_8640000',
'7_misses_goodrounds_median_median_259200',
'7_misses_goodrounds_median_median_8640000',
'7_misses_goodrounds_median_sum_259200',
'7_misses_goodrounds_median_sum_8640000',
'7_misses_goodrounds_sum_max_259200',
'7_misses_goodrounds_sum_max_8640000',
'7_misses_goodrounds_sum_mean_259200',
'7_misses_goodrounds_sum_mean_8640000',
'7_misses_goodrounds_sum_median_259200',
'7_misses_goodrounds_sum_median_8640000',
'7_misses_goodrounds_sum_sum_259200',
'7_misses_goodrounds_sum_sum_8640000',
'7_num_goodrounds_max_259200',
'7_num_goodrounds_max_8640000',
'7_num_goodrounds_mean_259200',
'7_num_goodrounds_mean_8640000',
'7_num_goodrounds_median_259200',
'7_num_goodrounds_median_8640000',
'7_num_goodrounds_sum_259200',
'7_num_goodrounds_sum_8640000',
'7ad3efc6_259200',
'8_false_attempts_median_259200',
'8_false_attempts_median_8640000',
'8_false_attempts_sum_259200',
'8_first_goodattemp__median_259200',
'8_first_goodattemp__median_8640000',
'8_first_goodattemp__sum_259200',
'8_max_round_max_259200',
'8_max_round_max_8640000',
'8_max_round_mean_259200',
'8_max_round_mean_8640000',
'8_max_round_median_259200',
'8_max_round_median_8640000',
'8_max_round_sum_259200',
'8_max_round_sum_8640000',
'8_misses_goodrounds_median_max_259200',
'8_misses_goodrounds_median_sum_259200',
'8_misses_goodrounds_median_sum_8640000',
'8_misses_goodrounds_sum_sum_259200',
'8_misses_goodrounds_sum_sum_8640000',
'8_num_goodrounds_max_259200',
'8_num_goodrounds_max_8640000',
'8_num_goodrounds_median_259200',
'8_num_goodrounds_median_8640000',
'8_num_goodrounds_sum_259200',
'8_true_attempts_median_259200',
'8_true_attempts_median_8640000',
'8_true_attempts_sum_259200',
'9_false_attempts_median_259200',
'9_false_attempts_median_8640000',
'9_false_attempts_sum_259200',
'9_first_goodattemp__median_259200',
'9_first_goodattemp__sum_259200',
'9_max_round_max_259200',
'9_max_round_max_8640000',
'9_max_round_median_259200',
'9_misses_goodrounds_median_sum_259200',
'9_misses_goodrounds_sum_sum_259200',
'9_num_goodrounds_max_259200',
'9_num_goodrounds_max_8640000',
'9_num_goodrounds_median_259200',
'9_num_goodrounds_median_8640000',
'9_true_attempts_median_259200',
'9_true_attempts_median_8640000',
'9_true_attempts_sum_8640000',
'activity_10_max_259200',
'activity_10_median_259200',
'activity_30_max_259200',
'activity_4_max_8640000',
'activity_5_max_259200',
'activity_5_max_8640000',
'activity_5_median_259200',
'activity_5_median_8640000',
'got_history_259200',
'got_history_8640000',
'relevant_activity_misses_goodrounds_median_8640000__mean',
'2_num_goodrounds_max_259200',
'4_num_of_misses_total_misses_259200',
'6_misses_goodrounds_sum_sum_8640000',
'12_first_goodattemp__median_259200',
'6_max_round_max_259200',
'9_max_round_median_8640000',
'12_num_goodrounds_max_259200',
'32_num_attempts_sum_259200',
'8_num_goodrounds_sum_8640000',
'17_misses_goodrounds_median_max_259200',
'2_misses_goodrounds_median_median_259200',
'5_max_round_sum_8640000',
'1_num_attempts_sum_8640000',
'36_num_goodrounds_max_259200',
'19_max_round_sum_8640000',
'8_misses_goodrounds_median_mean_259200',
'36_first_goodattemp__median_259200',
'6_num_attempts_sum_259200',
'17_misses_goodrounds_median_max_8640000',
'18_misses_goodrounds_median_median_259200',
'32_num_goodrounds_sum_259200',
'4_num_goodrounds_sum_8640000',
'4_duration_goodrounds_median_sum_259200',
'23_max_round_max_259200',
'2_num_attempts_sum_259200',
'9_max_round_sum_259200',
'10_duration_goodrounds_median_max_259200',
'26_misses_goodrounds_sum_sum_8640000',
'19_num_goodrounds_max_8640000',
'19_misses_goodrounds_median_mean_259200',
'7ad3efc6_8640000',
'8_misses_goodrounds_median_max_8640000',
'4_num_attempts_sum_259200',
'4_misses_goodrounds_sum_max_259200',
'19_max_round_max_259200',
'36_num_goodrounds_sum_259200',
'23_num_attempts_sum_259200',
'9_first_goodattemp__median_8640000',
'10_share_cor_sum_259200',
'9_first_goodattemp__sum_8640000',
'9_true_attempts_sum_259200',
'1_first_goodattemp__median_8640000',
'32_max_round_median_259200',
'10_num_goodrounds_mean_259200',
'4_duration_goodrounds_median_sum_8640000',
'17_num_goodrounds_max_8640000',
'2_max_round_sum_8640000',
'26_duration_goodrounds_median_sum_259200',
'6_first_goodattemp__mean_259200',
'12_misses_goodrounds_median_sum_259200',
'1_share_cor_sum_8640000',
'32_misses_goodrounds_sum_median_259200',
'26_misses_goodrounds_sum_max_259200',
'8_first_goodattemp__sum_8640000',
'1_misses_goodrounds_sum_median_8640000',
'2_max_round_median_259200',
'32_misses_goodrounds_sum_max_259200',
'19_duration_goodrounds_median_sum_259200',
'8_true_attempts_sum_8640000',
'1_max_round_max_8640000',
'6_num_goodrounds_sum_8640000',
'4_misses_goodrounds_sum_sum_259200',
'2_misses_goodrounds_sum_sum_259200',
'19_num_goodrounds_max_259200',
'9_accuracy_in_Ass_median_259200',
'10_first_goodattemp__mean_259200',
'19_share_cor_sum_259200',
'10_misses_goodrounds_sum_mean_259200',
'18_num_goodrounds_sum_8640000',
'17_first_goodattemp__mean_259200',
'17_max_round_mean_259200',
'30_misses_goodrounds_median_max_8640000',
'19_duration_goodrounds_mean_sum_259200',
'32_num_goodrounds_max_8640000',
'36_num_attempts_sum_259200',
'6_misses_goodrounds_median_sum_8640000',
'activity_4_median_8640000',
'8_misses_goodrounds_median_median_259200',
'10_misses_goodrounds_sum_median_8640000',
'1_max_round_max_259200',
'532a2afb_259200',
'26_num_goodrounds_sum_8640000',
'2_max_round_max_259200',
'12_num_goodrounds_sum_259200',
'9_max_round_sum_8640000',
'10_misses_goodrounds_median_max_8640000',
'32_misses_goodrounds_median_max_8640000',
'19_max_round_median_259200',
'10_misses_goodrounds_sum_max_259200',
'3ccd3f02_259200',
'a1bbe385_259200',
'acc_Chest_Sorter__Assessment__259200',
'6_max_round_mean_259200',
'26_num_goodrounds_max_259200',
'19_misses_goodrounds_sum_max_259200',
'4_duration_goodrounds_median_max_259200',
'10_misses_goodrounds_median_mean_8640000',
'17_max_round_max_259200',
'17_share_cor_sum_259200',
'23_max_round_median_259200',
'26_misses_goodrounds_median_mean_259200',
'30_misses_goodrounds_median_median_259200',
'19_num_goodrounds_sum_8640000',
'32_num_of_misses_total_misses_259200',
'10_duration_goodrounds_mean_median_259200',
'26_first_goodattemp__median_8640000',
'6_max_round_max_8640000',
'23_misses_goodrounds_sum_sum_259200',
'2040_259200',
'17_duration_goodrounds_median_sum_259200',
'2_first_goodattemp__median_8640000',
'Crystals_Rule_259200',
'f806dc10_259200',
'8_misses_goodrounds_median_median_8640000',
'1_share_cor_sum_259200',
'8_misses_goodrounds_sum_max_259200',
'9_num_attempts_sum_259200',
'26_num_goodrounds_max_8640000',
'26_duration_goodrounds_mean_sum_259200',
'10_share_cor_median_259200',
'36_max_round_sum_8640000',
'18_misses_goodrounds_sum_sum_8640000',
'36_first_goodattemp__median_8640000',
'3dcdda7f_259200',
'4_duration_goodrounds_mean_sum_259200',
'10_duration_goodrounds_mean_mean_259200',
'30_duration_goodrounds_median_sum_259200',
'4_num_goodrounds_sum_259200',
'b1d5101d_259200',
'17_misses_goodrounds_sum_sum_8640000',
'activity_30_median_259200',
'4d6737eb_259200',
'30_misses_goodrounds_sum_max_259200',
'12_max_round_sum_8640000',
'1_misses_goodrounds_median_max_259200',
'18_misses_goodrounds_median_mean_259200',
'8_share_cor_median_259200',
'6_duration_goodrounds_median_sum_259200',
'26_max_round_max_8640000',
'23_num_goodrounds_max_8640000',
'10_num_of_misses_total_misses_259200',
'activity_4_max_259200',
'4_false_attempts_sum_8640000',
'26_misses_goodrounds_sum_median_259200',
'36_first_goodattemp__sum_8640000',
'23_first_goodattemp__mean_259200',
'36_num_goodrounds_max_8640000',
'26_misses_goodrounds_median_max_8640000',
'8_misses_goodrounds_median_mean_8640000',
'9_max_round_mean_259200',
'36_misses_goodrounds_median_mean_8640000',
'26_misses_goodrounds_median_median_8640000',
'18_misses_goodrounds_sum_max_259200',
'1_misses_goodrounds_sum_median_259200',
'26_num_attempts_median_259200',
'4_misses_goodrounds_sum_mean_259200',
'bdf49a58_259200',
'6_duration_goodrounds_mean_sum_259200',
'f7e47413_259200',
'23_num_goodrounds_sum_8640000',
'26_first_goodattemp__mean_259200',
'32_misses_goodrounds_median_median_8640000',
'26_duration_goodrounds_mean_mean_259200',
'30_misses_goodrounds_median_median_8640000',
'8_num_attempts_sum_259200',
'activity_4_median_259200',
'18_num_attempts_sum_259200',
'6_misses_goodrounds_median_mean_259200',
'a29c5338_259200',
'5c3d2b2f_259200',
'30_num_attempts_sum_259200',
'17_num_of_misses_total_misses_259200',
'1_num_attempts_mean_259200',
'8_accuracy_in_Ass_median_259200',
'709b1251_259200',
'17_max_round_median_8640000',
'32_first_goodattemp__median_8640000',
'ab3136ba_259200',
'9_duration_goodrounds_median_sum_259200',
'6_num_attempts_median_259200',
'8_misses_goodrounds_sum_max_8640000',
'32_num_attempts_median_259200',
'f28c589a_259200',
'4_duration_goodrounds_median_median_259200',
'36_misses_goodrounds_sum_median_259200',
'4_misses_goodrounds_median_median_259200',
'12_misses_goodrounds_median_sum_8640000',
'18_duration_goodrounds_mean_sum_259200',
'activity_8_max_259200',
'e9c52111_259200',
'17_num_attempts_sum_259200',
'262136f4_259200',
'2_misses_goodrounds_median_max_259200',
'26_duration_goodrounds_median_mean_259200',
'10_duration_goodrounds_median_sum_8640000',
'17_max_round_max_8640000',
'19_misses_goodrounds_median_max_8640000',
'2_num_goodrounds_max_8640000',
'30_misses_goodrounds_median_mean_259200',
'8af75982_259200',
'c0415e5c_259200',
'30_first_goodattemp__mean_8640000',
'4d6737eb_8640000',
'12_num_goodrounds_max_8640000',
'17_duration_goodrounds_mean_sum_259200',
'26_num_attempts_sum_8640000',
'2_num_goodrounds_sum_8640000',
'4_share_cor_sum_8640000',
'26_misses_goodrounds_sum_mean_259200',
'b7530680_259200',
'32_misses_goodrounds_median_median_259200',
'1_num_attempts_median_8640000',
'5010_259200',
'1_misses_goodrounds_sum_mean_8640000',
'30_false_attempts_sum_8640000',
'12_misses_goodrounds_median_max_259200',
'4_accuracy_in_Ass_median_259200',
'32_misses_goodrounds_sum_mean_259200',
'4_share_cor_sum_259200',
'23_num_attempts_sum_8640000',
'2_misses_goodrounds_sum_median_259200',
'9d4e7b25_259200',
'6_share_cor_sum_259200',
'923afab1_259200',
'2_share_cor_sum_259200',
'1_share_cor_median_259200',
'2_first_goodattemp__mean_259200',
'6_num_of_misses_total_misses_259200',
'1_duration_goodrounds_mean_median_259200',
'18_misses_goodrounds_sum_median_259200',
'8_first_goodattemp__mean_259200',
'19_misses_goodrounds_sum_sum_8640000',
'36_max_round_max_259200',
'17_num_goodrounds_mean_259200',
'8_num_attempts_sum_8640000',
'3babcb9b_259200',
'4_duration_goodrounds_mean_sum_8640000',
'6_misses_goodrounds_sum_median_8640000',
'4220_259200',
'1_num_of_misses_total_misses_259200',
'f806dc10_8640000',
'a2df0760_259200',
'709b1251_8640000',
'6_num_goodrounds_mean_259200',
'4_false_attempts_mean_259200',
'2dc29e21_259200',
'd122731b_259200',
'19_misses_goodrounds_sum_mean_259200',
'8_share_cor_sum_259200',
'23_max_round_max_8640000',
'10_num_attempts_sum_259200',
'5154fc30_259200',
'9_share_cor_median_259200',
'36_max_round_median_259200',
'18_misses_goodrounds_sum_mean_259200',
'acc_Cauldron_Filler__Assessment__259200',
'Bubble_Bath_259200',
'10_duration_goodrounds_median_median_8640000',
'2_num_attempts_median_259200',
'3d0b9317_259200',
'9_num_goodrounds_sum_259200',
'23_duration_goodrounds_mean_sum_259200',
'18_duration_goodrounds_median_sum_259200',
'6_max_round_median_8640000',
'7f0836bf_259200',
'6_misses_goodrounds_sum_max_8640000',
'a52b92d5_259200',
'19_first_goodattemp__median_8640000',
'5f0eb72c_259200',
'18_misses_goodrounds_median_max_8640000',
'Leaf_Leader_259200',
'10_misses_goodrounds_sum_max_8640000',
'11_num_of_misses_total_misses_259200',
'19_duration_goodrounds_median_max_259200',
'32_max_round_mean_259200',
'17_num_attempts_median_259200',
'6_duration_goodrounds_mean_sum_8640000',
'19_max_round_mean_259200',
'1_max_round_median_8640000',
'26_duration_goodrounds_median_median_259200',
'1_duration_goodrounds_mean_mean_259200',
'9_first_goodattemp__mean_259200',
'10_accuracy_in_Ass_median_259200',
'4110_259200',
'532a2afb_8640000',
'23_misses_goodrounds_median_sum_8640000',
'84b0e0c8_259200',
'36_num_attempts_sum_8640000',
'2_share_cor_median_259200',
'23_duration_goodrounds_median_sum_259200',
'8d7e386c_259200',
'19_first_goodattemp__mean_259200',
'4_num_attempts_median_259200',
'5_num_goodrounds_sum_8640000',
'795e4a37_259200',
'18_duration_goodrounds_median_median_259200',
'91561152_259200',
'5000_259200',
'4_share_cor_median_259200',
'a5be6304_8640000',
'19_misses_goodrounds_median_median_8640000',
'4_accuracy_in_Ass_mean_259200',
'9_num_attempts_median_259200',
'12_first_goodattemp__median_8640000',
'8_duration_goodrounds_median_sum_259200',
'10_duration_goodrounds_mean_max_259200',
'8_misses_goodrounds_sum_median_8640000',
'26_misses_goodrounds_median_mean_8640000',
'26_duration_goodrounds_mean_median_259200',
'32_num_attempts_sum_8640000',
'1_max_round_median_259200',
'907a054b_259200',
'36_num_of_misses_total_misses_259200',
'3d63345e_259200',
'19_share_cor_median_259200',
'22_num_of_misses_total_misses_259200',
'832735e1_259200',
'19_duration_goodrounds_mean_max_259200',
'6_misses_goodrounds_sum_mean_259200',
'26_share_cor_sum_259200',
'18_num_of_misses_total_misses_259200',
'36_misses_goodrounds_sum_mean_259200',
'93b353f2_259200',
'4_misses_goodrounds_sum_max_8640000',
'9_num_goodrounds_sum_8640000',
'activity_9_max_259200',
'bc8f2793_259200',
'activity_10_max_8640000',
'77261ab5_259200',
'10_num_of_misses_total_misses_8640000',
'32_first_goodattemp__mean_259200',
'4_duration_goodrounds_median_mean_259200',
'00c73085_8640000',
'32_num_goodrounds_sum_8640000',
'a16a373e_259200',
'1_misses_goodrounds_median_mean_8640000',
'2_max_round_mean_259200',
'activity_19_max_259200',
'2230fab4_259200',
'6_num_attempts_mean_259200',
'736f9581_8640000',
'6_duration_goodrounds_median_sum_8640000',
'a2df0760_8640000',
'17_misses_goodrounds_median_mean_259200',
'19_duration_goodrounds_mean_mean_259200',
'Pan_Balance_259200',
'19_duration_goodrounds_mean_median_259200',
'32_misses_goodrounds_sum_sum_8640000',
'5c3d2b2f_8640000',
'30_misses_goodrounds_median_mean_8640000',
'Air_Show_259200',
'17_duration_goodrounds_mean_mean_259200',
'activity_6_max_259200',
'17_duration_goodrounds_mean_sum_8640000',
'23_max_round_median_8640000',
'69fdac0a_259200',
'Dino_Dive_259200',
'28f975ea_8640000',
'18_duration_goodrounds_mean_max_259200',
'1_duration_goodrounds_median_median_259200',
'18_misses_goodrounds_median_median_8640000',
'9_misses_goodrounds_sum_max_259200',
'19_num_attempts_sum_259200',
'8_false_attempts_sum_8640000',
'9_num_of_misses_total_misses_259200',
'1_duration_goodrounds_mean_sum_8640000',
'32_duration_goodrounds_mean_median_259200',
'1cc7cfca_259200',
'30_false_attempts_mean_259200',
'19_duration_goodrounds_median_sum_8640000',
'4ef8cdd3_259200',
'17_num_attempts_sum_8640000',
'8b757ab8_259200',
'26_duration_goodrounds_mean_sum_8640000',
'Bottle_Filler__Activity__259200',
'2_num_attempts_sum_8640000',
'36_share_cor_sum_259200',
'19_duration_goodrounds_median_median_259200',
'4c2ec19f_259200',
'activity_8_max_8640000',
'4_num_attempts_sum_8640000',
'df4940d3_259200',
'4_first_goodattemp__mean_8640000',
'2dcad279_259200',
'12_misses_goodrounds_sum_sum_259200',
'9e4c8c7b_259200',
'8_share_cor_sum_8640000',
'923afab1_8640000',
'2081_259200',
'b1d5101d_8640000',
'18_duration_goodrounds_mean_mean_259200',
'17_duration_goodrounds_mean_median_259200',
'32_duration_goodrounds_median_sum_259200',
'19_max_round_median_8640000',
'36_num_goodrounds_sum_8640000',
'10_duration_goodrounds_median_mean_8640000',
'4_num_of_misses_total_misses_8640000',
'4_num_attempts_mean_259200',
'10_false_attempts_sum_8640000',
'1_duration_goodrounds_mean_max_259200',
'30_share_cor_median_259200',
'6_max_round_mean_8640000',
'9_share_cor_sum_259200',
'6_duration_goodrounds_median_max_259200',
'32_num_of_misses_total_misses_8640000',
'8_duration_goodrounds_median_sum_8640000',
'5348fd84_259200',
'36_misses_goodrounds_sum_sum_8640000',
'1_misses_goodrounds_sum_mean_259200',
'18_num_attempts_median_259200',
'18_num_attempts_sum_8640000',
'795e4a37_8640000',
'23_misses_goodrounds_median_median_259200',
'Chicken_Balancer__Activity__259200',
'10_true_attempts_mean_259200',
'2_num_of_misses_total_misses_259200',
'23_max_round_mean_259200',
'18_duration_goodrounds_median_max_259200',
'6_duration_goodrounds_mean_max_259200',
'32_duration_goodrounds_median_median_259200',
'26_num_attempts_mean_259200',
'74e5f8a7_259200',
'36_duration_goodrounds_mean_sum_259200',
'30_duration_goodrounds_mean_sum_259200',
'1_duration_goodrounds_median_sum_8640000',
'18_share_cor_sum_259200',
'9_misses_goodrounds_median_median_259200',
'26_misses_goodrounds_sum_max_8640000',
'32_duration_goodrounds_mean_sum_259200',
'6_misses_goodrounds_median_max_8640000',
'8_duration_goodrounds_median_mean_259200',
'36_duration_goodrounds_median_sum_259200',
'5010_8640000',
'00c73085_259200',
             
            
'relevant_activity_max_round_8640000__mean',
'relevant_activity_num_goodrounds_8640000__sum',
'activity_30_max_8640000',
'32_duration_sum_259200',
'Crystals_Rule_259200',
'acc_Cauldron_Filler__Assessment__259200',
'828e68f9_259200',
'6_duration_sum_259200',
'736f9581_259200',
'1_duration_sum_259200',
'Bubble_Bath_259200',
'8b757ab8_8640000',
'17_duration_sum_259200',
'36_duration_sum_259200',
'acc_Chest_Sorter__Assessment__259200',
'0d1da71f_259200',
'30_misses_goodrounds_sum_max_8640000',
'30_duration_goodrounds_median_sum_8640000',
'bd701df8_259200',
'8_num_attempts_median_259200',
'17_num_of_misses_total_misses_8640000',
'Pan_Balance_259200',
'23_num_attempts_median_259200',
'Air_Show_259200',
'Chicken_Balancer__Activity__259200',
'3d0b9317_8640000',
'3babcb9b_8640000',
'23_misses_goodrounds_median_max_259200',
'11_num_of_misses_total_misses_8640000',
'17_misses_goodrounds_sum_median_259200',
'Leaf_Leader_259200',
'65a38bf7_8640000',
'acc_Mushroom_Sorter__Assessment__259200',
'18_duration_sum_259200',
'Bottle_Filler__Activity__259200',
'26_duration_sum_259200',
'30_num_attempts_median_259200',
'ab3136ba_8640000',
'30_misses_goodrounds_sum_median_8640000',
'17_duration_sum_8640000',
'6_num_of_misses_total_misses_8640000',
'30_duration_goodrounds_median_mean_259200',
'0330ab6a_259200',
'5_num_goodrounds_max_259200',
'11_duration_sum_259200',
'd2e9262e_259200',
'30_duration_goodrounds_median_median_259200',
'12_misses_goodrounds_median_median_259200',
'6_num_attempts_sum_8640000',
'28f975ea_259200',
'f3cd5473_259200',
'a5be6304_259200',
'19_duration_sum_259200',
'12_max_round_median_259200',
'2_duration_goodrounds_mean_sum_259200',
'acc_Bird_Measurer__Assessment__259200',
'Crystals_Rule_8640000',
'bdf49a58_8640000',
'5154fc30_8640000',
'71fe8f75_8640000',
'bd612267_259200',
'18_duration_goodrounds_mean_sum_8640000',
'26_num_of_misses_total_misses_259200',
'a52b92d5_8640000',
'17_duration_goodrounds_median_sum_8640000',
'3dcdda7f_8640000',
'19_duration_goodrounds_mean_sum_8640000',
'cdd22e43_259200',
'6_first_goodattemp__mean_8640000',
'7ec0c298_259200',
'18_num_goodrounds_max_8640000',
# '12_num_attempts_sum_259200',
# '8af75982_8640000',
# '36_first_goodattemp__mean_259200',
# 'a1e4395d_8640000',
# '2_misses_goodrounds_median_mean_259200',
# '26_num_of_misses_total_misses_8640000',
# 'Bird_Measurer__Assessment__259200',
# '1_duration_sum_8640000',
# '20_num_of_misses_total_misses_8640000',
# '12_num_goodrounds_sum_8640000',
# '30_share_cor_sum_259200',
# '30_num_attempts_sum_8640000',
# '9_duration_goodrounds_median_sum_8640000',
# '9_duration_goodrounds_median_mean_259200',
# '32_duration_goodrounds_median_sum_8640000',
# '12_first_goodattemp__mean_259200',
# '65a38bf7_259200',
# '19_num_attempts_sum_8640000',
# '23_misses_goodrounds_sum_sum_8640000',
# 'cfbd47c8_259200',
# '2230fab4_8640000',
# '23_misses_goodrounds_sum_max_259200',
# '9_duration_goodrounds_median_median_259200',
# 'a1e4395d_259200',
# '32_misses_goodrounds_median_mean_259200',
# '9_duration_goodrounds_median_max_259200',
# 'activity_6_median_259200',
# '1_num_of_misses_total_misses_8640000',
# '23_num_of_misses_total_misses_8640000',
# '4_misses_goodrounds_median_max_259200',
# '9_misses_goodrounds_sum_median_259200',
# '8_num_attempts_median_8640000',
# '19_num_attempts_median_259200',
# '90efca10_259200',
# 'f28c589a_8640000',
# '26_max_round_median_8640000',
# '17_max_round_mean_8640000',
# '4_misses_goodrounds_sum_median_8640000',
# '74e5f8a7_8640000',
# '26_share_cor_sum_8640000',
# '6_duration_sum_8640000',
# 'Scrub_A_Dub_259200',
# '23_num_of_misses_total_misses_259200',
# '23_duration_sum_259200',
# '8_duration_goodrounds_median_max_259200',
# '828e68f9_8640000',
# '17_share_cor_sum_8640000',
# '7f0836bf_8640000',
# 'Happy_Camel_259200',
# '4_duration_goodrounds_mean_max_259200',
# '26_misses_goodrounds_sum_mean_8640000',
# '832735e1_8640000',
# '19_share_cor_sum_8640000',
# '23_share_cor_sum_259200',
# '18_duration_goodrounds_median_sum_8640000',
# '23_duration_goodrounds_median_sum_8640000',
# '49ed92e9_259200',
# '2_misses_goodrounds_sum_sum_8640000',
# '2075_259200',
# '2fb91ec1_259200',
# '42_num_of_misses_total_misses_259200',
# 'Bubble_Bath_8640000',
# '6_misses_goodrounds_median_median_8640000',
# '12_max_round_max_259200',
# '18_max_round_median_259200',
# '30_num_of_misses_total_misses_259200',
# '3d8c61b0_8640000',
# '32_num_goodrounds_mean_259200',
# '5_duration_sum_259200',
# '67439901_259200',
# '36_num_attempts_median_259200',
# 'Egg_Dropper__Activity__259200',
# 'df4940d3_8640000',
# 'ea321fb1_259200',
# '26_duration_goodrounds_median_sum_8640000',
# '9_share_cor_sum_8640000',
# 'Flower_Waterer__Activity__259200',
# '26_misses_goodrounds_sum_median_8640000',
# '4a4c3d21_259200',
# '19_max_round_max_8640000',
# '77261ab5_8640000',
# '4ef8cdd3_8640000',
# '18_duration_sum_8640000',
# '8_share_cor_mean_259200',
# '23_duration_goodrounds_mean_sum_8640000',
# '12_num_attempts_sum_8640000',
# '3d63345e_8640000',
# '9_misses_goodrounds_median_max_259200',
# '17_duration_goodrounds_median_mean_259200',
# '18_misses_goodrounds_sum_max_8640000',
# '4031_259200',
# '8d7e386c_8640000',
# '1_max_round_mean_8640000',
# '10_duration_goodrounds_median_max_8640000',
# '2_max_round_median_8640000',
# '18_max_round_max_259200',
# '2040_8640000',
# '17_first_goodattemp__mean_8640000',
# 'e9c52111_8640000',
# 'Happy_Camel_8640000',
# '6_share_cor_sum_8640000',
# '4_duration_sum_259200',
# 'activity_30_median_8640000',
# '23_duration_sum_8640000',
# '2035_259200',
# '262136f4_8640000',
# 'Bottle_Filler__Activity__8640000',
# '18_num_of_misses_total_misses_8640000',
# 'b80e5e84_8640000',
# '51102b85_259200',
# '8_misses_goodrounds_sum_median_259200',
# '12_first_goodattemp__sum_8640000',
# '17_duration_goodrounds_median_max_259200',
# 'Dino_Dive_259200',
# '56cd3b43_259200',
# '598f4598_259200',
# 'Air_Show_8640000',
# '23_num_attempts_median_8640000',
# '36_misses_goodrounds_sum_max_259200',
# '18_max_round_max_8640000',
# 'b7530680_8640000',
# 'Bird_Measurer__Assessment__8640000',
# 'e79f3763_259200',
# '8_false_attempts_mean_259200',
# 'bbfe0445_259200',
# '30_num_attempts_median_8640000',
# '30614231_259200',
# '3d8c61b0_259200',
# '32_duration_goodrounds_mean_sum_8640000',
# '5000_8640000',
# 'bb3e370b_259200',
# 'bc8f2793_8640000',
# 'c0415e5c_8640000',
# '1_first_goodattemp__mean_8640000',
# 'b80e5e84_259200',
# '9_duration_goodrounds_mean_sum_259200',
# '0330ab6a_8640000',
# '56bcd38d_259200',
# '19_share_cor_mean_259200',
# '10_share_cor_mean_259200',
# '4_misses_goodrounds_sum_mean_8640000',
# 'activity_32_max_259200',
# 'activity_2_max_259200',
# 'a29c5338_8640000',
# '8_duration_goodrounds_median_median_259200',
# '6_share_cor_median_259200',
# '9_misses_goodrounds_sum_sum_8640000',
# 'a1bbe385_8640000',
# 'f7e47413_8640000',
# '84b0e0c8_8640000',
# '4_accuracy_in_Ass_median_8640000',
# '2070_259200',
# '12_misses_goodrounds_sum_sum_8640000',
# '90efca10_8640000',
# '9_duration_sum_259200',
# '17_misses_goodrounds_median_mean_8640000',
# '23_misses_goodrounds_sum_median_259200',
# 'ea321fb1_8640000',
# '3ccd3f02_8640000',
# '0d1da71f_8640000',
# 'activity_6_max_8640000',
# '35_num_of_misses_total_misses_259200',
# '69fdac0a_8640000',
# '32_duration_goodrounds_median_max_259200',
# '19_num_of_misses_total_misses_259200',
# 'cf82af56_259200',
# '10_false_attempts_mean_259200',
# '2_share_cor_sum_8640000',
# '17_misses_goodrounds_sum_median_8640000',
# 'activity_9_median_259200',
# 'a0faea5d_259200',
# 'Leaf_Leader_8640000',
# 'cfbd47c8_8640000',
# '30_misses_goodrounds_sum_median_259200',
# '12_duration_goodrounds_mean_sum_259200',
# '19_num_of_misses_total_misses_8640000',
# '21_num_of_misses_total_misses_259200',
# '5e3ea25a_259200',
# '26_duration_sum_8640000',
# '42_duration_sum_259200',
# 'activity_9_max_8640000',
# 'de26c3a6_259200',
# '8fee50e2_8640000',
# '8_num_attempts_mean_259200',
# 'd122731b_8640000',
# '4_duration_sum_8640000',
# '23_num_attempts_mean_259200',
# '23_duration_goodrounds_median_median_259200',
# '30_share_cor_sum_8640000',
# '2_max_round_max_8640000',
# '32_misses_goodrounds_median_mean_8640000',
# '6_duration_goodrounds_median_median_259200',
# '17_num_attempts_mean_259200',
# '7ec0c298_8640000',
# '32_duration_sum_8640000',
# '2dcad279_8640000',
# '23_duration_goodrounds_mean_median_259200',
# '8_misses_goodrounds_sum_mean_259200',
# '18_misses_goodrounds_sum_median_8640000',
# '8_num_of_misses_total_misses_259200',
# '8fee50e2_259200',
# '6_duration_goodrounds_median_mean_259200',
# 'activity_19_median_259200',
# 'activity_12_max_259200',
# '23_duration_goodrounds_mean_max_259200',
# 'bd701df8_8640000',
# '7ab78247_8640000',
# '5f0eb72c_8640000',
# '17_misses_goodrounds_sum_mean_259200',
# '32_num_attempts_mean_259200',
# '2060_259200',
# '2_misses_goodrounds_sum_max_259200',
# '4_duration_goodrounds_median_median_8640000',
# '907a054b_8640000',
# '4095_259200',
# '23_first_goodattemp__mean_8640000',
# '5_duration_goodrounds_median_sum_259200',
# 'bcceccc6_259200',
# '12_duration_goodrounds_median_sum_259200',
# '4031_8640000',
# 'acc_Cauldron_Filler__Assessment__8640000',
# '9e4c8c7b_8640000',
# '30_misses_goodrounds_sum_mean_259200',
# '2_misses_goodrounds_sum_mean_259200',
# '71fe8f75_259200',
# '2_num_attempts_mean_259200',
# 'activity_2_median_259200',
# '4110_8640000',
# '23_duration_goodrounds_mean_mean_259200',
# '9_num_attempts_sum_8640000',
# '37db1c2f_259200',
# 'activity_17_max_8640000',
# 'Egg_Dropper__Activity__8640000',
# '36_max_round_mean_259200',
# '6_duration_goodrounds_mean_median_259200',
# '19_max_round_mean_8640000',
# '4_duration_goodrounds_mean_median_259200',
# 'accumulated_quality_by_ass_9_true_attempts_259200',
# 'All_Star_Sorting_259200',
# 'Cauldron_Filler__Assessment__259200',
# '32_misses_goodrounds_sum_median_8640000',
# '17_duration_goodrounds_mean_max_259200',
# '7_num_of_misses_total_misses_259200',
# '36_num_of_misses_total_misses_8640000',
# '23_duration_goodrounds_median_max_259200',
# 'Chicken_Balancer__Activity__8640000',
# 'd02b7a8e_259200',
# '17_duration_goodrounds_median_median_259200',
# '7_duration_sum_259200',
# '1_duration_goodrounds_mean_mean_8640000',
# '2_duration_goodrounds_median_sum_259200',
# '7ab78247_259200',
# 'Pan_Balance_8640000',
# 'Bug_Measurer__Activity__8640000',
# '23_misses_goodrounds_median_mean_259200',
# '18_share_cor_sum_8640000',
# '4a4c3d21_8640000',
# '2dc29e21_8640000',
# '93b353f2_8640000',
# '42_num_of_misses_total_misses_8640000',
# 'activity_32_median_259200',
# 'Scrub_A_Dub_8640000',
# '17_num_attempts_median_8640000',
# 'acc_Cart_Balancer__Assessment__259200',
# '17_num_goodrounds_mean_8640000',
# '4_true_attempts_mean_259200',
# '22_num_of_misses_total_misses_8640000',
# '4c2ec19f_8640000',
# 'Mushroom_Sorter__Assessment__259200',
# '23_duration_goodrounds_median_mean_259200',
# '5d042115_8640000',
# '5d042115_259200',
# '20_duration_sum_259200',
# '9_share_cor_mean_259200',
# '22_duration_sum_259200',
# 'd2e9262e_8640000',
# '6_num_attempts_median_8640000',
# '5_num_of_misses_total_misses_259200',
# '10_misses_goodrounds_sum_mean_8640000',
# 'Chow_Time_8640000',
# '23_share_cor_sum_8640000',
# '26_duration_goodrounds_mean_max_259200',
# '18_num_attempts_mean_259200',
# '12_max_round_mean_259200',
# 'd3f1e122_259200',
# '1_duration_goodrounds_median_median_8640000',
# 'acc_Bird_Measurer__Assessment__8640000',
# '67439901_8640000',
# 'acc_Cart_Balancer__Assessment__8640000',
# '792530f8_8640000',
# 'activity_36_max_259200',
# '36_duration_sum_8640000',
# '36_share_cor_median_259200',
# 'bb3e370b_8640000',
# '2081_8640000',
# '36_num_attempts_mean_259200',
# '91561152_8640000',
# '8_duration_goodrounds_mean_sum_259200',
# '1_max_round_mean_259200',
# 'activity_8_median_8640000',
# '32_duration_goodrounds_median_mean_259200',
# '36_duration_goodrounds_median_max_259200',
# 'Dino_Drink_259200',
# '1_num_attempts_mean_8640000',
# '12_max_round_max_8640000',
# '12_share_cor_sum_259200',
# '2_num_goodrounds_mean_259200',
# '2_misses_goodrounds_median_median_8640000',
# '10_first_goodattemp__mean_8640000',
# '8_duration_goodrounds_median_mean_8640000',
# '9_num_goodrounds_mean_259200',
# '4_misses_goodrounds_median_mean_259200',
# '19_misses_goodrounds_median_mean_8640000',
# '23_misses_goodrounds_median_median_8640000',
# '51102b85_8640000',
# '11_duration_sum_8640000',
# '4bb2f698_259200',
# '30_share_cor_median_8640000',
# '19_duration_goodrounds_median_mean_259200',
# '20_duration_sum_8640000',
# '30614231_8640000',
# '26_num_goodrounds_mean_259200',
# '18_num_goodrounds_mean_259200',
# '36_share_cor_mean_259200',
# '17_misses_goodrounds_sum_max_8640000',
# 'Watering_Hole__Activity__259200',
# '18_num_goodrounds_max_259200',
# '32_duration_goodrounds_mean_mean_259200',
# '2_num_of_misses_total_misses_8640000',
# '1_misses_goodrounds_median_max_8640000',
# '2083_259200',
# '32_first_goodattemp__mean_8640000',
# '9_true_attempts_mean_259200',
# '1_duration_goodrounds_mean_median_8640000',
# '7_num_of_misses_total_misses_8640000' ,
'relevant_activity_duration_goodrounds_median_8640000__mean'
             ]

# to_exclude = []
# ajusted_test = reduce_test.copy()
# for feature in ajusted_test.columns:
#     if feature not in ['accuracy_group', 'installation_id', 'accuracy_group', 'session_title','hour','month','weekday']:
#         data = reduce_train[feature]
#         train_mean = data.mean()
#         data = ajusted_test[feature] 
#         test_mean = data.mean()
#         try:
#             ajust_factor = train_mean / test_mean
#             if ajust_factor > 100 or ajust_factor < 0.01:
#                 to_exclude.append(feature)
#                 print(feature, train_mean, test_mean)
# #             else:
# #                 ajusted_test[feature] *= ajust_factor
#         except:
#             to_exclude.append(feature)
#             print(feature, train_mean, test_mean)


# In[15]:


from collections import Counter, defaultdict
from sklearn.utils import check_random_state

class RepeatedStratifiedGroupKFold():

    def __init__(self, n_splits=5, n_repeats=1, random_state=None):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        
    # Implementation based on this kaggle kernel:
    #    https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
    def split(self, X, y=None, groups=None):
        k = self.n_splits
        def eval_y_counts_per_fold(y_counts, fold):
            y_counts_per_fold[fold] += y_counts
            std_per_label = []
            for label in range(labels_num):
                label_std = np.std(
                    [y_counts_per_fold[i][label] / y_distr[label] for i in range(k)]
                )
                std_per_label.append(label_std)
            y_counts_per_fold[fold] -= y_counts
            return np.mean(std_per_label)
            
        rnd = check_random_state(self.random_state)
        for repeat in range(self.n_repeats):
            labels_num = np.max(y) + 1
            y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
            y_distr = Counter()
            for label, g in zip(y, groups):
                y_counts_per_group[g][label] += 1
                y_distr[label] += 1

            y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
            groups_per_fold = defaultdict(set)
        
            groups_and_y_counts = list(y_counts_per_group.items())
            rnd.shuffle(groups_and_y_counts)

            for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
                best_fold = None
                min_eval = None
                for i in range(k):
                    fold_eval = eval_y_counts_per_fold(y_counts, i)
                    if min_eval is None or fold_eval < min_eval:
                        min_eval = fold_eval
                        best_fold = i
                y_counts_per_fold[best_fold] += y_counts
                groups_per_fold[best_fold].add(g)

            all_groups = set(groups)
            for i in range(k):
                train_groups = all_groups - groups_per_fold[i]
                test_groups = groups_per_fold[i]

                train_indices = [i for i, g in enumerate(groups) if g in train_groups]
                test_indices = [i for i, g in enumerate(groups) if g in test_groups]

                yield train_indices, test_indices


# In[16]:


y = reduce_train['accuracy_group']

cols_to_drop = [ 'installation_id', 'accuracy_group']
cols_to_drop.extend(to_exclude)
n_fold = 5
n_repeats= 5
folds = RepeatedStratifiedGroupKFold(n_splits=n_fold, n_repeats=n_repeats, random_state = 5)

splits = list(folds.split(np.empty_like(y), y,reduce_train['installation_id'] ))

#TRUNCATED SPLITS
for idx, s in enumerate(splits):
    val = reduce_train[['installation_id']].iloc[s[1]]
    val = val.sample(frac=1, random_state = 14)
    splits[idx] = (s[0], list(val.groupby('installation_id').tail(1).index))


# In[17]:


reduce_train.drop(list(set(reduce_train.columns) & set(cols_to_drop)),axis = 1,inplace = True)
reduce_test.drop(list(set(reduce_test.columns) & set(cols_to_drop)),axis = 1,inplace = True)


# In[18]:


reduce_train.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in reduce_train.columns]
reduce_test.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in reduce_test.columns]
categoricals = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in categoricals]


# In[19]:


params = {
#         'n_estimators':2000,
        'boosting_type': 'gbdt',
        'metric': 'rmse',
        'objective': 'regression',
        'eval_metric': 'cappa',
    
#         'n_jobs': -1,
        'seed': 42,

        'num_leaves': 130,
        'feature_fraction_bynode' : 0.6,
        'learning_rate': 0.01,
        'bagging_fraction': 0.6951448659512921,
        'bagging_freq': 3,
        'feature_fraction': 0.6867901263802068,
        'verbosity': 100,
        'early_stop': 150,
        'verbose_eval': 1000,
        'num_rounds': 10000,
        'raw_seed': 1234,
        'max_bin': 150,
        'min_child_samples': 150,
        'lambda_l1': 0.42651295024341174,
        'lambda_l2': 0.15395842517107572,
        'max_depth': -1,
        'min_split_gain': 0.023658591149106636
}


# In[20]:


def run_lightgbm(x_train, y_train, x_valid, y_valid, x_test, index):
    params['seed'] = params['raw_seed'] + index
    num_rounds = params['num_rounds']
    verbose_eval = params['verbose_eval']
    early_stop = params['early_stop']

    x_train_proc, x_valid_proc, x_test_proc = x_train, x_valid, x_test

    dtrain = lgb.Dataset(x_train_proc, y_train, 
                         categorical_feature=categoricals)
    dvalid = lgb.Dataset(x_valid_proc, y_valid,
                         categorical_feature=categoricals)

    model = lgb.train(params,
                      train_set=dtrain,
                      valid_sets=(dtrain, dvalid),
                      num_boost_round=num_rounds,
                      verbose_eval=verbose_eval,
                      early_stopping_rounds=early_stop)
    
    feature_importance = pd.DataFrame(list(zip(list(x_train.columns), model.feature_importance(importance_type='gain'))),
                                           columns=['feature', 'importance'])

    return  model.predict(x_valid_proc), model.predict(x_test_proc), feature_importance


# In[21]:


get_ipython().run_cell_magic('time', '', "FE = pd.DataFrame(columns=['feature', 'importance'])\n\ntrain_preds_lgb = np.ones((n_repeats, len(reduce_train))) * -1\ntest_preds_lgb = np.zeros((n_repeats, len(reduce_test), n_fold))\n\nfor i, (train_index, val_index) in enumerate(splits):\n    train_preds_lgb[i // n_fold, val_index], test_preds_lgb[i // n_fold, :, i % n_fold], fe = run_lightgbm(reduce_train.iloc[train_index], y.values[train_index],\n                                                                                                             reduce_train.iloc[val_index], y.values[val_index], reduce_test, i)\n    \n    FE = FE.append(fe)")


# In[22]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

fe_aggr = FE.groupby('feature')['importance'].median().reset_index()
fe_aggr.to_csv('fe_aggr.csv',index = False)


plt.figure(figsize = (8, 50))
sns.barplot(data = fe_aggr.sort_values(by = "importance", ascending = False).head(150), x = "importance", y = "feature")
plt.show()


# In[23]:


y_train  = y


# In[24]:


from sklearn.metrics import cohen_kappa_score, mean_squared_error, roc_auc_score
def fix_distribution(y_train, pred):
    base = pd.Series([0, 0, 0, 0, 0], index=np.arange(0, 5))
    thresholds = (base + pd.Series(y_train).value_counts()).fillna(0).cumsum()
    thresholds = thresholds / len(y_train) * len(pred)
    
    pred_ranks = pd.Series(pred).rank()
    ranked_scores = np.zeros(len(pred))

    for j, threshold in list(enumerate(thresholds))[::-1]:
        ranked_scores[pred_ranks <= threshold] = j
    return ranked_scores

SCORES = []
for r in range(n_repeats):
    idx = train_preds_lgb[r] != -1
    score = cohen_kappa_score(y_train[idx], fix_distribution(y_train[idx], train_preds_lgb[r, idx]), weights='quadratic')
    SCORES.append(score)

print(SCORES)
print('MEAN_CAPPA_', np.mean(SCORES))
print('MEDIAN_CAPPA_', np.median(SCORES))
print('STD_CAPPA_', np.std(SCORES))


# In[25]:


all_preds = test_preds_lgb.mean(axis = 0).mean(axis = 1)
fixed_scores = fix_distribution(y_train, np.array(all_preds))


# In[26]:


sample_submission['accuracy_group'] = fixed_scores.astype(int)
sample_submission.to_csv('submission.csv', index=False)


# In[27]:


sample_submission['accuracy_group'].value_counts(normalize=True)

