#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.




get_ipython().system('pip install fastai==0.7.0')




from fastai import *
from fastai.structured import *




from sklearn.metrics import *
from sklearn.ensemble import *




import pathlib
from tqdm import tqdm_notebook as tqdm




pd.set_option("display.expand_frame_repr", False)




PATH = '/kaggle/input/data-science-bowl-2019'
PATH_W = '/kaggle/working'

path = pathlib.Path(PATH)
path_w = pathlib.Path(PATH_W)




get_ipython().system('ls {path}')




get_ipython().system("head -n 300000 {path}/'train.csv' > {path_w}/'sample_train.csv'")
get_ipython().system("head -n 300000 {path}/'specs.csv' > {path_w}/'sample_specs.csv'")




sample_train = pd.read_csv(path_w/'sample_train.csv')
sample_specs = pd.read_csv(path_w/'sample_specs.csv')
train_labels = pd.read_csv(path/'train_labels.csv')




train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')
train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')
specs = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')
test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')
submission = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')




train.query('game_session=="0848ef14a8dc6892" & event_code!=4110')




#The following code copied from https://www.kaggle.com/artgor/quick-and-dirty-regression

def read_data():
    print('Reading train.csv file....')
    train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')
    print('Training.csv file have {} rows and {} columns'.format(train.shape[0], train.shape[1]))

    print('Reading test.csv file....')
    test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')
    print('Test.csv file have {} rows and {} columns'.format(test.shape[0], test.shape[1]))

    print('Reading train_labels.csv file....')
    train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')
    print('Train_labels.csv file have {} rows and {} columns'.format(train_labels.shape[0], train_labels.shape[1]))

    print('Reading specs.csv file....')
    specs = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')
    print('Specs.csv file have {} rows and {} columns'.format(specs.shape[0], specs.shape[1]))

    print('Reading sample_submission.csv file....')
    sample_submission = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')
    print('Sample_submission.csv file have {} rows and {} columns'.format(sample_submission.shape[0], sample_submission.shape[1]))
    return train, test, train_labels, specs, sample_submission

def encode_title(train, test, train_labels):
    # encode title
    train['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), train['title'], train['event_code']))
    test['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), test['title'], test['event_code']))
    all_title_event_code = list(set(train["title_event_code"].unique()).union(test["title_event_code"].unique()))
    # make a list with all the unique 'titles' from the train and test set
    list_of_user_activities = list(set(train['title'].unique()).union(set(test['title'].unique())))
    # make a list with all the unique 'event_code' from the train and test set
    list_of_event_code = list(set(train['event_code'].unique()).union(set(test['event_code'].unique())))
    list_of_event_id = list(set(train['event_id'].unique()).union(set(test['event_id'].unique())))
    # make a list with all the unique worlds from the train and test set
    list_of_worlds = list(set(train['world'].unique()).union(set(test['world'].unique())))
    # create a dictionary numerating the titles
    activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))
    activities_labels = dict(zip(np.arange(len(list_of_user_activities)), list_of_user_activities))
    activities_world = dict(zip(list_of_worlds, np.arange(len(list_of_worlds))))
    assess_titles = list(set(train[train['type'] == 'Assessment']['title'].value_counts().index).union(set(test[test['type'] == 'Assessment']['title'].value_counts().index)))
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

def get_data(user_sample, test_set=False):
    '''
    The user_sample is a DataFrame from train or test where the only one 
    installation_id is filtered
    And the test_set parameter is related with the labels processing, that is only requered
    if test_set=False
    '''
    # Constants and parameters declaration
    last_activity = 0
    
    user_activities_count = {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}
    
    # new features: time spent in each activity
    last_session_time_sec = 0
    accuracy_groups = {0:0, 1:0, 2:0, 3:0}
    all_assessments = []
    accumulated_accuracy_group = 0
    accumulated_accuracy = 0
    accumulated_correct_attempts = 0 
    accumulated_uncorrect_attempts = 0
    accumulated_actions = 0
    counter = 0
    time_first_activity = float(user_sample['timestamp'].values[0])
    durations = []
    last_accuracy_title = {'acc_' + title: -1 for title in assess_titles}
    event_code_count: Dict[str, int] = {ev: 0 for ev in list_of_event_code}
    event_id_count: Dict[str, int] = {eve: 0 for eve in list_of_event_id}
    title_count: Dict[str, int] = {eve: 0 for eve in activities_labels.values()} 
    title_event_code_count: Dict[str, int] = {t_eve: 0 for t_eve in all_title_event_code}
    
    # itarates through each session of one instalation_id
    for i, session in user_sample.groupby('game_session', sort=False):
        # i = game_session_id
        # session is a DataFrame that contain only one game_session
        
        # get some sessions information
        session_type = session['type'].iloc[0]
        session_title = session['title'].iloc[0]
        session_title_text = activities_labels[session_title]
                    
            
        # for each assessment, and only this kind off session, the features below are processed
        # and a register are generated
        if (session_type == 'Assessment') & (test_set or len(session)>1):
            # search for event_code 4100, that represents the assessments trial
            all_attempts = session.query(f'event_code == {win_code[session_title]}')
            # then, check the numbers of wins and the number of losses
            true_attempts = all_attempts['event_data'].str.contains('true').sum()
            false_attempts = all_attempts['event_data'].str.contains('false').sum()
            # copy a dict to use as feature template, it's initialized with some itens: 
            # {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}
            features = user_activities_count.copy()
            features.update(last_accuracy_title.copy())
            features.update(event_code_count.copy())
            features.update(event_id_count.copy())
            features.update(title_count.copy())
            features.update(title_event_code_count.copy())
            features.update(last_accuracy_title.copy())
            
            # get installation_id for aggregated features
            features['installation_id'] = session['installation_id'].iloc[-1]
            # add title as feature, remembering that title represents the name of the game
            features['session_title'] = session['title'].iloc[0]
            # the 4 lines below add the feature of the history of the trials of this player
            # this is based on the all time attempts so far, at the moment of this assessment
            features['accumulated_correct_attempts'] = accumulated_correct_attempts
            features['accumulated_uncorrect_attempts'] = accumulated_uncorrect_attempts
            accumulated_correct_attempts += true_attempts 
            accumulated_uncorrect_attempts += false_attempts
            # the time spent in the app so far
            if durations == []:
                features['duration_mean'] = 0
            else:
                features['duration_mean'] = np.mean(durations)
            durations.append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds)
            # the accurace is the all time wins divided by the all time attempts
            features['accumulated_accuracy'] = accumulated_accuracy/counter if counter > 0 else 0
            accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0
            accumulated_accuracy += accuracy
            last_accuracy_title['acc_' + session_title_text] = accuracy
            # a feature of the current accuracy categorized
            # it is a counter of how many times this player was in each accuracy group
            if accuracy == 0:
                features['accuracy_group'] = 0
            elif accuracy == 1:
                features['accuracy_group'] = 3
            elif accuracy == 0.5:
                features['accuracy_group'] = 2
            else:
                features['accuracy_group'] = 1
            features.update(accuracy_groups)
            accuracy_groups[features['accuracy_group']] += 1
            # mean of the all accuracy groups of this player
            features['accumulated_accuracy_group'] = accumulated_accuracy_group/counter if counter > 0 else 0
            accumulated_accuracy_group += features['accuracy_group']
            # how many actions the player has done so far, it is initialized as 0 and updated some lines below
            features['accumulated_actions'] = accumulated_actions
            
            # there are some conditions to allow this features to be inserted in the datasets
            # if it's a test set, all sessions belong to the final dataset
            # it it's a train, needs to be passed throught this clausule: session.query(f'event_code == {win_code[session_title]}')
            # that means, must exist an event_code 4100 or 4110
            if test_set:
                all_assessments.append(features)
            elif true_attempts+false_attempts > 0:
                all_assessments.append(features)
                
            counter += 1
        
        # this piece counts how many actions was made in each event_code so far
        def update_counters(counter: dict, col: str):
                num_of_session_count = Counter(session[col])
                for k in num_of_session_count.keys():
                    x = k
                    if col == 'title':
                        x = activities_labels[k]
                    counter[x] += num_of_session_count[k]
                return counter
            
        event_code_count = update_counters(event_code_count, "event_code")
        event_id_count = update_counters(event_id_count, "event_id")
        title_count = update_counters(title_count, 'title')
        title_event_code_count = update_counters(title_event_code_count, 'title_event_code')

        # counts how many actions the player has done so far, used in the feature of the same name
        accumulated_actions += len(session)
        if last_activity != session_type:
            user_activities_count[session_type] += 1
            last_activitiy = session_type 
                        
    # if it't the test_set, only the last assessment must be predicted, the previous are scraped
    if test_set:
        return all_assessments[-1]
    # in the train_set, all assessments goes to the dataset
    return all_assessments

def get_train_and_test(train, test):
    compiled_train = []
    compiled_test = []
    for i, (ins_id, user_sample) in tqdm(enumerate(train.groupby('installation_id', sort = False)), total = 17000):
        compiled_train += get_data(user_sample)
    for ins_id, user_sample in tqdm(test.groupby('installation_id', sort = False), total = 1000):
        test_data = get_data(user_sample, test_set = True)
        compiled_test.append(test_data)
    reduce_train = pd.DataFrame(compiled_train)
    reduce_test = pd.DataFrame(compiled_test)
    categoricals = ['session_title']
    return reduce_train, reduce_test, categoricals

# read data
train, test, train_labels, specs, sample_submission = read_data()
# get usefull dict with maping encode
train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code = encode_title(train, test, train_labels)
# tranform function to get the train and test set
reduce_train, reduce_test, categoricals = get_train_and_test(train, test)




def preprocess(reduce_train, reduce_test):
    for df in [reduce_train, reduce_test]:
        df['installation_session_count'] = df.groupby(['installation_id'])['Clip'].transform('count')
        df['installation_duration_mean'] = df.groupby(['installation_id'])['duration_mean'].transform('mean')
        #df['installation_duration_std'] = df.groupby(['installation_id'])['duration_mean'].transform('std')
        df['installation_title_nunique'] = df.groupby(['installation_id'])['session_title'].transform('nunique')
        
        df['sum_event_code_count'] = df[[2050, 4100, 4230, 5000, 4235, 2060, 4110, 5010, 2070, 2075, 2080, 2081, 2083, 3110, 4010, 3120, 3121, 4020, 4021, 
                                        4022, 4025, 4030, 4031, 3010, 4035, 4040, 3020, 3021, 4045, 2000, 4050, 2010, 2020, 4070, 2025, 2030, 4080, 2035, 
                                        2040, 4090, 4220, 4095]].sum(axis = 1)
        
        df['installation_event_code_count_mean'] = df.groupby(['installation_id'])['sum_event_code_count'].transform('mean')
        #df['installation_event_code_count_std'] = df.groupby(['installation_id'])['sum_event_code_count'].transform('std')
        
    features = reduce_train.loc[(reduce_train.sum(axis=1) != 0), (reduce_train.sum(axis=0) != 0)].columns # delete useless columns
    features = [x for x in features if x not in ['accuracy_group', 'installation_id']] + ['acc_' + title for title in assess_titles]
   
    return reduce_train, reduce_test, features
# call feature engineering function
reduce_train, reduce_test, features = preprocess(reduce_train, reduce_test)




temp_copy_train = reduce_train.copy()




int_encode = list(range(len(temp_copy_train.installation_id.unique())))
# temp_copy_train.installation_id.unique()
installation_id_int_encoding = {x:y for x,y in zip(temp_copy_train.installation_id.unique(),int_encode)}
installation_id_int_encoding




all_installation_codes = []
for x in temp_copy_train.installation_id:
    all_installation_codes.append(installation_id_int_encoding[x])
    
all_installation_codes




temp_copy_train['installation_id'] = all_installation_codes
temp_copy_train.installation_id




train_cats(temp_copy_train)




#removoing the y_fld from reduce_train
y_trn = temp_copy_train['accuracy_group']
df_trn = temp_copy_train.drop('accuracy_group',axis=1)




def split_vals(a,n): return a[:n],a[n:]

n_valid = 5307
n_train = len(df_trn) - n_valid
X_train, X_valid = split_vals(df_trn,n_train)
y_train, y_valid = split_vals(y_trn,n_train)
raw_train, raw_valid = split_vals(reduce_train,n_train)




get_ipython().run_line_magic('pinfo', 'train_cats')




from collections import defaultdict




# sample_train = sample_train.query('(event_code==4100 | event_code==4110) & type=="Assessment"')
len(sample_train.game_session.unique())




# def label_assign(sample_trn,game_sess_id):
#     sample_df = defaultdict(lambda: '')
# #     [sample_df[i] for i in list(sample_trn.columns)]
    
#     game_sess = sample_trn.groupby('game_session',sort=False)
#     #unique sess ids
#     sess_grps = list(game_sess.groups.keys())

#     #getting all rows of first group i.e. first session
#     f_grp = game_sess.get_group(game_sess_id)

#     #correct and icorrect attempts count
#     num_corr = 0
#     num_incorr = 0

#     #printing attempted rows 
#     attempts = f_grp.query('event_code==4100 | event_code==4110')['event_data']
#     num_attempts = len(attempts)
#     num_corr = attempts.str.contains('true').sum()
#     temp = 0
#     if(num_corr>1):
#         temp = num_corr
#         num_corr=1

#     num_incorr = attempts.str.contains('false').sum()

#     #calculating the accuracy
#     acc = num_corr/(num_corr+num_incorr)
# #     print(acc)
#     if(acc < 0.5):
#         acc_grp = 1
#     elif(acc == 0.5):
#         acc_grp = 2
#     elif(acc == 1):
#         acc_grp = 3
#     elif(acc == 0):
#         acc_grp = 0
#     else:
#         acc_grp = 0
        
        
#     sample_df['num_correct'] = num_corr
#     sample_df['num_incorrect'] = num_incorr
#     sample_df['accuracy'] = acc
#     sample_df['accuracy_group'] = acc_grp
    
#     return sample_df, game_sess.groups[game_sess_id]    #ssample dict, indexes of all the rows of the groupby clause




# sample_train['num_correct'] = 0
# sample_train['num_incorrect'] = 0
# sample_train['accuracy'] = 0
# sample_train['accuracy_group'] = 0

# temp_1 = label_assign(sample_train,'cace4c493ac347e3')
# temp_1




# y = sample_train.loc[temp_1[1]].query('event_code==4100 | event_code==4110')
# y_indexes = y.loc[y['event_data'].str.contains('true')].index
# false_idxs = y.loc[y['event_data'].str.contains('false')].index

# #values for true indices
# sample_train.loc[y_indexes,'num_correct'] = temp_1[0]['num_correct']
# sample_train.loc[y_indexes,'num_incorrect'] = temp_1[0]['num_incorrect']
# sample_train.loc[y_indexes,'accuracy'] = temp_1[0]['accuracy']
# sample_train.loc[y_indexes,'accuracy_group'] = temp_1[0]['accuracy_group']

# # #values for false indices
# # sample_train.loc[false_idxs,'num_correct'] = 0
# # sample_train.loc[false_idxs,'num_incorrect'] = 0
# # sample_train.loc[false_idxs,'accuracy'] = 0
# # sample_train.loc[false_idxs,'accuracy_group'] = 0

# # #values for indices other than event_code 4100 and 4110
# # other_idxs = sample_train.query('(event_code!=4100 | event_code!=4110) & game_session=="cace4c493ac347e3"').index
# # sample_train.loc[other_idxs,'num_correct'] = 0
# # sample_train.loc[other_idxs,'num_incorrect'] = 0
# # sample_train.loc[other_idxs,'accuracy'] = 0
# # sample_train.loc[other_idxs,'accuracy_group'] = 0




# sample_train['num_correct'] = 0
# sample_train['num_incorrect'] = 0
# sample_train['accuracy'] = 0
# sample_train['accuracy_group'] = 0

# # code for labeling all the game_sessions:
# all_sess_grps = list(sample_train.groupby('game_session',sort=False).groups.keys())
# for sess in tqdm(all_sess_grps,total=len(all_sess_grps)):
#     temp_1 = label_assign(sample_train,str(sess))
#     y = sample_train.loc[temp_1[1]].query('event_code==4100 | event_code==4110')
#     y_indexes = y.loc[y['event_data'].str.contains('true')].index
#     false_idxs = y.loc[y['event_data'].str.contains('false')].index

#     #values for true indices
#     sample_train.loc[y_indexes,'num_correct'] = temp_1[0]['num_correct']
#     sample_train.loc[y_indexes,'num_incorrect'] = temp_1[0]['num_incorrect']
#     sample_train.loc[y_indexes,'accuracy'] = temp_1[0]['accuracy']
#     sample_train.loc[y_indexes,'accuracy_group'] = temp_1[0]['accuracy_group']
    




sample_train.to_feather({path_w}/'saved_sample_trn_raw')




from sklearn.metrics import *




def print_score(m):
    results = [cohen_kappa_score(y_train,m.predict(X_train),weights='quadratic'),cohen_kappa_score(y_valid,m.predict(X_valid),weights='quadratic'),
               m.score(X_train,y_train),m.score(X_valid,y_valid)]
    print("Scores = ",results)




set_rf_samples(5000)




from sklearn.tree import *




model = DecisionTreeClassifier(min_samples_leaf=3,max_features=0.5)
model.fit(df_trn,y_trn)




model.score(df_trn,y_trn)




print_score(model)




train_data = pd.read_csv(path/'train.csv')




train_data = train_data.query('(event_code==4100 | event_code==4110) & type=="Assessment"')
train_data




corr_count = defaultdict(lambda :0)
incorr_count = defaultdict(lambda :0)
title = defaultdict(lambda :'')
game_sess = defaultdict(lambda :'')
install_id = defaultdict(lambda :'')

for idx in train_data.index:
    row = train_data.loc[idx]
    title[row.game_session] = row.title
    game_sess[row.game_session] = row.game_session
    install_id[row.game_session] = row.installation_id
    #import pdb; pdb.set_trace()

    if(row.event_code==4100):
        if('true' in row.event_data):
            corr_count[row.game_session] +=1
        else:
            incorr_count[row.game_session] +=1
            
    elif(row.event_code==4110):
        if('true' in row.event_data):
            corr_count[row.game_session] +=1
        else:
            incorr_count[row.game_session] +=1     
        
labels = pd.DataFrame({'game_session':game_sess,'installation_id':install_id,'title':title,'num_correct':corr_count
                              ,'num_incorrect':incorr_count},index=None)
labels.fillna(value=0,inplace=True)
labels.reset_index(inplace=True)
labels.drop(columns='index',inplace=True)

accuracy = labels.num_correct/(labels.num_correct+labels.num_incorrect)
labels['accuracy'] = accuracy 

temp = []
for x in labels.index:
    row = labels.loc[x]
    if(row.num_correct==1):
        temp.append(3)
    elif(row.num_correct==2):
        temp.append(2)
    elif(row.num_correct>=3):
        temp.append(1)
    elif(row.num_correct==0):
        temp.append(0)
        
labels['accuracy_group'] = temp
    




train_w_gt = pd.merge(left=train_data,right=labels,on='game_session',how='left')
train_w_gt




train_w_gt.drop(columns='event_data',inplace=True)
train_w_gt




add_datepart(train_w_gt,fldname='timestamp')
train_w_gt.dtypes




train_cats(train_w_gt)




df_trn, y_trn, nas = proc_df(train_w_gt,y_fld='accuracy_group')




def split_vals(a,n): return a[:n],a[n:]

n_valid = 1200
n_train = len(train_w_gt) - n_valid
X_train, X_valid = split_vals(df_trn,n_train)
y_train, y_valid = split_vals(y_trn,n_train)
raw_train, raw_valid = split_vals(train_w_gt,n_train)




X_valid




set_rf_samples(50000)




model = RandomForestClassifier(n_estimators=10,min_samples_leaf=3,max_features=0.5,n_jobs=-1)
model.fit(X_train,y_train)
print_score(model)




X_valid.iloc[0]






