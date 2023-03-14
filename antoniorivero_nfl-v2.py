#!/usr/bin/env python
# coding: utf-8



from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from keras.callbacks.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Activation

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# from scipy.spatial import distance_matrix




pd.options.display.max_columns = 50
pd.options.display.max_rows = 200




train_data = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv')




train_data.head()




plt.figure(figsize = (18,8))
plt.subplot(121)
plt.title('Yard Distribution for all Plays')
sns.distplot(train_data.Yards, kde = False)
plt.subplot(122)
plt.title('Yard Distribution for all Plays (with less than 40 Yards gained)')
sns.boxplot(train_data.Yards[lambda x: x < 40])




plt.figure(figsize = (40,8))
sns.boxplot(x = 'WindSpeed', y = 'Yards', data = train_data.sample(frac = 0.2))




# explore variables to see which have too much levels
plt.figure(figsize = (18,10))
ax1 = plt.subplot(211)
plt.title('All Categorical Variables')
((train_data.loc[:,train_data.dtypes == 'object']
                           .nunique().sort_values()
                          ) 
).plot(kind = 'bar', rot = 25, ax = ax1)


ax2 = plt.subplot(212)
plt.title('Categorical Variables with less than 300 levels')
((train_data.loc[:,train_data.dtypes == 'object']
                           .nunique().sort_values()
                          ) 
[lambda x: x < 300]
).plot(kind = 'bar', rot = 25, ax = ax2)
plt.tight_layout()




def is_offense(row):
    if row.Team == 'home':
        if row.HomeTeamAbbr == row.PossessionTeam:
            return True
        else:
            return False
    else:
        if row.VisitorTeamAbbr == row.PossessionTeam:
            return True
        else:
            return False
        
def avg_dist_between_teammates(play_team):
    dist_mat = distance_matrix(play_team[['X','Y']], play_team[['X','Y']])
    np.fill_diagonal(dist_mat, np.nan)
    return np.nanmean(dist_mat)


def off_vs_def_position_stats(row):
    columns = 'X_mean X_std Y_mean Y_std'.split()
    stats = {}
    stats['id'] = row['PlayId']
    if row.is_offense:
        prefix = 'Offense_'
        for i in columns:
                stats[prefix+i] = row[i]
    else:
        prefix = 'Defense_'
        for i in columns:
            stats[prefix+i] = row[i]

    return stats
    
        




def data_prep(data):
    data = data.copy()
    data = data.drop('WindDirection WindSpeed Temperature Humidity                          GameWeather StadiumType Location Week PlayerCollegeName                          HomeScoreBeforePlay VisitorScoreBeforePlay YardLine JerseyNumber                          DisplayName Stadium TimeHandoff TimeSnap FieldPosition Quarter Season GameClock'.split(), 
                     axis = 1, errors = 'ignore')

#     team_distance = data.groupby(['PlayId','Team'])[['X','Y']].apply(avg_dist_between_teammates).unstack(1).rename(columns = {'away':'away_distance','home':'home_distance'})
#     print('Is Offense')
    data['is_offense'] = (data.Team == 'home') == (data.HomeTeamAbbr == data.PossessionTeam)

#     data = data.merge(team_distance, left_on = 'PlayId', right_index = True)
#     print('Player Age')
    data['player_age'] = np.round((pd.to_datetime('today') - pd.to_datetime(data.PlayerBirthDate))/pd.Timedelta('1 y'))
    
#     print(3)
    data = data[['NflId', 'NflIdRusher'] + [i for i in list(data.columns) if i not in ['NflId', 'NflIdRusher']]]

#     data['DefTeamDist'] = data.apply(lambda x: x.home_distance if ((x.is_offense == False) == (x.Team == 'home')) else x.away_distance, axis = 1)
#     data['OffTeamDist'] = data.apply(lambda x: x.home_distance if ((x.is_offense == True) == (x.Team == 'home')) else x.away_distance, axis = 1)
    
#     print(4)
    pos_stats = data.groupby(['PlayId','Team'])[['X','Y']].agg([np.mean, np.std])
    
    pos_stats.columns = [*map(lambda x: '_'.join(x), pos_stats.columns.ravel())]
    
#     print(5)
    data = data.merge(pos_stats, left_on = 'PlayId Team'.split(), right_index = True)
    
#     print(6)
    def_vs_off_stats = data.groupby(['Team', 'PlayId'])['X_mean X_std Y_mean Y_std'.split()].max().unstack(0)
    def_vs_off_stats.columns = [*map(lambda x: '_'.join(x), def_vs_off_stats.columns.ravel())]
    
#     print(7)
    data = data.merge(def_vs_off_stats, left_on = 'PlayId', right_index = True)

    # Think how to include Orientation in the ABT. Maybe one column per player.
    # Same with PlayDirection

    data = data.drop('PlayerHeight HomeTeamAbbr VisitorTeamAbbr is_offense Dir Orientation team_distance PlayerBirthDate PossessionTeam Team'.split(), axis = 1, errors = 'ignore')

    data = data.drop('X_mean X_std Y_mean Y_std'.split(), axis = 1)
    
    cat_vars = data.dtypes[data.dtypes == 'object'].index
    
#     print(8)
    data = pd.get_dummies(data, columns = cat_vars)
    
    data = data[data.NflIdRusher == data.NflId]

    data = data.drop('NflId NflIdRusher GameId'.split(), axis = 1)
    data = data.set_index('PlayId')

    # Revisar si esto es lo que quiero hacer! Ver por qué hay tantos NaNs en el Offense 
    data = data.dropna()
    return data




def train_model(data):
    data = data.copy()
    data = data[data.Yards < 15]
    X = data.drop(['Yards'], axis = 1)
    y = pd.get_dummies(data['Yards'])
    
    labels = list(y.columns)
    
    model = Sequential()
    model.add(Dense(200, input_shape=(X.shape[1],), activation='sigmoid'))
    model.add(Dense(200, activation='sigmoid'))
#     model.add(Dense(80, activation='sigmoid'))
    
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    model.fit(X, y, epochs=20, batch_size=32, callbacks = [EarlyStopping('accuracy', patience = 2)])
    
#     l_reg = LogisticRegression(multi_class='ovr')
#     l_reg.fit(X, y)
    
    return model, labels




def make_prediction(test_df, sample_sub, model, labels):
    test_df = test_df.copy()
    preped_df = preped_train_data.drop('Yards',axis = 1).align(data_prep(test_df), axis = 1, join = 'left')[1].fillna(0)
    temp_results = pd.DataFrame(model.predict_proba(preped_df), columns = ['Yards'+str(i) for i in labels])
    sample_sub.update(temp_results)
    sample_sub = sample_sub.cumsum(axis = 1)
    sample_sub[sample_sub > 1] = 1
    return sample_sub




from kaggle.competitions import nflrush
env = nflrush.make_env()




print('Preping Data')
preped_train_data = data_prep(train_data)

print('Training')
model, labels = train_model(preped_train_data)




print('Predicting')
for (test_df, sample_prediction_df) in env.iter_test():
    sample_prediction_df = sample_prediction_df.replace(sample_prediction_df, 0).reset_index(drop=True)
    predictions_df = make_prediction(test_df, sample_prediction_df, model, labels)
    env.predict(predictions_df)

print('Writing Submissions')
env.write_submission_file()

