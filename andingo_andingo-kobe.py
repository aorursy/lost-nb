# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import datetime

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFdr

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/data.csv')
data['time_remaining'] = 60*data.minutes_remaining + data.seconds_remaining
angle = np.arctan(abs(data.loc_y) / abs(data.loc_x))
angle[np.isnan(angle)] = np.pi/2
I = (data.loc_x < 0) & (data.loc_y >= 0)
angle[I] = np.pi - angle[I]
I = (data.loc_x < 0) & (data.loc_y < 0)
angle[I] += np.pi
data['angle'] = angle
sym_angle = angle
sym_angle[I] = np.pi - angle[I]
data['sym_angle'] = sym_angle
data['game_date'] = [datetime.datetime.strptime(x, '%Y-%m-%d').date() for x in data.game_date]

nan_inds = np.where(np.isnan(data.shot_made_flag))[0]
categories = np.unique(data.combined_shot_type)

X = data[['time_remaining','angle','sym_angle','shot_distance','period']]
Y = data['shot_made_flag']

preds = {'shot_id': [], 'shot_made_flag': []}
for ind in nan_inds[:1000]:
    print ind
    if ind < 1000:
        preds['shot_made_flag'].append(.4)
        continue
    
    subY = Y.iloc[:ind]
    I = ~np.isnan(subY)
    subY = subY[I]
    subX = X.iloc[:ind][I]
    
    cat = data['combined_shot_type'].iloc[ind]
    date = data['game_date'].iloc[ind]
    shot_id = data['shot_id'].iloc[ind]
    preds['shot_id'].append(shot_id)
    
    I = data['combined_shot_type'].loc[subX.index]==cat
    
    x = subX[I]
    y = subY[I]
    
    if len(y) < 1000:
        preds['shot_made_flag'].append(.4)
        continue

    dates = data['game_date'].loc[subX.index][I]
    
    t = np.array([(date - x).days / 365. for x in dates])
    wgt = np.exp(-.3*t)
    wgt /= wgt.sum()
    
    select = SelectFdr()
    select.fit(x, y)
    
    cols = subX.columns[select.pvalues_ < .2]
    
    if len(cols) == 0:
        preds['shot_made_flag'].append(.4)
        continue
    
    x = x[cols]
    
    clf = RandomForestClassifier(500)
    clf.fit(x, y, wgt)
    
    prob = clf.predict_proba(X[cols].iloc[ind:ind+1])[0, 1]
    preds['shot_made_flag'].append(prob)


