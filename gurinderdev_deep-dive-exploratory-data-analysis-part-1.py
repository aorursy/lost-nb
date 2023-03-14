#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import json
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.style.use('seaborn')
get_ipython().run_line_magic('matplotlib', 'inline')

# # Input data files are available in the "../input/" directory.
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # Any results you write to the current directory are saved as output.




cd '/kaggle/input/data-science-bowl-2019'




# get the data into a pandas dataframe
train = pd.read_csv('train.csv')




no_unique ={}
unique_values = {}
dataframes ={}

for col in train.columns:
    no_unique[col] = train[col].nunique()
    if no_unique[col] <11:
        unique_values[col] = train[col].unique()
    else: 
        unique_values[col] = 'Too many to list.Eg: {}'.format(train[col].unique()[:4])
    
    dataframes[col] = pd.DataFrame([{
         'column_name' : col
     ,   'no_unique_values' : no_unique[col]
    ,   'unique_values' : unique_values[col]
    
    }], index = [col])

df_unique = pd.concat([dataframes[col] for col in train.columns], ignore_index = True)
df_unique.sort_values(by = 'no_unique_values').reset_index()




g =train.groupby('world')
x = g['installation_id'].nunique()
x.reset_index().plot.bar(x ='world' , y = 'installation_id' )




# # which world is the played the most 
g['game_time'].sum().reset_index().plot.bar(x ='world' , y = 'game_time' )




# # what is the breakup of game and type 
x = train.groupby(['type','world' ])['game_time'].sum().sort_values()

# build a dataframe that has all the types with worlds and corresponding sum of time
type_ = train.type.unique()
df ={}
for t in type_:
    df[t] = (x.get(t).reset_index()).set_index('world').rename(columns = {'game_time':t})
df_f = pd.concat([df[t] for t in type_ ] , ignore_index = False , axis = 1, sort = True)
df_f = df_f.fillna(0)

# get individual series as we want to create a chart with all of the differnt types 
worlds = list(df_f.index)
clip = list(df_f.iloc[:,0])
Activity =  list(df_f.iloc[:,1])
Game =  list(df_f.iloc[:,2])
Assessment = list( list(df_f.iloc[:,3]))
del worlds[2]
del clip[2]
del Activity[2]
del Game[2]
del Assessment[2]

# plot the chart 
plt.xlabel('Worlds')
plt.ylabel('Time')
plt.title('Time spent by each activity in different Worlds')

plt.plot(worlds, clip , label = 'Clip')
plt.plot(worlds, Activity , label = 'Activity')
plt.plot(worlds, Game, label = 'Game' )
plt.plot(worlds, Assessment , label = 'Assessment')
plt.legend(loc = 'best')




# how many game sessions were spent on the different types
types_ = list(train.groupby('type')['game_session'].count().sort_values().index)
values = list(train.groupby('type')['game_session'].count().sort_values())
plt.xlabel('Types')
plt.ylabel('Number of Sessions')
plt.title('Breakdown of Sessions by Type')
plt.plot(types_ , values)
 




# plot the charts

i = 0
plt.figure()

for column in list(train.describe().columns):
    i += 1
    ax_gen  = 'ax'+str(i)
    fig_gen = 'fig'+ str(i)
    fig_gen , ax_gen = plt.subplots()  
    ax_gen.hist(x = train[column].dropna())        
    ax_gen.hist(x = train[column].dropna())
    ax_gen.set_title(column)
#     plt.tight_layout()
#     ax_gen.show()

