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




import numpy as np
import pandas as pd
pd.set_option('max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
import datetime
import lightgbm as lgb
from scipy import stats
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split, KFold
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
stop = set(stopwords.words('english'))
import os
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import xgboost as xgb
import lightgbm as lgb
from sklearn import model_selection
from sklearn.metrics import accuracy_score
import json
import ast
import eli5
import shap
from catboost import CatBoostRegressor
from urllib.request import urlopen
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import time
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn import linear_model




trainAdditionalFeatures = pd.read_csv('../input/tmdb-competition-additional-features/TrainAdditionalFeatures.csv')
testAdditionalFeatures = pd.read_csv('../input/tmdb-competition-additional-features/TestAdditionalFeatures.csv')

train = pd.read_csv('../input/tmdb-box-office-prediction/train.csv')
test = pd.read_csv('../input/tmdb-box-office-prediction/test.csv')

train = pd.merge(train, trainAdditionalFeatures, how='left', on=['imdb_id'])
test = pd.merge(test, testAdditionalFeatures, how='left', on=['imdb_id'])
test['revenue'] = -np.inf
train.loc[train['id'] == 16,'revenue'] = 192864          # Skinning
train.loc[train['id'] == 90,'budget'] = 30000000         # Sommersby          
train.loc[train['id'] == 118,'budget'] = 60000000        # Wild Hogs
train.loc[train['id'] == 149,'budget'] = 18000000        # Beethoven
train.loc[train['id'] == 313,'revenue'] = 12000000       # The Cookout 
train.loc[train['id'] == 451,'revenue'] = 12000000       # Chasing Liberty
train.loc[train['id'] == 464,'budget'] = 20000000        # Parenthood
train.loc[train['id'] == 470,'budget'] = 13000000        # The Karate Kid, Part II
train.loc[train['id'] == 513,'budget'] = 930000          # From Prada to Nada
train.loc[train['id'] == 797,'budget'] = 8000000         # Welcome to Dongmakgol
train.loc[train['id'] == 819,'budget'] = 90000000        # Alvin and the Chipmunks: The Road Chip
train.loc[train['id'] == 850,'budget'] = 90000000        # Modern Times
train.loc[train['id'] == 1007,'budget'] = 2              # Zyzzyx Road 
train.loc[train['id'] == 1112,'budget'] = 7500000        # An Officer and a Gentleman
train.loc[train['id'] == 1131,'budget'] = 4300000        # Smokey and the Bandit   
train.loc[train['id'] == 1359,'budget'] = 10000000       # Stir Crazy 
train.loc[train['id'] == 1542,'budget'] = 1              # All at Once
train.loc[train['id'] == 1570,'budget'] = 15800000       # Crocodile Dundee II
train.loc[train['id'] == 1571,'budget'] = 4000000        # Lady and the Tramp
train.loc[train['id'] == 1714,'budget'] = 46000000       # The Recruit
train.loc[train['id'] == 1721,'budget'] = 17500000       # Cocoon
train.loc[train['id'] == 1865,'revenue'] = 25000000      # Scooby-Doo 2: Monsters Unleashed
train.loc[train['id'] == 1885,'budget'] = 12             # In the Cut
train.loc[train['id'] == 2091,'budget'] = 10             # Deadfall
train.loc[train['id'] == 2268,'budget'] = 17500000       # Madea Goes to Jail budget
train.loc[train['id'] == 2491,'budget'] = 6              # Never Talk to Strangers
train.loc[train['id'] == 2602,'budget'] = 31000000       # Mr. Holland's Opus
train.loc[train['id'] == 2612,'budget'] = 15000000       # Field of Dreams
train.loc[train['id'] == 2696,'budget'] = 10000000       # Nurse 3-D
train.loc[train['id'] == 2801,'budget'] = 10000000       # Fracture
train.loc[train['id'] == 335,'budget'] = 2 
train.loc[train['id'] == 348,'budget'] = 12
train.loc[train['id'] == 470,'budget'] = 13000000 
train.loc[train['id'] == 513,'budget'] = 1100000
train.loc[train['id'] == 640,'budget'] = 6 
train.loc[train['id'] == 696,'budget'] = 1
train.loc[train['id'] == 797,'budget'] = 8000000 
train.loc[train['id'] == 850,'budget'] = 1500000
train.loc[train['id'] == 1199,'budget'] = 5 
train.loc[train['id'] == 1282,'budget'] = 9               # Death at a Funeral
train.loc[train['id'] == 1347,'budget'] = 1
train.loc[train['id'] == 1755,'budget'] = 2
train.loc[train['id'] == 1801,'budget'] = 5
train.loc[train['id'] == 1918,'budget'] = 592 
train.loc[train['id'] == 2033,'budget'] = 4
train.loc[train['id'] == 2118,'budget'] = 344 
train.loc[train['id'] == 2252,'budget'] = 130
train.loc[train['id'] == 2256,'budget'] = 1 
train.loc[train['id'] == 2696,'budget'] = 10000000

#Clean Data
test.loc[test['id'] == 6733,'budget'] = 5000000
test.loc[test['id'] == 3889,'budget'] = 15000000
test.loc[test['id'] == 6683,'budget'] = 50000000
test.loc[test['id'] == 5704,'budget'] = 4300000
test.loc[test['id'] == 6109,'budget'] = 281756
test.loc[test['id'] == 7242,'budget'] = 10000000
test.loc[test['id'] == 7021,'budget'] = 17540562       #  Two Is a Family
test.loc[test['id'] == 5591,'budget'] = 4000000        # The Orphanage
test.loc[test['id'] == 4282,'budget'] = 20000000       # Big Top Pee-wee
test.loc[test['id'] == 3033,'budget'] = 250 
test.loc[test['id'] == 3051,'budget'] = 50
test.loc[test['id'] == 3084,'budget'] = 337
test.loc[test['id'] == 3224,'budget'] = 4  
test.loc[test['id'] == 3594,'budget'] = 25  
test.loc[test['id'] == 3619,'budget'] = 500  
test.loc[test['id'] == 3831,'budget'] = 3  
test.loc[test['id'] == 3935,'budget'] = 500  
test.loc[test['id'] == 4049,'budget'] = 995946 
test.loc[test['id'] == 4424,'budget'] = 3  
test.loc[test['id'] == 4460,'budget'] = 8  
test.loc[test['id'] == 4555,'budget'] = 1200000 
test.loc[test['id'] == 4624,'budget'] = 30 
test.loc[test['id'] == 4645,'budget'] = 500 
test.loc[test['id'] == 4709,'budget'] = 450 
test.loc[test['id'] == 4839,'budget'] = 7
test.loc[test['id'] == 3125,'budget'] = 25 
test.loc[test['id'] == 3142,'budget'] = 1
test.loc[test['id'] == 3201,'budget'] = 450
test.loc[test['id'] == 3222,'budget'] = 6
test.loc[test['id'] == 3545,'budget'] = 38
test.loc[test['id'] == 3670,'budget'] = 18
test.loc[test['id'] == 3792,'budget'] = 19
test.loc[test['id'] == 3881,'budget'] = 7
test.loc[test['id'] == 3969,'budget'] = 400
test.loc[test['id'] == 4196,'budget'] = 6
test.loc[test['id'] == 4221,'budget'] = 11
test.loc[test['id'] == 4222,'budget'] = 500
test.loc[test['id'] == 4285,'budget'] = 11
test.loc[test['id'] == 4319,'budget'] = 1
test.loc[test['id'] == 4639,'budget'] = 10
test.loc[test['id'] == 4719,'budget'] = 45
test.loc[test['id'] == 4822,'budget'] = 22
test.loc[test['id'] == 4829,'budget'] = 20
test.loc[test['id'] == 4969,'budget'] = 20
test.loc[test['id'] == 5021,'budget'] = 40 
test.loc[test['id'] == 5035,'budget'] = 1 
test.loc[test['id'] == 5063,'budget'] = 14 
test.loc[test['id'] == 5119,'budget'] = 2 
test.loc[test['id'] == 5214,'budget'] = 30 
test.loc[test['id'] == 5221,'budget'] = 50 
test.loc[test['id'] == 4903,'budget'] = 15
test.loc[test['id'] == 4983,'budget'] = 3
test.loc[test['id'] == 5102,'budget'] = 28
test.loc[test['id'] == 5217,'budget'] = 75
test.loc[test['id'] == 5224,'budget'] = 3 
test.loc[test['id'] == 5469,'budget'] = 20 
test.loc[test['id'] == 5840,'budget'] = 1 
test.loc[test['id'] == 5960,'budget'] = 30
test.loc[test['id'] == 6506,'budget'] = 11 
test.loc[test['id'] == 6553,'budget'] = 280
test.loc[test['id'] == 6561,'budget'] = 7
test.loc[test['id'] == 6582,'budget'] = 218
test.loc[test['id'] == 6638,'budget'] = 5
test.loc[test['id'] == 6749,'budget'] = 8 
test.loc[test['id'] == 6759,'budget'] = 50 
test.loc[test['id'] == 6856,'budget'] = 10
test.loc[test['id'] == 6858,'budget'] =  100
test.loc[test['id'] == 6876,'budget'] =  250
test.loc[test['id'] == 6972,'budget'] = 1
test.loc[test['id'] == 7079,'budget'] = 8000000
test.loc[test['id'] == 7150,'budget'] = 118
test.loc[test['id'] == 6506,'budget'] = 118
test.loc[test['id'] == 7225,'budget'] = 6
test.loc[test['id'] == 7231,'budget'] = 85
test.loc[test['id'] == 5222,'budget'] = 5
test.loc[test['id'] == 5322,'budget'] = 90
test.loc[test['id'] == 5350,'budget'] = 70
test.loc[test['id'] == 5378,'budget'] = 10
test.loc[test['id'] == 5545,'budget'] = 80
test.loc[test['id'] == 5810,'budget'] = 8
test.loc[test['id'] == 5926,'budget'] = 300
test.loc[test['id'] == 5927,'budget'] = 4
test.loc[test['id'] == 5986,'budget'] = 1
test.loc[test['id'] == 6053,'budget'] = 20
test.loc[test['id'] == 6104,'budget'] = 1
test.loc[test['id'] == 6130,'budget'] = 30
test.loc[test['id'] == 6301,'budget'] = 150
test.loc[test['id'] == 6276,'budget'] = 100
test.loc[test['id'] == 6473,'budget'] = 100
test.loc[test['id'] == 6842,'budget'] = 30

# from this kernel: https://www.kaggle.com/gravix/gradient-in-a-box
dict_columns = ['belongs_to_collection', 'genres', 'production_companies',
                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']

def text_to_dict(df):
    for column in dict_columns:
        df[column] = df[column].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x) )
    return df
        
train = text_to_dict(train)
test = text_to_dict(test)

def fix_date(x):
    """
    Fixes dates which are in 20xx
    """
    if not isinstance(x, str): return x
    year = x.split('/')[2]
    if int(year) <= 19:
        return x[:-2] + '20' + year
    else:
        return x[:-2] + '19' + year

train.loc[train['release_date'].isnull() == True, 'release_date'] = '01/01/19'
test.loc[test['release_date'].isnull() == True, 'release_date'] = '01/01/19'
    
#train["RevByBud"] = train["revenue"] / train["budget"]
    
train['release_date'] = train['release_date'].apply(lambda x: fix_date(x))
test['release_date'] = test['release_date'].apply(lambda x: fix_date(x))




train.head(10)




train['producer_avg_popularity'].isnull().sum()




# Adding new colums ,collection name and collection indication
# Removing the 'belong_to_collection' column
train['collection_name'] = train['belongs_to_collection'].apply(lambda x: x[0]['name'] if x != {} else 0)
train['has_collection'] = train['belongs_to_collection'].apply(lambda x: len(x) if x != {} else 0)

test['collection_name'] = test['belongs_to_collection'].apply(lambda x: x[0]['name'] if x != {} else 0)
test['has_collection'] = test['belongs_to_collection'].apply(lambda x: len(x) if x != {} else 0)

train = train.drop(['belongs_to_collection'], axis=1)
test = test.drop(['belongs_to_collection'], axis=1)




# Most of the movies don't have homepage. So we will add new colums of is homepage exist
train['has_homepage'] = 0
train.loc[train['homepage'].isnull() == False, 'has_homepage'] = 1
test['has_homepage'] = 0
test.loc[test['homepage'].isnull() == False, 'has_homepage'] = 1

train = train.drop(['homepage'], axis=1)
test = test.drop(['homepage'], axis=1)




# checks the unique genres of the movies
genre_list = []
for i in range(len(train)):
        for j in range((len(train['genres'][i]))):
                genre_list.append(train['genres'][i][j]['name'])
for i in range(len(test)):
        for j in range((len(test['genres'][i]))):
                genre_list.append(test['genres'][i][j]['name'])
genres = []
for x in genre_list:
    if x not in genres:
            genres.append(x)
for x in genres:
    train[x] = 0
    test[x] = 0
print(genres)
len(genres)




# train['genres'][1][0]['name']




# updated the new columns of each genre

for i in range(len(train)):
        for j in range(len(train['genres'][i])):
             train[train['genres'][i][j]['name']][i] = 1
for i in range(len(test)):
        for j in range(len(test['genres'][i])):
             test[test['genres'][i][j]['name']][i] = 1




# Adding Producer and Director column to data
train['Producer'] = train['crew'].apply(lambda x : [str(i['name']) for i in x if i['job']=='Producer'])
test['Producer'] = test['crew'].apply(lambda x : [str(i['name']) for i in x if i['job']=='Producer'])

train['ProducerT'] = train['Producer'].apply(lambda x: str(x)[2:-2])
test['ProducerT'] = test['Producer'].apply(lambda x: str(x)[2:-2])

train['Director'] = train['crew'].apply(lambda x : [i['name'] for i in x if i['job']=='Director'])
test['Director'] = test['crew'].apply(lambda x : [i['name'] for i in x if i['job']=='Director'])

train['DirectorT'] = train['Director'].apply(lambda x: str(x)[2:-2])
test['DirectorT'] = test['Director'].apply(lambda x: str(x)[2:-2])




train[] = train['crew'].apply(lambda x : [i['name'] for i in x if i['department']=='Crew'])
temp




# Delete Poster column
train = train.drop(['poster_path'], axis=1)
test = test.drop(['poster_path'], axis=1)




# Shows the departments 
departments = []
for i in range(len(train['crew'][0])):
    if train['crew'][0][i]['department'] not in departments:
        departments.append(train['crew'][0][i]['department'])
departments




# Separate the release date to weekday,month quarter and year
different_dates = ['week_day','month', 'quarter', 'year']
for i in different_dates:
    train[i] = 0
    test[i] = 0
dates = pd.to_datetime(train['release_date'])
for i in range(len(train)):
    train['week_day'][i] = dates[i].day_name()
    train['month'][i] = dates[i].month_name()
    train['quarter'][i] = dates[i].quarter
    train['year'][i] = dates[i].year
dates = pd.to_datetime(test['release_date'])
for i in range(len(test)):
    test['week_day'][i] = dates[i].day_name()
    test['month'][i] = dates[i].month_name()
    test['quarter'][i] = dates[i].quarter
    test['year'][i] = dates[i].year

train = train.drop(['release_date'], axis=1)
test = test.drop(['release_date'], axis=1)




# Creates new dataFrame of each derector, producer and its value
# Complete missing values of rating by filling it with mean value of train+test data
ab = pd.concat((train['rating'],test['rating']),axis=0)
mn = ab.mean()
train['rating'] = train['rating'].fillna(mn)
test['rating'] = test['rating'].fillna(mn)

# Creates list and data frame for producers and directors
directors_list = np.concatenate((train['DirectorT'].unique(),test['DirectorT'].unique()))
unique_directors = np.unique(directors_list).tolist()
df = pd.DataFrame(unique_directors, columns = ['name' ])
df['movies_num'] = 0
df['score'] = 0
df['popularity'] = 0.0
df['director_avg_score'] = 0.0
df['director_avg_popularity'] = 0.0
df.set_index('name', inplace=True)


producers_list = np.concatenate((train['ProducerT'].unique(),test['ProducerT'].unique()))
unique_producers = np.unique(producers_list).tolist()
df2 = pd.DataFrame(unique_producers, columns = ['name' ])
df2['movies_num'] = 0
df2['score'] = 0
df2['popularity'] = 0.0
df2['producer_avg_score'] = 0.0
df2['producer_avg_popularity'] = 0.0
df2.set_index('name', inplace=True)

# Calculates the average rating and popularity of each Director/Producer
for i in range(len(train)):
    df.at[train['DirectorT'][i],'score'] += train['rating'][i]
    df.at[train['DirectorT'][i],'popularity'] += train['popularity'][i]
    df.at[train['DirectorT'][i],'movies_num'] += 1
     
    df2.at[train['ProducerT'][i],'score'] += train['rating'][i]
    df2.at[train['ProducerT'][i],'popularity'] += train['popularity'][i]
    df2.at[train['ProducerT'][i],'movies_num'] += 1
        
for i in range(len(test)):
    df.at[test['DirectorT'][i],'score'] += test['rating'][i]
    df.at[test['DirectorT'][i],'popularity'] += test['popularity'][i]
    df.at[test['DirectorT'][i],'movies_num'] += 1
    
    df2.at[test['ProducerT'][i],'score'] += test['rating'][i]
    df2.at[test['ProducerT'][i],'popularity'] += test['popularity'][i]
    df2.at[test['ProducerT'][i],'movies_num'] += 1

for i in range(len(unique_directors)):
    df.at[unique_directors[i],'director_avg_score'] = df.at[unique_directors[i],'score']/df.at[unique_directors[i],'movies_num']
    df.at[unique_directors[i],'director_avg_popularity'] = df.at[unique_directors[i],'popularity']/df.at[unique_directors[i],'movies_num']
for i in range(len(unique_producers)):   
    df2.at[unique_producers[i],'producer_avg_score'] = df2.at[unique_producers[i],'score']/df2.at[unique_producers[i],'movies_num']
    df2.at[unique_producers[i],'producer_avg_popularity'] = df2.at[unique_producers[i],'popularity']/df2.at[unique_producers[i],'movies_num']
    
# Creates new columns of the average score/popularity of directors/producers in specific film
test['director_avg_score'] = 0.0
test['director_avg_popularity'] = 0.0
train['director_avg_score'] = 0.0
train['director_avg_popularity'] = 0.0

test['producer_avg_score'] = 0.0
test['producer_avg_popularity'] = 0.0
train['producer_avg_score'] = 0.0
train['producer_avg_popularity'] = 0.0

for i in range(len(train)):
    if len(train['Director'][i])==0:
        train['director_avg_popularity'][i] = 0.0
        train['director_avg_score'][i] = 0.0
    else :
        train['director_avg_popularity'][i] = df.at[train['DirectorT'][i],'director_avg_popularity']/len(train['Director'][i])
        train['director_avg_score'][i] = df.at[train['DirectorT'][i],'director_avg_score']/len(train['Director'][i])
    
    if len(train['Producer'][i])==0:
        train['producer_avg_popularity'][i] = 0.0
        train['producer_avg_score'][i] = 0.0
    else :
        train['producer_avg_popularity'][i] = df2.at[train['ProducerT'][i],'producer_avg_popularity']/len(train['Producer'][i])
        train['producer_avg_score'][i] = df2.at[train['ProducerT'][i],'producer_avg_score']/len(train['Producer'][i])
    

for i in range(len(test)):
    if len(test['Director'][i])==0:
        test['director_avg_popularity'][i] = 0.0
        test['director_avg_score'][i] = 0.0
    else :
        test['director_avg_popularity'][i] = df.at[test['DirectorT'][i],'director_avg_popularity']/len(test['Director'][i])
        test['director_avg_score'][i] = df.at[test['DirectorT'][i],'director_avg_score']/len(test['Director'][i])
    
    if len(test['Producer'][i])==0:
        test['producer_avg_popularity'][i] = 0.0
        test['producer_avg_score'][i] = 0.0
    else :
        test['producer_avg_popularity'][i] = df2.at[test['ProducerT'][i],'producer_avg_popularity']/len(test['Producer'][i])
        test['producer_avg_score'][i] = df2.at[test['ProducerT'][i],'producer_avg_score']/len(test['Producer'][i])



# Crew popularity is more relevant than avg rating/popularity 
train['crew_popularity'] = train['producer_avg_popularity'] + train['director_avg_popularity']
test['crew_popularity'] = test['producer_avg_popularity'] + test['director_avg_popularity']

# Deletes producer,director and crew columns
drop_colums = ['Producer', 'Director', 'DirectorT', 'ProducerT', 'crew','director_avg_popularity', 'producer_avg_popularity','producer_avg_score','director_avg_score']
for i in drop_colums:
    train = train.drop([i], axis=1)
    test = test.drop([i], axis=1)




# df.loc[unique_directors[0]]['score']/df.loc[unique_directors[0]]['movies_num']
df.loc[unique_directors[0]]['score']
for i in range(len(unique_directors)):
    df.loc[unique_directors[i]]['score'] = df.loc[unique_directors[i]]['score']/df.loc[unique_directors[i]]['movies_num']




# len(train['Producer'][2])
# df.loc[train['DirectorT'][0]]['movies_num']
# df.loc[train['DirectorT'][0]]['movies_sum'] +=1
# if len(test['Director'][3])==0:
#     print('True')
# else:
#     print('False')
    
# df.loc[train['DirectorT'][1]]['score'] = df.loc[train['DirectorT'][1]]['score'] + train['rating'][1]
# df.at[train['DirectorT'][1],'score']
# ab = train['rating']
# ab.isnull().sum()
# ab = ab.fillna(ab.mean())
# train['rating'] = train['rating'].fillna(train['rating'].mean())
ab = pd.concat((train['rating'],test['rating']),axis=0)
mn = ab.mean()
# train['rating'] = train['rating'].fillna(mn)
# test['rating'] = test['rating'].fillna(mn)
# print(len(test),len(train),len(ab),ab.mean())
# list(train['rating'])
mn




# Creates list top twenty producers
top_twenty_producers = train['DirectorT'].value_counts()[1:21]
test_top_twenty_producers = test['DirectorT'].value_counts()[1:21]
temp = top_twenty_producers.reset_index()
temp2 = test_top_twenty_producers.reset_index()
top_Directors = []
for i in range(len(top_twenty_producers)):
    if temp['index'][i] not in top_Directors:
        top_Directors.append(temp['index'][i])
    if temp2['index'][i] not in top_Directors:
        top_Directors.append(temp2['index'][i])

top_Directors




# train.head()
# Crew popularity is more relevant than avg rating/popularity 
train['crew_rating'] = train['producer_avg_score'] + train['director_avg_score']
train['crew_popularity'] = train['producer_avg_popularity'] + train['director_avg_popularity']




sns.catplot(x='crew_popularity', y='revenue', data=train);
plt.title('Revenue for film with and without homepage')




sns.catplot(x='producer_avg_popularity', y='revenue', data=train);
plt.title('Revenue for film with and without homepage')




a = train['runtime']
np.mean(a)
# sns.distplot(train['runtime'])




g = sns.relplot(x="director", y ="revenue", data=train)

