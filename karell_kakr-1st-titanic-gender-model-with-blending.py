#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
# from xgboost import XGBClassifier
from sklearn import preprocessing, model_selection
from sklearn.metrics import accuracy_score

pd.options.mode.chained_assignment = None

train = pd.read_csv('../input/2019-1st-ml-month-with-kakr/train.csv')
test = pd.read_csv('../input/2019-1st-ml-month-with-kakr/test.csv')

print(train.info())
train.sample(5)


# In[2]:


x_train = train.drop(['PassengerId', 'Survived'], axis=1)
x_test = test.drop(['PassengerId'], axis=1)
y_train = train.Survived

all = pd.concat([x_train, x_test]).reset_index(drop=True)

print('[All data] \n', all.isnull().sum())


# In[3]:


age_med = np.nanmedian(all.Age)
fare_med = np.nanmedian(all.Fare)

print(age_med, '//', 
      fare_med, '//',
      all.Cabin.isnull().sum(), '//',
      all.Embarked.isnull().sum(), '\n')

print(all.groupby('Embarked').Embarked.count(), '\n')

# Age median : 28
# Fare median : 14.4542
# Cabin : unknown 1014
# Embarked : two missing values input 'S'

all.Age = all.Age.fillna(age_med)
all.Age = pd.qcut(all.Age, 4, labels=False)
all.Fare = all.Fare.fillna(fare_med)
all.Fare = pd.qcut(all.Fare, 4, labels=False)
all.Cabin = all.Cabin.fillna('unknown')
all.Cabin = all.Cabin.apply(lambda x: x[0])
all.Embarked = all.Embarked.fillna('S')

print(all.isnull().sum())
#print(all.info())


# In[4]:


def data_Preprocessing(df):
    df['Gender'] = df.Name.apply(lambda x: re.split(', |\\. ', x)[1])

    name_man = [df.Gender[df.Sex == 'male'].unique()]
    name_woman = df.Gender[df.Sex == 'female'].unique().tolist()
    name_man = np.delete(name_man, 1).tolist()

    df.loc[df.Gender.isin(name_man), 'Gender'] = 'man'
    df.loc[df.Gender.isin(name_woman), 'Gender'] = 'woman'
    df.loc[df.Gender == 'Master', 'Gender'] = 'boy'

    df['Family_name'] = df.Name.apply(lambda x: x.split(',')[0])
    df.loc[df.Gender == 'man', 'Family_name'] = 'alone'
    df['Family_freq'] = df.groupby('Family_name')['Family_name'].transform('count')
    df.loc[df.Family_freq <= 1, 'Family_name'] = 'alone'

    df.Ticket = df.Ticket.apply(lambda x: x[:-1] + 'X')
    df['Group'] = df.Family_name + '-' +                   df.Pclass.map(str) + '-' +                   df.Fare.map(str) + '-' +                   df.Ticket + '-' +                   df.Embarked

    # 581 example
    for i in range(0, len(df)):
        if df.loc[i,'Gender'] != 'man' and df.loc[i,'Family_name'] == 'alone':
            
            df.loc[i,'Family_name'] =             df.Family_name[df.Ticket == df.Ticket[i]].iloc[0]

    df = df.drop(['Name', 'Ticket', 'Sex', 'SibSp', 'Parch'], axis=1)

    return df


# In[5]:


df = data_Preprocessing(all)

df.loc[0:890, 'Survived'] = y_train
temp = df.loc[0:890]
temp['Family_freq'] = temp.groupby('Family_name')['Family_name'].transform('count')
temp['Family_survival'] = temp.groupby('Family_name')['Survived'].transform('sum') / temp.groupby('Family_name')['Family_name'].transform('count')

temp['Group_freq'] = temp.groupby('Group')['Group'].transform('count')
temp['Group_survival'] = temp.groupby('Group')['Survived'].transform('sum') / temp.groupby('Group')['Group'].transform('count')


df['Family_survival'] = temp.Family_survival
df['Group_survival'] = temp.Group_survival

del temp

for i in range(891, len(df)):
    df.loc[i, 'Family_survival'] = df.Family_survival[df.Family_name == df.Family_name[i]].iloc[0]
    
df.loc[(df.Family_survival.isnull()) & (df.Pclass == 3), 'Family_survival'] = 0
df.loc[(df.Family_survival.isnull()) & (df.Pclass != 3), 'Family_survival'] = 1

df['predict'] = int(0)
df.loc[df.Gender == 'woman', 'predict'] = 1
df.loc[(df.Gender == 'boy') & (df.Family_survival == 1), 'predict'] = 1
df.loc[(df.Gender == 'woman') & (df.Family_survival == 0), 'predict'] = 0


# In[6]:


sub_GM = test[['PassengerId']]
sub_GM['Survived'] = np.int64(df.loc[891:len(df), 'predict'].values)
sub_GM.to_csv('kernel_GM.csv', index=False)

sub_GM.sample(10)


# In[7]:


blend = pd.read_csv('../input/titanic-first-blending/blendind_material.csv')
blend.head(5)


# In[8]:


for i in range(0, len(blend)):
    for c in range(3, 9):
        blend.loc[i, 'Sum'] = blend.loc[i, 'Sum'] + blend.iloc[i,c]
        
    if blend.loc[i, 'best'] == 0 and blend.loc[i, 'Sum'] > 2.5:
        blend.loc[i, 'Survived'] = 1
            
    if blend.loc[i, 'best'] == 1 and blend.loc[i, 'Sum'] < 2.5:
        blend.loc[i, 'Survived'] = 0

blend.head(10)  


# In[9]:


blend = blend.iloc[:, 0:2]
blend.to_csv('kernel_blend_1st.csv', index=False)


# In[10]:


blend2 = pd.read_csv('../input/titanic-second-blending/blendind_material2.csv')

for i in range(0, len(blend2)):
    for c in range(2,7):
        blend2.iloc[i,7] = blend2.iloc[i,7] + blend2.iloc[i,c]
        
    if blend2.loc[i, 'Survived'] == 0 and blend2.loc[i, 'Sum'] > 2.5:
        blend2.loc[i, 'Survived'] = 1
    if blend2.loc[i, 'Survived'] == 1 and blend2.loc[i, 'Sum'] < 2.5:
        blend2.loc[i, 'Survived'] = 0

blend2.head(10)   


# In[11]:


blend2 = blend2.iloc[:, 0:2]
blend2.to_csv('kernel_blend_2nd.csv', index=False)

