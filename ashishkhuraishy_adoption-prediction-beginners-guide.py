#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import json
import os
import time
import random

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, cohen_kappa_score


# In[ ]:


train_df = pd.read_csv('../input/train/train.csv')
test_df = pd.read_csv('../input/test/test.csv')

breed_labels = pd.read_csv('../input/breed_labels.csv')
color_labels = pd.read_csv('../input/color_labels.csv')
state_labels = pd.read_csv('../input/state_labels.csv')

train_sen = os.listdir('../input/train_sentiment/')
train_meta = os.listdir('../input/train_metadata/')

test_sen = os.listdir('../input/test_sentiment/')
test_meta = os.listdir('../input/test_metadata/')

train = 'train'
test = 'test'

print(train_df.shape)
print(test_df.shape)
print(len(train_sen))
print(len(test_sen))
print(len(train_meta))
print(len(test_meta))


# In[ ]:


train_df.head()


# In[ ]:


breed_labels.head()


# In[ ]:


state_labels.head()


# In[ ]:


color_labels.head()


# In[ ]:


train_df['Type'].value_counts().plot.bar()


# In[ ]:


def sen_score(df, sen_source, test_train ):   
   sen = []
   for i in df['PetID']:
       a = i+'.json'
       if a in sen_source:
           x = '../input/%s_sentiment/%s' % (test_train, a)
           with open(x, 'r') as f:
                   sentiment = json.load(f)

           y = sentiment['documentSentiment']['score']
       else:
           y = 0

       sen.append(y)
   return sen
   
train_df['sen_score'] = sen_score(train_df, train_sen, train)
test_df['sen_score'] = sen_score(test_df, test_sen, test)


# In[ ]:


def sen_mag(df, sen_source, test_train ):   
   sen = []
   for i in df['PetID']:
       a = i+'.json'
       if a in sen_source:
           x = '../input/%s_sentiment/%s' % (test_train, a)
           with open(x, 'r') as f:
                   sentiment = json.load(f)

           y = sentiment['documentSentiment']['magnitude']
       else:
           y = 0

       sen.append(y)
   return sen
   
train_df['sen_mag'] = sen_mag(train_df, train_sen, train)
test_df['sen_mag'] = sen_mag(test_df, test_sen, test)


# In[ ]:


train_df['sen_score'].plot.hist()


# In[ ]:


train_df['sen_mag'].plot.hist()


# In[ ]:


train_df['PhotoAmt'].value_counts().plot.bar()


# In[ ]:


def meta_red(df, meta_source, train_test = 'train_test'):    
    meta = []
    for i in df['PetID']:
        a = i+'-1.json'
        if a in meta_source:
                x = '../input/%s_metadata/%s' % (train_test, a)
                with open(x, 'r') as f:
                    meta_r = json.load(f)
                    y = meta_r['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['red']
        else:
            y = 0

        meta.append(y)
    return meta
 
train_df['meta_red'] = meta_red(train_df, train_meta, train)
test_df['meta_red'] = meta_red(test_df, test_meta, test)


# In[ ]:


def meta_green(df, meta_source, train_test = 'train_test'):    
    meta = []
    for i in df['PetID']:
        a = i+'-1.json'
        if a in meta_source:
                x = '../input/%s_metadata/%s' % (train_test, a)
                with open(x, 'r') as f:
                    meta_r = json.load(f)
                    y = meta_r['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['green']
        else:
            y = 0

        meta.append(y)
    return meta

train_df['meta_green'] = meta_green(train_df, train_meta, train)
test_df['meta_green'] = meta_green(test_df, test_meta, test)


# In[ ]:


def meta_blue(df, meta_source, train_test = 'train_test'):    
    meta = []
    for i in df['PetID']:
        a = i+'-1.json'
        if a in meta_source:
                x = '../input/%s_metadata/%s' % (train_test, a)
                with open(x, 'r') as f:
                    meta_r = json.load(f)
                    y = meta_r['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['blue']
        else:
            y = 0

        meta.append(y)
    return meta

train_df['meta_blue'] = meta_blue(train_df, train_meta, train)
test_df['meta_blue'] = meta_blue(test_df, test_meta, test)


# In[ ]:


def meta_score(df, meta_source, train_test = 'train_test'):    
    meta = []
    for i in df['PetID']:
        a = i+'-1.json'
        if a in meta_source:
                x = '../input/%s_metadata/%s' % (train_test, a)
                with open(x, 'r') as f:
                    meta_r = json.load(f)
                    y = meta_r['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']
        else:
            y = 0

        meta.append(y)
    return meta

train_df['meta_score'] = meta_score(train_df, train_meta, train)
test_df['meta_score'] = meta_score(test_df, test_meta, test)


# In[ ]:


def meta_pixelfraction(df, meta_source, train_test = 'train_test'):    
    meta = []
    for i in df['PetID']:
        a = i+'-1.json'
        if a in meta_source:
                x = '../input/%s_metadata/%s' % (train_test, a)
                with open(x, 'r') as f:
                    meta_r = json.load(f)
                    y = meta_r['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']
        else:
            y = 0

        meta.append(y)
    return meta

train_df['meta_pixelfraction'] = meta_pixelfraction(train_df, train_meta, train)
test_df['meta_pixelfraction'] = meta_pixelfraction(test_df, test_meta, test)


# In[ ]:


def meta_ver_x(df, meta_source, train_test = 'train_test'):    
    meta = []
    for i in df['PetID']:
        a = i+'-1.json'
        if a in meta_source:
                x = '../input/%s_metadata/%s' % (train_test, a)
                with open(x, 'r') as f:
                    meta_r = json.load(f)
                    y = meta_r['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][1]['x']
        else:
            y = 0

        meta.append(y)
    return meta

train_df['meta_ver_x'] = meta_ver_x(train_df, train_meta, train)
test_df['meta_ver_x'] = meta_ver_x(test_df, test_meta, test)


# In[ ]:


def meta_ver_y(df, meta_source, train_test = 'train_test'):    
    meta = []
    for i in df['PetID']:
        a = i+'-1.json'
        if a in meta_source:
                x = '../input/%s_metadata/%s' % (train_test, a)
                with open(x, 'r') as f:
                    meta_r = json.load(f)
                    y = meta_r['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][3]['y']
        else:
            y = 0

        meta.append(y)
    return meta

train_df['meta_ver_y'] = meta_ver_y(train_df, train_meta, train)
test_df['meta_ver_y'] = meta_ver_y(test_df, test_meta, test)


# In[ ]:


def meta_conf(df, meta_source, train_test = 'train_test'):    
    meta = []
    for i in df['PetID']:
        a = i+'-1.json'
        if a in meta_source:
                x = '../input/%s_metadata/%s' % (train_test, a)
                with open(x, 'r') as f:
                    meta_r = json.load(f)
                    y = meta_r['cropHintsAnnotation']['cropHints'][0]['confidence']
        else:
            y = 0

        meta.append(y)
    return meta

train_df['meta_conf'] = meta_conf(train_df, train_meta, train)
test_df['meta_conf'] = meta_conf(test_df, test_meta, test)


# In[ ]:


train_df.head()


# In[ ]:


train_df['Age_in_yrs'] = [i//12 for i in train_df['Age'] ]
train_df['Age_in_yrs'].value_counts().plot.bar()


# In[ ]:


test_df['Age_in_yrs'] = [i//12 for i in test_df['Age']]
test_df['Age_in_yrs'].value_counts().plot.bar()


# In[ ]:


train_df.head()


# In[ ]:


def Cross(df):
    cross = [1 if df['Breed1'][i] and df['Breed2'][i] != 0 else 0 for i in range(len(df['Breed1']))]
    df['Cross_Y/N'] = cross
    return df
train_df = Cross(train_df)
test_df  = Cross(test_df)

 


# In[ ]:


def cross(df):    
    cross = []
    a = 0
    for i in df['Breed1']:
        cross.append(i*(df['Breed2'][a]+1))
        a += 1
    return cross

train_df['Cross_BreedScore'] = cross(train_df)
test_df['Cross_BreedScore']  = cross(test_df)


# In[ ]:


train_df.head()


# In[ ]:


train_df.isnull().sum().plot.bar()


# In[ ]:


drop = ['Name', 'Age', 'Breed1','Breed2', 'RescuerID', 'PetID', 'Description']
train = train_df.drop(drop, axis = 1)
test  = test_df.drop(drop, axis = 1)

X = train.drop(['AdoptionSpeed'], axis = 1)
y = train.AdoptionSpeed


# In[ ]:


y.value_counts().plot.bar()


# In[ ]:


test.head()


# In[ ]:


train.head()


# In[ ]:


def train_model(clf,X, y, val_x, val_y):
    model = clf
    #t1 = time()
    model.fit(X, y)
    #t2 = round((t1-time()), 3)
    #t3 = time()
    pred = model.predict(val_x)
    #t4 = round((t3-time()), 3)
    score = accuracy_score(pred, val_y)
    ck_score = cohen_kappa_score(pred, val_y)
    cross_score = cross_val_score(clf, X, y, scoring='accuracy', cv = 10)
    
    print("Model : %s" % clf)
    #print("Training Time : %d" % t2)
    #print("Prediction Time : %d" % t4)
    print("Accuracy : %s" % score)
    print("Cohem_Kappa : %s" % ck_score)
    print("Cross_Val_Score : %s" % cross_score.mean())
    


# In[ ]:


#Train-Test Split

train_X , val_X, train_y, val_y = train_test_split(X, y, random_state = 2, test_size = 0.2)


# In[ ]:


clf1 = RandomForestClassifier()
clf2 = AdaBoostClassifier()
clf3 = xgb.XGBClassifier()


# In[ ]:


train_model(clf1, train_X, train_y, val_X, val_y)


# In[ ]:


train_model(clf2, train_X, train_y, val_X, val_y)


# In[ ]:


train_model(clf3, train_X, train_y, val_X, val_y)


# In[ ]:


# You can try different combinations and check each scores until you are satisfied.
model = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,    
       colsample_bytree=1, gamma=0.2, learning_rate=0.1
                          , max_delta_step=0,
       max_depth=5, min_child_weight=1, missing=None, n_estimators=140,
       n_jobs=1, nthread=4, objective='multi:softprob', random_state=42,
       reg_alpha=0.001, reg_lambda=1, scale_pos_weight=1, seed=27,
       silent=True, subsample=1)                                                  


# In[ ]:


model.fit(train_X, train_y)


# In[ ]:


pred = model.predict(val_X)
score = cohen_kappa_score(pred, val_y)
acc = accuracy_score(pred, val_y) 

print("Cohen Kappa : %s" % score) # 0.2175871739419205
print("Accuracy : %s" % acc) # 0.4171390463487829


# In[ ]:


n = random.randint(0, 100) # just for checking how my model works
print(list(val_y)[n])
print(list(pred)[n])


# In[ ]:


cross_val_score(model, X, y, scoring = 'accuracy', cv= 10).mean() #0.39845585964987384


# In[ ]:


model.fit(X, y)


# In[ ]:


result = model.predict(test)
result


# In[ ]:


submission = pd.DataFrame({'PetID' : test_df.PetID, 'AdoptionSpeed' : result })
submission.to_csv('submission.csv', index=False)


# In[ ]:


submission['AdoptionSpeed'].value_counts().plot.bar()

