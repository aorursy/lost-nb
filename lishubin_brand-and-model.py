#!/usr/bin/env python
# coding: utf-8



import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import KFold,StratifiedKFold
from sklearn.metrics import log_loss
import xgboost as xgb




phone = pd.read_csv('../input/phone_brand_device_model.csv',encoding='utf-8')
phone.head(3)




gatrain = pd.read_csv('../input/gender_age_train.csv')
gatest = pd.read_csv('../input/gender_age_test.csv')
gatrain.head(3)




dup = phone.groupby('device_id').size()
dup = dup[dup>1]
dup.shape





dup = phone.loc[phone.device_id.isin(dup.index)]
first = dup.groupby('device_id').first()
last = dup.groupby('device_id').last()




phone = phone.drop_duplicates('device_id', keep='first')




lebrand = LabelEncoder().fit(phone.phone_brand)
phone['brand'] = lebrand.transform(phone.phone_brand)
m = phone.phone_brand.str.cat(phone.device_model)
lemodel = LabelEncoder().fit(m)
phone['model'] = lemodel.transform(m)
phone['old_model'] = LabelEncoder().fit_transform(phone.device_model)




train = gatrain.merge(phone)
train.head()




train['y'] = LabelEncoder().fit_transform(train['group'])
train['gender'] = train['gender'].apply(lambda x: int(x=='M'))
train.head()






params = {
    "objective": "multi:softprob",
    "num_class": 12,
    "booster": "gbtree",
    "max_depth": 10,
    "eval_metric": "mlogloss",
    #"min_child_weight":50,
    "eta": 0.02,
    "silent": 1,
    "alpha": 3,
}




def encode_cat(Xtrain,Xtest):
    model_gender = Xtrain.groupby('model')['gender'].agg(['size','mean'])
    brand_gender = Xtrain.groupby('brand')['gender'].agg(['size','mean'])
    self.prob = (c.add(self.prior_weight*self.prior)).div(c.sum(axis=1)+self.prior_weight, axis=0)
    #model_gender.iloc[np.where(model_gender['size']<50)[0],1] = -1
    model_gender.columns = ['model_size','model_mean']
    #brand_gender.iloc[np.where(brand_gender['size']<50)[0],1] = -1
    brand_gender.columns = ['brand_size','brand_mean']
    Xtest['model_gender_mean'] = Xtest['model'].map(model_gender['model_mean'])
    Xtrain['model_gender_mean'] = Xtrain['model'].map(model_gender['model_mean'])
    Xtest['brand_gender_mean'] = Xtest['brand'].map(brand_gender['brand_mean'])
    Xtrain['brand_gender_mean'] = Xtrain['brand'].map(brand_gender['brand_mean'])

    
    model_age = Xtrain.groupby('model')['age'].agg(['size','mean'])
    brand_age = Xtrain.groupby('brand')['age'].agg(['size','mean'])
    
    model_age.iloc[np.where(model_age['size']<50)[0],1] = -1
    model_age.columns = ['model_age_size','model_age_mean']
    
    brand_age.iloc[np.where(brand_age['size']<50)[0],1] = -1
    brand_age.columns = ['brand_age_size','brand_age_mean']

    Xtest['model_age_mean'] = Xtest['model'].map(model_age['model_age_mean'])
    Xtrain['model_age_mean'] = Xtrain['model'].map(model_age['model_age_mean'])
    Xtest['brand_age_mean'] = Xtest['brand'].map(brand_age['brand_age_mean'])
    Xtrain['brand_age_mean'] = Xtrain['brand'].map(brand_age['brand_age_mean'])

    
    Xtrain.fillna(-1,inplace = True)
    Xtest.fillna(-1,inplace = True)
    #return Xtrain[['brand','model','model_age','brand_age']],Xtest[['brand','model','model_age','brand_age']]
    return Xtrain[['model_gender_mean','brand_gender_mean','model_age_mean','brand_age_mean']],Xtest[['model_gender_mean','brand_gender_mean','model_age_mean','brand_age_mean']]




def encode_cat(Xtrain,Xtest):
    #clf,prior = GenderAgeGroupProb(prior_weight=10).fit(Xtrain,'model')
    #model_group = Xtrain.groupby('model')['y'].agg(['size','mean'])
    #brand_group = Xtrain.groupby('brand')['y'].agg(['size','mean'])
    #model_group.iloc[np.where(model_group['size']<10)[0],1] = -1
    #model_group.columns = ['model_size','model_mean']
    #brand_group.iloc[np.where(brand_group['size']<10)[0],1] = -1
    #brand_group.columns = ['brand_size','brand_mean']
    #Xtest['model_group_mean'] = Xtest['model'].map(clf)
    #Xtrain['model_group_mean'] = Xtrain['model'].map(model_group)
    #Xtest['brand_group_mean'] = Xtest['brand'].map(brand_group['brand_mean'])
    #Xtrain['brand_group_mean'] = Xtrain['brand'].map(brand_group['brand_mean'])
    #Xtrain = Xtrain.merge(clf,right_index=True,left_on ='model')
    #Xtest = Xtest.merge(clf,right_index=True,left_on ='model')
    #Xtrain.fillna(0,inplace = True)
    #Xtest.loc[Xtest.iloc[:,0].isnull(),:] = prior

    #Xtest.fillna(0,inplace = True)
    #return Xtrain[['brand','model','model_age','brand_age']],Xtest[['brand','model','model_age','brand_age']]
    return Xtrain[['model']],Xtest[['model']]




brand_unique = train.groupby('brand')['model'].nunique()




train['brand_unique_model'] = train['brand'].map(brand_unique)




model_count = train.groupby('model')['model'].size()
model_count.head()




train['model_count'] = train['model'].map(model_count)
train.head()




Xtest.loc[Xtest.iloc[:,10].isnull(),10:] = prior




np.sum(Xtest.iloc[:,10].isnull())




class GenderAgeGroupProb(object):
    def __init__(self, prior_weight=10.):
        self.prior_weight = prior_weight
    
    def fit(self, df, by):
        self.by = by
        #self.label = 'pF_' + by
        self.prior = df['group'].value_counts().sort_index()/df.shape[0]
        # fit gender probs by grouping column
        c = df.groupby([by, 'group']).size().unstack().fillna(0)
        total = c.sum(axis=1)
        self.prob = (c.add(self.prior_weight*self.prior)).div(c.sum(axis=1)+self.prior_weight, axis=0)
        return self.prob,self.prior
    
    def predict_proba(self, df):
        pred = df[[self.by]].merge(self.prob, how='left', 
                                left_on=self.by, right_index=True).fillna(self.prior)[self.prob.columns]
        pred.loc[pred.iloc[:,0].isnull(),:] = self.prior
        return pred.values
    
def score(ptrain, by, prior_weight=10.):
    kf = KFold(ptrain.shape[0], n_folds=10, shuffle=True, random_state=0)
    pred = np.zeros((ptrain.shape[0],n_classes))
    for itrain, itest in kf:
        train = ptrain.iloc[itrain,:]
        test = ptrain.iloc[itest,:]
        ytrain, ytest = y[itrain], y[itest]
        clf = GenderAgeGroupProb(prior_weight=prior_weight).fit(train,by)
        pred[itest,:] = clf.predict_proba(test)
    return log_loss(y, pred)




train.head()




y = train['y']
kf = KFold(train.shape[0], n_folds=5, shuffle=True, random_state=1024)
pred = np.zeros((train.shape[0],12))
for itrain, itest in kf:
    Xtrain = train.ix[itrain,]
    Xtest = train.ix[itest,]
    ytrain, ytest = y[itrain], y[itest]
    Xtrain,Xtest = encode_cat(Xtrain,Xtest)
    print(Xtrain.shape)
    dtrain = xgb.DMatrix(Xtrain,label = ytrain)
    dvalid = xgb.DMatrix(Xtest,label = ytest)
    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, 600, evals=watchlist,
                early_stopping_rounds=25, verbose_eval=20)
    
    #gbm = xgb.train(params, dtrain, 600, evals=watchlist,
                #early_stopping_rounds=25, verbose_eval=20)
    temp_pred = gbm.predict(dvalid)
    pred[itest,:] = temp_pred
    print(log_loss(ytest, temp_pred))




Xtrain.head()




log_loss(train['y'].values.tolist(),pred)




log_loss(train['y'].values.tolist(),pred)




log_loss(train['y'].values.tolist(),pred)




df = Xtrain
df.head()




df[['gender']]==0




def fit(self, df, by):
    self.by = by
    self.label = 'pF_' + by
    self.prior = (df['gender']=='F').mean()
    # fit age groups
    prob = train.groupby(['gender','group']).size()
    pF = prob['F']/prob['F'].sum()
    pM = prob['M']/prob['M'].sum()
    self.agegroups = pd.concat((pF,pM),axis=0)
    # fit gender probs by grouping column
    c = (df[['gender']]=='F').groupby(df[by]).agg(['sum','count'])
    s = c[('gender','sum')]
    n = c[('gender','count')]
    prob = (s + self.prior*self.prior_weight) / (n + self.prior_weight)
    self.prob = pd.DataFrame(prob, columns=[self.label])
    return self

