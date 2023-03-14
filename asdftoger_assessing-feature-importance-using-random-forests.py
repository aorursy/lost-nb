#!/usr/bin/env python
# coding: utf-8



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from rfpimp import *
import os
print(os.listdir("../input"))

#%auto reload_ext
#http://explained.ai/rf-importance/index.html

from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

from fastai.structured import *
from fastai.imports import *
from rfpimp import *
from plotnine import *
from pdpbox import pdp

SEED=100
np.random.seed(SEED)




df_raw=pd.read_json('../input/train.json')
display(df_raw.head())
display(df_raw.tail())




df_raw.interest_level=df_raw.interest_level.map({'low':1,'medium':2,'high':3})
cols=['bathrooms', 'bedrooms', 'longitude', 'latitude', 'price','interest_level']
df=df_raw.loc[:,cols]
df['random']=np.random.rand(len(df))#Add a uniform random distributed column to assess feature importances  
df.head()




def filter_train_test_split(df,X_cols,y_cols,train_size=0.9):
    X=df.loc[:,X_cols]
    y=df.loc[:,y_cols]
    X_train,X_valid,y_train,y_valid=train_test_split(X,y,train_size=train_size,random_state=SEED)
    return X_train,X_valid,y_train.values.ravel(),y_valid.values.ravel()
def sklearn_importances(X_train,m):
    I=pd.DataFrame(data={'Feature': X_train.columns, 'Importance': m.feature_importances_})
    I=I.set_index('Feature')
    I.sort_values('Importance',ascending=False,inplace=True)
    return I
def importance_comparison_plot(gini_imp,perm_imp,drop_imp):
    fig,axarr=plt.subplots(1,3,figsize=(17,5))
    gini_imp.plot(kind='barh',color='#D9E6F5',width=0.9,ax=axarr[0],title='Gini importances',legend=False)
    perm_imp.plot(kind='barh',color='#D9E6F5',width=0.9,ax=axarr[1],title='Permutation importances',legend=False)
    drop_imp.plot(kind='barh',color='#D9E6F5',width=0.9,ax=axarr[2],title='Drop column importances',legend=False)
    plt.tight_layout()




X_train,X_valid,y_train,y_valid=filter_train_test_split(df,['bathrooms', 'bedrooms', 'longitude', 'latitude','random'],['price'])
m=RandomForestRegressor(n_estimators=100,n_jobs=-1)
_ = get_ipython().run_line_magic('time', 'm.fit(X_train,y_train)')
gini_imp=sklearn_importances(X_train,m)
perm_imp=importances(m,X_valid,y_valid,n_samples=-1)
drop_imp=dropcol_importances(m,X_train,y_train,X_valid,y_valid)




importance_comparison_plot(gini_imp,perm_imp,drop_imp)




X_train,X_valid,y_train,y_valid=filter_train_test_split(df,['bathrooms', 'bedrooms', 'longitude', 'latitude','random','price'],['interest_level'])
m=RandomForestRegressor(n_estimators=100,n_jobs=-1)
_ = get_ipython().run_line_magic('time', 'm.fit(X_train,y_train)')
gini_imp=sklearn_importances(X_train,m)
perm_imp=importances(m,X_valid,y_valid,n_samples=-1)
drop_imp=dropcol_importances(m,X_train,y_train,X_valid,y_valid)




importance_comparison_plot(gini_imp,perm_imp,drop_imp)




df['neg_price']=-df.price
df['mean_bin']=np.where(df.price > df.price.mean(), 1, 0)
df['median_bin']=np.where(df.price > df.price.median(), 1, 0)
percentiles=np.linspace(0,100,6)#Create 5 different percentile ranges
price_percentiles=np.percentile(df.price,percentiles)
df['percentiles_bin']=df.price
for i in range(len(price_percentiles)-1):
    df['percentiles_bin']=np.where((df['percentiles_bin']>=price_percentiles[i])&(df['percentiles_bin']<=price_percentiles[i+1]),i,df['percentiles_bin'])
df['addmean_price']=df.price+df.price.mean()
df['normalised_price']=(df.price-df.price.mean())/df.price.std()
#Add constant valued column
df['mean_price']=df.price.mean()
df['median_price']=df.price.median()




plot_corr_heatmap(df,figsize=(10,10))
plt.tight_layout()




X_train,X_valid,y_train,y_valid=filter_train_test_split(df,['bathrooms', 'bedrooms', 'longitude', 'latitude','random','price','neg_price'],['interest_level'])
m=RandomForestRegressor(n_estimators=100,n_jobs=-1)
_ = get_ipython().run_line_magic('time', 'm.fit(X_train,y_train)')
gini_imp=sklearn_importances(X_train,m)
perm_imp=importances(m,X_valid,y_valid,n_samples=-1)
drop_imp=dropcol_importances(m,X_train,y_train,X_valid,y_valid)




importance_comparison_plot(gini_imp,perm_imp,drop_imp)




X_train,X_valid,y_train,y_valid=filter_train_test_split(df,['bathrooms', 'bedrooms', 'longitude', 'latitude','random','price','mean_bin'],['interest_level'])
m=RandomForestRegressor(n_estimators=100,n_jobs=-1)
_ = get_ipython().run_line_magic('time', 'm.fit(X_train,y_train)')
gini_imp=sklearn_importances(X_train,m)
perm_imp=importances(m,X_valid,y_valid,n_samples=-1)
drop_imp=dropcol_importances(m,X_train,y_train,X_valid,y_valid)




importance_comparison_plot(gini_imp,perm_imp,drop_imp)




X_train,X_valid,y_train,y_valid=filter_train_test_split(df,['bathrooms', 'bedrooms', 'longitude', 'latitude','random','price','median_bin'],['interest_level'])
#Binarise price
m=RandomForestRegressor(n_estimators=100,n_jobs=-1)
_ = get_ipython().run_line_magic('time', 'm.fit(X_train,y_train)')
gini_imp=sklearn_importances(X_train,m)
perm_imp=importances(m,X_valid,y_valid,n_samples=-1)
drop_imp=dropcol_importances(m,X_train,y_train,X_valid,y_valid)




importance_comparison_plot(gini_imp,perm_imp,drop_imp)




X_train,X_valid,y_train,y_valid=filter_train_test_split(df,['bathrooms', 'bedrooms', 'longitude', 'latitude','random','price','percentiles_bin'],['interest_level'])
#Binarise price
m=RandomForestRegressor(n_estimators=100,n_jobs=-1)
_ = get_ipython().run_line_magic('time', 'm.fit(X_train,y_train)')
gini_imp=sklearn_importances(X_train,m)
perm_imp=importances(m,X_valid,y_valid,n_samples=-1)
drop_imp=dropcol_importances(m,X_train,y_train,X_valid,y_valid)




importance_comparison_plot(gini_imp,perm_imp,drop_imp)




X_train,X_valid,y_train,y_valid=filter_train_test_split(df,['bathrooms', 'bedrooms', 'longitude', 'latitude','random','price','addmean_price'],['interest_level'])
#Binarise price
m=RandomForestRegressor(n_estimators=100,n_jobs=-1)
_ = get_ipython().run_line_magic('time', 'm.fit(X_train,y_train)')
gini_imp=sklearn_importances(X_train,m)
perm_imp=importances(m,X_valid,y_valid,n_samples=-1)
drop_imp=dropcol_importances(m,X_train,y_train,X_valid,y_valid)




importance_comparison_plot(gini_imp,perm_imp,drop_imp)




X_train,X_valid,y_train,y_valid=filter_train_test_split(df,['bathrooms', 'bedrooms', 'longitude', 'latitude','random','price','normalised_price'],['interest_level'])
#Binarise price
m=RandomForestRegressor(n_estimators=100,n_jobs=-1)
_ = get_ipython().run_line_magic('time', 'm.fit(X_train,y_train)')
gini_imp=sklearn_importances(X_train,m)
perm_imp=importances(m,X_valid,y_valid,n_samples=-1)
drop_imp=dropcol_importances(m,X_train,y_train,X_valid,y_valid)




importance_comparison_plot(gini_imp,perm_imp,drop_imp)




X_train,X_valid,y_train,y_valid=filter_train_test_split(df,['bathrooms', 'bedrooms', 'longitude', 'latitude','random','price','mean_price'],['interest_level'])
#Binarise price
m=RandomForestRegressor(n_estimators=100,n_jobs=-1)
_ = get_ipython().run_line_magic('time', 'm.fit(X_train,y_train)')
gini_imp=sklearn_importances(X_train,m)
perm_imp=importances(m,X_valid,y_valid,n_samples=-1)
drop_imp=dropcol_importances(m,X_train,y_train,X_valid,y_valid)




importance_comparison_plot(gini_imp,perm_imp,drop_imp)




X_train,X_valid,y_train,y_valid=filter_train_test_split(df,['bathrooms', 'bedrooms', 'longitude', 'latitude','random','price','median_price'],['interest_level'])
#Binarise price
m=RandomForestRegressor(n_estimators=100,n_jobs=-1)
_ = get_ipython().run_line_magic('time', 'm.fit(X_train,y_train)')
gini_imp=sklearn_importances(X_train,m)
perm_imp=importances(m,X_valid,y_valid,n_samples=-1)
drop_imp=dropcol_importances(m,X_train,y_train,X_valid,y_valid)




importance_comparison_plot(gini_imp,perm_imp,drop_imp)




X_train,X_valid,y_train,y_valid=filter_train_test_split(df,['bathrooms', 'bedrooms', 'longitude', 'latitude','random','price','median_price','mean_price','percentiles_bin','mean_bin','median_bin'],['interest_level'])
#Binarise price
m=RandomForestRegressor(n_estimators=100,n_jobs=-1)
_ = get_ipython().run_line_magic('time', 'm.fit(X_train,y_train)')
gini_imp=sklearn_importances(X_train,m)
perm_imp=importances(m,X_valid,y_valid,features=['bathrooms', 'bedrooms', 'longitude', 'latitude','random','price',['median_price','mean_price','percentiles_bin','mean_bin','median_bin']],n_samples=-1)
drop_imp=dropcol_importances(m,X_train,y_train,X_valid,y_valid)




importance_comparison_plot(gini_imp,perm_imp,drop_imp)

