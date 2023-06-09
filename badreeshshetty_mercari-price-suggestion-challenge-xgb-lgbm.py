#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import scipy as sp
import gc
import re
import nltk
from datetime import datetime
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from scipy.sparse import csr_matrix,hstack
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')




import os 
print(os.listdir("../input/"))




df_train = pd.read_csv('../input/mercari-price-suggestion-challenge/train.tsv', sep='\t', encoding='utf-8')
df_test = pd.read_csv('../input/mercari-price-suggestion-challenge/test_stg2.tsv', sep='\t', encoding='utf-8')




df_train.head()




df_test.head()




print('Train shape:{}\nTest shape:{}'.format(df_train.shape, df_test.shape))




#Information on Train Dataset 
df_train.info()




#Information on Test Dataset 
df_test.info()




#Evaluation Metric
def rmsle(predicted, actual):
    assert len(predicted) == len(actual)
    return np.sqrt(np.mean(np.power(np.log1p(predicted)-np.log1p(actual), 2)))




#Check for Null Values in Train Dataset
null_columns=df_train.columns[df_train.isnull().any()]
df_train[null_columns].isnull().sum()




#Check for Null Values in Test Dataset
null_columns=df_test.columns[df_test.isnull().any()]
df_test[null_columns].isnull().sum()




#Filling Missing Values
def fill_missing(data):
    data["category_name"].fillna("Missing",inplace=True)
    data["brand_name"].fillna("Missing",inplace=True)
    data["item_description"].fillna("Missing",inplace=True)




fill_missing(df_train)




fill_missing(df_test)




df_train.isnull().any().sum()




df_test.isnull().any().sum()




#Spliiting into general_category,cat_1,cat_2
def split_cat(s):
    try:
        return s.split('/')[0],s.split('/')[1],s.split('/')[2],
    except:
        return ['No','No','No'] 




df_train[['gen_cat','cat1','cat2']] = pd.DataFrame(df_train.category_name.apply(split_cat).tolist(),
                                   columns = ['gen_cat','cat1','cat2'])
df_test[['gen_cat','cat1','cat2']] = pd.DataFrame(df_test.category_name.apply(split_cat).tolist(),
                                   columns = ['gen_cat','cat1','cat2'])




df_train.drop("category_name",axis=1,inplace=True)
df_test.drop("category_name",axis=1,inplace=True)




train_rows=df_train.shape[0]
y_train=df_train["price"]
test_id=df_test["test_id"]
df_all= pd.concat([df_train,df_test])




df_all.drop(["train_id","test_id"], axis=1,inplace=True)




stop_words = set(stopwords.words('english'))
def clean_text(text):
    text=text.lower()
    text = re.sub('[^\w\s]', '', text)
    x = [word for word in text.split() if word not in stop_words]
    return " ".join(x)




df_all['item_description_clean']=df_all['item_description'].apply(clean_text)




def item_desc_len(item):
    item_l=[i for i in item.split()]
    return(len(item_l))




df_all['description_len']=df_all['item_description_clean'].apply(item_desc_len)




df_all.drop("item_description",axis=1,inplace=True)




df_all.head()




count_vect_n=CountVectorizer()
X_name=count_vect_n.fit_transform(df_all["name"])
X_name.shape




count_vect_cn=CountVectorizer()
X_gen_cat=count_vect_cn.fit_transform(df_all["gen_cat"])
X_cat1=count_vect_cn.fit_transform(df_all["cat1"])
X_cat2=count_vect_cn.fit_transform(df_all["cat2"])
X_gen_cat.shape




tfidf_vect=TfidfVectorizer(ngram_range=(1,2),stop_words="english")
X_descript=tfidf_vect.fit_transform(df_all["item_description_clean"])
X_descript.shape




lb=LabelBinarizer(sparse_output=True)
X_brand=lb.fit_transform(df_all["brand_name"])
X_brand.shape




X_item_ship=pd.get_dummies(df_all[["item_condition_id","shipping","description_len"]],sparse=True)
X_item_ship.shape




X_dummies=csr_matrix(X_item_ship)
X_dummies.shape




X=hstack((X_name,X_gen_cat,X_cat1,X_cat2,X_descript,X_brand,X_item_ship,X_dummies)).tocsr()




X_train=X[:train_rows]
X_test=X[train_rows:]




print("There are %d unique brand names" % df_all['brand_name'].nunique())




all_brand_name_10=df_all["brand_name"].value_counts().head(10)
all_brand_name_10=all_brand_name_10[1:]




plt.figure(figsize=(20, 15))
sns.barplot(all_brand_name_10.index.values.astype('str'), all_brand_name_10.values, alpha=0.8);
plt.xticks(rotation=70,fontsize=15)
plt.yticks(fontsize=15);




print("There are %d unique General names" % df_all['gen_cat'].nunique())




all_cat_10=df_all["gen_cat"].value_counts().head(11)
all_cat_10




plt.figure(figsize=(20, 15))
sns.barplot(all_cat_10.index.values.astype('str'), all_cat_10.values, alpha=0.8);
plt.xticks(rotation=70,fontsize=15)
plt.yticks(fontsize=15);




print("There are %d unique Category 1 Names"% df_all['cat1'].nunique())




all_cat1_10=df_all["cat1"].value_counts().head(10)
all_cat1_10




plt.figure(figsize=(20, 15))
sns.barplot(all_cat1_10.index.values.astype('str'), all_cat1_10.values, alpha=0.8);
plt.xticks(rotation=70,fontsize=15)
plt.yticks(fontsize=15);




print("There are %d unique Category 2 Names"% df_all['cat2'].nunique())




all_cat2_10=df_all["cat2"].value_counts().head(10)
all_cat2_10




plt.figure(figsize=(20, 15))
sns.barplot(all_cat2_10.index.values.astype('str'), all_cat2_10.values, alpha=0.8);
plt.xticks(rotation=70,fontsize=15)
plt.yticks(fontsize=15);




plt.figure(figsize=(20, 15))
plt.hist(df_train['price'],bins=50,range=[0,250],edgecolor='white',label='price')
plt.title('Train Price Distribution',fontsize=15)
plt.xlabel('Price',fontsize=15)
plt.ylabel('Items',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15);




plt.figure(figsize=(20, 15))
plt.hist(np.log1p(df_train['price']),bins=50,edgecolor='white',label='price')
plt.title('Log(Train Price) Distribution ',fontsize=15)
plt.xlabel('Price',fontsize=15)
plt.ylabel('Items',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15);




df_train['shipping'].unique()




plt.figure(figsize=(20, 15))
plt.hist(df_train[df_train['shipping']==1]['price'], bins=50, density=True, range=[0,250],
         alpha=0.5, label='price when shipping==1')
plt.hist(df_train[df_train['shipping']==0]['price'], bins=50, density=True, range=[0,250],
         alpha=0.5, label='price when shipping==0')
plt.title('Train Price over Shipping type Distribution', fontsize=15)
plt.xlabel('Price', fontsize=15)
plt.ylabel('Normalized Items', fontsize=15)
plt.legend(fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15);




del df_train
del df_test
del null_columns
del X
del X_name
del X_gen_cat 
del X_cat1
del X_cat2 
del X_descript
del X_brand
del X_item_ship
del X_dummies
gc.collect()




start=datetime.now()
model = Ridge(alpha=.05, copy_X=True, fit_intercept=True, max_iter=100,
      normalize=False, random_state=42, solver='auto', tol=0.001)
model.fit(X_train, y_train)
print("Time taken to run this cell :", datetime.now() - start)




predsR_train=model.predict(X_train)




predsR = model.predict(X_test)




train_X, valid_X, train_y, valid_y = train_test_split(X_train, y_train, test_size = 0.15, random_state = 42) 
d_train = lgb.Dataset(train_X, label=train_y)
d_valid = lgb.Dataset(valid_X, label=valid_y)
watchlist = [d_train, d_valid]




params1 = {
        'learning_rate': 0.5,
        'application': 'regression',
        'max_depth': 3,
        'num_leaves': 60,
        'verbosity': -1,
        'metric': 'RMSE',
        'data_random_seed': 1,
        'bagging_fraction': 0.5,
        'nthread': 4,
        'max_bin':8192   
    }




start=datetime.now()
model = lgb.train(params1, train_set=d_train, num_boost_round=8000, valid_sets=watchlist, early_stopping_rounds=250, verbose_eval=1000) 
print("Time taken to run this cell :", datetime.now() - start)




predsL1_train = model.predict(X_train)




predsL1 = model.predict(X_test)




train_X2, valid_X2, train_y2, valid_y2 = train_test_split(X_train, y_train, test_size = 0.10, random_state = 42) 
d_train2 = lgb.Dataset(train_X2, label=train_y2)
d_valid2 = lgb.Dataset(valid_X2, label=valid_y2)
watchlist2 = [d_train2, d_valid2]




params2 = {
       'learning_rate': 1,
       'application': 'regression',
       'max_depth': 3,
       'num_leaves': 140,
       'verbosity': -1,
       'metric': 'RMSE',
       'data_random_seed': 2,
       'bagging_fraction': 1,
       'nthread': 4,
       'max_bin':8192
   }




start=datetime.now()
model = lgb.train(params2, train_set=d_train2, num_boost_round=8000, valid_sets=watchlist2,     early_stopping_rounds=250, verbose_eval=1000) 
print("Time taken to run this cell :", datetime.now() - start)




predsL2_train = model.predict(X_train)




predsL2 = model.predict(X_test)




preds_train = predsR_train*0.35 + predsL1_train*0.35 + predsL2_train*0.3




preds = predsR*0.35 + predsL1*0.35 + predsL2*0.3




rmsle_n=rmsle(preds_train,y_train)
rmsle_n




#start=datetime.now()
#lgb_g=lgb.LGBMRegressor(learning_rate=0.005,max_depth=40,n_estimators=4000,num_leaves=200)
#lgb_g.fit(X_train,y_train)
#print("Time taken to run this cell :", datetime.now() - start)




#import pickle
#lgb_g = pickle.load(open("../input/mercarilgbstg2/lgb_g_2.pickle.dat", "rb"))




#preds_train_L=lgb_g.predict(X_train)




#preds_test_L=lgb_g.predict(X_test)




submission = pd.DataFrame({"Test_id": test_id,"Price": preds})




submission.head()




submission.tail()




submission.to_csv('submission.csv', index=False)































