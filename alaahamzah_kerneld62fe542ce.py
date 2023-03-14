#!/usr/bin/env python
# coding: utf-8



#! pip install lightgbm




#import pyximport; pyximport.install()
import gc
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import SGDRegressor
import lightgbm as lgb

NUM_BRANDS = 4000
NUM_CATEGORIES = 1000
NAME_MIN_DF = 10
MAX_FEATURES_ITEM_DESCRIPTION = 20000




def rmsle(y, y0):
     assert len(y) == len(y0)
     return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))
    
def split_cat(text):
    try: return text.split("/")
    except: return ("No Label", "No Label", "No Label")
    
def handle_missing_inplace(dataset):
    dataset['general_cat'].fillna(value='missing', inplace=True)
    dataset['subcat_1'].fillna(value='missing', inplace=True)
    dataset['subcat_2'].fillna(value='missing', inplace=True)
    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].fillna(value='missing', inplace=True)


def cutting(dataset):
    pop_brand = dataset['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['brand_name'].isin(pop_brand), 'brand_name'] = 'missing'
    pop_category1 = dataset['general_cat'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    pop_category2 = dataset['subcat_1'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    pop_category3 = dataset['subcat_2'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    dataset.loc[~dataset['general_cat'].isin(pop_category1), 'general_cat'] = 'missing'
    dataset.loc[~dataset['subcat_1'].isin(pop_category2), 'subcat_1'] = 'missing'
    dataset.loc[~dataset['subcat_2'].isin(pop_category3), 'subcat_2'] = 'missing'


def to_categorical(dataset):
    dataset['general_cat'] = dataset['general_cat'].astype('category')
    dataset['subcat_1'] = dataset['subcat_1'].astype('category')
    dataset['subcat_2'] = dataset['subcat_2'].astype('category')
    dataset['item_condition_id'] = dataset['item_condition_id'].astype('category')




start_time = time.time()

train = pd.read_table('../input/train.tsv', engine='c')
test = pd.read_table('../input/test_stg2.tsv', engine='c')




#train[(train.price < 1.0)].index




dftt = train[(train.price < 1.0)]
train = train.drop(train[(train.price < 1.0)].index)
del dftt['price']

nrow_train = train.shape[0] - dftt.shape[0]
nrow_test = train.shape[0] + dftt.shape[0]

nrow_train = train.shape[0] #-dftt.shape[0]
y = np.log1p(train["price"])
merge = pd.concat([train, dftt, test])
submission = test[['test_id']]




#merge.loc[merge["train_id"]==,:]
#merge[merge["price"].isnull()]
#submission.head()




merge['general_cat'], merge['subcat_1'], merge['subcat_2'] = zip(*merge['category_name'].apply(lambda x: split_cat(x)))
merge.drop('category_name', axis=1, inplace=True)




handle_missing_inplace(merge)




numbersList = [1, 2, 3]
strList = ['one', 'two']
numbersTuple = ('ONE', 'TWO', 'THREE', 'FOUR')

result = zip(numbersList, numbersTuple)

# Converting to set
resultSet = set(result)
print(resultSet)

result = zip(numbersList, strList, numbersTuple)

# Converting to set
resultSet = set(result)
print(resultSet)




cutting(merge)
#merge['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:5000]

#.loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]




to_categorical(merge)




cv = CountVectorizer(min_df=NAME_MIN_DF,ngram_range=(1, 2),
                         stop_words='english')
X_name = cv.fit_transform(merge['name'])




del train
del test
gc.collect()
cv = CountVectorizer()
X_category1 = cv.fit_transform(merge['general_cat'])
X_category2 = cv.fit_transform(merge['subcat_1'])
X_category3 = cv.fit_transform(merge['subcat_2'])




gc.collect()
tv = TfidfVectorizer(max_features=MAX_FEATURES_ITEM_DESCRIPTION,
                         ngram_range=(1, 3),
                         stop_words='english')
X_description = tv.fit_transform(merge['item_description'])




gc.collect()




X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']],
                                          sparse=True).values)




lb = LabelBinarizer(sparse_output=True)
X_brand = lb.fit_transform(merge['brand_name'])
sparse_merge = hstack((X_dummies, X_description, X_brand, X_category1, X_category2, X_category3, X_name)).tocsr()
X = sparse_merge[:nrow_train]
X_test = sparse_merge[nrow_test:]





gc.collect()
print(sparse_merge[1:10,:].todense())




model = Ridge(alpha=.5, copy_X=True, fit_intercept=True, max_iter=100,
     normalize=False, random_state=101, solver='auto', tol=0.01)
model.fit(X, y)




predsR = model.predict(X=X_test)




# train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size = 0.15, random_state = 144) 
# d_train = lgb.Dataset(train_X, label=train_y)
# d_valid = lgb.Dataset(valid_X, label=valid_y)
# watchlist = [d_train, d_valid]




# gc.collect()
# params = {
#         'learning_rate': 0.65,
#         'application': 'regression',
#         'max_depth': 3,
#         'num_leaves': 60,
#         'verbosity': -1,
#         'metric': 'RMSE',
#         'data_random_seed': 1,
#         'bagging_fraction': 0.5,
#         'nthread': 4
#     }

# params2 = {
#         'learning_rate': 0.85,
#         'application': 'regression',
#         'max_depth': 3,
#         'num_leaves': 140,
#         'verbosity': -1,
#         'metric': 'RMSE',
#         'data_random_seed': 2,
#         'bagging_fraction': 1,
#         'nthread': 4
#     }
# model = lgb.train(params, train_set=d_train, num_boost_round=8000, valid_sets=watchlist, \
# early_stopping_rounds=100, verbose_eval=1000) 
# predsL = model.predict(X_test)




submission['price'] = np.expm1(predsR)
submission.to_csv("submission_Ridge20000.csv", index=False)






