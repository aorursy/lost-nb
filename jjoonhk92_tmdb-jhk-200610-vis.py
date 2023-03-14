#!/usr/bin/env python
# coding: utf-8



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import ast
import json

from collections import Counter

import itertools
from itertools import zip_longest

import re
from wordcloud import WordCloud
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

import eli5

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('precision', '3')
pd.set_option('precision', 3)

import warnings
warnings.filterwarnings('ignore')




#データを読み取る
train = pd.read_csv('../input/tmdb-box-office-prediction/train.csv')
test = pd.read_csv('../input/tmdb-box-office-prediction/test.csv')




#train = pd.concat([train, test])




#columnsを確認し、除外する変数をdrop
train.drop(columns=['status','imdb_id','poster_path','original_title'], inplace = True)
print(train.columns)




train['log_revenue'] = np.log10(train['revenue'])




train['overview'].isna().sum()




train['overview'].fillna('none', inplace = True)
train['overview'].shape




train['overview'].head()




stop_words = stopwords.words('english')




def lower_text(text):
    return text.lower()

#記号の排除
def remove_punct(text):
    text = text.replace('-', ' ')  # - は単語の区切りとみなす
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)

def remove_stopwords(words, stopwords):#不要な単語を削除
    words = [word for word in words if word not in stopwords]
    return words




print(string.punctuation)




train['overview'] = train['overview'].apply(lambda x: lower_text(x))
train['overview'] = train['overview'].apply(lambda x: remove_punct(x))
#train['overview'] = train['overview'].apply(lambda x: remove_stopwords(x,stop_words))




train.info()




train['overview']




train['overview'] = train['overview'].apply(lambda x: ''.join(map(str,x)))




train['overview']




plt.figure(figsize = (12, 12))
text = ''.join(train['overview'].values)
wordcloud = WordCloud(max_font_size=None, background_color='white', width=1200, height=1000).generate(text)
plt.imshow(wordcloud)
plt.title('Top words in overview')
plt.axis("off")
plt.show()




vec_tfidf = TfidfVectorizer()

X = vec_tfidf.fit_transform(train['overview'])

print('Vocabulary size: {}'.format(len(vec_tfidf.vocabulary_)))
print('Vocabulary content: {}'.format(vec_tfidf.vocabulary_))




vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            analyzer='word',
            token_pattern=r'\w{1,}',
            ngram_range=(1, 2),
            min_df=5)

overview_text = vectorizer.fit_transform(train['overview'].fillna(''))
linreg = LinearRegression()
linreg.fit(overview_text, train['log_revenue'])




eli5.show_weights(linreg, vec = vectorizer, top=20, feature_filter=lambda x: x != '<BIAS>')




train['tagline'].fillna('none', inplace = True)
train['tagline'] = train['tagline'].apply(lambda x: lower_text(x))
train['tagline'] = train['tagline'].apply(lambda x: remove_punct(x))




train['tagline'] = train['tagline'].apply(lambda x: ''.join(map(str,x)))




plt.figure(figsize = (12,12))
text = ''.join(train['tagline'].values)
wordcloud = WordCloud(max_font_size=None, background_color='white', width=1200, height=1000).generate(text)
plt.imshow(wordcloud)
plt.title('Top words in overview')
plt.axis("off")
plt.show()




X = vec_tfidf.fit_transform(train['tagline'])

vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            analyzer='word',
            token_pattern=r'\w{1,}',
            ngram_range=(1, 2),
            min_df=5)

overview_text = vectorizer.fit_transform(train['tagline'].fillna(''))
linreg = LinearRegression()
linreg.fit(overview_text, train['log_revenue'])




eli5.show_weights(linreg, vec = vectorizer, top=20, feature_filter=lambda x: x != '<BIAS>')




def select_lgbm(x):
    params = {'learning_rate': [0.01,0.05,0.1 
              'max_depth': [8,16,32]
              'boosting': 'gbdt', 
              'objective': 'regression', 
              'metric': 'mse', 
              'is_training_metric': True, 
              'num_leaves': 144, 
              'feature_fraction': 0.9, 
              'bagging_fraction': 0.7, 
              'bagging_freq': 5, 
              }

lgb = LGBMClassifier()
grid = GridSearchCV(lgb, param_grid=params)

print('best_parameter: ', grid.best_params_)
print('rmse:', )




def gridcv_lgb(cv, data, train, test):

    best_models, best_params = [], []
    for folds_num, (train_idx, val_idx) in enumerate(cv.split(data)):
        print(f"\n------- Fold: ({folds_num + 1} / {cv.get_n_splits()}) ------\n")
        train, valid = data.iloc[train_idx], data.iloc[val_idx]
        X_train, X_valid = train.iloc[:train_days], train.iloc[test_days:]
        y_train, y_valid = train.iloc[train_days:,0], valid.iloc[:,0]

        train_set = lgb.Dataset(X_train, label=y_train)
        valid_set = lgb.Dataset(X_valid, label=y_valid)
        
        best_rmse = float('inf')
        for subsam in [0.3, 0.5, 0.7]:
            for lr in [0.1, 0.05, 0.01]:
                for num_iter in [2**9, 2**10, 2**11]:
                    for num_leaves in [2**5, 2**6, 2**7, 2**8]:
                        for m_depth in [2**2, 2**3, 2**4]:

                            set_param = {'boosting_type': 'gbdt',
                                         'objective': 'regression',
                                         'metric': 'rmse',
                                         'subsample': subsam,
                                         'learning_rate': lr,
                                         'max_bin': 50,
                                         'num_iterations': num_iter,
                                         'num_leaves': num_leaves,
                                         'max_depth': m_depth,
                                         'verbosity': -1}

                            model = lgb.train(params=set_param,
                                              train_set=train_set,
                                              valid_sets=[valid_set],
                                              early_stopping_rounds=10,
                                              verbose_eval=False)

                            pred = model.predict(X_valid)
                            rmse = np.sqrt(mse(y_valid.iloc[:30], pred[:30]))

                            if rmse < best_rmse:
                                best_rmse = rmse
                                best_model = model
                                best_param = set_param
                                print('RMSE: {:.5f}'.format(rmse))

        best_models.append(best_model)
        best_params.append(best_param)
    
    return best_models, best_params


def make_pred(model, data, train_days, pred_days, test):
    pred = model.predict(data.iloc[len(data)-train_days:len(data)])
    pred = pd.DataFrame(pred, columns=['prediction'])
    
    rmse = np.sqrt(mse(test.iloc[:pred_days], pred.iloc[:pred_days]))
    print('RMSE: {:.5f}'.format(rmse))
    
    # 評価用データと予測データを図で比較
    pred.index = test.index
    for_plot = pd.concat([test.iloc[:pred_days],
                          pred.iloc[:pred_days]],axis=1)

    for_plot.plot(figsize=(12,6));
    
    return pred, rmse






