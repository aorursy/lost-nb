#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import missingno as msno
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns
import math

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

df_train = pd.read_csv('../input/train.tsv', sep='\t')
df_test = pd.read_csv('../input/test.tsv', sep='\t')
sample = pd.read_csv('../input/sample_submission.csv')
# Any results you write to the current directory are saved as output.




print(df_train.shape)
df_train.head()




print(df_test.shape)
df_test.head()




print(df_train.dtypes.unique())




pp = pd.value_counts(df_train.dtypes)
pp.plot.bar()
plt.show()




cols_missing_val_train = df_train.columns[df_train.isnull().any()].tolist()
print(cols_missing_val_train)
print('\n')

cols_missing_val_test = df_test.columns[df_test.isnull().any()].tolist()
print(cols_missing_val_test)




# Train data
msno.bar(df_train[cols_missing_val_train],figsize=(20,8),color="#19455e",fontsize=18,labels=True,)




# Test data
msno.bar(df_test[cols_missing_val_test],figsize=(20,8),color="#50085e",fontsize=18,labels=True,)




# Train data
msno.matrix(df_train[cols_missing_val_train],width_ratios=(10,1),            figsize=(20,8),color=(0.5,0.5,0.2),fontsize=18,sparkline=True,labels=True)




#--- Test dataframe ---
msno.matrix(df_test[cols_missing_val_test],width_ratios=(10,1),            figsize=(20,8),color=(0.9,0.2,0.2),fontsize=18,sparkline=True,labels=True)




df_train['category_name'] = df_train['category_name'].fillna("Unknown_category")
df_test['category_name'] = df_test['category_name'].fillna("Unknown_category")




print(df_train.shape)
print(df_test.shape)




df_train['brand_name'] = df_train['brand_name'].fillna("Unknown")
df_test['brand_name'] = df_test['brand_name'].fillna("Unknown")




df_train = df_train[pd.notnull(df_train['item_description'])]




print(df_train.shape)




df_test['price'] = -1
df = pd.concat([df_train, df_test])

print(df.shape)
df.head()




a = 'This / is / me'
b = a.split('/')
print(b)




ha = "some-sample-filename-to-split"
"-".join(ha.split("-", 3)[:3])




df['category_name'].head(15)




df.shape




'''def my_test2(row):
     #return row['a'] % row['c']
    return row.count('/') + 1

df['category_num'] = df.apply(my_test2, axis = 1)'''

#for i in df['category_name']:
#    df['category_num'] = i.count('/') + 1




df.head()




primary_category_list = []
for i in df['category_name']:
    b = i.split('/')
    primary_category_list.append(b[0])
    
    primary_category_list = set(primary_category_list)
    primary_category_list = list(primary_category_list)
    
len(primary_category_list)    




primary_category_list




from wordcloud import WordCloud, STOPWORDS

wordcloud = WordCloud(
                          background_color='white',
                          #stopwords=stopwords,
                          max_words=200,
                          max_font_size=30,
                          min_font_size=15,
                          random_state=42
                         ).generate(str(primary_category_list))

print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
#fig.savefig("primary_categories.png", dpi=900)




dual_category_list = []
for i in df['category_name']:
    b = "/".join(i.split("/", 2)[:2])
    dual_category_list.append(b)
    
    dual_category_list = set(dual_category_list)
    dual_category_list = list(dual_category_list)
    
len(dual_category_list) 




dual_category_list




wordcloud1 = WordCloud(
                          background_color='white',
                          #stopwords=stopwords,
                          max_words=200,
                          max_font_size=30,
                          min_font_size=15,
                          random_state=42
                         ).generate(str(dual_category_list))

print(wordcloud1)
fig = plt.figure(1)
plt.imshow(wordcloud1)
plt.axis('off')
plt.show()
#fig.savefig("dual_categories.png", dpi=900)




triple_category_list = []
for i in df['category_name']:
    b = "/".join(i.split("/", 3)[:3])
    triple_category_list.append(b)
    
    triple_category_list = set(triple_category_list)
    triple_category_list = list(triple_category_list)
    
len(triple_category_list) 




triple_category_list




category_name_list = []
for i in df['category_name']:
    b = i.split('/')
    for j in b:
        category_name_list.append(j)
        
    category_name_list = set(category_name_list)
    category_name_list = list(category_name_list)
    
len(category_name_list)    




category_name_list




'''for col in category_name_list:
    df[col] = np.int(0)
    df[col] = df[col].astype(np.int8)'''




'''print(df.shape)
df.head()'''




# df.drop('category_name', axis=1, inplace=True)




len(df)




df.name.nunique()




df.drop('item_description', axis=1, inplace=True)
#df.drop('name', axis=1, inplace=True)




df_cat = pd.DataFrame(df.category_name.str.split('/',2).tolist(),
                                   columns = ['category_1','category_2', 'category_3'])
df['category_1'] = df_cat['category_1']
df['category_2'] = df_cat['category_2']
df['category_3'] = df_cat['category_3']




df.drop('category_name', axis=1, inplace=True)




df.head(15)




df.shape




df['name'], _ = pd.factorize(df['name'])
#df['brand_name'] = df['brand_name'].astype(np.int16)
df['name'].nunique()




df['brand_name'], _ = pd.factorize(df['brand_name'])
df['brand_name'] = df['brand_name'].astype(np.int16)
df['brand_name'].unique()




#for r in df.category_name:
#    df['cat_num'] = r.count('/') + 1

for r in range(0, len(df)):
    df['cat_num'] = df['category_name'].iloc[r].count('/') + 1
  
#df['category_name'][100]

#df['category_name'].iloc[100]

#df.type()
#df.category_name.iter

df.head()




sample = df
cat_1 = sample.category_1.unique()
cat_2 = sample.category_2.unique()
cat_3 = sample.category_3.unique()

for i in sample.category_1:
    for c in cat_1:
        x = 

sample.head()

sample['category_1'].mean()

df_cat = pd.DataFrame(df.category_name.str.split('/',2).tolist(), columns = ['c1','c2', 'c3']))
df_cat.head()




df['category_1'], _ = pd.factorize(df['category_1'])
df['category_1'] = df['category_1'].astype(np.int8)




df['category_2'], _ = pd.factorize(df['category_2'])
df['category_2'] = df['category_2'].astype(np.int8)




df['category_3'], _ = pd.factorize(df['category_3'])
df['category_3'] = df['category_3'].astype(np.int16)




#df['category_2'].nunique()




df.head()




df_test = df[df['price'] == -1]
df_train = df[df['price'] != -1]




df_train['price'].head()




#price = df_train['price']
#price = np.log1p(df_train['price'])
price = df_train['price']

#pp = np.exp(price)




df_train.drop(['test_id', 'train_id', 'price'], axis=1, inplace=True)
df_test.drop(['test_id', 'train_id', 'price'], axis=1, inplace=True)




print(df_train.shape)
df_train.head()




price.shape




print(df_test.shape)
df_test.head()




len(sample)




del df




from sklearn.model_selection import train_test_split

#features= [c for c in df_train.columns.values if c  not in ['id', 'target']]
#numeric_features= [c for c in df.columns.values if c  not in ['id','text','author','processed']]
#target = 'author'

X_train, X_test, y_train, y_test = train_test_split(df_train, price, test_size=0.33, random_state=42)
X_train.head()




from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

    #('features',feats),
    ('classifier', RandomForestRegressor(random_state = 42))
    #('classifier', GradientBoostingClassifier(random_state = 42))
])


preds = pipeline.predict(X_test)








from sklearn.model_selection import GridSearchCV

hyperparameters = { #'features__text__tfidf__max_df': [0.9, 0.95],
                    #'features__text__tfidf__ngram_range': [(1,1), (1,2)],
                    #'classifier__learning_rate': [0.1, 0.2],
                    'classifier__n_estimators': [30],
                    'classifier__max_depth': [6, 8],
                    'classifier__min_samples_leaf': [4, 6]
                  }
clf = GridSearchCV(pipeline, hyperparameters, cv = 3)
 
# Fit and tune model
clf.fit(X_train, y_train)




clf.best_params_




#refitting on entire training data using best settings
clf.refit

preds = clf.predict(X_test)
#probs = clf.predict_proba(X_test)

np.mean(preds == y_test)




preds = clf.predict(df_test)




out = pd.DataFrame()
df_test = pd.read_csv('../input/test.tsv', sep='\t')
out['test_id'] = df_test['test_id']
out['price'] = preds

out.to_csv("Random_Forest_1.csv",index=False)

out.head()




'''from sklearn import ensemble
clf =  ensemble.GradientBoostingRegressor(learning_rate = 0.7, n_estimators = 300, max_depth = 3, warm_start = True, verbose=1, random_state=45, max_features = 0.8)

df = pd.DataFrame()
df['price'] = np.log1p(price)

clf.fit(df_train, df['price'])'''




'''predicted = clf.predict(df_test) 

print(df_train.columns)
print( clf.feature_importances_)'''




#out = pd.DataFrame()




'''df_test = pd.read_csv('../input/test.tsv', sep='\t')
out['test_id'] = df_test['test_id']
out['price'] = np.exp(predicted) - 1'''




out.head()




out.to_csv("Grad_Boost_1.csv",index=False)




def rmsle(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return np.sqrt(np.mean(np.power(np.log(y_true + 1) - np.log(y_pred + 1), 2)))





from sklearn.ensemble import ExtraTreesRegressor

df = pd.DataFrame()
#df['price'] = np.log(price)
#df['price'] = 1 + np.log(price)
df['price'] = np.log1p(price)

# A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
'''def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i, pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5'''


'''et_model = ExtraTreesRegressor(n_jobs=-1, n_estimators=100,  random_state=42, min_samples_leaf=2)

et_model.fit(df[col], df['price'])

y_tr_1 = et_model.predict(df[col])

print('RMSLE: {0:.5f}'.format(rmsle(np.exp(df['price'])-1, np.exp(y_tr_1)-1)))


y_pred_1 = et_model.predict(test_df[col])'''

et_model = ExtraTreesRegressor()

et_model.fit(df_train, df['price'])

y_tr = et_model.predict(df_train)

#print(rmsle(np.exp(df['price'])-1, np.exp(y_tr)-1))
print(rmsle(np.exp(df['price']), np.exp(y_tr)))

y_pred = et_model.predict(df_test)
 
    




df_test = pd.read_csv('../input/test.tsv', sep='\t')

out = pd.DataFrame()
out['test_id'] = df_test['test_id'].astype(np.int)
#out['price'] = np.exp(y_pred)-1
out['price'] = np.exp(y_pred) - 1

out.to_csv("Extra_Trees_Reg_output.csv",index=False)

