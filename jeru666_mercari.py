#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


print(df_train.shape)
df_train.head()


# In[3]:


print(df_test.shape)
df_test.head()


# In[4]:


print(df_train.dtypes.unique())


# In[5]:


pp = pd.value_counts(df_train.dtypes)
pp.plot.bar()
plt.show()


# In[6]:


cols_missing_val_train = df_train.columns[df_train.isnull().any()].tolist()
print(cols_missing_val_train)
print('\n')

cols_missing_val_test = df_test.columns[df_test.isnull().any()].tolist()
print(cols_missing_val_test)


# In[7]:


# Train data
msno.bar(df_train[cols_missing_val_train],figsize=(20,8),color="#19455e",fontsize=18,labels=True,)


# In[8]:


# Test data
msno.bar(df_test[cols_missing_val_test],figsize=(20,8),color="#50085e",fontsize=18,labels=True,)


# In[9]:


# Train data
msno.matrix(df_train[cols_missing_val_train],width_ratios=(10,1),            figsize=(20,8),color=(0.5,0.5,0.2),fontsize=18,sparkline=True,labels=True)


# In[10]:


#--- Test dataframe ---
msno.matrix(df_test[cols_missing_val_test],width_ratios=(10,1),            figsize=(20,8),color=(0.9,0.2,0.2),fontsize=18,sparkline=True,labels=True)


# In[11]:


df_train['category_name'] = df_train['category_name'].fillna("Unknown_category")
df_test['category_name'] = df_test['category_name'].fillna("Unknown_category")


# In[12]:


print(df_train.shape)
print(df_test.shape)


# In[13]:


df_train['brand_name'] = df_train['brand_name'].fillna("Unknown")
df_test['brand_name'] = df_test['brand_name'].fillna("Unknown")


# In[14]:


df_train = df_train[pd.notnull(df_train['item_description'])]


# In[15]:


print(df_train.shape)


# In[16]:


df_test['price'] = -1
df = pd.concat([df_train, df_test])

print(df.shape)
df.head()


# In[17]:


a = 'This / is / me'
b = a.split('/')
print(b)


# In[18]:


ha = "some-sample-filename-to-split"
"-".join(ha.split("-", 3)[:3])


# In[19]:


df['category_name'].head(15)


# In[20]:


df.shape


# In[21]:


'''def my_test2(row):
     #return row['a'] % row['c']
    return row.count('/') + 1

df['category_num'] = df.apply(my_test2, axis = 1)'''

#for i in df['category_name']:
#    df['category_num'] = i.count('/') + 1


# In[22]:


df.head()


# In[23]:


primary_category_list = []
for i in df['category_name']:
    b = i.split('/')
    primary_category_list.append(b[0])
    
    primary_category_list = set(primary_category_list)
    primary_category_list = list(primary_category_list)
    
len(primary_category_list)    


# In[24]:


primary_category_list


# In[25]:


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


# In[26]:


dual_category_list = []
for i in df['category_name']:
    b = "/".join(i.split("/", 2)[:2])
    dual_category_list.append(b)
    
    dual_category_list = set(dual_category_list)
    dual_category_list = list(dual_category_list)
    
len(dual_category_list) 


# In[27]:


dual_category_list


# In[28]:


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


# In[29]:


triple_category_list = []
for i in df['category_name']:
    b = "/".join(i.split("/", 3)[:3])
    triple_category_list.append(b)
    
    triple_category_list = set(triple_category_list)
    triple_category_list = list(triple_category_list)
    
len(triple_category_list) 


# In[30]:


triple_category_list


# In[31]:


category_name_list = []
for i in df['category_name']:
    b = i.split('/')
    for j in b:
        category_name_list.append(j)
        
    category_name_list = set(category_name_list)
    category_name_list = list(category_name_list)
    
len(category_name_list)    


# In[32]:


category_name_list


# In[33]:


'''for col in category_name_list:
    df[col] = np.int(0)
    df[col] = df[col].astype(np.int8)'''


# In[34]:


'''print(df.shape)
df.head()'''


# In[35]:


# df.drop('category_name', axis=1, inplace=True)


# In[36]:


len(df)


# In[37]:


df.name.nunique()


# In[38]:


df.drop('item_description', axis=1, inplace=True)
#df.drop('name', axis=1, inplace=True)


# In[39]:


df_cat = pd.DataFrame(df.category_name.str.split('/',2).tolist(),
                                   columns = ['category_1','category_2', 'category_3'])
df['category_1'] = df_cat['category_1']
df['category_2'] = df_cat['category_2']
df['category_3'] = df_cat['category_3']


# In[40]:


df.drop('category_name', axis=1, inplace=True)


# In[41]:


df.head(15)


# In[42]:


df.shape


# In[43]:


df['name'], _ = pd.factorize(df['name'])
#df['brand_name'] = df['brand_name'].astype(np.int16)
df['name'].nunique()


# In[44]:


df['brand_name'], _ = pd.factorize(df['brand_name'])
df['brand_name'] = df['brand_name'].astype(np.int16)
df['brand_name'].unique()


# In[45]:


#for r in df.category_name:
#    df['cat_num'] = r.count('/') + 1

for r in range(0, len(df)):
    df['cat_num'] = df['category_name'].iloc[r].count('/') + 1
  
#df['category_name'][100]

#df['category_name'].iloc[100]

#df.type()
#df.category_name.iter

df.head()


# In[46]:


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


# In[47]:


df['category_1'], _ = pd.factorize(df['category_1'])
df['category_1'] = df['category_1'].astype(np.int8)


# In[48]:


df['category_2'], _ = pd.factorize(df['category_2'])
df['category_2'] = df['category_2'].astype(np.int8)


# In[49]:


df['category_3'], _ = pd.factorize(df['category_3'])
df['category_3'] = df['category_3'].astype(np.int16)


# In[50]:


#df['category_2'].nunique()


# In[51]:


df.head()


# In[52]:


df_test = df[df['price'] == -1]
df_train = df[df['price'] != -1]


# In[53]:


df_train['price'].head()


# In[54]:


#price = df_train['price']
#price = np.log1p(df_train['price'])
price = df_train['price']

#pp = np.exp(price)


# In[55]:


df_train.drop(['test_id', 'train_id', 'price'], axis=1, inplace=True)
df_test.drop(['test_id', 'train_id', 'price'], axis=1, inplace=True)


# In[56]:


print(df_train.shape)
df_train.head()


# In[57]:


price.shape


# In[58]:


print(df_test.shape)
df_test.head()


# In[59]:


len(sample)


# In[60]:


del df


# In[61]:


from sklearn.model_selection import train_test_split

#features= [c for c in df_train.columns.values if c  not in ['id', 'target']]
#numeric_features= [c for c in df.columns.values if c  not in ['id','text','author','processed']]
#target = 'author'

X_train, X_test, y_train, y_test = train_test_split(df_train, price, test_size=0.33, random_state=42)
X_train.head()


# In[62]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

pipeline = Pipeline([
    #('features',feats),
    ('classifier', RandomForestRegressor(random_state = 42))
    #('classifier', GradientBoostingClassifier(random_state = 42))
])

pipeline.fit(X_train, y_train)

preds = pipeline.predict(X_test)


# In[63]:


pipeline.get_params().keys()


# In[64]:


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


# In[65]:


clf.best_params_


# In[66]:


#refitting on entire training data using best settings
clf.refit

preds = clf.predict(X_test)
#probs = clf.predict_proba(X_test)

np.mean(preds == y_test)


# In[67]:


preds = clf.predict(df_test)


# In[68]:


out = pd.DataFrame()
df_test = pd.read_csv('../input/test.tsv', sep='\t')
out['test_id'] = df_test['test_id']
out['price'] = preds

out.to_csv("Random_Forest_1.csv",index=False)

out.head()


# In[69]:


'''from sklearn import ensemble
clf =  ensemble.GradientBoostingRegressor(learning_rate = 0.7, n_estimators = 300, max_depth = 3, warm_start = True, verbose=1, random_state=45, max_features = 0.8)

df = pd.DataFrame()
df['price'] = np.log1p(price)

clf.fit(df_train, df['price'])'''


# In[70]:


'''predicted = clf.predict(df_test) 

print(df_train.columns)
print( clf.feature_importances_)'''


# In[71]:


#out = pd.DataFrame()


# In[72]:


'''df_test = pd.read_csv('../input/test.tsv', sep='\t')
out['test_id'] = df_test['test_id']
out['price'] = np.exp(predicted) - 1'''


# In[73]:


out.head()


# In[74]:


out.to_csv("Grad_Boost_1.csv",index=False)


# In[75]:


def rmsle(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return np.sqrt(np.mean(np.power(np.log(y_true + 1) - np.log(y_pred + 1), 2)))


# In[76]:



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
 
    


# In[77]:


df_test = pd.read_csv('../input/test.tsv', sep='\t')

out = pd.DataFrame()
out['test_id'] = df_test['test_id'].astype(np.int)
#out['price'] = np.exp(y_pred)-1
out['price'] = np.exp(y_pred) - 1

out.to_csv("Extra_Trees_Reg_output.csv",index=False)

