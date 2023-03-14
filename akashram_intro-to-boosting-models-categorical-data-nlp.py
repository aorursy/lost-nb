#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import os

import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nltk.tokenize import RegexpTokenizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')




train_df = pd.read_csv('../input/data-science-bowl-2019/train.csv')
test_df = pd.read_csv('../input/data-science-bowl-2019/test.csv')
train_labels_df = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')
specs_df = pd.read_csv('../input/data-science-bowl-2019/specs.csv')
sample_submission_df = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')

data = pd.read_csv('/kaggle/input/jobs-on-naukricom/home/sdf/marketing_sample_for_naukri_com-jobs__20190701_20190830__30k_data.csv')
data.head(5)




data.dtypes




rc = data['Role Category'].value_counts().reset_index()
rc.columns = ['Role Category', 'Count']
rc['Percent'] = rc['Count']/rc['Count'].sum() * 100
rc




rc = rc[:10]

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

rcParams['figure.figsize'] = 13, 10
ax = sns.barplot(x="Role Category", y="Count", data=rc)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)




rc = data['Role Category'].value_counts().nlargest(n=10)

fig = px.pie(rc, 
       values = rc.values, 
       names = rc.index, 
       title="Top 10 Role Categories", 
       color=rc.values)
       
fig.update_traces(opacity=0.5,
                  marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5)
fig.update_layout(title_x=0.5)
fig.show()




location = data['Location'].value_counts().nlargest(n=10)

fig = px.bar(y=location.values,
       x=location.index,
       orientation='v',
       color=location.index,
       text=location.values,
       color_discrete_sequence= px.colors.qualitative.Bold)

fig.update_traces(texttemplate='%{text:.2s}', 
                  textposition='outside', 
                  marker_line_color='rgb(8,48,107)', 
                  marker_line_width=1.5, 
                  opacity=0.7)

fig.update_layout(width=800, 
                  showlegend=False, 
                  xaxis_title="City",
                  yaxis_title="Count",
                  title="Top 10 cities by job count")
fig.show()




data1 = data[:10 ] ## taking just 10 records for demo

lis_sum_t = data1[['Job Title', 'Job Experience Required']]
two_cls = pd.crosstab(lis_sum_t['Job Title'], lis_sum_t['Job Experience Required'])

two_cls.plot.bar(stacked=True)
#plt.legend(title='mark')
plt.show()




place_map = {'Location': {'Hyderabad': 1, 'Pune': 2, 'Bengaluru': 3, 'Mumbai': 4,
                                  'Gurgaon': 5, 'Pune,Pune': 6}}




labels = data1['Location'].astype('category').cat.categories.tolist()

replace_map_comp = {'Location' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}

print(replace_map_comp)




#data1.replace(replace_map_comp, inplace=True)
data1['Location']
data1['Location'].value_counts()




from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()

data2 = data[:10]

data2['l_code'] = lb_make.fit_transform(data1['Location'])

data2.head() #Results in appending a new column to df




from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()

lb_results = lb.fit_transform(data2['Location'])

lb_results_df = pd.DataFrame(lb_results, columns=lb.classes_)

print(lb_results_df.head())




result_df = pd.concat([data2, lb_results_df], axis=1)

result_df.head(2)




get_ipython().system('pip install category_encoders')




import category_encoders as ce

encoder = ce.BinaryEncoder(cols=['Location'])
df_binary = encoder.fit_transform(data2)
df_binary.head()




encoder = ce.BackwardDifferenceEncoder(cols=['Job Title'])

df_bd = encoder.fit_transform(data2)

df_bd.head()




data2['Job Experience Required'].value_counts()




data2['Job Experience Required'] = data2['Job Experience Required'].str.replace("yrs", "")




def split_mean(x):
    split_list = x.split('-')
    mean = (float(split_list[0])+float(split_list[1]))/2
    return mean

data2['exp_mean'] = data2['Job Experience Required'].apply(lambda x: split_mean(x))

data2.head()




data["About_me"] = "I am a" + data["Job Title"]+" from "+ data["Location"]+". "+" I work in the "+ data["Industry"]+" Industry as a "+ data['Role']
data['len'] = data['About_me'].str.len()




import spacy
nlp = spacy.load("en_core_web_sm")
import re
import nltk
import gensim




import plotly.express as px
fig = px.histogram(data, x="len")
fig.show()




pd.set_option('display.max_colwidth', -1)
m = data['len'].max()
ab = data[data['len'] == m]
ab['About_me']




import re 

data['About_me'] = data['About_me'].fillna('').astype('str')
data['detail_abt'] = data['About_me'].apply(lambda x: nltk.sent_tokenize(x))




avg_len = data['About_me'].str.split().apply(lambda x : [len(i) for i in x]).map(lambda x: np.mean(x))

import plotly.express as px

fig = px.histogram(avg_len)

fig.show()




import nltk
stopwords = nltk.corpus.stopwords.words('english')
#stop=set(stopwords.words('english'))
print(stopwords[:10])




# Now, we’ll  create the corpus.

corpus=[]

new = data['About_me'].str.split()

new = new.values.tolist()

corpus=[word for i in new for word in i]

from collections import defaultdict, Counter

dic=defaultdict(int)

for word in corpus:
    if word in stopwords:
        dic[word]+=1




#plotly.offline.initnotebookmode(connected = True)

top = sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10]

x,y = zip(*top)

x = list(x)
y = list(y)

import plotly.graph_objects as go

fig = go.Figure(go.Bar(
            x=y,
            y=x,
            orientation='h',
            textposition='auto',
            marker=dict(color='rgba(58, 71, 80, 0.6)')))

fig.show()




counter=Counter(corpus)
most=counter.most_common()

x, y= [], []

for word,count in most[:40]:
    if (word not in stopwords):
        x.append(word)
        y.append(count)
        

x = list(x)
y = list(y)
fig = go.Figure(go.Bar(
            x=y,
            y=x,
            orientation='h',
            marker=dict(color='rgba(246, 78, 139, 0.6)')))

fig.show()        




from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer

def get_top_ngram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) 
                  for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:10]




top_n_bigrams=get_top_ngram(data['About_me'],2)[:10]

x,y=map(list,zip(*top_n_bigrams))

#sns.barplot(x=y,y=x)

x = list(x)
y = list(y)

fig = go.Figure(go.Bar(
            x=y,
            y=x,
            orientation='h',
            marker_color='rgb(26, 118, 255)'))

fig.show()     




top_tri_grams=get_top_ngram(data['About_me'], n=3)
x,y=map(list,zip(*top_tri_grams))

#sns.barplot(x=y,y=x)

x = list(x)
y = list(y)
fig = go.Figure(go.Bar(
            x=y,
            y=x,
            orientation='h'))
fig.show()     




from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)

def show_wordcloud(data):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=100,
        max_font_size=30,
        scale=3,
        random_state=1)
   
    wordcloud=wordcloud.generate(str(data))

    fig = plt.figure(1, figsize=(16, 12))
    plt.axis('off')

    plt.imshow(wordcloud)
    plt.show()

show_wordcloud(corpus)




# There are three pre-trained models for English in spaCy. I will use en_core_web_sm for our task but you can try other models.

import spacy
nlp = spacy.load("en_core_web_sm")




doc = nlp('India and Iran have agreed to boost the economic viability of the strategic Chabahar port through various measures, including larger subsidies to merchant shipping firms using the facility, people familiar with the development said on Thursday.')

[(x.text,x.label_) for x in doc.ents]




from spacy import displacy

displacy.render(doc, style='ent')




def ner(text):
    doc=nlp(text)
    return [X.label_ for X in doc.ents]

data1 = data[:1000]

ent = data1['About_me'].apply(lambda x : ner(x))

ent=[x for sub in ent for x in sub]

counter=Counter(ent)

count=counter.most_common()




x,y=map(list,zip(*count))

#sns.barplot(x=y,y=x)

x = list(x)
y = list(y)

fig = go.Figure(go.Bar(
            x=y,
            y=x,
            orientation='h',
            marker_color='rgb(55, 83, 109)'))
fig.show()




def ner(text,ent="GPE"):
    doc=nlp(text)
    return [X.text for X in doc.ents if X.label_ == ent]

gpe = data1['About_me'].apply(lambda x: ner(x))

gpe=[i for x in gpe for i in x]

counter=Counter(gpe)

x,y=map(list,zip(*counter.most_common(10)))

# sns.barplot(y,x)

x = list(x)
y = list(y)

fig = go.Figure(go.Bar(
            x=y,
            y=x,
            orientation='h',
            ))
fig.show()




# I will use the nltk to do the parts of speech tagging but there are other libraries that do a good job (spacy, textblob).

# Let’s look at an example.

import nltk

sentence="The greatest comeback stories in 2019"

tokens = nltk.word_tokenize(sentence)

nltk.pos_tag(tokens)




doc = nlp('The greatest comeback stories in 2020')

displacy.render(doc, style='dep', jupyter=True, options={'distance': 90})




def pos(text):
    pos=nltk.pos_tag(nltk.word_tokenize(text))
    pos = list(map(list, zip(*pos)))[1]    
    return pos

data1 = data[:1000]

data1['About_me'] = data1['About_me'].str.replace('', 'dummy')

tags = data1['About_me'].apply(lambda x : pos(x))

tags=[x for l in tags for x in l]
counter=Counter(tags)
x,y=list(map(list,zip(*counter.most_common(7))))

# sns.barplot(x=y,y=x)

x = list(x)
y = list(y)

fig = go.Figure(go.Bar(
            x=y,
            y=x,
            orientation='h',
            marker_color='rgb(55, 83, 109)'))
fig.show()     




def get_adjs(text):
    adj=[]
    pos=nltk.pos_tag(nltk.word_tokenize(text))
    for word,tag in pos:
        if tag=='NN':
            adj.append(word)
    return adj


data2 = data[:1000]

words = data2['About_me'].apply(lambda x : get_adjs(x))

words=[x for l in words for x in l]
counter=Counter(words)

x,y=list(map(list,zip(*counter.most_common(7))))

# sns.barplot(x=y,y=x)

x = list(x)
y = list(y)

fig = go.Figure(go.Bar(
            x=y,
            y=x,
            orientation='h',
            marker_color='rgb(227, 119, 194)'))
fig.show()     




get_ipython().system('pip install textstat')

from textstat import flesch_reading_ease

score = data2['About_me'].apply(lambda x : flesch_reading_ease(x))

score.hist()




import plotly.graph_objects as go
import numpy as np

## Normalized Histogram

fig = go.Figure(data=[go.Histogram(x=score, histnorm='probability')])

fig.show()




data2['read_score'] = data2['About_me'].apply(lambda x : flesch_reading_ease(x))

alls = data2[data2['read_score'] < 5].head(5)

alls['About_me']




import xgboost
import gc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier 
import lightgbm as lgb
from numba import jit




def extract_time_features(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['month'] = df['timestamp'].dt.month
    df['hour'] = df['timestamp'].dt.hour
    df['year'] = df['timestamp'].dt.year
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['weekofyear'] = df['timestamp'].dt.weekofyear
    df['dayofyear'] = df['timestamp'].dt.dayofyear
    df['quarter'] = df['timestamp'].dt.quarter
    df['is_month_start'] = df['timestamp'].dt.is_month_start    
    return df




def get_object_columns(df, columns):
    df = df.groupby(['installation_id', columns])['event_id'].count().reset_index()
    df = df.pivot_table(index = 'installation_id', columns = [columns], values = 'event_id')
    df.columns = list(df.columns)
    df.fillna(0, inplace = True)
    return df

def get_numeric_columns(df, column):
    df = df.groupby('installation_id').agg({f'{column}': ['mean', 'sum', 'min', 'max', 'std', 'skew']})
    df[column].fillna(df[column].mean(), inplace = True)
    df.columns = [f'{column}_mean', f'{column}_sum', f'{column}_min', f'{column}_max', f'{column}_std', f'{column}_skew']
    return df

def get_numeric_columns_add(df, agg_column, column):
    df = df.groupby(['installation_id', agg_column]).agg({f'{column}': ['mean', 'sum', 'min', 'max', 'std', 'skew']}).reset_index()
    df = df.pivot_table(index = 'installation_id', columns = [agg_column], values = [col for col in df.columns if col not in ['installation_id', 'type']])
    df[column].fillna(df[column].mean(), inplace = True)
    df.columns = list(df.columns)
    return df




def perform_features_engineering(train_df, test_df, train_labels_df):
    print(f'Perform features engineering')
    numerical_columns = ['game_time']
    categorical_columns = ['type', 'world']

    comp_train_df = pd.DataFrame({'installation_id': train_df['installation_id'].unique()})
    comp_train_df.set_index('installation_id', inplace = True)
    comp_test_df = pd.DataFrame({'installation_id': test_df['installation_id'].unique()})
    comp_test_df.set_index('installation_id', inplace = True)

    test_df = extract_time_features(test_df)
    train_df = extract_time_features(train_df)

    for i in numerical_columns:
        comp_train_df = comp_train_df.merge(get_numeric_columns(train_df, i), left_index = True, right_index = True)
        comp_test_df = comp_test_df.merge(get_numeric_columns(test_df, i), left_index = True, right_index = True)
    
    for i in categorical_columns:
        comp_train_df = comp_train_df.merge(get_object_columns(train_df, i), left_index = True, right_index = True)
        comp_test_df = comp_test_df.merge(get_object_columns(test_df, i), left_index = True, right_index = True)
    
    for i in categorical_columns:
        for j in numerical_columns:
            comp_train_df = comp_train_df.merge(get_numeric_columns_add(train_df, i, j), left_index = True, right_index = True)
            comp_test_df = comp_test_df.merge(get_numeric_columns_add(test_df, i, j), left_index = True, right_index = True)
    
    
    comp_train_df.reset_index(inplace = True)
    comp_test_df.reset_index(inplace = True)
       
    labels_map = dict(train_labels_df.groupby('title')['accuracy_group'].agg(lambda x:x.value_counts().index[0]))
 
    labels = train_labels_df[['installation_id', 'title', 'accuracy_group']]
    
    labels['title'] = labels['title'].map(labels_map)
   
    comp_test_df['title'] = test_df.groupby('installation_id').last()['title'].map(labels_map).reset_index(drop = True)
   
    comp_train_df = labels.merge(comp_train_df, on = 'installation_id', how = 'left')
    print('We have {} training rows'.format(comp_train_df.shape[0]))
    
    return comp_train_df, comp_test_df




def qwk3(a1, a2, max_rat=3):
    assert(len(a1) == len(a2))
    a1 = np.asarray(a1, dtype=int)
    a2 = np.asarray(a2, dtype=int)
    hist1 = np.zeros((max_rat + 1, ))
    hist2 = np.zeros((max_rat + 1, ))
    o = 0
    for k in range(a1.shape[0]):
        i, j = a1[k], a2[k]
        hist1[i] += 1
        hist2[j] += 1
        o +=  (i - j) * (i - j)
    e = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            e += hist1[i] * hist2[j] * (i - j) * (i - j)
    e = e / a1.shape[0]
    return 1 - o / e




ada_train_df, ada_test_df = perform_features_engineering(train_df, test_df, train_labels_df)

null_columns = ada_test_df.columns[ada_test_df.isnull().any()]
ada_test_df[null_columns].isnull().sum()

ada_test_df['game_time_std'] = ada_test_df['game_time_std'].fillna(0)
ada_test_df['game_time_skew'] = ada_test_df['game_time_skew'].fillna(0)




def adaboost_it(ada_train_df, ada_test_df):
    print("Ada-Boosting...")
    t_splits = 5
    k_scores = []
    kf = KFold(n_splits = t_splits)
    features = [i for i in ada_train_df.columns if i not in ['accuracy_group', 'installation_id']]
    target = 'accuracy_group'
    oof_pred = np.zeros((len(ada_train_df), 4))
    y_pred = np.zeros((len(ada_test_df), 4))
    for fold, (tr_ind, val_ind) in enumerate(kf.split(ada_train_df)):
        print(f'Fold: {fold+1}')
        x_train, x_val = ada_train_df[features].iloc[tr_ind], ada_train_df[features].iloc[val_ind]
        y_train, y_val = ada_train_df[target][tr_ind], ada_train_df[target][val_ind]
               
        ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200,algorithm="SAMME.R", learning_rate=0.5)
        ada_clf.fit(x_train, y_train)
        oof_pred[val_ind] = ada_clf.predict_proba(x_val)
      
        y_pred += ada_clf.predict_proba(ada_test_df[features]) / t_splits
        
        val_crt_fold = qwk3(y_val, oof_pred[val_ind].argmax(axis = 1))
        print(f'Fold: {fold+1} quadratic weighted kappa score: {np.round(val_crt_fold,4)}')
        
    res = qwk3(ada_train_df['accuracy_group'], oof_pred.argmax(axis = 1))
    print(f'Quadratic weighted score: {np.round(res,4)}')
        
    return y_pred




y_pred = adaboost_it(ada_train_df, ada_test_df)

ada_test_df = ada_test_df.reset_index()
ada_test_df = ada_test_df[['installation_id']]
ada_test_df['accuracy_group'] = y_pred.argmax(axis = 1)
ada_sample_submission_df = sample_submission_df.merge(ada_test_df, on = 'installation_id')
ada_sample_submission_df.to_csv('ada_boost_submission.csv', index = False)




xgb_train_df, xgb_test_df = perform_features_engineering(train_df, test_df, train_labels_df)

features = [i for i in xgb_train_df.columns if i not in ['accuracy_group', 'installation_id']]
target = 'accuracy_group'


x_train  = xgb_train_df[features]
y_train = xgb_train_df[target]




## Grid search is very time consuming and therefore i have commented it for now.

#from sklearn.model_selection import GridSearchCV
#model = xgboost.XGBClassifier()

#param_dist = {"max_depth": [10,30,50],"min_child_weight" : [1,3,6],
 #             "n_estimators": [200],
  #            "learning_rate": [0.05, 0.1,0.16],}

#grid_search = GridSearchCV(model, param_grid=param_dist, cv = 3, verbose=10, n_jobs=-1)
#grid_search.fit(x_train, y_train)
#grid_search.best_estimator_




def xgb(xgb_train_df, xgb_test_df):
    print("XG-Boosting...")
    t_splits = 5
    k_scores = []
    kf = KFold(n_splits = t_splits)
    features = [i for i in xgb_train_df.columns if i not in ['accuracy_group', 'installation_id']]
    target = 'accuracy_group'
    oof_pred = np.zeros((len(xgb_train_df), 4))
    y_pred = np.zeros((len(xgb_test_df), 4))
    for fold, (tr_ind, val_ind) in enumerate(kf.split(xgb_train_df)):
        print(f'Fold: {fold+1}')
        x_train, x_val = xgb_train_df[features].iloc[tr_ind], xgb_train_df[features].iloc[val_ind]
        y_train, y_val = xgb_train_df[target][tr_ind], xgb_train_df[target][val_ind]
        
        xgb_clf = xgboost.XGBClassifier()
        xgb_clf.fit(x_train, y_train)
        oof_pred[val_ind] = xgb_clf.predict_proba(x_val)
      
        y_pred += xgb_clf.predict_proba(xgb_test_df[features]) / t_splits
        
        val_crt_fold = qwk3(y_val, oof_pred[val_ind].argmax(axis = 1))
        print(f'Fold: {fold+1} quadratic weighted kappa score: {np.round(val_crt_fold,4)}')
        
    res = qwk3(xgb_train_df['accuracy_group'], oof_pred.argmax(axis = 1))
    print(f'Quadratic weighted score: {np.round(res,4)}')
        
    return y_pred


y_pred = xgb(xgb_train_df, xgb_test_df)




xgb_test_df = xgb_test_df.reset_index()
xgb_test_df = xgb_test_df[['installation_id']]
xgb_test_df['accuracy_group'] = y_pred.argmax(axis = 1)
xgb_sample_submission_df = sample_submission_df.merge(xgb_test_df, on = 'installation_id')
xgb_sample_submission_df.to_csv('xgb_submission.csv', index = False)




xgb_sample_submission_df = xgb_sample_submission_df.drop('accuracy_group_x', axis=1)
xgb_sample_submission_df.columns = ['installation_id', 'accuracy_group']




xgb_sample_submission_df.to_csv('xgb_submission.csv', index = False)




cat_train_df, cat_test_df = perform_features_engineering(train_df, test_df, train_labels_df)

xc_train  = cat_train_df[features]
yc_train = cat_train_df[target]




#cat_test_df.columns
#import re

# cat_test_df = cat_test_df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
#cat_train_df.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in cat_train_df.columns]




import catboost as cb

def cat(cat_train_df, cat_test_df):
    print("Meeowwww...")
    t_splits = 3
    k_scores = []
    kf = KFold(n_splits = t_splits)
    features = [i for i in cat_train_df.columns if i not in ['accuracy_group', 'installation_id']]
    target = 'accuracy_group'
    oof_pred = np.zeros((len(cat_train_df), 4))
    y_pred = np.zeros((len(cat_test_df), 4))
    for fold, (tr_ind, val_ind) in enumerate(kf.split(cat_train_df)):
        print(f'Fold: {fold+1}')
        x_train, x_val = cat_train_df[features].iloc[tr_ind], cat_train_df[features].iloc[val_ind]
        y_train, y_val = cat_train_df[target][tr_ind], cat_train_df[target][val_ind]
        
        cat_clf = cb.CatBoostClassifier(depth=10, iterations= 200, l2_leaf_reg= 9, learning_rate= 0.15, silent=True)
        cat_clf.fit(xc_train, yc_train)
        oof_pred[val_ind] = cat_clf.predict_proba(x_val)
      
        y_pred += cat_clf.predict_proba(cat_test_df[features]) / t_splits
        
        val_crt_fold = qwk3(y_val, oof_pred[val_ind].argmax(axis = 1))
        print(f'Fold: {fold+1} quadratic weighted kappa score: {np.round(val_crt_fold,4)}')
        
    res = qwk3(cat_train_df['accuracy_group'], oof_pred.argmax(axis = 1))
    print(f'Quadratic weighted score: {np.round(res,4)}')
        
    return y_pred




y_pred_cat = cat(cat_train_df, cat_test_df)


cat_test_df = cat_test_df.reset_index()

cat_test_df = cat_test_df[['installation_id']]
cat_test_df['accuracy_group'] = y_pred_cat.argmax(axis = 1)

cat_sample_submission_df = sample_submission_df.merge(cat_test_df, on = 'installation_id')
cat_sample_submission_df.to_csv('submission.csv', index = False)

cat_sample_submission_df = cat_sample_submission_df.drop('accuracy_group_x', axis=1)

cat_sample_submission_df.columns = ['installation_id', 'accuracy_group']

cat_sample_submission_df.to_csv('submission.csv', index = False)




lgb_train_df, lgb_test_df = perform_features_engineering(train_df, test_df, train_labels_df)

xl_train  = lgb_train_df[features]
yl_train = lgb_train_df[target]




import lightgbm as lgb

def lgbc(lgb_train_df, lgb_test_df):
    print("I'm so light you know...")
    t_splits = 3
    k_scores = []
    kf = KFold(n_splits = t_splits)
    features = [i for i in lgb_train_df.columns if i not in ['accuracy_group', 'installation_id']]
    target = 'accuracy_group'
    oof_pred = np.zeros((len(lgb_train_df), 4))
    y_pred = np.zeros((len(lgb_test_df), 4))
    for fold, (tr_ind, val_ind) in enumerate(kf.split(lgb_train_df)):
        print(f'Fold: {fold+1}')
        x_train, x_val = lgb_train_df[features].iloc[tr_ind], lgb_train_df[features].iloc[val_ind]
        y_train, y_val = lgb_train_df[target][tr_ind], lgb_train_df[target][val_ind]
        
        lg = lgb.LGBMClassifier(silent=False)
        lg.fit(xl_train, yl_train)
        oof_pred[val_ind] = lg.predict_proba(x_val)
      
        y_pred += lg.predict_proba(lgb_test_df[features]) / t_splits
        
        val_crt_fold = qwk3(y_val, oof_pred[val_ind].argmax(axis = 1))
        print(f'Fold: {fold+1} quadratic weighted kappa score: {np.round(val_crt_fold,4)}')
        
    res = qwk3(lgb_train_df['accuracy_group'], oof_pred.argmax(axis = 1))
    print(f'Quadratic weighted score: {np.round(res,4)}')
        
    return y_pred




#lgb_train_df.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in lgb_train_df.columns]
# lgb_test_df.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in lgb_test_df.columns]

#y_pred_lgb = lgbc(lgb_train_df, lgb_test_df)

#lgb_test_df = lgb_test_df.reset_index()
#lgb_test_df = lgb_test_df[['installation_id']]

#lgb_test_df['accuracy_group'] = y_pred_lgb.argmax(axis = 1)

#lgb_sample_submission_df = sample_submission_df.merge(lgb_test_df, on = 'installation_id')

#lgb_sample_submission_df.to_csv('lgb_submission.csv', index = False)

#lgb_sample_submission_df = lgb_sample_submission_df.drop('accuracy_group_x', axis=1)

#lgb_sample_submission_df.columns = ['installation_id', 'accuracy_group']




data = [['ada', 0.42], ['xgb', 0.44], ['cat', 0.65], ['lgb', 0.62]]

df = pd.DataFrame(data, columns = ['Model', 'Validation Kappa Score']) 

import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Bar(x=df['Model'], y=df['Validation Kappa Score'], marker_color='#FFD700'))
fig.show()

