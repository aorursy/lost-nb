#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.




import numpy as np
import pandas as pd

from collections import Counter
#from collections import deque

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')

from wordcloud import WordCloud
import plotly as py
import plotly.graph_objs as go
from plotly import tools
import seaborn as sns

import ast                               # ast.literal_eval() to reformat strings into dictionaries
from urllib.request import urlopen
from PIL import Image                    # display jpg files

import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
#from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize
#from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
#from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor




train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.shape, test.shape




train.dtypes




# sanity check: target column
x = list(train.columns.values)
y = list(test.columns.values)
[item for item in x if item not in y]




train.head(2)




fig = plt.figure(figsize=(15, 5))
plt.title("Distribution of NAN values")
train.isna().sum().sort_values(ascending = True).plot(kind = 'barh')




# reformat strings into dictionaries
# ast.literal_eval() use instad of eval() inspired from https://www.kaggle.com/gravix/gradient-in-a-box
def refmt_str2dict(df, cols):
    for col in cols:
        df[col] = df[col].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x) )
    return df

dict_columns = ['belongs_to_collection', 'genres', 'production_companies', 'production_countries', 
                'spoken_languages', 'Keywords', 'cast', 'crew']

train = refmt_str2dict(train, dict_columns)




# path found here: https://www.kaggle.com/artgor/eda-feature-engineering-and-model-interpretation
TMDB_path = 'https://image.tmdb.org/t/p/w600_and_h900_bestv2/'




nrow = 3
fig = plt.figure(figsize=(20, nrow*5))
k = 0
for i in np.random.randint(train.shape[0], size=nrow*7):
    ax = fig.add_subplot(nrow, 7, k+1, xticks=[], yticks=[])
    img = Image.open(urlopen(TMDB_path + train['poster_path'][i]))
    plt.imshow(img)
    ax.set_title(f"{train['title'][i][0:22]}")
    k += 1




def display_posters(movies, nrow=1):
    fig = plt.figure(figsize=(20, nrow*5))
    max_plot = nrow*7
    if len(movies) <= max_plot:
        max_movies = len(movies)
    else:
        max_movies = max_plot
    for i in range(max_movies):
        ax = fig.add_subplot(nrow, 7, i+1, xticks=[], yticks=[])
        img = Image.open(urlopen(TMDB_path + movies['poster_path'][i]))
        plt.imshow(img)
        ax.set_title(f"{movies['title'][i][0:22]}")




# correct 'release_date' year
def fix_date(x):
    yr = x.split('/')[2]
    if int(yr) <= 19:
        return x[:-2] + '20' + yr
    else:
        return x[:-2] + '19' + yr

train['release_date'] = train['release_date'].apply(lambda x: fix_date(x))
train['release_date'] = pd.to_datetime(train['release_date'])




def process_date(df):
    date_attrs = ['year', 'month', 'day', 'weekday', 'weekofyear', 'quarter']
    for attr in date_attrs:
        new_col = 'release_date_' + attr
        df[new_col] = getattr(df['release_date'].dt, attr).astype(int)
    return df

train = process_date(train)




oldies = train[train['release_date_year'] < 1930].reset_index()
display_posters(oldies)




def WordCloud_fromDict(dict_name, key='name'):
    list_dict = list(train[dict_name].apply(lambda x: [i[key] for i in x] if x != {} else []).values)
#    list2txt = ' '.join([i for j in list_dict for i in j])
    list2txt = ' '.join(['_'.join(i.split(' ')) for j in list_dict for i in j])
    wordcloud = WordCloud(max_font_size = None, background_color = 'black', collocations = False,
                      width = 1200, height = 1000).generate(list2txt)
    return wordcloud




fig = plt.figure(figsize = (20, 10))
ax = fig.add_subplot(1,2,1)
ax.imshow(WordCloud_fromDict('genres'))
ax.set_title('GENRES')
ax.axis('off')
ax = fig.add_subplot(1,2,2)
ax.imshow(WordCloud_fromDict('Keywords'))
ax.set_title('KEYWORDS')
ax.axis('off')
plt.show()




list_genres = list(train['genres'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
list_keywords = list(train['Keywords'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
genres = set([y for x in list_genres for y in x])
keywords = set([y for x in list_keywords for y in x])
len(genres), len(keywords)




timeTravelMovies = train[train['Keywords'].apply(lambda x: 'time travel' in [i['name'] for i in x])].reset_index()
display_posters(timeTravelMovies)




nrow = 2
# checking lead cast 'cast_id'] == 1
list_cast_wURL = list(train['cast'].apply(
    lambda x: [(i['name'], i['profile_path']) for i in x if i['cast_id'] == 1] if x != {} else []).values)
top_cast = Counter([y for x in list_cast_wURL for y in x]).most_common(7*nrow)

fig = plt.figure(figsize=(20, nrow*5))
k = 0
for i in top_cast:
    ax = fig.add_subplot(nrow, 7, k+1, xticks=[], yticks=[])
    img = Image.open(urlopen(TMDB_path + i[0][1]))
    plt.imshow(img)
    ax.set_title(i[0][0][0:22])
    k += 1




BillMurrayMovies = train[train['cast'].apply(lambda x: 'Bill Murray' in [i['name'] for i in x])].reset_index()
display_posters(BillMurrayMovies)




NolanMovies = train[train['crew'].apply(lambda x: 'Christopher Nolan' in [i['name'] for i in x if                                                                          i['job'] == 'Director'])].reset_index()
display_posters(NolanMovies)




list_crew = list(train['crew'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
cast = set([y for x in list_cast_wURL for y in x])
crew = set([y for x in list_crew for y in x])
len(cast), len(crew)




fig = plt.figure(figsize = (20, 10))
ax = fig.add_subplot(1,2,1)
ax.imshow(WordCloud_fromDict('production_companies'))
ax.set_title('PRODUCTION COMPANIES')
ax.axis('off')
ax = fig.add_subplot(1,2,2)
ax.imshow(WordCloud_fromDict('production_countries'))
ax.set_title('PRODUCTION COUNTRIES')
ax.axis('off')
plt.show()




list_prodCompanies = list(train['production_companies'].apply(
    lambda x: [i['name'] for i in x] if x != {} else []).values)
list_prodCountries = list(train['production_countries'].apply(
    lambda x: [i['name'] for i in x] if x != {} else []).values)
prodCompanies = set([y for x in list_prodCompanies for y in x])
prodCountries = set([y for x in list_prodCountries for y in x])
len(prodCompanies), len(prodCountries)




AmblinMovies = train[train['production_companies'].apply(
    lambda x: 'Amblin Entertainment' in [i['name'] for i in x])].reset_index()
display_posters(AmblinMovies)




train[['budget', 'revenue', 'runtime', 'popularity']].describe()




plt.figure(figsize=(20,8))
plt.subplot(121)
plt.hist(np.log1p(train['budget']), bins = 50)   #some zero values in 'budget'
plt.title('budget_rescaled')
plt.subplot(122)
plt.hist(np.log1p(train['revenue']), bins = 50)
plt.title('revenue_rescaled')
plt.subplots_adjust(hspace=0.5)
plt.show()




yr_release = train['release_date_year'].value_counts().sort_index()
yr_budget = train.groupby(['release_date_year'])['budget'].mean()
yr_revenue = train.groupby(['release_date_year'])['revenue'].mean()
yr_popularity = train.groupby(['release_date_year'])['popularity'].mean()

plt.figure(figsize=(20,8))
plt.subplot(311)
plt.title('Movie count')
plt.plot(yr_release.index, yr_release.values)
plt.subplot(312)
plt.plot(yr_budget.index, yr_budget.values)
plt.plot(yr_revenue.index, yr_revenue.values)
plt.title('$')
plt.subplot(313)
plt.plot(yr_popularity.index, yr_popularity.values)
plt.title('Popularity')
plt.subplots_adjust(hspace=0.3)
plt.show()




OneMovie = train[train['title'] == 'The Terminator']
keywords = list(OneMovie['Keywords'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)[0]
keywords_str = ' '.join(keywords)
text_merged = list(OneMovie['overview'])[0] + ' | ' + list(OneMovie['tagline'])[0] + ' | ' + keywords_str
text_merged




# The following will not be used, as removing stopwords, getting stem, etc. will be done via TfidfVectorizer...
#stop_words = set(stopwords.words('english'))
#tokens = word_tokenize(text_merged)
#tokens_cleaned = [w for w in tokens if not w in stop_words]
#print(' '.join(tokens_cleaned))

#ps = PorterStemmer()
#tokens_stem = [ps.stem(w) for w in tokens_cleaned]
#tokens_stem_nopunct = [w.lower() for w in tokens_stem if w.isalpha()]
#print(' '.join(tokens_stem_nopunct))

#bagofwords = list(set(tokens_stem_nopunct))
#','.join(bagofwords)

text_merged2train = []
for id in train['id']:
    mov = train[train['id'] == id]
    keywords = list(mov['Keywords'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)[0]
    keywords_str = ' '.join(keywords)
    if pd.isnull(list(mov['overview'])[0]):
        overview = ''
    else:
        overview = list(mov['overview'])[0]
    if pd.isnull(list(mov['tagline'])[0]):
        tagline = ''
    else:
        tagline = list(mov['tagline'])[0]

#    text_merged = overview + ' | ' + tagline + ' | ' + keywords_str
    text_merged = keywords_str

    text_merged2train.append(text_merged)
#    tokens = word_tokenize(text_merged)
#    tokens_cleaned = [w for w in tokens if not w in stop_words]
#    tokens_stem = [ps.stem(w) for w in tokens_cleaned]
#    tokens_stem_nopunct = [w.lower() for w in tokens_stem if w.isalpha()]
#    bagofwords = list(set(tokens_stem_nopunct))
#    text_merged2train.append(bagofwords)

train['text_merged'] = text_merged2train




#check
#list(train[train['title'] == 'The Terminator']['text_merged'])




train['belongs_to_collection'].head(5)




train['belongs_to_collection'].apply(lambda x: 1 if x != {} else 0).value_counts()




train['belongs2coll_yn'] = train['belongs_to_collection'].apply(lambda x: len(x) if x != {} else 0)
train['homepage'].head(5)




train['homepage'].apply(lambda x: 1 if pd.isnull(x) == False else 0).value_counts()




train['homepage_yn'] = train['homepage'].apply(lambda x: 1 if pd.isnull(x) == False else 0)




train['imdb_id'].head(5)




' '.join(set(train['original_language'])), len(set(train['original_language']))




train['original_title'][0:5], train['title'][0:5]




list_languages = list(train['spoken_languages'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
languages = set([y for x in list_languages for y in x])
len(languages), Counter([i for j in list_languages for i in j]).most_common(5)




train['status'].apply(lambda x: 1 if x == 'Released' else 0).value_counts(), set(train['status'])




# strings to dictionaries
test = refmt_str2dict(test, dict_columns)




# One release date missing for the test set, for movie 'Jails, Hospitals & Hip-Hop'
# date '5/1/00' retrieved from https://www.imdb.com/title/tt0210130/
# but I don't want to use any external data wo will use dummy date
test.at[pd.isnull(test['release_date']), 'release_date'] = '1/1/11'
test['release_date'] = test['release_date'].apply(lambda x: fix_date(x))
test['release_date'] = pd.to_datetime(test['release_date'])

# create 'year', 'month', 'day', 'weekday', 'weekofyear', 'quarter' features
test = process_date(test)




text_merged2test = []
for id in test['id']:
    mov = test[test['id'] == id]
    keywords = list(mov['Keywords'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)[0]
    keywords_str = ' '.join(keywords)
    if pd.isnull(list(mov['overview'])[0]):
        overview = ''
    else:
        overview = list(mov['overview'])[0]
    if pd.isnull(list(mov['tagline'])[0]):
        tagline = ''
    else:
        tagline = list(mov['tagline'])[0]

#    text_merged = overview + ' | ' + tagline + ' | ' + keywords_str
    text_merged = keywords_str

    text_merged2test.append(text_merged)
#    tokens = word_tokenize(text_merged)
#    tokens_cleaned = [w for w in tokens if not w in stop_words]
#    tokens_stem = [ps.stem(w) for w in tokens_cleaned]
#    tokens_stem_nopunct = [w.lower() for w in tokens_stem if w.isalpha()]
#    bagofwords = list(set(tokens_stem_nopunct))
#    text_merged2test.append(bagofwords)
    
test['text_merged'] = text_merged2test




#list(test[test['title'] == 'Transcendence']['text_merged'])




test['belongs2coll_yn'] = test['belongs_to_collection'].apply(lambda x: len(x) if x != {} else 0)
test['homepage_yn'] = test['homepage'].apply(lambda x: 1 if pd.isnull(x) == False else 0)




yr_release_test = test['release_date_year'].value_counts().sort_index()
yr_budget_test = test.groupby(['release_date_year'])['budget'].mean()

plt.figure(figsize=(20,8))
plt.subplot(211)
plt.title('Movie count')
plt.plot(yr_release.index, yr_release.values)
plt.plot(yr_release_test.index, yr_release_test.values)
plt.subplot(212)
plt.plot(yr_budget.index, yr_budget.values)
plt.plot(yr_budget_test.index, yr_budget_test.values)
plt.title('$')
plt.subplots_adjust(hspace=0.3)
plt.show()




test[test['release_date_year'] == 1927]
#https://en.wikipedia.org/wiki/List_of_most_expensive_films
#Metropolis, the 1927 German film directed by Fritz Lang, often erroneously reported as having cost
#$200 million at the value of modern money. Metropolis cost $1.2–1.3 million at the time of its
#production, which would be about $12 million at 2009 prices




nrow = 2
# checking lead cast 'cast_id'] == 1
list_cast_INtest = list(test['cast'].apply(
    lambda x: [(i['name'], i['profile_path']) for i in x if i['cast_id'] == 1] if x != {} else []).values)
top_cast_INtest = Counter([y for x in list_cast_INtest for y in x]).most_common(7*nrow)

fig = plt.figure(figsize=(20, nrow*5))
k = 0
for i in top_cast_INtest:
    ax = fig.add_subplot(nrow, 7, k+1, xticks=[], yticks=[])
    img = Image.open(urlopen(TMDB_path + i[0][1]))
    plt.imshow(img)
    ax.set_title(i[0][0][0:22])
    k += 1




cols2drop = ['id','belongs_to_collection', 'homepage', 'imdb_id', 'original_title', 'overview',
            'poster_path', 'release_date', 'spoken_languages', 'status','tagline', 'Keywords']
train = train.drop(cols2drop, axis=1)
test = test.drop(cols2drop, axis=1)

#features that remain:
train.columns, test.columns




df4corr = train[['budget', 'popularity', 'runtime', 'release_date_year', 'release_date_month', 'release_date_day',                    'release_date_weekday', 'release_date_weekofyear', 'release_date_quarter', 'revenue']]




correlation = df4corr.corr()
plt.figure(figsize=(12, 12))  
sns.heatmap(correlation, annot=True, square=True, cmap='coolwarm')




_, axes = plt.subplots(2, 4, figsize=(20, 8))
sns.scatterplot(x = 'budget', y = 'revenue', data = train, marker="+", ax=axes[0,0])
sns.scatterplot(x = np.log1p(train['budget']), y = np.log1p(train['revenue']), marker="+", ax=axes[0,1])
sns.scatterplot(x = 'runtime', y = 'revenue', data = train, marker="+", ax=axes[0,2])
#sns.scatterplot(x = train['budget']/train['runtime'], y = train['revenue'], marker="+", ax=axes[0,2])
sns.scatterplot(x = 'popularity', y = 'revenue', data = train, marker="+", ax=axes[0,3])
#sns.scatterplot(x = train['budget']/train['popularity'], y = train['revenue'], marker="+", ax=axes[0,3])
#sns.scatterplot(x = np.where(train['popularity'] < 40, train['popularity'], 40), y = train['revenue'], \
#               marker="+", ax=axes[0,3])
sns.scatterplot(x = 'release_date_year', y = 'revenue', data = train, marker="+", ax=axes[1,0])
sns.stripplot(x = 'release_date_weekday', y = 'revenue', data = train, ax=axes[1,1])
sns.stripplot(x = 'release_date_month', y = 'revenue', data = train, ax=axes[1,2])




train['budget_yn'] = train['budget'].apply(lambda x: 0 if x == 0 else 1)
train['budget_per_year'] = train['budget']/train['release_date_year']
#train['budget_perRuntime'] = train['budget']/train['runtime']
#train['popularity_clipped'] = np.where(train['popularity'] < 40, train['popularity'], 40)




_, axes = plt.subplots(1, 3, figsize=(20, 8))
sns.stripplot(x = 'belongs2coll_yn', y = 'revenue', data = train, ax=axes[0])
sns.stripplot(x = 'homepage_yn', y = 'revenue', data = train, ax=axes[1])
sns.stripplot(x = 'budget_yn', y = 'revenue', data = train, ax=axes[2])




train['n_genres'] = train['genres'].apply(lambda x: len(x) if x != {} else 0)
train['n_production_companies'] = train['production_companies'].apply(lambda x: len(x) if x != {} else 0)
train['n_production_countries'] = train['production_countries'].apply(lambda x: len(x) if x != {} else 0)
train['n_cast'] = train['cast'].apply(lambda x: len(x) if x != {} else 0)
train['n_crew'] = train['crew'].apply(lambda x: len(x) if x != {} else 0)




_, axes = plt.subplots(2, 3, figsize=(20, 8))
sns.stripplot(x = 'n_genres', y = 'revenue', data = train, ax=axes[0,0])
sns.stripplot(x = 'n_production_companies', y = 'revenue', data = train, ax=axes[0,1])
sns.stripplot(x = 'n_production_countries', y = 'revenue', data = train, ax=axes[0,2])
sns.regplot(x = 'n_cast', y = 'revenue', data = train, marker='+', ax=axes[1,0])
sns.regplot(x = 'n_crew', y = 'revenue', data = train, marker='+', ax=axes[1,1])




train['genres_collapsed'] = train['genres'].apply(lambda x: ' '.                                                  join(sorted([i['name'] for i in x])) if x != {} else '')
for g in genres:
    train['genre_' + g] =  train['genres_collapsed'].apply(lambda x: 1 if g in x else 0)

keys_genre = ['genre_' + g for g in genres]




n_prodCompanies = 30
Counter([i for j in list_prodCompanies for i in j]).most_common(n_prodCompanies)




train['production_companies_collapsed'] = train['production_companies'].apply(lambda x: ' '.                                                    join(sorted([i['name'] for i in x])) if x != {} else '')
top_prodCompanies = [m[0] for m in Counter([i for j in list_prodCompanies for i in j]).most_common(n_prodCompanies)]
for comp in top_prodCompanies:
    train['production_company_' + comp] = train['production_companies_collapsed'].            apply(lambda x: 1 if comp in x else 0)
    
keys_production_company = ['production_company_' + comp for comp in top_prodCompanies]




n_prodCountries = 30
Counter([i for j in list_prodCountries for i in j]).most_common(n_prodCountries)




train['production_countries_collapsed'] = train['production_countries'].apply(lambda x: ' '.                                                    join(sorted([i['name'] for i in x])) if x != {} else '')
top_prodCountries = [m[0] for m in Counter([i for j in list_prodCountries for i in j]).most_common(n_prodCountries)]
for comp in top_prodCountries:
    train['production_country_' + comp] = train['production_countries_collapsed'].            apply(lambda x: 1 if comp in x else 0)

keys_production_country = ['production_country_' + comp for comp in top_prodCountries]




#train['EnglishLead_yn'] = train['original_language'].apply(lambda x: 1 if x == 'en' else 0)
top_languages = [m[0] for m in Counter(train['original_language']).most_common(5)]
for l in top_languages:
    train['language_' + l] = train['original_language'].            apply(lambda x: 1 if l in x else 0)

keys_language = ['language_' + l for l in top_languages]




# use lead actor for top_cast
n_lead = 50
list_lead = list(train['cast'].apply(lambda x: [i['name'] for i in x if i['cast_id'] == 1] if x != {} else []).values)
Counter([i for j in list_lead for i in j]).most_common(n_lead)




# different top leads in training and test sets - used combined set
fullset = pd.concat([train, test])

list_lead = list(fullset['cast'].apply(lambda x: [i['name'] for i in x if i['cast_id'] == 1] if x != {} else []).values)
Counter([i for j in list_lead for i in j]).most_common(n_lead)




# find top_lead even if not lead in movie
train['cast_collapsed'] = train['cast'].apply(lambda x: ' '.                                              join(sorted([i['name'] for i in x])) if x != {} else '')
top_lead = [m[0] for m in Counter([i for j in list_lead for i in j]).most_common(n_lead)]
for lead in top_lead:
    train['cast_' + lead] = train['cast_collapsed'].apply(lambda x: 1 if lead in x else 0)
    
keys_cast = ['cast_' + lead for lead in top_lead]




n_directors = 50
list_directors = list(train['crew'].apply(
    lambda x: [i['name'] for i in x if i['job'] == 'Director'] if x != {} else []).values)
Counter([i for j in list_directors for i in j]).most_common(n_directors)




#same as cast - use combined set
list_directors = list(fullset['crew'].apply(
    lambda x: [i['name'] for i in x if i['job'] == 'Director'] if x != {} else []).values)
Counter([i for j in list_directors for i in j]).most_common(n_directors)




train['directors_collapsed'] = train['crew'].apply(lambda x: ' '.        join(sorted([i['name'] for i in x if i['job'] == 'Director'])) if x != {} else '')
top_directors = [m[0] for m in Counter([i for j in list_directors for i in j]).most_common(n_directors)]
for d in top_directors:
    train['director_' + d] = train['directors_collapsed'].apply(lambda x: 1 if d in x else 0)
    
keys_director = ['director_' + d for d in top_directors]




# deprecated: was used when text transformation was already performed on 'text_merged'
#tokens_train = [word for l in train['text_merged'] for word in l]
#tokens_test = [word for l in test['text_merged'] for word in l]
#tokens = tokens_train + tokens_test
#lexicon = set(tokens)
#len(lexicon)

#lexicon10 = [m[0] for m in Counter(tokens).most_common(10)]
#lexicon10




vectorizer = TfidfVectorizer(
            analyzer = 'word',
            stop_words = 'english',
            ngram_range = (1, 2),
            min_df = 10,
            sublinear_tf = True)

overview_transf = vectorizer.fit_transform(train['text_merged'])
overview_transf




lexicon = list(vectorizer.vocabulary_.keys())
#lexicon




linreg = LinearRegression()
linreg.fit(overview_transf, train['revenue'])




top = 100   #500: 2.06288, 100 : 2.01548, 50: 2.01763 (test set result)
negcorrTop = np.argsort(linreg.coef_)[0:top]
poscorrTop = np.argsort(linreg.coef_)[len(linreg.coef_)-top:]




keywords_newPos = [lexicon[i] for i in poscorrTop]
keywords_newNeg = [lexicon[i] for i in negcorrTop]
keywords_new = keywords_newPos + keywords_newNeg
keywords_new[0:10]




for k in keywords_new:
    train['txt_' + k] =  train['text_merged'].apply(lambda x: 1 if k in x else 0)

keys_txt = ['txt_' + s for s in keywords_new]




test['budget_yn'] = test['budget'].apply(lambda x: 0 if x == 0 else 1)
test['budget_per_year'] = test['budget']/test['release_date_year']

test['n_genres'] = test['genres'].apply(lambda x: len(x) if x != {} else 0)
test['n_production_companies'] = test['production_companies'].apply(lambda x: len(x) if x != {} else 0)
test['n_production_countries'] = test['production_countries'].apply(lambda x: len(x) if x != {} else 0)
test['n_cast'] = test['cast'].apply(lambda x: len(x) if x != {} else 0)
test['n_crew'] = test['crew'].apply(lambda x: len(x) if x != {} else 0)

#test['EnglishLead_yn'] = test['original_language'].apply(lambda x: 1 if x == 'en' else 0)
for l in top_languages:
    test['language_' + l] = test['original_language'].            apply(lambda x: 1 if l in x else 0)

test['genres_collapsed'] = test['genres'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
for g in genres:
    test['genre_' + g] =  test['genres_collapsed'].apply(lambda x: 1 if g in x else 0)

test['production_companies_collapsed'] = test['production_companies'].apply(lambda x: ' '.                                                    join(sorted([i['name'] for i in x])) if x != {} else '')
for comp in top_prodCompanies:
    test['production_company_' + comp] = test['production_companies_collapsed'].              apply(lambda x: 1 if comp in x else 0)

test['production_countries_collapsed'] = test['production_countries'].apply(lambda x: ' '.                                                    join(sorted([i['name'] for i in x])) if x != {} else '')
for comp in top_prodCountries:
    test['production_country_' + comp] = test['production_countries_collapsed'].            apply(lambda x: 1 if comp in x else 0)

test['cast_collapsed'] = test['cast'].apply(lambda x: ' '.                                              join(sorted([i['name'] for i in x])) if x != {} else '')
for lead in top_lead:
    test['cast_' + lead] = test['cast_collapsed'].apply(lambda x: 1 if lead in x else 0)

test['directors_collapsed'] = test['crew'].apply(lambda x: ' '.        join(sorted([i['name'] for i in x if i['job'] == 'Director'])) if x != {} else '')
for d in top_directors:
    test['director_' + d] = test['directors_collapsed'].apply(lambda x: 1 if d in x else 0)
    
for k in keywords_new:
    test['txt_' + k] =  test['text_merged'].apply(lambda x: 1 if k in x else 0)




cols2drop = ['genres', 'production_companies', 'production_countries', 'genres_collapsed',
             'production_companies_collapsed', 'production_countries_collapsed', 'cast_collapsed', 
             'directors_collapsed', 'n_cast', 'n_crew']
train = train.drop(cols2drop, axis=1)
test = test.drop(cols2drop, axis=1)




def RMSE(y_obs, y_pred):
    n = len(y_obs)
    rmse = np.sqrt( 1/n*np.sum((y_pred-y_obs)**2) )
    return rmse

def RMSLE(y_obs, y_pred):
    n = len(y_obs)
    rmsle = np.sqrt( 1/n*np.sum((np.log(y_pred)-np.log(y_obs))**2) )
    return rmsle

def LinearRegression(X, y):
    intercept_term = np.ones(shape = y.shape)
    X = np.concatenate((intercept_term, X), 1)
    #closed-form solution:
    coeffs = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return coeffs




x = train['budget'].values
#x = train['budget_per_year'].values
X = np.reshape(x, (len(x),1))     #len(x) samples, 1 dimension
y = train['revenue'].values
y = np.reshape(y, (len(y),1))
np.shape(X), np.shape(y)




model_baseline_coeffs = LinearRegression(X, y)
model_baseline_pred_train = model_baseline_coeffs[0] + model_baseline_coeffs[1]*X
model_baseline_coeffs




plt.figure(figsize=(10,5))
plt.title('Linear regression (1d)')
plt.scatter(X, y)
plt.plot(X, model_baseline_pred_train, c = 'black')




#x=budget: 2.6656031635747532
#x=budget_per_date_year: 2.661467020660518
RMSLE(y, model_baseline_pred_train)




train_selected = train[['budget', 'popularity', 'runtime',
                        'release_date_year', 'release_date_weekday', 'release_date_month',
                        'budget_yn', 'belongs2coll_yn', 'homepage_yn',
                        'n_genres', 'n_production_companies', 'n_production_countries', 
                        'revenue']+keys_genre+keys_cast+keys_director+keys_production_company+
                       keys_production_country+keys_language]

#train_selected = train_selected.replace([np.inf, -np.inf], np.nan)
train_selected = train_selected.dropna(axis = 0)

X = train_selected.drop(['revenue'], axis=1)
y = train_selected['revenue']




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

model_RF = RandomForestRegressor(n_estimators = 100,
                                 max_depth = 20)
model_RF.fit(X_train, y_train)
model_RF_pred_valset = model_RF.predict(X_test)
RMSLE(y_test, model_RF_pred_valset)




model_RF.fit(X, y)
model_RF_pred_train = model_RF.predict(X)
RMSLE(y, model_RF_pred_train)




X_train, X_test, y_train, y_test = train_test_split(X, np.log1p(y), test_size=0.4)

model_CatBoost = CatBoostRegressor(silent=True)

model_CatBoost.fit(X_train, y_train)
model_CatBoost_pred_valset = model_CatBoost.predict(X_test)
RMSE(y_test, model_CatBoost_pred_valset)




model_CatBoost.fit(X, np.log1p(y))
model_CatBoost_pred_train = model_CatBoost.predict(X)
RMSE(np.log1p(y), model_CatBoost_pred_train)




X_train, X_test, y_train, y_test = train_test_split(X, np.log1p(y), test_size=0.4)

model_gboost = GradientBoostingRegressor()

model_gboost.fit(X_train, y_train)
model_gboost_pred_valset = model_gboost.predict(X_test)
RMSE(y_test, model_gboost_pred_valset)




model_gboost.fit(X, np.log1p(y))
model_gboost_pred_train = model_gboost.predict(X)
RMSE(np.log1p(y), model_gboost_pred_train)




model_ensemble_pred_train = (model_CatBoost_pred_train+model_gboost_pred_train)/2
RMSE(np.log1p(y), model_ensemble_pred_train)




x = test['budget'].values
X = np.reshape(x, (len(x),1))     #len(x) samples, 1 dimension
np.shape(X)




model_baseline_pred_test = model_baseline_coeffs[0] + model_baseline_coeffs[1]*X




submission = pd.read_csv('../input/sample_submission.csv')
submission['revenue'] = model_baseline_pred_test
submission.to_csv('submission_baseline.csv', index = False)




test_features = test[['budget', 'popularity', 'runtime',
                        'release_date_year', 'release_date_weekday', 'release_date_month',
                        'budget_yn', 'belongs2coll_yn', 'homepage_yn',
                        'n_genres', 'n_production_companies', 'n_production_countries']+
                     keys_genre+keys_cast+keys_director+keys_production_company+keys_production_country+
                    keys_language]

X = test_features
test_features.columns[test_features.isna().any()].tolist()




test_features['runtime'][test_features['runtime'].isna() == True]




test_features['runtime'][test_features['runtime'].isna() == True] = test_features['runtime'].median()




model_RF_pred_test = model_RF.predict(X)




submission = pd.read_csv('../input/sample_submission.csv')
submission['revenue'] = model_RF_pred_test
submission.to_csv('submission_RF.csv', index = False)




model_CatBoost_pred_test = model_CatBoost.predict(X)
model_CatBoost_pred_test = np.expm1(model_CatBoost_pred_test)




submission = pd.read_csv('../input/sample_submission.csv')
submission['revenue'] = model_CatBoost_pred_test
submission.to_csv('submission_CatBoost.csv', index = False)




model_gboost_pred_test = model_gboost.predict(X)
model_gboost_pred_test = np.expm1(model_gboost_pred_test)




submission = pd.read_csv('../input/sample_submission.csv')
submission['revenue'] = model_gboost_pred_test
submission.to_csv('submission_gboost.csv', index = False)




model_ensemble_pred_test = (model_CatBoost_pred_test + model_gboost_pred_test)/2




submission = pd.read_csv('../input/sample_submission.csv')
submission['revenue'] = model_ensemble_pred_test
submission.to_csv('submission_ensemble.csv', index = False)

