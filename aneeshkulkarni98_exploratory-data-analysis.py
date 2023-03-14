#!/usr/bin/env python
# coding: utf-8



get_ipython().run_line_magic('cd', '../input/tmdb-box-office-prediction')




import os
get_ipython().system('unzip train.csv.zip')
get_ipython().system('unzip test.csv.zip')




import pandas as pd
import numpy as np
import ast
df_train = pd.read_csv('train.csv')
for i, e in enumerate(df_train['cast'][:5]):
  print(ast.literal_eval(e)[0]['character'])




df_train.head()




import matplotlib.pyplot as plt
log_revenue = np.log(df_train['revenue'])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 5))
fig.subplots_adjust(wspace = 0.4, hspace = 0.4)
ax1.scatter(df_train['budget'], df_train['revenue'])
ax1.set_xlabel('Budget')
ax1.set_ylabel('Revenue')
ax2.scatter(df_train['budget'], log_revenue)
ax2.set_xlabel('Budget')
ax2.set_ylabel('Log Revenue')




fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 5))
fig.subplots_adjust(wspace = 0.4, hspace = 0.4)
ax1.scatter(df_train['popularity'], df_train['revenue'])
ax1.set_xlabel('Popularity')
ax1.set_ylabel('Revenue')
ax2.scatter(df_train['popularity'], log_revenue)
ax2.set_xlabel('Popularity')
ax2.set_ylabel('Log Revenue')




import seaborn as sns
plt.figure(figsize=(10, 10))
sns.boxplot(x = 'original_language', y = 'revenue', data = df_train, whis = 0.0, showfliers = False)




df_train[['release_month', 'release_day', 'release_year']] = df_train['release_date'].str.split('/', expand = True)                                                                                                .replace(np.nan, -1)




df_train['release_month'] = df_train['release_month'].astype(int)
df_train['release_day'] = df_train['release_day'].astype(int)
df_train['release_year'] = df_train['release_year'].astype(int)
df_train.loc[(df_train['release_year'] <= 19) & (df_train['release_year'] < 100), "release_year"] += 2000
df_train.loc[(df_train['release_year'] > 19)  & (df_train['release_year'] < 100), "release_year"] += 1900                                                                     




indices = list([x - 1 for x in df_train.release_month.value_counts().index])
indices




df_train.release_month.value_counts().sort_index()




import plotly.graph_objs as go
months = np.array(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
# plt.bar(df_train['release_month'].value_counts().index, df_train['release_month'].value_counts().values,
#         color = 'red')
monthly_data = df_train.release_month.value_counts().sort_index()
idx = [x - 1 for x in monthly_data.index]
data = go.Bar(x = months[idx],
                y = monthly_data.values,
                marker = dict(color = monthly_data.values,
                             line=dict(color='rgb(0,0,0)',width=1.5))
                )
fig = go.Figure(data = data)
fig.update_layout(title = 'Revenue Vs Release Month', xaxis_title = 'Release Month', yaxis_title = 'Revenue')
fig.show()




yearly_data = df_train.release_year.value_counts().sort_index()

data = go.Bar(x = yearly_data.index,
                y = yearly_data.values,
                marker = dict(color = yearly_data.values,
                             line=dict(color='rgb(0,0,0)',width=1.5))
                )
fig = go.Figure(data = data)
fig.update_layout(xaxis_tickangle = -90, title = 'Revenue Vs Year', xaxis_title = 'Year', yaxis_title = 'Revenue')
fig.show()




df_train[df_train['status'] == 'Rumored']




from collections import Counter
def extract_values(text):
  values = []
  for list_of_values in text:
    if list_of_values != '':
      list_ = ast.literal_eval(list_of_values)
      for element in list_:
        value = element['name']
        values.append(value)
  count_values = Counter(values)
  return count_values

count_genres = extract_values(df_train['genres'].replace(np.nan, ""))




x, y = zip(*count_genres.items())




data = go.Bar(x = x,
              y = y,
              marker = dict(color = 'rgb(255, 255, 0, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5))
              )
fig = go.Figure(data = data)
fig.show()




# CHECKING FOR NULL VALUES
nrows = df_train.shape[0]
null_check = df_train.isnull().sum()




pct_null = pd.DataFrame(null_check.sort_values(ascending = False) * 100 / nrows)




pct_null




count_prod_comp = extract_values(df_train['production_countries'].replace(np.nan, ""))
x, y = zip(*count_prod_comp.items())
data = go.Bar(x = x,
              y = y,
              marker = dict(color = 'rgb(255, 255, 0, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5))
              )
fig = go.Figure(data = data)
fig.update_layout(width = 1000, height = 1000, xaxis = dict(tickmode = 'array',
                           tickvals = list(range(0, len(x))), 
                           ticktext = x))
fig.show()




year = df_train['release_year']
revenue = df_train.groupby('release_year')["revenue"].aggregate('mean')
data = go.Scatter(x=revenue.index, y=revenue.values,
                    mode='lines',
                    name='lines')
fig = go.Figure(data)
fig.update_layout(title = 'Year Vs Average Revenue',
                  xaxis_title = 'Year',
                  yaxis_title = 'Average Revenue')




year = df_train['release_year']
revenue = df_train.groupby('release_year')["runtime"].aggregate('mean')
data = go.Scatter(x=revenue.index, y=revenue.values,
                    mode='lines+markers',
                    name='lines')
fig = go.Figure(data)
fig.update_layout(title = 'Year Vs Runtime',
                  xaxis_title = 'Year',
                  yaxis_title = 'Runtime')




revenue = df_train.groupby('release_month')["revenue"].aggregate('mean')
data = go.Bar(x = months[np.array(revenue.index) - 1],
              y = revenue.values,
              marker = dict(color = 'rgba(255, 255, 0, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5))
              )
fig.update_layout(title = 'Month Vs Average Revenue',
                  xaxis_title = 'Month',
                  yaxis_title = 'Average Revenue')
fig = go.Figure(data = data)
fig.show()




count_prod_companies = extract_values(df_train['production_companies'].replace(np.nan, ""))
print(pd.DataFrame({'Name':list(count_prod_companies.keys()), 'No. of Movies Produced':list(count_prod_companies.values())}).head())




df_train['homepage'].isnull().value_counts()




import plotly.express as px
import seaborn as sns
slice_data = df_train[['homepage', 'revenue']]
slice_data.loc[~slice_data['homepage'].isnull(), 'homepage'] = "With Homepage"
slice_data.loc[slice_data['homepage'].isnull(), 'homepage'] = "No Homepage"
fig = px.box(slice_data, x = "revenue", y = "homepage", orientation = 'h', points = 'suspectedoutliers')
fig.show()




count_keywords = extract_values(df_train['Keywords'].replace(np.nan, ""))
vals = pd.DataFrame({'Keywords':list(count_keywords.keys()), 'Counts':list(count_keywords.values())})
### Looking at the top keywords in the film ### 
vals = vals.sort_values('Counts', ascending = False)
data = go.Bar(x = vals['Keywords'][:20],
              y = vals['Counts'][:20],
              marker = dict(color = 'rgba(255, 0, 0, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5))
              )
fig = go.Figure(data = data)
fig.update_layout(title = 'Top 20 keywords',
                  xaxis_title = 'Keywords',
                  yaxis_title = 'Counts')
fig.show()




count_spoken_languages = extract_values(df_train['spoken_languages'].replace(np.nan, ""))
vals = pd.DataFrame({'Languages':list(count_spoken_languages.keys()), 'Counts':list(count_spoken_languages.values())})
### Looking at the top keywords in the film ### 
vals = vals.sort_values('Counts', ascending = False)
data = go.Bar(x = vals['Languages'][1:20],
              y = vals['Counts'][1:20],
              marker = dict(color = 'rgba(255, 0, 0, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5))
              )
fig = go.Figure(data = data)
fig.update_layout(title = 'Language Films',
                  xaxis_title = 'Languages',
                  yaxis_title = 'Counts')
fig.show()




get_ipython().system('pip install googletrans')




from googletrans import Translator
translator = Translator()
count_spoken_languages = extract_values(df_train['spoken_languages'].replace(np.nan, ""))

vals = pd.DataFrame({'Languages':list(count_spoken_languages.keys()), 'Counts':list(count_spoken_languages.values())})
### Looking at the top keywords in the film ### 
vals = vals.sort_values('Counts', ascending = False)
for i in range(vals.shape[0]):
  vals.loc[i, 'Languages'] = translator.translate(vals.loc[i, 'Languages'], dest = 'en').text
data = go.Bar(x = vals['Languages'][1:20],
              y = vals['Counts'][1:20],
              marker = dict(color = 'rgba(255, 0, 0, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5))
              )
fig = go.Figure(data = data)
fig.update_layout(title = 'Language Films',
                  xaxis_title = 'Languages',
                  yaxis_title = 'Counts')
fig.show()




df_train['revenue'].hist()




### NUMBER OF CAST VS REVENUE ###
def get_cast_length(row):
  l = ast.literal_eval(row)
  return len(l)
from sklearn.impute import SimpleImputer
si = SimpleImputer(fill_value = "[]")
df_train['cast'] = df_train['cast'].replace(np.nan, "[]")
df_train["cast"] = df_train['cast'].replace("","[]")




df_train["num_cast"] = df_train['cast'].apply(lambda x: get_cast_length(x))
df_train.loc[:,['num_cast','revenue']]




from plotly.subplots import make_subplots
data = go.Scatter(x = df_train['num_cast'], y = np.log(df_train['revenue']), mode = 'markers',
                  marker = dict(color = 'rgba(255, 0, 0, 0.5)'))
fig = go.Figure(data = data)
params = {'title':'No. Of Cast Vs Revenue', 'xaxis_title':'num_cast', 'yaxis_title':'Log of Revenue'}
fig.update_layout(**params)
fig.show()




agg_data = df_train.groupby('original_language').aggregate({'revenue':np.mean})




langs = df_train['original_language'].unique()
fig = go.Figure()
for lang in langs:
  x = np.log(df_train.loc[df_train['original_language'] == lang, 'revenue'])
  name = lang
  fig.add_trace(go.Box(x = x,
                name = name,
                ))
fig.update_traces(orientation = 'h')
params = {'xaxis_title':'Revenue', 'yaxis_title':'Original Language', 'width':1000, 'height':1000}
fig.update_layout(**params)
fig.show()




from collections import defaultdict
genre_revenue_data = defaultdict(list)
for index, row in df_train.iterrows():
  try:
    genres = ast.literal_eval(row['genres'])
    for genre in genres:
      genre_revenue_data[genre['name']].append(row['revenue'])
  except ValueError:
    continue




genre_revenue_data.keys()




fig = go.Figure()
for genre in genre_revenue_data.keys():
  x = genre_revenue_data[genre]
  name = genre
  fig.add_trace(go.Box(x = x,
                name = name,
                ))
fig.update_traces(orientation = 'h')
params = {'xaxis_title':'Revenue', 'yaxis_title':'Genre','width':1000,'height':1000}
fig.update_layout(**params)
fig.show()




count_prod_companies.most_common(15)




x, y = zip(*count_prod_companies.most_common(10))
data = go.Bar(x = x,
              y = y,
              marker = dict(color = 'rgb(255, 0, 0)', line = dict(color = 'rgb(0,0,0)', width = 1)))
fig = go.Figure(data = data)
fig.show()




from collections import defaultdict
genre_popularity_data = defaultdict(list)
for index, row in df_train.iterrows():
  try:
    genres = ast.literal_eval(row['genres'])
    for genre in genres:
      genre_popularity_data[genre['name']].append(row['popularity'])
  except ValueError:
    continue




fig = go.Figure()
for genre in genre_popularity_data.keys():
  x = genre_popularity_data[genre]
  name = genre
  fig.add_trace(go.Box(x = x,
                name = name, boxpoints = False
                ))
fig.update_traces(orientation = 'h')
params = {'xaxis_title':'Popularity', 'yaxis_title':'Genre', 'width':1000, 'height':1000}
fig.update_layout(**params)
fig.show()




q1, q3 = np.percentile(df_train['popularity'].sort_values(),[25,75])
iqr = q3 - q1
lower_range = q1 - (1.5 * iqr)
upper_range = q3 + (1.5 * iqr)
slice_data = df_train[['original_language', 'popularity']]
slice_data.loc[slice_data['popularity'] > upper_range, 'popularity'] = upper_range
slice_data.loc[slice_data['popularity'] < lower_range, 'popularity'] = lower_range
px.box(slice_data, title = 'Original Language vs Popularity' ,y = 'original_language', x = 'popularity', 
       orientation = 'h',width = 1000, height = 1000)




df_train.head()




df_train['production_companies'].isnull().value_counts()




company_revenue_data = defaultdict(list)
for index, row in df_train.iterrows():
  try:
    companies = ast.literal_eval(row['production_companies'])
    for company in companies:
      company_revenue_data[company['name']].append(row['revenue'])
  except ValueError:
    continue




company_revenue_mean = defaultdict(int)
for company in company_revenue_data.keys():
  company_revenue_mean[company] = np.mean(company_revenue_data[company])




company_revenue_mean = pd.Series(company_revenue_mean).sort_values(ascending = False).head(20)
data = go.Bar(x = company_revenue_mean.index,
              y = company_revenue_mean.values,
              marker = dict(color = 'rgb(255, 0, 0)', line = dict(color = 'rgb(0,0,0)', width = 1)))
fig = go.Figure(data = data)
params = {'title':'Top 20 Companies based on their average revenues',
          'xaxis_title':'Company','yaxis_title':'Revenue'}
fig.update_layout(**params)
fig.show()




country_revenue_data = defaultdict(list)
for index, row in df_train.iterrows():
  try:
    countries = ast.literal_eval(row['production_countries'])
    for country in countries:
      country_revenue_data[country['name']].append(row['revenue'])
  except ValueError:
    continue
country_revenue_mean = defaultdict(int)
for country in country_revenue_data.keys():
  country_revenue_mean[country] = np.mean(country_revenue_data[country])
country_revenue_mean = pd.Series(country_revenue_mean).sort_values(ascending = False).head(20)
data = go.Bar(x = country_revenue_mean.index,
              y = country_revenue_mean.values,
              marker = dict(color = 'rgb(255, 0, 0)', line = dict(color = 'rgb(0,0,0)', width = 1)))
fig = go.Figure(data = data)
params = {'title':'Top 20 Countries based on their average revenues',
          'xaxis_title':'Country','yaxis_title':'Revenue'}
fig.update_layout(**params)
fig.show()

