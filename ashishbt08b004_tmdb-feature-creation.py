#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


import pandas as pd
data = pd.read_csv('/kaggle/input/tmdb-box-office-prediction/train.csv')
data.head()


# In[3]:


import ast
movie_genres = []
a=[]
data['genres'].fillna("",inplace=True)
for i in range(len(data)):
    if data['genres'][i] != "":
        genre = [x['name'] for x in ast.literal_eval(data['genres'][i])]
    else:
        genre = []
    movie_genres.append(genre)
print(movie_genres)


# In[4]:


length = []
for j in movie_genres:
    length.append(len(j))
pd.Series(length).value_counts()


# In[5]:


k = []
for j in movie_genres:
    k += j
genre_list = list(set(k))
genre_list


# In[6]:


#sum[1 if 'Action' in j else 0 for j in movie_genres]
m = []
ct = {}
dummy_v ={}
for k in genre_list:
    m = []
    for j in movie_genres:
        if k in j:
            m.append(int(1))
        else:
            m.append(int(0))
    dummy_v[k] = m
    ct[k] = sum(m)
ct


# In[7]:


for key, value in sorted(ct.items(), key=lambda kv: kv[1], reverse=True):
    print("%s: %s" % (key, value))


# In[8]:


for key in sorted(ct.keys()):
    print("%s: %s" % (key, ct[key]))


# In[9]:


import ast
keywords = []
a=[]
data['Keywords'].fillna("",inplace=True)
for i in range(len(data)):
    if data['Keywords'][i] != "":
        keyword = [x['name'] for x in ast.literal_eval(data['Keywords'][i])]
    else:
        keyword = []
    keywords.append(keyword)
print(keywords)


# In[10]:


k = []
for j in keywords:
    k += j
keywords_list = list(set(k))
keywords_list


# In[11]:


ct_k = {}
dummy_v ={}
for k in keywords_list:
    m = []
    for j in keywords:
        if k in j:
            m.append(int(1))
        else:
            m.append(int(0))
    #dummy_v[k] = m
    ct_k[k] = sum(m)
ct_k


# In[12]:


length = []
for j in keywords:
    length.append(len(j))
pd.Series(length).value_counts()


# In[13]:


for key, value in sorted(ct_k.items(), key=lambda kv: kv[1], reverse=True):
    print("%s: %s" % (key, value))


# In[14]:


import ast
lans = []
a=[]
data['spoken_languages'].fillna("",inplace=True)
for i in range(len(data)):
    if data['spoken_languages'][i] != "":
        lan = [x['name'] for x in ast.literal_eval(data['spoken_languages'][i])]
    else:
        lan = []
    lans.append(lan)
print(lans)


# In[15]:


k = []
for j in lans:
    k += j
lans_list = list(set(k))
lans_list


# In[16]:


ct_l = {}
dummy_v ={}
for k in lans_list:
    m = []
    for j in lans:
        if k in j:
            m.append(int(1))
        else:
            m.append(int(0))
    #dummy_v[k] = m
    ct_l[k] = sum(m)
ct_l


# In[17]:


length = []
for j in lans:
    length.append(len(j))
#pd.Series(length).value_counts()


# In[18]:


import matplotlib.pyplot as plt
from wordcloud import WordCloud

text = ' '.join([i for j in movie_genres for i in j])
wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,
                      width=5000, height=3000).generate(text)
plt.imshow(wordcloud)
plt.title('Top genres')
plt.axis("off")
plt.show()


# In[19]:


plt.figure(figsize = (16, 12))
text = ' '.join(['_'.join(i.split(' ')) for j in keywords for i in j])
wordcloud = WordCloud(max_font_size=None, background_color='black', collocations=False,
                      width=1200, height=1000).generate(text)
plt.imshow(wordcloud)
plt.title('Top keywords')
plt.axis("off")
plt.show()


# In[20]:


[len(j) for j in data['crew']


# In[21]:


crews = []
data['crew'].fillna("",inplace=True)
for i in range(len(data)):
    if data['crew'][i] != "":
        crew = [x['name'] for x in ast.literal_eval(data['crew'][i])]
    else:
        crew = []
    crews.append(crew)
print(movie_genres)


# In[22]:


k = []
for j in crews:
    k += j
a = pd.Series(k).value_counts()[:16]
top_crew = list(a.index)
top_crew


# In[23]:


pd.Series([len(j) for j in crews]).value_counts()


# In[24]:


crews_gender = []
data['crew'].fillna("",inplace=True)
for i in range(len(data)):
    if data['crew'][i] != "":
        crew_gender = [x['gender'] for x in ast.literal_eval(data['crew'][i])]
    else:
        crew_gender = []
    crews_gender.append(crew_gender)
print(movie_genres)


# In[25]:


k = []
for j in crews_gender:
    k += j
pd.Series(k).value_counts()


# In[26]:


male_count = []
male_pc = []
for i in range(len(crews_gender)):
    counter = 0
    for j in crews_gender[i]:
            if j==1:
                counter += 1
    male_count.append(counter)
    #male_pc.append(counter/len(crews_gender[i]))
male_count    
                

