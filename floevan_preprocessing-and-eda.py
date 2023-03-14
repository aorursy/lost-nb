#!/usr/bin/env python
# coding: utf-8

# In[3]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

#!pip install chart_studio
#!pip install textstat

import numpy as np 
import pandas as pd 

# text processing libraries
import re
import string
import nltk
from nltk.corpus import stopwords
#import textstat
import random



# Visualisation libraries
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import plotly.graph_objs as go
import plotly.express as px
#import chart_studio.plotly as py
import plotly.figure_factory as ff
from plotly.offline import iplot



# sklearn 
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer



from collections import Counter

from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


from tqdm import tqdm

import os

import spacy
import random
from spacy.util import compounding
from spacy.util import minibatch


import warnings
warnings.filterwarnings("ignore")


# In[4]:


train = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')
test = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')
print('Training data shape: ', train.shape)
print('Testing data shape: ', test.shape)


# In[5]:


print("There are {} percentage of test data proportion compared to train data".format(round(test.shape[0]/train.shape[0]*100,2)))


# In[6]:


# First few rows of the training dataset
train.head()


# In[7]:


# First few rows of the testing dataset
test.head()


# In[8]:


train.info()


# In[9]:


train.isnull().sum()


# In[10]:


train.dropna(inplace = True)


# In[11]:


test.info()


# In[12]:


test.isnull().sum()


# In[13]:


print('Positive tweet example :', train[train['sentiment'] == 'positive']['text'].values[1])
print()
print('Neutral tweet example :', train[train['sentiment'] == 'neutral']['text'].values[1])
print()
print('Negative tweet example :', train[train['sentiment'] == 'negative']['text'].values[1])


# In[14]:


train['sentiment'].value_counts()


# In[15]:


train['sentiment'].value_counts(normalize = True)


# In[16]:


test['sentiment'].value_counts(normalize = True)


# In[17]:


def jaccard(str1,str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)/ (len(a) + len(b) - len(c)))


# In[18]:


results_jaccard=[]

for ind,row in train.iterrows():
    sentence1 = row.text
    sentence2 = row.selected_text

    jaccard_score = jaccard(sentence1,sentence2)
    results_jaccard.append([sentence1,sentence2,jaccard_score])


# In[19]:


jaccard = pd.DataFrame(results_jaccard, columns = ["text", "selected_text", "jaccard_score"])


# In[20]:


jaccard.head()


# In[21]:


train = train.merge(jaccard, how = 'outer')
train.head()


# In[22]:


train['Num_Word_ST'] = train['selected_text'].apply(lambda x: len(str(x).split()))


# In[23]:


train['Num_Word_Text'] = train['text'].apply(lambda x: len(str(x).split()))


# In[24]:


train['diff_in_word'] = train['Num_Word_Text'] - train['Num_Word_ST']


# In[25]:


train.head()


# In[26]:


plt.figure(figsize = (8,6))
plt.hist(train['Num_Word_ST'], bins = 30)
plt.title('Distribution of Number Of words for selected_text');


# In[27]:


plt.figure(figsize = (8,6))
plt.hist(train['Num_Word_Text'], bins = 30)
plt.title('Distribution of Number Of words for text');


# In[28]:


plt.figure(figsize = (8,6))
plt.hist(train['diff_in_word'], bins = 30)
plt.title('Distribution of Number Of words for diff_in_word');


# In[29]:


plt.figure(figsize = (8,6))
p = sns.kdeplot(train['Num_Word_ST'], shade = True, color = "r")
p = sns.kdeplot(train['Num_Word_Text'], shade = True, color = 'g')
plt.title('Kernel Distribution of Number Of words');


# In[30]:


plt.figure(figsize = (10,8))
sns.kdeplot(train['diff_in_word'], shade = True, color = 'b', legend = False)
plt.title('Kernel Distribution of diff_in_word');


# In[31]:


plt.figure(figsize = (10,8))
sns.kdeplot(train[train['sentiment'] == 'positive']['diff_in_word'], shade = True, color = 'b', legend = False )
plt.title('Kernel Distribution of diff_in_word for positive sentiment');


# In[32]:


plt.figure(figsize = (10,8))
sns.kdeplot(train[train['sentiment'] == 'negative']['diff_in_word'], shade = True, color = 'r', legend = False )
plt.title('Kernel Distribution of diff_in_word for negative sentiment');


# In[33]:


plt.figure(figsize = (10,8))
plt.hist(train[train['sentiment'] == 'neutral']['diff_in_word'], bins = 20)
plt.title('Distribution of diff_in_word for neutral sentiment');


# In[34]:


plt.figure(figsize = (10,8))
sns.kdeplot(train[train['sentiment'] == 'positive']['jaccard_score'], shade = True, color = 'r', legend = False).set_title('Distribution of jaccard_score for positive sentiment');


# In[35]:


plt.figure(figsize = (10,8))
sns.kdeplot(train[train['sentiment'] == 'negative']['jaccard_score'], shade = True, color = 'b', legend = False).set_title('Distribution of jaccard_score for negative sentiment');


# In[36]:


plt.figure(figsize = (10,8))
plt.hist(train[train['sentiment'] == 'neutral']['jaccard_score'], bins = 20)
plt.title('Distribution of jaccard_score for neutral sentiment');


# In[37]:


j = train[train['Num_Word_Text'] < 3]


# In[38]:


train.head()


# In[39]:


j.groupby('sentiment').mean()['jaccard_score']


# In[40]:


j[['text', 'selected_text']].head(15)


# In[41]:


stop_words = stopwords.words("english")
stop_words.extend(['im', 'u'])
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

def clean_text(text):
    text = text.lower() #make text lowercase and fill na
    text = re.sub('\[.*?\]', '', text) 
    text = re.sub('\\n', '',str(text))
    text = re.sub("\[\[User.*",'',str(text))
    text = re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",'',str(text))
    text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text) #remove hyperlinks
    text = re.sub(r'\:(.*?)\:', '', text) #remove emoticones
    text = re.sub(r'[\w\.-]+@[\w\.-]+', '', str(text)) #remove email
    text = re.sub(r'(?<=@)\w+', '', text) #remove @
    text = re.sub(r'[0-9]+', '', text) #remove numbers
    text = re.sub("[^A-Za-z0-9 ]", '', text) #remove non alphanumeric like ['@', '#', '.', '(', ')']
    text = re.sub(r'[!"\$%&\'()*+,\-.\/:;=#@?\[\\\]^_`{|}~]*', '', text) #remove punctuations from sentences
    text = re.sub('<.*?>+', '', str(text))
    text = re.sub('[%s]' % re.escape(string.punctuation), '', str(text))
    text = re.sub('\w*\d\w*', '', str(text))
    text = tokenizer.tokenize(text)
    text = [word for word in text if not word in stop_words]
    final_text = ' '.join( [w for w in text if len(w)>1] ) #remove word with one letter
    return final_text


# In[42]:


train['text_clean'] = train['text'].apply(lambda x: clean_text(x))
train['selected_text_clean'] = train['selected_text'].apply(lambda x : clean_text(x))


# In[43]:


train[['text', 'text_clean']].head()


# In[44]:


train[['selected_text', 'selected_text_clean']].head()


# In[45]:


train['list'] = train['selected_text_clean'].apply(lambda x: str(x).split())

top = Counter([item for sublist in train['list'] for item in sublist])

mostcommon = pd.DataFrame(top.most_common(20))
mostcommon.columns = ['Common Word', 'Count']
mostcommon.head(20)


# In[46]:


train['list'] = train['text_clean'].apply(lambda x: str(x).split())


def top(corpus, n = None):
    top = Counter([item for sublist in corpus['list'] for item in sublist])
    mostcommon = pd.DataFrame(top.most_common(20))
    mostcommon.columns = ['Common Word', 'Count']
    return mostcommon.head(20)

top(train)


# In[47]:


positive = train[train['sentiment'] == 'positive']
negative = train[train['sentiment'] == 'negative']
neutral = train[train['sentiment'] == 'neutral']


# In[48]:


print('The most common word in positive tweets')
top(positive)


# In[49]:


print('The most common word in negative tweets')
top(negative)


# In[50]:


print('The most common word in neutral tweets')
top(neutral)


# In[51]:


raw_text = [word for word_list in train['list'] for word in word_list]


# In[52]:


def words_unique(sentiment,numwords,raw_words):
    '''
    Input:
        segment - Segment category (ex. 'Neutral');
        numwords - how many specific words do you want to see in the final result; 
        raw_words - list  for item in train_data[train_data.segments == segments]['temp_list1']:
    Output: 
        dataframe giving information about the name of the specific ingredient and how many times it occurs in the chosen cuisine (in descending order based on their counts)..

    '''
    allother = []
    for item in train[train.sentiment != sentiment]['list']:
        for word in item:
            allother .append(word)
    allother  = list(set(allother ))
    
    specificnonly = [x for x in raw_text if x not in allother]
    
    mycounter = Counter()
    
    for item in train[train.sentiment == sentiment]['list']:
        for word in item:
            mycounter[word] += 1
    keep = list(specificnonly)
    
    for word in list(mycounter):
        if word not in keep:
            del mycounter[word]
    
    Unique_words = pd.DataFrame(mycounter.most_common(numwords), columns = ['words','count'])
    
    return Unique_words


# In[53]:


Unique_Positive= words_unique('positive', 20, raw_text)
print("The top 20 unique words in Positive Tweets are:")
Unique_Positive.style.background_gradient(cmap='Greens')


# In[54]:


Unique_Negative= words_unique('negative', 10, raw_text)
print("The top 10 unique words in Negative Tweets are:")
Unique_Negative.style.background_gradient(cmap='Reds')


# In[55]:


Unique_Neutral= words_unique('neutral', 10, raw_text)
print("The top 10 unique words in Neutral Tweets are:")
Unique_Neutral.style.background_gradient(cmap='Oranges')


# In[56]:


wc = WordCloud(stopwords = stop_words)
plt.figure(figsize = (18,12))
wc.generate(str(positive['text_clean']))
plt.imshow(wc)
plt.title('WordCloud of positive tweets');


# In[57]:


wc = WordCloud(stopwords = stop_words)
plt.figure(figsize = (18,12))
wc.generate(str(neutral['text_clean']))
plt.imshow(wc)
plt.title('WordCloud of neutral tweets');


# In[58]:


wc = WordCloud(stopwords = stop_words)
plt.figure(figsize = (18,12))
wc.generate(str(negative['text_clean']))
plt.imshow(wc)
plt.title('WordCloud of negative tweets');


# In[59]:


def get_top_n_gram(corpus,ngram_range,n=None):
    vec = CountVectorizer(ngram_range=ngram_range,stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


# In[60]:


positive_bigram = get_top_n_gram(positive['text_clean'], (2,2), 20)
negative_bigram = get_top_n_gram(negative['text_clean'], (2,2), 20)
neutral_bigram = get_top_n_gram(neutral['text_clean'], (2,2), 20)


# In[61]:


def process(corpus):
    corpus = pd.DataFrame(corpus, columns= ['Text', 'count']).sort_values('count', ascending = True)
    return corpus


# In[62]:


positive_bigram = process(positive_bigram)
negative_bigram = process(negative_bigram)
neutral_bigram = process(neutral_bigram)


# In[63]:


plt.figure(figsize = (10,8))
plt.barh(positive_bigram['Text'], positive_bigram['count'])
plt.title('Top 20 Bigrams in positive text');


# In[64]:


plt.figure(figsize = (10,8))
plt.barh(negative_bigram['Text'], negative_bigram['count'])
plt.title('Top 20 Bigrams in negative text');


# In[65]:


plt.figure(figsize = (10,8))
plt.barh(neutral_bigram['Text'], neutral_bigram['count'])
plt.title('Top 20 Bigrams in neutral text');


# In[66]:


positive_trigram = get_top_n_gram(positive['text_clean'], (3,3), 20)
negative_trigram = get_top_n_gram(negative['text_clean'], (3,3), 20)
neutral_trigram = get_top_n_gram(neutral['text_clean'], (3,3), 20)


# In[67]:


positive_trigram = process(positive_trigram)
negative_trigram = process(negative_trigram)
neutral_trigram = process(neutral_trigram)


# In[68]:


plt.figure(figsize = (10,8))
plt.barh(positive_trigram['Text'], positive_trigram['count'])
plt.title('Top 20 Trigrams in positive text');


# In[69]:


plt.figure(figsize = (10,8))
plt.barh(negative_trigram['Text'], negative_trigram['count'])
plt.title('Top 20 Trigrams in negative text');


# In[70]:


plt.figure(figsize = (10,8))
plt.barh(neutral_trigram['Text'], neutral_trigram['count'])
plt.title('Top 20 Trigrams in neutral text');


# In[71]:


get_ipython().run_cell_magic('HTML', '', '<iframe width="560" height="315" src="https://www.youtube.com/embed/XaQ0CBlQ4cY" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')


# In[ ]:




