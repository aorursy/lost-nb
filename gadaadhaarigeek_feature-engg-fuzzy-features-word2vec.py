#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


get_ipython().system('pip install distance')


# In[3]:


import zipfile
data = pd.read_csv(zipfile.ZipFile("/kaggle/input/quora-question-pairs/train.csv.zip").open("train.csv")).head(2000)

data.head()


# In[4]:


data = data.astype({"question1": str, "question2":str})


# In[5]:


# Intial level preprocessing

# Fill null values
data.fillna(" ")

# freq of each questions in q1 and q2, length, number of words
# number of common_words, unique words total, common_share
# Sum/difference of freqs of each question
data['freq_qid1'] = data.groupby('qid1')['qid1'].transform('count') 
data['freq_qid2'] = data.groupby('qid2')['qid2'].transform('count')
data['q1len'] = data['question1'].str.len() 
data['q2len'] = data['question2'].str.len()
data['q1_n_words'] = data['question1'].apply(lambda row: len(row.split(" ")))
data['q2_n_words'] = data['question2'].apply(lambda row: len(row.split(" ")))

def normalized_word_Common(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
    return 1.0 * len(w1 & w2)
data['word_common'] = data.apply(normalized_word_Common, axis=1)

def normalized_word_Total(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
    return 1.0 * (len(w1) + len(w2))
data['word_total'] = data.apply(normalized_word_Total, axis=1)

def normalized_word_share(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
    return 1.0 * len(w1 & w2)/(len(w1) + len(w2))
data['word_share'] = data.apply(normalized_word_share, axis=1)

data['freq_q1+q2'] = data['freq_qid1'] + data['freq_qid2']
data['freq_q1-q2'] = abs(data['freq_qid1'] - data['freq_qid2'])


# In[6]:


# Feature Engineered dataframe 
data.head(2)


# In[7]:


# To see the words with special chars
# Take a peek at the data
# x = data.head(100)
# def get_special_words(row):
#     list_ques = row["question1"].strip().lower().split(" ")
#     for word in list_ques:
#         if word.isalpha() == False:
#             pass
#             print(word)

# x.apply(get_special_words, axis=1)


# In[8]:


from nltk.corpus import stopwords
SAFE_DIV = .0001
STOP_WORDS = stopwords.words("english")

def preprocess(x):
    x = str(x).lower()
    x = x.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'").replace("won't", "will not")     .replace("cannot", "can not").replace("can't", "can not").replace("n't", "not").replace("what's", "what is").replace("it's", "it is")     .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are").replace("he's", "he is").replace("she's", "she is").replace("'s", " own")     .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ").replace("€", " euro ").replace("'ll", " will")
    x = re.sub(r"([0-9]+)000000", r"\1m", x)
    x = re.sub(r"([0-9]+)000", r"\1k", x)
    
    porter = PorterStemmer()
    pattern = re.compile("\W")
    
    if type(x) == type(""):
        x = re.sub(pattern, ' ', x)
        
    if type(x) == type(""):
        x = porter.stem(x)
        example1 = BeautifulSoup(x)
        x = example1.get_text()
        
    return x


# In[9]:


def get_token_features(q1, q2):
    token_features = [0.0] * 10
    
    # Converting the sentence into tokens
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features
    
    # get the non-stop words in questions
    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])
    
    # Get the stopwords in questions
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    
    # Get the common non-stopwords from question pair
    common_word_count = len(q1_words.intersection(q2_words))
    
    # Get the common stopwords count
    common_stop_count = len(q1_stops.intersection(q2_stops))
    
    # Get the common tokens from question pair
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))
    
    # common word count - min
    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    
    # common word count - max
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    
    # common stop count - min
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    
    # common stop count - max
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    
    # common token count - min
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    
    # common token count - max
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)    
    
    # whether last words of both the questions are same or not ?
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    
    # first word of both the questions are same or not
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])
    
    # absolute differen b/w the number of tokens in both the questions
    token_features[8] = abs(len(q1_tokens) - len(q2_tokens))
    
    # average number of tokens in both the questions
    token_features[9] = (len(q1_tokens) + len(q2_tokens))/2
    
    return token_features

def get_longest_common_substr_ratio(a, b):
    strs = list(distance.lcsubstrings(a, b))
    if len(strs) == 0:
        return 0
    else:
        return len(strs[0]) / (min(len(a), len(b)) + 1)

def extract_features(df):
    print("Preprocess questions...")
    df["question1"] = df["question1"].fillna("").apply(preprocess)
    df["question2"] = df["question2"].fillna("").apply(preprocess)
    
    print("Token features...")
    
    token_features = df.apply(lambda x: get_token_features(x["question1"], x["question2"]), axis=1)
    
    df["cwc_min"]       = list(map(lambda x: x[0], token_features))
    df["cwc_max"]       = list(map(lambda x: x[1], token_features))
    df["csc_min"]       = list(map(lambda x: x[2], token_features))
    df["csc_max"]       = list(map(lambda x: x[3], token_features))
    df["ctc_min"]       = list(map(lambda x: x[4], token_features))
    df["ctc_max"]       = list(map(lambda x: x[5], token_features))
    df["last_word_eq"]  = list(map(lambda x: x[6], token_features))
    df["first_word_eq"] = list(map(lambda x: x[7], token_features))
    df["abs_len_diff"]  = list(map(lambda x: x[8], token_features))
    df["mean_len"]      = list(map(lambda x: x[9], token_features))
    
    #Computing Fuzzy Features and Merging with Dataset
    
    # http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
    # https://stackoverflow.com/questions/31806695/when-to-use-which-fuzz-function-to-compare-2-strings
    # https://github.com/seatgeek/fuzzywuzzy    
    
    print("Creating fuzzy features... ")
    df["token_set_ratio"]       = df.apply(lambda x: fuzz.token_set_ratio(x["question1"], x["question2"]), axis=1)
    # The token sort approach involves tokenizing the string in question, sorting the tokens alphabetically, and 
    # then joining them back into a string We then compare the transformed strings with a simple ratio().
    df["token_sort_ratio"]      = df.apply(lambda x: fuzz.token_sort_ratio(x["question1"], x["question2"]), axis=1)
    df["fuzz_ratio"]            = df.apply(lambda x: fuzz.QRatio(x["question1"], x["question2"]), axis=1)
    df["fuzz_partial_ratio"]    = df.apply(lambda x: fuzz.partial_ratio(x["question1"], x["question2"]), axis=1)
    df["longest_substr_ratio"]  = df.apply(lambda x: get_longest_common_substr_ratio(x["question1"], x["question2"]), axis=1)    
    
    return df


# In[10]:


import re
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
import distance
import timeit
start = timeit.default_timer()
data = extract_features(data)
stop = timeit.default_timer()
print("Time elapsed: ", stop-start)


# In[11]:


data.head(2)


# In[12]:


data_duplicate = data[data["is_duplicate"] == 1]
data_nonduplicate = data[data["is_duplicate"] == 0]

# Converting 2d array of q1 and q2 and flatten the array: like {{1,2},{3,4}} to {1,2,3,4}
p = np.dstack([data_duplicate["question1"], data_duplicate["question2"]]).flatten()
n = np.dstack([data_nonduplicate["question1"], data_nonduplicate["question2"]]).flatten()

print ("Number of data points in class 1 (duplicate pairs) :",len(p)//2)
print ("Number of data points in class 0 (non duplicate pairs) :",len(n)//2)


# In[13]:


from wordcloud import WordCloud, STOPWORDS
# reading the text files and removing the Stop Words:

stopwords = set(STOPWORDS)
stopwords.add("said")
stopwords.add("br")
stopwords.add(" ")
stopwords.remove("not")

stopwords.remove("no")
#stopwords.remove("good")
#stopwords.remove("love")
stopwords.remove("like")
#stopwords.remove("best")
#stopwords.remove("!")
textp_w = " ".join(p)
textn_w = " ".join(n)
print ("Total number of words in duplicate pair questions :", len(textp_w.split()))
print ("Total number of words in non duplicate pair questions :", len(textn_w.split()))


# In[14]:


# word cloiud for duplicate pair question's text
import matplotlib.pyplot as plt
wc = WordCloud(background_color="white", height=400, width=800, max_words=len(textp_w), stopwords=stopwords)
wc.generate(textp_w)
print ("Word Cloud for Duplicate Question pairs")
plt.figure(figsize=(12, 8))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[15]:


# Wordcloud for non duplicate pairs of question text
wc = WordCloud(background_color="white", height=400, width=800, max_words=len(textn_w),stopwords=stopwords)
# generate word cloud
wc.generate(textn_w)
print ("Word Cloud for non-Duplicate Question pairs:")
plt.figure(figsize=(12, 8))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[16]:


# Pair plot of features ['ctc_min', 'cwc_min', 'csc_min', 'token_sort_ratio']
import seaborn as sns
n = data.shape[0]
sns.pairplot(data[['ctc_min', 'cwc_min', 'csc_min', 'token_sort_ratio', 'is_duplicate']][0:n], 
             hue='is_duplicate', vars=['ctc_min', 'cwc_min', 'csc_min', 'token_sort_ratio'])
plt.show()


# In[17]:


# Distribution of the token_sort_ratio
plt.figure(figsize=(10, 8))

plt.subplot(1,2,1)
sns.violinplot(x = 'is_duplicate', y = 'token_sort_ratio', data = data[0:] , )

plt.subplot(1,2,2)
sns.distplot(data[data['is_duplicate'] == 1.0]['token_sort_ratio'][0:] , label = "1", color = 'red')
sns.distplot(data[data['is_duplicate'] == 0.0]['token_sort_ratio'][0:] , label = "0" , color = 'blue' )
plt.show()


# In[18]:


plt.figure(figsize=(10, 8))

plt.subplot(1,2,1)
sns.violinplot(x = 'is_duplicate', y = 'fuzz_ratio', data = data[0:] , )

plt.subplot(1,2,2)
sns.distplot(data[data['is_duplicate'] == 1.0]['fuzz_ratio'][0:] , label = "1", color = 'red')
sns.distplot(data[data['is_duplicate'] == 0.0]['fuzz_ratio'][0:] , label = "0" , color = 'blue' )
plt.show()


# In[19]:


plt.figure(figsize=(10, 8))

plt.subplot(1,2,1)
sns.violinplot(x = 'is_duplicate', y = 'fuzz_partial_ratio', data = data[0:] , )

plt.subplot(1,2,2)
sns.distplot(data[data['is_duplicate'] == 1.0]['fuzz_partial_ratio'][0:] , label = "1", color = 'red')
sns.distplot(data[data['is_duplicate'] == 0.0]['fuzz_partial_ratio'][0:] , label = "0" , color = 'blue' )
plt.show()


# In[20]:


# All the three fuzzy features seem to have similar distribution amongst two classes


# In[21]:


# VISUALIZATION
# Using TSNE for Dimentionality reduction for 15 Features(Generated after cleaning the data) to 3 dimention

from sklearn.preprocessing import MinMaxScaler

X = MinMaxScaler().fit_transform(data[['cwc_min', 'cwc_max', 'csc_min', 'csc_max' , 'ctc_min' , 'ctc_max' , 'last_word_eq', 'first_word_eq' , 'abs_len_diff' , 'mean_len' , 'token_set_ratio' , 'token_sort_ratio' ,  'fuzz_ratio' , 'fuzz_partial_ratio' , 'longest_substr_ratio']])
y = data['is_duplicate'].values


# In[22]:


from sklearn.manifold import TSNE

# TSNE 2d
tsne2d = TSNE(
    n_components=2,
    init='random', # pca
    random_state=101,
    method='barnes_hut',
    n_iter=1000,
    verbose=2,
    angle=0.5
).fit_transform(X)


# In[23]:


df = pd.DataFrame({'x':tsne2d[:,0], 'y':tsne2d[:,1] ,'label':y})

# draw the plot in appropriate place in the grid
sns.lmplot(data=df, x='x', y='y', hue='label', fit_reg=False, size=8,palette="Set1",markers=['s','o'])
plt.title("perplexity : {} and max_iter : {}".format(30, 1000))
plt.show()


# In[24]:


from sklearn.manifold import TSNE

# TSNE 3d
tsne3d = TSNE(
    n_components=3,
    init='random', # pca
    random_state=101,
    method='barnes_hut',
    n_iter=1000,
    verbose=2,
    angle=0.5
).fit_transform(X)


# In[25]:


import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

trace1 = go.Scatter3d(
    x=tsne3d[:,0],
    y=tsne3d[:,1],
    z=tsne3d[:,2],
    mode='markers',
    marker=dict(
        sizemode='diameter',
        color = y,
        colorscale = 'Portland',
        colorbar = dict(title = 'duplicate'),
        line=dict(color='rgb(255, 255, 255)'),
        opacity=0.75
    )
)

trace_data = [trace1]
layout=dict(height=800, width=1200, title='3d embedding with engineered features')
fig=dict(data=trace_data, layout=layout)
py.iplot(fig, filename='3DBubble')


# In[26]:


questions = list(data["question1"]) + list(data["question2"])


# In[27]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(lowercase=False)
tfidf.fit_transform(questions)

word2tfidf = dict(zip(tfidf.get_feature_names(), tfidf.idf_))


# In[28]:


import spacy 
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")
vecs1 = []

# Set up in the spacy model
vect_dim = len(nlp(data["question1"][0])[0].vector)

for qu1 in tqdm(list(data["question1"])):
    mean_vec1 = np.zeros([1, vect_dim])
    if len(qu1) != 0:
        doc1 = nlp(qu1)
        mean_vec1 = np.zeros([len(doc1), len(doc1[0].vector)])
        for word1 in doc1:
            vec1 = word1.vector
            try: 
                idf = word2tfidf[str(word)]
            except:
                idf = 0
            mean_vec1 += vec1 * idf
    mean_vec1 = mean_vec1.mean(axis=0)
    vecs1.append(mean_vec1)        

data["q1_feats_tfidf_avg_w2v"] = list(vecs1)


# In[29]:


vecs2 = []

# Set up in the spacy model
vect_dim = len(nlp(data["question2"][0])[0].vector)

for qu2 in tqdm(list(data['question2'])):
    mean_vec2 = np.zeros([1, vect_dim])
    if len(qu2) != 0:
        doc2 = nlp(qu2) 
        mean_vec2 = np.zeros([len(doc1), len(doc2[0].vector)])        
        for word2 in doc2:
            # word2vec
            vec2 = word2.vector
            # fetch idf score
            try:
                idf = word2tfidf[str(word2)]
            except:
                idf = 0
            # computing idf weighted avg w2v
            mean_vec2 += vec2 * idf
    mean_vec2 = mean_vec2.mean(axis=0)
    vecs2.append(mean_vec2)

data['q2_feats_tfidf_avg_w2v'] = list(vecs2)


# In[30]:


# We can make BoW based average Word2Vec also


# In[31]:


labels1 = ['freq_qid1', 'freq_qid2', 'q1len', 'q2len', 'q1_n_words', 'q2_n_words', 'word_common', 'word_total', 
          'word_share', 'freq_q1+q2', 'freq_q1-q2']
without_preprocess_fe_df = data[labels1]


# In[32]:


labels2 = ['cwc_min', 'cwc_max', 'csc_min', 'csc_max', 'ctc_min', 'ctc_max', 'last_word_eq', 'first_word_eq', 
           'abs_len_diff', 'mean_len', 'token_set_ratio', 'token_sort_ratio', 'fuzz_ratio', 'fuzz_partial_ratio', 
           'longest_substr_ratio']
nlp_fuzzy_fe_df = data[labels2]


# In[33]:


labels3 = ['q1_feats_tfidf_avg_w2v', 'q2_feats_tfidf_avg_w2v']
avg_w2v_fe_df = data[labels3]


# In[34]:


df3_q1 = pd.DataFrame(avg_w2v_fe_df["q1_feats_tfidf_avg_w2v"].values.tolist(), index= avg_w2v_fe_df.index)
df3_q2 = pd.DataFrame(avg_w2v_fe_df["q2_feats_tfidf_avg_w2v"].values.tolist(), index= avg_w2v_fe_df.index)


# In[35]:


without_preprocess_fe_df.head()


# In[36]:


nlp_fuzzy_fe_df.head()


# In[37]:


# Questions 1 tfidf weighted word2vec
df3_q1.head()


# In[38]:


# Questions 2 tfidf weighted word2vec
df3_q2.head()


# In[39]:


print("Number of features in without preprocess dataframe :", without_preprocess_fe_df.shape[1])
print("Number of features in nlp dataframe :", nlp_fuzzy_fe_df.shape[1])
print("Number of features in question1 w2v  dataframe :", df3_q1.shape[1])
print("Number of features in question2 w2v  dataframe :", df3_q1.shape[1])
print("Number of features in final dataframe  :", without_preprocess_fe_df.shape[1]+nlp_fuzzy_fe_df.shape[1]+df3_q1.shape[1]+df3_q2.shape[1])

