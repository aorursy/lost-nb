#!/usr/bin/env python
# coding: utf-8



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




import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from subprocess import check_output
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import os
import gc

import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup




import os
print(os.listdir("../input"))




df=pd.read_csv('../input/quora-question-pairs/train.csv.zip')
df.shape




df.head()




df.info()




df.groupby('is_duplicate')['id'].count().plot.bar()




dup_0=df[df['is_duplicate']==0].shape[0]
dup_1=df.shape[0]-dup_0




print("Percentage of 0's or not duplicates in dataset: {0:.2f}".format(dup_0*100/(dup_0+dup_1)))
print("Percentage of 1's or duplicates in dataset: {0:.2f}".format(dup_1*100/(dup_0+dup_1)))




#number of unique questions in the dataset
from collections import Counter
q_list1=df['qid1'].tolist()
q_list2=df['qid2'].tolist()
q_list1.extend(q_list2)
c=Counter(q_list1)




print("Number of unique questions in dataset: ",len(c.values()))
print("Maximum number of times a question is repeated :  ",max(c.values()))
print("Number of times questions are repeated in dataset: ",sum(c.values())-len(c.values()))
t=0
for i in c.values():
    if i>1:
        t+=1
print("Number of unique questions that are repeated: ",t)




#checking whether there are any duplicate pairs
group_df_count=df.groupby(['qid1','qid2']).count().shape[0]
print("Number of duplicate pairs: ",group_df_count-df.shape[0])




plt.figure(figsize=(20,10))
plt.hist(c.values(),bins=160)
plt.yscale('log',nonposy='clip')
plt.title('log-histogram of questions occurrences')
plt.xlabel('number of occurrences of questions')
plt.ylabel('total number of questions')
plt.show()




df[df.isnull().any(axis=1)]




df.fillna(' ',inplace=True)




df[df.isnull().any(1)]




df['freq_qid1'] = df.groupby('qid1')['qid1'].transform('count') 
df['freq_qid2'] = df.groupby('qid2')['qid2'].transform('count')
df['q1len'] = df['question1'].str.len() 
df['q2len'] = df['question2'].str.len()
df['q1_n_words'] = df['question1'].apply(lambda row: len(row.split(" ")))
df['q2_n_words'] = df['question2'].apply(lambda row: len(row.split(" ")))
def normalized_word_Common(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
    return 1.0 * len(w1 & w2)
df['word_Common'] = df.apply(normalized_word_Common, axis=1)
def normalized_word_Total(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
    return 1.0 * (len(w1) + len(w2))
df['word_Total'] = df.apply(normalized_word_Total, axis=1)
def normalized_word_share(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
    return 1.0 * len(w1 & w2)/(len(w1) + len(w2))
df['word_share'] = df.apply(normalized_word_share, axis=1)
df['freq_q1+q2'] = df['freq_qid1']+df['freq_qid2']
df['freq_q1-q2'] = abs(df['freq_qid1']-df['freq_qid2'])




df.head()




print("minimum length of question 1: ",df['q1_n_words'].min())
print("minimum length of question 2: ",df['q2_n_words'].min())
t1=df['q1_n_words'].min()
t2=df['q2_n_words'].min()




print("number of question 1 with minimum length: ",df[df['q1_n_words']==t1].shape[0])
print("number of question 2 with minimum length: ",df[df['q2_n_words']==t2].shape[0])




plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
sns.violinplot(x='is_duplicate',y='word_share',data=df)
plt.title('violin plot')
plt.subplot(1,2,2)
sns.distplot(df[df['is_duplicate']==1]['word_share'],color='red',label='1-duplicate')
sns.distplot(df[df['is_duplicate']==0]['word_share'],color='blue',label='0-not duplicate')
plt.title('PDF for word_share')
plt.legend()
plt.show()




plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
sns.violinplot(x='is_duplicate',y='word_Common',data=df)
plt.title('Violin plot for word_Common')
plt.subplot(1,2,2)
plt.title('PDF for word_Common')
sns.distplot(df[df['is_duplicate']==1]['word_Common'],color='red',label='1-duplicate')
sns.distplot(df[df['is_duplicate']==0]['word_Common'],color='blue',label='0-not duplicate')
plt.legend()
plt.show()




plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
plt.title('Violin plot for freq_q1+q2')
sns.violinplot(x='is_duplicate',y='freq_q1+q2',data=df)
plt.subplot(1,2,2)
plt.title('PDF for freq_q1+q2')
sns.distplot(df[df['is_duplicate']==1]['freq_q1+q2'],color='red',label='1-duplicate')
sns.distplot(df[df['is_duplicate']==0]['freq_q1+q2'],color='blue',label='0-not duplicate')
plt.legend()
plt.show()




plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
plt.title('Violin plot for freq_q1-q2')
sns.violinplot(x='is_duplicate',y='freq_q1-q2',data=df)
plt.subplot(1,2,2)
plt.title('PDF for freq_q1-q2')
sns.distplot(df[df['is_duplicate']==1]['freq_q1-q2'],color='red',label='1-duplicate')
sns.distplot(df[df['is_duplicate']==0]['freq_q1-q2'],color='blue',label='0-not duplicate')
plt.legend()
plt.show()




plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
plt.title('Violin plot for q1_n_words')
sns.violinplot(x='is_duplicate',y='q1_n_words',data=df)
plt.subplot(1,2,2)
plt.title('PDF for q1_n_words')
sns.distplot(df[df['is_duplicate']==1]['q1_n_words'],color='red',label='1-duplicate')
sns.distplot(df[df['is_duplicate']==0]['q1_n_words'],color='blue',label='0-not duplicate')
plt.legend()
plt.show()




plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
sns.violinplot(x='is_duplicate',y='q2_n_words',data=df)
plt.title('Violin plot for q2_n_words')
plt.subplot(1,2,2)
plt.title('PDF for q2_n_words')
sns.distplot(df[df['is_duplicate']==1]['q2_n_words'],color='red',label='1-duplicate')
sns.distplot(df[df['is_duplicate']==0]['q2_n_words'],color='blue',label='0-not duplicate')
plt.legend()
plt.show()




plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
plt.title('Violin plot for q1len')
sns.violinplot(x='is_duplicate',y='q1len',data=df)
plt.subplot(1,2,2)
plt.title('PDF for q1len')
sns.distplot(df[df['is_duplicate']==1]['q1len'],color='red',label='1-duplicate')
sns.distplot(df[df['is_duplicate']==0]['q1len'],color='blue',label='0-not duplicate')
plt.legend()
plt.show()




plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
plt.title('Violin plot for q2len')
sns.violinplot(x='is_duplicate',y='q2len',data=df)
plt.subplot(1,2,2)
plt.title('PDF for q2len')
sns.distplot(df[df['is_duplicate']==1]['q2len'],color='red',label='1-duplicate')
sns.distplot(df[df['is_duplicate']==0]['q2len'],color='blue',label='0-not duplicate')
plt.legend()
plt.show()




# To get the results in 4 decemal points
SAFE_DIV = 0.0001 

STOP_WORDS = stopwords.words("english")


def preprocess(x):
    x = str(x).lower()
    x = x.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")                           .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")                           .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")                           .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")                           .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")                           .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")                           .replace("€", " euro ").replace("'ll", " will")
    x = re.sub(r"([0-9]+)000000", r"\1m", x)
    x = re.sub(r"([0-9]+)000", r"\1k", x)
    
    
    porter = PorterStemmer()
    pattern = re.compile('\W')
    
    if type(x) == type(''):
        x = re.sub(pattern, ' ', x)
    
    
    if type(x) == type(''):
        x = porter.stem(x)
        example1 = BeautifulSoup(x)
        x = example1.get_text()
               
    
    return x
    




get_ipython().run_cell_magic('bash', '', 'pip install distance')




import distance




def get_token_features(q1, q2):
    token_features = [0.0]*10
    
    # Converting the Sentence into Tokens: 
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features
    # Get the non-stopwords in Questions
    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])
    
    #Get the stopwords in Questions
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])
    
    # Get the common non-stopwords from Question pair
    common_word_count = len(q1_words.intersection(q2_words))
    
    # Get the common stopwords from Question pair
    common_stop_count = len(q1_stops.intersection(q2_stops))
    
    # Get the common Tokens from Question pair
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))
    
    
    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    
    # Last word of both question is same or not
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    
    # First word of both question is same or not
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])
    
    token_features[8] = abs(len(q1_tokens) - len(q2_tokens))
    
    #Average Token Length of both Questions
    token_features[9] = (len(q1_tokens) + len(q2_tokens))/2
    return token_features

# get the Longest Common sub string

def get_longest_substr_ratio(a, b):
    strs = list(distance.lcsubstrings(a, b))
    if len(strs) == 0:
        return 0
    else:
        return len(strs[0]) / (min(len(a), len(b)) + 1)

def extract_features(df):
    # preprocessing each question
    df["question1"] = df["question1"].fillna("").apply(preprocess)
    df["question2"] = df["question2"].fillna("").apply(preprocess)

    print("token features...")
    
    # Merging Features with dataset
    
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
    

    print("fuzzy features..")

    df["token_set_ratio"]       = df.apply(lambda x: fuzz.token_set_ratio(x["question1"], x["question2"]), axis=1)
    # The token sort approach involves tokenizing the string in question, sorting the tokens alphabetically, and 
    # then joining them back into a string We then compare the transformed strings with a simple ratio().
    df["token_sort_ratio"]      = df.apply(lambda x: fuzz.token_sort_ratio(x["question1"], x["question2"]), axis=1)
    df["fuzz_ratio"]            = df.apply(lambda x: fuzz.QRatio(x["question1"], x["question2"]), axis=1)
    df["fuzz_partial_ratio"]    = df.apply(lambda x: fuzz.partial_ratio(x["question1"], x["question2"]), axis=1)
    df["longest_substr_ratio"]  = df.apply(lambda x: get_longest_substr_ratio(x["question1"], x["question2"]), axis=1)
    return df




get_ipython().run_cell_magic('bash', '', 'pip install fuzzywuzzy')




from fuzzywuzzy import fuzz




df=extract_features(df)
df.head()




df.to_csv("nlp_features_train.csv", index=False)






