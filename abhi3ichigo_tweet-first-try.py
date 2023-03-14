#!/usr/bin/env python
# coding: utf-8



from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

import pandas as pd
import numpy as np
import re

# LSTM for sequence classification in the IMDB dataset
import numpy
#import gensim
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing import sequence
from keras import preprocessing
from keras.preprocessing.text import Tokenizer

import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

# Let's try TF-IDF, LSA, Clustering (Docs are sentiments)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import nltk
import pandas as pd
import numpy as np
import lxml.html
import re
import collections
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import distance
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

import lxml.html
import nltk
from nltk.tokenize import word_tokenize
from nltk import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer




#nltk.download('wordnet')
#nltk.download('stopwords')
#nltk.download('punkt')




def tf_idf_train(df,col_name):
    
    vec = TfidfVectorizer(ngram_range=(1,1))
    resp = vec.fit_transform(df[col_name])
        
    return resp,vec


##############################################################################################

def tf_idf_test(df,col_name,vec):
    
    resp = vec.transform(df[col_name])
    
    return resp,vec

#############################################################################################

def combine_text_by_sentiment(df):
    
    cols = ['sentiment','combined_text']
    
    df_tmp = df[['sentiment','selected_text']].copy ()
    
    #Creating empty dataframe 
    dfz1 = pd.DataFrame(columns=cols)
    
    #Add all text cols
    #df_tmp['questions_title'] = (df['questions_title'] + ' ' + df['questions_body'] + ' ' + df['answers_body'])
        
    group = df_tmp.groupby('sentiment');i=0
          
    for nm,gr in group:
        
        tmp_str = ''
        for each in gr['selected_text']:
            tmp_str = tmp_str + ' ' + each
        dfz1.loc[i] = [nm,tmp_str]
                
        i += 1
            
    return dfz1

###############################################################################################

def remove_stop_words(df,col_name):
    
    #Lowercase all words
    for idx,each in enumerate(df[col_name]):
        df.iloc[idx,df.columns.get_loc(col_name)] = each.lower()
    
    #Remove stopwords
    
    for idx,each in enumerate(df[col_name]):
        word_tokens = word_tokenize(each)
        df.iloc[idx,df.columns.get_loc(col_name)] = " ".join(select_stop_words(word_tokens))
    
    return df

###############################################################################################

def remove_spl_chars(df0,col_name):
    df = df0.copy()
    
    for idx,each in enumerate(df[col_name]):
        word_tokens = word_tokenize(each)
        tmp = []
        
        for each2 in word_tokens:
            #Removing special characters - •,! etc
            word = re.sub('[^\s\w]','',each2)
            tmp.append(word)

        df.iloc[idx,df.columns.get_loc(col_name)] = " ".join(tmp)

    return df

###############################################################################################

def lemma_text(df0,col_name):
    df = df0.copy()
    
     #Lowercase all words
    for idx,each in enumerate(df[col_name]):
        df.iloc[idx,df.columns.get_loc(col_name)] = each.lower()
    
    #Lemmatize Text
    lemmatizer = WordNetLemmatizer()

    for idx,each in enumerate(df[col_name]):
        word_tokens = word_tokenize(each)
        tmp = []
        
        for each2 in word_tokens:
            tmp.append(lemmatizer.lemmatize(each2))

        df.iloc[idx,df.columns.get_loc(col_name)] = " ".join(tmp)
    
    return df

###############################################################################################


def remove_html(df,col_name):
    
    for idx, each in enumerate(df[col_name]):
        #print(str(idx)+":"+col_name)
        
        df.iloc[idx,df.columns.get_loc(col_name)] = re.sub(r'http\S+', '', each)
    
    return df

###############################################################################################

#Function obtained from https://www.programcreek.com/python/example/106181/sklearn.feature_extraction.stop_words.ENGLISH_STOP_WORDS and is open source
def select_stop_words(word_list):
    """ Filter out cluster term names"""
    st = PorterStemmer()
    out = []
    for word in word_list:
        word_st = st.stem(word)
        if len(word_st) <= 2 or                re.match('\d+', word_st) or                 re.match('[^a-zA-Z0-9]', word_st) or                word in ENGLISH_STOP_WORDS:
            continue
        out.append(word)
    return out 




df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
df = df.dropna()
df.head()




df2 = df.copy()

cols = ['selected_text']
for each in cols:
    #Function call to html tag removal
    df2 = remove_html(df2,each)
    
    #Function to remove spl chars
    df2 = remove_spl_chars(df2,each)
    
    #Function call to remove stop words
    df2 = remove_stop_words(df2,each)
    
    #Function to lemmatie text
    df2 = lemma_text(df2,each)




df_1 = combine_text_by_sentiment(df2)
df_1.head()




tr_resp, tr_vec = tf_idf_train(df_1,'combined_text')




#tr_resp




#tr_vec




#tr_resp[0,100]




#tr_vec.get_feature_names()




df_t = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
df_t = df_t.dropna()
df_t.head()




df2_test = df_t.copy()
#df2_test['ori_text'] = df2_test['text']
    
cols = ['text']
df2_test = remove_html(df2_test,cols[0])
df2_test = remove_spl_chars(df2_test,cols[0])




#tr_vec.get_feature_names().index('is')
def get_res_test(sample_txt,sentiment_txt):
    
    sentiment_id = df_1[df_1['sentiment'] == sentiment_txt].index
    
    sample_lst = []
    for each in sample_txt.split():
        try:
            id = tr_vec.get_feature_names().index(each)
            sample_lst.append(tr_resp[sentiment_id,id])
        except:
            sample_lst.append(0)

    sample_txt_lst = sample_txt.split()
    res = ""

    start=-1
    end=-1
    for idx,each in enumerate(sample_lst):
        if (each > 0):
            start = idx
            break

            res = ' '.join([res,sample_txt_lst[idx]])

    for idx in range(len(sample_lst)-1,0,-1):
        if (sample_lst[idx] > 0):
            end = idx+1
            break

    res = ' '.join(sample_txt_lst[start:end])
    return res

#print(sample_txt[sample_txt.find(sample_txt_lst[start]):sample_txt.find(sample_txt_lst[end])])




df2_test['selected_text'] = df2_test.apply(lambda x: get_res_test(x.text, x.sentiment), axis=1)
df2_test.head()




df2_test = df2_test.drop(columns=['text', 'sentiment'])
df2_test.to_csv('/kaggle/working/submission.csv',index=False)






