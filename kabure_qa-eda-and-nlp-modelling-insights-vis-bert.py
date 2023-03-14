#!/usr/bin/env python
# coding: utf-8



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib_venn import venn2, venn3, venn2_circles, venn3_circles #to create intersection graphs
import matplotlib.pyplot as plt #to plot show the charts
import seaborn as sns
from scipy import stats

from nltk import word_tokenize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import os 
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))




df_train = pd.read_csv("../input/google-quest-challenge/train.csv")
df_test = pd.read_csv("../input/google-quest-challenge/test.csv")
df_sub = pd.read_csv("../input/google-quest-challenge/sample_submission.csv")




def resumetable(df):
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values    
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values

    for name in summary['Name'].value_counts().index:
        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 

    return summary




resumetable(df_train)[:10]




resumetable(df_test)[:10]




print(f"Shape of submission: {df_sub.shape}")
target_cols = df_train[df_train.columns[df_train.columns.isin(df_sub.columns[1:])]].columns




df_train.head(3)




## I will use the host column, split and get the first string
for i in range(len(df_train)):
    df_train.loc[i,'host_cat'] = df_train.host.str.split('.')[i][0]
df_train.drop('host', axis=1, inplace=True)




host = df_train.groupby(['host_cat'])['url'].nunique().sort_values(ascending=False)
category = df_train.groupby(['category'])['url'].nunique().sort_values(ascending=False)

plt.figure(figsize=(16,12))
plt.suptitle('Unique URL by Host and Categories', size=22)

plt.subplot(211)
g0 = sns.barplot(x=category.index, y=category.values, color='blue')
g0.set_title("Unique Answers by category", fontsize=22)
g0.set_xlabel("Category Name", fontsize=19)
g0.set_ylabel("Total Count", fontsize=19)
#g1.set_xticklabels(g1.get_xticklabels(),rotation=45)
for p in g0.patches:
    height = p.get_height()
    g0.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.1f}%'.format(height/category.sum()*100),
            ha="center",fontsize=11) 

plt.subplot(212)
g1 = sns.barplot(x=host[:20].index, y=host[:20].values, color='blue')
g1.set_title("TOP 20 HOSTS with more UNIQUE questions", fontsize=22)
g1.set_xlabel("Host Name", fontsize=19)
g1.set_ylabel("Total Count", fontsize=19)
g1.set_xticklabels(g1.get_xticklabels(),rotation=45)
for p in g1.patches:
    height = p.get_height()
    g1.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.1f}%'.format(height/host.sum()*100),
            ha="center",fontsize=11) 
    
plt.subplots_adjust(hspace = 0.3, top = 0.90)

plt.show()




print(f"Total Unique Users in 'Question User Name': {df_train['question_user_name'].nunique()}")
print(f"Total Unique Users in 'Answer User Name': {df_train['answer_user_name'].nunique()}")




plt.figure(figsize=(12,8))

venn2([set(df_train['question_user_name'].value_counts(dropna=False).index), 
       set(df_train['answer_user_name'].value_counts(dropna=False).index)],
      set_labels=('Question Users', 'Answer Users'), alpha=.5)
plt.title('Comparison of Question and Answer Users Intersection\n', fontsize=20)

plt.show()




import matplotlib.gridspec as gridspec # to do the grid of plots

grid = gridspec.GridSpec(3, 3)
plt.figure(figsize=(16,3*4))

plt.suptitle('Intersection QA USERS \nQuestions and Answers by different CATEGORIES', size=20)

for n, col in enumerate(df_train['category'].value_counts().index):
    ax = plt.subplot(grid[n])
    venn2([set(df_train[df_train.category == col]['question_user_name'].value_counts(dropna=False).index), 
           set(df_train[df_train.category == col]['answer_user_name'].value_counts(dropna=False).index)],
      set_labels=('Question Users', 'Answer Users'), )
    ax.set_title(str(col), fontsize=15)
    ax.set_xlabel('')
    #plt.subplots_adjust(top = 0.98, wspace=.9, hspace=.9)
    
plt.subplots_adjust(top = 0.9, hspace=.1)

plt.show()




grid = gridspec.GridSpec(5, 3)
plt.figure(figsize=(16,4.5*4))

plt.suptitle('Intersection QA USERS - TOP 15 \nQuestions and Answers by different HOSTS', size=20)
top_host = df_train['host_cat'].value_counts()[:15].index
for n, col in enumerate(top_host):
    ax = plt.subplot(grid[n])
    venn2([set(df_train[df_train.host_cat == col]['question_user_name'].value_counts(dropna=False).index), 
           set(df_train[df_train.host_cat == col]['answer_user_name'].value_counts(dropna=False).index)],
      set_labels=('Question Users', 'Answer Users'), )
    ax.set_title(str(col), fontsize=15)
    ax.set_xlabel('')
    #plt.subplots_adjust(top = 0.98, wspace=.9, hspace=.9)
    
plt.subplots_adjust(top = 0.9, hspace=.1)

plt.show()




# Tokenize each item in the review column
word_tokens = [word_tokenize(question) for question in df_train.question_body]

# Create an empty list to store the length of the reviews
len_tokens = []

# Iterate over the word_tokens list and determine the length of each item
for i in range(len(word_tokens)):
     len_tokens.append(len(word_tokens[i]))

# Create a new feature for the lengh of each review
df_train['question_n_words'] = len_tokens




grid = gridspec.GridSpec(5, 3)
plt.figure(figsize=(16,6*4))

plt.suptitle('Title and Question Lenghts by Different Categories \nThe Mean in RED - Also 5% and 95% lines', size=20)
count=0
top_cats=df_train['category'].value_counts().index
for n, col in enumerate(top_cats):
    for i, q_t in enumerate(['question_title', 'question_body', 'question_n_words']):
        ax = plt.subplot(grid[count])
        if q_t == 'question_n_words':
            sns.distplot(df_train[df_train['category'] == col][q_t], bins = 50, 
                         color='g', label="RED - 50%") 
            ax.set_title(f"Distribution of {str(col)} \nQuestion #Total Words Distribution", fontsize=15)
            ax.axvline(df_train[df_train['category'] == col][q_t].quantile(.95))
            ax.axvline(df_train[df_train['category'] == col][q_t].quantile(.05))
            mean_val = df_train[df_train['category'] == col][q_t].mean()
            ax.axvline(mean_val, color='red' )
            ax.set_xlabel('')            
        else:
            sns.distplot(df_train[df_train['category'] == col][q_t].str.len(), bins = 50, 
                         color='g', label="RED - 50%") 
            ax.set_title(f"Distribution of {str(col)} \n{str(q_t)}", fontsize=15)
            ax.axvline(df_train[df_train['category'] == col][q_t].str.len().quantile(.95))
            ax.axvline(df_train[df_train['category'] == col][q_t].str.len().quantile(.05))
            mean_val = df_train[df_train['category'] == col][q_t].str.len().mean()
            ax.axvline(mean_val, color='red' )
            #ax.text(x=mean_val*1.1, y=.02, s='Holiday in US', alpha=0.7, color='#334f8d')
            ax.set_xlabel('')
        count+=1
        
plt.subplots_adjust(top = 0.90, hspace=.4, wspace=.15)
plt.show()




grid = gridspec.GridSpec(10, 3)

plt.figure(figsize=(16,8*4))
count=0
plt.suptitle('Distribution of QA metrics (Target Features)', size=20)
# top_host = df_train['host_cat'].value_counts()[:15].index
for n, col in enumerate(target_cols):
    #if df_train[target_cols].std()[col] > .15:
    ax = plt.subplot(grid[count])
    sns.boxplot(x='category', y=col, data=df_train)
    ax.set_title(str(col), fontsize=13)
    ax.set_xlabel('')
    ax.set_ylabel(' ')
    count+=1
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45)

plt.subplots_adjust(top = 0.95, hspace=.9, wspace=.2)

plt.show()




pca = PCA(n_components=3, random_state=42)

principalComponents = pca.fit_transform(df_train[target_cols])

principalDf = pd.DataFrame(principalComponents)

# df.drop(cols, axis=1, inplace=True)
prefix='Target_PCA'
principalDf.rename(columns=lambda x: str(prefix)+str(x), inplace=True)

df_train = pd.concat([df_train, principalDf], axis=1)




print("TOP 3 PCA Explanability: ")
[print(str(f"{i+1} - {round(pca*100,3)}%")) for i, pca in enumerate(pca.explained_variance_ratio_[:3])]
print(f"Sum of 3 Principal components: {round(pca.explained_variance_ratio_[:3].sum()*100,3)}%")




plt.figure(figsize=(15,6))
g = sns.scatterplot(x='Target_PCA0', y='Target_PCA1', data=df_train, hue='category')
g.set_title("PCA Components Distribution by Categories", fontsize=22)
g.set_xlabel("TARGET PCA 0", fontsize=16)
g.set_ylabel("TARGET PCA 1", fontsize=16)

plt.show()




g = sns.FacetGrid(df_train[df_train.host_cat.isin(top_host)], col='host_cat',
                  col_wrap=3, height=3, aspect=1.5, hue='category')

g.map(sns.scatterplot, "Target_PCA0", "Target_PCA1", alpha=.5 ).add_legend();
g.set_titles('{col_name}', fontsize=17)
plt.show()




plt.figure(figsize=(16,10))
sns.heatmap(df_train[target_cols].corr(),vmin=-1,cmap='YlGnBu')




from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)
newStopWords = ['amp', 'gt', 'lt', 'div', 'id',
                'fi', 'will', 'use', 'one', 'nbsp', 'need']
stopwords.update(newStopWords)




grid = gridspec.GridSpec(5, 2)

plt.figure(figsize=(16,7*4))

plt.suptitle('Word Cloud OF CATEGORY FEATURE', size=20)

for n, col in enumerate(df_train['category'].value_counts().index):
    ax = plt.subplot(grid[n])  
    
    wordcloud = WordCloud(
        background_color='black',
        stopwords=stopwords,
        max_words=250,
        max_font_size=100, 
        width=400, height=280,
        random_state=42,
    ).generate(" ".join(df_train[df_train['category'] == col]['answer'].astype(str)))

    #print(wordcloud)

    plt.imshow(wordcloud)
    plt.title(f"Category: {col}",fontsize=18)
    plt.axis('off')
plt.subplots_adjust(top = 0.95, hspace=.2, wspace=.1 )

plt.show()




#newStopWords = ['fruit', "Drink", "black"]

#stopwords.update(newStopWords)

import matplotlib.gridspec as gridspec # to do the grid of plots
grid = gridspec.GridSpec(5, 2)

plt.figure(figsize=(16,7*4))

plt.suptitle('Answers Word Cloud \nTOP 10 hosts with more questions', size=20)

for n, col in enumerate(df_train['host_cat'].value_counts()[:10].index):
    ax = plt.subplot(grid[n])   
    wordcloud = WordCloud(
        background_color='black',
        stopwords=stopwords,
        max_words=250,
        max_font_size=100, 
        width=400, height=280,
        random_state=42,
    ).generate(" ".join(df_train[df_train['host_cat'] == col]['answer'].astype(str)))

    #print(wordcloud)

    plt.imshow(wordcloud)
    plt.title(f"Host: {col}",fontsize=18)
    plt.axis('off')
    
plt.subplots_adjust(top = 0.95, hspace=.2, wspace=.1 )

plt.show()




from textblob import TextBlob

pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity

df_train['ans_polarity']= df_train['answer'].apply(pol)
df_train['ans_subjectivity']= df_train['answer'].apply(sub)

df_train[['answer', 'category', 'ans_polarity', 'ans_subjectivity']].head()




plt.figure(figsize=(16,5))

g = sns.scatterplot(x='ans_polarity', y='ans_subjectivity', 
                    data=df_train, hue='category')
g.set_title("Sentiment Analyzis (Polarity x Subjectivity) by 'Category' Feature", fontsize=21)
g.set_xlabel("Polarity distribution",fontsize=18)
g.set_ylabel("Subjective ",fontsize=18)

plt.show()




polarity_answers = df_train.groupby('category')['ans_polarity', 'ans_subjectivity'].describe().reset_index()

polarity_answers




stopwords.update(['amp', 'lt', 'gt', 'frac'])




import re
contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", 
                    "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not",
                    "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would",
                    "he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", 
                    "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have",
                    "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                    "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have",
                    "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", 
                    "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not",
                    "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", 
                    "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
                    "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                    "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", 
                    "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have",
                    "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", 
                    "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is",
                    "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is",
                    "they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have",
                    "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would",
                    "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", 
                    "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", 
                    "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", 
                    "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have",
                    "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", 
                    "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", 
                    "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
                    "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are",
                    "y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will",
                    "you'll've": "you will have", "you're":"you are", 'you re':"you are",'youre': "you are", "you've": "you have", }

puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>',
          '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£','·', '_', '{', '}', '©', '^', '®', '`', 
          '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
          '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾',
          '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─',  '▒', '：', '¼', '⊕', '▼', '▪', '†', '■',
          '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
          '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬',
          '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', #'lt', 'gt', 'amp', 'div', 'ex', 'le', 'http', 'www', 'vo', '\n'
         ]

def clean_text(x):
    x = str(x)
    
    for punct in puncts:
        if punct in x:
            x = x.replace(punct, '')
    return x

def _get_contractions(contraction_dict):
    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    return contraction_dict, contraction_re

contractions, contractions_re = _get_contractions(contraction_dict)

def replace_contractions(text):
    def replace(match):
        return contractions[match.group(0)]
    return contractions_re.sub(replace, text)




# Here, the order is important
df_train['answer'] = df_train['answer'].apply(replace_contractions)
df_train['answer'] = df_train['answer'].apply(clean_text)




from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

grid = gridspec.GridSpec(5, 3)
plt.figure(figsize=(16,6*4))

for n, cat in enumerate(top_host[:9]):
    
    ax = plt.subplot(grid[n])   
    # print(f'PRINCIPAL WORDS CATEGORY: {cat}')
    # vectorizer = CountVectorizer(ngram_range = (3,3)) 
    # X1 = vectorizer.fit_transform(df_train[df_train['host_cat'] == cat]['answer'])  
 
    min_df_val = round(len(df_train[df_train['host_cat'] == cat]) - len(df_train[df_train['host_cat'] == cat]) * .99)
    max_df_val = round(len(df_train[df_train['host_cat'] == cat]) - len(df_train[df_train['host_cat'] == cat]) * .3)
    
    # Applying TFIDF 
    vectorizer = TfidfVectorizer(ngram_range = (2,2), min_df=5, stop_words='english',
                                 max_df=.5) 
    X2 = vectorizer.fit_transform(df_train[df_train['host_cat'] == cat]['answer']) 
    features = (vectorizer.get_feature_names()) 
    scores = (X2.toarray()) 

    # Getting top ranking features 
    sums = X2.sum(axis = 0) 
    data1 = [] 
    
    for col, term in enumerate(features): 
        data1.append( (term, sums[0,col] )) 

    ranking = pd.DataFrame(data1, columns = ['term','rank']) 
    words = (ranking.sort_values('rank', ascending = False))[:10]
    
    sns.barplot(x='term', y='rank', data=words, ax=ax, 
                color='blue', orient='v')
    ax.set_title(f"Top rank Trigram of: {cat}")
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
    ax.set_ylabel(' ')
    ax.set_xlabel(" ")

plt.subplots_adjust(top = 0.95, hspace=.9, wspace=.1)

plt.show()




extra_cols = ['host_cat', 'question_n_words', 'Target_PCA0',
              'Target_PCA1', 'Target_PCA2', 'ans_polarity', 'ans_subjectivity']




y_train = df_train[target_cols].copy()
X_train = df_train.drop(list(extra_cols) + list(target_cols), axis=1)
del df_train

X_test = df_test.copy()
del df_test




## Shell 
get_ipython().system('pip install ../input/sacremoses > /dev/null')

import sys
sys.path.insert(0, "../input/transformers/")




import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
# import tensorflow_hub as hub
import tensorflow as tf
# import bert_tokenization as tokenization
import tensorflow.keras.backend as K
import os
from scipy.stats import spearmanr
from math import floor, ceil
from transformers import *

from transformers import *

np.set_printoptions(suppress=True)
print(tf.__version__)




np.set_printoptions(suppress=True)

from transformers import *

BERT_PATH = '../input/bert-base-uncased-huggingface-transformer/'
tokenizer = BertTokenizer.from_pretrained(BERT_PATH+'bert-base-uncased-vocab.txt')

MAX_SEQUENCE_LENGTH = 512




## The function to creat the masks using to the title, question and answer
def _convert_to_transformer_inputs(title, question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for transformer (including bert)"""
    
    def return_id(str1, str2, truncation_strategy, length):

        inputs = tokenizer.encode_plus(str1, str2,
            add_special_tokens=True,
            max_length=length,
            truncation_strategy=truncation_strategy)
        
        input_ids =  inputs["input_ids"]
        input_masks = [1] * len(input_ids)
        input_segments = inputs["token_type_ids"]
        padding_length = length - len(input_ids)
        padding_id = tokenizer.pad_token_id
        input_ids = input_ids + ([padding_id] * padding_length)
        input_masks = input_masks + ([0] * padding_length)
        input_segments = input_segments + ([0] * padding_length)
        
        return [input_ids, input_masks, input_segments]
    
    input_ids_q, input_masks_q, input_segments_q = return_id(
        title + ' ' + question, None, 'longest_first', max_sequence_length)
    
    input_ids_a, input_masks_a, input_segments_a = return_id(
        answer, None, 'longest_first', max_sequence_length)
    
    return [input_ids_q, input_masks_q, input_segments_q,
            input_ids_a, input_masks_a, input_segments_a]

# Computing the inputs
def compute_input_arrays(df, columns, tokenizer, max_sequence_length):
    
    input_ids_q, input_masks_q, input_segments_q = [], [], []
    input_ids_a, input_masks_a, input_segments_a = [], [], []
    
    for _, instance in tqdm(df[columns].iterrows()):
        
        t, q, a = instance.question_title, instance.question_body, instance.answer

        ids_q, masks_q, segments_q, ids_a, masks_a, segments_a =         _convert_to_transformer_inputs(t, q, a, tokenizer, max_sequence_length)
        
        input_ids_q.append(ids_q)
        input_masks_q.append(masks_q)
        input_segments_q.append(segments_q)

        input_ids_a.append(ids_a)
        input_masks_a.append(masks_a)
        input_segments_a.append(segments_a)
        
    return [np.asarray(input_ids_q, dtype=np.int32), 
            np.asarray(input_masks_q, dtype=np.int32), 
            np.asarray(input_segments_q, dtype=np.int32),
            np.asarray(input_ids_a, dtype=np.int32), 
            np.asarray(input_masks_a, dtype=np.int32), 
            np.asarray(input_segments_a, dtype=np.int32)]


def compute_output_arrays(df, columns):
    return np.asarray(df[columns])





## Computing the error metric to the model optimization
def compute_spearmanr_ignore_nan(trues, preds):
    rhos = []
    for tcol, pcol in zip(np.transpose(trues), np.transpose(preds)):
        rhos.append(spearmanr(tcol, pcol).correlation)
    return np.nanmean(rhos)

def create_model():
    q_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    a_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    
    q_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    a_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    
    q_atn = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    a_atn = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    
    config = BertConfig() # print(config) to see settings
    config.output_hidden_states = False # Set to True to obtain hidden states
    # caution: when using e.g. XLNet, XLNetConfig() will automatically use xlnet-large config
    
    # normally ".from_pretrained('bert-base-uncased')", but because of no internet, the 
    # pretrained model has been downloaded manually and uploaded to kaggle. 
    bert_model = TFBertModel.from_pretrained(
        BERT_PATH+'bert-base-uncased-tf_model.h5', config=config)
    
    # if config.output_hidden_states = True, obtain hidden states via bert_model(...)[-1]
    q_embedding = bert_model(q_id, attention_mask=q_mask, token_type_ids=q_atn)[0]
    a_embedding = bert_model(a_id, attention_mask=a_mask, token_type_ids=a_atn)[0]
    
    q = tf.keras.layers.GlobalAveragePooling1D()(q_embedding)
    a = tf.keras.layers.GlobalAveragePooling1D()(a_embedding)
    
    x = tf.keras.layers.Concatenate()([q, a])
    
    x = tf.keras.layers.Dropout(0.2)(x)
    
    x = tf.keras.layers.Dense(30, activation='sigmoid')(x)

    model = tf.keras.models.Model(inputs=[q_id, q_mask, q_atn, a_id, a_mask, a_atn,], outputs=x)
    
    return model




outputs = compute_output_arrays(y_train, y_train.columns)
inputs = compute_input_arrays(X_train, X_train.columns, tokenizer, MAX_SEQUENCE_LENGTH)
test_inputs = compute_input_arrays(X_test, X_test.columns, tokenizer, MAX_SEQUENCE_LENGTH)




## Creating Kfold with 5 splits 
gkf = GroupKFold(n_splits=5).split(X=X_train.question_body, groups=X_train.question_body)

## to receive predictions
valid_preds = []
test_preds = []

## Looping throught the folds
for fold, (train_idx, valid_idx) in enumerate(gkf):
    
    # will actually only do 2 folds (out of 5) to manage < 2h
    if fold in [0 , 2, 4]:
        
        ## Train index from Kfold 
        train_inputs = [inputs[i][train_idx] for i in range(len(inputs))]
        train_outputs = outputs[train_idx]
        ## Valid index from Kfold 
        valid_inputs = [inputs[i][valid_idx] for i in range(len(inputs))]
        valid_outputs = outputs[valid_idx]
        
        K.clear_session()
        
        ## Instantiating the Bert Model
        model = create_model()
        optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
        model.compile(loss='binary_crossentropy', optimizer=optimizer)
        
        ## Fiting the model
        model.fit(train_inputs, train_outputs, epochs=2, batch_size=6)
        
        # model.save_weights(f'bert-{fold}.h5')
        valid_preds.append(model.predict(valid_inputs))
        # predicting the test set and appending to test_preds
        test_preds.append(model.predict(test_inputs))
        
        # Calculating the error in the valid set
        rho_val = compute_spearmanr_ignore_nan(valid_outputs, valid_preds[-1])
        print('validation score = ', rho_val)




df_sub.iloc[:, 1:] = np.average(test_preds, axis=0) # for weighted average set weights=[...]

df_sub.to_csv('submission.csv', index=False)
















