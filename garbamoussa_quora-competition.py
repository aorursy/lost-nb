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














get_ipython().run_line_magic('ls', '')





import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from zipfile import ZipFile
from math import radians, cos, sin, asin, sqrt
from datetime import datetime
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk import wordnet, pos_tag
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords, wordnet as wn
import re
import string




train = pd.read_csv('/kaggle/input/quora-insincere-questions-classification/train.csv')
test = pd.read_csv('/kaggle/input/quora-insincere-questions-classification/test.csv')
sub_df_target = pd.read_csv('/kaggle/input/quora-insincere-questions-classification/sample_submission.csv')




train.shape




train.head()




train1 = train.sample(frac=0.3,random_state=200)




train1.shape




test.shape




test1 = test.sample(frac=0.3,random_state=200)




test1.shape




import pandas as pd 
import numpy as np
import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')

#cora_data = pd.read_csv('all/train.csv')
#cora_test_data = pd.read_csv('all/test.csv')
cora_data = train1.copy()
cora_test_data = test1.copy()




cora_data.info()




#Verification presence données manquantes
cora_data[cora_data['question_text'].isnull()]




#Verification presence données manquantes
cora_data[cora_data['target'].isnull()]




percent_target = cora_data.groupby('target').count()
percent_target['percent'] = 100*(percent_target['question_text']/cora_data['target'].count())
percent_target.reset_index(level=0, inplace=True)
percent_target




import matplotlib.pyplot as plt

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Toxic contents','Positive Questions'
sizes = [6, 94]
explode = (0.1, 0)  # only "explode" the 1st slice (i.e. 'Toxic contents')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.0f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()




cora_data_neg_sample = cora_data[cora_data['target'] == 1] #Negatives comments
cora_data_positive_sample = cora_data[cora_data['target'] == 0].reindex()  #Positive Comments

cora_resampling = pd.concat([pd.DataFrame(cora_data_positive_sample.sample(6000)), #130000
                               pd.DataFrame(cora_data_neg_sample.sample(3650))])




100*(cora_resampling.groupby('target')['question_text'].count())/cora_resampling['target'].count()




import matplotlib.pyplot as plt

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Toxic contents','Positive Questions'
sizes = [38, 62]
explode = (0.1, 0)  # only "explode" the 1st slice (i.e. 'Toxic contents')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.0f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()




from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk import wordnet, pos_tag
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords, wordnet as wn
import re
import string

#Cleaning data

def clean_str(chaine):
    chaine = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", chaine)     
    chaine = re.sub(r"\'s", " \'s", chaine) 
    chaine = re.sub(r"\'ve", " \'ve", chaine) 
    chaine = re.sub(r"n\'t", " n\'t", chaine) 
    chaine = re.sub(r"\'re", " \'re", chaine) 
    chaine = re.sub(r"\'d", " \'d", chaine) 
    chaine = re.sub(r"\'ll", " \'ll", chaine) 
    chaine = re.sub(r",", " , ", chaine) 
    chaine = re.sub(r"!", " ! ", chaine) 
    chaine = re.sub(r"\(", " \( ", chaine) 
    chaine = re.sub(r"\)", " \) ", chaine) 
    chaine = re.sub(r"\?", " \? ", chaine) 
    chaine = re.sub(r"\s{2,}", " ", chaine)
    chaine = chaine.lower() #convert all text in lower case
    chaine = chaine.replace(' +', ' ') # Remove double space
    chaine = chaine.strip() # Remove trailing space at the beginning or end
    chaine = chaine.replace('[^a-zA-Z]', ' ' )# Everything not a alphabet character replaced with a space
    #words =  [word for word in chaine.split() if word not in [i for i in string.punctuation]] #Remove punctuations
    words =  [word for word in chaine.split() if word.isalpha()] #droping numbers and punctuations
    return ' '.join(words)

#Tokenization and punctuation removing and stopwords
def tokeniZ_stopWords(chaine):
    chaine = word_tokenize(chaine)
    list_stopWords = set(stopwords.words('english'))
    words = [word for word in chaine if word not in list_stopWords]
    return words

#Stemming 
ps = PorterStemmer()
sb = SnowballStemmer('english')

#Lemmatization
def lemat_words(tokens_list):
    from collections import defaultdict
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    lemma_function = WordNetLemmatizer()
    return [lemma_function.lemmatize(token, tag_map[tag[0]]) for token, tag in pos_tag(tokens_list)]
    #for token, tag in pos_tag(tokens_list):
     #   lemma = lemma_function.lemmatize(token, tag_map[tag[0]])

# Define Ngrams function
def get_ngrams(text, n ):
    n_grams = ngrams(word_tokenize(text), n)
    return [ ' '.join(grams) for grams in n_grams]




#Cleaning the data 
cora_resampling['clean_question'] = cora_resampling['question_text'].apply(clean_str)




#Tokenizing and stopwords removing
cora_resampling['tokeniZ_stopWords_question'] = cora_resampling['clean_question'].apply(tokeniZ_stopWords)




#Words Stemming
cora_resampling['stemming_question'] = [[ps.stem(word) for word in words] for words in cora_resampling['tokeniZ_stopWords_question'] ]
cora_resampling['stemming_question_for_tfidf'] = [' '.join(words) for words in cora_resampling['stemming_question']] 




#Words lemmatization
cora_resampling['lemmatize_question'] = cora_resampling['tokeniZ_stopWords_question'].apply(lemat_words)
cora_resampling['lemmatize_question_for_tfidf'] = [' '.join(x) for x in cora_resampling['lemmatize_question'] ]




#Calcul longueur des commentaires
cora_resampling['question_lenght'] = cora_resampling['question_text'].apply(len)




#Calcul du nombre de ponctuation par question
from string import punctuation
cora_resampling['number_punctuation'] = cora_resampling['question_text'].apply(
    lambda doc: len([word for word in str(doc) if word in punctuation])) 




#Number of unique words in the text
cora_resampling['number_of_Unique_words'] = cora_resampling['clean_question'].apply([lambda x : len(set(str(x).split()))])




#Number of stopwords in the text
list_stopWords = set(stopwords.words('english'))
cora_resampling['number_of_StopWords'] = cora_resampling['clean_question'].apply(
    lambda x : len([w for w in x.lower().split() if w in list_stopWords ]))




#Number of upper case words
cora_resampling['number_of_uppercase'] = cora_resampling['question_text'].apply(
    lambda x : len([w for w in x.split() if w.isupper()]))




#Average length of words in the text (whithout stop words)
cora_resampling['average_of_wordsLength'] = cora_resampling['clean_question'].apply(
    lambda x : np.mean([len(w) for w in x.split()]))




#Number of words in the text
cora_resampling['number_of_words'] = cora_resampling['clean_question'].apply([lambda x : len(str(x).split())])




cora_resampling.info()




cora_resampling[['question_lenght', 'number_punctuation', 'number_of_words',
       'number_of_Unique_words', 'number_of_StopWords', 'number_of_uppercase',
       'average_of_wordsLength']].sample(5)




import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')




list_var=['question_lenght', 'number_punctuation', 'number_of_Unique_words', 
          'number_of_StopWords', 'number_of_uppercase', 'average_of_wordsLength']
def var_hist_global(df,X='target',Y=list_var, Title='Features Engineering - Histograms', KDE=False):
    fig, ((ax1, ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3, 2 ,figsize=(14,16))#, sharey=True )
    aX = [ax1, ax2,ax3,ax4,ax5,ax6]
    for i in range(len(list_var)):   
        sns.distplot( df[list_var[i]][df[X]== 1 ].dropna(), label="unsinceres questions" , ax= aX[i], kde= KDE , color = 'red')           
        sns.distplot( df[list_var[i]][df[X]== 0 ].dropna(), label="Sinceres questions"   , ax= aX[i], kde= KDE , color = "olive")
    plt.legend()
    plt.title(Title)
    #plt.show()
    plt.savefig("Features_Engineering_Histograms")
    
var_hist_global(df=cora_resampling,X='target',Y=list_var, Title='Histogramme Quora Questions', KDE=True)




# Calculate number of obs per group & median to position labels
list_var = ['question_lenght', 'number_of_Unique_words', 'number_of_StopWords']
def violin_boxplott(df,X='target',Y=list_var, Title='Features Engineering - Box plot'): 
    fig, (ax1, ax2 ,ax3) = plt.subplots(1,3 ,figsize=(14,8))#, sharey=True )
    medians = cora_resampling.groupby(['target'])['question_lenght', 'number_of_Unique_words', 'number_of_StopWords'].median().values
 
    sns.boxplot( y=list_var[0],  x=X , data = df, ax= ax1 , palette=['olive','red'])
    sns.boxplot( y=list_var[1],  x=X , data = df, ax= ax2 , palette=['olive','red'])
    sns.boxplot( y=list_var[2],  x=X , data = df, ax= ax3 , palette=['olive','red'])
    #plt.title(Title)
    plt.savefig("Features_Engineering_Boxplot")
violin_boxplott(df=cora_resampling)




from wordcloud import WordCloud, STOPWORDS
# Code recuperer de : https://www.kaggle.com/aashita/word-clouds-of-various-shapes ##
def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0,16.0), 
                   title = None, title_size=40, image_color=False):
    stopwords = set(STOPWORDS)
    more_stopwords = {'one', 'quora','br', 'Po', 'th', 'sayi', 'fo', 'Unknown','will','say','now','must','want','much','talks','buy','dont','use','etc','go','ago','lot','ki', 'ba'}
    stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(background_color='black',
                    stopwords = stopwords,
                    max_words = max_words,
                    max_font_size = max_font_size, 
                    random_state = 42,
                    width=800, 
                    height=400,
                    mask = mask)
    wordcloud.generate(str(text))
    
    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
        plt.title(title, fontdict={'size': title_size,  
                                  'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud);
        plt.title(title, fontdict={'size': title_size, 'color': 'black', 
                                  'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.savefig(title)
    plt.tight_layout()  




#plot_wordcloud(cora_data_neg_sample["question_text"], title="Word Cloud of insincere Questions")
#plot_wordcloud(cora_resampling['tokeniZ_stopWords_question'][cora_resampling['target']== 1]) 
plot_wordcloud(cora_resampling['lemmatize_question_for_tfidf'][cora_resampling['target']== 1], title="Word Cloud of insincere Questions") 




#plot_wordcloud(cora_data_positive_sample["question_text"], title="Word Cloud of sincere Questions")
plot_wordcloud(cora_resampling['lemmatize_question_for_tfidf'][cora_resampling['target']== 0], title="Word Cloud of sincere Questions")




# Code source : https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-qiqc
from collections import defaultdict
from wordcloud import  STOPWORDS
import plotly.graph_objs as go
from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)

#train1_df = train_df[train_df["target"]==1]
#train0_df = train_df[train_df["target"]==0]
train1_df = cora_resampling[cora_resampling['target']==1]
train0_df = cora_resampling[cora_resampling['target']==0]
## custom function for ngram generation ##
def generate_ngrams(text, n_gram=1):
    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]

## custom function for horizontal bar chart ##
def horizontal_bar_chart(df, color):
    trace = go.Bar(
        y=df["word"].values[::-1],
        x=df["wordcount"].values[::-1],
        showlegend=False,
        orientation = 'h',
        marker=dict(
            color=color,
        ),
    )
    return trace

## Get the bar chart from sincere questions ##
freq_dict = defaultdict(int)
for sent in train0_df["lemmatize_question_for_tfidf"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace0 = horizontal_bar_chart(fd_sorted.head(20), 'blue')

## Get the bar chart from insincere questions ##
freq_dict = defaultdict(int)
for sent in train1_df["lemmatize_question_for_tfidf"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace1 = horizontal_bar_chart(fd_sorted.head(50), 'blue')

# Creating two subplots
fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04,
                          subplot_titles=["Frequent words of sincere questions", 
                                          "Frequent words of insincere questions"])
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")
py.iplot(fig, filename='word-plots')

#plt.figure(figsize=(10,16))
#sns.barplot(x="ngram_count", y="ngram", data=fd_sorted.loc[:50,:], color="b")
#plt.title("Frequent words for Insincere Questions", fontsize=16)
#plt.show()




freq_dict = defaultdict(int)
for sent in train0_df["lemmatize_question_for_tfidf"]:
    for word in generate_ngrams(sent,2):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace0 = horizontal_bar_chart(fd_sorted.head(50), 'orange')


freq_dict = defaultdict(int)
for sent in train1_df["lemmatize_question_for_tfidf"]:
    for word in generate_ngrams(sent,2):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace1 = horizontal_bar_chart(fd_sorted.head(20), 'orange')

# Creating two subplots
fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04,horizontal_spacing=0.15,
                          subplot_titles=["Frequent bigrams of sincere questions", 
                                          "Frequent bigrams of insincere questions"])
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Bigram Count Plots - N_gram(2,2)")
py.iplot(fig, filename='word-plots')




freq_dict = defaultdict(int)
for sent in train0_df["lemmatize_question_for_tfidf"]:
    for word in generate_ngrams(sent,3):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace0 = horizontal_bar_chart(fd_sorted.head(50), 'green')


freq_dict = defaultdict(int)
for sent in train1_df["lemmatize_question_for_tfidf"]:
    for word in generate_ngrams(sent,3):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace1 = horizontal_bar_chart(fd_sorted.head(20), 'green')

# Creating two subplots
fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04, horizontal_spacing=0.2,
                          subplot_titles=["Frequent trigrams of sincere questions", 
                                          "Frequent trigrams of insincere questions"])
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig['layout'].update(height=1200, width=1200, paper_bgcolor='rgb(233,233,233)', title="Trigram Count Plots - N_gram(3,3)")
py.iplot(fig, filename='word-plots')




cora_resampling.columns




from sklearn.model_selection import train_test_split
X_cora_train, X_cora_test, y_cora_train, y_cora_test = train_test_split(
            cora_resampling[['clean_question', 'stemming_question_for_tfidf', 'lemmatize_question_for_tfidf',
                             'tokeniZ_stopWords_question', 'stemming_question', 'lemmatize_question',
                             'question_lenght', 'number_punctuation', 'number_of_StopWords', 'number_of_Unique_words', 'number_of_uppercase','average_of_wordsLength']]
            ,cora_resampling['target'], 
            test_size=0.3, random_state=42)
X_cora_train.shape, X_cora_test.shape, y_cora_train.shape, y_cora_test.shape




from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(  ngram_range=(1,1), 
                                     analyzer='word',
                                     stop_words='english', 
                                     lowercase=True, 
                                     max_df=0.9, # remove too frequent words
                                     min_df=10, # remove too rare words
                                     max_features = None, # max words in vocabulary, will keep most frequent words
                                     binary=False #If True, all non zero counts are set to 1. This is useful for discrete probabilistic models that model binary events rather than integer counts.
                                  )




#Stemmed questions vectorzation
X_tfidf_vectorizer_train = tfidf_vectorizer.fit_transform(X_cora_train['stemming_question_for_tfidf'])
X_tfidf_vectorizer_test = tfidf_vectorizer.transform(X_cora_test['stemming_question_for_tfidf'])




#Lemmentized questions vectorization
X_tfidf_Lem_vect_train = tfidf_vectorizer.fit_transform(X_cora_train['lemmatize_question_for_tfidf'])
X_tfidf_Lem_vect_test = tfidf_vectorizer.transform(X_cora_test['lemmatize_question_for_tfidf'])




#bigram questions vectorization
bigram_vectorizer = TfidfVectorizer(  ngram_range=(1,2), 
                                     analyzer='word',
                                     stop_words='english', 
                                     lowercase=True, 
                                     max_df=0.9, # remove too frequent words
                                     min_df=10, # remove too rare words
                                     max_features = None, # max words in vocabulary, will keep most frequent words
                                     binary=False #If True, all non zero counts are set to 1. This is useful for discrete probabilistic models that model binary events rather than integer counts.
                                  )
X_bigram_vectorizer_train = bigram_vectorizer.fit_transform(X_cora_train['stemming_question_for_tfidf'])
X_bigram_vectorizer_test = bigram_vectorizer.transform(X_cora_test['stemming_question_for_tfidf'])




#T3gram questions vectorization
t3gram_vectorizer = TfidfVectorizer(  ngram_range=(1,4), 
                                     analyzer='word',
                                     stop_words='english', 
                                     lowercase=True, 
                                     max_df=0.9, # remove too frequent words
                                     min_df=10, # remove too rare words
                                     max_features = None, # max words in vocabulary, will keep most frequent words
                                     binary=False #If True, all non zero counts are set to 1. This is useful for discrete probabilistic models that model binary events rather than integer counts.
                                  )
X_t3gram_vectorizer_train = t3gram_vectorizer.fit_transform(X_cora_train['stemming_question_for_tfidf'])
X_t3gram_vectorizer_test = t3gram_vectorizer.transform(X_cora_test['stemming_question_for_tfidf'])




#Range single word to t3gram questions vectorization
st3gram_vectorizer = TfidfVectorizer(  ngram_range=(1,3), 
                                     analyzer='word',
                                     stop_words='english', 
                                     lowercase=True, 
                                     max_df=0.9, # remove too frequent words
                                     min_df=10, # remove too rare words
                                     max_features = None, # max words in vocabulary, will keep most frequent words
                                     binary=False #If True, all non zero counts are set to 1. This is useful for discrete probabilistic models that model binary events rather than integer counts.
                                  )
X_Singt3gram_vectorizer_train = st3gram_vectorizer.fit_transform(X_cora_train['stemming_question_for_tfidf'])
X_Singt3gram_vectorizer_test  = st3gram_vectorizer.transform(X_cora_test['stemming_question_for_tfidf'])




X_Singt3gram_vectorizer_train




#Word2Vec with preprocessiong questions (without stopwords) 
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

d2v_training_data = []
for i, doc in enumerate(X_cora_train['stemming_question']):
    d2v_training_data.append(TaggedDocument(words=doc,tags=[i]))

# ========== learning doc embeddings with doc2vec ==========

# PV stands for 'Paragraph Vector'
# PV-DBOW (distributed bag-of-words) dm=0

d2v = Doc2Vec(d2v_training_data, vector_size=300, window=10, alpha=0.1, min_alpha=1e-4, dm=0, negative=1, epochs=10, min_count=2, workers=4)
d2v_vecs = np.zeros((len(X_cora_train['stemming_question']), 300))
for i in range(len(X_cora_train['stemming_question'])):
    d2v_vecs[i,:] = d2v.docvecs[i]
    
d2v_test = np.zeros((len(X_cora_test['stemming_question']), 300))
for i in range(len(X_cora_test['stemming_question'])):
    d2v_test[i,:] = d2v.infer_vector(X_cora_test['stemming_question'].iloc[i])




#Word2Vec with lemmatize words
d2v_training_data = []
for i, doc in enumerate(X_cora_train['lemmatize_question']):
    d2v_training_data.append(TaggedDocument(words=doc,tags=[i]))

# ========== learning doc embeddings with doc2vec ==========

# PV stands for 'Paragraph Vector'
# PV-DBOW (distributed bag-of-words) dm=0

d2v = Doc2Vec(d2v_training_data, vector_size=200, window=5, alpha=0.1, min_alpha=1e-4, 
              dm=0, negative=1, epochs=10, min_count=2, workers=4)
d2v_vecs_bigram = np.zeros((len(X_cora_train['lemmatize_question']), 200))
for i in range(len(X_cora_train['lemmatize_question'])):
    d2v_vecs_bigram[i,:] = d2v.docvecs[i]
    
d2v_test_bigram = np.zeros((len(X_cora_test['lemmatize_question']), 200))
for i in range(len(X_cora_test['lemmatize_question'])):
    d2v_test_bigram[i,:] = d2v.infer_vector(X_cora_test['lemmatize_question'].iloc[i])




from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, SelectPercentile
from sklearn.pipeline import Pipeline




features = SelectKBest(mutual_info_classif,k=2).fit(X_cora_train[['question_lenght', 'number_punctuation', 'number_of_StopWords', 'number_of_Unique_words', 
                                        'number_of_uppercase','average_of_wordsLength']].fillna(0),y_cora_train)
independance_test = np.zeros((6,2))
for idx,i in enumerate(['question_lenght', 'number_punctuation', 'number_of_StopWords', 'number_of_Unique_words', 'number_of_uppercase','average_of_wordsLength']):
    #independance_test[idx,0]= features.pvalues_[idx]
    independance_test[idx,1]= features.scores_[idx]
    #print (i,features.pvalues_[idx],features.scores_[idx])
    #print('%s  %s'%(i,features.scores_[idx]))




list_var=['question_lenght', 'number_punctuation', 'number_of_StopWords', 'number_of_Unique_words', 'number_of_uppercase','average_of_wordsLength']
independance_df = pd.DataFrame({'Variables': list_var, 'p_values': independance_test[:,0], 'MI': independance_test[:,1]},index=None)
independance_df




plt.figure(figsize=(12, 10))
_ = sns.heatmap(cora_resampling[['question_lenght', 'number_punctuation', 'number_of_StopWords', 
                                 'number_of_Unique_words', 'number_of_uppercase','average_of_wordsLength']].corr()
                ,cmap="YlGnBu", annot=True, fmt=".2f")
plt.savefig("Correlation Matrice")
plt.show()




from sklearn.metrics import accuracy_score,roc_auc_score, f1_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve
from sklearn.model_selection import StratifiedShuffleSplit
from matplotlib import pyplot as plt
plt.style.use('seaborn-whitegrid')

def plot_learning_curve(estimator1, X, y, estimator2, ylim=(0, 1.1), cv=2, n_jobs=-1, 
                        train_sizes=np.linspace(.1, 1.0, 5), scoring=None):
    
    
    plt.figure(figsize=(12,6))
    #plt.title("Learning curves for %s" % type(estimator1).__name__)
    #plt.title("Learning curves for %s" %(estimator1))
    plt.grid()
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.xscale('log')
    train_sizes, train_scores, test_scores = learning_curve(
        estimator1, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
        scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.subplot(1, 2, 1)
    plt.title("Learning curves for %s" % type(estimator1).__name__)
    plt.plot(
        train_sizes, train_scores_mean, 'o-',
        color="r", #linewidth=3, 
        label="Training score")
    plt.plot(
        train_sizes, test_scores_mean, 'o-',
        color="olive", 
        label="Cross-validation score")
    plt.fill_between(
        train_sizes, train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std, alpha=0.1,
        color="firebrick")
    plt.fill_between(
        train_sizes, test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std, alpha=0.1, color="darkgoldenrod")
    plt.legend(loc="best")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator2, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
        scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.subplot(1, 2, 2)
    plt.title("Learning curves for %s with 70 percent of best features" % type(estimator1).__name__)
    plt.plot(
        train_sizes, train_scores_mean, 'o-',
        color="r", #linewidth=3, 
        label="Training score")
    plt.plot(
        train_sizes, test_scores_mean, 'o-',
        color="olive", 
        label="Cross-validation score")
    plt.fill_between(
        train_sizes, train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std, alpha=0.1,
        color="firebrick")
    plt.fill_between(
        train_sizes, test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std, alpha=0.1, color="darkgoldenrod")
    plt.legend(loc="best")
    plt.savefig("Learning curves for %s" % type(estimator1).__name__)




df_models = pd.DataFrame({'Models':[], 'Sample':[], 'Accuracy':[],'Accuracy with 70% best features':[]})
data=[]

def modelize(list_clf,X,y,X_test,y_test):
    selector = SelectPercentile(mutual_info_classif,percentile=70)
    for clf in list_clf:
        Clf1 = clf().fit(X,y)
        Clf2 = Pipeline([('Feature Selection',selector),('Classification',clf())]).fit(X,y)
        cv = StratifiedShuffleSplit(n_splits=2 , test_size=.3, random_state=51)
        print('Model : %s' %type(Clf1).__name__)
        print('With all features                                              /    With 70% of the best features')
        print('                 Accuracy Score             F1 Score                Accuracy Score             F1 Score')
        print('training :       %f                   %f           /    %f                   %f' 
              %(accuracy_score(y,Clf1.predict(X)),f1_score(y,Clf1.predict(X),average='weighted'),accuracy_score(y,Clf2.predict(X)),f1_score(y,Clf2.predict(X),average='weighted')))
        print('Test     :       %f                   %f           /    %f                   %f' 
              %(accuracy_score(y_test,Clf1.predict(X_test)),f1_score(y_test,Clf1.predict(X_test),average='weighted'),accuracy_score(y_test,Clf2.predict(X_test)),f1_score(y_test,Clf2.predict(X_test),average='weighted')))
        data.insert(0,{'Models':type(Clf1).__name__, 'Sample':'Training', 'Accuracy':accuracy_score(y,Clf1.predict(X)),'Accuracy with 70% best features':accuracy_score(y,Clf2.predict(X)) })
        data.insert(0,{'Models':type(Clf1).__name__, 'Sample':'Test', 'Accuracy':accuracy_score(y_test,Clf1.predict(X_test)),'Accuracy with 70% best features':accuracy_score(y_test,Clf2.predict(X_test)) })
        #plot_learning_curve(Clf1, X, y, Clf2, cv, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), scoring='accuracy')
        print('=============================================')




from sklearn.linear_model import LogisticRegressionCV , PassiveAggressiveClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier , ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import NuSVC, LinearSVC, SVC, OneClassSVM
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB




generalized_linear_model = [LogisticRegressionCV,PassiveAggressiveClassifier]
support_vector_machines =  [NuSVC, LinearSVC]   
decisionTreeClassification=[DecisionTreeClassifier]
ensemble_methods = [RandomForestClassifier , ExtraTreesClassifier,AdaBoostClassifier, GradientBoostingClassifier]
naive_bayes_model = [GaussianNB, MultinomialNB, ComplementNB]




#Generalized Linear Model
modelize(generalized_linear_model,X_tfidf_vectorizer_train,y_cora_train,X_tfidf_vectorizer_test,y_cora_test)




def modelize_svc(list_clf,X,y,X_test,y_test):
    selector = SelectPercentile(mutual_info_classif,percentile=70)
    for clf in list_clf:
        Clf1 = clf(kernel='poly').fit(X,y)
        Clf2 = Pipeline([('Feature Selection',selector),('Classification',clf(kernel='poly'))]).fit(X,y)
        cv = StratifiedShuffleSplit(n_splits=2 , test_size=.3, random_state=51)
        print('Model : %s' %type(Clf1).__name__)
        print('With all features                                              /    With 70% of the best features')
        print('                 Accuracy Score             F1 Score                Accuracy Score             F1 Score')
        print('training :       %f                   %f           /    %f                   %f' 
              %(accuracy_score(y,Clf1.predict(X)),f1_score(y,Clf1.predict(X),average='weighted'),accuracy_score(y,Clf2.predict(X)),f1_score(y,Clf2.predict(X),average='weighted')))
        print('Test     :       %f                   %f           /    %f                   %f' 
              %(accuracy_score(y_test,Clf1.predict(X_test)),f1_score(y_test,Clf1.predict(X_test),average='weighted')
                ,accuracy_score(y_test,Clf2.predict(X_test)),f1_score(y_test,Clf2.predict(X_test),average='weighted')))
        data.insert(0,{'Models':type(Clf1).__name__, 'Sample':'Training', 'Accuracy':accuracy_score(y_test,Clf1.predict(X_test))
                       ,'Accuracy with 70% best features':accuracy_score(y_test,Clf2.predict(X_test)) })
        data.insert(0,{'Models':type(Clf1).__name__, 'Sample':'Test', 'Accuracy':accuracy_score(y_test,Clf1.predict(X_test))
                       ,'Accuracy with 70% best features':accuracy_score(y_test,Clf2.predict(X_test)) })
        #plot_learning_curve(Clf1, X, y, Clf2, cv, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), scoring='accuracy')
        print('=============================================')
#Support Vector Machine
#modelize_svc([SVC],X_tfidf_vectorizer_train,y_cora_train,X_tfidf_vectorizer_test,y_cora_test)




#Support Vector Machine
modelize(support_vector_machines,X_tfidf_vectorizer_train,y_cora_train,X_tfidf_vectorizer_test,y_cora_test)




#Decision Tree
modelize(decisionTreeClassification,X_tfidf_vectorizer_train,y_cora_train,X_tfidf_vectorizer_test,y_cora_test)




#Ensemble Methods
modelize(ensemble_methods,X_tfidf_vectorizer_train,y_cora_train,X_tfidf_vectorizer_test,y_cora_test)       




#Naives bayes
modelize([ MultinomialNB, ComplementNB],X_tfidf_vectorizer_train,y_cora_train,X_tfidf_vectorizer_test,y_cora_test)




#Neural Network - Consomme beaucoup trop d'energie
Neural_network= [MLPClassifier]
modelize(Neural_network,X_tfidf_vectorizer_train,y_cora_train,X_tfidf_vectorizer_test,y_cora_test)




models_df =pd.concat([pd.DataFrame(data),df_models],ignore_index=True)  
models_df.sort_values(by=['Accuracy','Models','Sample'], ascending=False, axis=0, inplace=True)
test_df = pd.DataFrame(np.array(models_df[['Accuracy', 'Accuracy with 70% best features']]),
                       index=[models_df['Models'],models_df['Sample']], columns=['Accuracy', 'Accuracy with 70% best features'])




test_df




test_df1=pd.merge(models_df[models_df['Sample']=='Test'],models_df[models_df['Sample']=='Training'],how='inner',on='Models')




final_model = pd.DataFrame( np.array(test_df1[['Sample_y','Accuracy_y', 'Accuracy with 70% best features_y','Sample_x','Accuracy_x', 'Accuracy with 70% best features_x']]),
            index=test_df1['Models'], columns=['','Accuracy', 'Accuracy with 70% best features','','Accuracy1', 'Accuracy with 70% best features'])




#final_model= pd.read_csv('models_stemmisation.csv')
final_model.sort_values(by=['Accuracy1'], ascending=False, axis=0, inplace=True)
final_model.rename(index=str,columns={'Unnamed: 1':'','Unnamed: 4':'', 'Accuracy':'Accuracy1', 'Accuracy with 70% best features.1':'Accuracy with 70% best features'},inplace=True)




final_model




#Nous utiliserons dans cette étape les modèles preselectionnés dans l'étape précedente.
generalized_linear_model2 = [LogisticRegressionCV,PassiveAggressiveClassifier]
support_vector_machines2 =  [LinearSVC,NuSVC]
ensemble_methods2 = [RandomForestClassifier , ExtraTreesClassifier]
Neural_network= [MLPClassifier]
naive_bayes_model = [MultinomialNB, ComplementNB]




df_models = pd.DataFrame({'Models':[], 'Sample':[], 'Accuracy':[],'Accuracy with 70% best features':[]})
data=[]




modelize(generalized_linear_model,X_tfidf_Lem_vect_train,y_cora_train,X_tfidf_Lem_vect_test,y_cora_test)




modelize(support_vector_machines,X_tfidf_Lem_vect_train,y_cora_train,X_tfidf_Lem_vect_test,y_cora_test)




modelize(ensemble_methods,X_tfidf_Lem_vect_train,y_cora_train,X_tfidf_Lem_vect_test,y_cora_test)




modelize(naive_bayes_model,X_tfidf_Lem_vect_train,y_cora_train,X_tfidf_Lem_vect_test,y_cora_test)




get_ipython().run_cell_magic('time', '', 'modelize(Neural_network,X_tfidf_Lem_vect_train,y_cora_train,X_tfidf_Lem_vect_test,y_cora_test)')




get_ipython().run_cell_magic('time', '', 'modelize([DecisionTreeClassifier, SVC],X_tfidf_Lem_vect_train,y_cora_train,X_tfidf_Lem_vect_test,y_cora_test)')




models_df =pd.concat([pd.DataFrame(data),df_models],ignore_index=True)  
#models_df.sort_values(by=['Accuracy','Models','Sample'], ascending=False, axis=0, inplace=True)
#test_df = pd.DataFrame(np.array(models_df[['Accuracy', 'Accuracy with 70% best features']]),
#                       index=[models_df['Models'],models_df['Sample']], columns=['Accuracy', 'Accuracy with 70% best features'])
test_df1=pd.merge(models_df[models_df['Sample']=='Test'],models_df[models_df['Sample']=='Training'],how='inner',on='Models')
test_df1.sort_values(by=['Accuracy_x'], ascending=False, axis=0, inplace=True)
final_model2 = pd.DataFrame( np.array(test_df1[['Sample_y','Accuracy_y', 'Accuracy with 70% best features_y','Sample_x','Accuracy_x', 'Accuracy with 70% best features_x']]),
            index=test_df1['Models'], columns=['','Accuracy', 'Accuracy with 70% best features','','Accuracy1', 'Accuracy with 70% best features'])

final_model2.rename(index=str,columns={'Unnamed: 1':'','Unnamed: 4':'', 'Accuracy':'Accuracy1', 'Accuracy with 70% best features.1':'Accuracy with 70% best features'},inplace=True)
final_model2.to_csv('models_stemmisation1_2.csv')
final_model2




df_models = pd.DataFrame({'Models':[], 'Sample':[], 'Accuracy':[],'Accuracy with 70% best features':[]})
data=[]




modelize(generalized_linear_model2,X_bigram_vectorizer_train,y_cora_train,X_bigram_vectorizer_test,y_cora_test)




modelize(support_vector_machines2,X_bigram_vectorizer_train,y_cora_train,X_bigram_vectorizer_test,y_cora_test)




modelize(ensemble_methods2,X_bigram_vectorizer_train,y_cora_train,X_bigram_vectorizer_test,y_cora_test)




modelize(naive_bayes_model,X_bigram_vectorizer_train,y_cora_train,X_bigram_vectorizer_test,y_cora_test)




modelize(Neural_network,X_bigram_vectorizer_train,y_cora_train,X_bigram_vectorizer_test,y_cora_test)




models_df =pd.concat([pd.DataFrame(data),df_models],ignore_index=True)  
#models_df.sort_values(by=['Accuracy','Models','Sample'], ascending=False, axis=0, inplace=True)
#test_df = pd.DataFrame(np.array(models_df[['Accuracy', 'Accuracy with 70% best features']]),
#                       index=[models_df['Models'],models_df['Sample']], columns=['Accuracy', 'Accuracy with 70% best features'])
test_df1=pd.merge(models_df[models_df['Sample']=='Test'],models_df[models_df['Sample']=='Training'],how='inner',on='Models')
test_df1.sort_values(by=['Accuracy_x'], ascending=False, axis=0, inplace=True)
final_model2 = pd.DataFrame( np.array(test_df1[['Sample_y','Accuracy_y', 'Accuracy with 70% best features_y','Sample_x','Accuracy_x', 'Accuracy with 70% best features_x']]),
            index=test_df1['Models'], columns=['','Accuracy', 'Accuracy with 70% best features','','Accuracy', 'Accuracy with 70% best features'])

final_model2.rename(index=str,columns={'Unnamed: 1':'','Unnamed: 4':'', 'Accuracy.1':'Accuracy', 'Accuracy with 70% best features.1':'Accuracy with 70% best features'},inplace=True)
final_model2.to_csv('models_stemmisation1_2.csv')
final_model2




df_models = pd.DataFrame({'Models':[], 'Sample':[], 'Accuracy':[],'Accuracy with 70% best features':[]})
data=[]




modelize(generalized_linear_model2,X_Singt3gram_vectorizer_train,y_cora_train,X_Singt3gram_vectorizer_test,y_cora_test) 




modelize(support_vector_machines2,X_Singt3gram_vectorizer_train,y_cora_train,X_Singt3gram_vectorizer_test,y_cora_test)




modelize(ensemble_methods2,X_Singt3gram_vectorizer_train,y_cora_train,X_Singt3gram_vectorizer_test,y_cora_test)




modelize(naive_bayes_model,X_Singt3gram_vectorizer_train,y_cora_train,X_Singt3gram_vectorizer_test,y_cora_test)




modelize(Neural_network,X_Singt3gram_vectorizer_train,y_cora_train,X_Singt3gram_vectorizer_test,y_cora_test)




models_df =pd.concat([pd.DataFrame(data),df_models],ignore_index=True)  
#models_df.sort_values(by=['Accuracy','Models','Sample'], ascending=False, axis=0, inplace=True)
#test_df = pd.DataFrame(np.array(models_df[['Accuracy', 'Accuracy with 70% best features']]),
#                       index=[models_df['Models'],models_df['Sample']], columns=['Accuracy', 'Accuracy with 70% best features'])
test_df1=pd.merge(models_df[models_df['Sample']=='Test'],models_df[models_df['Sample']=='Training'],how='inner',on='Models')
test_df1.sort_values(by=['Accuracy_x'], ascending=False, axis=0, inplace=True)
final_model2 = pd.DataFrame( np.array(test_df1[['Sample_y','Accuracy_y', 'Accuracy with 70% best features_y','Sample_x','Accuracy_x', 'Accuracy with 70% best features_x']]),
            index=test_df1['Models'], columns=['','Accuracy', 'Accuracy with 70% best features','','Accuracy', 'Accuracy with 70% best features'])

final_model2.rename(index=str,columns={'Unnamed: 1':'','Unnamed: 4':'', 'Accuracy.1':'Accuracy', 'Accuracy with 70% best features.1':'Accuracy with 70% best features'},inplace=True)
final_model2.to_csv('models_stemmisaton1_3.csv')
final_model2




df_models = pd.DataFrame({'Models':[], 'Sample':[], 'Accuracy':[],'Accuracy with 70% best features':[]})
data=[]




modelize(generalized_linear_model2,X_t3gram_vectorizer_train,y_cora_train,X_t3gram_vectorizer_test,y_cora_test)




modelize(support_vector_machines2,X_t3gram_vectorizer_train,y_cora_train,X_t3gram_vectorizer_test,y_cora_test)




modelize(ensemble_methods2,X_t3gram_vectorizer_train,y_cora_train,X_t3gram_vectorizer_test,y_cora_test)




modelize(naive_bayes_model,X_t3gram_vectorizer_train,y_cora_train,X_t3gram_vectorizer_test,y_cora_test)




modelize(Neural_network,X_t3gram_vectorizer_train,y_cora_train,X_t3gram_vectorizer_test,y_cora_test)




models_df =pd.concat([pd.DataFrame(data),df_models],ignore_index=True)  
#models_df.sort_values(by=['Accuracy','Models','Sample'], ascending=False, axis=0, inplace=True)
#test_df = pd.DataFrame(np.array(models_df[['Accuracy', 'Accuracy with 70% best features']]),
#                       index=[models_df['Models'],models_df['Sample']], columns=['Accuracy', 'Accuracy with 70% best features'])
test_df1=pd.merge(models_df[models_df['Sample']=='Test'],models_df[models_df['Sample']=='Training'],how='inner',on='Models')
test_df1.sort_values(by=['Accuracy_x'], ascending=False, axis=0, inplace=True)
final_model2 = pd.DataFrame( np.array(test_df1[['Sample_y','Accuracy_y', 'Accuracy with 70% best features_y','Sample_x','Accuracy_x', 'Accuracy with 70% best features_x']]),
            index=test_df1['Models'], columns=['','Accuracy', 'Accuracy with 70% best features','','Accuracy', 'Accuracy with 70% best features'])

final_model2.rename(index=str,columns={'Unnamed: 1':'','Unnamed: 4':'', 'Accuracy.1':'Accuracy', 'Accuracy with 70% best features.1':'Accuracy with 70% best features'},inplace=True)
final_model2.to_csv('models_stemmisaton1_4.csv')
final_model2




df_models = pd.DataFrame({'Models':[], 'Sample':[], 'Accuracy':[],'Accuracy with 70% best features':[]})
data=[]




def modelize(list_clf,X,y,X_test,y_test):
    selector = SelectPercentile(mutual_info_classif,percentile=70)
    for clf in list_clf:
        Clf1 = clf().fit(X,y)
        Clf2 = Pipeline([('Feature Selection',selector),('Classification',clf())]).fit(X,y)
        cv = StratifiedShuffleSplit(n_splits=2 , test_size=.3, random_state=51)
        print('Model : %s' %type(Clf1).__name__)
        print('With all features                                              /    With 70% of the best features')
        print('                 Accuracy Score             F1 Score                Accuracy Score             F1 Score')
        print('training :       %f                   %f           /    %f                   %f' 
              %(accuracy_score(y,Clf1.predict(X)),f1_score(y,Clf1.predict(X),average='weighted'),accuracy_score(y,Clf2.predict(X)),f1_score(y,Clf2.predict(X),average='weighted')))
        print('Test     :       %f                   %f           /    %f                   %f' 
              %(accuracy_score(y_test,Clf1.predict(X_test)),f1_score(y_test,Clf1.predict(X_test),average='weighted'),accuracy_score(y_test,Clf2.predict(X_test)),f1_score(y_test,Clf2.predict(X_test),average='weighted')))
        data.insert(0,{'Models':type(Clf1).__name__, 'Sample':'Training', 'Accuracy':accuracy_score(y,Clf1.predict(X)),'Accuracy with 70% best features':accuracy_score(y,Clf2.predict(X)) })
        data.insert(0,{'Models':type(Clf1).__name__, 'Sample':'Test', 'Accuracy':accuracy_score(y_test,Clf1.predict(X_test)),'Accuracy with 70% best features':accuracy_score(y_test,Clf2.predict(X_test)) })
        plot_learning_curve(Clf1, X, y, Clf2, cv, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), scoring='accuracy')
        print('=============================================')




modelize(generalized_linear_model2,d2v_vecs,y_cora_train,d2v_test,y_cora_test) 




get_ipython().run_cell_magic('time', '', 'modelize(support_vector_machines2,d2v_vecs,y_cora_train,d2v_test,y_cora_test)')




modelize(ensemble_methods2,d2v_vecs,y_cora_train,d2v_test,y_cora_test)




modelize(Neural_network,d2v_vecs,y_cora_train,d2v_test,y_cora_test)




modelize([DecisionTreeClassifier,AdaBoostClassifier,GradientBoostingClassifier, SVC],d2v_vecs,y_cora_train,d2v_test,y_cora_test)




models_df =pd.concat([pd.DataFrame(data),df_models],ignore_index=True)  
#models_df.sort_values(by=['Accuracy','Models','Sample'], ascending=False, axis=0, inplace=True)
#test_df = pd.DataFrame(np.array(models_df[['Accuracy', 'Accuracy with 70% best features']]),
#                       index=[models_df['Models'],models_df['Sample']], columns=['Accuracy', 'Accuracy with 70% best features'])
test_df1=pd.merge(models_df[models_df['Sample']=='Test'],models_df[models_df['Sample']=='Training'],how='inner',on='Models')
test_df1.sort_values(by=['Accuracy_x'], ascending=False, axis=0, inplace=True)
final_model2 = pd.DataFrame( np.array(test_df1[['Sample_y','Accuracy_y', 'Accuracy with 70% best features_y','Sample_x','Accuracy_x', 'Accuracy with 70% best features_x']]),
            index=test_df1['Models'], columns=['','Accuracy', 'Accuracy with 70% best features','','Accuracy', 'Accuracy with 70% best features'])

final_model2.rename(index=str,columns={'Unnamed: 1':'','Unnamed: 4':'', 'Accuracy.1':'Accuracy', 'Accuracy with 70% best features.1':'Accuracy with 70% best features'},inplace=True)
final_model2.to_csv('models_stemmisatonDo2vec.csv')
final_model2




df_models = pd.DataFrame({'Models':[], 'Sample':[], 'Accuracy':[],'Accuracy with 70% best features':[]})
data=[]




def modelize(list_clf,X,y,X_test,y_test):
    selector = SelectPercentile(mutual_info_classif,percentile=70)
    for clf in list_clf:
        Clf1 = clf().fit(X,y)
        Clf2 = Pipeline([('Feature Selection',selector),('Classification',clf())]).fit(X,y)
        cv = StratifiedShuffleSplit(n_splits=2 , test_size=.3, random_state=51)
        print('Model : %s' %type(Clf1).__name__)
        print('With all features                                              /    With 70% of the best features')
        print('                 Accuracy Score             F1 Score                Accuracy Score             F1 Score')
        print('training :       %f                   %f           /    %f                   %f' 
              %(accuracy_score(y,Clf1.predict(X)),f1_score(y,Clf1.predict(X),average='weighted'),accuracy_score(y,Clf2.predict(X)),f1_score(y,Clf2.predict(X),average='weighted')))
        print('Test     :       %f                   %f           /    %f                   %f' 
              %(accuracy_score(y_test,Clf1.predict(X_test)),f1_score(y_test,Clf1.predict(X_test),average='weighted'),accuracy_score(y_test,Clf2.predict(X_test)),f1_score(y_test,Clf2.predict(X_test),average='weighted')))
        data.insert(0,{'Models':type(Clf1).__name__, 'Sample':'Training', 'Accuracy':accuracy_score(y,Clf1.predict(X)),'Accuracy with 70% best features':accuracy_score(y,Clf2.predict(X)) })
        data.insert(0,{'Models':type(Clf1).__name__, 'Sample':'Test', 'Accuracy':accuracy_score(y_test,Clf1.predict(X_test)),'Accuracy with 70% best features':accuracy_score(y_test,Clf2.predict(X_test)) })
        plot_learning_curve(Clf1, X, y, Clf2, cv, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), scoring='accuracy')
        print('=============================================')




modelize(generalized_linear_model2,d2v_vecs,y_cora_train,d2v_test,y_cora_test) 




modelize(support_vector_machines2,d2v_vecs,y_cora_train,d2v_test,y_cora_test)




modelize(ensemble_methods2,d2v_vecs,y_cora_train,d2v_test,y_cora_test)




modelize(Neural_network,d2v_vecs,y_cora_train,d2v_test,y_cora_test)




modelize([DecisionTreeClassifier,AdaBoostClassifier,GradientBoostingClassifier, SVC],d2v_vecs,y_cora_train,d2v_test,y_cora_test)




models_df =pd.concat([pd.DataFrame(data),df_models],ignore_index=True)  
#models_df.sort_values(by=['Accuracy','Models','Sample'], ascending=False, axis=0, inplace=True)
#test_df = pd.DataFrame(np.array(models_df[['Accuracy', 'Accuracy with 70% best features']]),
#                       index=[models_df['Models'],models_df['Sample']], columns=['Accuracy', 'Accuracy with 70% best features'])
test_df1=pd.merge(models_df[models_df['Sample']=='Test'],models_df[models_df['Sample']=='Training'],how='inner',on='Models')
test_df1.sort_values(by=['Accuracy_x'], ascending=False, axis=0, inplace=True)
final_model2 = pd.DataFrame( np.array(test_df1[['Sample_y','Accuracy_y', 'Accuracy with 70% best features_y','Sample_x','Accuracy_x', 'Accuracy with 70% best features_x']]),
            index=test_df1['Models'], columns=['','Accuracy', 'Accuracy with 70% best features','','Accuracy', 'Accuracy with 70% best features'])

final_model2.rename(index=str,columns={'Unnamed: 1':'','Unnamed: 4':'', 'Accuracy.1':'Accuracy', 'Accuracy with 70% best features.1':'Accuracy with 70% best features'},inplace=True)
final_model2.to_csv('models_stemmisatonDo2vec.csv')
final_model2




df_models = pd.DataFrame({'Models':[], 'Sample':[], 'Accuracy':[],'Accuracy with 70% best features':[]})
data=[]
def modelize(list_clf,X,y,X_test,y_test):
    selector = SelectPercentile(mutual_info_classif,percentile=70)
    for clf in list_clf:
        Clf1 = clf().fit(X,y)
        Clf2 = Pipeline([('Feature Selection',selector),('Classification',clf())]).fit(X,y)
        cv = StratifiedShuffleSplit(n_splits=2 , test_size=.3, random_state=51)
        print('Model : %s' %type(Clf1).__name__)
        print('With all features                                              /    With 70% of the best features')
        print('                 Accuracy Score             F1 Score                Accuracy Score             F1 Score')
        print('training :       %f                   %f           /    %f                   %f' 
              %(accuracy_score(y,Clf1.predict(X)),f1_score(y,Clf1.predict(X),average='weighted'),accuracy_score(y,Clf2.predict(X)),f1_score(y,Clf2.predict(X),average='weighted')))
        print('Test     :       %f                   %f           /    %f                   %f' 
              %(accuracy_score(y_test,Clf1.predict(X_test)),f1_score(y_test,Clf1.predict(X_test),average='weighted'),accuracy_score(y_test,Clf2.predict(X_test)),f1_score(y_test,Clf2.predict(X_test),average='weighted')))
        data.insert(0,{'Models':type(Clf1).__name__, 'Sample':'Training', 'Accuracy':accuracy_score(y,Clf1.predict(X)),'Accuracy with 70% best features':accuracy_score(y,Clf2.predict(X)) })
        data.insert(0,{'Models':type(Clf1).__name__, 'Sample':'Test', 'Accuracy':accuracy_score(y_test,Clf1.predict(X_test)),'Accuracy with 70% best features':accuracy_score(y_test,Clf2.predict(X_test)) })
        plot_learning_curve(Clf1, X, y, Clf2, cv, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), scoring='accuracy')
        print('=============================================')




modelize(generalized_linear_model2,d2v_vecs_bigram,y_cora_train,d2v_test_bigram,y_cora_test) 




modelize(support_vector_machines2,d2v_vecs_bigram,y_cora_train,d2v_test_bigram,y_cora_test)




modelize(ensemble_methods2,d2v_vecs_bigram,y_cora_train,d2v_test_bigram,y_cora_test)




modelize(Neural_network,d2v_vecs_bigram,y_cora_train,d2v_test_bigram,y_cora_test)




models_df =pd.concat([pd.DataFrame(data),df_models],ignore_index=True)  
#models_df.sort_values(by=['Accuracy','Models','Sample'], ascending=False, axis=0, inplace=True)
#test_df = pd.DataFrame(np.array(models_df[['Accuracy', 'Accuracy with 70% best features']]),
#                       index=[models_df['Models'],models_df['Sample']], columns=['Accuracy', 'Accuracy with 70% best features'])
test_df1=pd.merge(models_df[models_df['Sample']=='Test'],models_df[models_df['Sample']=='Training'],how='inner',on='Models')
test_df1.sort_values(by=['Accuracy_x'], ascending=False, axis=0, inplace=True)
final_model2 = pd.DataFrame( np.array(test_df1[['Sample_y','Accuracy_y', 'Accuracy with 70% best features_y','Sample_x','Accuracy_x', 'Accuracy with 70% best features_x']]),
            index=test_df1['Models'], columns=['','Accuracy', 'Accuracy with 70% best features','','Accuracy', 'Accuracy with 70% best features'])

final_model2.rename(index=str,columns={'Unnamed: 1':'','Unnamed: 4':'', 'Accuracy.1':'Accuracy', 'Accuracy with 70% best features.1':'Accuracy with 70% best features'},inplace=True)
final_model2.to_csv('models_lemmATISatIonDo2vec.csv')
final_model2




from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, KFold
from sklearn.feature_selection import SelectPercentile
from hyperopt import hp,fmin,Trials, tpe, STATUS_FAIL, STATUS_OK, space_eval, anneal
from hyperopt.pyll import scope
from hyperopt.pyll import stochastic

from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.decomposition import SparsePCA
from sklearn.preprocessing import normalize, StandardScaler

import time
random_state = 42
kf = KFold(n_splits=2,random_state=random_state)
n_iter= 50




#Parameters Optimization with HyperOpt 
#Tree-structure Parzen Estimator: TPE is a default algorithm for the Hyperopt. It uses Bayesian approach for optimization. 
#At every step it is trying to build probabilistic model of the function and choose the most promising parameters for the next step.
#1. We need to create a function to minimize.
def extraTree_accuracy_cv(params, random_state=random_state, cv=kf, X=X_t3gram_vectorizer_train, y=y_cora_train):
    # the function gets a set of variable parameters in "param"
    # the function gets a set of variable parameters in "param"
    params = {'n_estimators': int(params['n_estimators']),  #The number of trees in the forest.
              'max_features': str(params['max_features']), #The number of features to consider when looking for the best split.
              'min_samples_split': int(params['min_samples_split']), #The minimum number of samples required to split an internal node
              'max_depth': int(params['max_depth'])} #The maximum depth of the tree
    # we use this params to create a new LinearSVC Classifier
    model = ExtraTreesClassifier(random_state=random_state, **params, n_jobs = -1)
    # and then conduct the cross validation with the same folds as before
    try:
        return {'loss' : -cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1).mean(),
                'time' : time.time(),
                'status' : STATUS_OK }
    except (Exception, e):
        return {'status' : STATUS_FAIL,
                'time' : time.time(),
                'exception' : str(e)}




get_ipython().run_cell_magic('time', '', '# possible values of parameters\nspace={\'n_estimators\': hp.quniform(\'n_estimators\', 100, 2000, 1),\n       \'max_depth\' : hp.quniform(\'max_depth\', 2, 80, 1),\n       \'max_features\': hp.choice(\'max_features\', ["sqrt","log2"]),\n       \'min_samples_split\': hp.quniform(\'min_samples_split\', 2, 10, 1)  \n      }\n\n# trials will contain logging information\ntrials = Trials()\n\nbest=fmin(fn=extraTree_accuracy_cv, # function to optimize\n          space=space, \n          algo=tpe.suggest, # optimization algorithm, hyperotp will select its parameters automatically\n          max_evals=n_iter, # maximum number of iterations\n          trials=trials, # logging\n          rstate=np.random.RandomState(random_state) # fixing random state for the reproducibility\n         )\n# computing the score on the test set\nmodel = ExtraTreesClassifier(random_state=random_state, \n                             n_estimators=int(best[\'n_estimators\']), \n                             max_depth=int(best[\'max_depth\']),\n                             min_samples_split= int(space_eval(space,best)[\'min_samples_split\']),\n                             max_features=str(space_eval(space,best)[\'max_features\']),\n                             n_jobs=-1)\nmodel.fit(X_t3gram_vectorizer_train,y_cora_train)\ntpe_test_score=accuracy_score(y_cora_test, model.predict(X_t3gram_vectorizer_test))')




print(stochastic.sample(space))




print("Best Accuracy score on train set {:.3f} params {}".format( -extraTree_accuracy_cv(space_eval(space,best))['loss'], space_eval(space,best)))
print('Accuracy score on validation sample {:.3f}'.format(tpe_test_score))




print("Best Accuracy score on train set {:.3f} params {}".format( -extraTree_accuracy_cv(space_eval(space,best))['loss'], space_eval(space,best)))
print('Accuracy score on validation sample {:.3f}'.format(tpe_test_score))




#extraTree_accuracy_cv(best), 
space_eval(space,best)




tpe_results=np.array([[x['result']['loss'],
                      x['misc']['vals']['max_depth'][0],
                      x['misc']['vals']['n_estimators'][0],
                      x['misc']['vals']['max_features'][0],
                      x['misc']['vals']['min_samples_split'][0]
                      
                      
                      ] for x in trials.trials])

tpe_results_df=pd.DataFrame(tpe_results,
                           columns=['score', 'max_depth', 'n_estimators', 'max_features', 'min_samples_split'])
tpe_results_df.plot(subplots=True,figsize=(10, 10))




get_ipython().run_cell_magic('time', '', "#Lets generate the learning curve of the optimized model\nselector = SelectPercentile(f_classif,percentile=70)\ndef modelize(list_clf,X,y,X_test,y_test):\n    for clf in list_clf:\n        Clf1 = clf(random_state=random_state, n_estimators=int(best['n_estimators']), max_depth=int(best['max_depth']), min_samples_split= int(space_eval(space,best)['min_samples_split']),\n                             max_features=str(space_eval(space,best)['max_features']), n_jobs=-1).fit(X,y)\n        Clf2 = Pipeline([('Feature Selection',selector),('Classification',clf(random_state=random_state, n_estimators=int(best['n_estimators']), max_depth=int(best['max_depth']), min_samples_split= int(space_eval(space,best)['min_samples_split']),\n                             max_features=str(space_eval(space,best)['max_features']), n_jobs=-1))]).fit(X,y)\n        cv = StratifiedShuffleSplit(n_splits=3 , test_size=.3, random_state=random_state)\n        print('Model : %s' %type(Clf1).__name__)\n        print('With all features                                              /    With 70% of the best features')\n        print('                 Accuracy Score             F1 Score                Accuracy Score             F1 Score')\n        print('training :       %f                   %f           /    %f                   %f' \n              %(accuracy_score(y,Clf1.predict(X)),f1_score(y,Clf1.predict(X),average='weighted'),accuracy_score(y,Clf2.predict(X)),f1_score(y,Clf2.predict(X),average='weighted')))\n        print('Test     :       %f                   %f           /    %f                   %f' \n              %(accuracy_score(y_test,Clf1.predict(X_test)),f1_score(y_test,Clf1.predict(X_test),average='weighted')\n                ,accuracy_score(y_test,Clf2.predict(X_test)),f1_score(y_test,Clf2.predict(X_test),average='weighted')))\n        plot_learning_curve(Clf1, X, y, Clf2, cv, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), scoring='accuracy')\n        print('=============================================')\n    \nmodelize([ExtraTreesClassifier],X_t3gram_vectorizer_train,y_cora_train,X_t3gram_vectorizer_test,y_cora_test)")




#Parameters Optimization with HyperOpt 
#Tree-structure Parzen Estimator: TPE is a default algorithm for the Hyperopt. It uses Bayesian approach for optimization. 
#At every step it is trying to build probabilistic model of the function and choose the most promising parameters for the next step.
#1. We need to create a function to minimize.
def RF_cv(params, random_state=random_state, cv=kf, X=X_t3gram_vectorizer_train, y=y_cora_train):
    # the function gets a set of variable parameters in "param"
    # the function gets a set of variable parameters in "param"
    params = {'n_estimators': int(params['n_estimators']),  #The number of trees in the forest.
              'max_features': str(params['max_features']), #The number of features to consider when looking for the best split.
              'min_samples_split': int(params['min_samples_split']), #The minimum number of samples required to split an internal node
              'max_depth': int(params['max_depth'])} #The maximum depth of the tree
    # we use this params to create a new LinearSVC Classifier
    model = RandomForestClassifier(random_state=random_state, **params, n_jobs = -1)
    # and then conduct the cross validation with the same folds as before
    try:
        return {'loss' : -cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1).mean(),
                'time' : time.time(),
                'status' : STATUS_OK }
    except (Exception, e):
        return {'status' : STATUS_FAIL,
                'time' : time.time(),
                'exception' : str(e)}




get_ipython().run_cell_magic('time', '', '# possible values of parameters\nspace={\'n_estimators\': hp.quniform(\'n_estimators\', 20, 500, 1),\n       \'max_depth\' : hp.quniform(\'max_depth\', 2, 100, 1),\n       \'max_features\': hp.choice(\'max_features\', ["sqrt","log2"]),\n       \'min_samples_split\': hp.quniform(\'min_samples_split\', 2, 10, 1)  \n      }\n\n# trials will contain logging information\ntrials = Trials()\n\nbest=fmin(fn=RF_cv, # function to optimize\n          space=space, \n          algo=anneal.suggest, # optimization algorithm, hyperotp will select its parameters automatically\n          max_evals=n_iter, # maximum number of iterations\n          trials=trials, # logging\n          rstate=np.random.RandomState(random_state) # fixing random state for the reproducibility\n         )\n# computing the score on the test set\nmodel = RandomForestClassifier(random_state=random_state, \n                             n_estimators=int(best[\'n_estimators\']),\n                             #n_estimators= 302,  \n                             max_depth=int(best[\'max_depth\']),\n                             min_samples_split= int(space_eval(space,best)[\'min_samples_split\']),\n                             max_features=str(space_eval(space,best)[\'max_features\']),\n                             n_jobs=-1)\nmodel.fit(X_t3gram_vectorizer_train,y_cora_train)\ntpe_test_score=accuracy_score(y_cora_test, model.predict(X_t3gram_vectorizer_test))')




hp.quniform('n_estimators', 20, 500, 50)




print("Best Accuracy score on train set {:.3f} params {}".format( -RF_cv(space_eval(space,best))['loss'], space_eval(space,best)))
print('Accuracy score on validation sample {:.3f}'.format(tpe_test_score))




tpe_results=np.array([[x['result']['loss'],
                      x['misc']['vals']['max_depth'][0],
                      #x['misc']['vals']['n_estimators'][0],
                     # x['misc']['vals']['max_features'][0],
                      x['misc']['vals']['min_samples_split'][0]
                      
                      
                      ] for x in trials.trials])

tpe_results_df=pd.DataFrame(tpe_results,
                           columns=['score', 'max_depth', #'n_estimators',# 'max_features', 
                                    'min_samples_split'])
tpe_results_df.plot(subplots=True,figsize=(10, 10))




get_ipython().run_cell_magic('time', '', "#Lets generate the learning curve of the optimized model\nselector = SelectPercentile(f_classif,percentile=70)\ndef modelize(list_clf,X,y,X_test,y_test):\n    for clf in list_clf:\n        #Clf1 = clf(random_state=random_state, n_estimators=int(best['n_estimators']), max_depth=int(best['max_depth']), min_samples_split= int(space_eval(space,best)['min_samples_split']),\n        #                     max_features=str(space_eval(space,best)['max_features']), n_jobs=-1).fit(X,y)\n        #Clf2 = Pipeline([('Feature Selection',selector),('Classification',clf(random_state=random_state, n_estimators=int(best['n_estimators']), max_depth=int(best['max_depth']), min_samples_split= int(space_eval(space,best)['min_samples_split']),\n        #                     max_features=str(space_eval(space,best)['max_features']), n_jobs=-1))]).fit(X,y)\n        Clf1 = clf(random_state=random_state, n_estimators=302, max_depth=96, n_jobs=-1).fit(X,y)\n        Clf2 = Pipeline([('Feature Selection',selector),('Classification',clf(random_state=random_state, n_estimators=302, max_depth=96,n_jobs=-1) )]).fit(X,y)\n        cv = StratifiedShuffleSplit(n_splits=3 , test_size=.3, random_state=random_state)\n        print('Model : %s' %type(Clf1).__name__)\n        print('With all features                                              /    With 70% of the best features')\n        print('                 Accuracy Score             F1 Score                Accuracy Score             F1 Score')\n        print('training :       %f                   %f           /    %f                   %f' \n              %(accuracy_score(y,Clf1.predict(X)),f1_score(y,Clf1.predict(X),average='weighted'),accuracy_score(y,Clf2.predict(X)),f1_score(y,Clf2.predict(X),average='weighted')))\n        print('Test     :       %f                   %f           /    %f                   %f' \n              %(accuracy_score(y_test,Clf1.predict(X_test)),f1_score(y_test,Clf1.predict(X_test),average='weighted')\n                ,accuracy_score(y_test,Clf2.predict(X_test)),f1_score(y_test,Clf2.predict(X_test),average='weighted')))\n        plot_learning_curve(Clf1, X, y, Clf2, cv, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), scoring='accuracy')\n        print('=============================================')\n    \nmodelize([RandomForestClassifier],X_t3gram_vectorizer_train,y_cora_train,X_t3gram_vectorizer_test,y_cora_test)")




from sklearn.model_selection import validation_curve
from matplotlib import pyplot as plt

def plot_validation_curve(estimator, X, y, param_name, param_range,
                          ylim=(0, 1.1), cv=5, n_jobs=-1, scoring=None):
    estimator_name = type(estimator).__name__
    plt.figure(figsize=(10,6))
    plt.title("Validation curves for %s on %s"
              % (param_name, estimator_name))
    plt.grid()
    plt.xlim(min(param_range), max(param_range))
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.xscale('log')
    
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name, param_range,
        cv=cv, n_jobs=n_jobs, scoring=scoring)

    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.plot(
        param_range, train_scores_mean, 'o-',
        color="firebrick", linewidth=3, 
        label="Training score")
    plt.plot(
        param_range, test_scores_mean, 'o-',
        color="darkgoldenrod", 
        label="Cross-validation score")
    plt.fill_between(
        param_range, train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std, alpha=0.1,
        color="firebrick")
    plt.fill_between(
        param_range, test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std, alpha=0.1, color="darkgoldenrod")
    plt.legend(loc="best")




get_ipython().run_cell_magic('time', '', "%matplotlib inline\ncv = StratifiedShuffleSplit(n_splits=10 #nombre de fois qu'on regenere le train test\n                            , train_size=.8, random_state=random_state)\n\nclf = ExtraTreesClassifier(n_jobs=-1)\nparam_name = 'max_depth'\nparam_range = np.linspace(1,1500,15,dtype=int)\nprint(param_range)\nplot_validation_curve(\n    clf, X_t3gram_vectorizer_train, y_cora_train, param_name, param_range, cv=cv.split(X_t3gram_vectorizer_train, y_cora_train), scoring='accuracy')")




param_range




get_ipython().run_cell_magic('time', '', "%matplotlib inline\ncv = StratifiedShuffleSplit(n_splits=1 #nombre de fois qu'on regenere le train test\n                            , train_size=.7, random_state=random_state)\n\nclf = ExtraTreesClassifier(n_jobs=-1)\nparam_name = 'n_estimators'\nparam_range = np.linspace(1,500,10,dtype=int)\nprint(param_range)\nplot_validation_curve(\n    clf, X_t3gram_vectorizer_train, y_cora_train, param_name, param_range, cv=cv.split(X_t3gram_vectorizer_train, y_cora_train), scoring='accuracy')")




get_ipython().run_cell_magic('time', '', '\n#Parameters Optimization with HyperOpt \n#Tree-structure Parzen Estimator: TPE is a default algorithm for the Hyperopt. It uses Bayesian approach for optimization. \n#At every step it is trying to build probabilistic model of the function and choose the most promising parameters for the next step.\n#1. We need to create a function to minimize.\ndef ExtraTC_cv(params, random_state=random_state, cv=kf, X=X_t3gram_vectorizer_train, y=y_cora_train):\n    # the function gets a set of variable parameters in "param"\n    # the function gets a set of variable parameters in "param"\n    params = {#\'n_estimators\': int(params[\'n_estimators\']),  #The number of trees in the forest.\n              \'max_features\': str(params[\'max_features\']), #The number of features to consider when looking for the best split.\n              \'min_samples_split\': int(params[\'min_samples_split\']), #The minimum number of samples required to split an internal node\n              \'max_depth\': int(params[\'max_depth\'])} #The maximum depth of the tree\n    # we use this params to create a new LinearSVC Classifier\n    model = ExtraTreesClassifier(random_state=random_state, **params, n_jobs = -1, n_estimators=110)\n    # and then conduct the cross validation with the same folds as before\n    try:\n        return {\'loss\' : -cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1).mean(),\n                \'time\' : time.time(),\n                \'status\' : STATUS_OK }\n    except (Exception, e):\n        return {\'status\' : STATUS_FAIL,\n                \'time\' : time.time(),\n                \'exception\' : str(e)}\n    \n# possible values of parameters\nspace={#\'n_estimators\': hp.quniform(\'n_estimators\', 80, 180, 1),\n       \'max_depth\' : hp.quniform(\'max_depth\', 215, 430, 1),\n       \'max_features\': hp.choice(\'max_features\', ["sqrt","log2"]),\n       \'min_samples_split\': hp.quniform(\'min_samples_split\', 2, 10, 1)  \n      }\n\n# trials will contain logging information\ntrials = Trials()\n\nbest=fmin(fn=ExtraTC_cv, # function to optimize\n          space=space, \n          algo=anneal.suggest, # optimization algorithm, hyperotp will select its parameters automatically\n          max_evals=n_iter, # maximum number of iterations\n          trials=trials, # logging\n          rstate=np.random.RandomState(random_state) # fixing random state for the reproducibility\n         )\n# computing the score on the test set\nmodel = ExtraTreesClassifier(random_state=random_state, \n                             #n_estimators=int(best[\'n_estimators\']),\n                             n_estimators= 110,  \n                             max_depth=int(best[\'max_depth\']),\n                             min_samples_split= int(space_eval(space,best)[\'min_samples_split\']),\n                             max_features=str(space_eval(space,best)[\'max_features\']),\n                             n_jobs=-1)\nmodel.fit(X_t3gram_vectorizer_train,y_cora_train)\ntpe_test_score=accuracy_score(y_cora_test, model.predict(X_t3gram_vectorizer_test))')




print("Best Accuracy score on train set {:.3f} params {}".format( -ExtraTC_cv(space_eval(space,best))['loss'], space_eval(space,best)))
print('Accuracy score on validation sample {:.3f}'.format(tpe_test_score))




get_ipython().run_cell_magic('time', '', "#Lets generate the learning curve of the optimized model\nselector = SelectPercentile(f_classif,percentile=70)\ndef modelize(list_clf,X,y,X_test,y_test):\n    for clf in list_clf:\n        Clf1 = clf(random_state=random_state, n_estimators=110, max_depth=int(best['max_depth']), min_samples_split= int(space_eval(space,best)['min_samples_split']),\n                             max_features=str(space_eval(space,best)['max_features']), n_jobs=-1).fit(X,y)\n        Clf2 = Pipeline([('Feature Selection',selector),('Classification',clf(random_state=random_state, n_estimators=110, max_depth=int(best['max_depth']), min_samples_split= int(space_eval(space,best)['min_samples_split']),\n                             max_features=str(space_eval(space,best)['max_features']), n_jobs=-1))]).fit(X,y)\n        #Clf1 = clf(random_state=random_state, n_estimators=302, max_depth=96, n_jobs=-1).fit(X,y)\n        #Clf2 = Pipeline([('Feature Selection',selector),('Classification',clf(random_state=random_state, n_estimators=302, max_depth=96,n_jobs=-1) )]).fit(X,y)\n        cv = StratifiedShuffleSplit(n_splits=3 , test_size=.3, random_state=random_state)\n        print('Model : %s' %type(Clf1).__name__)\n        print('With all features                                              /    With 70% of the best features')\n        print('                 Accuracy Score             F1 Score                Accuracy Score             F1 Score')\n        print('training :       %f                   %f           /    %f                   %f' \n              %(accuracy_score(y,Clf1.predict(X)),f1_score(y,Clf1.predict(X),average='weighted'),accuracy_score(y,Clf2.predict(X)),f1_score(y,Clf2.predict(X),average='weighted')))\n        print('Test     :       %f                   %f           /    %f                   %f' \n              %(accuracy_score(y_test,Clf1.predict(X_test)),f1_score(y_test,Clf1.predict(X_test),average='weighted')\n                ,accuracy_score(y_test,Clf2.predict(X_test)),f1_score(y_test,Clf2.predict(X_test),average='weighted')))\n        plot_learning_curve(Clf1, X, y, Clf2, cv, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), scoring='accuracy')\n        print('=============================================')\n    \nmodelize([ExtraTreesClassifier],X_t3gram_vectorizer_train,y_cora_train,X_t3gram_vectorizer_test,y_cora_test)")




get_ipython().run_cell_magic('time', '', "%matplotlib inline\ncv = StratifiedShuffleSplit(n_splits=1 #nombre de fois qu'on regenere le train test\n                            , train_size=.7, random_state=random_state)\n\nclf = ExtraTreesClassifier(n_jobs=-1)\nparam_name = 'max_depth'\nparam_range = np.linspace(1,1500,15,dtype=int)\nprint(param_range)\nplot_validation_curve(\n    clf, X_t3gram_vectorizer_train_, y_cora_train, param_name, param_range, cv=cv.split(X_t3gram_vectorizer_train_, y_cora_train), scoring='accuracy')")




get_ipython().run_cell_magic('time', '', "%matplotlib inline\ncv = StratifiedShuffleSplit(n_splits=1 #nombre de fois qu'on regenere le train test\n                            , train_size=.7, random_state=random_state)\n\nclf = ExtraTreesClassifier(n_jobs=-1)\nparam_name = 'n_estimators'\nparam_range = np.linspace(1,700,7,dtype=int)\nprint(param_range)\nplot_validation_curve(\n    clf, X_t3gram_vectorizer_train, y_cora_train, param_name, param_range, cv=cv.split(X_t3gram_vectorizer_train, y_cora_train), scoring='accuracy')")




get_ipython().run_cell_magic('time', '', '\ndef ExtraTC_cv(params, random_state=random_state, cv=kf, X=X_t3gram_vectorizer_train, y=y_cora_train):\n    # the function gets a set of variable parameters in "param"\n    # the function gets a set of variable parameters in "param"\n    params = {#\'n_estimators\': int(params[\'n_estimators\']),  #The number of trees in the forest.\n              \'max_features\': str(params[\'max_features\']), #The number of features to consider when looking for the best split.\n              \'min_samples_split\': int(params[\'min_samples_split\']), #The minimum number of samples required to split an internal node\n              \'max_depth\': int(params[\'max_depth\'])} #The maximum depth of the tree\n    # we use this params to create a new LinearSVC Classifier\n    model = ExtraTreesClassifier(random_state=random_state, **params, n_jobs = -1, n_estimators=110)\n    # and then conduct the cross validation with the same folds as before\n    return {\'loss\' : -cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1).mean(),\n                \'time\' : time.time(),\n                \'status\' : STATUS_OK }\n\n    \n# possible values of parameters\nspace={#\'n_estimators\': hp.quniform(\'n_estimators\', 80, 180, 1),\n       \'max_depth\' : hp.quniform(\'max_depth\', 400, 500, 1),\n       \'max_features\': hp.choice(\'max_features\', ["sqrt","log2"]),\n       \'min_samples_split\': hp.quniform(\'min_samples_split\', 2, 10, 1)  \n      }\n\n# trials will contain logging information\ntrials = Trials()\n\nbest=fmin(fn=ExtraTC_cv, # function to optimize\n          space=space, \n          algo=anneal.suggest, # optimization algorithm, hyperotp will select its parameters automatically\n          max_evals=n_iter, # maximum number of iterations\n          trials=trials, # logging\n          rstate=np.random.RandomState(random_state) # fixing random state for the reproducibility\n         )\n# computing the score on the test set\nmodel = ExtraTreesClassifier(random_state=random_state, \n                             #n_estimators=int(best[\'n_estimators\']),\n                             n_estimators= 117,  \n                             max_depth=int(best[\'max_depth\']),\n                             min_samples_split= int(space_eval(space,best)[\'min_samples_split\']),\n                             max_features=str(space_eval(space,best)[\'max_features\']),\n                             n_jobs=-1)\nmodel.fit(X_t3gram_vectorizer_train_,y_cora_train)')




print("Best Accuracy score on train set {:.3f} params {}".format( -ExtraTC_cv(space_eval(space,best))['loss'], space_eval(space,best)))
tpe_test_score=accuracy_score(y_cora_test, model.predict(X_t3gram_vectorizer_test))
print('Accuracy score on validation sample {:.3f}'.format(tpe_test_score))
tpe_test_score=balanced_accuracy_score(y_cora_test, model.predict(X_t3gram_vectorizer_test))
print('balanced Accuracy score on validation sample {:.3f}'.format(tpe_test_score))




get_ipython().run_cell_magic('time', '', "#Lets generate the learning curve of the optimized model\nselector = SelectPercentile(f_classif,percentile=70)\ndef modelize(list_clf,X,y,X_test,y_test):\n    for clf in list_clf:\n        Clf1 = clf(random_state=random_state, n_estimators=117, max_depth=int(best['max_depth']), min_samples_split= int(space_eval(space,best)['min_samples_split']),\n                             max_features=str(space_eval(space,best)['max_features']), n_jobs=-1).fit(X,y)\n        Clf2 = Pipeline([('Feature Selection',selector),('Classification',clf(random_state=random_state, n_estimators=110, max_depth=int(best['max_depth']), min_samples_split= int(space_eval(space,best)['min_samples_split']),\n                             max_features=str(space_eval(space,best)['max_features']), n_jobs=-1))]).fit(X,y)\n        #Clf1 = clf(random_state=random_state, n_estimators=302, max_depth=96, n_jobs=-1).fit(X,y)\n        #Clf2 = Pipeline([('Feature Selection',selector),('Classification',clf(random_state=random_state, n_estimators=302, max_depth=96,n_jobs=-1) )]).fit(X,y)\n        cv = StratifiedShuffleSplit(n_splits=3 , test_size=.3, random_state=random_state)\n        print('Model : %s' %type(Clf1).__name__)\n        print('With all features                                              /    With 70% of the best features')\n        print('                 Balanced Accuracy Score    F1 Score                Balanced Accuracy Score    F1 Score')\n        print('training :       %f                   %f           /    %f                   %f' \n              %(balanced_accuracy_score(y,Clf1.predict(X)),f1_score(y,Clf1.predict(X),average='weighted'),\n                balanced_accuracy_score(y,Clf2.predict(X)),f1_score(y,Clf2.predict(X),average='weighted')))\n        print('Test     :       %f                   %f           /    %f                   %f' \n              %(balanced_accuracy_score(y_test,Clf1.predict(X_test)),f1_score(y_test,Clf1.predict(X_test),average='weighted'),\n                balanced_accuracy_score(y_test,Clf2.predict(X_test)),f1_score(y_test,Clf2.predict(X_test),average='weighted')))\n        plot_learning_curve(Clf1, X, y, Clf2, cv, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), scoring='accuracy')\n        print('=============================================')\n    \nmodelize([ExtraTreesClassifier],X_t3gram_vectorizer_train_,y_cora_train,X_t3gram_vectorizer_test,y_cora_test)")




get_ipython().run_cell_magic('time', '', "#OPTIMISATION DES HYPERPARAMETRES DU MODEL LINEARSVC AVEC GRIDSEARCH - We want to decrease the classifier's complexty (decrease parameter C of LinearSVC)\nfrom sklearn.model_selection import GridSearchCV\nparam_grid={'max_depth': np.linspace( 10, 200, 10,dtype=int), # Maximum number of levels in tree\n            'n_estimators': np.linspace(100,1000, 10 ,dtype=int), # Number of trees in random forest\n            'max_features' : ['auto', 'log2'], # Number of features to consider at every split\n            'min_samples_split' : [2, 5, 10], # Minimum number of samples required to split a node\n            #'min_samples_leaf': [1, 2, 4], # Minimum number of samples required at each leaf node\n           }\nmodel = RandomForestClassifier(random_state=random_state, n_jobs=-1)\nkf = KFold(n_splits=3,random_state=random_state)\n\ngs=GridSearchCV(model, param_grid, scoring='accuracy', n_jobs=-1, cv=kf, verbose=False)\ngs.fit(X_t3gram_vectorizer_train,y_cora_train)\ngs_test_score=accuracy_score(y_cora_test, gs.predict(X_t3gram_vectorizer_test))")




print("Best Accuracy on training sample: {:.3f} with hyperparameters {}".format(gs.best_score_, gs.best_params_))
print("Best Accuracy on validation sample: {:.3f} ".format(gs_test_score))




print("Best Accuracy on training sample: {:.3f} with hyperparameters {}".format(gs.best_score_, gs.best_params_))
print("Best Accuracy on validation sample: {:.3f} ".format(gs_test_score))




gs_results_df=pd.DataFrame(np.transpose([gs.cv_results_['mean_test_score'],
                                         gs.cv_results_['param_max_depth'].data,
                                         gs.cv_results_['param_max_features'].data,
                                         gs.cv_results_['param_min_samples_split'].data,
                                         gs.cv_results_['param_n_estimators'].data]),
                                           
                           columns=['score', 'max_depth', 'max_features', 'min_samples_split', 'n_estimators',])
gs_results_df.plot(subplots=True,figsize=(10, 10))




print((gs.cv_results_).keys())




gs_results_df.sample(5)




#gs_results_df.plot([gs_results_df['max_depth'],gs_results_df['score']])
plt.plot(gs_results_df['max_depth'],gs_results_df['score'])




get_ipython().run_cell_magic('time', '', "#OPTIMISATION DES HYPERPARAMETRES DU MODEL LINEARSVC AVEC GRIDSEARCH - We want to decrease the classifier's complexty (decrease parameter C of LinearSVC)\nfrom sklearn.model_selection import GridSearchCV\nparam_grid={'C': np.linspace(0.0000001, 1, 250)}\nmodel = LinearSVC()\nkf = KFold(n_splits=2,random_state=42)\nn_iter= 50\nrandom_state = 42\ngs=GridSearchCV(model, param_grid, scoring='accuracy', n_jobs=-1, cv=kf, verbose=False)\ngs.fit(X_Singt3gram_vectorizer_train,y_cora_train)\ngs_test_score=accuracy_score(y_cora_test, gs.predict(X_Singt3gram_vectorizer_test))")




print("Best Accuracy on training sample: {:.3f} with hyperparameters {}".format(gs.best_score_, gs.best_params_))
print("Best Accuracy on validation sample: {:.3f} ".format(gs_test_score))




print("Best Accuracy for training{:.3f} params {}".format(gs.best_score_, gs.best_params_))
print("Best Accuracy for validation: {:.3f}".format(gs_test_score))




print("Best Accuracy on training sample: {:.3f} with hyperparameters {}".format(gs.best_score_, gs.best_params_))
print("Best Accuracy on validation sample: {:.3f} ".format(gs_test_score))




gs_results_df=pd.DataFrame(np.transpose([gs.cv_results_['mean_test_score'],
                                         gs.cv_results_['param_C'].data]),
                           columns=['score', 'c'])
gs_results_df.plot(subplots=True,figsize=(10, 10))




get_ipython().run_cell_magic('time', '', "#Let's RECALL THE lINEARsvc with the penality = l1, which results in sparse solutions. Sparse solutions correspond to an implicit feature selection.\nfrom sklearn.model_selection import GridSearchCV\nparam_grid={'C': np.linspace(0.0000001, 1, 250),\n            'penalty' : ['l1'],\n            'dual': [False]}\nmodel = LinearSVC()\nkf = KFold(n_splits=2,random_state=42)\nn_iter= 50\nrandom_state = 42\ngs=GridSearchCV(model, param_grid, scoring='accuracy', n_jobs=-1, cv=kf, verbose=False)\ngs.fit(X_Singt3gram_vectorizer_train,y_cora_train)\ngs_test_score=accuracy_score(y_cora_test, gs.predict(X_Singt3gram_vectorizer_test))")




print("Best Accuracy on training sample: {:.3f} with hyperparameters {}".format(gs.best_score_, gs.best_params_))
print("Best Accuracy on validation sample: {:.3f} ".format(gs_test_score))




print("Best Accuracy on training sample: {:.3f} with hyperparameters {}".format(gs.best_score_, gs.best_params_))
print("Best Accuracy on validation sample: {:.3f} ".format(gs_test_score))




gs_results_df=pd.DataFrame(np.transpose([gs.cv_results_['mean_test_score'],
                                         gs.cv_results_['param_C'].data]),
                           columns=['score', 'c'])
gs_results_df.plot(subplots=True,figsize=(10, 10))




get_ipython().run_cell_magic('time', '', "#Lets generate the learning curve of the optimized model\nselector = SelectPercentile(f_classif,percentile=70)\ndef modelize_LinearSVC(list_clf,X,y,X_test,y_test):\n    for clf in list_clf:\n        Clf1 = clf(C=0.46231161155778894).fit(X,y)\n        Clf2 = Pipeline([('Feature Selection',selector),('Classification',clf(C=0.46231161155778894))]).fit(X,y)\n        cv = StratifiedShuffleSplit(n_splits=3 , test_size=.3, random_state=42)\n        print('Model : %s' %type(Clf1).__name__)\n        print('With all features                                              /    With 70% of the best features')\n        print('                 Accuracy Score             F1 Score                Accuracy Score             F1 Score')\n        print('training :       %f                   %f           /    %f                   %f' \n              %(accuracy_score(y,Clf1.predict(X)),f1_score(y,Clf1.predict(X),average='weighted'),accuracy_score(y,Clf2.predict(X)),f1_score(y,Clf2.predict(X),average='weighted')))\n        print('Test     :       %f                   %f           /    %f                   %f' \n              %(accuracy_score(y_test,Clf1.predict(X_test)),f1_score(y_test,Clf1.predict(X_test),average='weighted')\n                ,accuracy_score(y_test,Clf2.predict(X_test)),f1_score(y_test,Clf2.predict(X_test),average='weighted')))\n        plot_learning_curve(Clf1, X, y, Clf2, cv, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), scoring='accuracy')\n        print('=============================================')\n    \nmodelize_LinearSVC([LinearSVC],X_Singt3gram_vectorizer_train,y_cora_train,X_Singt3gram_vectorizer_test,y_cora_test)")




#Nouvel Echantillonage  
cora_data_neg_sample = cora_data[cora_data['target'] == 1] #Negatives comments
cora_data_positive_sample = cora_data[cora_data['target'] == 0].reindex()  #Positive Comments

cora_resampling = pd.concat([pd.DataFrame(cora_data_positive_sample.sample(20000)),
                               pd.DataFrame(cora_data_neg_sample)])
100*(cora_resampling.groupby('target')['question_text'].count())/cora_resampling['target'].count()




cora_resampling.info()




#Cleaning the data 
cora_resampling['clean_question'] = cora_resampling['question_text'].apply(clean_str)
#Tokenizing and stopwords removing
cora_resampling['tokeniZ_stopWords_question'] = cora_resampling['clean_question'].apply(tokeniZ_stopWords)
#Words Stemming
cora_resampling['stemming_question'] = [[ps.stem(word) for word in words] for words in cora_resampling['tokeniZ_stopWords_question'] ]
cora_resampling['stemming_question_for_tfidf'] = [' '.join(words) for words in cora_resampling['stemming_question']] 




cora_resampling.columns




X_cora_train, X_cora_test, y_cora_train, y_cora_test = train_test_split(
            cora_resampling['stemming_question_for_tfidf']
            ,cora_resampling['target'], 
            test_size=0.3, random_state=42)
X_cora_train.shape, X_cora_test.shape, y_cora_train.shape, y_cora_test.shape




#Vectorization with tf-idf
X_Singt3gram_vectorizer_train_ = st3gram_vectorizer.fit_transform(X_cora_train_)
X_Singt3gram_vectorizer_test_  = st3gram_vectorizer.transform(X_cora_test_)




get_ipython().run_cell_magic('time', '', "#OPTIMISATION DES HYPERPARAMETRES DU MODEL LINEARSVC AVEC GRIDSEARCH - We want to increase the regularization of the classifier (decrease parameter C of LinearSVC)\nfrom sklearn.model_selection import GridSearchCV\nparam_grid={'C': np.linspace(0.0000001, 1, 250)}\nmodel = LinearSVC()\nkf = KFold(n_splits=2,random_state=42)\nn_iter= 50\nrandom_state = 42\ngs=GridSearchCV(model, param_grid, scoring='accuracy', n_jobs=-1, cv=kf, verbose=False)\ngs.fit(X_Singt3gram_vectorizer_train_,y_cora_train_)\ngs_test_score=accuracy_score(y_cora_test_, gs.predict(X_Singt3gram_vectorizer_test_))")




print("Best Accuracy on training sample: {:.3f} with hyperparameters {}".format(gs.best_score_, gs.best_params_))
print("Best Accuracy on validation sample: {:.3f} ".format(gs_test_score))




from sklearn.metrics import balanced_accuracy_score




get_ipython().run_cell_magic('time', '', "#Let's try to increase the data regularization by using the Standardization method\n#Lets generate the learning curve of the optimized model\nselector = SelectPercentile(f_classif,percentile=60)\ndef modelize_LinearSVC(list_clf,X,y,X_test,y_test):\n    for clf in list_clf:\n        Clf1 = clf(C=0.4056225493975904).fit(X,y)\n        Clf2 = Pipeline([('Feature Selection',selector),('Classification',clf(C=0.4056225493975904))]).fit(X,y)\n        cv = StratifiedShuffleSplit(n_splits=3 , test_size=.3, random_state=51)\n        print('Model : %s' %type(Clf1).__name__)\n        print('With all features                                              /    With 70% of the best features')\n        print('                 Balanced Accuracy Score    F1 Score                Balanced Accuracy Score    F1 Score')\n        print('training :       %f                   %f           /    %f                   %f' \n              %(balanced_accuracy_score(y,Clf1.predict(X)),f1_score(y,Clf1.predict(X),average='weighted'),balanced_accuracy_score(y,Clf2.predict(X)),f1_score(y,Clf2.predict(X),average='weighted')))\n        print('Test     :       %f                   %f           /    %f                   %f' \n              %(balanced_accuracy_score(y_test,Clf1.predict(X_test)),f1_score(y_test,Clf1.predict(X_test),average='weighted')\n                ,balanced_accuracy_score(y_test,Clf2.predict(X_test)),f1_score(y_test,Clf2.predict(X_test),average='weighted')))\n        plot_learning_curve(Clf1, X, y, Clf2, cv, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), scoring='balanced_accuracy')\n        print('=============================================')\n    \nmodelize_LinearSVC([LinearSVC],X_Singt3gram_vectorizer_train,y_cora_train,X_Singt3gram_vectorizer_test,y_cora_test)")




get_ipython().run_cell_magic('time', '', "#Let's try to increase the data regularization by using the Standardization method\n#Lets generate the learning curve of the optimized model\nselector = SelectPercentile(f_classif,percentile=60)\ndef modelize_LinearSVC(list_clf,X,y,X_test,y_test):\n    for clf in list_clf:\n        Clf1 = clf(C=0.4056225493975904).fit(X,y)\n        Clf2 = Pipeline([('Feature Selection',selector),('Classification',clf(C=0.4056225493975904))]).fit(X,y)\n        cv = StratifiedShuffleSplit(n_splits=3 , test_size=.3, random_state=51)\n        print('Model : %s' %type(Clf1).__name__)\n        print('With all features                                              /    With 70% of the best features')\n        print('                 Balanced Accuracy Score    F1 Score                Balanced Accuracy Score    F1 Score')\n        print('training :       %f                   %f           /    %f                   %f' \n              %(balanced_accuracy_score(y,Clf1.predict(X)),f1_score(y,Clf1.predict(X),average='weighted'),balanced_accuracy_score(y,Clf2.predict(X)),f1_score(y,Clf2.predict(X),average='weighted')))\n        print('Test     :       %f                   %f           /    %f                   %f' \n              %(balanced_accuracy_score(y_test,Clf1.predict(X_test)),f1_score(y_test,Clf1.predict(X_test),average='weighted')\n                ,balanced_accuracy_score(y_test,Clf2.predict(X_test)),f1_score(y_test,Clf2.predict(X_test),average='weighted')))\n        plot_learning_curve(Clf1, X, y, Clf2, cv, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), scoring='balanced_accuracy')\n        print('=============================================')\nnorm=StandardScaler(with_mean=False)\nmodelize_LinearSVC([LinearSVC],norm.fit_transform(X_Singt3gram_vectorizer_train),y_cora_train,norm.transform(X_Singt3gram_vectorizer_test),y_cora_test)")




get_ipython().run_cell_magic('time', '', "#Let's try to increase the data regularization by using the normalization method\nmodelize_LinearSVC([LinearSVC],normalize(X_Singt3gram_vectorizer_train,norm='l2'),y_cora_train,normalize(X_Singt3gram_vectorizer_test,norm='l2'),y_cora_test)")




#Parameters Optimization with HyperOpt 
#1. We need to create a function to minimize.
def MLP_accuracy_cv(params, random_state=random_state, cv=kf, X=X_t3gram_vectorizer_train, y=y_cora_train):
    # the function gets a set of variable parameters in "param" 
    params = {'hidden_layer_sizes': tuple(params['hidden_layer_sizes']), 
              'activation': str(params['activation']),#Activation functions for the hidden layers
              'solver': str(params['solver']), #The solver for weight optimization.
              'alpha': int(params['alpha']), #L2 penalty (regularization term) parameter
              'learning_rate': str(params['learning_rate'])} #Learning rate schedule for weight updates
    # we use this params to create a new LinearSVC Classifier
    model = MLPClassifier(random_state=random_state ,# **params)
                          hidden_layer_sizes = params['hidden_layer_sizes'],
                          activation = params['activation'],
                          solver = params['solver'],
                          alpha = params['alpha'],
                          learning_rate = params['learning_rate'])
    # and then conduct the cross validation with the same folds as before
    try:
        return {'loss' : -cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1).mean(),
                'time' : time.time(),
                'status' : STATUS_OK }
    except (Exception, e):
        return {'status' : STATUS_FAIL,
                'time' : time.time(),
                'exception' : str(e)}




get_ipython().run_cell_magic('time', '', '# possible values of parameters\nspace={\'hidden_layer_sizes\': hp.choice(\'hidden_layer_sizes\' , [(20,10,5,),(5,25,50,),(100,25,5,)]),\n       \'activation\' : hp.choice(\'activation\' , ["identity", "logistic", "tanh", "relu"]), \n       \'solver\' : hp.choice( \'solver\' , ["lbfgs", "sgd", "adam"]),\n       \'alpha\': hp.uniform(\'alpha\',0.0001,0.9),\n       \'learning_rate\': hp.choice(\'learning_rate\' , ["constant", "invscaling", "adaptive"])\n      }\n      \n\n# trials will contain logging information\ntrials = Trials()\n\nbest=fmin(fn=MLP_accuracy_cv, # function to optimize\n          space=space, \n          algo=tpe.suggest, # optimization algorithm, hyperotp will select its parameters automatically\n          max_evals=n_iter, # maximum number of iterations\n          trials=trials, # logging\n          rstate=np.random.RandomState(random_state)) # fixing random state for the reproducibility')




get_ipython().run_cell_magic('time', '', "# computing the score on the test set\nmodel = MLPClassifier(random_state=random_state, activation = str(space_eval(space,best)['activation'] ),\n                                                 solver = str(space_eval(space,best)['solver']),\n                                                 alpha = int(best['alpha']),\n                                                 hidden_layer_sizes=tuple(space_eval(space,best)['hidden_layer_sizes']),\n                                                 learning_rate = str(space_eval(space,best)['learning_rate'] ))\nmodel.fit(X_t3gram_vectorizer_train,y_cora_train)\ntpe_test_score=accuracy_score(y_cora_test, model.predict(X_t3gram_vectorizer_test))")




get_ipython().run_cell_magic('time', '', 'print("Best Accuracy score on train set {:.3f} params {}".format( -MLP_accuracy_cv(space_eval(space,best))[\'loss\'], space_eval(space,best)))\nprint(\'Accuracy score on validation sample {:.3f}\'.format(tpe_test_score))')




get_ipython().run_cell_magic('time', '', "#Lets generate the learning curve of the optimized model\nselector = SelectPercentile(f_classif,percentile=70)\ndef modelize_MLP(list_clf,X,y,X_test,y_test):\n    for clf in list_clf:\n        Model = MLPClassifier(random_state=random_state, activation = str(space_eval(space,best)['activation'] ),\n                                                 solver = str(space_eval(space,best)['solver']),\n                                                 alpha = int(best['alpha']),\n                                                 hidden_layer_sizes=tuple(space_eval(space,best)['hidden_layer_sizes']),\n                                                 learning_rate = str(space_eval(space,best)['learning_rate'] ))\n        \n        Clf1 = Model.fit(X,y)\n        Clf2 = Pipeline([('Feature Selection',selector),('Classification',MLPClassifier(random_state=random_state, activation = str(space_eval(space,best)['activation'] ),\n                                                 solver = str(space_eval(space,best)['solver']),\n                                                 alpha = int(best['alpha']),\n                                                 hidden_layer_sizes=tuple(space_eval(space,best)['hidden_layer_sizes']),\n                                                 learning_rate = str(space_eval(space,best)['learning_rate'] )))]).fit(X,y)\n        cv = StratifiedShuffleSplit(n_splits=3 , test_size=.3, random_state=random_state)\n        print('Model : %s' %type(Clf1).__name__)\n        print('With all features                                              /    With 70% of the best features')\n        print('                 Accuracy Score             F1 Score                Accuracy Score             F1 Score')\n        print('training :       %f                   %f           /    %f                   %f' \n              %(accuracy_score(y,Clf1.predict(X)),f1_score(y,Clf1.predict(X),average='weighted'),accuracy_score(y,Clf2.predict(X)),f1_score(y,Clf2.predict(X),average='weighted')))\n        print('Test     :       %f                   %f           /    %f                   %f' \n              %(accuracy_score(y_test,Clf1.predict(X_test)),f1_score(y_test,Clf1.predict(X_test),average='weighted')\n                ,accuracy_score(y_test,Clf2.predict(X_test)),f1_score(y_test,Clf2.predict(X_test),average='weighted')))\n        plot_learning_curve(Clf1, X, y, Clf2, cv, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), scoring='accuracy')\n        print('=============================================')\n        \nmodelize_MLP([MLPClassifier],X_t3gram_vectorizer_train,y_cora_train,X_t3gram_vectorizer_test,y_cora_test)")




cora_data_neg_sample = cora_data[cora_data['target'] == 1] #Negatives comments
cora_data_positive_sample = cora_data[cora_data['target'] == 0].reindex()  #Positive Comments

cora_resampling = pd.concat([pd.DataFrame(cora_data_positive_sample.sample(4000)),
                               pd.DataFrame(cora_data_neg_sample.sample(2500))])
100*(cora_resampling.groupby('target')['question_text'].count())/cora_resampling['target'].count()




#Cleaning the data 
cora_resampling['clean_question'] = cora_resampling['question_text'].apply(clean_str)
#Tokenizing and stopwords removing
cora_resampling['tokeniZ_stopWords_question'] = cora_resampling['clean_question'].apply(tokeniZ_stopWords)
#Words lemmatization
cora_resampling['lemmatize_question'] = cora_resampling['tokeniZ_stopWords_question'].apply(lemat_words)
cora_resampling['lemmatize_question_for_tfidf'] = [' '.join(x) for x in cora_resampling['lemmatize_question'] ]
#Vectorization with tf-idf
X_Singt3gram_vectorizer_train = st3gram_vectorizer.fit_transform(X_cora_train['lemmatize_question_for_tfidf'])
X_Singt3gram_vectorizer_test  = st3gram_vectorizer.transform(X_cora_test['lemmatize_question_for_tfidf'])




get_ipython().run_cell_magic('time', '', "#Lets generate the learning curve of the optimized model\nselector = SelectPercentile(f_classif,percentile=70)\ndef modelize_MLP(list_clf,X,y,X_test,y_test):\n    for clf in list_clf:\n        Model = MLPClassifier(random_state=random_state, activation = str(space_eval(space,best)['activation'] ),\n                                                 solver = str(space_eval(space,best)['solver']),\n                                                 alpha = int(best['alpha']),\n                                                 hidden_layer_sizes=tuple(space_eval(space,best)['hidden_layer_sizes']),\n                                                 learning_rate = str(space_eval(space,best)['learning_rate'] ))\n        \n        Clf1 = Model.fit(X,y)\n        Clf2 = Pipeline([('Feature Selection',selector),('Classification',MLPClassifier(random_state=random_state, activation = str(space_eval(space,best)['activation'] ),\n                                                 solver = str(space_eval(space,best)['solver']),\n                                                 alpha = int(best['alpha']),\n                                                 hidden_layer_sizes=tuple(space_eval(space,best)['hidden_layer_sizes']),\n                                                 learning_rate = str(space_eval(space,best)['learning_rate'] )))]).fit(X,y)\n        cv = StratifiedShuffleSplit(n_splits=3 , test_size=.3, random_state=random_state)\n        print('Model : %s' %type(Clf1).__name__)\n        print('With all features                                              /    With 70% of the best features')\n        print('                 Balanced Accuracy Score    F1 Score                Balanced Accuracy Score    F1 Score')\n        print('training :       %f                   %f           /    %f                   %f' \n              %(balanced_accuracy_score(y,Clf1.predict(X)),f1_score(y,Clf1.predict(X),average='weighted'),balanced_accuracy_score(y,Clf2.predict(X)),f1_score(y,Clf2.predict(X),average='weighted')))\n        print('Test     :       %f                   %f           /    %f                   %f' \n              %(balanced_accuracy_score(y_test,Clf1.predict(X_test)),f1_score(y_test,Clf1.predict(X_test),average='weighted')\n                ,balanced_accuracy_score(y_test,Clf2.predict(X_test)),f1_score(y_test,Clf2.predict(X_test),average='weighted')))\n        plot_learning_curve(Clf1, X, y, Clf2, cv, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), scoring='balanced_accuracy')\n        print('=============================================')\n        \nmodelize_MLP([MLPClassifier],X_t3gram_vectorizer_train,y_cora_train,X_t3gram_vectorizer_test,y_cora_test)")




from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold
random_state = 42




get_ipython().run_cell_magic('time', '', "#Let's try to increase the model complexity.\n\nparam_grid={'C': np.linspace(1, 12, 100),\n            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga', 'liblinear']}\nmodel = LogisticRegression( n_jobs=-1)\nkf = KFold(n_splits=3,random_state=random_state)\n\ngs=GridSearchCV(model, param_grid, scoring='accuracy',  n_jobs=-1, cv=kf, verbose=False)\ngs.fit(X_t3gram_vectorizer_train,y_cora_train)\ngs_test_score=accuracy_score(y_cora_test, gs.predict(X_t3gram_vectorizer_test))")




print("Best Accuracy on training sample: {:.3f} with hyperparameters {}".format(gs.best_score_, gs.best_params_))
print("Best Accuracy on validation sample: {:.3f} ".format(gs_test_score))




print("Best Accuracy on training sample: {:.3f} with hyperparameters {}".format(gs.best_score_, gs.best_params_))
print("Best Accuracy on validation sample: {:.3f} ".format(gs_test_score))




gs_results_df=pd.DataFrame(np.transpose([gs.cv_results_['mean_test_score'],
                                         gs.cv_results_['param_C'].data]),
                           columns=['score', 'c'])
gs_results_df.plot(subplots=True,figsize=(10, 10))




get_ipython().run_cell_magic('time', '', "#Lets generate the learning curve of the optimized model\nselector = SelectPercentile(f_classif,percentile=71)\ndef modelize_LogReg(list_clf,X,y,X_test,y_test):\n    for clf in list_clf:\n        Clf1 = clf(C=3.3855421686746987).fit(X,y)\n        Clf2 = Pipeline([('Feature Selection',selector),('Classification',clf(C=3.3855421686746987))]).fit(X,y)\n        cv = StratifiedShuffleSplit(n_splits=3 , test_size=.3, random_state=42)\n        print('Model : %s' %type(Clf1).__name__)\n        print('With all features                                              /    With 70% of the best features')\n        print('                 Accuracy Score             F1 Score                Accuracy Score             F1 Score')\n        print('training :       %f                   %f           /    %f                   %f' \n              %(balanced_accuracy_score(y,Clf1.predict(X)),f1_score(y,Clf1.predict(X),average='weighted'),\n                balanced_accuracy_score(y,Clf2.predict(X)),f1_score(y,Clf2.predict(X),average='weighted')))\n        print('Test     :       %f                   %f           /    %f                   %f' \n              %(balanced_accuracy_score(y_test,Clf1.predict(X_test)),f1_score(y_test,Clf1.predict(X_test),average='weighted'),\n                balanced_accuracy_score(y_test,Clf2.predict(X_test)),f1_score(y_test,Clf2.predict(X_test),average='weighted')))\n        plot_learning_curve(Clf1, X, y, Clf2, cv, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), scoring='accuracy')\n        print('=============================================')\n    \nmodelize_LogReg([LogisticRegression],X_t3gram_vectorizer_train,y_cora_train,X_t3gram_vectorizer_test,y_cora_test)")




get_ipython().run_cell_magic('time', '', "#Let's try to increase the model complexity.\n\nparam_grid={'C': np.linspace(1, 100, 250), 'penalty': ['l1']}\nmodel = LogisticRegression()\nkf = KFold(n_splits=3,random_state=random_state)\n\ngs=GridSearchCV(model, param_grid, scoring='accuracy',  n_jobs=-1, cv=kf, verbose=False)\ngs.fit(X_t3gram_vectorizer_train,y_cora_train)\ngs_test_score=accuracy_score(y_cora_test, gs.predict(X_t3gram_vectorizer_test))")




print("Best Accuracy on training sample: {:.3f} with hyperparameters {}".format(gs.best_score_, gs.best_params_))
print("Best Accuracy on validation sample: {:.3f} ".format(gs_test_score))




gs_results_df=pd.DataFrame(np.transpose([gs.cv_results_['mean_test_score'],
                                         gs.cv_results_['param_C'].data]),
                           columns=['score', 'c'])
gs_results_df.plot(subplots=True,figsize=(10, 10))




get_ipython().run_cell_magic('time', '', "#Lets generate the learning curve of the optimized model\nselector = SelectPercentile(f_classif,percentile=70)\ndef modelize_LogReg(list_clf,X,y,X_test,y_test):\n    for clf in list_clf:\n        Clf1 = clf(C= 2.9879518072289155, penalty= 'l1').fit(X,y)\n        Clf2 = Pipeline([('Feature Selection',selector),('Classification',clf(C=2.9879518072289155, penalty='l1'))]).fit(X,y)\n        cv = StratifiedShuffleSplit(n_splits=3 , test_size=.3, random_state=42)\n        print('Model : %s' %type(Clf1).__name__)\n        print('With all features                                              /    With 70% of the best features')\n        print('                 Accuracy Score             F1 Score                Accuracy Score             F1 Score')\n        print('training :       %f                   %f           /    %f                   %f' \n              %(balanced_accuracy_score(y,Clf1.predict(X)),f1_score(y,Clf1.predict(X),average='weighted'),\n                balanced_accuracy_score(y,Clf2.predict(X)),f1_score(y,Clf2.predict(X),average='weighted')))\n        print('Test     :       %f                   %f           /    %f                   %f' \n              %(balanced_accuracy_score(y_test,Clf1.predict(X_test)),f1_score(y_test,Clf1.predict(X_test),average='weighted')\n                ,balanced_accuracy_score(y_test,Clf2.predict(X_test)),f1_score(y_test,Clf2.predict(X_test),average='weighted')))\n        plot_learning_curve(Clf1, X, y, Clf2, cv, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), scoring='accuracy')\n        print('=============================================')\nnorm=StandardScaler(with_mean=False)\nmodelize_LogReg([LogisticRegression],norm.fit_transform(X_t3gram_vectorizer_train),y_cora_train,norm.transform(X_t3gram_vectorizer_test),y_cora_test)")




get_ipython().run_cell_magic('time', '', "#Lets generate the learning curve of the optimized model\nselector = SelectPercentile(f_classif,percentile=70)\ndef modelize_LogReg(list_clf,X,y,X_test,y_test):\n    for clf in list_clf:\n        Clf1 = clf(C= 3.3855421686746987, penalty= 'l2').fit(X,y)\n        Clf2 = Pipeline([('Feature Selection',selector),('Classification',clf(C=3.3855421686746987, penalty='l2'))]).fit(X,y)\n        cv = StratifiedShuffleSplit(n_splits=3 , test_size=.3, random_state=42)\n        print('Model : %s' %type(Clf1).__name__)\n        print('With all features                                              /    With 70% of the best features')\n        print('                 Accuracy Score             F1 Score                Accuracy Score             F1 Score')\n        print('training :       %f                   %f           /    %f                   %f' \n              %(balanced_accuracy_score(y,Clf1.predict(X)),f1_score(y,Clf1.predict(X),average='weighted'),\n                balanced_accuracy_score(y,Clf2.predict(X)),f1_score(y,Clf2.predict(X),average='weighted')))\n        print('Test     :       %f                   %f           /    %f                   %f' \n              %(balanced_accuracy_score(y_test,Clf1.predict(X_test)),f1_score(y_test,Clf1.predict(X_test),average='weighted')\n                ,balanced_accuracy_score(y_test,Clf2.predict(X_test)),f1_score(y_test,Clf2.predict(X_test),average='weighted')))\n        plot_learning_curve(Clf1, X, y, Clf2, cv, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), scoring='accuracy')\n        print('=============================================')\nnorm=StandardScaler(with_mean=False)\nmodelize_LogReg([LogisticRegression],norm.fit_transform(X_t3gram_vectorizer_train),y_cora_train,norm.transform(X_t3gram_vectorizer_test),y_cora_test)")




#Nouvel Echantillonage  
cora_data_neg_sample = cora_data[cora_data['target'] == 1] #Negatives comments
cora_data_positive_sample = cora_data[cora_data['target'] == 0].reindex()  #Positive Comments

cora_resampling = pd.concat([pd.DataFrame(cora_data_positive_sample.sample(20000)),
                               pd.DataFrame(cora_data_neg_sample)])
100*(cora_resampling.groupby('target')['question_text'].count())/cora_resampling['target'].count()

#Cleaning the data 
cora_resampling['clean_question'] = cora_resampling['question_text'].apply(clean_str)
#Tokenizing and stopwords removing
cora_resampling['tokeniZ_stopWords_question'] = cora_resampling['clean_question'].apply(tokeniZ_stopWords)
#Words Stemming
cora_resampling['stemming_question'] = [[ps.stem(word) for word in words] for words in cora_resampling['tokeniZ_stopWords_question'] ]
cora_resampling['stemming_question_for_tfidf'] = [' '.join(words) for words in cora_resampling['stemming_question']] 

X_cora_train, X_cora_test, y_cora_train, y_cora_test = train_test_split(
            cora_resampling['stemming_question_for_tfidf']
            ,cora_resampling['target'], 
            test_size=0.3, random_state=42)
X_cora_train.shape, X_cora_test.shape, y_cora_train.shape, y_cora_test.shape

#Vectorization with tf-idf
X_Singt3gram_vectorizer_train = st3gram_vectorizer.fit_transform(X_cora_train)
X_Singt3gram_vectorizer_test  = st3gram_vectorizer.transform(X_cora_test)

X_t3gram_vectorizer_train = t3gram_vectorizer.fit_transform(X_cora_train)
X_t3gram_vectorizer_test  = t3gram_vectorizer.transform(X_cora_test)




get_ipython().run_cell_magic('time', '', "#OPTIMISATION DES HYPERPARAMETRES DU MODEL LINEARSVC AVEC GRIDSEARCH - We want to increase the regularization of the classifier (decrease parameter C of LinearSVC)\nfrom sklearn.model_selection import GridSearchCV\nparam_grid={'C': np.linspace(1, 10, 50), 'penalty': ['l2']}\nmodel = LogisticRegression(n_jobs=-1)\nkf = KFold(n_splits=3,random_state=random_state)\n\ngs=GridSearchCV(model, param_grid, scoring='accuracy', n_jobs=-1, cv=kf, verbose=False)\ngs.fit(X_t3gram_vectorizer_train,y_cora_train)\ngs_test_score=balanced_accuracy_score(y_cora_test, gs.predict(X_t3gram_vectorizer_test))")




print("Best Accuracy on training sample: {:.3f} with hyperparameters {}".format(gs.best_score_, gs.best_params_))
print("Best Accuracy on validation sample: {:.3f} ".format(gs_test_score))




print("Best Accuracy on training sample: {:.3f} with hyperparameters {}".format(gs.best_score_, gs.best_params_))

gs_test_scorer=accuracy_score(y_cora_test, gs.predict(X_t3gram_vectorizer_test))
print("Best Accuracy on validation sample: {:.3f} ".format(gs_test_scorer))
print("Best balanced Accuracy on validation sample: {:.3f} ".format(gs_test_score))




get_ipython().run_cell_magic('time', '', "from sklearn.metrics import balanced_accuracy_score\n\n#Let's try to increase the data regularization by using the Standardization method\n#Lets generate the learning curve of the optimized model\nselector = SelectPercentile(f_classif,percentile=30)\ndef modelize_LogReg(list_clf,X,y,X_test,y_test):\n    for clf in list_clf:\n        Clf1 = clf(C=3.272727272727273).fit(X,y)\n        Clf2 = Pipeline([('Feature Selection',selector),('Classification',clf(C=3.272727272727273))]).fit(X,y)\n        cv = StratifiedShuffleSplit(n_splits=3 , test_size=.3, random_state=51)\n        print('Model : %s' %type(Clf1).__name__)\n        print('With all features                                              /    With 70% of the best features')\n        print('                 Balanced Accuracy Score    F1 Score                Balanced Accuracy Score    F1 Score')\n        print('training :       %f                   %f           /    %f                   %f' \n              %(balanced_accuracy_score(y,Clf1.predict(X)),f1_score(y,Clf1.predict(X),average='weighted'),balanced_accuracy_score(y,Clf2.predict(X)),f1_score(y,Clf2.predict(X),average='weighted')))\n        print('Test     :       %f                   %f           /    %f                   %f' \n              %(balanced_accuracy_score(y_test,Clf1.predict(X_test)),f1_score(y_test,Clf1.predict(X_test),average='weighted')\n                ,balanced_accuracy_score(y_test,Clf2.predict(X_test)),f1_score(y_test,Clf2.predict(X_test),average='weighted')))\n        plot_learning_curve(Clf1, X, y, Clf2, cv, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), scoring='balanced_accuracy')\n        print('=============================================')\n    \nmodelize_LogReg([LogisticRegression],X_t3gram_vectorizer_train,y_cora_train,X_t3gram_vectorizer_test,y_cora_test)")




get_ipython().run_cell_magic('time', '', 'norm=StandardScaler(with_mean=False)\nmodelize_LogReg([LogisticRegression],norm.fit_transform(X_t3gram_vectorizer_train),y_cora_train,norm.transform(X_t3gram_vectorizer_test),y_cora_test)')




get_ipython().run_cell_magic('time', '', 'selector = SelectPercentile(f_classif,percentile=30)\nmodelize_LogReg([LogisticRegression],normalize(X_t3gram_vectorizer_train),y_cora_train,normalize(X_t3gram_vectorizer_test),y_cora_test)')




get_ipython().run_cell_magic('time', '', "random_state=42\nkf = KFold(n_splits=3,random_state=random_state)\n\nsearchCV = LogisticRegressionCV(\n        Cs=list(np.linspace(0.0001, 100, 250))\n        ,penalty='l1'\n        ,scoring='accuracy'\n        ,cv=kf\n        ,random_state=random_state\n        ,max_iter=10000\n        ,fit_intercept=True\n        ,solver='saga'\n        ,tol=10\n    )\nsearchCV.fit(X_Singt3gram_vectorizer_train, y_cora_train)")




print ('Max accuracy training sample:', searchCV.scores_[1].mean(axis=0).max())
print('Max accuracy validation sample:', accuracy_score(searchCV.predict(X_Singt3gram_vectorizer_test),y_cora_test))




get_ipython().run_cell_magic('time', '', "from sklearn.metrics import balanced_accuracy_score\n\n#Let's try to increase the data regularization by using the Standardization method\n#Lets generate the learning curve of the optimized model\nselector = SelectPercentile(f_classif,percentile=70)\ndef modelize_LogReg(list_clf,X,y,X_test,y_test):\n    for clf in list_clf:\n        Clf1 = clf.fit(X,y)\n        Clf2 = Pipeline([('Feature Selection',selector),('Classification',clf)]).fit(X,y)\n        cv = StratifiedShuffleSplit(n_splits=3 , test_size=.3, random_state=51)\n        print('Model : %s' %type(Clf1).__name__)\n        print('With all features                                              /    With 70% of the best features')\n        print('                 Balanced Accuracy Score    F1 Score                Balanced Accuracy Score    F1 Score')\n        print('training :       %f                   %f           /    %f                   %f' \n              %(balanced_accuracy_score(y,Clf1.predict(X)),f1_score(y,Clf1.predict(X),average='weighted'),balanced_accuracy_score(y,Clf2.predict(X)),f1_score(y,Clf2.predict(X),average='weighted')))\n        print('Test     :       %f                   %f           /    %f                   %f' \n              %(balanced_accuracy_score(y_test,Clf1.predict(X_test)),f1_score(y_test,Clf1.predict(X_test),average='weighted')\n                ,balanced_accuracy_score(y_test,Clf2.predict(X_test)),f1_score(y_test,Clf2.predict(X_test),average='weighted')))\n        plot_learning_curve(Clf1, X, y, Clf2, cv, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), scoring='balanced_accuracy')\n        print('=============================================')\n    \nmodelize_LogReg([LogisticRegressionCV(Cs=list(np.linspace(0.0001, 100, 250)),penalty='l2',scoring='accuracy',cv=kf\n     ,random_state=random_state,max_iter=10000,fit_intercept=True,solver='newton-cg',tol=10)],X_Singt3gram_vectorizer_train,y_cora_train,X_Singt3gram_vectorizer_test,y_cora_test)")




get_ipython().run_cell_magic('time', '', "#Lets generate the learning curve of the optimized model\nselector = SelectPercentile(f_classif,percentile=70)\ndef modelize_LogReg(list_clf,X,y,X_test,y_test):\n    for clf in list_clf:\n        Clf1 = clf(C= 3.3855421686746987, penalty= 'l2').fit(X,y)\n        Clf2 = Pipeline([('Feature Selection',selector),('Classification',clf(C=3.3855421686746987, penalty='l2'))]).fit(X,y)\n        cv = StratifiedShuffleSplit(n_splits=3 , test_size=.3, random_state=42)\n        print('Model : %s' %type(Clf1).__name__)\n        print('With all features                                              /    With 70% of the best features')\n        print('                 Accuracy Score             F1 Score                Accuracy Score             F1 Score')\n        print('training :       %f                   %f           /    %f                   %f' \n              %(balanced_accuracy_score(y,Clf1.predict(X)),f1_score(y,Clf1.predict(X),average='weighted'),\n                balanced_accuracy_score(y,Clf2.predict(X)),f1_score(y,Clf2.predict(X),average='weighted')))\n        print('Test     :       %f                   %f           /    %f                   %f' \n              %(balanced_accuracy_score(y_test,Clf1.predict(X_test)),f1_score(y_test,Clf1.predict(X_test),average='weighted')\n                ,balanced_accuracy_score(y_test,Clf2.predict(X_test)),f1_score(y_test,Clf2.predict(X_test),average='weighted')))\n        plot_learning_curve(Clf1, X, y, Clf2, cv, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), scoring='accuracy')\n        print('=============================================')\nnorm=StandardScaler(with_mean=False)\nmodelize_LogReg([LogisticRegression],norm.fit_transform(X_t3gram_vectorizer_train),y_cora_train,norm.transform(X_t3gram_vectorizer_test),y_cora_test)")




get_ipython().run_cell_magic('time', '', "random_state=42\nkf = KFold(n_splits=3,random_state=random_state)\n\nsearchCV = LogisticRegressionCV(\n        Cs=list(np.linspace(0.0001, 100, 250))\n        ,penalty='l1'\n        ,scoring='accuracy'\n        ,cv=kf\n        ,random_state=random_state\n        ,max_iter=10000\n        ,fit_intercept=True\n        ,solver='saga'\n        ,tol=10\n    )\nsearchCV.fit(X_Singt3gram_vectorizer_train, y_cora_train)")




print ('Max accuracy training sample:', searchCV.scores_[1].mean(axis=0).max())
print('Max accuracy validation sample:', accuracy_score(searchCV.predict(X_Singt3gram_vectorizer_test),y_cora_test))




get_ipython().run_cell_magic('time', '', "random_state=42\nkf = KFold(n_splits=3,random_state=random_state)\n\nsearchCV = LogisticRegressionCV(\n        Cs=list(np.linspace(0.0001, 100, 250))\n        ,penalty='l2'\n        ,scoring='accuracy'\n        ,cv=kf\n        ,random_state=random_state\n        ,max_iter=10000\n        ,fit_intercept=True\n        #,solver='saga'\n        ,tol=10\n    )\nsearchCV.fit(norm.fit_transform(X_Singt3gram_vectorizer_train), y_cora_train)")




print ('Max accuracy training sample:', searchCV.scores_[1].mean(axis=0).max())
print ('Max accuracy validation sample:', accuracy_score(searchCV.predict(norm.transform(X_Singt3gram_vectorizer_test)),y_cora_test))




searchCV.get_params














from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, KFold
from sklearn.feature_selection import SelectPercentile
from hyperopt import hp,fmin,Trials, tpe, STATUS_FAIL, STATUS_OK, space_eval, anneal
from hyperopt.pyll import scope
from hyperopt.pyll import stochastic

from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.decomposition import SparsePCA
from sklearn.preprocessing import normalize, StandardScaler

import time
random_state = 42
kf = KFold(n_splits=2,random_state=random_state)
n_iter= 50




get_ipython().run_cell_magic('time', '', "import time\nrandom_state = 42\n\nkf = KFold(n_splits=2,random_state=random_state)\nn_iter= 50\nModel_final_MLPClassifier = Pipeline([('Feature Selection',SelectPercentile(f_classif,percentile=70)),('Classification',MLPClassifier(random_state=random_state, activation = str(space_eval(space,best)['activation'] ),\n                                                 solver = str(space_eval(space,best)['solver']),\n                                                 alpha = int(best['alpha']),\n                                                 hidden_layer_sizes=tuple(space_eval(space,best)['hidden_layer_sizes']),\n                                                 learning_rate = str(space_eval(space,best)['learning_rate'] )))]).fit(X_t3gram_vectorizer_train_,y_cora_train_)\n\nModel_final_LogReg = Pipeline([('Feature Selection',SelectPercentile(f_classif,percentile=70)),('Classification',LogisticRegression(C=3.272727272727273))]).fit(X_t3gram_vectorizer_train_,y_cora_train_)\nModel_final_LinearSVC = Pipeline([('Feature Selection',SelectPercentile(f_classif,percentile=70)),('Classification',LinearSVC(C=0.4056225493975904))]).fit(X_Singt3gram_vectorizer_train_,y_cora_train_)\nModel_final_ExtraTreesClassifier = ExtraTreesClassifier(max_depth= 443.0, max_features= 'sqrt', min_samples_split= 5, n_estimators=117).fit(X_t3gram_vectorizer_train_,y_cora_train_)")




#mATRICES DE CONFUSIONS
from sklearn.metrics import confusion_matrix
cm_ETree = confusion_matrix(y_cora_test_, Model_final_ExtraTreesClassifier.predict(X_t3gram_vectorizer_test_))
cm_LinearSVC = confusion_matrix(y_cora_test_, Model_final_LinearSVC.predict(X_Singt3gram_vectorizer_test_))
cm_LogReg = confusion_matrix(y_cora_test_, Model_final_LogReg.predict(X_t3gram_vectorizer_test_))
cm_MLP = confusion_matrix(y_cora_test_, Model_final_MLPClassifier.predict(X_t3gram_vectorizer_test_))




#MATRICES DE CONFUSIONS NORMALISEES
cm_ExtraTreesClassifier = cm_ETree.astype('float') / cm_ETree.sum(axis=1)[:, np.newaxis]
cm_LinearSVC = cm_LinearSVC.astype('float') / cm_LinearSVC.sum(axis=1)[:, np.newaxis]
cm_LogisticRegression = cm_LogReg.astype('float') / cm_LogReg.sum(axis=1)[:, np.newaxis]
cm_MLPClassifier = cm_MLP.astype('float') / cm_MLP.sum(axis=1)[:, np.newaxis]




from sklearn.utils.multiclass import unique_labels
classes = unique_labels(y_cora_test_)
#dict_models = {'cm_ExtraTreesClassifier' : 'ExtraTreesClassifier', 'cm_LinearSVC': 'LinearSVC', 'cm_LogisticRegression': 'LogisticRegression', 'cm_MLPClassifier': 'MLPClassifier'}

fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2, figsize=(10,10))
aX = [ax1, ax2,ax3,ax4]
dict_models = ['ExtraTreesClassifier',  'LinearSVC',  'LogisticRegression',  'MLPClassifier']
for IDX, i in enumerate([cm_ExtraTreesClassifier, cm_LinearSVC, cm_LogisticRegression, cm_MLPClassifier]):
    im = aX[IDX].imshow(i, interpolation='nearest', cmap=plt.cm.Blues)
    #aX[IDX].figure.colorbar(im, aX[IDX]=aX[IDX])

    # We want to show all ticks...
    aX[IDX].set(xticks=np.arange(i.shape[1]), yticks=np.arange(i.shape[0]),
           xticklabels=classes, yticklabels=classes, # ... and label them with the respective list entries
           title= dict_models[IDX],
           ylabel='True label',
           xlabel='Predicted label')
    
    # Rotate the tick labels and set their alignment.
    #plt.setp(aX[i].get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    fmt = '.2f' if normalize else 'd'
    thresh = i.max() / 2.
    for t in range(i.shape[0]):
        for j in range(i.shape[1]):
            aX[IDX].text(j, t, format(i[t, j], fmt),
                    ha="center", va="center",
                    color="white" if i[t, j] > thresh else "black")




cm_MLPClassifier




from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
fpr1, tpr1, thresholds1 = roc_curve(y_cora_test_, Model_final_MLPClassifier.predict(X_t3gram_vectorizer_test_))
fpr2, tpr2, thresholds2 = roc_curve(y_cora_test_, Model_final_ExtraTreesClassifier.predict(X_t3gram_vectorizer_test_))
fpr3, tpr3, thresholds3 = roc_curve(y_cora_test_, Model_final_LinearSVC.predict(X_Singt3gram_vectorizer_test_))
fpr4, tpr4, thresholds4 = roc_curve(y_cora_test_, Model_final_LogReg.predict(X_t3gram_vectorizer_test_))

print('AUC - MLPClassifier: %.2f ' %(auc(fpr1, tpr1)))
print('AUC - ExtraTreesClassifier: %.2f ' %(auc(fpr2, tpr2)))
print('AUC - LinearSVCr: %.2f ' %(auc(fpr3, tpr3)))
print('AUC - LogisticRegression: %.2f ' %(auc(fpr4, tpr4)))




plt.figure(figsize=(12,10))
lw = 2
plt.plot(fpr1, tpr1, #color='darkorange',
         lw=lw, label='MLPClassifier')
plt.plot(fpr2, tpr2, #color='bleu', 
         lw=lw, label='ExtratreesClassifier')
plt.plot(fpr3, tpr3, #color='Olive', 
         lw=lw, label='LinearSVC')
plt.plot(fpr4, tpr4, #color='red', 
         lw=lw, label='LogisticRegression')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Proportion mal classée")
plt.ylabel("Proportion bien classée")
plt.title('ROC')
plt.legend(loc="lower right")
plt.savefig('AUC')




sub_df = pd.read_csv(zf_test.open('test.csv'))
sub_df_target  = pd.read_csv(zf_test.open('sample_submission.csv'))
#sub_df = pd.read_csv('all/test.csv')
#sub_df_target = pd.read_csv('all/sample_submission.csv')




sub_df_target.info()




sub_df.info()




sub_df.shape




sub_df.head()




#Cleaning the data 
sub_df['clean_question'] = sub_df['question_text'].apply(clean_str)

#Tokenizing and stopwords removing
sub_df['tokeniZ_stopWords_question'] = sub_df['clean_question'].apply(tokeniZ_stopWords)

#Words Stemming
sub_df['stemming_question'] = [[ps.stem(word) for word in words] for words in sub_df['tokeniZ_stopWords_question'] ]
sub_df['stemming_question_for_tfidf'] = [' '.join(words) for words in sub_df['stemming_question']] 




#T3gram questions vectorization
X_t3gram_vect_sub_test  = t3gram_vectorizer.transform(sub_df['stemming_question_for_tfidf'])




#X_t3gram_vect_sub_test




Model_final_MLPClassifier = Pipeline([('Feature Selection',SelectPercentile(f_classif,percentile=70)),('Classification',MLPClassifier(random_state=random_state, 
                                                activation = str(space_eval(space,best)['activation'] ),
                                                 solver = str(space_eval(space,best)['solver']),
                                                 alpha = int(best['alpha']),
                                                 hidden_layer_sizes=tuple(space_eval(space,best)['hidden_layer_sizes']),
                                                 learning_rate = str(space_eval(space,best)['learning_rate'] )))]).fit(X_t3gram_vectorizer_train,y_cora_train)




sub_df['target'] = Model_final_MLPClassifier.predict(X_t3gram_vect_sub_test)




final_submission1 = pd.merge(sub_df,sub_df_target,how='inner',on='qid' )




final_submission = (final_submission1[['qid','target']]).rename(columns={'target':'prediction'})




final_submission.to_csv('final_submission.csv', index=False, sep=',')




final_submission.head()




get_ipython().run_line_magic('ls', '')






