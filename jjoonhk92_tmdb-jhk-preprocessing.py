#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import ast
import json

from collections import Counter

import itertools
from itertools import zip_longest

import re
from wordcloud import WordCloud
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

import eli5

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('precision', '3')
pd.set_option('precision', 3)

import warnings
warnings.filterwarnings('ignore')


# In[3]:


#データを読み取る
#
train = pd.read_csv('../input/tmdb-box-office-prediction/train.csv')
#
test = pd.read_csv('../input/tmdb-box-office-prediction/test.csv')


# In[4]:


print(train.shape,test.shape)
train.columns


# In[5]:


train.loc[train['id'] == 391,'runtime'] = 96 #The Worst Christmas of My Lifeの上映時間を調べて入力
train.loc[train['id'] == 592,'runtime'] = 90 #А поутру они проснулисьの上映時間を調べて入力
train.loc[train['id'] == 925,'runtime'] = 86 #¿Quién mató a Bambi?の上映時間を調べて入力
train.loc[train['id'] == 978,'runtime'] = 93 #La peggior settimana della mia vitaの上映時間を調べて入力
train.loc[train['id'] == 1256,'runtime'] = 92 #Cry, Onion!の上映時間を調べて入力
train.loc[train['id'] == 1542,'runtime'] = 93 #All at Onceの上映時間を調べて入力
train.loc[train['id'] == 1875,'runtime'] = 93 #Vermistの上映時間を調べて入力
train.loc[train['id'] == 2151,'runtime'] = 108 #Mechenosetsの上映時間を調べて入力
train.loc[train['id'] == 2499,'runtime'] = 86 #Na Igre 2. Novyy Urovenの上映時間を調べて入力
train.loc[train['id'] == 2646,'runtime'] = 98 #My Old Classmateの上映時間を調べて入力
train.loc[train['id'] == 2786,'runtime'] = 111 #Revelationの上映時間を調べて入力
train.loc[train['id'] == 2866,'runtime'] = 96 #Tutto tutto niente nienteの上映時間を調べて入力


# In[6]:


test.loc[test['id'] == 3244,'runtime'] = 93 #La caliente niña Julietta	の上映時間を調べて入力
test.loc[test['id'] == 4490,'runtime'] = 90 #Pancho, el perro millonarioの上映時間を調べて入力
test.loc[test['id'] == 4633,'runtime'] = 108 #Nunca en horas de claseの上映時間を調べて入力
test.loc[test['id'] == 6818,'runtime'] = 90 #Miesten välisiä keskustelujaの上映時間を調べて入力

test.loc[test['id'] == 4074,'runtime'] = 103 #Shikshanachya Aaicha Ghoの上映時間を調べて入力
test.loc[test['id'] == 4222,'runtime'] = 91 #Street Knightの上映時間を調べて入力
test.loc[test['id'] == 4431,'runtime'] = 96 #Plus oneの上映時間を調べて入力
test.loc[test['id'] == 5520,'runtime'] = 86 #Glukhar v kinoの上映時間を調べて入力
test.loc[test['id'] == 5845,'runtime'] = 83 #Frau Müller muss weg!の上映時間を調べて入力
test.loc[test['id'] == 5849,'runtime'] = 140 #Shabdの上映時間を調べて入力
test.loc[test['id'] == 6210,'runtime'] = 104 #The Last Breathの上映時間を調べて入力
test.loc[test['id'] == 6804,'runtime'] = 140 #Chaahat Ek Nasha...の上映時間を調べて入力
test.loc[test['id'] == 7321,'runtime'] = 87 #El truco del mancoの上映時間を調べて入力


# In[7]:


df = pd.concat([train, test]).set_index("id")


# In[8]:


#columnsを確認し、除外する変数をdrop
print(df.columns)
# 使わない列を消す
df = df.drop(["poster_path", "status", "original_title"], axis=1) # "overview",  "imdb_id", 


# In[9]:


# logを取っておく
df["log_revenue"] = np.log10(df["revenue"])
# homepage: 有無に
df["homepage"] = ~df["homepage"].isnull()


# In[10]:


dfdic_feature = {}


# In[11]:


get_ipython().run_cell_magic('time', '', "# JSON text を辞書型のリストに変換\nimport ast\ndict_columns = ['belongs_to_collection', 'genres', 'production_companies',\n                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']\n\nfor col in dict_columns:\n       df[col]=df[col].apply(lambda x: [] if pd.isna(x) else ast.literal_eval(x) )")


# In[12]:


# 各ワードの有無を表す 01 のデータフレームを作成
def count_word_list(series):
    len_max = series.apply(len).max() # ジャンル数の最大値
    tmp = series.map(lambda x: x+["nashi"]*(len_max-len(x))) # listの長さをそろえる
    
    word_set = set(sum(list(series.values), [])) # 全ジャンル名のset
    for n in range(len_max):
        word_dfn = pd.get_dummies(tmp.apply(lambda x: x[n]))
        word_dfn = word_dfn.reindex(word_set, axis=1).fillna(0).astype(int)
        if n==0:
            word_df = word_dfn
        else:
            word_df = word_df + word_dfn
    
    return word_df#.drop("nashi", axis=1)


# In[13]:


#budgetが0の物を予測（テスト）、0でない物をtrainingデータとする
budget0 = df[df["budget"] == 0]
budget = df[df["budget"] != 0]
train_X = budget[["popularity","runtime"]]
train_y = budget["budget"]
test_X = budget0[["popularity","runtime"]]
test_y = budget0["budget"]


# In[14]:


train_X.fillna(0, inplace = True)
test_X.fillna(0, inplace = True)


# In[15]:


#budgetが0の物を線形回帰で予測
from sklearn.linear_model import RidgeCV
rcv= RidgeCV(cv=3, alphas = 10**np.arange(-2, 2, 0.1))
rcv.fit(train_X, train_y)
y_pred = rcv.predict(test_X)


# In[16]:


budget0


# In[17]:


budget0.index = range(0,2023)


# In[18]:


budget_pred = pd.DataFrame(y_pred,columns=["pred"])
budget_pred = pd.concat([budget.index,budget_pred],axis = 1)
budget_pred


# In[19]:


#予算が0を下回っているものはおかしいので0に戻す。
budget_pred.loc[budget_pred["pred"] < 0, "pred"] = 0


# In[20]:


df = pd.merge(df, budget_pred, on="id", how="left") 
df.loc[budget_pred["id"]-1, "budget"] = df.loc[budget_pred["id"]-1, "pred"]
df = df.drop("pred", axis=1)


# In[21]:


df["genre_names"] = df["genres"].apply(lambda x : [ i["name"] for i in x])


# In[22]:


dfdic_feature["genre"] = count_word_list(df["genre_names"])
# TV movie は1件しかないので削除
dfdic_feature["genre"] = dfdic_feature["genre"].drop("TV Movie", axis=1)
dfdic_feature["genre"].head()


# In[23]:


# train内の作品数が10件未満の言語は "small" に集約
n_language = df.loc[:train.index[-1], "original_language"].value_counts()
large_language = n_language[n_language>=10].index
df.loc[~df["original_language"].isin(large_language), "original_language"] = "small"


# In[24]:


df["original_language"] = df["original_language"].astype("category")


# In[25]:


# one_hot_encoding
dfdic_feature["original_language"] = pd.get_dummies(df["original_language"])
#dfdic_feature["original_language"] = dfdic_feature["original_language"].loc[:, dfdic_feature["original_language"].sum()>0]
dfdic_feature["original_language"].head()


# In[26]:


df["production_names"] = df["production_companies"].apply(lambda x : [ i["name"] for i in x])
#.fillna("[{'name': 'nashi'}]").map(to_name_list)


# In[27]:


get_ipython().run_line_magic('time', 'tmp = count_word_list(df["production_names"])')


# In[28]:


# train内の件数が多い物のみ選ぶ
def select_top_n(df, topn=9999, nmin=2):  # topn:上位topn件, nmin:作品数nmin以上
#    if "small" in df.columns:
#        df = df.drop("small", axis=1)
    n_word = (df.loc[train["id"]]>0).sum().sort_values(ascending=False)
    # 作品数がnmin件未満
    smallmin = n_word[n_word<nmin].index
    # 上位topn件に入っていない
    smalln = n_word.iloc[topn+1:].index
    small = set(smallmin) | set(smalln)
    # 件数の少ないタグのみの作品
    df["small"] = df[small].sum(axis=1) #>0
    
    return df.drop(small, axis=1)


# In[29]:


# trainに50本以上作品のある会社
dfdic_feature["production_companies"] = select_top_n(tmp, nmin=50)
dfdic_feature["production_companies"].head()


# In[30]:


# 国名のリストに
df["country_names"] = df["production_countries"].apply(lambda x : [ i["name"] for i in x])
df_country = count_word_list(df["country_names"])


# In[31]:


# 2か国だったら、0.5ずつに
df_country = (df_country.T/df_country.sum(axis=1)).T.fillna(0)


# In[32]:


# 30作品以上の国のみ
dfdic_feature["production_countries"] = select_top_n(df_country, nmin=30)
dfdic_feature["production_countries"].head()


# In[33]:


df["keyword_list"] = df["Keywords"].apply(lambda x : [ i["name"] for i in x])


# In[34]:


def encode_topn_onehot(series, topn):
    # 多いワード順に
    word_count = pd.Series(collections.Counter(sum(list(series.values), [])))
    word_count = word_count.sort_values(ascending=False)
    
    df_topn = df[[]].copy()  # index のみのDF
    # 上位topn件のキーワードのみ
    for word in word_count.iloc[:topn].index:  # .drop("nashi")
        df_topn[word] = series.apply(lambda x: word in x)*1
    
    return df_topn
    


# In[35]:


dfdic_feature["Keywords"] = encode_topn_onehot(df["keyword_list"], 100)


# In[36]:


df["num_Keywords"] = df["keyword_list"].apply(len)


# In[37]:


df["language_names"] = df["spoken_languages"].apply(lambda x : [ i["name"] for i in x])
df["n_language"] = df["language_names"].apply(len)
# 欠損値は１にする(データを見ると無声映画ではない)
df.loc[df["n_language"]==0, "n_language"] = 1


# In[38]:


# 英語が含まれるか否か
df["speak_English"] = df["language_names"].apply(lambda x : "English" in x)


# In[39]:


import datetime


# In[40]:


# 公開日の欠損1件 id=3829
# May,2000 (https://www.imdb.com/title/tt0210130/) 
# 日は不明。1日を入れておく
df.loc[3829, "release_date"] = "5/1/00"


# In[41]:


df["release_year"] = pd.to_datetime(df["release_date"]).dt.year.astype(int)
# 年の20以降を、2020年より後の未来と判定してしまうので、補正。
df.loc[df["release_year"]>2020, "release_year"] = df.loc[df["release_year"]>2020, "release_year"]-100

df["release_month"] = pd.to_datetime(df["release_date"]).dt.month.astype(int)
df["release_day"] = pd.to_datetime(df["release_date"]).dt.day.astype(int)


# In[42]:


# datetime型に
df["release_date"] = df.apply(lambda s: datetime.datetime(
    year=s["release_year"],month=s["release_month"],day=s["release_day"]), axis=1)


# In[43]:


df["release_dayofyear"] = df["release_date"].dt.dayofyear
df["release_dayofweek"] = df["release_date"].dt.dayofweek


# In[44]:


# 月、曜日は カテゴリ型に
df["release_month"] = df["release_month"].astype('category')
df["release_dayofweek"] = df["release_dayofweek"].astype('category')


# In[45]:


# collection 名を抽出
df["collection_name"] = df["belongs_to_collection"].apply(lambda x : x[0]["name"] if len(x)>0 else "nashi")
# 無い場合、"nashi"に


# In[46]:


# シリーズの作品数
#df = pd.merge( df, df.groupby("collection_name").count()[["budget"]].rename(columns={"budget":"count_collection"}), 
#         on="collection_name", how="left")
# indexがずれるので、戻す
#df.index = df.index+1

df["count_collection"] = df["collection_name"].apply(lambda x : (df["collection_name"]==x).sum())
# シリーズ以外の場合0
df.loc[df["collection_name"]=="nashi", "count_collection"] = 0


# In[47]:


# シリーズ何作目か
df["number_in_collection"] = df.sort_values("release_date").groupby("collection_name").cumcount()+1
# シリーズ以外の場合0
df.loc[df["collection_name"]=="nashi", "number_in_collection"] = 0


# In[48]:


get_ipython().run_cell_magic('time', '', '# 同シリーズの自分より前の作品の平均log(revenue)\ndf["collection_av_logrevenue"] = [ df.loc[(df["collection_name"]==row["collection_name"]) & \n                                          (df["number_in_collection"]<row["number_in_collection"]),\n                                          "log_revenue"].mean() \n     for key,row in df.iterrows() ]')


# In[49]:


# 欠損(nashi) の場合、nashi での平均
df.loc[df["collection_name"]=="nashi", "collection_av_logrevenue"] = df.loc[df["collection_name"]=="nashi", "log_revenue"].mean()


# In[50]:


# train に無くtestだけにあるシリーズの場合、シリーズもの全部の平均
collection_mean = df.loc[df["collection_name"]!="nashi", "log_revenue"].mean()  # シリーズもの全部の平均
df["collection_av_logrevenue"] = df["collection_av_logrevenue"].fillna(collection_mean)  


# In[51]:


df_features = pd.concat(dfdic_feature, axis=1)


# In[52]:


# 欠測と0は、0ではないものの平均で埋める
df["runtime"] = df["runtime"].fillna(df.loc[df["runtime"]>0, "runtime"].mean())
df.loc[df["runtime"]==0, "runtime"] = df.loc[df["runtime"]>0, "runtime"].mean()


# In[53]:


#plt.scatter(df["budget"]+1, df["log_revenue"], s=1)
#plt.xscale("log")
#plt.xrange([])


# In[54]:


df.columns


# In[55]:


df[["original_language", "collection_name"]] = df[["original_language", "collection_name"]].astype("category")


# In[56]:


df_use = df[['budget', 'homepage', 'popularity','runtime','n_language', 
             "num_Keywords", "speak_English",
             'release_year', 'release_month','release_dayofweek', 
             'collection_av_logrevenue' ,"count_collection","number_in_collection"
            ]]
df_use.head()


# In[57]:


df_use = pd.get_dummies(df_use)


# In[ ]:





# In[58]:


train_add = pd.read_csv('../input/tmdb-competition-additional-features/TrainAdditionalFeatures.csv')
test_add = pd.read_csv('../input/tmdb-competition-additional-features/TestAdditionalFeatures.csv')
train_add.head()


# In[59]:


df = pd.merge(df, pd.concat([train_add, test_add]), on="imdb_id", how="left")


# In[60]:


add_cols = ["popularity2", "rating", "totalVotes"]
df[add_cols] = df[add_cols].fillna(df[add_cols].mean())


# In[61]:


train2 = pd.read_csv('../input/tmdb-box-office-prediction-more-training-data/additionalTrainData.csv')
train3 = pd.read_csv('../input/tmdb-box-office-prediction-more-training-data/trainV3.csv')
train3.head()


# In[ ]:





# In[62]:


#全て小文字に変換
def lower_text(text):
    return text.lower()

#記号の排除
def remove_punct(text):
    text = text.replace('-', ' ')  # - は単語の区切りとみなす
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)

def remove_stopwords(words, stopwords):#不要な単語を削除
    words = [word for word in words if word not in stopwords]
    return words


# In[63]:


from nltk.corpus import stopwords
stop_words = stopwords.words('english')


# In[64]:


# 英語でよく使う単語が入っていない文章を確認
df.loc[df["overview"].apply(lambda x : str(x)).apply(lambda x : lower_text(x)
                                ).str.contains("nan|the|where|with|from|and|for|his|her|over")==False, "overview"]
#train3.loc[train3["overview"].apply(lambda x : str(x)).apply(lambda x : lower_text(x)).str.contains("nan|the|where|with|from|and|for|his|her|over")==False, "overview"]


# In[65]:


no_english_overview_id = [157, 2863, 4616]   # 上のデータを目で確認
no_english_tagline_id = [3255, 3777, 4937]   # Tfidf で非英語の単語があったもの


# In[66]:


from gensim.models import word2vec


# In[67]:


col_text = ["overview", "tagline"] # "title", 
all_text = pd.concat([df[col_text], train2[col_text], train3[col_text]])


# In[68]:


# 英語以外と"nan"は除外
all_text.loc[no_english_overview_id, "overview"] = np.nan
all_text.loc[no_english_tagline_id, "tagline"] = np.nan
all_text.loc[all_text["tagline"]=="nan", "tagline"] = np.nan


# In[69]:


all_texts = all_text.stack()
all_texts=all_texts.apply(lambda x : str(x))
all_texts=all_texts.apply(lambda x : lower_text(x))
all_texts=all_texts.apply(lambda x : remove_punct(x))


# In[70]:


all_texts.to_csv("./alltexts_for_w2v.txt", index=False, header=False)
docs = word2vec.LineSentence("alltexts_for_w2v.txt")


# In[71]:


get_ipython().run_cell_magic('time', '', '\nmodel = word2vec.Word2Vec(docs, sg=1, size=100, min_count=5, window=5, iter=100)\nmodel.save("./alltexts_w2v1_sg.model")')


# In[72]:


# model = word2vec.Word2Vec.load("./alltexts_w2v1_cbow.model")
# model = word2vec.Word2Vec.load("./alltexts_w2v1_sg.model")


# In[73]:


model.most_similar(positive=['father'])


# In[74]:


model.most_similar(positive=['human'])


# In[75]:


# 単語ベクトルの mean, max を文章ベクトルにする
def get_doc_vector(doc, method="mean", weight=None):
    split_doc = doc.split(" ")
    if weight==None:
        weight = dict(zip(model.wv.vocab.keys(), np.ones(len(model.wv.vocab))))
        
    word_vecs = [ model[word]*weight[word] for word in split_doc if word in model.wv.vocab.keys() ]
    
    if len(word_vecs)==0:
        doc_vec = []
    elif method=="mean":
        doc_vec =  np.mean(word_vecs, axis=0)
    elif method=="max":
        doc_vec =  np.max(word_vecs, axis=0)
    elif method=="meanmax":
        doc_vec =  np.mean(word_vecs, axis=0)+np.max(word_vecs, axis=0)
    return doc_vec


# In[76]:


#単語数
df['overview_word_count'] = df['overview'].apply(lambda x: len(str(x).split()))
#文字数
df['overview_char_count'] = df['overview'].apply(lambda x: len(str(x)))
# 記号の個数
df['overview_punctuation_count'] = df['overview'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))


# In[77]:


# 前処理
df['_overview']=df['overview'].apply(lambda x : str(x)
                            ).apply(lambda x : lower_text(x)).apply(lambda x : remove_punct(x))


# In[78]:


#短縮形を元に戻す
shortened = {
    '\'m': ' am',
    '\'re': ' are',
    'don\'t': 'do not',
    'doesn\'t': 'does not',
    'didn\'t': 'did not',
    'won\'t': 'will not',
    'wanna': 'want to',
    'gonna': 'going to',
    'gotta': 'got to',
    'hafta': 'have to',
    'needa': 'need to',
    'outta': 'out of',
    'kinda': 'kind of',
    'sorta': 'sort of',
    'lotta': 'lot of',
    'lemme': 'let me',
    'gimme': 'give me',
    'getcha': 'get you',
    'gotcha': 'got you',
    'letcha': 'let you',
    'betcha': 'bet you',
    'shoulda': 'should have',
    'coulda': 'could have',
    'woulda': 'would have',
    'musta': 'must have',
    'mighta': 'might have',
    'dunno': 'do not know',
}
df["overview"] = df["overview"].replace(shortened)
train["overview"] = train["overview"].replace(shortened)


# In[79]:


df["overview"]=df["overview"].apply(lambda x : remove_punct(x))
train["overview"]=train["overview"].apply(lambda x : remove_punct(x))


# In[80]:


# 連続した数字を0で置換
def normalize_number(text):
    replaced_text = re.sub(r'\d+', '0', text)
    return replaced_text


# In[81]:


df["overview"]=df["overview"].apply(lambda x : normalize_number(x))
train["overview"]=train["overview"].apply(lambda x : normalize_number(x))


# In[82]:


#レンマ化
from nltk.stem.wordnet import WordNetLemmatizer

wnl = WordNetLemmatizer()
df["overview"]=df["overview"].apply(wnl.lemmatize)
train["overview"]=train["overview"].apply(wnl.lemmatize)


# In[83]:


#空白ごとの文章の分割
df["overview"]=df["overview"].apply(lambda x : str(x).split())
train["overview"]=train["overview"].apply(lambda x : str(x).split())


# In[84]:


df_overview = df["overview"]


# In[85]:


def most_common(docs, n=100):#(文章、上位n個の単語)#上位n個の単語を抽出
    fdist = Counter()
    for doc in docs:
        for word in doc:
            fdist[word] += 1
    common_words = {word for word, freq in fdist.most_common(n)}
    print('{}/{}'.format(n, len(fdist)))
    return common_words


# In[86]:


most_common(df_overview,100)


# In[87]:


def get_stop_words(docs, n=100, min_freq=1):#上位n個の単語、頻度がmin_freq以下の単語を列挙（あまり特徴のない単語等）
    fdist = Counter()
    for doc in docs:
        for word in doc:
            fdist[word] += 1
    common_words = {word for word, freq in fdist.most_common(n)}
    rare_words = {word for word, freq in fdist.items() if freq <= min_freq}
    stopwords = common_words.union(rare_words)
    print('{}/{}'.format(len(stopwords), len(fdist)))
    return stopwords


# In[88]:


stopwords = get_stop_words(df_overview)
stopwords


# In[89]:


def remove_stopwords(words, stopwords):#不要な単語を削除
    words = [word for word in words if word not in stopwords]
    return words


# In[90]:


df["overview"]=df["overview"].apply(lambda x : remove_stopwords(x,stopwords))
train["overview"]=train["overview"].apply(lambda x : remove_stopwords(x,stopwords))


# In[91]:


df["overview"]=[" ".join(review) for review in df["overview"].values]
train["overview"]=[" ".join(review) for review in train["overview"].values]


# In[92]:


from sklearn.feature_extraction.text import TfidfVectorizer#ベクトル化
vec_tfidf = TfidfVectorizer()
X = vec_tfidf.fit_transform(df["overview"])
Tfid_overview = pd.DataFrame(X.toarray(), columns=vec_tfidf.get_feature_names())

X2 = vec_tfidf.fit_transform(df["overview"])
Tfid_train_overview = pd.DataFrame(X2.toarray(), columns=vec_tfidf.get_feature_names())


# In[93]:


df['_tagline']=df['tagline'].apply(lambda x : str(x)
                                 ).apply(lambda x : lower_text(x)).apply(lambda x : remove_punct(x))


# In[94]:


#ベクトル化
# from sklearn.feature_extraction.text import TfidfVectorizer
# vec_tfidf = TfidfVectorizer()
# X = vec_tfidf.fit_transform(df['tagline'])
# Tfidf_tagline = pd.DataFrame(X.toarray(), columns=vec_tfidf.get_feature_names())
# X = vec_tfidf.fit_transform(df['overview'].dropna())
# Tfidf_overview = pd.DataFrame(X.toarray(), columns=vec_tfidf.get_feature_names())


# In[95]:


get_ipython().run_line_magic('time', 'df_tagline =  df["_tagline"].apply(get_doc_vector, method="meanmax").apply(pd.Series)')


# In[96]:


df_tagline = df_tagline.fillna(0).add_prefix("tagline_")


# In[97]:


#単語数
df['title_word_count'] = df['title'].apply(lambda x: len(str(x).split()))
#文字数
df['title_char_count'] = df['title'].apply(lambda x: len(str(x)))
# 記号の個数
df['title_punctuation_count'] = df['title'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))


# In[98]:


df['tagline']=df['tagline'].apply(lambda x : str(x))
df["tagline"] = df["tagline"].replace(shortened)
df['tagline']=df['tagline'].apply(lambda x : lower_text(x))
df['tagline']=df['tagline'].apply(lambda x : remove_punct(x))
df["tagline"]=df["tagline"].apply(lambda x : normalize_number(x))
df['tagline']=df['tagline'].apply(lambda x : str(x).split())


# In[99]:


tagline = df["tagline"]


# In[100]:


most_common(tagline)


# In[101]:


stopwords = get_stop_words(tagline)


# In[102]:


df['tagline']=df['tagline'].apply(lambda x : remove_stopwords(x,stopwords))


# In[103]:


nan = {"nan"}
def remove_nan(words):
    words = [word for word in words if word not  in nan]
    return words


# In[104]:


df['tagline']=df['tagline'].apply(lambda x : remove_nan(x))


# In[105]:


df['tagline']=[" ".join(review) for review in df['tagline'].values]


# In[106]:


#ベクトル化
X = vec_tfidf.fit_transform(df['tagline'])
Tfid_tagline = pd.DataFrame(X.toarray(), columns=vec_tfidf.get_feature_names())


# In[107]:


df_use2 = df[["runtime",'budget','tagline_char_count']]


# In[108]:


df_use2 = pd.concat([df_use2,Tfid_overview],axis=1)


# In[109]:


#使用する変数
df_use2 = df_use2.loc[:,~df_use.columns.duplicated()]


# In[110]:


# Keywords を全部並べたものを、文とみなしてベクトル化
get_ipython().run_line_magic('time', 'df_keyword_w2v = df["keyword_list"].apply(" ".join).apply(get_doc_vector, method="mean").apply(pd.Series)')
df_keyword_w2v = df_keyword_w2v.fillna(0).add_prefix("keyword_")


# In[111]:


#castの中にある俳優の名前をリスト化させる
list_of_cast_names = list(df['cast'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
df['num_cast'] = df['cast'].apply(lambda x: len(x) if x != {} else 0)
df['all_cast'] = df['cast'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')


top_cast_names = [m[0] for m in Counter([i for j in list_of_cast_names for i in j]).most_common(30)]
for g in top_cast_names:
    df['cast_name_' + g] = df['all_cast'].apply(lambda x: 1 if g in x else 0)


# In[112]:


list_of_cast_genders = list(df['cast'].apply(lambda x: [i['gender'] for i in x] if x != {} else []).values)

df['genders_0_cast'] = df['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))
df['genders_1_cast'] = df['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))
df['genders_2_cast'] = df['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))    

#df = df.drop(['cast'], axis=1)

df['cast_gen0_ratio'] = df['genders_0_cast'].sum()/df['num_cast'].sum()
df['cast_gen1_ratio'] = df['genders_1_cast'].sum()/df['num_cast'].sum()
df['cast_gen2_ratio'] = df['genders_2_cast'].sum()/df['num_cast'].sum()


# In[113]:


#crewのname
list_of_crew_names = list(df['crew'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
df['num_crew'] = df['crew'].apply(lambda x: len(x) if x != {} else 0)
df['all_crew'] = df['crew'].apply(lambda x: ','.join(sorted([i['name'] for i in x])) if x != {} else '')
top_crew_names = [m[0] for m in Counter([i for j in list_of_crew_names for i in j]).most_common(15)]
for g in top_crew_names:
    df['crew_name_' + g] = df['all_crew'].apply(lambda x: 1 if g in x else 0)


# In[114]:


list_of_crew_department = list(df['crew'].apply(lambda x: [i['department'] for i in x] if x != {} else []).values)
df['all_department'] = df['crew'].apply(lambda x: '　'.join(sorted([i['department']for i in x])) if x != {} else '')
top_crew_department = [m[0] for m in Counter(i for j in list_of_crew_department for i in j).most_common(12)]
for g in top_crew_department:
    df['crew_department_' + g] = df['crew'].apply(lambda x: sum([1 for i in x if i['department'] == g]))


# In[115]:


list_of_crew_job = list(df['crew'].apply(lambda x: [i['job'] for i in x] if x != {} else []).values)
top_crew_job = [m[0] for m in Counter(i for j in list_of_crew_job for i in j).most_common(10)]
for g in top_crew_job:
    df['crew_job_' + g] = df['crew'].apply(lambda x: sum([1 for i in x if i['job'] == g]))


# In[116]:


df['genders_0_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))
df['genders_1_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))
df['genders_2_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))
df['crew_gen0_ratio'] = df['genders_0_crew'].sum()/df['num_crew'].sum()
df['crew_gen1_ratio'] = df['genders_1_crew'].sum()/df['num_crew'].sum()
df['crew_gen2_ratio'] = df['genders_2_crew'].sum()/df['num_crew'].sum()


# In[ ]:





# In[117]:


all_crew_job = [m[0] for m in Counter([i for j in list_of_crew_job for i in j]).most_common()]


# In[118]:


all_crew_department = [m[0] for m in Counter([i for j in list_of_crew_department for i in j]).most_common()]


# In[119]:


def select_department(list_dict, department):
    return [ dic['name'] for dic in list_dict if dic['department']==department]


# In[120]:


for z in all_crew_department:
    df['{}_list'.format(z)] = df["crew"].apply(select_department, department=z)
    globals()[z] = [m[0] for m in Counter([i for j in df['{}_list'.format(z)] for i in j]).most_common(15)]
    for i in globals()[z]:
        df['crew_{}_{}'.format(z,i)] = df['{}_list'.format(z)].apply(lambda x: sum([1 for i in x]))


# In[121]:


def select_job(list_dict, job):
    return [ dic["name"] for dic in list_dict if dic["job"]==job]


# In[122]:


for z in top_crew_job:
    df['{}_list'.format(z)] = df["crew"].apply(select_job, job=z)
    globals()[z] = [m[0] for m in Counter([i for j in df['{}_list'.format(z)] for i in j]).most_common(15)]
    for i in globals()[z]:
        df['crew_{}_{}'.format(z,i)] = df['{}_list'.format(z)].apply(lambda x: sum([1 for i in x]))


# In[123]:


df.columns


# In[124]:


df_use3=df[['num_cast','all_cast','cast_name_Samuel L. Jackson','cast_name_Robert De Niro','cast_name_Bruce Willis',
'cast_name_Morgan Freeman','cast_name_Liam Neeson','cast_name_Willem Dafoe','cast_name_Steve Buscemi',
'cast_name_Sylvester Stallone','cast_name_Nicolas Cage','cast_name_Matt Damon','cast_name_J.K. Simmons',
'cast_name_John Goodman','cast_name_Julianne Moore','cast_name_Christopher Walken','cast_name_Robin Williams',
'cast_name_Johnny Depp','cast_name_Stanley Tucci','cast_name_Harrison Ford','cast_name_Richard Jenkins',
'cast_name_Ben Stiller','cast_name_Susan Sarandon','cast_name_Brad Pitt','cast_name_Tom Hanks',
'cast_name_Keith David','cast_name_John Leguizamo','cast_name_Woody Harrelson','cast_name_Bill Murray','cast_name_Dennis Quaid','cast_name_James Franco','cast_name_Dustin Hoffman','genders_0_cast','genders_1_cast',
'genders_2_cast','cast_gen0_ratio','cast_gen1_ratio','cast_gen2_ratio','num_crew','all_crew','crew_name_Avy Kaufman','crew_name_Steven Spielberg',
'crew_name_Robert Rodriguez','crew_name_Mary Vernieu','crew_name_Deborah Aquila','crew_name_Bob Weinstein','crew_name_Harvey Weinstein','crew_name_Hans Zimmer','crew_name_Tricia Wood','crew_name_James Newton Howard',
'crew_name_James Horner','crew_name_Luc Besson','crew_name_Francine Maisler','crew_name_Kerry Barden','crew_name_Jerry Goldsmith','all_department','crew_department_Production','crew_department_Sound',
'crew_department_Art','crew_department_Crew','crew_department_Writing','crew_department_Costume & Make-Up','crew_department_Camera','crew_department_Directing','crew_department_Editing','crew_department_Visual Effects','crew_department_Lighting','crew_department_Actors','crew_job_Producer','crew_job_Executive Producer','crew_job_Director','crew_job_Screenplay','crew_job_Editor','crew_job_Casting','crew_job_Director of Photography','crew_job_Original Music Composer','crew_job_Art Direction','crew_job_Production Design',
'genders_0_crew','genders_1_crew','genders_2_crew','crew_gen0_ratio','crew_gen1_ratio','crew_gen2_ratio',
'crew_Production_Avy Kaufman','crew_Production_Mary Vernieu','crew_Production_Deborah Aquila','crew_Production_Bob Weinstein','crew_Production_Harvey Weinstein','crew_Production_Tricia Wood','crew_Production_Francine Maisler','crew_Production_Kerry Barden','crew_Production_Billy Hopkins','crew_Production_Steven Spielberg','crew_Production_Suzanne Smith',
'crew_Production_Arnon Milchan','crew_Production_Scott Rudin','crew_Production_John Papsidera','crew_Production_Tim Bevan','crew_Sound_James Newton Howard','crew_Sound_Hans Zimmer','crew_Sound_James Horner','crew_Sound_Jerry Goldsmith','crew_Sound_John Williams',
'crew_Sound_Alan Silvestri','crew_Sound_Danny Elfman',"crew_Sound_Dan O'Connell",'crew_Sound_Mark Isham','crew_Sound_John Debney','crew_Sound_Marco Beltrami',
'crew_Sound_Kevin Kaska','crew_Sound_Christophe Beck','crew_Sound_Graeme Revell','crew_Sound_Carter Burwell','crew_Art_Helen Jarvis','crew_Art_Ray Fisher','crew_Art_Rosemary Brandenburg',
'crew_Art_Cedric Gibbons','crew_Art_Walter M. Scott','crew_Art_Nancy Haigh','crew_Art_Robert Gould','crew_Art_J. Michael Riva','crew_Art_Maher Ahmad','crew_Art_Henry Bumstead','crew_Art_Leslie A. Pope',
'crew_Art_Gene Serdena','crew_Art_Jann Engel','crew_Art_David F. Klassen','crew_Art_Cindy Carr','crew_Crew_J.J. Makaro','crew_Crew_Brian N. Bentley',
'crew_Crew_Brian Avery','crew_Crew_James Bamford','crew_Crew_Mark Edward Wright','crew_Crew_Karin Silvestri','crew_Crew_Gregory Nicotero','crew_Crew_G.A. Aguilar',
'crew_Crew_Doug Coleman','crew_Crew_Sean Button',"crew_Crew_Chris O'Connell",'crew_Crew_Tim Monich','crew_Crew_Denny Caira',
'crew_Crew_Susan Hegarty','crew_Crew_Michael Queen','crew_Writing_Luc Besson','crew_Writing_Stephen King','crew_Writing_Woodyallen','crew_Writing_John Hughes',
'crew_Writing_Ian Fleming','crew_Writing_Robert Mark Kamen','crew_Writing_Sylvester Stallone','crew_Writing_David Koepp','crew_Writing_Terry Rossio',
'crew_Writing_George Lucas','crew_Writing_Stan Lee','crew_Writing_Akiva Goldsman','crew_Writing_Brian Helgeland','crew_Writing_Ted Elliott','crew_Writing_William Goldman','crew_Costume & Make-Up_Ve Neill',
'crew_Costume & Make-Up_Bill Corso','crew_Costume & Make-Up_Colleen Atwood','crew_Costume & Make-Up_Camille Friend','crew_Costume & Make-Up_Edith Head','crew_Costume & Make-Up_Louise Frogley','crew_Costume & Make-Up_Ellen Mirojnick',
'crew_Costume & Make-Up_Mary Zophres','crew_Costume & Make-Up_Edouard F. Henriques','crew_Costume & Make-Up_Jean Ann Black','crew_Costume & Make-Up_Marlene Stewart','crew_Costume & Make-Up_Ann Roth','crew_Costume & Make-Up_Deborah La Mia Denaver',
'crew_Costume & Make-Up_Alex Rouse','crew_Costume & Make-Up_Shay Cunliffe','crew_Camera_Hans Bjerno','crew_Camera_Roger Deakins','crew_Camera_Dean Semler',
'crew_Camera_David B. Nowell','crew_Camera_Mark Irwin','crew_Camera_John Marzano','crew_Camera_Matthew F. Leonetti','crew_Camera_Dean Cundey',
'crew_Camera_Frank Masi','crew_Camera_Oliver Wood','crew_Camera_Robert Elswit','crew_Camera_Pete Romano','crew_Camera_Merrick Morton',
'crew_Camera_Robert Richardson','crew_Camera_Philippe Rousselot','crew_Directing_Steven Spielberg','crew_Directing_Clint Eastwood','crew_Directing_Woodyallen',
'crew_Directing_Ridley Scott','crew_Directing_Karen Golden','crew_Directing_Alfred Hitchcock','crew_Directing_Kerry Lyn McKissick','crew_Directing_Ron Howard','crew_Directing_Dianne Dreyer','crew_Directing_Wilma Garscadden-Gahret',
'crew_Directing_Martin Scorsese','crew_Directing_Brian De Palma','crew_Directing_Ana Maria Quintana','crew_Directing_Dug Rotstein',
'crew_Directing_Tim Burton','crew_Editing_Michael Kahn','crew_Editing_Chris Lebenzon','crew_Editing_Jim Passon',
'crew_Editing_Gary Burritt','crew_Editing_Dale E. Grahn','crew_Editing_Joel Cox','crew_Editing_Mark Goldblatt',
'crew_Editing_Conrad Buff IV','crew_Editing_John C. Stuver','crew_Editing_Pietro Scalia','crew_Editing_Paul Hirsch',
'crew_Editing_Don Zimmerman','crew_Editing_Robert Troy','crew_Editing_Steven Rosenblum','crew_Editing_Dennis McNeill',
'crew_Visual Effects_Dottie Starling','crew_Visual Effects_Phil Tippett','crew_Visual Effects_James Baker','crew_Visual Effects_Hugo Dominguez',
'crew_Visual Effects_Larry White','crew_Visual Effects_Ray McIntyre Jr.','crew_Visual Effects_James Baxter','crew_Visual Effects_Aaron Williams',"crew_Visual Effects_Julie D'Antoni",'crew_Visual Effects_Frank Thomas','crew_Visual Effects_Milt Kahl','crew_Visual Effects_Peter Chiang','crew_Visual Effects_Chuck Duke','crew_Visual Effects_Dave Kupczyk','crew_Visual Effects_Craig Barron','crew_Lighting_Justin Hammond','crew_Lighting_Howard R. Campbell',
'crew_Lighting_Arun Ram-Mohan','crew_Lighting_Chuck Finch','crew_Lighting_Russell Engels','crew_Lighting_Frank Dorowsky',
'crew_Lighting_Bob E. Krattiger','crew_Lighting_Ian Kincaid','crew_Lighting_Thomas Neivelt','crew_Lighting_Dietmar Haupt','crew_Lighting_James J. Gilson',
'crew_Lighting_Dan Cornwall','crew_Lighting_Andy Ryan','crew_Lighting_Lee Walters','crew_Lighting_Jay Kemp','crew_Actors_Francois Grobbelaar',
"crew_Actors_Mick 'Stuntie' Milligan",'crew_Actors_Sol Gorss','crew_Actors_Mark De Alessandro','crew_Actors_Leigh Walsh',
'crew_Producer_Joel Silver','crew_Producer_Brian Grazer','crew_Producer_Scott Rudin','crew_Producer_Neal H. Moritz',
'crew_Producer_Tim Bevan','crew_Producer_Eric Fellner','crew_Producer_Jerry Bruckheimer','crew_Producer_Arnon Milchan',
'crew_Producer_Gary Lucchesi','crew_Producer_John Davis','crew_Producer_Jason Blum','crew_Producer_Tom Rosenberg','crew_Producer_Kathleen Kennedy',
'crew_Producer_Luc Besson','crew_Producer_Steven Spielberg','crew_Executive Producer_Bob Weinstein','crew_Executive Producer_Harvey Weinstein','crew_Executive Producer_Bruce Berman',
'crew_Executive Producer_Steven Spielberg','crew_Executive Producer_Toby Emmerich','crew_Executive Producer_Stan Lee','crew_Executive Producer_Ryan Kavanaugh','crew_Executive Producer_Ben Waisbren','crew_Executive Producer_Michael Paseornek','crew_Executive Producer_Thomas Tull','crew_Executive Producer_Arnon Milchan','crew_Executive Producer_Nathan Kahane','crew_Executive Producer_John Lasseter','crew_Executive Producer_Tessa Ross',
'crew_Executive Producer_Gary Barber','crew_Director_Steven Spielberg','crew_Director_Clint Eastwood','crew_Director_Woodyallen','crew_Director_Ridley Scott',
'crew_Director_Alfred Hitchcock','crew_Director_Ron Howard','crew_Director_Brian De Palma','crew_Director_Martin Scorsese','crew_Director_Tim Burton',
'crew_Director_Blake Edwards','crew_Director_Joel Schumacher','crew_Director_Oliver Stone','crew_Director_Robert Zemeckis','crew_Director_Steven Soderbergh',
'crew_Director_Wes Craven','crew_Screenplay_Sylvester Stallone','crew_Screenplay_Luc Besson','crew_Screenplay_John Hughes','crew_Screenplay_Akiva Goldsman','crew_Screenplay_David Koepp','crew_Screenplay_William Goldman','crew_Screenplay_Robert Mark Kamen','crew_Screenplay_Oliver Stone',
'crew_Screenplay_Woodyallen','crew_Screenplay_Richard Maibaum','crew_Screenplay_John Logan','crew_Screenplay_Terry Rossio','crew_Screenplay_Harold Ramis',
'crew_Screenplay_Brian Helgeland','crew_Screenplay_Ted Elliott','crew_Editor_Michael Kahn','crew_Editor_Chris Lebenzon','crew_Editor_Joel Cox',
'crew_Editor_Mark Goldblatt','crew_Editor_Conrad Buff IV','crew_Editor_Pietro Scalia','crew_Editor_Paul Hirsch','crew_Editor_Don Zimmerman',
'crew_Editor_Christian Wagner','crew_Editor_Anne V. Coates','crew_Editor_William Goldenberg','crew_Editor_Michael Tronick','crew_Editor_Daniel P. Hanley',
'crew_Editor_Paul Rubell','crew_Editor_Stephen Mirrione','crew_Casting_Avy Kaufman','crew_Casting_Mary Vernieu','crew_Casting_Deborah Aquila',
'crew_Casting_Tricia Wood','crew_Casting_Kerry Barden','crew_Casting_Francine Maisler','crew_Casting_Billy Hopkins','crew_Casting_Suzanne Smith',
'crew_Casting_John Papsidera','crew_Casting_Denise Chamian','crew_Casting_Jane Jenkins','crew_Casting_Janet Hirshenson','crew_Casting_Mike Fenton',
'crew_Casting_Mindy Marin','crew_Casting_Sarah Finn','crew_Director of Photography_Dean Semler','crew_Director of Photography_Roger Deakins','crew_Director of Photography_Mark Irwin',
'crew_Director of Photography_Matthew F. Leonetti','crew_Director of Photography_Dean Cundey','crew_Director of Photography_Oliver Wood','crew_Director of Photography_Robert Elswit','crew_Director of Photography_Robert Richardson',
'crew_Director of Photography_Philippe Rousselot','crew_Director of Photography_Dante Spinotti','crew_Director of Photography_Julio Macat','crew_Director of Photography_Dariusz Wolski','crew_Director of Photography_Don Burgess',
'crew_Director of Photography_Janusz Kami≈Ñski','crew_Director of Photography_Peter Deming','crew_Original Music Composer_James Newton Howard','crew_Original Music Composer_James Horner','crew_Original Music Composer_Hans Zimmer',
'crew_Original Music Composer_Jerry Goldsmith','crew_Original Music Composer_John Williams','crew_Original Music Composer_Danny Elfman','crew_Original Music Composer_Christophe Beck','crew_Original Music Composer_Alan Silvestri',
'crew_Original Music Composer_John Powell','crew_Original Music Composer_Marco Beltrami','crew_Original Music Composer_Howard Shore','crew_Original Music Composer_Graeme Revell','crew_Original Music Composer_John Debney',
'crew_Original Music Composer_Carter Burwell','crew_Original Music Composer_Mark Isham','crew_Art Direction_Cedric Gibbons','crew_Art Direction_Hal Pereira','crew_Art Direction_Helen Jarvis',
'crew_Art Direction_Lyle R. Wheeler','crew_Art Direction_David Lazan','crew_Art Direction_Andrew Max Cahn','crew_Art Direction_Jack Martin Smith','crew_Art Direction_Robert Cowper',
'crew_Art Direction_Stuart Rose','crew_Art Direction_David F. Klassen','crew_Art Direction_Dan Webster','crew_Art Direction_Steven Lawrence','crew_Art Direction_Jesse Rosenthal',
'crew_Art Direction_Richard L. Johnson','crew_Art Direction_Kevin Constant','crew_Production Design_J. Michael Riva','crew_Production Design_Jon Hutman','crew_Production Design_Carol Spier',
'crew_Production Design_Ida Random','crew_Production Design_Dennis Gassner','crew_Production Design_Perry Andelin Blake','crew_Production Design_David Gropman',
'crew_Production Design_Mark Friedberg','crew_Production Design_Rick Carter','crew_Production Design_Stuart Craig','crew_Production Design_Jim Clay',
'crew_Production Design_Kristi Zea','crew_Production Design_David Wasco','crew_Production Design_Wynn Thomas','crew_Production Design_Dante Ferretti]]


# In[125]:


df


# In[126]:


df_features.index = df.index

df_use.index = df.index
df_use2.index = df.index


# In[127]:


df_use4 = df[add_cols]


# In[128]:


df_input = pd.concat([df_use, df_use2, df_use3, df_use4, df_features], axis=1) # .drop("belongs_to_collection", axis=1)


# In[129]:


#Tfid_tagline.index = df_use.index
#df_use_Tfid = Tfid_tagline.loc[:, Tfid_tagline[:3000].nunique()>1]
#df_use_Tfid.shape


# In[130]:


# 全て繋げた特徴量
df_input = pd.concat([df_input, df_tagline, df_overview, df_keyword_w2v, df_castname, df_crewname], axis=1)


# In[131]:


# 欠測ナシを確認
df_input.isnull().sum().sum()


# In[132]:


#cols = df_input.loc[:, df_input.isnull().sum()>0].columns
#df_input.loc[:, cols] = df_input[cols].fillna(df_input[cols].mean())


# In[133]:


# 保存
import pickle
with open('df_input.pkl', 'wb') as f:
      pickle.dump(df_input , f)


# In[134]:


df["ln_revenue"] = np.log(df["revenue"]+1)


# In[135]:


# 数値化できい列を確認
no_numeric = df_input.apply(lambda s:pd.to_numeric(s, errors='coerce')).isnull().all()
no_numeric[no_numeric]


# In[136]:


X_all = df_input  # .drop(["collection_av_logrevenue"], axis=1)
y_all = df["ln_revenue"]
y_all.index = X_all.index


# In[137]:


[ c for c in X_all.columns if "revenue" in str(c)]


# In[138]:


# 標準化
# X_train_all_mean = X_all[:3000].mean()
# X_train_all_std  = X_all[:3000].std()
# X_all = (X_all-X_train_all_mean)/X_train_all_std


# In[139]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error 
from sklearn.preprocessing import StandardScaler


# In[140]:


train_X, val_X, train_y, val_y = train_test_split(X_all[:train.index[-1]], 
                                                  y_all[:train.index[-1]], 
                                                  test_size=0.25, random_state=1)


# In[141]:


from sklearn.ensemble import RandomForestRegressor


# In[142]:


clf2 = RandomForestRegressor(n_jobs=3, random_state=1)  # max_depth=, min_samples_split=, 
clf2.fit(train_X, train_y)


# In[143]:


val_pred = clf2.predict(val_X)
print("RMSLE score for validation data")
np.sqrt(mean_squared_error(val_pred, val_y))


# In[144]:


plt.scatter(np.exp(val_pred)+1, np.exp(val_y)+1, s=3)
plt.xlabel("prediction")
plt.ylabel("true revenue")
plt.xscale("log")
plt.yscale("log")


# In[145]:


clf2 = RandomForestRegressor(n_jobs=3, random_state=1, n_estimators=500)  # 
clf2.fit(X_all[:train.index[-1]], y_all[:train.index[-1]])


# In[146]:


df_importance = pd.DataFrame([clf2.feature_importances_], columns=train_X.columns, index=["importance"]).T
df_importance.sort_values("importance", ascending=False).head(20)


# In[147]:


test_pred = clf2.predict(X_all[3000:])


# In[148]:


test_revenue = np.exp(test_pred)-1


# In[149]:


sample_submission = pd.read_csv('../input/tmdb-box-office-prediction/sample_submission.csv')


# In[150]:


submission_RF = sample_submission.copy()
submission_RF["revenue"] = test_revenue


# In[151]:


submission_RF


# In[152]:


submission_RF.to_csv('submission_RF.csv', index=False)


# In[ ]:




