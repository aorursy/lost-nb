#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings

warnings.filterwarnings("ignore")


# In[2]:


import pandas as pd

df_train = pd.read_csv("../input/nlp-dataset/labeledTrainData.tsv", header = 0,
                      delimiter = "\t", quoting = 3)
df_test = pd.read_csv("../input/nlp-dataset/testData.tsv", header = 0,
                      delimiter = "\t", quoting = 3)
df_unlabeled = pd.read_csv("../input/nlp-dataset/unlabeledTrainData.tsv", header = 0,
                      delimiter = "\t", quoting = 3)

print(df_train.shape)
print(df_test.shape)
print(df_unlabeled.shape)

print(df_train["review"].size)
print(df_test["review"].size)
print(df_unlabeled["review"].size)


# In[3]:


df_train.head()


# In[4]:


df_test.head()

# 예측해야하는 sentiment 피쳐가 없다.


# In[5]:


import re
import nltk

import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from multiprocessing import Pool

class KaggleWord2VecUtility(object):

    @staticmethod
    def review_to_wordlist(review, remove_stopwords=False):
        # 1. HTML 제거
        review_text = BeautifulSoup(review, "html.parser").get_text()
        # 2. 특수문자를 공백으로 바꿔줌
        review_text = re.sub('[^a-zA-Z]', ' ', review_text)
        # 3. 소문자로 변환 후 나눈다.
        words = review_text.lower().split()
        # 4. 불용어 제거
        if remove_stopwords:
            stops = set(stopwords.words('english'))
            words = [w for w in words if not w in stops]
        # 5. 어간추출
        stemmer = SnowballStemmer('english')
        words = [stemmer.stem(w) for w in words]
        # 6. 리스트 형태로 반환
        return(words)

    @staticmethod
    def review_to_join_words( review, remove_stopwords=False ):
        words = KaggleWord2VecUtility.review_to_wordlist(            review, remove_stopwords=False)
        join_words = ' '.join(words)
        return join_words

    @staticmethod
    def review_to_sentences( review, remove_stopwords=False ):
        # punkt tokenizer를 로드한다.
        """
        이 때, pickle을 사용하는데
        pickle을 통해 값을 저장하면 원래 변수에 연결 된 참조값 역시 저장된다.
        저장된 pickle을 다시 읽으면 변수에 연결되었던
        모든 레퍼런스가 계속 참조 상태를 유지한다.
        """
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        # 1. nltk tokenizer를 사용해서 단어로 토큰화 하고 공백 등을 제거한다.
        raw_sentences = tokenizer.tokenize(review.strip())
        # 2. 각 문장을 순회한다.
        sentences = []
        for raw_sentence in raw_sentences:
            # 비어있다면 skip
            if len(raw_sentence) > 0:
                # 태그제거, 알파벳문자가 아닌 것은 공백으로 치환, 불용어제거
                sentences.append(                    KaggleWord2VecUtility.review_to_wordlist(                    raw_sentence, remove_stopwords))
        return sentences


    # 참고 : https://gist.github.com/yong27/7869662
    # http://www.racketracer.com/2016/07/06/pandas-in-parallel/
    # 속도 개선을 위해 멀티 스레드로 작업하도록
    @staticmethod
    def _apply_df(args):
        df, func, kwargs = args
        return df.apply(func, **kwargs)

    @staticmethod
    def apply_by_multiprocessing(df, func, **kwargs):
        # 키워드 항목 중 workers 파라메터를 꺼냄
        workers = kwargs.pop('workers')
        # 위에서 가져온 workers 수로 프로세스 풀을 정의
        pool = Pool(processes=workers)
        # 실행할 함수와 데이터프레임을 워커의 수 만큼 나눠 작업
        result = pool.map(KaggleWord2VecUtility._apply_df, [(d, func, kwargs)
                for d in np.array_split(df, workers)])
        pool.close()
        # 작업 결과를 합쳐서 반환
        return pd.concat(result)
    
    
# KaggleWord2VecUtility를 class로 생성하여 사용 
# 코드 출처: https://github.com/corazzon/KaggleStruggle/blob/master/word2vec-nlp-tutorial/KaggleWord2VecUtility.py


# In[6]:


KaggleWord2VecUtility.review_to_wordlist(df_train["review"][0])[:10]


# In[7]:


sentences = []
for review in df_train["review"]:
    sentences += KaggleWord2VecUtility.review_to_sentences(
    review, remove_stopwords = False)
    
# KaggleWord2VecUtility을 사용하여 train 데이터를 정제해준다.


# In[8]:


for review in df_unlabeled["review"]:
    sentences += KaggleWord2VecUtility.review_to_sentences(
    review, remove_stopwords = False)
    
# KaggleWord2VecUtility을 사용하여 unlabeled train 데이터를 정제해준다.    


# In[9]:


len(sentences)


# In[10]:


sentences[0][:10]


# In[11]:


import logging
logging.basicConfig(
    format = "%(asctime)s : %(levelname)s : %(message)s",
    level = logging.INFO)


# In[12]:


# 파라미터 값을 지정해준다. 

num_features = 300 # 문자 벡터 차원 수 (size)
min_word_count = 40 # 최소 문자 수 (min_count)
num_workers = 4 # 병렬 처리 스레드 수 (workers)
context = 10 # 문자열 창 크기 (window)
downsampling = 1e-3 # 문자 빈도 수 Downsample (sample)

# 초기화 및 모델 학습
from gensim.models import word2vec

model = word2vec.Word2Vec(sentences,
                         workers = num_workers,
                         size = num_features,
                         min_count = min_word_count,
                         window = context,
                         sample = downsampling)

model


# In[13]:


# 학습이 완료되면 필요없는 메모리를 unload 시킨다.
model.init_sims(replace = True)

model_name = "300features_40minwindows_10text"
model.save(model_name)


# In[14]:


# 유사도가 없는 단어 추출
model.wv.doesnt_match("man woman child kitchen".split())


# In[15]:


model.wv.doesnt_match("france england germany berlin".split())


# In[16]:


# 가장 유사한 단어를 추출
model.wv.most_similar("man")


# In[17]:


model.wv.most_similar("queen")


# In[18]:


model.wv.most_similar("film")


# In[19]:


model.wv.most_similar("happi")


# In[20]:


from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt
import gensim
import gensim.models as g

# 그래프에서 마이너스 폰트 깨지는 문제에 대한 대처
mpl.rcParams["axes.unicode_minus"] = False

model_name = "300features_40minwindows_10text"
model = g.Doc2Vec.load(model_name)

vocab = list(model.wv.vocab)
X = model[vocab]

print(len(X))
print(X[0][:10])
tsne = TSNE(n_components = 2)

# 100개의 단어에 대해서만 시각화
X_tsne = tsne.fit_transform(X[:100,:])


# In[21]:


df = pd.DataFrame(X_tsne, index = vocab[:100], columns = ["x", "y"])
df.shape


# In[22]:


df.head()


# In[23]:


fig = plt.figure()
fig.set_size_inches(40, 20)
ax = fig.add_subplot(1, 1, 1)

ax.scatter(df["x"], df["y"])

for word, pos in df.iterrows():
    ax.annotate(word, pos, fontsize = 30)

plt.show()


# In[24]:


import numpy as np


# def를 이용해서 주어진 문장에서 단어 벡터의 평균을 구하는 함수를 만든다.
def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,),dtype = "float32")
    
    # 속도를 위해 0으로 채운 배열로 초기화 한다.
    nwords = 0.
    # Index2word는 모델의 사전에 있는 단어명을 담은 리스트이다.
    # 속도를 위해 set 형태로 초기화 한다.
    index2word_set = set(model.wv.index2word)
    # 루프를 돌며 모델 사전에 포함이 되는 단어라면 피처에 추가한다.
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model[word])
    # 결과를 단어수로 나누어 평균을 구한다.        
    featureVec = np.divide(featureVec, nwords)
    return featureVec


# In[25]:


def getAvgFeatureVecs(reviews, model, num_features):
    # 리뷰 단어 목록의 각각에 대한 평균 feature 벡터를 계산하고
    # 2d Numpy Array로 반환한다.
    
    # 카운터를 초기화 한다.
    counter = 0.
    # 속도를 위해 2D 넘파이 배열을 미리 할당한다.
    reviewFeatureVecs = np.zeros(
        (len(reviews), num_features), dtype = "float32")
    
    for review in reviews:
        # 매 1000개 리뷰마다 상태를 출력
        if counter%1000. == 0.:
            print("Review %d of %d"%(counter, len(reviews)))
        # 평균 피처 벡터를 만들기 위해 위에서 정의한 함수를 호출한다.    
        reviewFeatureVecs[int(counter)] = makeFeatureVec(review,
                                                         model,
                                                        num_features)
        # 카운터를 증가시킨다.
        counter = counter + 1.
    return reviewFeatureVecs


# In[26]:


# 멀티스레드로 4개의 워커를 사용해 처리한다.

def getCleanReviews(reviews):
    clean_reviews = []
    clean_reviews = KaggleWord2VecUtility.apply_by_multiprocessing(        reviews["review"], KaggleWord2VecUtility.review_to_wordlist,        workers = 4)
    return clean_reviews


# In[27]:


get_ipython().run_line_magic('time', 'trainDataVecs = getAvgFeatureVecs(    getCleanReviews(df_train), model, num_features)')


# In[28]:


get_ipython().run_line_magic('time', 'testDataVecs = getAvgFeatureVecs(    getCleanReviews(df_test), model, num_features)')


# In[29]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators = 100, n_jobs = -1, random_state = 42)


# In[30]:


get_ipython().run_line_magic('time', 'rf.fit(trainDataVecs, df_train["sentiment"])')


# In[31]:


from sklearn.model_selection import cross_val_score
get_ipython().run_line_magic('time', 'score= np.mean(cross_val_score(    rf, trainDataVecs, df_train["sentiment"], cv = 10, scoring = "roc_auc"))')


# In[32]:


score


# In[33]:


result = rf.predict(testDataVecs)


# In[34]:


output = pd.DataFrame(data = {"id": df_test["id"], "sentiment": result})
output.to_csv("./Word2Vec_Tutorial_{:.5f}.csv".format(score),
             index = False, quoting = 3)


# In[35]:


output_sentiment = output["sentiment"].value_counts()
print(output_sentiment[0] - output_sentiment[1])
output_sentiment


# In[ ]:




