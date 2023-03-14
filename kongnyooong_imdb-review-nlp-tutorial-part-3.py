#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

from bs4 import BeautifulSoup
import re
import time

from nltk.corpus import stopwords
import nltk.data
from nltk.stem.snowball import SnowballStemmer
from multiprocessing import Pool

import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')




df_train = pd.read_csv("../input/nlp-dataset/labeledTrainData.tsv", header = 0,
                      delimiter = "\t", quoting = 3)
df_test = pd.read_csv("../input/nlp-dataset/testData.tsv", header = 0,
                      delimiter = "\t", quoting = 3)
df_unlabeled = pd.read_csv("../input/nlp-dataset/unlabeledTrainData.tsv", header = 0,
                      delimiter = "\t", quoting = 3)




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




sentences = []
for review in df_train["review"]:
    sentences += KaggleWord2VecUtility.review_to_sentences(
    review, remove_stopwords = False)
    
# KaggleWord2VecUtility을 사용하여 train 데이터를 정제해준다.




for review in df_unlabeled["review"]:
    sentences += KaggleWord2VecUtility.review_to_sentences(
    review, remove_stopwords = False)




num_features = 300 # 문자 벡터 차원 수 (size)
min_word_count = 40 # 최소 문자 수 (min_count)
num_workers = 4 # 병렬 처리 스레드 수 (workers)
context = 10 # 문자열 창 크기 (window)
downsampling = 1e-3 # 문자 빈도 수 Downsample (sample)

model = Word2Vec(sentences, workers = num_workers,
                 size = num_features, min_count = min_word_count,
                 window = context, sample = downsampling)

model




# 숫자로 단어를 표현
# Word2Vec 모델은 어휘의 각 단어에 대한 feature 벡터로 구성되며
# 'syn0'이라는 넘파이 배열로 저장된다.
# syn0의 행 수는 모델 어휘의 단어 수
# 컬럼 수는 part 2에서 설정한 피처 벡터의 크기
type(model.wv.syn0)




model.wv.syn0.shape




model.wv["flower"].shape




model.wv["flower"][:10]




# 단어 벡터에서 k-means를 실행하고 일부 클러스터를 찍어본다.
start = time.time()

# 클러스터의 크기 "k"를 어휘 크기의 1/5 이나 평균 5단어로 설정한다.
word_vectors = model.wv.syn0
num_clusters = word_vectors.shape[0] / 5
num_clusters = int(num_clusters)

# K-means를 정의하고  학습시킨다.
kmeans_clustering = KMeans(n_clusters = num_clusters)
idx = kmeans_clustering.fit_predict(word_vectors)

# 끝난시간에서 시작시간을 빼서 걸린 시간을 구한다.
end = time.time()
elapsed = end - start

print("Time taken for K Means clustering: ", elapsed, "seconds.")




# 각 어휘 단어를 클러스터 번호에 매핑되게 word/index 사전을 만든다.
idx = list(idx)
names = model.wv.index2word
word_centroid_map = {names[i]: idx[i] for i in range(len(names))}
# word_cenetroid_map = dict(zip(model.wv.index2word, idx))

# 첫번째 클러스터의 처음 10개를 출력
for cluster in range(0, 10):
    # 클러스터 번호를 출력
    print("\n Cluster {}".format(cluster))
    
    # 클러스터 번호와 클러스터에 있는 단어를 찍는다.
    words = []
    for i in range(0, len(list(word_centroid_map.values()))):
        if(list(word_centroid_map.values())[i] == cluster):
            words.append(list(word_centroid_map.keys())[i])
    print(words)




"""
판다스로 데이터 프레임 형태의 데이터로 읽어온다.

그리고 이전 튜토리얼에서 했던 것 처럼 clean_train_review와
clean_test_review로 텍스트를 정제한다.
"""




clean_train_reviews = []
for review in df_train["review"]:
    clean_train_reviews.append(
        KaggleWord2VecUtility.review_to_wordlist(review,
                                                remove_stopwords = True))




clean_test_reviews = []
for review in df_test["review"]:
    clean_test_reviews.append(
        KaggleWord2VecUtility.review_to_wordlist(review,
                                                remove_stopwords = True))




# bag of centroids 생성
# 속도를 위해 centroid 학습 세트 bag을 미리 할당한다.
train_centroids = np.zeros((df_train["review"].size, num_clusters),
                          dtype = "float32")

train_centroids[:5]




# centroid는 두 클러스터의 중심점을 정의 한 다음 중심점의 거리를 측정한 것
def create_bag_of_centroids(wordlist, word_centroid_map):
    
    # 클러스터의 수는 word/centroid map에서 가장 높은 클러스트 인덱스와 같다.
    num_centroids = max(word_centroid_map.values())+1
    
    # 속도를 위해 bag of centroids vector를 미리 할당한다.
    bag_of_centroids = np.zeros(num_centroids, dtype = "float32")
    
    # 루프를 돌며 단어가 word_centroid_map에 있다면
    # 해당되는 클러스터의 수를 하나씩 증가시켜 준다.
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
            
    # bag of centroids를 반환한다.
    return bag_of_centroids




# 학습 리뷰를 bags of centroids로 변환한다.
counter = 0
for review in clean_train_reviews:
    train_centroids[counter] = create_bag_of_centroids(review,
                                                      word_centroid_map)
    counter += 1

# 테스트 리뷰도 같은 방법으로 반복해 준다.
test_centroids = np.zeros((df_test["review"].size, num_clusters),
                         dtype = "float32")

counter = 0
for review in clean_test_reviews:
    test_centroids[counter] = create_bag_of_centroids(review,
                                                      word_centroid_map)
    counter += 1




# RandomForest를 사용하여 학습시키고 예측
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 100)

# train 데이터의 레이블을 통해 학습시키고 예측한다.
get_ipython().run_line_magic('time', 'rf = rf.fit(train_centroids, df_train["sentiment"])')




from sklearn.model_selection import cross_val_score
get_ipython().run_line_magic('time', 'score = np.mean(cross_val_score(rf, train_centroids,                                df_train["sentiment"],                                      cv = 10,                                      scoring = "roc_auc"))')




get_ipython().run_line_magic('time', 'result = rf.predict(test_centroids)')




score




output = pd.DataFrame(data = {"id": df_test["id"], "sentiment":result})
output.to_csv("./submit_BagofCentroids_{:.3f}.csv".format(score),index = False,quoting = 3)

