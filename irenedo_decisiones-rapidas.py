#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import gensim, logging
from gensim.models import word2vec

stops = set(stopwords.words("english"))
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
pal = sns.color_palette()
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.




df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_part = df_train[0:2500]

len(df_train)









import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import gensim, logging
from gensim.models import Word2Vec
from nltk.corpus import brown, movie_reviews, treebank
from gensim import  corpora , models, similarities

#b = Word2Vec(brown.sents())
#mr = Word2Vec(movie_reviews.sents())
#t = Word2Vec(treebank.sents())



q1 = df_part['question1'].values.tolist()
q2 = df_part['question2'].values.tolist()
corpus = q1 + q2

print(corpus[2])

tok_corpus = [word_tokenize(sent) for sent in corpus] #decode('utf-8')
#model = gensim.models.Word2Vec(tok_corpus, min_count=32, size=32)
#model.most_similar(tok_corpus[376][2])
#model.similarity(tok_corpus[765][3],tok_corpus[298][3])
#print(tok_corpus[33][3])

"""
#tokens1 = word_tokenize(q1[2])
#tokens2 = word_tokenize(q2[2])
#filtered_tokens1 = [token.lower() for token in tokens1 if token.isalnum()]#para limpiar las frases(minusculas y alfanumeric)
#filtered_tokens2 = [token.lower() for token in tokens2 if token.isalnum()]#para limpiar las frases(minusculas y alfanumeric)  
#clean_tokens1 = [token for token in filtered_tokens1 if token not in stops]
#clean_tokens2 = [token for token in filtered_tokens2 if token not in stops]
#c = model# + b

for i in clean_tokens1:
    w = c.most_similar(i, topn=2)
    print(i)
    print(w[0][0])
    print(w[1][0])
#corpus = clean_tokens1 + clean_tokens2
#model = gensim.models.Word2Vec(corpus)
print(clean_tokens1)

"""




"""
path = r'C:\Users\Inaki\Documents\TFG\corpus_word2vec/mymodel'
model = Word2Vec.load(path)

print('house') in model
print (model['house'])
"""




print(b)




#model = gensim.models.Word2Vec.load('../input/train.csv')


for i in clean_tokens1:
    w = b.most_similar(i, topn=2)
    print(i)
    print(w[0][0])
    print(w[1][0])
    
#w = word_tokenize(df_part['question1'])




import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import gensim, logging
from gensim.models import word2vec

q1 = df_part['question1'].values.tolist()
q2 = df_part['question2'].values.tolist()
tokens1 = word_tokenize(q1[1])
tokens2 = word_tokenize(q2[2])

filtered_tokens1 = [token.lower() for token in tokens1 if token.isalnum()]#para limpiar las frases(minusculas y alfanumeric)
filtered_tokens2 = [token.lower() for token in tokens2 if token.isalnum()]#para limpiar las frases(minusculas y alfanumeric)  
clean_tokens1 = [token for token in filtered_tokens1 if token not in stops]
clean_tokens2 = [token for token in filtered_tokens2 if token not in stops]
    
corpus = clean_tokens1 + clean_tokens2
model = gensim.models.Word2Vec(corpus)

print(clean_tokens1[0])
model.most_similar([clean_tokens1[0]])




acierto2 =  {} #tabla para porcentajes de aciertos en la estimación
#ratiow2 = same_word_ratio_2(df_train)
 
counter = 0
for index in range(len(df_train)):
    if (df_train['is_duplicate'][index] == 1):
        counter = counter + 1
        
porcentaje = (counter*100)/len(df_train)
print(porcentaje)




train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
test_qs = pd.Series(df_test['question1'].tolist() + df_test['question2'].tolist()).astype(str)
q1 = pd.Series(df_train['question1'][0]) #para sacar la primera pregunta de question1
print(q1)
dist_train = train_qs.apply(len)
dist_test = test_qs.apply(len)
#plt.figure(figsize=(15, 10))
#plt.hist(dist_train, bins=200, range=[0, 200], color=pal[2], normed=True, label='train')
#plt.hist(dist_test, bins=200, range=[0, 200], color=pal[1], normed=True, alpha=0.5, label='test')
#plt.title('Normalised histogram of character count in questions', fontsize=15)
#plt.legend()
#plt.xlabel('Number of characters', fontsize=15)
#plt.ylabel('Probability', fontsize=15)

#print('mean-train {:.2f} std-train {:.2f} mean-test {:.2f} std-test {:.2f} max-train {:.2f} max-test {:.2f}'.format(dist_train.mean(), 
#                          dist_train.std(), dist_test.mean(), dist_test.std(), dist_train.max(), dist_test.max()))




from nltk.corpus import stopwords

stops = set(stopwords.words("english"))

def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R
#print(stops)




train_word_match = df_train.apply(word_match_share, axis=1, raw=True)
decision = {} #tabla con los datos estimados, duplicados o no.
acierto =  {}#tabla para porcentajes de aciertos en la estimación
duplicados = df_train['is_duplicate'] #tabla con los datos de pares duplicados
len(df_train)
#print(duplicados)




for index in range(len(train_word_match)):
    if (train_word_match[index] > 0.35):
        decision[index] = 1
    else:
        decision[index] = 0
#print (decision) 
counter = 0
for index in range(len(train_word_match)):
    if (decision[index] == duplicados[index]):
        acierto[index]=1
        counter = counter +1
    else:
        acierto[index]=0
        
porcentaje = (counter*100)/404290
print(porcentaje)
print(len(decision))




from collections import Counter

# If a word appears only once, we ignore it completely (likely a typo)
# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller
def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)

eps = 5000 
words = (" ".join(train_qs)).lower().split()
words_q1 = (" ".join(q1)).lower().split()
for w in range(len(words_q1)):
    print(words_q1[w])
 

    
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}#estudiar bien que hace




ratiow3 = []

q1 = (pd.Series(df_part['question1'][105780]).astype(str))[0]
q2 = (pd.Series(df_part['question2'][105780]).astype(str))[0] 
if q2 == [] : 
    q2 =""
if q1 == []:
    q1 ="" 
w1 = word_tokenize(q1) 
w2 = word_tokenize(q2) 

filtered_tokens1 = [token.lower() for token in w1 if token.isalnum()]#para limpiar las frases(minusculas y alfanumeric)
filtered_tokens2 = [token.lower() for token in w2 if token.isalnum()]#para limpiar las frases(minusculas y alfanumeric)

clean_tokens1 = [token for token in filtered_tokens1 if token not in stops]
clean_tokens2 = [token for token in filtered_tokens2 if token not in stops]
print(q1)
print(q2)
print(w1)
print(w2)
print(filtered_tokens1)
print(filtered_tokens2)
print(clean_tokens1)
print(clean_tokens2)




#funciona bien

ratiow3 = []

for i in range(len(df_part)):
    n_words = 0
    q1 = (pd.Series(df_part['question1'][i]).astype(str))[0]
    q2 = (pd.Series(df_part['question1'][i]).astype(str))[0] 
    if q2 == []: 
        q2 =""
    if q1 == []:
        q1 ="" 
    w1 = word_tokenize(q1) 
    w2 = word_tokenize(q2) 
    
    filtered_tokens1 = [token.lower() for token in w1 if token.isalnum()]#para limpiar las frases(minusculas y alfanumeric)
    filtered_tokens2 = [token.lower() for token in w2 if token.isalnum()]#para limpiar las frases(minusculas y alfanumeric)
    
    clean_tokens1 = [token for token in filtered_tokens1 if token not in stops]
    clean_tokens2 = [token for token in filtered_tokens2 if token not in stops] 
    
    media = (len(clean_tokens1) + len(clean_tokens2)) 
    minimo = min(len(clean_tokens1), len(clean_tokens2)) 
    maximo = max(len(clean_tokens1), len(clean_tokens2)) 
    
    for w1 in clean_tokens1: 
        for w2 in clean_tokens2:
            if (w1==w2):
                n_words = n_words + 1 
                #print(n_words)
    if(maximo == 0): 
        maximo = 1
    r = n_words/maximo 
    print(i) 
    ratiow3.append(r)
    




#q1 = pd.Series(df_train['question1'][7]).astype(str)
#q2 = pd.Series(df_train['question2'][7]).astype(str)

#qv = q1.values.tolist()
x = df_part['question1'].values.tolist()
y = df_part['question2'].values.tolist()
print(x[5])

#art = x.decode('utf-8') 
w = word_tokenize(x[5])# w son los tokens de la frase 5
print(w)
filtered_tokens = [token.lower() for token in w if token.isalnum()]#para limpiar las frases(minusculas y alfanumeric)
clean_tokens = [token for token in filtered_tokens if token not in stops] #quitar stopwords
print(filtered_tokens)
print(clean_tokens)
print(clean_tokens[3])

#print(clean_tokens[2])#así se puede recorrer palabra por alabra
#token_corpus = [nltk.word_tokenize(sent.decode('utf-8'))for sent in corpus]
#print(token_corpus)
#print(q1)
#print(q2)
n_words = 0
#for n in range(5):
    #q1 = pd.Series(df_train['question1'][n]).astype(str)
    #q2 = pd.Series(df_train['question2'][n]).astype(str)
    #qv1 = q1.values.tolist()
    #qv2 = q2.values.tolist()
    #for w1 in q1:
      #  for w2 in q2:
            #if (w1==w2):
                #n_words = n_words + 1
#print(n_words)     




print(x)





ratiow2 = []
question1 = df_part['question1'].values.tolist()
question2 = df_part['question2'].values.tolist()

for i in range(len(df_part)):
    n_words = 0
    if(question1[i]==" "): #¿?¿?¿?¿?¿?¿?¿?¿?¿?¿ que hacer?
        question1[i] = 'a a a'#hay preguntas que dan error y hay que intentar cambiarlas para que no den error
    if(question2[i]==" "): # la 105800 por ejemplo falla
        question2[i] = 'a a a'
    q_tokens1 = word_tokenize(question1[i]) #tokenizar frase por frase, las frases con sus tokens separados
    q_tokens2 = word_tokenize(question2[i])
    
    filtered_tokens1 = [token.lower() for token in q_tokens1 if token.isalnum()]#para limpiar las frases(minusculas y alfanumeric)
    filtered_tokens2 = [token.lower() for token in q_tokens2 if token.isalnum()]
    
    clean_tokens1 = [token for token in filtered_tokens1 if token not in stops] #quitar stopwords
    clean_tokens2 = [token for token in filtered_tokens2 if token not in stops]  
        #meter lo de word2vec--->>> most_similar() y model.wv.simila

    media = (len(clean_tokens1) + len(clean_tokens2))/2
    minimo = min(len(clean_tokens1), len(clean_tokens2))
    maximo = max(len(clean_tokens1), len(clean_tokens2))

    for w1 in clean_tokens1:
        for w2 in clean_tokens2:
            if (w1==w2):
                n_words = n_words + 1
    #print(n_words)
    if(maximo == 0):
        maximo = 1
        
    r = n_words/maximo
   
    ratiow2.append(r)


            




print(clean_tokens1 & clean_tokens2)




decision2 = {} #tabla con los datos estimados, duplicados o no.
duplicados2 = df_part['is_duplicate']
acierto2 =  {} #tabla para porcentajes de aciertos en la estimación
#ratiow2 = same_word_ratio_2(df_train)

for index in range(len(ratiow2)):
    if (ratiow2[index] > 0.85):
        decision2[index] = 1
    else:
        decision2[index] = 0
#print (decision) 
counter = 0
for index in range(len(ratiow2)):
    if (decision2[index] == duplicados2[index]):
        acierto2[index]=1
        counter = counter +1
    else:
        acierto2[index]=0
        
porcentaje = (counter*100)/len(df_part)
print(porcentaje)




print(ratiow2)




stops = set(stopwords.words("english"))
#probar despues con el stemer y el lemmatizador

def same_word_ratio_2(row):
    
    ratiow = []
    question1 = row['question1'].values.tolist()
    question2 = row['question2'].values.tolist()
    for i in range(len(row)):
        n_words = 0     
        q_tokens1 = word_tokenize(question1[i]) #tokenizar frase por frase, las frases con sus tokens separados
        q_tokens2 = word_tokenize(question2[i])
        
        filtered_tokens1 = [token.lower() for token in q_tokens1 if token.isalnum()]#para limpiar las frases(minusculas y alfanumeric)
        filtered_tokens2 = [token.lower() for token in q_tokens2 if token.isalnum()]
        
        clean_tokens1 = [token for token in filtered_tokens1 if token not in stops] #quitar stopwords
        clean_tokens2 = [token for token in filtered_tokens2 if token not in stops]  
            #meter lo de word2vec--->>> most_similar() y model.wv.simila
    
        media = (len(clean_tokens1) + len(clean_tokens2))/2
        minimo = min(len(clean_tokens1), len(clean_tokens2))
        maximo = max(len(clean_tokens1), len(clean_tokens2))
        # return 1.0 * len(clean_tokens1 & w2)/(len(clean_tokens1) + len(w2))
        
        for w1 in clean_tokens1:
            for w2 in clean_tokens2:
                if (w1==w2):
                    n_words = n_words + 1
        #print(n_words)
        if(media == 0):
            r = n_words
        else:
            r = n_words/media
        ratiow.append(r)
    return ratiow




stops = set(stopwords.words("english"))


def same_word_ratio(row):
    
    ratiow = []
    for i in range(len(row)):
        n_words = 0
        q1 = pd.Series(row['question1'][i]).astype(str) #para sacar la primera pregunta de question1
        q2 = pd.Series(row['question2'][i]).astype(str)
        words_q1 = (" ".join(q1)).lower().split()
        words_q2 = (" ".join(q2)).lower().split()
        media = (len(words_q1) + len(words_q2))/2
        minimo = min(len(words_q1), len(words_q2))
        for w1 in words_q1:
            for w2 in words_q2:
                if (w1==w2) and (w1 not in stops):
                    n_words = n_words + 1
        #print(n_words)            
        r = n_words/minimo
        ratiow.append(r)
    return ratiow
    
    




decision2 = {} #tabla con los datos estimados, duplicados o no.

acierto2 =  {} #tabla para porcentajes de aciertos en la estimación
duplicados2 = df_part['is_duplicate'] #tabla con los datos de pares duplicados
#print(duplicados)
ratiow = same_word_ratio(df_part)#llamada a la funcion hecha

for index in range(len(ratiow)):
    if (ratiow[index] > 3):
        decision2[index] = 1
    else:
        decision2[index] = 0
#print (decision) 
counter = 0
for index in range(len(ratiow)):
    if (decision2[index] == duplicados2[index]):
        acierto2[index]=1
        counter = counter +1
    else:
        acierto2[index]=0
        
porcentaje = (counter*100)/len(df_part) 
print(porcentaje)




decision2 = {} #tabla con los datos estimados, duplicados o no.

acierto2 =  {} #tabla para porcentajes de aciertos en la estimación
ratiow2 = same_word_ratio_2(df_train)

for index in range(len(ratiow2)):
    if (ratiow2[index] > 0.25):
        decision2[index] = 1
    else:
        decision2[index] = 0
#print (decision) 
counter = 0
for index in range(len(ratiow2)):
    if (decision2[index] == duplicados2[index]):
        acierto2[index]=1
        counter = counter +1
    else:
        acierto2[index]=0
        
porcentaje = (counter*100)/404290 
print(porcentaje)




def no_detections(row, df_x ): #row: una matriz de decision con unos y ceros estimados
                               #df_x: matriz original con las preguntas
    
    no_detections = {}
    no_detections_q = {}
    no_detections_q1 = {}
    no_detections_q2 = {}
    counter = 0
    
    for index in range(len(row)):
        if ((row[index] == 0) and (df_x['is_duplicate'][index] == 1)):
            no_detections[index] = 1
            #no_detections_q2 = pd.DataFrame({'id': df_x['test_id'][index], 'question1': df_x['question1'][index], 'question2': df_x['question2'][index]})
            no_detections_q1[index] = df_x['question1'][index]
            no_detections_q2[index] = df_x['question2'][index]
            counter = counter + 1
        else:
            no_detections[index] = 0
            
    no_detections_q = pd.DataFrame({'question1': no_detections_q1, 'question2': no_detections_q2})   
    return no_detections_q




def falsa_alarma(row, df_x ): #row: una matriz de decision con unos y ceros estimados
                               #df_x: matriz original con las preguntas
    
    falsa_alarma = {}
    falsa_alarma_q = {}
    falsa_alarma_q1 = {}
    falsa_alarma_q2 = {}
    counter = 0
    
    for index in range(len(row)):
        if ((row[index] == 1) and (df_x['is_duplicate'][index] == 0)):
            falsa_alarma[index] = 1
            #falsa_alarma_q2 = pd.DataFrame({'id': df_x['test_id'][index], 'question1': df_x['question1'][index], 'question2': df_x['question2'][index]})
            falsa_alarma_q1[index] = df_x['question1'][index]
            falsa_alarma_q2[index] = df_x['question2'][index]
            counter = counter + 1
        else:
            falsa_alarma[index]=0
            
    falsa_alarma_q = pd.DataFrame({'question1': falsa_alarma_q1, 'question2': falsa_alarma_q2})
    return falsa_alarma_q




no_detection = no_detections(decision2,df_train)
#print(no_detection)




falsa_alarma = falsa_alarma(decision2, df_train)
print(falsa_alarma)




ratiow_test = same_word_ratio(df_test)




decision2_test = {}


for index in range(len(ratiow_test)):
    if (ratiow_test[index] > 0.25):
        decision2_test[index] = 1
    else:
        decision2_test[index] = 0
len(ratiow_test)
#for w in range(20):
#   print(decision2_test[w])




print('Most common words and weights: \n')
print(sorted(weights.items(), key=lambda x: x[1] if x[1] > 0 else 9999)[:10])
print('\nLeast common words and weights: ')
(sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10])




def tfidf_word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    
    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
    
    R = np.sum(shared_weights) / np.sum(total_weights)
    return R




tfidf_train_word_match = df_train.apply(tfidf_word_match_share, axis=1, raw=True)
decision_tfidf = {} #tabla con los datos estimados, duplicados o no.
acierto_tfidf =  {} #tabla para porcentajes de aciertos en la estimación
duplicados_tfidf = df_train['is_duplicate'] #tabla con los datos de pares duplicados
print(duplicados_tfidf)




for index in range(len(tfidf_train_word_match)):
    if (tfidf_train_word_match[index] > 0.35):
        decision_tfidf[index] = 1
    else:
        decision_tfidf[index] = 0
#print (decision_tfidf) 

counter = 0
for index in range(len(tfidf_train_word_match)):
    if (decision_tfidf[index] == duplicados_tfidf[index]):
        acierto_tfidf[index]=1
        counter = counter +1
    else:
        acierto_tfidf[index]=0
        
porcentaje_tfidf = (counter*100)/404265 
print(porcentaje_tfidf)









word_match_test = df_test.apply(word_match_share, axis=1, raw=True)
decision_test = {} #tabla con los datos estimados, duplicados o no.
#duplicados = df_train['is_duplicate'] #tabla con los datos de pares duplicados
len(df_test)
#print(duplicados)




for index in range(len(word_match_test)):
    if (word_match_test[index] > 0.35):
        decision_test[index] = 1
    else:
        decision_test[index] = 0
#print (decision) 
for w in range(20):
    print(decision_test[w])
len(decision_test)




ratiow_test = same_word_ratio(df_test)










#submission = pd.DataFrame({'test_id': df_test['test_id'],'is_duplicate': ratiow_test})
submission = pd.DataFrame({'test_id': df_test['test_id'], 'is_duplicate': decision2_test})

submission.to_csv("submission.csv", index=False)
#sub = pd.DataFrame()
#sub['test_id'] = df_test['test_id']
#sub['is_duplicate'] = decision_test
#sub.to_csv('simple_counw.csv', index=False)
submission.head()

