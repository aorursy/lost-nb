#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# In[ ]:


treinoOriginal = pd.read_csv("../input/teste-com-knn/train.csv",
            sep=r'\s*,\s*',
            engine='python',
            na_values="NaN")


# In[ ]:


treinoOriginal


# In[ ]:


nomes = ['escolari', 'meaneduc', 'cielorazo','eviv3', 'pisomoscer','etecho3']


# In[ ]:


medias = [treinoOriginal[x].mean(skipna = True) for x in nomes]


# In[ ]:


values = {key: value for (key, value) in zip(nomes, medias)}


# In[ ]:


values


# In[ ]:


n_treino = treinoOriginal.fillna(value = values)


# In[ ]:


n_treino


# In[ ]:


n_treino['Target'].value_counts().plot(kind='pie')


# In[ ]:


treinoOriginal['escolari'].value_counts(normalize=True).plot(kind='bar')


# In[ ]:


treinoOriginal['cielorazo'].value_counts(normalize=True).plot(kind='bar')


# In[ ]:


treinoOriginal['eviv3'].value_counts(normalize=True).plot(kind='bar')


# In[ ]:


treinoOriginal['pisomoscer'].value_counts(normalize=True).plot(kind='bar')


# In[ ]:


treinoOriginal['etecho3'].value_counts(normalize=True).plot(kind='bar')


# In[ ]:


Rodando um loop de kNN para tentar descobrir um índice ótimo. A escolha do índice foi baseada no valor
obtido pela média dos cross_val_scores.
Note que guardamos os resultados para cada indice na lista performance_neighbors, caso seja
interessante verificar seu valor depois.


# In[ ]:


melhor = 0
indice = 0
performance_neighbors = []
Xtreino = n_treino[['escolari','meaneduc', 'cielorazo','eviv3','pisomoscer','etecho3']]
Ytreino = n_treino.Target
for i in range (1, 60):
    knn = KNeighborsClassifier(n_neighbors = i)
    scores = cross_val_score(knn, Xtreino, Ytreino, cv=10)
    performance_neighbors.append(sum(scores)/20)
    if sum(scores)/20 > melhor:
        melhor = sum(scores)/20
        indice = i


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=indice)
knn.fit(Xtreino, Ytreino)


# In[ ]:


testesOriginal = pd.read_csv("../input/teste-com-knn/test.csv",
                            sep=r'\s*,\s*',
                            engine='python',
                            na_values="?")


# In[ ]:


mediasTestes = [testesOriginal[x].mean(skipna = True) for x in nomes]
valuesTestes = {key: value for (key, value) in zip(nomes, mediasTestes)}


# In[ ]:


valuesTestes


# In[ ]:


n_testes = testesOriginal.fillna(value = valuesTestes)


# In[ ]:


YtestPred = knn.predict(n_testes[['escolari','meaneduc', 'cielorazo','eviv3','pisomoscer','etecho3']])


# In[ ]:


YtestPred


# In[ ]:


Transformando para a formatação correta e exportando


# In[ ]:


ids = testesOriginal.iloc[:,0].values
ids = ids.ravel()
dataset = pd.DataFrame({'Id':ids[:],'income':YtestPred[:]})
dataset.to_csv("submition.csv", index = False)

