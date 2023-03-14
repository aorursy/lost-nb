#!/usr/bin/env python
# coding: utf-8



import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt




treinoOriginal = pd.read_csv("../input/teste-com-knn/train.csv",
            sep=r'\s*,\s*',
            engine='python',
            na_values="NaN")




treinoOriginal




nomes = ['escolari', 'meaneduc', 'cielorazo','eviv3', 'pisomoscer','etecho3']




medias = [treinoOriginal[x].mean(skipna = True) for x in nomes]




values = {key: value for (key, value) in zip(nomes, medias)}




values




n_treino = treinoOriginal.fillna(value = values)




n_treino




n_treino['Target'].value_counts().plot(kind='pie')




treinoOriginal['escolari'].value_counts(normalize=True).plot(kind='bar')




treinoOriginal['cielorazo'].value_counts(normalize=True).plot(kind='bar')




treinoOriginal['eviv3'].value_counts(normalize=True).plot(kind='bar')




treinoOriginal['pisomoscer'].value_counts(normalize=True).plot(kind='bar')




treinoOriginal['etecho3'].value_counts(normalize=True).plot(kind='bar')




Rodando um loop de kNN para tentar descobrir um índice ótimo. A escolha do índice foi baseada no valor
obtido pela média dos cross_val_scores.
Note que guardamos os resultados para cada indice na lista performance_neighbors, caso seja
interessante verificar seu valor depois.




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




knn = KNeighborsClassifier(n_neighbors=indice)
knn.fit(Xtreino, Ytreino)




testesOriginal = pd.read_csv("../input/teste-com-knn/test.csv",
                            sep=r'\s*,\s*',
                            engine='python',
                            na_values="?")




mediasTestes = [testesOriginal[x].mean(skipna = True) for x in nomes]
valuesTestes = {key: value for (key, value) in zip(nomes, mediasTestes)}




valuesTestes




n_testes = testesOriginal.fillna(value = valuesTestes)




YtestPred = knn.predict(n_testes[['escolari','meaneduc', 'cielorazo','eviv3','pisomoscer','etecho3']])




YtestPred




Transformando para a formatação correta e exportando




ids = testesOriginal.iloc[:,0].values
ids = ids.ravel()
dataset = pd.DataFrame({'Id':ids[:],'income':YtestPred[:]})
dataset.to_csv("submition.csv", index = False)

