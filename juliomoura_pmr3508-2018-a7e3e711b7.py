#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


hdb = pd.read_csv("../input/test.csv")


# In[ ]:


hdb.shape


# In[ ]:


hdb.head()


# In[ ]:


hdbTrain = pd.read_csv("../input/train.csv")


# In[ ]:


hdbTrain.shape


# In[ ]:


hdbTrain.head()


# In[ ]:


#Apenas para visualização da distribuição das classificações na coluna 'target'
import matplotlib.pyplot as plt


# In[ ]:


hdbTrain["Target"].value_counts().plot(kind="bar")


# In[ ]:


hdbTrain_r = hdbTrain.dropna()
hdbTrain_r.shape


# In[ ]:


hdbTrain = hdbTrain.fillna(method = 'ffill')
hdbTrain = hdbTrain.fillna(0)

hdb = hdb.fillna(method = 'ffill')
hdb = hdb.fillna(0)


# In[ ]:


hdbTrain.head()


# In[ ]:


XhdbTrain = hdbTrain[['v2a1', 'hacdor', 'rooms', 'hacapo', 'v14a', 'refrig', 'v18q', 'v18q1', 'r4h1', 'r4h2', 'r4h3',
                      'r4m1', 'r4m2', 'r4m3', 'r4t1', 'r4t2', 'r4t3', 'tamhog', 'tamviv', 'escolari', 'rez_esc', 'hhsize',
                      'paredblolad', 'paredzocalo', 'paredpreb', 'pareddes', 'paredmad', 'paredzinc', 'paredfibras','paredother',
                      'pisomoscer', 'pisocemento', 'pisoother', 'pisonatur', 'pisonotiene', 'pisomadera', 'techozinc', 'techoentrepiso',
                      'techocane', 'cielorazo','abastaguadentro', 'abastaguafuera', 'abastaguano', 'public', 'noelec', 'coopele', 'sanitario1',
                      'sanitario2', 'sanitario3', 'sanitario5', 'sanitario6', 'energcocinar2','energcocinar3', 'energcocinar4', 'elimbasu1', 
                      'elimbasu2','elimbasu3','elimbasu4', 'elimbasu5', 'elimbasu6', 'epared1', 'epared2', 'epared3', 'etecho1','etecho2',
                      'etecho3', 'eviv1', 'eviv2','eviv3', 'dis','male','female', 'estadocivil1', 'estadocivil2','estadocivil3','estadocivil4',
                      'estadocivil5', 'estadocivil6', 'estadocivil7','parentesco1', 'parentesco2','parentesco3','parentesco4','parentesco5',
                      'parentesco6', 'parentesco7', 'parentesco8', 'parentesco9','parentesco10','parentesco11', 'parentesco12',
                      'hogar_nin', 'hogar_adul', 'hogar_mayor','hogar_total', 'meaneduc', 'instlevel1',
                      'instlevel2', 'instlevel3','instlevel4','instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9', 'bedrooms',
                      'overcrowding', 'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5', 'computer', 'television', 'mobilephone',
                      'qmobilephone', 'lugar1', 'lugar2', 'lugar3', 'lugar4', 'lugar5', 'lugar6', 'area1', 'area2', 'age', 'SQBescolari', 'SQBage',
                      'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned', 'agesq']]


# In[ ]:


XhdbTrain.info()


# In[ ]:


YhdbTrain = hdbTrain.Target


# In[ ]:


XhdbTest = hdb[['v2a1', 'hacdor', 'rooms', 'hacapo', 'v14a', 'refrig', 'v18q', 'v18q1', 'r4h1', 'r4h2', 'r4h3',
                      'r4m1', 'r4m2', 'r4m3', 'r4t1', 'r4t2', 'r4t3', 'tamhog', 'tamviv', 'escolari', 'rez_esc', 'hhsize',
                      'paredblolad', 'paredzocalo', 'paredpreb', 'pareddes', 'paredmad', 'paredzinc', 'paredfibras','paredother',
                      'pisomoscer', 'pisocemento', 'pisoother', 'pisonatur', 'pisonotiene', 'pisomadera', 'techozinc', 'techoentrepiso',
                      'techocane', 'cielorazo','abastaguadentro', 'abastaguafuera', 'abastaguano', 'public', 'noelec', 'coopele', 'sanitario1',
                      'sanitario2', 'sanitario3', 'sanitario5', 'sanitario6', 'energcocinar2','energcocinar3', 'energcocinar4', 'elimbasu1', 
                      'elimbasu2','elimbasu3','elimbasu4', 'elimbasu5', 'elimbasu6', 'epared1', 'epared2', 'epared3', 'etecho1','etecho2',
                      'etecho3', 'eviv1', 'eviv2','eviv3', 'dis','male','female', 'estadocivil1', 'estadocivil2','estadocivil3','estadocivil4',
                      'estadocivil5', 'estadocivil6', 'estadocivil7','parentesco1', 'parentesco2','parentesco3','parentesco4','parentesco5',
                      'parentesco6', 'parentesco7', 'parentesco8', 'parentesco9','parentesco10','parentesco11', 'parentesco12',
                      'hogar_nin', 'hogar_adul', 'hogar_mayor','hogar_total', 'meaneduc', 'instlevel1',
                      'instlevel2', 'instlevel3','instlevel4','instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9', 'bedrooms',
                      'overcrowding', 'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5', 'computer', 'television', 'mobilephone',
                      'qmobilephone', 'lugar1', 'lugar2', 'lugar3', 'lugar4', 'lugar5', 'lugar6', 'area1', 'area2', 'age', 'SQBescolari', 'SQBage',
                      'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned', 'agesq']]


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


from sklearn.model_selection import cross_val_score


# In[ ]:


mediaMaiorK = 0.0
for k in range(1,51):
    knn = KNeighborsClassifier(n_neighbors = k)
    scores = cross_val_score(knn, XhdbTrain, YhdbTrain, cv=10)
    
    if sum(scores)/10 > mediaMaiorK:
        mediaMaiorK = sum(scores)/10
        melhorK = k
        
melhorK


# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 48)


# In[ ]:


knn.fit(XhdbTrain, YhdbTrain)


# In[ ]:


scores = cross_val_score(knn, XhdbTrain, YhdbTrain, cv=10)


# In[ ]:


scores


# In[ ]:


YtestPred = knn.predict(XhdbTrain)


# In[ ]:


YtestPred


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy_score(YhdbTrain,YtestPred)


# In[ ]:


Ysubmission = knn.predict(XhdbTest)
Ysubmission


# In[ ]:


Submissão dos resultados:


# In[ ]:


submission = pd.DataFrame()
submission['Target'] = np.array(Ysubmission).mean(axis=0).round().astype(int)
submission.to_csv('submission.csv', index = False)

