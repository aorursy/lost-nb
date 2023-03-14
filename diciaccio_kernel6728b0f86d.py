#!/usr/bin/env python
# coding: utf-8



from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn import datasets, metrics, tree
from sklearn.ensemble import  RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')




# Read data




#




#




# individuiamo la riga in  base al valore di ID





#

















clf = 





# calcolo dell'indice AUC e dell'errore di classificazione sul training e sul test
from sklearn.metrics import roc_auc_score




# Calcolo matrice di confusione e gli altri indici sul validation




# costruiamo la matrice di confusione con una soglia diversa da 0.5






Y_test = clf.predict_proba(X_test)
soluz = pd.DataFrame({'ID':XT[:,0].astype(int),'TARGET':Y_test[:,1]})
soluz.to_csv('fine.csv', sep=",",index=False)

