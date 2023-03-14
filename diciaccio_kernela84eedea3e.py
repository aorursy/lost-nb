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




import os
print(os.listdir("../input"))




Dtrain = pd.read_csv('../input/train1.csv',header=0, sep=",",index_col=0)
Dtest = pd.read_csv('../input/test1.csv',header=0, sep=",",index_col=False)
print(Dtrain.shape,Dtest.shape)
Dtest.head()




DD=Dtrain
k = 0
for j in DD.columns:
    k = k+1
    print("%4d" % k,"%28s" % j,"\t%12.1f" % DD[j].min()," %12.1f" % DD[j].max(), 
                          " Range: %12.1f" % (DD[j].max()-DD[j].min())," STD: %12.2f" % DD[j].std())




if Dtrain.isnull().values.any():
    print(Dtrain.isnull().sum())




# individuiamo la riga in  base al valore di ID
Z = Dtrain['TARGET'].isna()
a = Dtrain[Z]
print(a['TARGET'])
print(a.index.values)
# eliminiamo la riga individuata
Dtrain = Dtrain.drop(index=a.index.values)
# controlliamo che non ci siano più valori missing
print("Are there missing values?",Dtrain.isnull().values.any())




Dtest.isnull().values.any()




XX = Dtrain.values
XT = Dtest.values
xtrain = XX[:,:-1]
ytrain = XX[:,-1]
# attenzione: XT contiene anche l'indice che ci servirà per scrivere il file
# con la soluzione finale
X_test = XT[:,1:]
print(xtrain.shape,ytrain.shape,X_test.shape)









X_train, X_val, y_train, y_val = train_test_split(xtrain,ytrain,test_size=0.25, random_state=33)


print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)





clf =  RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=10, min_samples_split=2, 
                              min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=10, 
                              max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, 
                              bootstrap=True, oob_score=False, n_jobs=1, random_state=33, verbose=0, 
                              warm_start=False, class_weight=None)
clf.fit(X_train,y_train)




# calcolo dell'indice AUC e dell'errore di classificazione sul training e sul test
from sklearn.metrics import roc_auc_score
score = clf.score(X_train,y_train)
print("Score on training=",score)
ys = clf.predict_proba(X_train)
auc = roc_auc_score(y_train,ys[:,1])
print("AUC on training=",auc)

score = clf.score(X_val,y_val)
print("Score on validation=",score)
yprob = clf.predict_proba(X_val)
auc = roc_auc_score(y_val,yprob[:,1])
print("AUC on validation=",auc)




# Calculation of confusion matrix and other validation indexes
predicted = clf.predict(X_val)
print("TEST: \n Classification report for classifier %s:\n\n%s\n"
      % (clf, metrics.classification_report(y_val, predicted)))
print("Confusion matrix sul Validation:\n%s" % metrics.confusion_matrix(y_val, predicted))




# Build the confusion matrix with a threshold different from 0.5

ix = (yprob[:,1]>0.1)
print(ix)
yyy = np.zeros((len(y_val)))
yyy[ix] = 1
print("TEST: \n Classification report for classifier %s:\n\n%s\n"
      % (clf, metrics.classification_report(y_val, yyy)))
print("Confusion matrix sul Validation:\n%s" % metrics.confusion_matrix(y_val, yyy))




Y_test = clf.predict_proba(X_test)
soluz = pd.DataFrame({'ID':XT[:,0].astype(int),'TARGET':Y_test[:,1]})
soluz.to_csv('fine.csv', sep=",",index=False)

