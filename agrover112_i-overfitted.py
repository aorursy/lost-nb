#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




loading=pd.read_csv("../input/train.csv").drop('id', axis=1)
loading2=pd.read_csv("../input/test.csv").drop('id',axis=1)




y=loading['target']
X=loading.drop('target', axis=1)




#Test -train splittting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20,random_state = 0)




#from sklearn.svm import SVC
#classifier=SVC(kernel='linear',class_weight='balanced',gamma='auto',probability=True)#This one overfit BEAUTIFULLY
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(class_weight='balanced',penalty='l1',C=0.1,random_state=0,solver='liblinear')#L1='Lasso',l2="Ridge"
classifier.fit(X_train,y_train)
#print(classifier.score(X_train,y_train))




def auc_curve():
    from sklearn import metrics
    y_pred_proba=classifier.predict_proba(X_test)[::,1]
    fpr,tpr,_=metrics.roc_curve(y_test,y_pred_proba)
    auc=metrics.roc_auc_score(y_test,y_pred_proba)
    plt.plot(fpr,tpr,label="auc="+str(auc))
    plt.legend(loc=4)
    plt.show()
    return y_pred_proba
y_pred_proba=auc_curve()




y_pred=classifier.predict(X_test)




from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm




from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(classifier,X_test,y_test,cv=10)
print(accuracies.mean())




sub=pd.read_csv("../input/sample_submission.csv")




sub['target']=classifier.predict_proba(loading2)[::,1]
sub.to_csv('submission.csv', index=False)

