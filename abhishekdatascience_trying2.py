#!/usr/bin/env python
# coding: utf-8



#Abhishek Bhardwaj

import pandas as pd
import numpy as np
from sklearn import linear_model,svm,tree
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score,     recall_score, confusion_matrix, classification_report,     accuracy_score, f1_score

def getdummy(df,Category):
	df_temp=pd.get_dummies(df[Category]).astype(int).rename(columns=lambda x: str(Category)+"_"+ str(x))
	temp=pd.merge(df, df_temp, right_index=True, left_index=True)
	temp=temp.drop(Category,1)
	return temp


df_train = pd.read_csv('../input/act_train.csv')
df_train = df_train.rename(columns=lambda x: "act"+"_"+ str(x))
df_people = pd.read_csv('../input/people.csv')
df=pd.merge(df_train, df_people, left_on='act_people_id', right_on='people_id',how='left')
# xt=list(df)
# for x in xt:
# 	print x,(len(df[x])-df[x].count())*100.0/len(df)
df_subset=df[['act_activity_category', 'act_char_10', 'act_outcome', 'char_1', 'group_1', 'char_2', 'date', 'char_3', 'char_4', 'char_5', 'char_6', 'char_7', 'char_8', 'char_9', 'char_10', 'char_11', 'char_12', 'char_13', 'char_14', 'char_15', 'char_16', 'char_17', 'char_18', 'char_19', 'char_20', 'char_21', 'char_22', 'char_23', 'char_24', 'char_25', 'char_26', 'char_27', 'char_28', 'char_29', 'char_30', 'char_31', 'char_32', 'char_33', 'char_34', 'char_35', 'char_36', 'char_37', 'char_38']]
df_subset=df_subset.dropna()
df=df_subset
 

xt1 = df.select_dtypes(include=['bool']).dtypes.index
for x in xt1:
	df=getdummy(df,x)
xt1 = df.select_dtypes(include=['object']).dtypes.index
for x in xt1:
	df=getdummy(df,x)

df_test = pd.read_csv('../input/act_test.csv')
df_test = df_test.rename(columns=lambda x: "acttest"+"_"+ str(x))
dft=pd.merge(df_test, df_people, left_on='acttest_people_id', right_on='people_id',how='left')
dft=dft[['acttest_activity_category', 'acttest_char_10', 'char_1', 'group_1', 'char_2', 'date', 'char_3', 'char_4', 'char_5', 'char_6', 'char_7', 'char_8', 'char_9', 'char_10', 'char_11', 'char_12', 'char_13', 'char_14', 'char_15', 'char_16', 'char_17', 'char_18', 'char_19', 'char_20', 'char_21', 'char_22', 'char_23', 'char_24', 'char_25', 'char_26', 'char_27', 'char_28', 'char_29', 'char_30', 'char_31', 'char_32', 'char_33', 'char_34', 'char_35', 'char_36', 'char_37', 'char_38']]

xt1 = dft.select_dtypes(include=['bool']).dtypes.index
for x in xt1:
	dft=getdummy(dft,x)
xt1 = dft.select_dtypes(include=['object']).dtypes.index
for x in xt1:
	dft=getdummy(dft,x)



#ValidationSet
train, test = train_test_split(df, train_size = 0.7)
X = train.drop("act_outcome",1)
Y = train['act_outcome'].astype(int)

#Support Vector Machine 
clf = svm.SVC(gamma=0.01, C=100.)
clf.fit(X, Y) 
#LogisticRegression
h = .02 
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(X, Y)
#DecisionTree
dt = tree.DecisionTreeClassifier()
dt.fit(X, Y)
#RandomForest
rf = RandomForestClassifier(n_estimators=1000)
rf.fit(X,Y)
#RidgeRegression
ridge = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
ridge.fit(X,Y)

#Validating
X = test.drop("act_outcome",1)
Y = test['act_outcome'].astype(int)
print("done")

