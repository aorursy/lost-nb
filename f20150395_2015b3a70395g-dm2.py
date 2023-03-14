#!/usr/bin/env python
# coding: utf-8



import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import preprocessing




data_orig = pd.read_csv("../input/train.csv")
data = data_orig




data = data.replace({'?':np.nan})




data.isnull().sum(axis = 0)




nan_col = ['Worker Class','Fill','Teen','PREV','Live','MOVE','REG','MSA','State','Area','Reason','MLU','MOC','MIC','Enrolled']




data = data.drop(nan_col, axis = 1)




pd.set_option('display.max_columns', 60)
data.head()




null_columns = data.columns[data.isna().any()]
null_columns




for c in null_columns:
    data[c] = data[c].fillna(data[c].mode()[0])




for col in data.columns:
    print(col , data[col].unique())
    print(" ")




categorical_col = ['COB SELF','COB MOTHER','COB FATHER','Detailed']
data = data.drop(categorical_col, axis = 1)
data.head()




new_data = pd.get_dummies(data, columns=['Married_Life','Cast', 'Hispanic', 'Sex','Full/Part', 'Tax Status',
                                        'Summary', 'Citizen','Schooling'])




X_train = new_data
X_train  = X_train.drop(['Class'], axis=1)
y_train = new_data['Class']




from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier




#Using RandomForestClassifier for feature selection
plt.figure(figsize=(60,60))
model = AdaBoostClassifier(random_state=42)
model = model.fit(X_train,y_train)
features = X_train.columns
importances = model.feature_importances_
impfeatures_index = np.argsort(importances)
#print([features[i] for i in impfeatures_index])
sns.barplot(x = [importances[i] for i in impfeatures_index], y = [features[i] for i in impfeatures_index])
plt.xlabel('value', fontsize=32)
plt.ylabel('parameter', fontsize=32)
plt.tick_params(axis='both', which='major', labelsize=25)
plt.tick_params(axis='both', which='minor', labelsize=25)
plt.show()




#Selecting top features based on their importance according to the above graph
impfeatures = features[impfeatures_index[-25:]]
X_train = X_train[[features for features in impfeatures]]




from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42)




from imblearn.over_sampling import RandomOverSampler




ros = RandomOverSampler(random_state=0)
X_resampled1, y_resampled1 = ros.fit_resample(X_train, y_train)




score_train_RF = []
score_test_RF = []

for i in range(1,18,1):
    rf = RandomForestClassifier(n_estimators=i, random_state = 42,class_weight='balanced')
    rf.fit(X_resampled1,y_resampled1)
    sc_train = rf.score(X_resampled1,y_resampled1)
    score_train_RF.append(sc_train)
    sc_test = rf.score(X_val,y_val)
    score_test_RF.append(sc_test)




plt.figure(figsize=(10,6))
train_score,=plt.plot(range(1,18,1),score_train_RF,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='green', markersize=5)
test_score,=plt.plot(range(1,18,1),score_test_RF,color='red',linestyle='dashed',  marker='o',
         markerfacecolor='blue', markersize=5)
plt.legend( [train_score,test_score],["Train Score","Test Score"])
plt.title('Fig4. Score vs. No. of Trees')
plt.xlabel('No. of Trees')
plt.ylabel('Score')




rf = RandomForestClassifier(n_estimators=14, random_state = 42,class_weight='balanced')
rf.fit(X_resampled1,y_resampled1)
rf.score(X_val,y_val)




from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
y_pred_RF = rf.predict(X_val)
confusion_matrix(y_val, y_pred_RF)




from sklearn.metrics import roc_auc_score




print(roc_auc_score(y_val,y_pred_RF))




print(classification_report(y_val, y_pred_RF))




from sklearn.tree import DecisionTreeClassifier




from sklearn.tree import DecisionTreeClassifier

train_acc = []
test_acc = []
for i in range(2,30):
    dTree = DecisionTreeClassifier(min_samples_split=i, random_state = 42,class_weight='balanced')
    dTree.fit(X_resampled1,y_resampled1)
    acc_train = dTree.score(X_resampled1,y_resampled1)
    train_acc.append(acc_train)
    acc_test = dTree.score(X_val,y_val)
    test_acc.append(acc_test)




plt.figure(figsize=(10,6))
train_score,=plt.plot(range(2,30),train_acc,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='green', markersize=5)
test_score,=plt.plot(range(2,30),test_acc,color='red',linestyle='dashed',  marker='o',
         markerfacecolor='blue', markersize=5)
plt.legend( [train_score, test_score],["Train Accuracy", "Validation Accuracy"])
plt.title('Accuracy vs min_samples_split')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')




dTree = DecisionTreeClassifier(class_weight='balanced', random_state = 42)
dTree.fit(X_resampled1,y_resampled1)
dTree.score(X_val,y_val)




y_pred_DT = dTree.predict(X_val)
print(confusion_matrix(y_val, y_pred_DT))




print(roc_auc_score(y_val,y_pred_DT))




print(classification_report(y_val, y_pred_DT))




from sklearn.ensemble import AdaBoostClassifier




score_train_AB = []
score_test_AB = []

for i in range(1,20,1):
    ab = AdaBoostClassifier(n_estimators=i, random_state = 42)
    ab.fit(X_resampled1,y_resampled1)
    sc_train = ab.score(X_resampled1,y_resampled1)
    score_train_AB.append(sc_train)
    sc_test = ab.score(X_val,y_val)
    score_test_AB.append(sc_test)




plt.figure(figsize=(10,6))
train_score,=plt.plot(range(1,20,1),score_train_AB,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='green', markersize=5)
test_score,=plt.plot(range(1,20,1),score_test_AB,color='red',linestyle='dashed',  marker='o',
         markerfacecolor='blue', markersize=5)
plt.legend( [train_score,test_score],["Train Score","Test Score"])
plt.title('Fig4. Score vs. No. of Trees')
plt.xlabel('No. of Trees')
plt.ylabel('Score')




ab = AdaBoostClassifier(n_estimators=500, random_state = 42)
ab.fit(X_resampled1,y_resampled1)
ab.score(X_val,y_val)




from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
y_pred_AB = ab.predict(X_val)
confusion_matrix(y_val, y_pred_AB)




print(roc_auc_score(y_val,y_pred_AB))




print(classification_report(y_val, y_pred_AB))




from sklearn.naive_bayes import GaussianNB as NB




nb = NB()
nb.fit(X_resampled1,y_resampled1)
nb.score(X_val,y_val)




y_pred_NB = nb.predict(X_val)
print(confusion_matrix(y_val, y_pred_NB))




print(roc_auc_score(y_val,y_pred_AB))




print(classification_report(y_val, y_pred_NB))




X_test1 = pd.read_csv('../input/test.csv')
X_test1 = X_test1.replace({'?':np.nan})




ID = X_test1['ID']




del_col = ['Worker Class','Fill','Teen','PREV','Live','MOVE','REG','MSA',
           'State','Area','Reason','MLU','MOC','MIC','Enrolled',
          'COB SELF','COB MOTHER','COB FATHER','Detailed'
          ]

X_test1 = X_test1.drop(del_col, axis = 1)




null_columns_test = X_test1.columns[X_test1.isna().any()]
null_columns_test




for c in null_columns_test:
    X_test1[c] = X_test1[c].fillna(X_test1[c].mode()[0])




X_test1 = pd.get_dummies(X_test1, columns=['Married_Life','Cast', 'Hispanic', 'Sex','Full/Part', 'Tax Status',
                                        'Summary', 'Citizen','Schooling'])




X_test1 = X_test1[[features for features in impfeatures]]




preds = ab.predict(X_test1)




df = pd.DataFrame(columns=['ID', 'Class'])
df['ID']=ID
df['Class']=preds
df.head()




#df.to_csv('2015b3a70395g.csv',index=False)




from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html='<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title} </a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)
create_download_link(df)






