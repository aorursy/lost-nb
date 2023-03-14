#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
import xgboost as xgb
from xgboost import XGBClassifier




from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import *
from sklearn.model_selection import train_test_split
def train_and_evaluate(clf, X_train, y_train):
    cv = KFold(X_train.shape[0], 5, shuffle=True, random_state=33)
    scores = cross_val_score(clf, X_train, y_train, cv=cv)
    print ('Average coefficient of determination using 5-fold cross validation:', np.mean(scores))




train_variants_df = pd.read_csv("../input/training_variants")
test_variants_df = pd.read_csv("../input/test_variants")
train_text_df = pd.read_csv("../input/training_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
test_text_df = pd.read_csv("../input/test_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
print("Train and Test variants shape : ",train_variants_df.shape, test_variants_df.shape)
print("Train and Test text shape : ",train_text_df.shape, test_text_df.shape)

# combine into one df
train_merge_df = pd.concat([train_text_df,train_variants_df],axis=1)




train ,test = train_test_split(train_merge_df,test_size=0.2) 




X_train = train['Text'].values
X_test = test['Text'].values
y_train = train['Class'].values
y_test = test['Class'].values
X=train_merge_df['Text'].values
y=train_merge_df['Class'].values




tfidf = TfidfVectorizer(
    min_df=5, max_features=16000, strip_accents='unicode',lowercase =True, 
    analyzer='word', token_pattern=r'\w+', ngram_range=(1,4), use_idf=True, 
    smooth_idf=True, sublinear_tf=True, stop_words = 'english'
).fit(X)




X_train_text = tfidf.transform(X_train)
X_test_text = tfidf.transform(X_test)
X_all_text=tfidf.transform(X)




# add feature name and make as df
feature_names=tfidf.get_feature_names()
array_feature_names=np.asarray(feature_names)
test=X_all_text.todense()
X_all_text_name=pd.DataFrame(test,columns=feature_names)





clf=LinearSVC(penalty='l1',dual=False,tol=1e-3)
clf.fit(X_train_text,y_train)
clf.score(X_test_text,y_test)




sgd=SGDClassifier(loss='hinge',penalty='l1',n_iter=50,alpha=0.00001,fit_intercept=True)
train_and_evaluate(sgd, X_all_text, y_train)




# Variation type that has highest number of occurrences in each class
fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(15,15))

for i in range(3):
    for j in range(3):
        gene_count_grp = train_variants_df[train_variants_df["Class"]==((i*3+j)+1)].groupby('Variation')["ID"].count().reset_index()
        sorted_gene_group = gene_count_grp.sort_values('ID', ascending=False)
        sorted_gene_group_top_7 = sorted_gene_group[:7]
        sns.barplot(x="Variation", y="ID", data=sorted_gene_group_top_7, ax=axs[i][j])
        print sorted_gene_group_top_7




Train_all=pd.concat([train_variants_df,X_all_text_name],axis=1)




Train_all['var_del']=Train_all['Variation'].map(lambda s: 1 if 'Deletion' in s or 'del' in s else 0)
Train_all['var_stop']=Train_all['Variation'].map(lambda s: 1 if '*' in s or 'Truncating Mutations' in s else 0)
Train_all['var_fusion']=Train_all['Variation'].map(lambda s: 1 if 'Fusion' in s or 'del' in s else 0)
Train_all['var_amp']=Train_all['Variation'].map(lambda s: 1 if 'Amplification' in s else 0)
Train_all['var_ovex']=Train_all['Variation'].map(lambda s: 1 if 'Overexpression' in s else 0)
Train_all['var_methy']=Train_all['Variation'].map(lambda s: 1 if 'methyl' in s or 'sil' in s else 0)
Train_all['var_ins']=Train_all['Variation'].map(lambda s: 1 if 'ins' in s else 0)
Train_all['var_splice']=Train_all['Variation'].map(lambda s: 1 if 'splice' in s else 0)
Train_all['var_dup']=Train_all['Variation'].map(lambda s: 1 if 'dup' in s else 0)
Train_all["var_simple"] = Train_all.Variation.str.contains(r'^[A-Z]\d{1,7}[A-Z]',case=0)





X_all=Train_all.drop(['ID','Class'],axis=1)
y_all=Train_all['Class']




X_all_cat = pd.get_dummies(X_all, prefix=['Gene','Variation'] )




train_all,test_all = train_test_split(X_all_cat.as_matrix(),test_size=0.2) 




svc=LinearSVC(penalty='l1',dual=False,tol=1e-3)
sgd=SGDClassifier(loss='hinge',penalty='l2',n_iter=50,alpha=0.00001,fit_intercept=True)
xgb = XGBClassifier(
    max_depth=3,
    n_estimators=500,
    subsample=0.5,
    learning_rate=0.1
    )




from sklearn.ensemble import VotingClassifier
eclf = VotingClassifier(estimators=[('svc', svc), ('sgd', sgd), ('xgb', xgb)], voting='hard',n_jobs=-1)




eclf.fit(X_all_cat.as_matrix(),y_all)




train_and_evaluate(eclf, X_all_cat.as_matrix(), y_all)




train,test = X_all_cat.align(Text_X_all_cat, join='inner', axis=1)




eclf_align = VotingClassifier(estimators=[('svc', svc), ('sgd', sgd), ('xgb', xgb)], voting='hard',n_jobs=-1)
eclf_align=eclf_align.fit(train,y_all)




prediction_test=eclf_align.predict(test)

