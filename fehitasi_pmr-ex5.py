#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np




import os




trein= pd.read_csv('../input/ola245/treino_data23', 
        
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
trein.info()




trein.head()




trein.describe()




features=trein.keys()
features




coav=['word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d',
       'word_freq_our', 'word_freq_over', 'word_freq_remove',
       'word_freq_internet', 'word_freq_order', 'word_freq_mail',
       'word_freq_receive', 'word_freq_will', 'word_freq_people',
       'word_freq_report', 'word_freq_addresses', 'word_freq_free',
       'word_freq_business', 'word_freq_email', 'word_freq_you',
       'word_freq_credit', 'word_freq_your', 'word_freq_font', 'word_freq_000',
       'word_freq_money', 'word_freq_hp', 'word_freq_hpl', 'word_freq_george',
       'word_freq_650', 'word_freq_lab', 'word_freq_labs', 'word_freq_telnet',
       'word_freq_857', 'word_freq_data', 'word_freq_415', 'word_freq_85',
       'word_freq_technology', 'word_freq_1999', 'word_freq_parts',
       'word_freq_pm', 'word_freq_direct', 'word_freq_cs', 'word_freq_meeting',
       'word_freq_original', 'word_freq_project', 'word_freq_re',
       'word_freq_edu', 'word_freq_table', 'word_freq_conference',
       'char_freq_;', 'char_freq_(', 'char_freq_[', 'char_freq_!',
       'char_freq_$', 'char_freq_#', 'capital_run_length_average',
       'capital_run_length_longest', 'capital_run_length_total', 'ham', 'Id']
len(coav)




test= pd.read_csv('../input/ola245/test_features23', 
        
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")














#Matriz de Correlação
import seaborn as sns
corr = trein[['word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d',
       'word_freq_our', 'word_freq_over', 'word_freq_remove',
       'word_freq_internet', 'word_freq_order', 'word_freq_mail',
       'word_freq_receive', 'word_freq_will', 'word_freq_people',
       'word_freq_report', 'word_freq_addresses', 'word_freq_free',
       'word_freq_business', 'word_freq_email', 'word_freq_you',
       'word_freq_credit', 'word_freq_your', 'word_freq_font', 'word_freq_000',
       'word_freq_money', 'word_freq_hp', 'word_freq_hpl', 'word_freq_george',
       'word_freq_650', 'word_freq_lab', 'word_freq_labs', 'word_freq_telnet',
       'word_freq_857', 'word_freq_data', 'word_freq_415', 'word_freq_85',
       'word_freq_technology', 'word_freq_1999', 'word_freq_parts',
       'word_freq_pm', 'word_freq_direct', 'word_freq_cs', 'word_freq_meeting',
       'word_freq_original', 'word_freq_project', 'word_freq_re',
       'word_freq_edu', 'word_freq_table', 'word_freq_conference',
       'char_freq_;', 'char_freq_(', 'char_freq_[', 'char_freq_!',
       'char_freq_$', 'char_freq_#', 'capital_run_length_average']]

mask=np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)]=True
f, ax = plt.subplots(figsize=(17,17))
cmap=sns.diverging_palette(220, 10, as_cmap=False)
sns.heatmap(corr, mask=mask, cmap =cmap, vmax=1, vmin=-1, center=0, square=True, linewidth=.5,cbar_kws={"shrink":.5})
corr = trein.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220,10,as_cmap=True), square=True, ax=ax)




sns.pairplot(trein,hue="ham", vars=['word_freq_650','word_freq_lab','word_freq_labs','word_freq_telnet','word_freq_857','word_freq_415','word_freq_85','word_freq_technology','word_freq_direct'])




g = sns.PairGrid(trein,x_vars=['word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d',
       'word_freq_our', 'word_freq_over', 'word_freq_remove',
       'word_freq_internet', 'word_freq_order', 'word_freq_mail',
       'word_freq_receive', 'word_freq_will', 'word_freq_people',
       'word_freq_report', 'word_freq_addresses', 'word_freq_free',
       'word_freq_business', 'word_freq_email', 'word_freq_you',
       'word_freq_credit', 'word_freq_your', 'word_freq_font', 'word_freq_000',
       'word_freq_money', 'word_freq_hp', 'word_freq_hpl', 'word_freq_george',
       'word_freq_650', 'word_freq_lab', 'word_freq_labs', 'word_freq_telnet',
       'word_freq_857', 'word_freq_data', 'word_freq_415', 'word_freq_85',
       'word_freq_technology', 'word_freq_1999', 'word_freq_parts',
       'word_freq_pm', 'word_freq_direct', 'word_freq_cs', 'word_freq_meeting',
       'word_freq_original', 'word_freq_project', 'word_freq_re',
       'word_freq_edu', 'word_freq_table', 'word_freq_conference',
       'char_freq_;', 'char_freq_(', 'char_freq_[', 'char_freq_!',
       'char_freq_$', 'char_freq_#', 'capital_run_length_average',
       'capital_run_length_longest', 'capital_run_length_total'], y_vars=["ham"])
g.map(plt.scatter)




from sklearn import tree




x=trein[['word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d',
       'word_freq_our', 'word_freq_over', 'word_freq_remove',
       'word_freq_internet', 'word_freq_order', 'word_freq_mail',
       'word_freq_receive', 'word_freq_will', 'word_freq_people',
       'word_freq_report', 'word_freq_addresses', 'word_freq_free',
       'word_freq_business', 'word_freq_email', 'word_freq_you',
       'word_freq_credit', 'word_freq_your', 'word_freq_font', 'word_freq_000',
       'word_freq_money', 'word_freq_hp', 'word_freq_hpl', 'word_freq_george',
       'word_freq_650', 'word_freq_lab', 'word_freq_labs', 'word_freq_telnet',
       'word_freq_857', 'word_freq_data', 'word_freq_415', 'word_freq_85',
       'word_freq_technology', 'word_freq_1999', 'word_freq_parts',
       'word_freq_pm', 'word_freq_direct', 'word_freq_cs', 'word_freq_meeting',
       'word_freq_original', 'word_freq_project', 'word_freq_re',
       'word_freq_edu', 'word_freq_table', 'word_freq_conference',
       'char_freq_;', 'char_freq_(', 'char_freq_[', 'char_freq_!',
       'char_freq_$', 'char_freq_#', 'capital_run_length_average',
       'capital_run_length_longest', 'capital_run_length_total']]




y= trein[["ham"]]




clf = tree.DecisionTreeClassifier()
tree = clf.fit(x,y)




from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree, x, y, cv=10)
scores




from sklearn.metrics import confusion_matrix
y_test=tree.predict(x)
confusion_matrix(y, y_test)




from sklearn.metrics import accuracy_score
accuracy_score(y,y_test)




from sklearn.metrics import fbeta_score
fbeta_score(y, y_test, average='macro', beta=0.5)




from sklearn.metrics import f1_score
f1_score(y, y_test, average='binary')




tree




x=test[['word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d',
       'word_freq_our', 'word_freq_over', 'word_freq_remove',
       'word_freq_internet', 'word_freq_order', 'word_freq_mail',
       'word_freq_receive', 'word_freq_will', 'word_freq_people',
       'word_freq_report', 'word_freq_addresses', 'word_freq_free',
       'word_freq_business', 'word_freq_email', 'word_freq_you',
       'word_freq_credit', 'word_freq_your', 'word_freq_font', 'word_freq_000',
       'word_freq_money', 'word_freq_hp', 'word_freq_hpl', 'word_freq_george',
       'word_freq_650', 'word_freq_lab', 'word_freq_labs', 'word_freq_telnet',
       'word_freq_857', 'word_freq_data', 'word_freq_415', 'word_freq_85',
       'word_freq_technology', 'word_freq_1999', 'word_freq_parts',
       'word_freq_pm', 'word_freq_direct', 'word_freq_cs', 'word_freq_meeting',
       'word_freq_original', 'word_freq_project', 'word_freq_re',
       'word_freq_edu', 'word_freq_table', 'word_freq_conference',
       'char_freq_;', 'char_freq_(', 'char_freq_[', 'char_freq_!',
       'char_freq_$', 'char_freq_#', 'capital_run_length_average',
       'capital_run_length_longest', 'capital_run_length_total']]
resposta1=tree.predict(x)




output = pd.DataFrame(test.Id)
output["ham"] = resposta1
df = pd.DataFrame (output)
df.to_csv("sub.csv", index=False)




print(int(resposta1[1]))




x=trein[['word_freq_remove','word_freq_free','word_freq_3d','word_freq_table','word_freq_000','word_freq_credit','word_freq_remove','word_freq_free','word_freq_3d','word_freq_table','word_freq_000','word_freq_addresses'
    ,'word_freq_money','word_freq_receive','word_freq_parts','word_freq_order', 'word_freq_internet','word_freq_over','word_freq_report']]
y= trein["ham"]
xtest=test[['word_freq_remove','word_freq_free','word_freq_3d','word_freq_table','word_freq_000','word_freq_credit','word_freq_remove','word_freq_free','word_freq_3d','word_freq_table','word_freq_000','word_freq_addresses'
    ,'word_freq_money','word_freq_receive','word_freq_parts','word_freq_order', 'word_freq_internet','word_freq_over','word_freq_report']]




from sklearn.svm import SVC
clf = SVC(gamma='auto')
clf.fit(x, y) 




y2=clf.predict(x)




from sklearn.metrics import confusion_matrix
confusion_matrix(y,y2)




resposta2=clf.predict(xtest)
output = pd.DataFrame(test.Id)
output["ham"] = resposta2
df = pd.DataFrame (output)
df.to_csv("sub.csv", index=False)




from sklearn.naive_bayes import GaussianNB




x=trein[['word_freq_remove','word_freq_free','word_freq_3d','word_freq_table','word_freq_000','word_freq_credit','word_freq_remove','word_freq_free','word_freq_3d','word_freq_table','word_freq_000','word_freq_addresses'
    ,'word_freq_money','word_freq_receive','word_freq_parts','word_freq_order', 'word_freq_internet','word_freq_over','word_freq_report']]
y= trein[["ham"]]




clf = GaussianNB()
y=np.array(y)

nbg=clf.fit(x,y)




from sklearn.model_selection import cross_val_score
scores = cross_val_score(nbg, x, y, cv=10)
scores




from sklearn.metrics import confusion_matrix
y_test=nbg.predict(x)
confusion_matrix(y, y_test)




x=test[['word_freq_remove','word_freq_free','word_freq_3d','word_freq_table','word_freq_000','word_freq_credit','word_freq_remove','word_freq_free','word_freq_3d','word_freq_table','word_freq_000','word_freq_addresses'
    ,'word_freq_money','word_freq_receive','word_freq_parts','word_freq_order', 'word_freq_internet','word_freq_over','word_freq_report']]
resposta3=nbg.predict(x)




output = pd.DataFrame(test.Id)
output["ham"] = resposta3
df = pd.DataFrame (output)
df.to_csv("sub.csv", index=False)




from sklearn.naive_bayes import MultinomialNB




x=trein[['word_freq_remove','word_freq_free','word_freq_3d','word_freq_table','word_freq_000','word_freq_credit','word_freq_remove','word_freq_free','word_freq_3d','word_freq_table','word_freq_000','word_freq_addresses'
    ,'word_freq_money','word_freq_receive','word_freq_parts','word_freq_order', 'word_freq_internet','word_freq_over','word_freq_report']]
y= trein["ham"]
xtest=test[['word_freq_remove','word_freq_free','word_freq_3d','word_freq_table','word_freq_000','word_freq_credit','word_freq_remove','word_freq_free','word_freq_3d','word_freq_table','word_freq_000','word_freq_addresses'
    ,'word_freq_money','word_freq_receive','word_freq_parts','word_freq_order', 'word_freq_internet','word_freq_over','word_freq_report']]




clf = MultinomialNB().fit(x, y)
clf




resposta4 = clf.predict(xtest)




output = pd.DataFrame(test.Id)
output["ham"] = resposta4
df = pd.DataFrame (output)
df.to_csv("sub.csv", index=False)




from sklearn.metrics import confusion_matrix
ytest=clf.predict(x)
confusion_matrix(y, ytest)




from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import RandomizedSearchCV




x=trein[['word_freq_remove','word_freq_free','word_freq_3d','word_freq_table','word_freq_000','word_freq_credit','word_freq_remove','word_freq_free','word_freq_3d','word_freq_table','word_freq_000','word_freq_addresses'
    ,'word_freq_money','word_freq_receive','word_freq_parts','word_freq_order', 'word_freq_internet','word_freq_over','word_freq_report']]
y= trein["ham"]




grid.fit(x,y)
grid.grid_scores_




grid.best_params_ 




knn = KNeighborsClassifier(n_neighbors=11, p=1, weights= 'uniform' )
knn.fit(x,y)




xtest=test[['word_freq_remove','word_freq_free','word_freq_3d','word_freq_table','word_freq_000','word_freq_credit','word_freq_remove','word_freq_free','word_freq_3d','word_freq_table','word_freq_000','word_freq_addresses'
    ,'word_freq_money','word_freq_receive','word_freq_parts','word_freq_order', 'word_freq_internet','word_freq_over','word_freq_report']]




resposta5= knn.predict(xtest)




output = pd.DataFrame(test.Id)
output["ham"] = resposta5
df = pd.DataFrame (output)
df.to_csv("sub.csv", index=False)




from sklearn.metrics import confusion_matrix
ytest=clf.predict(x)
confusion_matrix(y, ytest)




from sklearn.naive_bayes import BernoulliNB




x=trein[['word_freq_remove','word_freq_free','word_freq_3d','word_freq_table','word_freq_000','word_freq_credit','word_freq_remove','word_freq_free','word_freq_3d','word_freq_table','word_freq_000','word_freq_addresses'
    ,'word_freq_money','word_freq_receive','word_freq_parts','word_freq_order', 'word_freq_internet','word_freq_over','word_freq_report']]
y= trein["ham"]
xtest=test[['word_freq_remove','word_freq_free','word_freq_3d','word_freq_table','word_freq_000','word_freq_credit','word_freq_remove','word_freq_free','word_freq_3d','word_freq_table','word_freq_000','word_freq_addresses'
    ,'word_freq_money','word_freq_receive','word_freq_parts','word_freq_order', 'word_freq_internet','word_freq_over','word_freq_report']]




clf = BernoulliNB().fit(x, y)
clf




y_pred = clf.predict(xtest)




from sklearn.metrics import confusion_matrix
resposta6=clf.predict(xtest)
confusion_matrix(y, ytest)




output = pd.DataFrame(test.Id)
output["ham"] = resposta6
df = pd.DataFrame (output)
df.to_csv("sub.csv", index=False)




resposta6




from sklearn import metrics




fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)






