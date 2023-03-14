#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


import os


# In[ ]:


trein= pd.read_csv('../input/ola245/treino_data23', 
        
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
trein.info()


# In[ ]:


trein.head()


# In[ ]:


trein.describe()


# In[ ]:


features=trein.keys()
features


# In[ ]:


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


# In[ ]:


test= pd.read_csv('../input/ola245/test_features23', 
        
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")


# In[ ]:





# In[ ]:





# In[ ]:


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


# In[ ]:


sns.pairplot(trein,hue="ham", vars=['word_freq_650','word_freq_lab','word_freq_labs','word_freq_telnet','word_freq_857','word_freq_415','word_freq_85','word_freq_technology','word_freq_direct'])


# In[ ]:


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


# In[ ]:


from sklearn import tree


# In[ ]:


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


# In[ ]:


y= trein[["ham"]]


# In[ ]:


clf = tree.DecisionTreeClassifier()
tree = clf.fit(x,y)


# In[ ]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree, x, y, cv=10)
scores


# In[ ]:


from sklearn.metrics import confusion_matrix
y_test=tree.predict(x)
confusion_matrix(y, y_test)


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(y,y_test)


# In[ ]:


from sklearn.metrics import fbeta_score
fbeta_score(y, y_test, average='macro', beta=0.5)


# In[ ]:


from sklearn.metrics import f1_score
f1_score(y, y_test, average='binary')


# In[ ]:


tree


# In[ ]:


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


# In[ ]:


output = pd.DataFrame(test.Id)
output["ham"] = resposta1
df = pd.DataFrame (output)
df.to_csv("sub.csv", index=False)


# In[ ]:


print(int(resposta1[1]))


# In[ ]:


x=trein[['word_freq_remove','word_freq_free','word_freq_3d','word_freq_table','word_freq_000','word_freq_credit','word_freq_remove','word_freq_free','word_freq_3d','word_freq_table','word_freq_000','word_freq_addresses'
    ,'word_freq_money','word_freq_receive','word_freq_parts','word_freq_order', 'word_freq_internet','word_freq_over','word_freq_report']]
y= trein["ham"]
xtest=test[['word_freq_remove','word_freq_free','word_freq_3d','word_freq_table','word_freq_000','word_freq_credit','word_freq_remove','word_freq_free','word_freq_3d','word_freq_table','word_freq_000','word_freq_addresses'
    ,'word_freq_money','word_freq_receive','word_freq_parts','word_freq_order', 'word_freq_internet','word_freq_over','word_freq_report']]


# In[ ]:


from sklearn.svm import SVC
clf = SVC(gamma='auto')
clf.fit(x, y) 


# In[ ]:


y2=clf.predict(x)


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y,y2)


# In[ ]:


resposta2=clf.predict(xtest)
output = pd.DataFrame(test.Id)
output["ham"] = resposta2
df = pd.DataFrame (output)
df.to_csv("sub.csv", index=False)


# In[ ]:


from sklearn.naive_bayes import GaussianNB


# In[ ]:


x=trein[['word_freq_remove','word_freq_free','word_freq_3d','word_freq_table','word_freq_000','word_freq_credit','word_freq_remove','word_freq_free','word_freq_3d','word_freq_table','word_freq_000','word_freq_addresses'
    ,'word_freq_money','word_freq_receive','word_freq_parts','word_freq_order', 'word_freq_internet','word_freq_over','word_freq_report']]
y= trein[["ham"]]


# In[ ]:


clf = GaussianNB()
y=np.array(y)

nbg=clf.fit(x,y)


# In[ ]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(nbg, x, y, cv=10)
scores


# In[ ]:


from sklearn.metrics import confusion_matrix
y_test=nbg.predict(x)
confusion_matrix(y, y_test)


# In[ ]:


x=test[['word_freq_remove','word_freq_free','word_freq_3d','word_freq_table','word_freq_000','word_freq_credit','word_freq_remove','word_freq_free','word_freq_3d','word_freq_table','word_freq_000','word_freq_addresses'
    ,'word_freq_money','word_freq_receive','word_freq_parts','word_freq_order', 'word_freq_internet','word_freq_over','word_freq_report']]
resposta3=nbg.predict(x)


# In[ ]:


output = pd.DataFrame(test.Id)
output["ham"] = resposta3
df = pd.DataFrame (output)
df.to_csv("sub.csv", index=False)


# In[ ]:


from sklearn.naive_bayes import MultinomialNB


# In[ ]:


x=trein[['word_freq_remove','word_freq_free','word_freq_3d','word_freq_table','word_freq_000','word_freq_credit','word_freq_remove','word_freq_free','word_freq_3d','word_freq_table','word_freq_000','word_freq_addresses'
    ,'word_freq_money','word_freq_receive','word_freq_parts','word_freq_order', 'word_freq_internet','word_freq_over','word_freq_report']]
y= trein["ham"]
xtest=test[['word_freq_remove','word_freq_free','word_freq_3d','word_freq_table','word_freq_000','word_freq_credit','word_freq_remove','word_freq_free','word_freq_3d','word_freq_table','word_freq_000','word_freq_addresses'
    ,'word_freq_money','word_freq_receive','word_freq_parts','word_freq_order', 'word_freq_internet','word_freq_over','word_freq_report']]


# In[ ]:


clf = MultinomialNB().fit(x, y)
clf


# In[ ]:


resposta4 = clf.predict(xtest)


# In[ ]:


output = pd.DataFrame(test.Id)
output["ham"] = resposta4
df = pd.DataFrame (output)
df.to_csv("sub.csv", index=False)


# In[ ]:


from sklearn.metrics import confusion_matrix
ytest=clf.predict(x)
confusion_matrix(y, ytest)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import RandomizedSearchCV


# In[ ]:


x=trein[['word_freq_remove','word_freq_free','word_freq_3d','word_freq_table','word_freq_000','word_freq_credit','word_freq_remove','word_freq_free','word_freq_3d','word_freq_table','word_freq_000','word_freq_addresses'
    ,'word_freq_money','word_freq_receive','word_freq_parts','word_freq_order', 'word_freq_internet','word_freq_over','word_freq_report']]
y= trein["ham"]


# In[ ]:


grid.fit(x,y)
grid.grid_scores_


# In[ ]:


grid.best_params_ 


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=11, p=1, weights= 'uniform' )
knn.fit(x,y)


# In[ ]:


xtest=test[['word_freq_remove','word_freq_free','word_freq_3d','word_freq_table','word_freq_000','word_freq_credit','word_freq_remove','word_freq_free','word_freq_3d','word_freq_table','word_freq_000','word_freq_addresses'
    ,'word_freq_money','word_freq_receive','word_freq_parts','word_freq_order', 'word_freq_internet','word_freq_over','word_freq_report']]


# In[ ]:


resposta5= knn.predict(xtest)


# In[ ]:


output = pd.DataFrame(test.Id)
output["ham"] = resposta5
df = pd.DataFrame (output)
df.to_csv("sub.csv", index=False)


# In[ ]:


from sklearn.metrics import confusion_matrix
ytest=clf.predict(x)
confusion_matrix(y, ytest)


# In[ ]:


from sklearn.naive_bayes import BernoulliNB


# In[ ]:


x=trein[['word_freq_remove','word_freq_free','word_freq_3d','word_freq_table','word_freq_000','word_freq_credit','word_freq_remove','word_freq_free','word_freq_3d','word_freq_table','word_freq_000','word_freq_addresses'
    ,'word_freq_money','word_freq_receive','word_freq_parts','word_freq_order', 'word_freq_internet','word_freq_over','word_freq_report']]
y= trein["ham"]
xtest=test[['word_freq_remove','word_freq_free','word_freq_3d','word_freq_table','word_freq_000','word_freq_credit','word_freq_remove','word_freq_free','word_freq_3d','word_freq_table','word_freq_000','word_freq_addresses'
    ,'word_freq_money','word_freq_receive','word_freq_parts','word_freq_order', 'word_freq_internet','word_freq_over','word_freq_report']]


# In[ ]:


clf = BernoulliNB().fit(x, y)
clf


# In[ ]:


y_pred = clf.predict(xtest)


# In[ ]:


from sklearn.metrics import confusion_matrix
resposta6=clf.predict(xtest)
confusion_matrix(y, ytest)


# In[ ]:


output = pd.DataFrame(test.Id)
output["ham"] = resposta6
df = pd.DataFrame (output)
df.to_csv("sub.csv", index=False)


# In[ ]:


resposta6


# In[ ]:


from sklearn import metrics


# In[ ]:


fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)


# In[ ]:




