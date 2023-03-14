#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


kaggledf = pd.read_csv("../input/train_data.csv")


# In[ ]:


kaggledf.shape


# In[ ]:


kaggledf.head()


# In[ ]:


kaggledf.describe


# In[ ]:


kaggledf.shape


# In[ ]:


kaggledf = kaggledf.dropna(subset=['default'])


# In[ ]:


kaggledf.shape


# In[ ]:


kaggledf.ftypes


# In[ ]:


for column in kaggledf:
    if column != 'ids':
        if kaggledf[column].dtype == np.float64:
            print(column, ' - Numeric - MISSING = ', kaggledf.loc[kaggledf[column].isnull(),['ids']].count())
        else:
            print(column, ' - String - MISSING = ', kaggledf.loc[kaggledf[column].isna(),['ids']].count())


# In[ ]:


kaggledfnonum = kaggledf.select_dtypes(exclude='float64')


# In[ ]:


for column in kaggledfnonum:
    if column != 'ids':
        print('\n', kaggledfnonum.groupby(column).size())


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


y = kaggledf.default #y recebe target


# In[ ]:


vars_num = ['score_3', 'score_4', 'score_5', 'score_6', 'risk_rate', 'amount_borrowed', 
            'borrowed_in_months', 'credit_limit', 'income', 'ok_since', 'n_bankruptcies', 
            'n_defaulted_loans', 'n_accounts', 'n_issues']
vars_obj = ['score_1', 'score_2', 'reason', 'gender', 'facebook_profile', 'state', 
            'zip', 'state', 'job_name', 'real_state']


# In[ ]:


X = pd.concat([kaggledf[vars_num]], axis=1) 


# In[ ]:


from sklearn.preprocessing import OneHotEncoder, LabelEncoder


# In[ ]:


le = LabelEncoder()


# In[ ]:


vars_obj_nomiss = ['score_1', 'score_2', 'state', 'zip', 'real_state']
vars_obj_miss = ['reason', 'gender', 'facebook_profile', 'job_name']


# In[ ]:


X.shape


# In[ ]:


X = pd.concat([X, kaggledf[vars_obj_nomiss]], axis=1)


# In[ ]:


for var in vars_obj_nomiss:
    le.fit(kaggledf[var])
    X[var] = le.transform(kaggledf[var])


# In[ ]:


X.shape


# In[ ]:


X[vars_obj_nomiss].head()


# In[ ]:


kaggledf[kaggledf['reason'].isna()] #lines 14097 19623 19947 50929


# In[ ]:


kaggledf.loc[[14097,19623,19947,50929],['reason']]


# In[ ]:


kaggledf[['reason']].sample(5)


# In[ ]:


import random


# In[ ]:


kaggledf['reason'][kaggledf['reason'].isna()] = random.choice(kaggledf[kaggledf['reason'] != ""]['reason'])


# In[ ]:


kaggledf.loc[[14097,19623,19947,50929,2,65,71,113,131,184,188,193,208],['gender']]


# In[ ]:


#for line in kaggledf['ids'][kaggledf['gender'].isna()]:
#    kaggledf['gender'][kaggledf['ids'] == line] = random.choice(['m', 'f'])
#ISSO FICOU EXTREMAMENTE LENTO


# In[ ]:


kaggledf['gender'] = kaggledf['gender'].fillna(method = "bfill") #usando Backfill
kaggledf['facebook_profile'] = kaggledf['facebook_profile'].fillna(method = "bfill") #usando Backfill
kaggledf['job_name'] = kaggledf['job_name'].fillna(value = "99999") #usando Backfill


# In[ ]:


from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='median')


# In[ ]:


imp.fit(kaggledf[['credit_limit']])


# In[ ]:


imp.transform(kaggledf[['credit_limit']])


# In[ ]:


kaggledf[['credit_limit']] 


# In[ ]:


kaggledf['credit_limit'] = imp.transform(kaggledf[['credit_limit']]) #Imput da Mediana em credit_limit


# In[ ]:


for i in ['credit_limit', 'ok_since', 'n_bankruptcies', 'n_defaulted_loans', 'n_issues']:
    imp.fit(kaggledf[[i]])
    kaggledf[i] = imp.transform(kaggledf[[i]]) #Imput da Mediana em cada uma var do loop


# In[ ]:


for column in kaggledf:
    if column != 'ids':
        if kaggledf[column].dtype == np.float64:
            print(column, ' - Numeric - MISSING = ', kaggledf.loc[kaggledf[column].isna(),['ids']].count())
        else:
            print(column, ' - String - MISSING = ', kaggledf.loc[kaggledf[column].isna(),['ids']].count())


# In[ ]:


X.dtypes


# In[ ]:


X.shape


# In[ ]:


kaggledf.shape


# In[ ]:


vars_mediana = ['credit_limit', 'ok_since', 'n_bankruptcies', 'n_defaulted_loans', 'n_issues']


# In[ ]:


for i in vars_mediana:
    X[i] = kaggledf[i]


# In[ ]:


X = pd.concat([X, kaggledf['gender']], axis=1)
X = pd.concat([X, kaggledf['facebook_profile']], axis=1)
X = pd.concat([X, kaggledf['reason']], axis=1)


# In[ ]:


le.fit(kaggledf['gender'])
X['gender'] = le.transform(kaggledf['gender'])

le.fit(kaggledf['facebook_profile'])
X['facebook_profile'] = le.transform(kaggledf['facebook_profile'])

le.fit(kaggledf['reason'])
X['reason'] = le.transform(kaggledf['reason'])


# In[ ]:


X.shape


# In[ ]:


kaggledf.dtypes


# In[ ]:


X.head()


# In[ ]:


X.dtypes


# In[ ]:


ohe = OneHotEncoder(categorical_features=[14,15,16,17,18,19,20,21])


# In[ ]:


ohe.fit(X[['score_3', 'score_4', 'score_5', 'score_6', 'risk_rate', 'amount_borrowed', 
                'borrowed_in_months', 'credit_limit', 'income', 'ok_since', 'n_bankruptcies', 
                'n_defaulted_loans', 'n_accounts', 'n_issues', 'score_1', 'score_2', 
                'state', 'zip', 'real_state', 'gender', 'facebook_profile', 'reason']])


# In[ ]:


X = ohe.transform(X[['score_3', 'score_4', 'score_5', 'score_6', 'risk_rate', 'amount_borrowed', 
                'borrowed_in_months', 'credit_limit', 'income', 'ok_since', 'n_bankruptcies', 
                'n_defaulted_loans', 'n_accounts', 'n_issues', 'score_1', 'score_2', 
                'state', 'zip', 'real_state', 'gender', 'facebook_profile', 'reason']])


# In[ ]:


X


# In[ ]:





# In[ ]:





# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


le.fit(y)
y = le.transform(y)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[ ]:


print(X_train.shape)
print(X_test.shape)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


# In[ ]:


lo = LogisticRegression(C=10)
lo.fit(X_train, y_train)
predictions = lo.predict(X_test)
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, predictions)
print("accuracy_score: %.4f" % acc)


# In[ ]:


tuned_parameters = [{'C': [0.1,1,10]}]

clf = GridSearchCV(LogisticRegression(), tuned_parameters, cv=3, scoring='accuracy')
clf.fit(X_train, y_train)


# In[ ]:


print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
        % (mean, std * 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, clf.predict(X_test))
print("Accuracy: %.4f" % acc)
print()


# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


dt = DecisionTreeClassifier(max_depth=3)
dt.fit(X_train, y_train)
predictions = dt.predict(X_test)
acc = accuracy_score(y_test, predictions)
print("accuracy_score: %.4f" % acc)


# In[ ]:


tuned_parameters = [{'max_depth': [1,3,5,10,20],
                     'min_samples_split': [3,5,10]}]

clf = GridSearchCV(DecisionTreeClassifier(), tuned_parameters, cv=3, scoring='accuracy')
clf.fit(X_train, y_train)


# In[ ]:


print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
        % (mean, std * 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, clf.predict(X_test))
print("Accuracy: %.4f" % acc)
print()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rf = RandomForestClassifier(n_estimators=100, max_depth=3, max_features=5)
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)
acc = accuracy_score(y_test, predictions)
print("accuracy_score: %.4f" % acc)


# In[ ]:


tuned_parameters = [{'n_estimators': [100,150],
                     'max_depth': [6,9,12,20],
                     'max_features': [9,12,16]}]

clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=10, scoring='accuracy', n_jobs=8)
clf.fit(X_train, y_train)


# In[ ]:


print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
        % (mean, std * 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, clf.predict(X_test))
print("Accuracy: %.4f" % acc)
print()


# In[ ]:


submitdf = pd.read_csv("../input/teste_data.csv")


# In[ ]:


submitdf.dtypes


# In[ ]:


for column in submitdf:
    if column != 'ids':
        if submitdf[column].dtype == np.float64:
            print(column, ' - Numeric - MISSING = ', submitdf.loc[submitdf[column].isna(),['ids']].count())
        else:
            print(column, ' - String - MISSING = ', submitdf.loc[submitdf[column].isna(),['ids']].count())


# In[ ]:


vars = ['score_3', 'score_4', 'score_5', 'score_6', 'risk_rate', 'amount_borrowed', 
        'borrowed_in_months', 'credit_limit', 'income', 'ok_since', 'n_bankruptcies', 
        'n_defaulted_loans', 'n_accounts', 'n_issues', 'score_1', 'score_2', 'state', 'zip', 
        'real_state', 'gender', 'facebook_profile', 'reason']


# In[ ]:


val = pd.concat([submitdf[vars]], axis=1)


# In[ ]:


val.dtypes


# In[ ]:


X.dtypes


# In[ ]:


vars_obj_nomiss = ['score_1', 'score_2', 'state', 'zip', 'real_state']
for var in vars_obj_nomiss:
    le.fit(submitdf[var])
    val[var] = le.transform(submitdf[var])

val['reason'][val['reason'].isna()] = random.choice(submitdf[submitdf['reason'] != ""]['reason'])


val['gender'] = submitdf['gender'].fillna(method = "bfill") #usando Backfill
val['facebook_profile'] = submitdf['facebook_profile'].fillna(method = "bfill") #usando Backfill


# In[ ]:


from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='median')
for i in ['credit_limit', 'ok_since', 'n_bankruptcies', 'n_defaulted_loans', 'n_issues']:
    imp.fit(submitdf[[i]])
    val[i] = imp.transform(submitdf[[i]]) #Imput da Mediana em cada uma var do loop


# In[ ]:


le.fit(val['gender'])
val['gender'] = le.transform(val['gender'])

le.fit(val['facebook_profile'])
val['facebook_profile'] = le.transform(val['facebook_profile'])

le.fit(val['reason'])
val['reason'] = le.transform(val['reason'])


# In[ ]:


val.dtypes


# In[ ]:


dt = DecisionTreeClassifier(max_depth=3)
dt.fit(X_train, y_train)


# In[ ]:


predictions = dt.predict_proba(val)


# In[ ]:


predictions


# In[ ]:


predictionsdf = pd.DataFrame(data=predictions)


# In[ ]:


predictionsdf.head()


# In[ ]:





# In[ ]:


to_submit = pd.concat([submitdf.ids, predictionsdf[1]], axis=1)


# In[ ]:


to_submit.to_csv('submit0004.csv')


# In[ ]:





# In[ ]:





# In[ ]:




