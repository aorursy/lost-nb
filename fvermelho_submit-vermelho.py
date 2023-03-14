#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')




kaggledf = pd.read_csv("../input/train_data.csv")




kaggledf.shape




kaggledf.head()




kaggledf.describe




kaggledf.shape




kaggledf = kaggledf.dropna(subset=['default'])




kaggledf.shape




kaggledf.ftypes




for column in kaggledf:
    if column != 'ids':
        if kaggledf[column].dtype == np.float64:
            print(column, ' - Numeric - MISSING = ', kaggledf.loc[kaggledf[column].isnull(),['ids']].count())
        else:
            print(column, ' - String - MISSING = ', kaggledf.loc[kaggledf[column].isna(),['ids']].count())




kaggledfnonum = kaggledf.select_dtypes(exclude='float64')




for column in kaggledfnonum:
    if column != 'ids':
        print('\n', kaggledfnonum.groupby(column).size())




from sklearn.model_selection import train_test_split




y = kaggledf.default #y recebe target




vars_num = ['score_3', 'score_4', 'score_5', 'score_6', 'risk_rate', 'amount_borrowed', 
            'borrowed_in_months', 'credit_limit', 'income', 'ok_since', 'n_bankruptcies', 
            'n_defaulted_loans', 'n_accounts', 'n_issues']
vars_obj = ['score_1', 'score_2', 'reason', 'gender', 'facebook_profile', 'state', 
            'zip', 'state', 'job_name', 'real_state']




X = pd.concat([kaggledf[vars_num]], axis=1) 




from sklearn.preprocessing import OneHotEncoder, LabelEncoder




le = LabelEncoder()




vars_obj_nomiss = ['score_1', 'score_2', 'state', 'zip', 'real_state']
vars_obj_miss = ['reason', 'gender', 'facebook_profile', 'job_name']




X.shape




X = pd.concat([X, kaggledf[vars_obj_nomiss]], axis=1)




for var in vars_obj_nomiss:
    le.fit(kaggledf[var])
    X[var] = le.transform(kaggledf[var])




X.shape




X[vars_obj_nomiss].head()




kaggledf[kaggledf['reason'].isna()] #lines 14097 19623 19947 50929




kaggledf.loc[[14097,19623,19947,50929],['reason']]




kaggledf[['reason']].sample(5)




import random




kaggledf['reason'][kaggledf['reason'].isna()] = random.choice(kaggledf[kaggledf['reason'] != ""]['reason'])




kaggledf.loc[[14097,19623,19947,50929,2,65,71,113,131,184,188,193,208],['gender']]




#for line in kaggledf['ids'][kaggledf['gender'].isna()]:
#    kaggledf['gender'][kaggledf['ids'] == line] = random.choice(['m', 'f'])
#ISSO FICOU EXTREMAMENTE LENTO




kaggledf['gender'] = kaggledf['gender'].fillna(method = "bfill") #usando Backfill
kaggledf['facebook_profile'] = kaggledf['facebook_profile'].fillna(method = "bfill") #usando Backfill
kaggledf['job_name'] = kaggledf['job_name'].fillna(value = "99999") #usando Backfill




from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='median')




imp.fit(kaggledf[['credit_limit']])




imp.transform(kaggledf[['credit_limit']])




kaggledf[['credit_limit']] 




kaggledf['credit_limit'] = imp.transform(kaggledf[['credit_limit']]) #Imput da Mediana em credit_limit




for i in ['credit_limit', 'ok_since', 'n_bankruptcies', 'n_defaulted_loans', 'n_issues']:
    imp.fit(kaggledf[[i]])
    kaggledf[i] = imp.transform(kaggledf[[i]]) #Imput da Mediana em cada uma var do loop




for column in kaggledf:
    if column != 'ids':
        if kaggledf[column].dtype == np.float64:
            print(column, ' - Numeric - MISSING = ', kaggledf.loc[kaggledf[column].isna(),['ids']].count())
        else:
            print(column, ' - String - MISSING = ', kaggledf.loc[kaggledf[column].isna(),['ids']].count())




X.dtypes




X.shape




kaggledf.shape




vars_mediana = ['credit_limit', 'ok_since', 'n_bankruptcies', 'n_defaulted_loans', 'n_issues']




for i in vars_mediana:
    X[i] = kaggledf[i]




X = pd.concat([X, kaggledf['gender']], axis=1)
X = pd.concat([X, kaggledf['facebook_profile']], axis=1)
X = pd.concat([X, kaggledf['reason']], axis=1)




le.fit(kaggledf['gender'])
X['gender'] = le.transform(kaggledf['gender'])

le.fit(kaggledf['facebook_profile'])
X['facebook_profile'] = le.transform(kaggledf['facebook_profile'])

le.fit(kaggledf['reason'])
X['reason'] = le.transform(kaggledf['reason'])




X.shape




kaggledf.dtypes




X.head()




X.dtypes




ohe = OneHotEncoder(categorical_features=[14,15,16,17,18,19,20,21])




ohe.fit(X[['score_3', 'score_4', 'score_5', 'score_6', 'risk_rate', 'amount_borrowed', 
                'borrowed_in_months', 'credit_limit', 'income', 'ok_since', 'n_bankruptcies', 
                'n_defaulted_loans', 'n_accounts', 'n_issues', 'score_1', 'score_2', 
                'state', 'zip', 'real_state', 'gender', 'facebook_profile', 'reason']])




X = ohe.transform(X[['score_3', 'score_4', 'score_5', 'score_6', 'risk_rate', 'amount_borrowed', 
                'borrowed_in_months', 'credit_limit', 'income', 'ok_since', 'n_bankruptcies', 
                'n_defaulted_loans', 'n_accounts', 'n_issues', 'score_1', 'score_2', 
                'state', 'zip', 'real_state', 'gender', 'facebook_profile', 'reason']])




X














from sklearn.model_selection import train_test_split




le.fit(y)
y = le.transform(y)




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)




print(X_train.shape)
print(X_test.shape)




from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV




lo = LogisticRegression(C=10)
lo.fit(X_train, y_train)
predictions = lo.predict(X_test)
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, predictions)
print("accuracy_score: %.4f" % acc)




tuned_parameters = [{'C': [0.1,1,10]}]

clf = GridSearchCV(LogisticRegression(), tuned_parameters, cv=3, scoring='accuracy')
clf.fit(X_train, y_train)




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




from sklearn.tree import DecisionTreeClassifier




dt = DecisionTreeClassifier(max_depth=3)
dt.fit(X_train, y_train)
predictions = dt.predict(X_test)
acc = accuracy_score(y_test, predictions)
print("accuracy_score: %.4f" % acc)




tuned_parameters = [{'max_depth': [1,3,5,10,20],
                     'min_samples_split': [3,5,10]}]

clf = GridSearchCV(DecisionTreeClassifier(), tuned_parameters, cv=3, scoring='accuracy')
clf.fit(X_train, y_train)




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




from sklearn.ensemble import RandomForestClassifier




rf = RandomForestClassifier(n_estimators=100, max_depth=3, max_features=5)
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)
acc = accuracy_score(y_test, predictions)
print("accuracy_score: %.4f" % acc)




tuned_parameters = [{'n_estimators': [100,150],
                     'max_depth': [6,9,12,20],
                     'max_features': [9,12,16]}]

clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=10, scoring='accuracy', n_jobs=8)
clf.fit(X_train, y_train)




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




submitdf = pd.read_csv("../input/teste_data.csv")




submitdf.dtypes




for column in submitdf:
    if column != 'ids':
        if submitdf[column].dtype == np.float64:
            print(column, ' - Numeric - MISSING = ', submitdf.loc[submitdf[column].isna(),['ids']].count())
        else:
            print(column, ' - String - MISSING = ', submitdf.loc[submitdf[column].isna(),['ids']].count())




vars = ['score_3', 'score_4', 'score_5', 'score_6', 'risk_rate', 'amount_borrowed', 
        'borrowed_in_months', 'credit_limit', 'income', 'ok_since', 'n_bankruptcies', 
        'n_defaulted_loans', 'n_accounts', 'n_issues', 'score_1', 'score_2', 'state', 'zip', 
        'real_state', 'gender', 'facebook_profile', 'reason']




val = pd.concat([submitdf[vars]], axis=1)




val.dtypes




X.dtypes




vars_obj_nomiss = ['score_1', 'score_2', 'state', 'zip', 'real_state']
for var in vars_obj_nomiss:
    le.fit(submitdf[var])
    val[var] = le.transform(submitdf[var])

val['reason'][val['reason'].isna()] = random.choice(submitdf[submitdf['reason'] != ""]['reason'])


val['gender'] = submitdf['gender'].fillna(method = "bfill") #usando Backfill
val['facebook_profile'] = submitdf['facebook_profile'].fillna(method = "bfill") #usando Backfill




from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='median')
for i in ['credit_limit', 'ok_since', 'n_bankruptcies', 'n_defaulted_loans', 'n_issues']:
    imp.fit(submitdf[[i]])
    val[i] = imp.transform(submitdf[[i]]) #Imput da Mediana em cada uma var do loop




le.fit(val['gender'])
val['gender'] = le.transform(val['gender'])

le.fit(val['facebook_profile'])
val['facebook_profile'] = le.transform(val['facebook_profile'])

le.fit(val['reason'])
val['reason'] = le.transform(val['reason'])




val.dtypes




dt = DecisionTreeClassifier(max_depth=3)
dt.fit(X_train, y_train)




predictions = dt.predict_proba(val)




predictions




predictionsdf = pd.DataFrame(data=predictions)




predictionsdf.head()









to_submit = pd.concat([submitdf.ids, predictionsdf[1]], axis=1)




to_submit.to_csv('submit0004.csv')
















