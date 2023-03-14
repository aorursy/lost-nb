#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.model_selection import StratifiedKFold, GridSearchCV,                                    StratifiedShuffleSplit, learning_curve, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_auc_score
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')


path_train = r'/kaggle/input/unimelb/unimelb_training.csv'
path_test = r'/kaggle/input/unimelb/unimelb_test.csv'
train = pd.read_csv(path_train)
test = pd.read_csv(path_test)
train.head()




X = train.drop('Grant.Status', axis = 1)
print(X.shape)

y = train['Grant.Status']
print(y.shape)




#check NaN values and column types (object - categorical, float - numerical)
X.info(verbose=True,null_counts=True)




#check class distribution
print("Class 1 examples {:.2f}\nClass 0 examples {:.2f}".format(sum(y)/y.shape[0], 
                                                                1-sum(y)/y.shape[0]))




#separate X on X_real( numerical ), X_cat (categorical)
#and don't use one col with int dtype (object's number)

X_real = X.select_dtypes(include = 'float64')
print(X_real.shape)

X_cat = X.select_dtypes(exclude = ['float64','int64'])
print(X_cat.shape)




#count fully NaN columns
print(X_real.isna().all().sum())
print(X_cat.isna().all().sum())




#drop fully NaN col
X_real = X_real.dropna(how ='all', axis = 1)
print(X_real.shape)




#view percent of NaN
na_real = X_real.isna().sum()
print(na_real.shape)

na_real_percent = (X_real.isna().sum()/X_real.isna().count()*100)
print(na_real_percent.shape)

df_na_real = pd.concat([na_real,na_real_percent], axis = 1, keys = ['NaN_cnt','Percent'])

df_na_real = df_na_real.sort_values(by=['Percent'], ascending = False)
df_na_real.head(10)




print(" NaN value in train data less then 90% for:    {} real features".format(df_na_real[df_na_real.Percent <= 90].shape[0]))
print(" NaN value in train data less then 70% for:    {} real features".format(df_na_real[df_na_real.Percent <= 70].shape[0]))
print(" NaN value in train data less then 50% for:    {} real features".format(df_na_real[df_na_real.Percent <= 50].shape[0]))
print(" NaN value in train data less then 20% for:    {} real features".format(df_na_real[df_na_real.Percent <= 20].shape[0]))
print(" NaN value in train data less then 10% for:    {} real features".format(df_na_real[df_na_real.Percent <= 10].shape[0]))




real_features = df_na_real[df_na_real.Percent <= 20].index




na_cat = X_cat.isna().sum()

na_cat_percent = (X_cat.isna().sum()/X_cat.isna().count()*100)

df_na_cat = pd.concat([na_cat,na_cat_percent], axis = 1, keys = ['NaN_cnt','Percent'])                                        .sort_values(by=['Percent'], ascending = False)

df_na_cat.head(10)




print(" NaN value in train data less then 90% for:    {} cat features".format(df_na_cat[df_na_cat.Percent <= 90].shape[0]))
print(" NaN value in train data less then 80% for:    {} cat features".format(df_na_cat[df_na_cat.Percent <= 80].shape[0]))
print(" NaN value in train data less then 60% for:    {} cat features".format(df_na_cat[df_na_cat.Percent <= 60].shape[0]))
print(" NaN value in train data less then 20% for:    {} cat features".format(df_na_cat[df_na_cat.Percent <= 20].shape[0]))
print(" NaN value in train data less then 10% for:    {} cat features".format(df_na_cat[df_na_cat.Percent <= 10].shape[0]))




cat_features = df_na_cat[df_na_cat.Percent <= 80].index




print(train.shape)
print(test.shape)

train_test  = pd.concat( [train.drop('Grant.Status', axis = 1),
                          test.drop('Grant.Status', axis = 1)],
                        axis = 0)
print(train_test.shape)




#categorical features encode / transform NaN cat values to str type 'nan' before encoding
print(train_test[cat_features].shape)
train_test_cat = pd.get_dummies(train_test[cat_features].astype('str'))
print(train_test_cat.shape)




#add encoded cat features to real features / fill NaN real with mean
print(train_test[real_features].shape)
print(train_test_cat.shape)
train_test_processed = pd.concat([train_test[real_features].fillna(train_test[real_features].mean()),                                  train_test_cat], axis = 1)
print(train_test_processed.shape)




TRAIN = train_test_processed.iloc[:8708,:]
TEST = train_test_processed.iloc[8708:,:]
print("train shape before processing {} and after {}".format(train.shape, TRAIN.shape))
print("test shape before processing {} and after {}".format(test.shape, TEST.shape))




SSS = StratifiedShuffleSplit(n_splits=1,test_size=0.30, random_state=0)

X = TRAIN
# y = train['Grant.Status'] - was defined in the beginning
train_ind, test_ind = next(SSS.split(X,y))

sub_X_train = TRAIN.iloc[train_ind,:]
sub_y_train = y.iloc[train_ind]
sub_X_test = TRAIN.iloc[test_ind,:]
sub_y_test = y.iloc[test_ind]

print(sub_X_train.shape)
print(sub_y_train.shape)
print(sub_X_test.shape)
print(sub_y_test.shape)




clf = LogisticRegression(random_state=0, solver = 'liblinear', penalty = 'l2', n_jobs=-1)

param_grid = {'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10]}

grid_cv = GridSearchCV(clf, param_grid, cv=5)

grid_cv.fit(sub_X_train, sub_y_train)

print ("roc_auc with L2:", roc_auc_score(sub_y_test, grid_cv.predict_proba(sub_X_test).T[1]))




plt.hist(grid_cv.best_estimator_.coef_[0], bins = 20)
plt.xlabel('coef')
plt.ylabel('freq')
plt.grid()




clf = LogisticRegression(random_state=0, solver = 'liblinear', penalty = 'l1', n_jobs = -1)

param_grid = {'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10]}

grid_cv = GridSearchCV(clf, param_grid, cv=5)

grid_cv.fit(sub_X_train, sub_y_train)

print ("roc_auc with L1:", roc_auc_score(sub_y_test, grid_cv.predict_proba(sub_X_test).T[1]))




plt.hist(grid_cv.best_estimator_.coef_[0], bins = 20)
plt.xlabel('coef')
plt.ylabel('freq')
plt.grid()




sns.pairplot(sub_X_train[['Faculty.No..1', 'SEO.Code.4', 'Sponsor.Code_1A']])




print(sub_X_train.shape)
print(sub_X_test.shape)
print(TEST.shape)




#separate on real and cat features, because just real should be scaled

X_real_train = sub_X_train[real_features].values
X_cat_train = sub_X_train[train_test_cat.columns].values

X_test_test = sub_X_test[real_features].values
X_cat_test = sub_X_test[train_test_cat.columns].values

TEST_real = TEST[real_features].values
TEST_cat = TEST[train_test_cat.columns].values




#scaling

scaler = StandardScaler()

X_real_train_scaled = scaler.fit_transform(X_real_train)

X_real_test_scaled = scaler.fit_transform(X_test_test)

TEST_real_scaled = scaler.fit_transform(TEST_real)




#mergin scaled and cat features
sub_X_train_scaled = np.hstack((X_real_train_scaled,X_cat_train))
print(sub_X_train_scaled.shape)

sub_X_test_scaled = np.hstack((X_real_test_scaled,X_cat_test))
print(sub_X_test_scaled.shape)

TEST_scaled = np.hstack((TEST_real_scaled,TEST_cat))
print(TEST_scaled.shape)




clf = LogisticRegression(random_state=0, solver = 'liblinear', penalty = 'l2',n_jobs = -1)

param_grid = {'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10]}

grid_cv = GridSearchCV(clf, param_grid, cv=5)

grid_cv.fit(sub_X_train_scaled, sub_y_train)

print ("roc_auc with L2:", roc_auc_score(sub_y_test, grid_cv.predict_proba(sub_X_test_scaled).T[1]))




clf = LogisticRegression(random_state=0, solver = 'liblinear', penalty = 'l1')

param_grid = {'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10]}

grid_cv = GridSearchCV(clf, param_grid, cv=5)

grid_cv.fit(sub_X_train_scaled, sub_y_train)

print ("roc_auc with L1:", roc_auc_score(sub_y_test, grid_cv.predict_proba(sub_X_test_scaled).T[1]))




get_ipython().run_cell_magic('time', '', "train_sizes, train_scores, test_scores = learning_curve(grid_cv.best_estimator_, sub_X_train, sub_y_train, \n                                                                       train_sizes=np.arange(0.1,1., 0.2), \n                                                                       cv=3, scoring='accuracy', n_jobs=-1)")




plt.grid(True)
plt.plot(train_sizes, train_scores.mean(axis = 1), 'g-', marker='o', label='train')
plt.plot(train_sizes, test_scores.mean(axis = 1), 'r-', marker='o', label='test')
plt.ylim((0.0, 1.05))
plt.legend(loc='lower right')
plt.xlabel('number of objects in Train')
plt.ylabel('accuracy')
plt.title('Lerning curve LR')
plt.show()




clf = RandomForestClassifier(random_state = 0, n_jobs=-1)

param_grid ={'n_estimators':[50,100,200,500],
             'max_depth':[10,30,50,100]}

grid_cv = GridSearchCV(clf, param_grid, cv=3)

grid_cv.fit(sub_X_train, sub_y_train)

print('roc_auc with RF: {}'.format(roc_auc_score(sub_y_test, grid_cv.predict_proba(sub_X_test).T[1])))




fig = plt.figure(figsize=(10,5))
scores = []

for i in range(0, len(grid_cv.cv_results_['mean_test_score']), len(param_grid['max_depth'])):
    scores.append(grid_cv.cv_results_['mean_test_score'][i:i+4])

for scr in scores:
    plt.plot(param_grid['n_estimators'], scr )

plt.legend(list(map(lambda x: 'max depth ' + str(x), param_grid['max_depth'])), loc = 5)
plt.xlabel('n_estimator')
plt.ylabel('accuracy_score')
plt.title('accuracy vs n_estimator for diff max_depth')
plt.grid()




train_sizes, train_scores, test_scores = learning_curve(grid_cv.best_estimator_, sub_X_train, sub_y_train, 
                                                                       train_sizes=np.arange(0.1,1., 0.2), 
                                                                       cv=3, scoring='accuracy', n_jobs=-1)




plt.grid(True)
plt.plot(train_sizes, train_scores.mean(axis = 1), 'g-', marker='o', label='train')
plt.plot(train_sizes, test_scores.mean(axis = 1), 'r-', marker='o', label='test')
plt.ylim((0.0, 1.05))
plt.legend(loc='lower right')
plt.xlabel('number of objects in Train')
plt.ylabel('accuracy')
plt.title('Lerning curve RF')
plt.show()




clf = XGBClassifier(random_state = 0)

param_grid ={'n_estimators':[50,100,200,300],
             'max_depth':[2,6,10,12]}

grid_cv = RandomizedSearchCV(clf, param_grid, cv=3)

grid_cv.fit(sub_X_train, sub_y_train)

print('roc_auc with RF: {}'.format(roc_auc_score(sub_y_test, grid_cv.predict_proba(sub_X_test).T[1])))




get_ipython().run_cell_magic('time', '', "train_sizes, train_scores, test_scores = learning_curve(grid_cv.best_estimator_, sub_X_train, sub_y_train, \n                                                                       train_sizes=np.arange(0.1,1., 0.2), \n                                                                       cv=5, scoring='accuracy', n_jobs=-1)")




plt.grid(True)
plt.plot(train_sizes, train_scores.mean(axis = 1), 'g-', marker='o', label='train')
plt.plot(train_sizes, test_scores.mean(axis = 1), 'r-', marker='o', label='test')
plt.ylim((0.0, 1.05))
plt.legend(loc='lower right')
plt.xlabel('number of objects in Train')
plt.ylabel('accuracy')
plt.title('Lerning curve XGB')
plt.show()

