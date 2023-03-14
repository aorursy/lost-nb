#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from category_encoders import LeaveOneOutEncoder
from sklearn.preprocessing import MinMaxScaler,StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier, LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
from mlxtend.classifier import StackingCVClassifier
import copy




train = pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv", index_col=['id'])
test = pd.read_csv("/kaggle/input/cat-in-the-dat/test.csv", index_col=['id'])




train.dtypes




display(train.head())




X = train.drop("target", axis = 1)
y = train.loc[:,"target"]




X.bin_3 = X.bin_3.apply(lambda x: 1 if x == "T" else 0)
X.bin_4 = X.bin_4.apply(lambda x: 1 if x == "Y" else 0)

print(X.columns)




# h = FeatureHasher(input_type='string', n_features=1000)
# X[['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']].values
# hash_X = h.fit_transform(X[['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']].values)
# hash_X = pd.DataFrame(hash_X.toarray())

# hash_X.columns
# X = X.drop(["nom_5", "nom_6", "nom_7", "nom_8", "nom_9"], axis=1).join(hash_X)

loo_encoder = LeaveOneOutEncoder(cols=["nom_5", "nom_6", "nom_7", "nom_8", "nom_9"])
loo_X = loo_encoder.fit_transform(X[["nom_5", "nom_6", "nom_7", "nom_8", "nom_9"]], y)
X = X.drop(["nom_5", "nom_6", "nom_7", "nom_8", "nom_9"], axis=1).join(loo_X)

X = X.drop(["nom_0", "nom_1", "nom_2", "nom_3", "nom_4"], axis=1)         .join(pd.get_dummies(X[["nom_0", "nom_1", "nom_2", "nom_3", "nom_4"]]))

print(X.columns)




X.ord_1.replace(to_replace = ['Novice', 'Contributor','Expert', 'Master', 'Grandmaster'],
                         value = [0, 1, 2, 3, 4], inplace = True)
X.ord_2.replace(to_replace = ['Freezing', 'Cold', 'Warm', 'Hot','Boiling Hot', 'Lava Hot'],
                         value = [0, 1, 2, 3, 4, 5], inplace = True)

for i in ["ord_3", "ord_4"]:
   le = LabelEncoder()
   X[[i]] = le.fit_transform(X[[i]])

oe = OrdinalEncoder(categories='auto')
X.ord_5 = oe.fit_transform(X.ord_5.values.reshape(-1,1))

print(X.columns)




def date_cyc_enc(df, col, max_vals):
   df[col + '_sin'] = np.sin(2 * np.pi * df[col]/max_vals)
   df[col + '_cos'] = np.cos(2 * np.pi * df[col]/max_vals)
   return df

X = date_cyc_enc(X, 'day', 7)
X = date_cyc_enc(X, 'month', 12)
X.drop(['day', 'month'], axis=1, inplace = True)

print(X.columns)




# lr = LogisticRegression()
# scores_lr = cross_val_score(lr, X, y, cv=5, n_jobs=-1)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores_lr.mean(), scores_lr.std() * 2))




# rc = RidgeClassifier()
# scores_rc = cross_val_score(rc, X, y, cv=5, n_jobs=-1)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores_rc.mean(), scores_rc.std() * 2))




# lda = LinearDiscriminantAnalysis()
# scores_lda = cross_val_score(lda, X_new, y, cv=5, n_jobs=1)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores_lda.mean(), scores_lda.std() * 2))




# linear_svm = LinearSVC(penalty="l2")
# scores_linear_svm = cross_val_score(linear_svm, X, y, cv=5, n_jobs=-1)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores_linear_svm.mean(), scores_linear_svm.std() * 2))




# fr = DecisionTreeClassifier(random_state=0)
# scores_dt = cross_val_score(fr, X, y, cv=5, n_jobs=2)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores_dt.mean(), scores_dt.std() * 2))




# sgdc = SGDClassifier()
# scores_sgdc = cross_val_score(sgdc, X, y, cv=5, n_jobs=4)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores_sgdc.mean(), scores_sgdc.std() * 2))




# ab = AdaBoostClassifier()
# scores_ab= cross_val_score(ab, X, y, cv=5, n_jobs=-1)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores_ab.mean(), scores_ab.std() * 2))




# gbm = GradientBoostingClassifier()
# scores_gbm= cross_val_score(gbm, X, y, cv=5, n_jobs=-1)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores_gbm.mean(), scores_gbm.std() * 2))




# rf = RandomForestClassifier()
# scores_rf= cross_val_score(rf, X, y, cv=5, n_jobs=-1)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores_rf.mean(), scores_rf.std() * 2))




# et = ExtraTreesClassifier()
# scores_et= cross_val_score(et, X, y, cv=5, n_jobs=-1)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores_et.mean(), scores_et.std() * 2))




# xgb = XGBClassifier()
# scores_xgb= cross_val_score(xgb, X, y, cv=5, n_jobs=-1)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores_xgb.mean(), scores_xgb.std() * 2))




# params = {
#         'min_child_weight': [1, 5, 10, 13, 15],
#         'gamma': [0.5, 1, 1.5, 2, 5],
#         'subsample': [0.2, 0.4, 0.6, 0.8, 1.0],
#         'colsample_bytree': [0.6, 0.8, 1.0],
#         'max_depth': [3, 4, 5, 10, 20]
#         }
# xgb = XGBClassifier(silent=True, nthread=1)
# folds = 3
# param_comb = 5

# skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

# random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, 
#                                    scoring='accuracy', n_jobs=-1, cv=skf.split(X, y), 
#                                    verbose=3, random_state=1001 )
# random_search.fit(X, y)

# print('\n All results:')
# print(random_search.cv_results_)
# print('\n Best estimator:')
# print(random_search.best_estimator_)
# print('\n Best hyperparameters:')
# print(random_search.best_params_)




# means = random_search.cv_results_['mean_test_score']
# stds = random_search.cv_results_['std_test_score']
# for mean, std, params in zip(means, stds, random_search.cv_results_['params']):
#     print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))




# params2 = {
#         'n_estimators': [50, 100, 300, 800],
#         'learning_rate': [0.01, 0.1, 0.5, 1],
#         'max_depth': [3, 10, 20, 50],
#         'min_samples_split': [100, 200, 500, 800],
#         'subsample': [0.2, 0.4, 0.6, 0.8, 1.0]
#         }
# gbm = GradientBoostingClassifier(random_state=1001)

# random_search2 = RandomizedSearchCV(gbm, param_distributions=params2, n_iter=param_comb, 
#                                    scoring='accuracy', n_jobs=-1, cv=skf.split(X, y), 
#                                    verbose=3, random_state=1001 )
# random_search2.fit(X, y)

# print('\n All results:')
# print(random_search2.cv_results_)
# print('\n Best estimator:')
# print(random_search2.best_estimator_)
# print('\n Best hyperparameters:')
# print(random_search2.best_params_)




# means = random_search2.cv_results_['mean_test_score']
# stds = random_search2.cv_results_['std_test_score']
# for mean, std, params in zip(means, stds, random_search2.cv_results_['params']):
#     print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))




# lr_params = {'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'], 
#              'C': [0.01, 0.1, 0.5, 1]
#             }
# lr = LogisticRegression(random_state=1001)

# random_search3 = RandomizedSearchCV(lr, param_distributions=lr_params, n_iter=param_comb, 
#                                    scoring='accuracy', n_jobs=-1, cv=skf.split(X, y), 
#                                    verbose=3, random_state=1001 )
# random_search3.fit(X, y)

# print('\n All results:')
# print(random_search3.cv_results_)
# print('\n Best estimator:')
# print(random_search3.best_estimator_)
# print('\n Best hyperparameters:')
# print(random_search3.best_params_)




# means = random_search3.cv_results_['mean_test_score']
# stds = random_search3.cv_results_['std_test_score']
# for mean, std, params in zip(means, stds, random_search3.cv_results_['params']):
#     print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))




# xgb_clf = XGBClassifier(booster='gbtree', gamma=5, colsample_bytree=0.8,
#                         learning_rate=0.1, max_depth=10, 
#                         min_child_weight=10, n_estimators=100, 
#                         silent=True, subsample=0.8)

# ab_clf = AdaBoostClassifier(n_estimators=200,
#                             base_estimator=DecisionTreeClassifier(
#                                 min_samples_leaf=2,
#                                 random_state=1001),
#                             random_state=1001)

# gbm_clf = GradientBoostingClassifier(n_estimators=300, min_samples_split=100,
#                                  max_depth=50, learning_rate=1, subsample=0.8,
#                                  random_state=1001)

# lr = LogisticRegression()

# stack = StackingCVClassifier(classifiers=[xgb_clf, gbm_clf, ab_clf], 
#                             meta_classifier=lr,
#                             cv=5,
#                             stratify=True,
#                             shuffle=True,
#                             use_probas=True,
#                             use_features_in_secondary=True,
#                             verbose=1,
#                             random_state=1001,
#                             n_jobs=-1)
# stack = stack.fit(X, y)




X_train = train.drop("target", axis = 1)
y_train = train.loc[:,"target"]




X_train.bin_3 = X_train.bin_3.apply(lambda x: 1 if x == "T" else 0)
X_train.bin_4 = X_train.bin_4.apply(lambda x: 1 if x == "Y" else 0)

X_train.ord_1.replace(to_replace = ['Novice', 'Contributor','Expert', 'Master', 'Grandmaster'],
                         value = [0, 1, 2, 3, 4], inplace = True)
X_train.ord_2.replace(to_replace = ['Freezing', 'Cold', 'Warm', 'Hot','Boiling Hot', 'Lava Hot'],
                         value = [0, 1, 2, 3, 4, 5], inplace = True)

for i in ["ord_3", "ord_4"]:
   le = LabelEncoder()
   X_train[[i]] = le.fit_transform(X_train[[i]])

oe = OrdinalEncoder(categories='auto')
X_train.ord_5 = oe.fit_transform(X_train.ord_5.values.reshape(-1,1))




X_test = copy.deepcopy(test)




X_test.bin_3 = X_test.bin_3.apply(lambda x: 1 if x == "T" else 0)
X_test.bin_4 = X_test.bin_4.apply(lambda x: 1 if x == "Y" else 0)

X_test.ord_1.replace(to_replace = ['Novice', 'Contributor','Expert', 'Master', 'Grandmaster'],
                         value = [0, 1, 2, 3, 4], inplace = True)
X_test.ord_2.replace(to_replace = ['Freezing', 'Cold', 'Warm', 'Hot','Boiling Hot', 'Lava Hot'],
                         value = [0, 1, 2, 3, 4, 5], inplace = True)

for i in ["ord_3", "ord_4"]:
    le = LabelEncoder()
    X_test[[i]] = le.fit_transform(X_test[[i]])

oe = OrdinalEncoder(categories='auto')
X_test.ord_5 = oe.fit_transform(X_test.ord_5.values.reshape(-1,1))




data = pd.concat([X_train, X_test])
print(data.shape)




columns = data.columns
dummies = pd.get_dummies(data,
                         columns=columns,
#                          drop_first=True,
                         sparse=True)




print(dummies.shape)
print(X_train.shape[0])




X_train = dummies.iloc[:X_train.shape[0], :]
X_test = dummies.iloc[X_train.shape[0]:, :]




del dummies
del data
print (X_train.shape)
print(X_test.shape)




X_train = X_train.sparse.to_coo().tocsr()
X_test = X_test.sparse.to_coo().tocsr()

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")




lr = LogisticRegression()
lr.fit(X_train, y_train)
pred = lr.predict_proba(X_test)
pred[:10,1]




# lr = LogisticRegression(solver="lbfgs", C=0.1, max_iter=10000)
# lr.fit(X_train, y_train)
# pred2 = lr.predict_proba(X_test)
# pred2[:10,1]




predictions = pd.Series(pred[:,1], index=test.index, dtype=y.dtype)
predictions.to_csv("./submission.csv", header=['target'], index=True, index_label='id')

