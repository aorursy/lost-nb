#!/usr/bin/env python
# coding: utf-8



import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.utils import resample
import scipy
from scipy.stats import chisquare
from scipy import stats
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,roc_auc_score
import warnings
warnings.filterwarnings(action='ignore')
pd.set_option('display.max_columns',300)




train = pd.read_csv('../input/train.csv',index_col='ID')




test = pd.read_csv('../input/test.csv',index_col='ID')




train.head()




test.head()




train.shape, test.shape




test['TARGET']=0




test.shape




df = train.append(test)




df.shape




constant_cols=df.nunique()[df.nunique()==1].index




df.drop(constant_cols,axis=1,inplace=True)




df.shape




def getDuplicateColumns(df):
    '''
    Get a list of duplicate columns.
    It will iterate over all the columns in dataframe and find the columns whose contents are duplicate.
    :param df: Dataframe object
    :return: List of columns whose contents are duplicates.
    '''
    duplicateColumnNames = set()
    # Iterate over all the columns in dataframe
    for x in range(df.shape[1]):
        # Select column at xth index.
        col = df.iloc[:, x]
        # Iterate over all the columns in DataFrame from (x+1)th index till end
        for y in range(x + 1, df.shape[1]):
            # Select column at yth index.
            otherCol = df.iloc[:, y]
            # Check if two columns at x and y index are equal
            if col.equals(otherCol):
                duplicateColumnNames.add(df.columns.values[y])
 
    return list(duplicateColumnNames)




duplicated_cols = getDuplicateColumns(df)




df.drop(duplicated_cols,axis=1,inplace=True)




df.shape




df_target = pd.DataFrame(train.TARGET.value_counts())
df_target['Percentage'] = 100*df_target['TARGET']/train.shape[0]
df_target




fig, ax=plt.subplots(figsize=(8,6))
sns.countplot('TARGET',data=train);




cor_mat = train.corr()
for i in range(5):
    for j in range(5):
        x = i*50
        y = j*50
        corr = cor_mat.iloc[range(x,x+50),range(y,y+50)]
        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(15, 12))
        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr,linewidths=.5, ax=ax)




X = train.iloc[:,:-1]
y = train.TARGET

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X, y)

feat_imp = pd.Series(rf.feature_importances_, index=X.columns)
feat_imp.sort_values(inplace=True)
ax = feat_imp.tail(20).plot(kind='barh', figsize=(10,7), title='Feature importance')




fig, axs = plt.subplots(nrows= 3, ncols=3, figsize=(18, 25))

axs[0, 0].boxplot(train['var38'])
axs[0, 0].set_title('var38')

axs[0, 1].boxplot(train['var15'])
axs[0, 1].set_title('var15')

axs[0, 2].boxplot(train['saldo_medio_var5_ult3'])
axs[0, 2].set_title('saldo_medio_var5_ult3')

axs[1, 0].boxplot(train['saldo_medio_var5_hace3'])
axs[1, 0].set_title('saldo_medio_var5_hace3')

axs[1, 1].boxplot(train['num_var45_ult3'])
axs[1, 1].set_title('num_var45_ult3')

axs[1, 2].boxplot(train['num_var45_hace3'])
axs[1, 2].set_title('num_var45_hace3')

axs[2, 0].boxplot(train['saldo_var30'])
axs[2, 0].set_title('saldo_var30')

axs[2, 1].boxplot(train['saldo_var42'])
axs[2, 1].set_title('saldo_var42')

axs[2, 2].boxplot(train['saldo_medio_var5_hace2'])
axs[2, 2].set_title('saldo_medio_var5_hace2')

pca = PCA()
principalComponents = pca.fit_transform(
    df.drop( "TARGET", axis=1)
)
variance = pca.explained_variance_ratio_  
var_rat = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=2) * 100)var_rat[:8]plt.figure(figsize=(10, 6))
plt.title("Cumulative explained variance with PCA", fontsize=24)
plt.xlabel("# Features", fontsize=18)
plt.ylabel("Cumulative explained variance", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.plot(var_rat);pca_df = pd.DataFrame(
    data=principalComponents[:, :8], columns=["PC{}".format(i) for i in range(1, 9)]
)

pcas = pca_df.shape[1]-1 # selectedpca_df['ID'] = df.index
pca_df.set_index('ID',inplace = True)
pca_df['TARGET'] = df['TARGET']pca_df.head()


fig, ax=plt.subplots(figsize=(8,6))
sns.countplot('TARGET',data=df);




train_index_start= train.shape[0] # catching row until test data
print(train_index_start)
target_index_start= train.shape[1] # catching column until target colum
print(target_index_start)




train = df.iloc[:train_index_start,:]
test  = df.iloc[train_index_start:,:] 




#!pip install imblearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=0)
X_train = train.drop('TARGET',axis =1)
y_train = train['TARGET']
X_resampled, y_resampled = rus.fit_resample(X_train, y_train) 




fig, ax=plt.subplots(figsize=(8,6))
sns.countplot(y_resampled);




ros = RandomOverSampler(random_state = 0)
X_rosampled, y_rosampled = ros.fit_resample(X_train,y_train) 




fig, ax = plt.subplots(figsize=(8,6))
sns.countplot(y_rosampled);




X_train, X_test, y_train, y_test = train_test_split(X_resampled,y_resampled , test_size=0.33, random_state=42)
print(X_test.shape, y_test.shape)
print(X_train.shape, y_train.shape)

X_train=pca_df.iloc[:train_index_start, :pcas]
X_test =pca_df.iloc[train_index_start:, :pcas]
y_train=pca_df.iloc[:train_index_start, pcas]
y_test =pca_df.iloc[train_index_start:, pcas]


clf = ExtraTreesClassifier(random_state=42)




selector = clf.fit(X_train, y_train)
test['TARGET'] = 0
test.shape




feats_sel = SelectFromModel(selector, prefit=True)
X_train = feats_sel.transform(X_train)
X_test = feats_sel.transform(X_test)
test = feats_sel.transform(test.drop("TARGET", axis = 1))




X_train.shape, X_test.shape




clf_xgb = GridSearchCV(
    estimator=xgb.XGBClassifier(seed=42),
    param_grid={
        "learning_rate": [0.1,0.01],
        "min_child_weight": [1,2,4],
        "max_depth": [4,6,8],
        "subsample": [0.75],
        "colsample_bytree":[0.75,0.8],
        "n_estimators": [100,200,300],
        "max_features": [3,4,6],
    },
    cv=3,
    scoring="roc_auc",
    verbose=1,
    n_jobs=-1,
)
clf_grid_result=clf_xgb.fit(X_train,y_train)




clf_grid_result.best_estimator_




clf_grid_result.best_score_




y_pred = clf_grid_result.predict(X_test)




fig, ax = plt.subplots(figsize=(8,6))
sns.countplot(y_pred);




clf_grid_result.scorer_




random_seed = 42

RF_parameters = {'n_estimators': [120, 240, 480],
                 'bootstrap': [True],
                 'max_depth': [16, 32, 80 ,120],
                 'max_features': [, 'sqrt', 'log2'],
                 'min_samples_leaf': [2, 6, 8, 16, 24, 36, 48],
                 'min_samples_split': [8 ,10, 15, 24, 36, 48],
                 'random_state':[random_seed],
                 "n_jobs": [-1],
                 'criterion':['gini', 'entropy']}

clf_rf = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid = RF_parameters,
    cv=3,
    scoring="roc_auc",
    verbose=1,
    n_jobs=-1,
)
clf_grid_result_rf=clf_rf.fit(X_train,y_train)




clf_grid_result_rf.best_estimator_




clf_grid_result_rf.best_score_




y_pred = clf_grid_result_rf.predict(X_test)




fig, ax = plt.subplots(figsize=(8,6))
sns.countplot(y_pred);




probabilities = clf_grid_result.predict_proba(test)

submission = pd.DataFrame({"ID":df.iloc[train_index_start:,0].index, "TARGET": probabilities[:,1]})
submission.to_csv("submission_xgb.csv", index=False)




probabilities = clf_grid_result_rf.predict_proba(test)

submission = pd.DataFrame({"ID":df.iloc[train_index_start:,0].index, "TARGET": probabilities[:,1]})
submission.to_csv("submission_rf.csv", index=False)

