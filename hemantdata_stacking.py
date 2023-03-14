#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.




import pandas as pd
import numpy as np




df=pd.read_csv("../input/talkingdata-adtracking-fraud-detection/train_sample.csv")




del df['attributed_time']




# Convert click_time into time,month,year and day
value='click_time'
df[value]=pd.to_datetime(df[value])
df['year']=df[value].dt.year
df['month']=df[value].dt.month
df['day']=df[value].dt.day
df['time']=df[value].dt.hour
df['day_week']=df[value].dt.weekday_name




del df['click_time']




df.head()




df.tail()




df.is_attributed.unique()




df.is_attributed.value_counts()




from sklearn.utils import shuffle
df = shuffle(df)
df




#Select the variables to be one-hot encoded
one_hot_features = ['day_week']
# Convert categorical variables into dummy/indicator variables (i.e. one-hot encoding).
one_hot_encoded = pd.get_dummies(df[one_hot_features],drop_first=True)
one_hot_encoded.info(verbose=True, memory_usage=True, null_counts=True)




# Replacing categorical columns with dummies
fdf = df.drop(one_hot_features,axis=1)
train = pd.concat([df, one_hot_encoded] ,axis=1)




train.info()




ones=train['is_attributed']==1
zeros=train['is_attributed']==0




ones=train[ones]
zeros=train[zeros]




import random
#n=random.randint(227,99773)
#print("Randaom Number is:",n)
sample_zeros=zeros.sample(n=2043)




ones.head()




zeros.head()




ones.shape




sample_zeros.shape




train_dataset=  pd.concat([ones, sample_zeros] ,axis=0)




train_dataset.shape




train_dataset['click_id'] = range(1, len(train_dataset)+1)




train_dataset.head()




#Standardize rows into uniform scale

X = train_dataset.drop(['is_attributed','click_id'],axis=1)
y = train_dataset['is_attributed']




del X['day_week']




y.value_counts()




# Importing Models
import warnings
warnings.filterwarnings("ignore")
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Importing other tools
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, classification_report, make_scorer
from sklearn.metrics import accuracy_score, recall_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV




# Defining random seed
seed=42

# Creating Models

lr = LogisticRegression()
lda = LinearDiscriminantAnalysis()
svc = SVC(random_state=seed, probability=True)
dtree = DecisionTreeClassifier(random_state=seed)
rf = RandomForestClassifier(10, random_state=seed)
gdb = GradientBoostingClassifier(random_state=seed)
adb = AdaBoostClassifier(random_state=seed)
xgb = XGBClassifier(random_state=seed)
knn = KNeighborsClassifier()
lgbm = LGBMClassifier(random_state=seed)

first_models = [ lr,
                lda,
                svc,
                dtree,
                rf,
                gdb,
                adb,
                xgb, 
                knn,
                lgbm]
first_model_names = ['Logistic Regression',
                     'LDA',
                     'SVM',
                     'Decision Tree', 
                     'Random Forest',
                     'GradientBoosting',
                     'AdaBoost',
                     'XGB', 
                     'K-Neighbors',
                     'Light GBM'] 

# Defining other steps
n_folds = 5
skf = model_selection.ShuffleSplit(n_splits = n_folds, test_size = .3, train_size = .7, random_state = seed ) 
std_sca = StandardScaler()




MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Time']
MLA_compare = pd.DataFrame(columns = MLA_columns)

#create table to compare MLA predictions
MLA_predict = train_dataset[['click_id']]

train_size = X.shape[0]
n_models = len(first_models)
oof_pred = np.zeros((train_size, n_models))
scores = []
row_index=0

for n, model in enumerate(first_models):
    print('-'*25,first_model_names[n],'-'*int(45-len(first_model_names[n])))
    model_pipeline = Pipeline(steps=[('Scaler', std_sca),
                                     ('Estimator', model)])
    MLA_name = model.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(model.get_params())
    
    cv_results = model_selection.cross_validate(model, X, y, cv  = skf, return_train_score=True,scoring=None)

    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()  
    
    model_pipeline.fit(X, y)
    MLA_predict[MLA_name] = model_pipeline.predict(X)
    row_index+=1
        




#print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html
MLA_compare['Differenec'] = (MLA_compare['MLA Test Accuracy Mean'] - MLA_compare['MLA Train Accuracy Mean'])*100
MLA_compare.sort_values(by = ['Differenec'], ascending = False, inplace = True)
MLA_compare




feature_names = X.columns
feat_imp_df = pd.DataFrame(columns=first_model_names, index=feature_names)

# Dropping the Models that don't have feature importances for this analysis
feat_imp_df.drop(['SVM','K-Neighbors'], axis=1, inplace=True)


# I'm using absolute values for logistic Regression and LDA because we only care about the magnitude of the coefficient, not its direction 
feat_imp_df['Logistic Regression'] = np.abs(lr.coef_.ravel())
feat_imp_df['LDA'] = np.abs(lda.coef_.flatten())
feat_imp_df['Decision Tree'] = dtree.feature_importances_
feat_imp_df['Random Forest'] = rf.feature_importances_
feat_imp_df['GradientBoosting'] = gdb.feature_importances_
feat_imp_df['AdaBoost'] = adb.feature_importances_
feat_imp_df['XGB'] = xgb.feature_importances_
feat_imp_df['Light GBM'] = lgbm.feature_importances_
feat_imp_df




# http://www.menucool.com/rgba-color-picker
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()

scaled_fi = pd.DataFrame(data=mms.fit_transform(feat_imp_df),
                         columns=feat_imp_df.columns,
                         index=feat_imp_df.index)
scaled_fi['Overall'] = scaled_fi.sum(axis=1)

ordered_ranking = scaled_fi.sort_values('Overall', ascending=False)
fig, ax = plt.subplots(figsize=(10,7), dpi=80)
sns.barplot(data=ordered_ranking, y=ordered_ranking.index, x='Overall', palette='magma')
for index,data in enumerate(tuple(ordered_ranking.Overall)):
    plt.text(y=index , x=data , s=f"{data}" , fontdict=dict(fontsize=15), color="#CC0000")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.xaxis.set_visible(False)
ax.grid(False)
ax.set_title('Feature Importances for all Models');




list(scaled_fi[:10].index)




X =  X[list(scaled_fi[:10].index)].head()




ordered_ranking.index[:-3:-1]




train_v2 = train_dataset.drop(ordered_ranking.index[:-3:-1], axis=1)

X_v1 = train_v2.drop(['is_attributed'],axis=1)
y_v1 = train_v2['is_attributed']




X_v1.head()




del X_v1['day_week']
del X_v1['click_id']




MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Time']
MLA_compare = pd.DataFrame(columns = MLA_columns)

#create table to compare MLA predictions
MLA_predict = train_dataset[['click_id']]

train_size = X_v1.shape[0]
n_models = len(first_models)
oof_pred = np.zeros((train_size, n_models))
scores = []
row_index=0

for n, model in enumerate(first_models):
  print('-'*25,first_model_names[n],'-'*int(45-len(first_model_names[n])))
  model_pipeline = Pipeline(steps=[('Scaler', std_sca),
                                   ('Estimator', model)])
  MLA_name = model.__class__.__name__
  MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
  MLA_compare.loc[row_index, 'MLA Parameters'] = str(model.get_params())
  
  #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
  cv_results = model_selection.cross_validate(model, X_v1, y_v1, cv  = skf, return_train_score=True)

  MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
  MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
  MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()  
  
  model_pipeline.fit(X_v1, y_v1)
  MLA_predict[MLA_name] = model_pipeline.predict(X_v1)
  
  #model_pipeline.fit(X, y)
  #val_pred = model_pipeline.predict(x_val)
  #oof_pred[X, n] = model_pipeline.predict_proba(X)[:,1]
  row_index+=1
      




#print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html
MLA_compare['Differenec'] = (MLA_compare['MLA Test Accuracy Mean'] - MLA_compare['MLA Train Accuracy Mean'])*100
MLA_compare.sort_values(by = ['Differenec'], ascending = False, inplace = True)
MLA_compare




first_models = [rf,lgbm,knn]
first_model_names = ['rf','lgbm', 'knn'] 




LGBM_param_grid = {'lgbm__learning_rate': [0.1],
    'lgbm__n_estimators': [100],
    'lgbm__num_leaves': [31], # large num_leaves helps improve accuracy but might lead to over-fitting
    'lgbm__boosting_type' : ['gbdt', 'dart'], # for better accuracy -> try dart
    
                  }

RF_param_grid = {
                
               'rf__n_estimators': [100,200], #default=10
            'rf__criterion': ['gini', 'entropy'], #default=”gini”
            'rf__max_depth': [2,4,8] ,#default=None
            'rf__oob_score': ['True']
}

GDB_param_grid = {
    "gdb__loss":["deviance"],
    "gdb__learning_rate": [0.01,0.1],
    
    "gdb__max_depth":[3,5],
   
    "gdb__criterion": ["friedman_mse",  "mae"],
    
    "gdb__n_estimators":[10]
    
}
XGB_param_grid = {
    'xgb__min_child_weight': [1, 5],
        'xgb__gamma': [0.5, 1],
        'xgb__max_depth': [3, 4, 5]
}

KNN_param_grid = {
    'knn__n_neighbors':[5,6,7],
          'knn__leaf_size':[1,2,3,5],
          
}




from sklearn.model_selection import GridSearchCV
columns = ['Name', 'Parameters', 'Train Accuracy Mean', 'Test Accuracy']
#models = [LGBMClassifier(),RandomForestClassifier()]
params_grid = [RF_param_grid,LGBM_param_grid,KNN_param_grid]

after_model_compare = pd.DataFrame(columns = columns)

row_index = 0
for n,alg in enumerate(first_models):
    print('-'*25,first_model_names[n],'-'*int(29-len(first_model_names[n])))
    print(alg)
    model_pipeline = Pipeline(steps=[('Scaler', std_sca),
                                     (first_model_names[n], first_models[n])])
    
    gs_alg = GridSearchCV(model_pipeline, param_grid = params_grid[0], cv = skf, scoring = 'accuracy', n_jobs=-1,return_train_score=True)
    params_grid.pop(0)

    #set name and parameters
    model_name = alg.__class__.__name__
    after_model_compare.loc[row_index, 'Name'] = model_name
    
    gs_alg.fit(X_v1, y_v1)
   
    after_model_compare.loc[row_index, 'Parameters'] = str(gs_alg.best_params_)
 
    
    after_model_compare.loc[row_index, 'Train Accuracy Mean'] = gs_alg.cv_results_['mean_train_score'][gs_alg.best_index_]
    after_model_compare.loc[row_index, 'Test Accuracy'] = gs_alg.cv_results_['mean_test_score'][gs_alg.best_index_]
    
    row_index+=1
    print(row_index, alg.__class__.__name__, 'trained...')

after_model_compare




after_model_compare['Difference'] = (after_model_compare['Test Accuracy']-after_model_compare['Train Accuracy Mean'])*100




after_model_compare




X_v1.shape




X_v1.head()




vote_est = [
    #Ensemble Methods: http://scikit-learn.org/stable/modules/ensemble.html
    
    ('lda', LinearDiscriminantAnalysis()),
    
    ('ada', AdaBoostClassifier(random_state=seed)),
    
    ('lr', LogisticRegression()),
    
    #Nearest Neighbor: http://scikit-learn.org/stable/modules/neighbors.html
    ('knn', neighbors.KNeighborsClassifier()),
    
    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html
   ('xgb', XGBClassifier())
  

]


#Hard Vote or majority rules
vote_hard = ensemble.VotingClassifier(estimators = vote_est , voting = 'hard')
vote_hard_cv = model_selection.cross_validate(vote_hard, X_v1, y_v1, cv  = skf,return_train_score=True)
vote_hard.fit(X_v1, y_v1)
#print("Hard Voting Training w/bin score mean: {:.2f}". format(vote_hard_cv['train_score'].mean()*100)) 
print("Hard Voting Train w/bin score mean: {:.2f}". format(vote_hard_cv['train_score'].mean()*100))
print("Hard Voting Test w/bin score mean: {:.2f}". format(vote_hard_cv['test_score'].mean()*100))
print("Hard Voting Test w/bin score 3*std: +/- {:.2f}". format(vote_hard_cv['test_score'].std()*100*5))
print('-'*10)


#Soft Vote or weighted probabilities
vote_soft = ensemble.VotingClassifier(estimators = vote_est , voting = 'soft')
vote_soft_cv = model_selection.cross_validate(vote_soft, X_v1, y_v1, cv  = skf,return_train_score=True)
vote_soft.fit(X_v1, y_v1)

#print("Soft Voting Training w/bin score mean: {:.2f}". format(vote_soft_cv['train_score'].mean()*100)) 
print("Soft Voting Train w/bin score mean: {:.2f}". format(vote_soft_cv['train_score'].mean()*100))
print("Soft Voting Test w/bin score mean: {:.2f}". format(vote_soft_cv['test_score'].mean()*100))
print("Soft Voting Test w/bin score 3*std: +/- {:.2f}". format(vote_soft_cv['test_score'].std()*100*5))
print('-'*10)




df_test=pd.read_csv("../input/talkingdata-adtracking-fraud-detection/test.csv")




# Convert click_time into time,month,year and day
value='click_time'
df_test[value]=pd.to_datetime(df_test[value])
df_test['year']=df_test[value].dt.year
df_test['month']=df_test[value].dt.month
df_test['day']=df_test[value].dt.day
df_test['time']=df_test[value].dt.hour
df_test['day_week']=df_test[value].dt.weekday_name




del df_test['click_time']
del df_test['click_id']




df_test.head()




del df_test['day_week']




data={'day_week_Tuesday': np.zeros(df_test.shape[0], dtype='int'),
     'day_week_Thursday': np.zeros(df_test.shape[0], dtype='int'),
     'day_week_Wednesday':np.zeros(df_test.shape[0], dtype='int')}
day = pd.DataFrame(data, columns = ['day_week_Thursday','day_week_Tuesday','day_week_Wednesday'])
day




test = pd.concat([df_test, day] ,axis=1)




test.head()




del test['year']
del test['month']




test.head()




submission=pd.read_csv("../input/talkingdata-adtracking-fraud-detection/sample_submission.csv")




submission.shape




submission.head()




submission['is_attributed']=vote_soft.predict(test)




submission.head()




submission.shape




submission.to_csv("submission.csv",index=False)











