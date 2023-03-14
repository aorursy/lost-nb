#!/usr/bin/env python
# coding: utf-8



# Some concepts borrowed from other kernels (will )
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import warnings
from collections import Counter
from sklearn.feature_selection import mutual_info_classif
warnings.filterwarnings('ignore')




train = pd.read_csv("train.csv")
#train.head()




train_copy = train
train_copy = train_copy.replace(-1, np.NaN)




import missingno as msno
# Nullity or missing values by columns
msno.matrix(df=train_copy.iloc[:,2:39], figsize=(20, 14), color=(0.42, 0.1, 0.05))

#ps_reg_03, ps_car_03_cat and ps_car_05_cat has many missing values, hence we need to be carefull while doing NA. For the time being we will just do na, but later try different things




data = [go.Bar(
            x = train["target"].value_counts().index.values,
            y = train["target"].value_counts().values,
            text='Distribution of target variable'
    )]

layout = go.Layout(
    title='Target variable distribution'
)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='basic-bar')

#Skewed target, Hence F1 score is more important than accuracy




#Remove colums if contains all NULL (none so here)
train = train.dropna(axis=1, how='all')




cols = train.columns.tolist()
print cols




target = train.target
train.drop('target', axis=1, inplace=True)
train.drop('id', axis=1, inplace=True)
train.dtypes

#float64 are continuous variable , int64 are either binary or categorical




cols = train.columns.tolist()
print len(cols)
print(train.skew())
# May require transformaion of the skewed variables




train.describe()
# Different stastical figures




from sklearn import preprocessing
def draw_histograms(df, variables, n_rows, n_cols):
    fig=plt.figure(figsize=(60,80))
    for i, var_name in enumerate(variables):
        ax=fig.add_subplot(n_rows,n_cols,i+1)
        df[var_name].hist(bins=10,ax=ax)
        ax.set_title(var_name)
    fig.tight_layout()  # Improves appearance a bit.
    plt.show()

le = preprocessing.LabelEncoder()
for x in range(len(cols)):
    typ = train[cols[x]].dtype
    if typ == 'int64':
        train[cols[x]] = train[cols[x]].fillna(value=0)
    elif typ == 'float64':
        train[cols[x]] = train[cols[x]].fillna(value=0.0)
    elif typ == 'object':
        pass
        train[cols[x]] = train[cols[x]].fillna(value=0)
draw_histograms(train, train.columns, 8, 8)

#looking at the histogram of the input variable, some type of normalization of feature scaling may be required for some variables




train.plot.box(return_type='axes', figsize=(90,70))
#Box plot of all varibales https://www.wellbeingatschool.org.nz/information-sheet/understanding-and-interpreting-box-plots




Counter(train.dtypes.values)




train_float = train.select_dtypes(include=['float64'])
train_int = train.select_dtypes(include=['int64'])




colormap = plt.cm.inferno
plt.figure(figsize=(16,12))
plt.title('Pearson correlation of continuous features', y=1.05, size=15)
sns.heatmap(train_float.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)

#pearson correlation of continuous varibales shows some strong correlation between variables, we may have to drop some correlated varibales or have to transform them to new variables




cat_features = [a for a in train.columns if a.endswith('cat')]
print cat_features




import scipy 
from scipy.stats import spearmanr
from pylab import rcParams




train_cat = train[cat_features]
train_cat.head()




bin_features = [a for a in train.columns if a.endswith('bin')]
print bin_features




bin_train = train[bin_features]
bin_train.head()




#rcParams['figure.figsize'] = 5, 4
#sns.set_style("whitegrid")




#sns.pairplot(train_cat)




from scipy.stats import chi2_contingency
for x in range(len(cat_features)):
    for y in range((x+1), len(cat_features)):
        col1 = train_cat[cat_features[x]]
        col2 = train_cat[cat_features[y]]
        table = pd.crosstab(col1,col2)
        chi2, p, dof, expected = chi2_contingency(table.values)
        print cat_features[x], cat_features[y], ':'
        print 'Chi-square statistics: %0.3f p_value: %0.3f' % (chi2, p)
        
#chisquare test show heavy correlation between categorical vribales, not sure about it




#from scipy.stats import chi2_contingency
for x in range(len(bin_features)):
    for y in range((x+1), len(bin_features)):
        col1 = bin_train[bin_features[x]]
        col2 = bin_train[bin_features[y]]
        table = pd.crosstab(col1,col2)
        chi2, p, dof, expected = chi2_contingency(table.values)
        print bin_features[x], bin_features[y], ':'
        print 'Chi-square statistics: %0.3f p_value: %0.3f' % (chi2, p)
        
#chisquare test of binary variables




#sns.pairplot(bin_train)




from sklearn.feature_selection import VarianceThreshold
train_bin_copy = bin_train
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
sel.fit_transform(train_bin_copy)

print train_bin_copy.columns
print bin_train.columns




from sklearn.feature_selection import VarianceThreshold
train_cat_copy = train_cat
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
sel.fit_transform(train_cat_copy)

print train_cat_copy.columns
print train_cat.columns




from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

print bin_train.head()
X_new = SelectKBest(chi2, k=5).fit_transform(bin_train, target)
X_new.shape

#SelectKbest does univariate feature selection , can ue chi2, ANOVA etc. http://scikit-learn.org/stable/modules/feature_selection.html
# K is number of most important features




print X_new




#Tree based feature selection, I would first do the hyperparameter tuning usig GBM and then use SelectFromModel to choose the best features
#  have used the the link to do hyperparameter tuning

from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search




#I ran following gridsearch method to come up with best parameters, hence commenting them

#param_test1 = {'n_estimators':range(20,81,10)}
#gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8,random_state=10), 
#param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
#gsearch1.fit(train, target)




#gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_




#param_test2 = {'max_depth':range(5,16,2), 'min_samples_split':range(200,1001,200)}
#gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, max_features='sqrt', subsample=0.8, random_state=10), 
#param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
#gsearch2.fit(train,target)
#gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_




#param_test3 = {'min_samples_leaf':range(30,71,10)}
#gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=7,min_samples_split=600,max_features='sqrt', subsample=0.8, random_state=10), 
#param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
#gsearch3.fit(train,target)
#gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_




#param_test4 = {'max_features':range(30,46,2)}
#gsearch4 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=7, min_samples_split=600, min_samples_leaf=50, subsample=0.8, random_state=10),
#param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
#gsearch4.fit(train,target)
#gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_




#param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
#gsearch5 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=7,min_samples_split=600, min_samples_leaf=50, random_state=10,max_features=31),
#param_grid = param_test5, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
#gsearch5.fit(train,target)
#gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_




import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search




#param_test1 = {
# 'max_depth':range(3,10,2),
# 'min_child_weight':range(1,6,2)
#}
#gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
# min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
# objective= 'binary:logistic', nthread=50, scale_pos_weight=1, seed=27), 
# param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
#gsearch1.fit(train,target)
#gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_




#param_test2 = {
# 'max_depth':[5,6,7],
# 'min_child_weight':[1,3,5]
#}
#gsearch2 = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=5,
# min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
# objective= 'binary:logistic', nthread=70, scale_pos_weight=1,seed=27), 
# param_grid = param_test2, scoring='roc_auc',n_jobs=1,iid=False, cv=5)
#gsearch2.fit(train,target)
#gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_




#param_test2b = {
# 'min_child_weight':[6,8,10,12]
#}
#gsearch2b = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=5,
# min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
# objective= 'binary:logistic', nthread=20, scale_pos_weight=1,seed=27), 
# param_grid = param_test2b, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
#gsearch2b.fit(train,target)




#gsearch2b.grid_scores_, gsearch2b.best_params_, gsearch2b.best_score_




#param_test3 = {
# 'gamma':[i/10.0 for i in range(0,5)]
#}
#gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
# min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
# objective= 'binary:logistic', nthread=20, scale_pos_weight=1,seed=27), 
# param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
#gsearch3.fit(train,target)
#gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_




#param_test4 = {
# 'subsample':[i/10.0 for i in range(6,10)],
# 'colsample_bytree':[i/10.0 for i in range(6,10)]
#}
#gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=5,
# min_child_weight=6, gamma=0.1, subsample=0.8, colsample_bytree=0.8,
# objective= 'binary:logistic', nthread=20, scale_pos_weight=1,seed=27), 
# param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
#gsearch4.fit(train,target)
#gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_




#param_test5 = {
# 'subsample':[i/100.0 for i in range(75,90,5)],
# 'colsample_bytree':[i/100.0 for i in range(75,90,5)]
#}
#gsearch5 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=5,
# min_child_weight=6, gamma=0.1, subsample=0.8, colsample_bytree=0.8,
# objective= 'binary:logistic', nthread=20, scale_pos_weight=1,seed=27), 
# param_grid = param_test5, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
#gsearch5.fit(train,target)




#gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_




#param_test7 = {
# 'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]
#}
#gsearch7 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=5,
# min_child_weight=6, gamma=0.1, subsample=0.8, colsample_bytree=0.8,
# objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
# param_grid = param_test7, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
#gsearch7.fit(train,target)
#gsearch7.grid_scores_, gsearch7.best_params_, gsearch7.best_score_




from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

def modelfit(alg, dtrain, targer,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain.values, label=target.values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain, target,eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain)
    dtrain_predprob = alg.predict_proba(dtrain)[:,1]
        
    #Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(target.values, dtrain_predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(target, dtrain_predprob)
                    
    feat_imp = pd.Series(alg.feature_importances_, dtrain.columns).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')

xgb4 = XGBClassifier(
 learning_rate =0.01,
 n_estimators=5000,
 max_depth=5,
 min_child_weight=6,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 reg_alpha=0,
 objective= 'binary:logistic',
 nthread=80,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb4, train, target)




from sklearn.feature_selection import SelectFromModel

clf = XGBClassifier(
 learning_rate =0.01,
 n_estimators=5000,
 max_depth=5,
 min_child_weight=6,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 reg_alpha=0,
 objective= 'binary:logistic',
 nthread=80,
 scale_pos_weight=1,
 seed=27)
clf = clf.fit(train, target)
clf.feature_importances_  

model = SelectFromModel(clf, prefit=True)
X_new = model.transform(train)
X_new.shape 

#This is choosing 23 features out of 50 features




from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4


def modelfit(alg, dtrain, target, performCV=True, printFeatureImportance=True, cv_folds=5):
    #Fit the algorithm on the data
    alg.fit(dtrain, target)
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain)
    dtrain_predprob = alg.predict_proba(dtrain)[:,1]
    
    #Perform cross-validation:
    if performCV:
        cv_score = cross_validation.cross_val_score(alg, dtrain, target, cv=cv_folds, scoring='roc_auc')
    
    #Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(target.values, dtrain_predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(target, dtrain_predprob)
    
    if performCV:
        print "CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score))
        
    #Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, dtrain.columns).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
        

gbm_tuned_2 = GradientBoostingClassifier(learning_rate=0.01, n_estimators=600,max_depth=7, min_samples_split=600,min_samples_leaf=50, subsample=0.80, random_state=10, max_features=31)
modelfit(gbm_tuned_2, train, target)




from sklearn.feature_selection import SelectFromModel

clf = GradientBoostingClassifier(learning_rate=0.01, n_estimators=600,max_depth=7, min_samples_split=600,min_samples_leaf=50, subsample=0.80, random_state=10, max_features=31)
clf = clf.fit(train, target)
clf.feature_importances_  

model = SelectFromModel(clf, prefit=True)
X_new = model.transform(train)
X_new.shape 






