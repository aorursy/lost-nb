#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

#load packages
import sys #access to system parameters https://docs.python.org/3/library/sys.html
print("Python version: {}". format(sys.version))

import pandas as pd #collection of functions for data processing and analysis modeled after R dataframes with SQL like features
print("pandas version: {}". format(pd.__version__))

import matplotlib #collection of functions for scientific and publication-ready visualization
print("matplotlib version: {}". format(matplotlib.__version__))

import numpy as np #foundational package for scientific computing
print("NumPy version: {}". format(np.__version__))

import scipy as sp #collection of functions for scientific computing and advance mathematics
print("SciPy version: {}". format(sp.__version__)) 

import IPython
from IPython import display #pretty printing of dataframes in Jupyter notebook
print("IPython version: {}". format(IPython.__version__)) 

import sklearn #collection of machine learning algorithms
print("scikit-learn version: {}". format(sklearn.__version__))

#misc libraries
import random
import time
import math


#ignore warnings
import warnings
warnings.filterwarnings('ignore')
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action="ignore", category=ConvergenceWarning)

print('-'*25)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.






#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder,OrdinalEncoder, StandardScaler,KBinsDiscretizer
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Evaluation
from sklearn.metrics import cohen_kappa_score,make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix
from wordcloud import WordCloud

#Configure Visualization Defaults
#%matplotlib inline = show plots in Jupyter Notebook browser
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.style.use('ggplot')
sns.set_style('darkgrid')
pylab.rcParams['figure.figsize'] = 12,8




#Machine Learning Algorithm (MLA) Selection and Initialization
MLA = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    #Gaussian Processes
    #gaussian_process.GaussianProcessClassifier(),
    
    #GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),
    
    #Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    
    #Nearest Neighbor
    neighbors.KNeighborsClassifier(),
    
    #SVM
    #svm.SVC(probability=True),
    #svm.NuSVC(probability=True),
    #svm.LinearSVC(),
    
    #Trees    
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
    
    #Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),

    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html
    XGBClassifier()    
    ]

def train_model(data, MLA_list = MLA, print_feature_table=False, print_feature_score=False, top_n=8):
    
    target = data['AdoptionSpeed']
    X_train = data.drop(['AdoptionSpeed'],axis=1)
    
    MLA_columns = ['MLA Name', 'MLA Parameters','MLA cohen_kappa_score','MLA Time']
    MLA_compare = pd.DataFrame(columns = MLA_columns)

    MLA_predict = data['AdoptionSpeed']
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    row_index = 0
    for alg in MLA_list:

        MLA_name = alg.__class__.__name__
        MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
        MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
    
        kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        kappa_score = make_scorer(cohen_kappa_score, weights='quadratic')
        cv_results = model_selection.cross_validate(alg, X_train, target, cv  = kf, scoring=kappa_score )
        
        MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
        MLA_compare.loc[row_index, 'MLA cohen_kappa_score'] = cv_results['test_score'].mean() 
             
        #MLA_predict[MLA_name] = alg.predict(X_train)
        row_index+=1

    #print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html
    MLA_compare.sort_values(by = ['MLA cohen_kappa_score'], ascending = False, inplace = True)
    sns.barplot(x='MLA cohen_kappa_score', y = 'MLA Name', data = MLA_compare, color = 'b')
    plt.title('Machine Learning Algorithm Accuracy Score \n')
    plt.xlabel('Accuracy Score (%)')
    plt.ylabel('Algorithm')
    
    
    if print_feature_table:
        for alg in MLA_list:
            alg.fit(X_train, target)
            if hasattr(alg, 'feature_importances_'):
                feat_imp = pd.DataFrame({'importance':alg.feature_importances_})    
                feat_imp['feature'] = X_train.columns
                feat_imp.sort_values(by='importance', ascending=False, inplace=True)
                feat_imp = feat_imp.iloc[:top_n]
    
                feat_imp.sort_values(by='importance', inplace=True)
                feat_imp = feat_imp.set_index('feature', drop=True)
                feat_imp.plot.barh(title=alg.__class__.__name__)
                plt.xlabel('Feature Importance Score')
                plt.show()
    
            if print_feature_score:
                from IPython.display import display
                print("Top {} features in descending order of importance".format(top_n))
                display(feat_imp.sort_values(by='importance', ascending=False))
    
    return MLA_compare




train = pd.read_csv('../input/train/train.csv')
test = pd.read_csv('../input/test/test.csv')

train['dataset_type'] = 'train'
test['dataset_type'] = 'test'




useColumns = ['Type', 'Age', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
       'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
       'Sterilized', 'Health', 'Quantity', 'Fee', 'State',
       'VideoAmt', 'PhotoAmt', 'AdoptionSpeed']

#train_model(train[useColumns], print_feature_table=True, top_n=20)




train['Name'] = train['Name'].fillna('Unnamed')
test['Name'] = test['Name'].fillna('Unnamed')

train['has_name'] = train['Name'].apply(lambda x: 0 if x == 'Unnamed' else 1)
test['has_name'] = test['Name'].apply(lambda x: 0 if x == 'Unnamed' else 1)




train['has_name'] = train['Name'].apply(lambda x: 0 if x == 'No Name' or x == 'Unnamed' else 1)
test['has_name'] = test['Name'].apply(lambda x: 0 if x == 'No Name' or x == 'Unnamed' else 1)




top_3 = [
    ensemble.AdaBoostClassifier(),
    ensemble.GradientBoostingClassifier(),
    XGBClassifier()    
    ]

useColumns = ['Type', 'Age', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
       'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
       'Sterilized', 'Health', 'Quantity', 'Fee', 'State',
       'VideoAmt', 'PhotoAmt', 'AdoptionSpeed', 'has_name']

#train_model(train[useColumns], print_feature_table=True, top_n=10)




# defining a function which returns a list of top names
def top_names(df, top_percent):
    df_withnames = df[df.has_name != 0]
    items = df_withnames.shape[0]
    top_names = []
    counter = 0
    for i,v in df_withnames.Name.value_counts().items():
        if (counter/items)>top_percent:
            break
        top_names.append(i)
        counter = counter + v  
    return top_names




topnames = top_names(train.append(test), 0.2)
train['has_topname'] = train['Name'].apply(lambda row: 1 if row in topnames else 0)
test['has_topname'] = test['Name'].apply(lambda row: 1 if row in topnames else 0)




useColumns = ['Type', 'Age', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
       'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
       'Sterilized', 'Health', 'Quantity', 'Fee', 'State',
       'VideoAmt', 'PhotoAmt', 'AdoptionSpeed', 'has_name', 'has_topname']

#train_model(train[useColumns], top_3, print_feature_table=True, top_n=10)




#Binning:
def binning(col, cut_points, labels=None):
  #Define min and max values:
  minval = col.min()
  maxval = col.max()

  #create list by adding min and max to cut_points
  break_points = [minval] + cut_points + [maxval]

  #if no labels provided, use default labels 0 ... (n-1)
  if not labels:
    labels = range(len(cut_points)+1)

  #Binning using cut function of pandas
  colBin = pd.cut(col,bins=break_points,labels=labels,include_lowest=True)
  return colBin




cut_points = [1,2,3,4,8,12,24,36,48]
labels = ["1m","2m","3m","4m","8m","1y","2y","3y","4y","5y and over"]
train["age_bin"] = binning(train["Age"], cut_points, labels)

# for our model we need to have int values, thus we use a LabelEncoder
labels = [1,2,3,4,5,6,7,8,9,10]
train["age_bin"] = binning(train["Age"], cut_points, labels)
test["age_bin"] = binning(test["Age"], cut_points, labels)
label = LabelEncoder()
train['age_bin'] = label.fit_transform(train['age_bin'])
test['age_bin'] = label.fit_transform(test['age_bin'])




top_3 = [
    ensemble.AdaBoostClassifier(),
    ensemble.GradientBoostingClassifier(),
    XGBClassifier()    
    ]

useColumns = ['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
       'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
       'Sterilized', 'Health', 'Quantity', 'Fee', 'State',
       'VideoAmt', 'PhotoAmt', 'AdoptionSpeed', 'has_name', 'has_topname', 'age_bin']

#train_model(train[useColumns], top_3, print_feature_table=True, top_n=10)




binner = KBinsDiscretizer(n_bins=5,encode='ordinal', strategy='kmeans')

# if we just use pandas train.Age we get a Series.
# scikit-learns wants a 2d array
# ValueError: Expected 2D array, got 1D array instead:
# train[['Age']].copy()
train['age_bin_kmeans'] = pd.DataFrame(binner.fit_transform(train[['Age']].copy()))
test['age_bin_kmeans'] = pd.DataFrame(binner.fit_transform(test[['Age']].copy()))




binner = KBinsDiscretizer(n_bins=5,encode='ordinal', strategy='quantile')

# if we just use pandas train.Age we get a Series.
# scikit-learns wants a 2d array
# ValueError: Expected 2D array, got 1D array instead:
# train[['Age']].copy()
train['age_bin_quantile'] = pd.DataFrame(binner.fit_transform(train[['Age']].copy()))
test['age_bin_quantile'] = pd.DataFrame(binner.fit_transform(test[['Age']].copy()))




useColumns = ['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
       'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
       'Sterilized', 'Health', 'Quantity', 'Fee', 'State',
       'VideoAmt', 'PhotoAmt', 'AdoptionSpeed', 'has_name', 'has_topname', 'age_bin', 'age_bin_kmeans', 'age_bin_quantile']

#train_model(train[useColumns], top_3, print_feature_table=True, top_n=10)




# https://www.101computing.net/how-old-is-your-cat/

cat_human_age = {1:0.5, 2:3, 3:4, 4:6, 5:8, 6:10, 7:12, 8:14, 9:15, 10:16, 11:17, 12:18, 24:24, 48:35, 72:42, 96:50, 120:60, 144:70, 168:80, 192:84 }
small_dog_human_age = {1:1, 2:2, 3:2.5, 4:3.5, 5:4.3, 6:5, 7:6.3, 8:7, 9:9, 10:11, 11:13, 12:15, 24:23, 48:32, 72:40, 96:48, 120:56, 144:64, 168:72, 192:80 }
normal_dog_human_age = {1:1, 2:2, 3:2.5, 4:3.5, 5:4.3, 6:5, 7:6.3, 8:7, 9:9, 10:11, 11:13, 12:15, 24:24, 48:34, 72:42, 96:51, 120:60, 144:69, 168:78, 192:87 }
big_dog_human_age = {1:1, 2:2, 3:2.5, 4:3.5, 5:4.3, 6:5, 7:6.3, 8:7, 9:9, 10:11, 11:13, 12:14, 24:22, 48:34, 72:45, 96:55, 120:66, 144:77, 168:88, 192:99 }

def human_age(row):
    months = row['Age']
    if months == 0:
        return 0
    if row['Type'] == 2:
        if cat_human_age.get(months) is not None:
            return cat_human_age.get(months)
        else:
            if months < 25:
                return 25
            else:
                return (25 + ((months/12) - 2) * 4)
    elif row['Type'] == 1 and row['MaturitySize'] == 1:
        if small_dog_human_age.get(months) is not None:
            return small_dog_human_age.get(months)
        else:
            if months < 24:
                return (months/12) * 11
            else:
                return (22 + ((months/12) - 2) * 4)
    elif row['Type'] == 1 and row['MaturitySize'] == 3:
        if big_dog_human_age.get(months) is not None:
            return big_dog_human_age.get(months)
        else:
            if months < 24:
                return (months/12) * 11
            else:
                return (22 + ((months/12) - 2) * 4)
    if normal_dog_human_age.get(months) is not None:
        return normal_dog_human_age.get(months)
    else:
        if months < 24:
            return (months/12) * 11
        else:
            return (22 + ((months/12) - 2) * 4)

def lifestage(row):
    age = row['human_age']
    if age < 10:
        return 'Kitten/Puppy'
    elif age < 25:
        return 'Junior'
    elif age < 40:
        return 'Prime'
    elif age < 60:
        return 'Mature'
    elif age < 74:
        return 'Senior'
    return 'Geriatic'




train['human_age'] = train.apply(human_age, axis=1)
train['lifestage'] = train.apply(lifestage, axis=1)

test['human_age'] = test.apply(human_age, axis=1)
test['lifestage'] = test.apply(lifestage, axis=1)




top_3 = [
    ensemble.AdaBoostClassifier(),
    ensemble.GradientBoostingClassifier(),
    XGBClassifier()    
    ]

useColumns = ['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
       'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
       'Sterilized', 'Health', 'Quantity', 'Fee', 'State',
       'VideoAmt', 'PhotoAmt', 'AdoptionSpeed', 'has_name', 'has_topname', 'age_bin',
              'age_bin_kmeans', 'age_bin_quantile','human_age']

#train_model(train[useColumns], top_3, print_feature_table=True, top_n=10)




mapper = {'Kitten/Puppy':1, 'Junior':2, 'Prime':3, 'Mature':4,'Senior':5,'Geriatic':6}

train.lifestage.replace(mapper, inplace=True)
test.lifestage.replace(mapper, inplace=True)




useColumns = ['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
       'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
       'Sterilized', 'Health', 'Quantity', 'Fee', 'State',
       'VideoAmt', 'PhotoAmt', 'AdoptionSpeed', 'has_name', 'has_topname',  'age_bin', 
              'age_bin_kmeans', 'age_bin_quantile','human_age', 'lifestage']

#train_model(train[useColumns], top_3, print_feature_table=True, top_n=10)




def mixed_breed(row):
    if row['Breed1'] == 307:
        return 1
    elif row['Breed2'] == 0:
        return 0 
    elif row['Breed2'] != row['Breed1']:
        return 1
    else:
        return 0

train['mixed_breed'] = train.apply(mixed_breed, axis=1)
test['mixed_breed'] = test.apply(mixed_breed, axis=1)




top_1 = [
    ensemble.GradientBoostingClassifier()   
]

useColumns = ['Type', 'Breed1', 'Breed2',  'Gender', 'Color1', 'Color2',
       'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
       'Sterilized', 'Health', 'Quantity', 'Fee', 'State',
       'VideoAmt', 'PhotoAmt', 'AdoptionSpeed', 'has_name', 'has_topname',  'age_bin', 
              'age_bin_kmeans', 'age_bin_quantile','human_age', 'lifestage','mixed_breed']

#train_model(pd.get_dummies(train[useColumns], columns=['Breed1','Breed2'], sparse=True), top_1, print_feature_table=True, top_n=10)




train.groupby('Breed1')["AdoptionSpeed"].value_counts().head(5)




subtrain = train[train.Breed1 == 0]
cumulative_sum = subtrain.groupby('Breed1')["AdoptionSpeed"].cumsum() - subtrain["AdoptionSpeed"]
cumulative_count = subtrain.groupby('Breed1').cumcount()
print(cumulative_sum)
print(cumulative_count)
cumulative_sum/cumulative_count




def cat_mean_encoding(df, column):
    cumulative_sum = df.groupby(column)["AdoptionSpeed"].cumsum() - df["AdoptionSpeed"]
    cumulative_count = df.groupby(column).cumcount()
    df[column + "_mean_encoding"] =  cumulative_sum/cumulative_count
    df[column + "_mean_encoding"].fillna(df.AdoptionSpeed.mean(), inplace=True)




# from: https://medium.com/@pouryaayria/k-fold-target-encoding-dfe9a594874b
from sklearn import base

class KFoldTargetEncoderTrain(base.BaseEstimator, base.TransformerMixin):

    def __init__(self, colnames,targetName,n_fold=5,verbosity=True,discardOriginal_col=False):

        self.colnames = colnames
        self.targetName = targetName
        self.n_fold = n_fold
        self.verbosity = verbosity
        self.discardOriginal_col = discardOriginal_col

    def fit(self, X, y=None):
        return self


    def transform(self,X):

        assert(type(self.targetName) == str)
        assert(type(self.colnames) == str)
        assert(self.colnames in X.columns)
        assert(self.targetName in X.columns)

        mean_of_target = X[self.targetName].mean()
        kf = StratifiedKFold(n_splits=self.n_fold, shuffle=True)
        
        col_mean_name = self.colnames + '_' + 'Kfold_Target_Enc'
        X[col_mean_name] = np.nan

        for tr_ind, val_ind in kf.split(X,X[self.targetName]):
            X_tr, X_val = X.iloc[tr_ind], X.iloc[val_ind]
            #print(tr_ind,val_ind)
            X.loc[X.index[val_ind], col_mean_name] = X_val[self.colnames].map(X_tr.groupby(self.colnames)[self.targetName].mean())

        X[col_mean_name].fillna(mean_of_target, inplace = True)

        if self.verbosity:

            encoded_feature = X[col_mean_name].values
            print('Correlation between the new feature, {} and, {} is {}.'.format(col_mean_name,
                                                                                      self.targetName,
                                                                                      np.corrcoef(X[self.targetName].values, encoded_feature)[0][1]))
        if self.discardOriginal_col:
            X = X.drop(self.targetName, axis=1)
            
        return X
    
    
    
class KFoldTargetEncoderTest(base.BaseEstimator, base.TransformerMixin):
    def __init__(self,train,colNames,encodedName):
        self.train = train
        self.colNames = colNames
        self.encodedName = encodedName
        
    def fit(self, X, y=None):
        return self

    def transform(self,X):

        mean = self.train[[self.colNames,self.encodedName]].groupby(self.colNames).mean().reset_index() 
        
        dd = {}
        for index, row in mean.iterrows():
            dd[row[self.colNames]] = row[self.encodedName]

        X[self.encodedName] = X[self.colNames]
        X[self.encodedName] = X[self.encodedName].map(dd)

        return X




targetc = KFoldTargetEncoderTrain('Breed1','AdoptionSpeed',n_fold=5)
train = targetc.fit_transform(train)

targetc = KFoldTargetEncoderTrain('Breed2','AdoptionSpeed',n_fold=5)
train = targetc.fit_transform(train)

test_targetc = KFoldTargetEncoderTest(train,'Breed1','Breed1_Kfold_Target_Enc')
test= test_targetc.fit_transform(test)

test_targetc = KFoldTargetEncoderTest(train,'Breed2','Breed2_Kfold_Target_Enc')
test= test_targetc.fit_transform(test)




cat_mean_encoding(train, 'Breed1')
cat_mean_encoding(train, 'Breed2')

test_targetc = KFoldTargetEncoderTest(train,'Breed1','Breed1_mean_encoding')
test= test_targetc.fit_transform(test)

test_targetc = KFoldTargetEncoderTest(train,'Breed2','Breed2_mean_encoding')
test= test_targetc.fit_transform(test)




print(train.Breed1_Kfold_Target_Enc.isna().sum())
print(train.Breed2_Kfold_Target_Enc.isna().sum())
print(test.Breed1_Kfold_Target_Enc.isna().sum())
print(test.Breed2_Kfold_Target_Enc.isna().sum())

print(train.Breed1_mean_encoding.isna().sum())
print(train.Breed2_mean_encoding.isna().sum())
print(test.Breed1_mean_encoding.isna().sum())
print(test.Breed2_mean_encoding.isna().sum())

test[~test.Breed1.isin(train.Breed1)].head(5)




test.Breed1_Kfold_Target_Enc.fillna(train.AdoptionSpeed.mean(), inplace=True)
test.Breed2_Kfold_Target_Enc.fillna(train.AdoptionSpeed.mean(), inplace=True)
test.Breed1_mean_encoding.fillna(train.AdoptionSpeed.mean(), inplace=True)
test.Breed2_mean_encoding.fillna(train.AdoptionSpeed.mean(), inplace=True)




useColumns = ['Type', 'Breed1_Kfold_Target_Enc','Breed2_Kfold_Target_Enc','Gender', 'Color1', 'Color2',
       'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
       'Sterilized', 'Health', 'Quantity', 'Fee', 'State',
       'VideoAmt', 'PhotoAmt', 'AdoptionSpeed', 'has_name', 'has_topname',  'age_bin', 
              'age_bin_kmeans', 'age_bin_quantile','human_age', 'lifestage','mixed_breed',
       'Breed1_mean_encoding', 'Breed2_mean_encoding']

#train_model(train[useColumns], top_3, print_feature_table=True, top_n=10)




# create pet count for each rescuer
rescuer_count = train.groupby(['RescuerID'])['PetID'].count().reset_index()
rescuer_count_dict = rescuer_count.set_index('RescuerID').T.to_dict('list')

#set number to each item
train.RescuerID = train.RescuerID.apply(lambda x: np.log(rescuer_count_dict.get(x)[0]))

# now we have a number count for each rescuer. now we create bins 
binner = KBinsDiscretizer(n_bins=6,encode='ordinal', strategy='kmeans')
train['rescuer_bin_kmeans'] = pd.DataFrame(binner.fit_transform(train[['RescuerID']].copy()))

# target encode
targetc = KFoldTargetEncoderTrain('rescuer_bin_kmeans','AdoptionSpeed',n_fold=5)
train = targetc.fit_transform(train)


# same for the test
rescuer_count = test.groupby(['RescuerID'])['PetID'].count().reset_index()
rescuer_count_dict = rescuer_count.set_index('RescuerID').T.to_dict('list')
test.RescuerID = test.RescuerID.apply(lambda x: np.log(rescuer_count_dict.get(x)[0]))
binner = KBinsDiscretizer(n_bins=5,encode='ordinal', strategy='kmeans')
test['rescuer_bin_kmeans'] = pd.DataFrame(binner.fit_transform(test[['RescuerID']].copy()))
test_targetc = KFoldTargetEncoderTest(train,'rescuer_bin_kmeans','rescuer_bin_kmeans_Kfold_Target_Enc')
test= test_targetc.fit_transform(test)




useColumns = ['Type', 'Breed1_Kfold_Target_Enc','Breed2_Kfold_Target_Enc','Gender', 'Color1', 'Color2',
       'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
       'Sterilized', 'Health', 'Quantity', 'Fee', 'State',
       'VideoAmt', 'PhotoAmt', 'AdoptionSpeed', 'has_name', 'has_topname',  'age_bin', 
              'age_bin_kmeans', 'age_bin_quantile','human_age', 'lifestage','mixed_breed',
       'Breed1_mean_encoding', 'Breed2_mean_encoding','rescuer_bin_kmeans','rescuer_bin_kmeans_Kfold_Target_Enc']

#train_model(train[useColumns], top_3, print_feature_table=True, top_n=10)




# state GDP: https://en.wikipedia.org/wiki/List_of_Malaysian_states_by_GDP
state_gdp = {
    41336: 116.679,
    41325: 40.596,
    41367: 23.02,
    41401: 190.075,
    41415: 5.984,
    41324: 37.274,
    41332: 42.389,
    41335: 52.452,
    41330: 67.629,
    41380: 5.642,
    41327: 81.284,
    41345: 80.167,
    41342: 121.414,
    41326: 280.698,
    41361: 32.270
}

# state population: https://en.wikipedia.org/wiki/Malaysia
state_population = {
    41336: 33.48283,
    41325: 19.47651,
    41367: 15.39601,
    41401: 16.74621,
    41415: 0.86908,
    41324: 8.21110,
    41332: 10.21064,
    41335: 15.00817,
    41330: 23.52743,
    41380: 2.31541,
    41327: 15.61383,
    41345: 32.06742,
    41342: 24.71140,
    41326: 54.62141,
    41361: 10.35977
}

state_area ={
41336:19102,
41325:9500,
41367:15099,
41401:243,
41415:91,
41324:1664,
41332:6686,
41335:36137,
41330:21035,
41380:821,
41327:1048,
41345:73631,
41342:124450,
41326:8104,
41361:13035}

train["state_gdp"] = train.State.map(state_gdp)
train["state_population"] = train.State.map(state_population)
train["state_area"] = train.State.map(state_area)
test["state_gdp"] = test.State.map(state_gdp)
test["state_population"] = test.State.map(state_population)
test["state_area"] = test.State.map(state_area)




targetc = KFoldTargetEncoderTrain('State','AdoptionSpeed',n_fold=5)
train = targetc.fit_transform(train)

test_targetc = KFoldTargetEncoderTest(train,'State','State_Kfold_Target_Enc')
test= test_targetc.fit_transform(test)




useColumns = ['Type', 'Breed1_Kfold_Target_Enc','Breed2_Kfold_Target_Enc','Gender', 'Color1', 'Color2',
       'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
       'Sterilized', 'Health', 'Quantity', 'Fee','State_Kfold_Target_Enc','state_gdp','state_population','state_area',
       'VideoAmt', 'PhotoAmt', 'AdoptionSpeed', 'has_name', 'has_topname',  'age_bin', 
              'age_bin_kmeans', 'age_bin_quantile','human_age', 'lifestage','mixed_breed',
       'Breed1_mean_encoding', 'Breed2_mean_encoding','rescuer_bin_kmeans','rescuer_bin_kmeans_Kfold_Target_Enc'
            ]

#train_model(train[useColumns], top_3, print_feature_table=True, top_n=10)




targetc = KFoldTargetEncoderTrain('Color1','AdoptionSpeed',n_fold=5)
train = targetc.fit_transform(train)

test_targetc = KFoldTargetEncoderTest(train,'Color1','Color1_Kfold_Target_Enc')
test= test_targetc.fit_transform(test)

targetc = KFoldTargetEncoderTrain('Color2','AdoptionSpeed',n_fold=5)
train = targetc.fit_transform(train)

test_targetc = KFoldTargetEncoderTest(train,'Color2','Color2_Kfold_Target_Enc')
test= test_targetc.fit_transform(test)

targetc = KFoldTargetEncoderTrain('Color3','AdoptionSpeed',n_fold=5)
train = targetc.fit_transform(train)

test_targetc = KFoldTargetEncoderTest(train,'Color3','Color3_Kfold_Target_Enc')
test= test_targetc.fit_transform(test)




useColumns = ['Type', 'Breed1_Kfold_Target_Enc','Breed2_Kfold_Target_Enc','Gender', 
              'Color1_Kfold_Target_Enc', 'Color2_Kfold_Target_Enc',
       'Color3_Kfold_Target_Enc', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
       'Sterilized', 'Health', 'Quantity', 'Fee','State_Kfold_Target_Enc','state_gdp','state_population','state_area',
       'VideoAmt', 'PhotoAmt', 'AdoptionSpeed', 'has_name', 'has_topname',  'age_bin', 
              'age_bin_kmeans', 'age_bin_quantile','human_age', 'lifestage','mixed_breed',
       'Breed1_mean_encoding', 'Breed2_mean_encoding','rescuer_bin_kmeans','rescuer_bin_kmeans_Kfold_Target_Enc'
            ]

#train_model(train[useColumns], top_3, print_feature_table=True, top_n=10)




train.Description.fillna('none', inplace=True)
test.Description.fillna('none', inplace=True)

train['desc_length'] = train.Description.apply(len)
test['desc_length'] = test.Description.apply(len)




tfv = TfidfVectorizer(min_df=2,  max_features=None,
                      strip_accents='unicode', analyzer='word', token_pattern=r'(?u)\b\w+\b',
                      ngram_range=(1, 4), use_idf=1, smooth_idf=1, sublinear_tf=1)
# make 5 pca's
svd_ = TruncatedSVD(n_components=5)

# here we use train and test data to create the tfidf.
train_test = pd.concat([train, test], ignore_index=True, sort=False)
tfidf_col = tfv.fit_transform(train_test.Description)
svd_col = svd_.fit_transform(tfidf_col)
svd_col = pd.DataFrame(svd_col)
svd_col = svd_col.add_prefix('TFIDF_')
pcas = pd.concat([train_test, svd_col], axis=1)

#extract test and train rows
train = pcas.loc[np.isfinite(pcas.AdoptionSpeed), :]
test = pcas.loc[~np.isfinite(pcas.AdoptionSpeed), :]




useColumns = ['Type', 'Breed1_Kfold_Target_Enc','Breed2_Kfold_Target_Enc','Gender', 
              'Color1_Kfold_Target_Enc', 'Color2_Kfold_Target_Enc',
       'Color3_Kfold_Target_Enc', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
       'Sterilized', 'Health', 'Quantity', 'Fee','State_Kfold_Target_Enc','state_gdp','state_population','state_area',
       'VideoAmt', 'PhotoAmt', 'AdoptionSpeed', 'has_name', 'has_topname',  'age_bin', 
              'age_bin_kmeans', 'age_bin_quantile','human_age', 'lifestage','mixed_breed',
       'Breed1_mean_encoding', 'Breed2_mean_encoding','rescuer_bin_kmeans','rescuer_bin_kmeans_Kfold_Target_Enc',
              'desc_length', 'TFIDF_0','TFIDF_1','TFIDF_2','TFIDF_3','TFIDF_4'             ]

#result = train_model(train[useColumns],  print_feature_table=False)




from sklearn.feature_selection import RFECV

useColumns = ['Type', 'Breed1_Kfold_Target_Enc','Breed2_Kfold_Target_Enc','Gender', 
              'Color1_Kfold_Target_Enc', 'Color2_Kfold_Target_Enc',
       'Color3_Kfold_Target_Enc', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
       'Sterilized', 'Health', 'Quantity', 'Fee','State_Kfold_Target_Enc','state_gdp','state_population','state_area',
       'VideoAmt', 'PhotoAmt', 'has_name', 'has_topname',  'age_bin', 
              'age_bin_kmeans', 'age_bin_quantile','human_age', 'lifestage','mixed_breed',
       'Breed1_mean_encoding', 'Breed2_mean_encoding','rescuer_bin_kmeans','rescuer_bin_kmeans_Kfold_Target_Enc'
            ]
features = train[useColumns]

kappa_score = make_scorer(cohen_kappa_score, weights='quadratic')
for alg in []:
    kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    rfecv = RFECV(estimator=alg, step=1, cv=kf, scoring=kappa_score, n_jobs=-1)
    rfecv.fit(features,train['AdoptionSpeed'])
    print("Optimal number of features : %d" % rfecv.n_features_)
    print(rfecv.support_)
    print(features.columns[rfecv.support_])
    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()




useColumns = ['Breed1_Kfold_Target_Enc', 'Breed2_Kfold_Target_Enc', 'FurLength',
       'Sterilized', 'State_Kfold_Target_Enc', 'PhotoAmt', 'age_bin',
       'human_age', 'Breed1_mean_encoding', 'Breed2_mean_encoding',
       'rescuer_bin_kmeans', 'rescuer_bin_kmeans_Kfold_Target_Enc']

params = {
 'n_estimators': [50, 100, 200],
 'learning_rate' : [0.01,0.05,0.1,0.3,1],
 'base_estimator': [ensemble.AdaBoostClassifier(), ensemble.GradientBoostingClassifier()]
 }

X_train = train[useColumns]
y_train = train['AdoptionSpeed']

kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
cv_split = kf.split(X_train, y_train)
kappa_score = make_scorer(cohen_kappa_score, weights='quadratic')
tuning = GridSearchCV(estimator = ensemble.AdaBoostClassifier(), param_grid = params,
                      scoring=kappa_score,cv=5, n_jobs=-1,iid=False)

#tuning.fit(X_train,y_train)
#tuning.best_params_, tuning.best_score_




params = {'learning_rate':[0.15,0.1,0.05,0.01,0.005,0.001], 
          'n_estimators':[100,250,500,750,1000,1250,1500,1750],
          'max_depth':[2,3,4,5,6,7],
          'min_samples_split':[2,4,6,8,10,20,40,60,100], 
          'min_samples_leaf':[1,3,5,7,9],
          'max_features':[2,3,4,5,6,7],
          'subsample':[0.7,0.75,0.8,0.85,0.9,0.95,1]}

kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
kappa_score = make_scorer(cohen_kappa_score, weights='quadratic')
tuning = GridSearchCV(estimator = ensemble.GradientBoostingClassifier(), param_grid = params, 
                      scoring=kappa_score,cv=kf, n_jobs=-1,iid=False)

train_cleaned = train.drop(['Name','human_age','lifestage','RescuerID','Description','PetID', 'dataset_type'],axis=1)
y_train = train_cleaned['AdoptionSpeed']
X_train = train_cleaned.drop(['AdoptionSpeed'],axis=1)
    
#tuning.fit(X_train,y_train)
#tuning.grid_scores_, tuning.best_params_, tuning.best_score_




# from https://www.kaggle.com/naveenasaithambi/optimizedrounder-improved

import scipy as sp

from collections import Counter
from functools import partial
from math import sqrt

from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.metrics import confusion_matrix as sk_cmatrix


# FROM: https://www.kaggle.com/myltykritik/simple-lgbm-image-features

# The following 3 functions have been taken from Ben Hamner's github repository
# https://github.com/benhamner/Metrics
def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(y, y_pred):
    """
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = y
    rater_b = y_pred
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)

class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0
    
    def _kappa_loss(self, coef, X, y):
        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4])
        return -cohen_kappa_score(y, preds, weights='quadratic')
    
    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X = X, y = y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')
    
    def predict(self, X, coef):
        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4])
        return preds
    
    def coefficients(self):
        return self.coef_['x']




useColumns = ['Type', 'Breed1_Kfold_Target_Enc','Breed2_Kfold_Target_Enc','Gender', 
              'Color1_Kfold_Target_Enc', 'Color2_Kfold_Target_Enc',
       'Color3_Kfold_Target_Enc', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
       'Sterilized', 'Health', 'Quantity', 'Fee','State_Kfold_Target_Enc','state_gdp','state_population','state_area',
       'VideoAmt', 'PhotoAmt','has_name', 'has_topname',  'age_bin', 
              'age_bin_kmeans', 'age_bin_quantile','human_age', 'lifestage','mixed_breed',
       'Breed1_mean_encoding', 'Breed2_mean_encoding','rescuer_bin_kmeans','rescuer_bin_kmeans_Kfold_Target_Enc',
              'desc_length', 'TFIDF_0','TFIDF_1','TFIDF_2','TFIDF_3','TFIDF_4'             ]

target = train.AdoptionSpeed.astype('int')
X_train=train[useColumns]
X_test=test[useColumns]

#clf = ensemble.GradientBoostingClassifier(n_estimators=200, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, target)
#pred = clf.predict(X_test)
           
#submit=pd.DataFrame()
#submit['PetID']=test['PetID']
#submit['AdoptionSpeed']=pred
#submit.to_csv('submission.csv',index=False)




useColumns = ['AdoptionSpeed','Type', 'Breed1_Kfold_Target_Enc','Breed2_Kfold_Target_Enc','Gender', 
              'Color1_Kfold_Target_Enc', 'Color2_Kfold_Target_Enc',
       'Color3_Kfold_Target_Enc', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
       'Sterilized', 'Health', 'Quantity', 'Fee','State_Kfold_Target_Enc','state_gdp','state_population','state_area',
       'VideoAmt', 'PhotoAmt','has_name', 'has_topname',  'age_bin', 
              'age_bin_kmeans', 'age_bin_quantile','human_age', 'lifestage','mixed_breed',
       'Breed1_mean_encoding', 'Breed2_mean_encoding','rescuer_bin_kmeans','rescuer_bin_kmeans_Kfold_Target_Enc',
              'desc_length', 'TFIDF_0','TFIDF_1','TFIDF_2','TFIDF_3','TFIDF_4'             ]

target = train.AdoptionSpeed.astype('int')
X_train=train[useColumns]
X_test=test[useColumns].drop(['AdoptionSpeed'], axis=1)




xgb_params = {
    'eval_metric': 'rmse',
    'seed': 1337,
    'silent': 1,
}




import xgboost as xgb

def run_xgb(params, X_train, X_test):
    n_splits = 5
    verbose_eval = 1000
    num_rounds = 30000
    early_stop = 500

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1337)

    oof_train = np.zeros((X_train.shape[0]))
    oof_test = np.zeros((X_test.shape[0], n_splits))

    i = 0

    for train_idx, valid_idx in kf.split(X_train, X_train['AdoptionSpeed'].values):

        X_tr = X_train.iloc[train_idx, :]
        X_val = X_train.iloc[valid_idx, :]

        y_tr = X_tr['AdoptionSpeed'].values
        X_tr = X_tr.drop(['AdoptionSpeed'], axis=1)

        y_val = X_val['AdoptionSpeed'].values
        X_val = X_val.drop(['AdoptionSpeed'], axis=1)

        d_train = xgb.DMatrix(data=X_tr, label=y_tr, feature_names=X_tr.columns)
        d_valid = xgb.DMatrix(data=X_val, label=y_val, feature_names=X_val.columns)

        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        model = xgb.train(dtrain=d_train, num_boost_round=num_rounds, evals=watchlist,
                         early_stopping_rounds=early_stop, verbose_eval=verbose_eval, params=params)

        valid_pred = model.predict(xgb.DMatrix(X_val, feature_names=X_val.columns), ntree_limit=model.best_ntree_limit)
        test_pred = model.predict(xgb.DMatrix(X_test, feature_names=X_test.columns), ntree_limit=model.best_ntree_limit)

        oof_train[valid_idx] = valid_pred
        oof_test[:, i] = test_pred

        i += 1
    return model, oof_train, oof_test




model, oof_train, oof_test = run_xgb(xgb_params, X_train, X_test)




optR = OptimizedRounder()
optR.fit(oof_train, X_train['AdoptionSpeed'].values)
coefficients = optR.coefficients()
valid_pred = optR.predict(oof_train, coefficients)
qwk = quadratic_weighted_kappa(X_train['AdoptionSpeed'].values, valid_pred)
print("QWK = ", qwk)




coefficients_ = coefficients.copy()
coefficients_[0] = 1.65
train_predictions = optR.predict(oof_train, coefficients_).astype(np.int8)
test_predictions = optR.predict(oof_test.mean(axis=1), coefficients_).astype(np.int8)




submission = pd.DataFrame({'PetID': test['PetID'].values, 'AdoptionSpeed': test_predictions})
submission.to_csv('submission.csv', index=False)

