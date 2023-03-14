#!/usr/bin/env python
# coding: utf-8

# In[1]:


# The first thing we need to do is to import all of the relevant python libraries that we will need for our analysis. 
# Libraries such as numpy, pandas, statsmodels, and scikit-learn are frequently utilised by the data science community.
# Import useful packages. For every package you have not in Anaconda you could type 
# "$pip install" and the name of the package you want in Anaconda Prompt and then continue here in Notebook
import scipy as sc
import sklearn as sk
import sklearn.metrics as metrics
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.graph_objs as go
import plotly.tools as tls
import warnings
from collections import Counter
from sklearn.feature_selection import mutual_info_classif
warnings.filterwarnings('ignore')
from pandas.tools.plotting import scatter_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import time
from sklearn import *
import random
from IPython.display import display, HTML


# In[2]:


# Get the path from where I could import my data.
import os
os.getcwd()


# In[3]:


# I have the dataset available in an easily accessible CSV, and I can use the convenient pandas method read_csv() to load it into our environment.
# Import train data
train = pd.read_csv('../input/train.csv')
# Once our dataset is loaded we can inspect the data using the head() method to have a quick look at what columns and what kind of data we have available to work with.
train.head()
# Data Description
# In this competition, we will predict the probability that an auto insurance policy holder files a claim.
# In the train and test data, features that belong to similar groupings are tagged as such in the feature names 
# (e.g., ind, reg, car, calc). In addition, feature names include the postfix bin to indicate binary features and 
# cat to indicate categorical features. Features without these designations are either continuous or ordinal. 
# Values of -1 indicate that the feature was missing from the observation. The target columns signifies whether or not a claim
# was filed for that policy holder.


# In[4]:


# Import test data. In this dataset we have not the target column which we are called to predict.
test = pd.read_csv('../input/test.csv')
test.head()


# In[5]:


## Exploration and Cleaning
# Now we can perform some basic exploratory analysis to get a better understanding of what is in our data. For example we would like to know:
# How much data we have
# If there are any missing values
# What data type each column is
# The distribution of data in each column
# We could also take this opportunity to plot some charts to help us get an idea of what variables / features will prove useful. For example, if we where thinking of doing some regression analysis, scatter charts could give us a visual indication of correlation between features.
# The pandas library has plenty of built in functions to help us quickly understand summary information about our dataset. Below we use the shape() method to check how many rows are in our dataset and the describe() method to confirm whether or not our columns have missing values.
# Train Dataset shape
trainshape = np.shape(train)
print("This train dataset is a 2D array and contains {0} rows and columns".format(trainshape))


# In[6]:


print("Number of rows: ", train.shape[0])
counts = train.describe().iloc[0]
display(
    pd.DataFrame(
        counts.tolist(), 
        columns=["Count of values"], 
        index=counts.index.values
    ).transpose()
)


# In[7]:


datacolumns = train.columns


# In[8]:


target = train.target
target.count(), target.min(), target.max(), target.mean(), target.std()


# In[9]:


# Let’s now take a look at the number of instances (rows) that belong to each class. We can view this as an absolute count.
print(train.groupby('target').size())

# and let's take a look at the claims distribution:
train['target'].value_counts().plot(kind='bar')
plt.title('Claims distribution')
plt.xlabel('Claim was filled or not')
plt.ylabel('Number of vehicles')
sns.despine


# In[10]:


# Let's see the type of our variables:
train.dtypes


# In[11]:


# The first rows of the float64 type variables are:
float_types = train.select_dtypes(include=['float64'])
float_types.head()


# In[12]:


# And the first 5 rows of the int64 type variables are:
int_types = train.select_dtypes(include=['int64'])
int_types.head()


# In[13]:


# At this stage we would normally begin the process of cleaning our data set, which could involve: Filling in missing values
# Let's see where we have NAs (True where the column includes NAs and False where the column does not include NAs)
train.isnull().any()


# In[14]:


# But in our dataset we have the NAs as -1. Let's see how many -1 we have per column.
(train==-1).sum()
# As we can see we have a lot of NAs as -1 in the columns: ps_reg_03, ps_car_03_cat, ps_car_05_cat, ps_car_07_cat and ps_car_14.


# In[15]:


# For the test dataset we have:
(test==-1).sum()


# In[16]:


# From the dataset description we know that some features fall into a number of groups; this is indicated by a prefix (for example, ind_, ps_, car_).
# -Categorical features have the _cat suffix.
# -Binary features have the _bin suffix.
# -Features without suffix are either continuous or ordinal.

# Prepare lists of numeric, categorical and binary columns
# Numeric Features
numdata = [x for x in datacolumns if x[-3:] not in ['bin', 'cat']]
# Categorical Features
catdata = [x for x in datacolumns if x[-3:]=='cat']
# Binary Features
bindata = [x for x in datacolumns if x[-3:]=='bin']


# In[17]:


# For the other categorical variables with missing values, we can leave the missing value -1 as it is.
# For the continuous variables ps_reg_03, ps_car_12 and ps_car_14 which are continuous with missing values we will replace them by the mean.
mean_imp = Imputer(missing_values=-1, strategy='mean')
mode_imp = Imputer(missing_values=-1, strategy='most_frequent')
train['ps_reg_03'] = mean_imp.fit_transform(train[['ps_reg_03']]).ravel()
train['ps_car_01_cat'] = mode_imp.fit_transform(train[['ps_car_01_cat']]).ravel()
train['ps_car_02_cat'] = mode_imp.fit_transform(train[['ps_car_02_cat']]).ravel()
train['ps_car_04_cat'] = mode_imp.fit_transform(train[['ps_car_04_cat']]).ravel()
train['ps_car_07_cat'] = mode_imp.fit_transform(train[['ps_car_07_cat']]).ravel()
train['ps_car_09_cat'] = mode_imp.fit_transform(train[['ps_car_09_cat']]).ravel()
train['ps_ind_02_cat'] = mode_imp.fit_transform(train[['ps_ind_02_cat']]).ravel()
train['ps_ind_04_cat'] = mode_imp.fit_transform(train[['ps_ind_04_cat']]).ravel()
train['ps_ind_05_cat'] = mode_imp.fit_transform(train[['ps_ind_05_cat']]).ravel()
train['ps_car_12'] = mode_imp.fit_transform(train[['ps_car_12']]).ravel()
train['ps_car_14'] = mode_imp.fit_transform(train[['ps_car_14']]).ravel()
train['ps_car_11'] = mode_imp.fit_transform(train[['ps_car_11']]).ravel()
# For the test dataset we do the same thing because we want to clean it too:
test['ps_reg_03'] = mean_imp.fit_transform(test[['ps_reg_03']]).ravel()
test['ps_car_01_cat'] = mode_imp.fit_transform(test[['ps_car_01_cat']]).ravel()
test['ps_car_02_cat'] = mode_imp.fit_transform(test[['ps_car_02_cat']]).ravel()
test['ps_car_04_cat'] = mode_imp.fit_transform(test[['ps_car_04_cat']]).ravel()
test['ps_car_07_cat'] = mode_imp.fit_transform(test[['ps_car_07_cat']]).ravel()
test['ps_car_09_cat'] = mode_imp.fit_transform(test[['ps_car_09_cat']]).ravel()
test['ps_ind_02_cat'] = mode_imp.fit_transform(test[['ps_ind_02_cat']]).ravel()
test['ps_ind_04_cat'] = mode_imp.fit_transform(test[['ps_ind_04_cat']]).ravel()
test['ps_ind_05_cat'] = mode_imp.fit_transform(test[['ps_ind_05_cat']]).ravel()
test['ps_car_12'] = mode_imp.fit_transform(test[['ps_car_12']]).ravel()
test['ps_car_14'] = mode_imp.fit_transform(test[['ps_car_14']]).ravel()
test['ps_car_11'] = mode_imp.fit_transform(test[['ps_car_11']]).ravel()


# In[18]:


# Let's see if we have any missing values after the changes we made.
# We leave the two variables ps_car_03_cat and ps_car_05_cat as they are for now because we will need them in the process below.
(train==-1).sum()


# In[19]:


# We leave the two variables ps_car_03_cat and ps_car_05_cat as they are for now because we will need them in the process below.
(test==-1).sum()


# In[20]:


for column in catdata:    
    # Figure initiation 
    fig, ax = plt.subplots(figsize=(15, 4))
    # Calculate the percentage of target=1 per category value
    cat_perc = train[[column, 'target']].groupby([column],as_index=False).mean()
    cat_perc.sort_values(by='target', ascending=False, inplace=True)
    # Bar plot
    # Order the bars descending on target mean
    sns.barplot(ax=ax, x=column, y='target', data=cat_perc, order=cat_perc[column])
    plt.ylabel('% target', fontsize=18)
    plt.xlabel(column, fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.show();


# In[21]:


# As we can see the ps_car_11_cat column has a lot of unique values as a categorical variable. 
# Let's see how many unique values this column has.
len(set(train['ps_car_11_cat']))


# In[22]:


for column in bindata:    
    # Figure initiation 
    fig, ax = plt.subplots(figsize=(8,4))
    # Calculate the percentage of target=1 per category value
    cat_perc = train[[column, 'target']].groupby([column],as_index=False).mean()
    cat_perc.sort_values(by='target', ascending=False, inplace=True)
    # Bar plot
    # Order the bars descending on target mean
    sns.barplot(ax=ax, x=column, y='target', data=cat_perc, order=cat_perc[column])
    plt.ylabel('% target', fontsize=18)
    plt.xlabel(column, fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.show();


# In[23]:


# Correlation matrix for binary data.
corrbin = train[bindata].corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrbin, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10})


# In[24]:


# Zoomed correlation matrix because of possible relationship between ps_ind_11_bin, ps_ind_12_bin and ps_ind_13_bin.
k = 5 #number of variables for heatmap
cols = corrbin.nlargest(k, 'ps_ind_12_bin')['ps_ind_12_bin'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
f, ax = plt.subplots(figsize=(12, 12))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[25]:


# Correlation matrix for numeric data.
corrnum = train[numdata].corr()
f, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(corrnum, square=True, cbar=True, annot=True, fmt='.2f', annot_kws={'size': 10})


# In[26]:


# My attention goes on two different relationship squares. So, I will have a zoomed correlation matrix.
# The first relationship square is between ps_reg_01, ps_reg_02 and ps_reg_03 and 
# the second relationship square is between ps_car_12, ps_car_13 and ps_car_15.
k = 11 #number of variables for heatmap
cols = corrnum.nlargest(k, 'ps_reg_01')['ps_reg_01'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
f, ax = plt.subplots(figsize=(12, 12))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[27]:


# Correlation matrix for categorical data.
corrcat = train[catdata].corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrcat, square=True, cbar=True, annot=True, fmt='.2f', annot_kws={'size': 10})


# In[28]:


# My attention goes on two different relationship squares. So, I will have a zoomed correlation matrix.
# The first relationship square is between ps_reg_01, ps_reg_02 and ps_reg_03 and 
# the second relationship square is between ps_car_12, ps_car_13 and ps_car_15.
k = 15 #number of variables for heatmap
cols = corrcat.nlargest(k, 'ps_car_05_cat')['ps_car_05_cat'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
f, ax = plt.subplots(figsize=(12, 12))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[29]:


#scatterplot Binary data
sns.set()
cols = ['ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin']
sns.pairplot(train[cols], size = 2.5)
plt.show();


# In[30]:


#scatterplot Numeric data
sns.set()
cols = ['ps_reg_01', 'ps_reg_02', 'ps_reg_03', 'ps_car_12', 'ps_car_13']
sns.pairplot(train[cols], size = 3)
plt.show();


# In[31]:


#scatterplot Categorical data
sns.set()
cols = ['ps_car_12', 'ps_car_13', 'ps_car_15']
sns.pairplot(train[cols], size = 3)
plt.show();


# In[32]:


# We see that ps_car_03_cat and ps_car_05_cat have a large proportion of records with missing values. 
# We will remove these variables. (I didn't make it before because of not existing in the index the barplots couldn't appeared)
del train["ps_car_03_cat"]
del train["ps_car_05_cat"]
# We delete the same column in the test dataset too:
del test["ps_car_03_cat"]
del test["ps_car_05_cat"]


# In[33]:


# Drop the columns that we have decided won't be used in prediction
train = train.drop(["ps_calc_15_bin", "ps_calc_16_bin", "ps_calc_17_bin", "ps_calc_18_bin", "ps_calc_19_bin", "ps_calc_20_bin", "ps_ind_06_bin", "ps_ind_07_bin", "ps_ind_08_bin", "ps_ind_09_bin"], axis=1)
features = train.drop(["target"], axis=1).columns
test = test.drop(["ps_calc_15_bin", "ps_calc_16_bin", "ps_calc_17_bin", "ps_calc_18_bin", "ps_calc_19_bin", "ps_calc_20_bin", "ps_ind_06_bin", "ps_ind_07_bin", "ps_ind_08_bin", "ps_ind_09_bin"], axis=1)


# In[34]:


# Drop the columns that we have decided won't be used in prediction
train = train.drop(["ps_calc_01", "ps_calc_02", "ps_calc_03", "ps_calc_04", "ps_calc_05", "ps_calc_06", "ps_calc_07", "ps_calc_08", "ps_calc_09", "ps_calc_10", "ps_calc_11", "ps_calc_12", "ps_calc_13", "ps_calc_14"], axis=1)
features = train.drop(["target"], axis=1).columns
test = test.drop(["ps_calc_01", "ps_calc_02", "ps_calc_03", "ps_calc_04", "ps_calc_05", "ps_calc_06", "ps_calc_07", "ps_calc_08", "ps_calc_09", "ps_calc_10", "ps_calc_11", "ps_calc_12", "ps_calc_13", "ps_calc_14"], axis=1)


# In[35]:


# Correlation matrix for all variables:
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


# In[36]:


# At this point I can construct my model. The first thing to do is split our train dataset into training and test sets.
# I will take a simple approach and take a 70:30 randomly sampled split..
df_train, df_test = train_test_split(train, test_size=0.3)


# In[37]:


# Set up our RandomForestClassifier instance and fit to data
clf = RandomForestClassifier(n_estimators=30)
clf.fit(df_train[features], df_train["target"])


# In[38]:


# Make predictions
predictions = clf.predict(df_test[features])
probs = clf.predict_proba(df_test[features])
display(predictions)


# In[39]:


# Let's see the Accuracy of RandomForest Classifier:
score = clf.score(df_test[features], df_test["target"])
print("Accuracy: ", score)


# In[40]:


# Actual False and True predictions
get_ipython().magic('matplotlib inline')
confusion_matrix = pd.DataFrame(
    confusion_matrix(df_test["target"], predictions), 
    columns=["Predicted False", "Predicted True"], 
    index=["Actual False", "Actual True"]
)
display(confusion_matrix)


# In[41]:


# Calculate the fpr and tpr for all thresholds of the classification
fpr, tpr, threshold = roc_curve(df_test["target"], probs[:,1])
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[42]:


# In the following results we will see the id as a high important label but we know that it is not true. 
# So we don't care about id. The top important variables are ps_car_13, ps_reg_03 etc.
fig = plt.figure(figsize=(20, 18))
ax = fig.add_subplot(111)

df_f = pd.DataFrame(clf.feature_importances_, columns=["importance"])
df_f["labels"] = features
df_f.sort_values("importance", inplace=True, ascending=False)
display(df_f.head(6))

index = np.arange(len(clf.feature_importances_))
bar_width = 0.5
rects = plt.barh(index , df_f["importance"], bar_width, alpha=0.4, color='b', label='Main')
plt.yticks(index, df_f["labels"])
plt.show()


# In[43]:


df_test["prob_true"] = probs[:, 1]
df_risky = df_test[df_test["prob_true"] > 0.5]
display(df_risky.head(5)[["prob_true"]])


# In[44]:


# We need to separate the target "dataset" from the whole dataset
import numpy as np
from sklearn import datasets
train_x = train.drop(["target"], axis=1)
train_y = train.target
np.unique(train_y)


# In[45]:


kf = StratifiedKFold(n_splits=5,random_state=5,shuffle=True)
test_full=0
cv_score=[]
i=1
for train_index,test_index in kf.split(train_x, train_y):    
    print('\n{} of kfold {}'.format(i, kf.n_splits))
    xtr, xvl = train_x.loc[train_index], train_x.loc[test_index]
    ytr, yvl = train_y[train_index], train_y[test_index]
    
    lr = LogisticRegression(class_weight='balanced', C=0.003)
    lr.fit(xtr, ytr)
    pred_test = lr.predict_proba(xvl)[:,1]
    score = roc_auc_score(yvl, pred_test)
    print('roc_auc_score', score)
    cv_score.append(score)
    test_full += lr.predict_proba(test)[:,1]
    i+=1


# In[46]:


# Make predictions
predictions = lr.predict(test[features])
probs = lr.predict_proba(test[features])
display(predictions)


# In[47]:


test_pred = test_full/5
submit = pd.DataFrame({'id':test['id'],'target':test_pred})
submit.head()


# In[48]:


# Let's split again our train dataset in train and test datasets.
test_size = 0.30
seed = 7
train_x_train, train_x_test = model_selection.train_test_split(train_x, test_size=test_size, random_state=seed)
train_y_train, train_y_test = model_selection.train_test_split(train_y, test_size=test_size, random_state=seed)


# In[49]:


np.random.seed(0)
indices = np.random.permutation(len(train_x))
indices


# In[50]:


# And let's test a KNN Classifier
knn = KNeighborsClassifier()
knn.fit(train_x_train, train_y_train)


# In[51]:


# And let's see the Accuracy of KNN Classifier:
score = knn.score(train_x_test, train_y_test)
print("Accuracy: ", score)


# In[52]:


# The same method about LogisticRegression now
logistic = linear_model.LogisticRegression(C=1e5)
logistic.fit(train_x_train, train_y_train)
logistic.score(train_x_test, train_y_test)
score = logistic.score(train_x_test, train_y_test)
# And let's see the accuracy of LogisticRegression Classifier
print("Accuracy: ", score)


# In[53]:


# Predictions and probs about LogisticRegression:
predictions = lr.predict(train_x_test)
probs = lr.predict_proba(train_x_test)
display(predictions)


# In[54]:


# From scikil_learn I found these graphs where you can clearly see the comparison of classification of the classifiers.
print(__doc__)

np.random.seed(0)

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import calibration_curve

train, test = datasets.make_classification(n_samples=100000, n_features=20,
                                    n_informative=2, n_redundant=2)

train_samples = 100  # Samples used for training the models

train_train = train[:train_samples]
train_test = train[train_samples:]
test_train = test[:train_samples]
test_test = test[train_samples:]

# Create classifiers
# We saw before LogisticRegression and RandomForest classifiers in detail 
lr = LogisticRegression()
rfc = RandomForestClassifier(n_estimators=100)
# We will add GausianNB and LinearSVC
gnb = GaussianNB()
svc = LinearSVC(C=1.0)

# Plot calibration plots

plt.figure(figsize=(10, 10))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))

ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
for clf, name in [(lr, 'Logistic'),
                  (gnb, 'Naive Bayes'),
                  (svc, 'Support Vector Classification'),
                  (rfc, 'Random Forest')]:
    clf.fit(train_train, test_train)
    if hasattr(clf, "predict_proba"):
        prob_pos = clf.predict_proba(train_test)[:, 1]
    else:  # use decision function
        prob_pos = clf.decision_function(train_test)
        prob_pos =             (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
    fraction_of_positives, mean_predicted_value =         calibration_curve(test_test, prob_pos, n_bins=10)

    ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
             label="%s" % (name, ))

    ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
             histtype="step", lw=2)

ax1.set_ylabel("Fraction of positives")
ax1.set_ylim([-0.05, 1.05])
ax1.legend(loc="lower right")
ax1.set_title('Calibration plots  (reliability curve)')

ax2.set_xlabel("Mean predicted value")
ax2.set_ylabel("Count")
ax2.legend(loc="upper center", ncol=2)

plt.tight_layout()
plt.show()


# In[55]:


# LogisticRegression returns well calibrated predictions as it directly optimizes log-loss.
# So, I submit the results with the LogisticRegression.
############################################## For educational reasons only ###############################################

