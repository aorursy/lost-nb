#!/usr/bin/env python
# coding: utf-8

# In[6]:


# This tells matplotlib not to try opening a new window for each plot.
get_ipython().run_line_magic('matplotlib', 'inline')

# General libraries.
import numpy as np
import matplotlib.pyplot as plt

# SK-learn libraries for learning.
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.grid_search import RandomizedSearchCV
from sklearn.mixture import GMM
from sklearn import linear_model
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

# SK-learn libraries for evaluation.
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report

# SK-learn libraries for feature preprocessing.
from sklearn import preprocessing

# SK-learn libraries for dimensionality reduction.
from sklearn.decomposition import PCA

# Data analysis and plotting 
import pandas as pd
import seaborn as sns
from scipy import stats


# In[ ]:


Next we load the training and test data sets.


# In[7]:


ff = "../input/train.csv"
f = open(ff)
column_names = f.readline()

data = np.loadtxt(f, delimiter=",")

y, X = data[:, -1].astype('u1'), data[:, :-1]

ff_test = "../input/test.csv" # you will need to edit this directory
f_test = open(ff_test)
column_names_test = f_test.readline() # you'd needs this ordinarily

data_test = np.loadtxt(f_test, delimiter=",")

# note there are no labels here!
X_test = data_test

print('The test dataset contains {0} observations with {1} features each.'.    format(X_test.shape[0], X_test.shape[1]))
print('\t(The 1st one is not really a feature but an observation ID.)')
print('The training dataset contains {0} observations with the same {1} features each.'.    format(X.shape[0], X.shape[1]))
print('For this training set we know the corresponding category (forest cover type) of the '       '{0} observations.'.format(y.shape[0]))


# In[ ]:


Let's take a look at the distribution of values, for the continuous features.


# In[8]:


Train_panda = pd.read_csv('../input/train.csv')
Train_panda.ix[:,1:11].hist(figsize=(16,12),bins=50)
plt.show()


# In[9]:


Test_panda = pd.read_csv('../input/test.csv')
Test_panda.ix[:,1:11].hist(figsize=(16,12),bins=50)
plt.show()


# In[10]:


# X[:,9] = np.where(X[:,9]==0, np.median(X[X[:,9]!=0,9]), X[:,9])
# X_test[:,9] = np.where(X_test[:,9]==0, np.median(X_test[X_test[:,9]!=0,9]), X_test[:,9])


# In[11]:


# Shuffle the data, but make sure that the features and accompanying labels stay in sync.
np.random.seed(0)
shuffle = np.random.permutation(np.arange(X.shape[0]))
X, y = X[shuffle], y[shuffle]

# Split into train (90%) and dev (10%)
train_size = int(X.shape[0] * 0.9)
# Also discard 1st feature (ID number that doesn't provide info about the label)
y_train, X_train = y[:train_size], X[:train_size, 1:]
y_dev, X_dev = y[train_size:], X[train_size:, 1:]
X_test = X_test[:, 1:]
print(X_dev.shape, X_train.shape)


# In[ ]:


As previously mentioned, the first 10 features of each observation (Elevation to Horizontal_Distance_To_Fire_Points) are continuous, with different ranges, while the remaining 44 are all binary. 4 of those 44 binary features correspond to Wilderness Area (i.e., there are 4 possible types), so any observation will have one 1 and three 0's in those columns. The last 40 features correspond to Soil Type (i.e., there are 40 possible types), so any observation will have one 1 and thirty-nine 0's in those columns.


# In[12]:


prop_wilderness = 100*X_train[:,10:14].sum(axis=0)/X_train[:,10:14].sum()
prop_soil = 100*X_train[:,14:54].sum(axis=0)/X_train[:,14:54].sum()

plt.figure(figsize=(8, 4))
plt.bar(np.arange(4), prop_wilderness, align="center")
plt.title("Percentage of Wilderness Area cases in the training dataset")
plt.xticks(np.arange(4), np.array([str(i) for i in np.arange(1,5)]))

plt.figure(figsize=(12, 4))
plt.bar(np.arange(40), prop_soil, align="center")
plt.title("Percentage of Soil Type cases in the training dataset")
plt.xticks(np.arange(40), np.array([str(i) for i in np.arange(1,41)]))

plt.show()


# In[13]:


prop_wilderness = 100*X_test[:,10:14].sum(axis=0)/X_test[:,10:14].sum()
prop_soil = 100*X_test[:,14:54].sum(axis=0)/X_test[:,14:54].sum()

plt.figure(figsize=(8, 4))
plt.bar(np.arange(4), prop_wilderness, align="center")
plt.title("Percentage of Wilderness Area cases in the test dataset")
plt.xticks(np.arange(4), np.array([str(i) for i in np.arange(1,5)]))

plt.figure(figsize=(12, 4))
plt.bar(np.arange(40), prop_soil, align="center")
plt.title("Percentage of Soil Type cases in the test dataset")
plt.xticks(np.arange(40), np.array([str(i) for i in np.arange(1,41)]))

plt.show()


# In[14]:


# Scale to range [0,1]
    # Only the continuous features
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = np.copy(X_train)
X_dev_minmax = np.copy(X_dev)
X_test_minmax = np.copy(X_test)
X_train_minmax[:, :10] = min_max_scaler.fit_transform(X_train[:, :10])
X_dev_minmax[:, :10]  = min_max_scaler.transform(X_dev[:, :10])
X_test_minmax[:, :10] = min_max_scaler.transform(X_test[:, :10])

# Scale to mean = 0, sd = 1
std_scaler = preprocessing.StandardScaler()
# X_train_std = std_scaler.fit_transform(X_train)
# X_dev_std = std_scaler.transform(X_dev)
# X_test_std = std_scaler.transform(X_test)
    # Only the continuous features
X_train_std = np.copy(X_train)
X_dev_std = np.copy(X_dev)
X_test_std = np.copy(X_test)
X_train_std[:, :10] = std_scaler.fit_transform(X_train[:, :10])
X_dev_std[:, :10] = std_scaler.transform(X_dev[:, :10])
X_test_std[:, :10] = std_scaler.transform(X_test[:, :10])


# In[15]:


# Create a mixed distance metric that accounts for the different characteristic of the features
    # to give a similar weight to all of them
# First 10 features are continuous. The square of differences is applied to the values scaled 
    # to [0,1] (maximum value of the sum = 10)
# Last 44 features correspond to 2 features (wilderness area and soil type), with 4 and 40
    # categories each. A variant of Hamming distance is applied to them, so the maximum value
    # is 2 if two observations differ in both features
# The total distance is the square of the sum of those 12 values, divided by the square of 12,
    # so the maximum distance between any two observations will be 1
# The ranges of the first 10 features may vary in the dev and test datasets, so the distances
    # might be slightly greater than 1
def mixed_distance(x, y):
    return np.sqrt(np.sum((x[:10]-y[:10])**2) + 0.5*np.sum(x[10:14]!=y[10:14]) +
                          0.5*np.sum(x[14:54]!=y[14:54])) / np.sqrt(12)

k = 1 # We also tried many other values of k
# Try our own metric
kNN_mixed = KNeighborsClassifier(n_neighbors=k, metric=mixed_distance)
kNN_mixed.fit(X_train_minmax, y_train)
print(kNN_mixed.score(X_dev_minmax, y_dev))
# Try euclidean distance with unscaled data
kNN = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
kNN.fit(X_train, y_train)
print(kNN.score(X_dev, y_dev))
# Try euclidean distance with scaled data
kNN = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
kNN.fit(X_train, y_train)
print(kNN.score(X_dev, y_dev))
# Try euclidean distance with unscaled data and only continuous features
kNN.fit(X_train[:,:10], y_train)
print(kNN.score(X_dev[:,:10], y_dev))


# In[16]:


# Estimate by cross-validation the optimal number of neighbors (k)
# Try between 1 and the number of features (54)
k = {'n_neighbors': np.concatenate([np.arange(1, X_train.shape[1]+1)]).tolist()}
# The optimal value is low, so let's narrow the search from 1 to 11
k = {'n_neighbors': np.concatenate([np.arange(1, 10+1)]).tolist()}
best_param_kNN = GridSearchCV(KNeighborsClassifier(), k, scoring='accuracy')
best_param_kNN.fit(X_train, y_train)
optimal_k = best_param_kNN.best_params_['n_neighbors']
print('The optimal value for k is {0}'.format(optimal_k))

# Plot results
f1_vector = np.array([best_param_kNN.grid_scores_[x][1] for x in 
                      range(len(k['n_neighbors']))])
plt.figure(figsize=(8, 8))
plt.plot(k['n_neighbors'], f1_vector, marker='x')
plt.axvline(x=optimal_k, linewidth=1, linestyle='--', color='red')
plt.axhline(y=best_param_kNN.best_score_, linewidth=1, linestyle='--', color='red')
plt.xlabel("k (Nearest Neighbors)")
plt.ylabel("F1 score")
plt.title('F1 score per value of k')
plt.ylim([0, (np.ceil(best_param_kNN.best_score_*20)+1)/20])
plt.xlim([0, len(k['n_neighbors'])+1])


# In[17]:


kNN = KNeighborsClassifier(n_neighbors=optimal_k)

kNN.fit(X_train, y_train)
print('Accuracy using non-scaled data:      {0:.4f}'.    format(kNN.score(X_dev, y_dev)))

kNN.fit(X_train_std, y_train)
print('Accuracy using standardized data:    {0:.4f}'.    format(kNN.score(X_dev_std, y_dev)))

kNN.fit(X_train_minmax, y_train)
print('Accuracy using scaled-to-range data: {0:.4f}'.    format(kNN.score(X_dev_minmax, y_dev)))


# In[ ]:


The model performs better with non-scaled data (it could be argued that we searched for the optimal value for k using those data, but we did the same -out of this notebook- with standardized and scaled-to-range data).
get_ipython().set_next_input('Which are the cover types most commonly misclassified');get_ipython().run_line_magic('pinfo', 'misclassified')


# In[18]:


kNN = KNeighborsClassifier(n_neighbors=optimal_k)
kNN.fit(X_train[:, :10], y_train)
predicted_y_dev = kNN.predict(X_dev[:, :10])
print(classification_report(y_dev, predicted_y_dev))
# Confusion Matrix
CM = metrics.confusion_matrix(y_dev, predicted_y_dev)
CM_percentage = np.around(100*CM.astype('f2') / CM.sum(axis=1)[:, np.newaxis], 1)

# plt.figure(figsize=(12, 12))
# ax = plt.gca()
# ax.axes.get_xaxis().set_visible(False)
# ax.axes.get_yaxis().set_visible(False)
# ax.axis('off')
# table2 = plt.table(cellText=CM_percentage,rowLabels=np.arange(1,8),
#                            colLabels=np.arange(1,8),loc='center')
# plt.show()

# Print a table with the confusion matrix (percentages of row, so each row correspond to the
    # true cover type, and the diagonal values correspond to the Recall / 100
cover_type = [c.rjust(5) for c in map(str, np.unique(y_dev))]
print("|    |{}|{}|{}|{}|{}|{}|{}|".format(*cover_type))
print('------------------------------------------------')
table = []
for i,j in enumerate(np.unique(y_dev)):
    table.append([j, CM_percentage[i,0], CM_percentage[i,1], CM_percentage[i,2],
                  CM_percentage[i,3], CM_percentage[i,4], CM_percentage[i,5],
                  CM_percentage[i,6]])
for i in table:
    print("|{:4}|{:5.1f}|{:5.1f}|{:5.1f}|{:5.1f}|{:5.1f}|{:5.1f}|{:5.1f}|".format(*i))


# In[ ]:


The cover types most typically misclassified are 1 and 2 (confused with each other).

Keep record of the predictions in the dev set, as well as the accuracy, to ensemble all the models in a later step:


# In[19]:


kNN = KNeighborsClassifier(n_neighbors=optimal_k)
kNN.fit(X_train, y_train)
pred_y_dev_kNN = kNN.predict(X_dev)
acc_kNN = metrics.accuracy_score(y_dev, pred_y_dev_kNN)
print(acc_kNN)

CM = metrics.confusion_matrix(y_dev, pred_y_dev_kNN)
acc = CM.astype('f8') / CM.sum(axis=1)[:, np.newaxis]
acc_kNN_perType = np.diag(acc)
print(acc_kNN_perType)


# In[ ]:


Predict the test set:


# In[20]:


pred_y_test_kNN = kNN.predict(X_test)


# In[ ]:


## Naive Bayes (NB)


# In[21]:


NB_model = GaussianNB()
NB_model.fit(X_train_std[:,:10], y_train)
dev_predicted_labels = NB_model.predict(X_dev_std[:,:10])
print(metrics.accuracy_score(y_true=y_dev, y_pred=dev_predicted_labels))
print(metrics.classification_report(y_dev, dev_predicted_labels))


# In[ ]:


Keep record of the predictions in the dev set, as well as the accuracy, to ensemble all the models in a later step:


# In[22]:


NB = GaussianNB()
NB.fit(X_train_std[:,:10], y_train)
pred_y_dev_NB = NB.predict(X_dev_std[:,:10])
acc_NB = metrics.accuracy_score(y_dev, pred_y_dev_NB)
print(acc_NB)

CM = metrics.confusion_matrix(y_dev, pred_y_dev_NB)
acc = CM.astype('f8') / CM.sum(axis=1)[:, np.newaxis]
acc_NB_perType = np.diag(acc)
print(acc_NB_perType)


# In[ ]:


Predict the test set:


# In[23]:


pred_y_test_NB = NB.predict(X_test_std[:,:10])


# In[24]:


param_grid = {'criterion': ['gini', 'entropy'], 'max_features': [2, 5, 10, 20, 54], 
              'max_depth': [5, 10, 20, 25, 30, 40]}
best_param_DT = GridSearchCV(DecisionTreeClassifier(), param_grid, scoring='accuracy')
best_param_DT.fit(X_train, y_train)
optimal_criterion_DT = best_param_DT.best_params_['criterion']
print('The optimal criterion is {0}'.format(optimal_criterion_DT))
optimal_max_features_DT = best_param_DT.best_params_['max_features']
print('The optimal maximum number of features is {0}'.format(optimal_max_features_DT))
optimal_max_depth_DT = best_param_DT.best_params_['max_depth']
print('The optimal maximum depth of the tree is {0}'.format(optimal_max_depth_DT))

DT = DecisionTreeClassifier(criterion=optimal_criterion_DT, max_features=optimal_max_features_DT, 
                            max_depth=optimal_max_depth_DT, random_state=0)
DT.fit(X_train, y_train)

y_dev_dec = DT.predict(X_dev)
print(metrics.classification_report(y_dev, y_dev_dec))
print(metrics.accuracy_score(y_dev, y_dev_dec))


# In[25]:


DT = DecisionTreeClassifier(criterion='entropy', max_features=54, 
                            max_depth=25, random_state=0)
DT.fit(X_train, y_train)
pred_y_dev_DT = DT.predict(X_dev)
acc_DT = metrics.accuracy_score(y_dev, pred_y_dev_DT)
print(acc_DT)

CM = metrics.confusion_matrix(y_dev, pred_y_dev_DT)
acc = CM.astype('f8') / CM.sum(axis=1)[:, np.newaxis]
acc_DT_perType = np.diag(acc)
print(acc_DT_perType)


# In[26]:


pred_y_test_DT = DT.predict(X_test)


# In[27]:


# Train and predict with the random forest classifier
param_grid = {'criterion': ['gini', 'entropy'], 'n_estimators': [10, 50, 150], 
              'min_samples_split': [2, 4], 'max_features': [2, 5, 10, 20, 54], 
              'max_depth': [10, 20, 25, 30, 40]}
best_param_RF = GridSearchCV(ensemble.RandomForestClassifier(), param_grid, scoring='accuracy')
best_param_RF.fit(X_train, y_train)
optimal_criterion_RF = best_param_RF.best_params_['criterion']
print('The optimal criterion is {0}'.format(optimal_criterion_RF))
optimal_n_estimators_RF = best_param_RF.best_params_['n_estimators']
print('The optimal number of trees in the forest is {0}'.format(optimal_n_estimators_RF))
optimal_min_samples_split_RF = best_param_RF.best_params_['min_samples_split']
print('The optimal minimum number of samples required to split an internal node is {0}'.    format(optimal_min_samples_split_RF))
optimal_max_features_RF = best_param_RF.best_params_['max_features']
print('The optimal maximum number of features is {0}'.format(optimal_max_features_RF))
optimal_max_depth_RF = best_param_DT.best_params_['max_depth']
print('The optimal maximum depth of the tree is {0}'.format(optimal_max_depth_RF))

RF = ensemble.RandomForestClassifier(criterion=optimal_criterion_RF, 
                                     n_estimators=optimal_n_estimators_RF, 
                                     min_samples_split=optimal_min_samples_split_RF, 
                                     max_features=optimal_max_features_RF, 
                                     max_depth=optimal_max_depth_RF, random_state=0)
RF.fit(X_train,y_train)
y_dev_RF = RF.predict(X_dev)
print(metrics.classification_report(y_dev, y_dev_RF))
print(metrics.accuracy_score(y_dev, y_dev_RF))


# In[28]:


RF = ensemble.RandomForestClassifier(criterion='entropy', n_estimators=150, 
                                     min_samples_split=2, max_features=20, 
                                     max_depth=25, random_state=0)
RF.fit(X_train,y_train)
pred_y_dev_RF = RF.predict(X_dev)
acc_RF = metrics.accuracy_score(y_dev, pred_y_dev_RF)
print(acc_RF)

CM = metrics.confusion_matrix(y_dev, pred_y_dev_RF)
acc = CM.astype('f8') / CM.sum(axis=1)[:, np.newaxis]
acc_RF_perType = np.diag(acc)
print(acc_RF_perType)


# In[ ]:


Predict the test set:


# In[29]:


pred_y_test_RF = RF.predict(X_test)


# In[29]:




