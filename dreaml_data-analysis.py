#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as pl
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import xgboost as xgb
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Pretty display for notebooks
get_ipython().run_line_magic('matplotlib', 'inline')

# Any results you write to the current directory are saved as output.


# In[2]:


data = pd.read_csv("../input/train.csv", parse_dates=['timestamp'])
prices = data['price_doc']
features_raw = data.drop(['price_doc','id'], axis =1)


# In[3]:


data.info()


# In[4]:


# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
features_raw.fillna(0, inplace =True)
# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler((0,1e6))
numerical = features_raw.select_dtypes(include=['float64','int64']).keys()
features_raw[numerical] = scaler.fit_transform(features_raw[numerical])

# Show an example of a record with scaling applied
print (features_raw.head(n = 1))


# In[5]:


#One-hot encode the 'features' data using pandas.get_dummies()
objects = features_raw.select_dtypes(include=['object']).keys()
lbl = LabelEncoder()
for col in objects:
        lbl.fit(list(features_raw[col].values)) 
        features_raw[col] = lbl.transform(list(features_raw[col].values))
# remove boolean with _no
features = features_raw
features = features.drop(features.filter(regex='_no', axis=1),axis=1)
features.drop('timestamp',axis=1,inplace =True)
# Print the number of features after one-hot encoding
encoded = list(features.columns)
print ("{} total features after one-hot encoding.".format(len(encoded)))

# Uncomment the following line to see the encoded feature names
print (encoded)


# In[6]:


#Minimum price of the data
minimum_price = np.min(prices)

#Maximum price of the data
maximum_price = np.max(prices)

#Mean price of the data
mean_price = np.mean(prices)

#Median price of the data
median_price = np.median(prices)

#Standard deviation of prices of the data
std_price = np.std(prices)

# Show the calculated statistics
print ("Statistics for russian housing dataset:\n")
print ("Minimum price: Rub {:,.2f}".format(minimum_price))
print ("Maximum price: Rub {:,.2f}".format(maximum_price))
print ("Mean price: Rub {:,.2f}".format(mean_price))
print ("Median price Rub {:,.2f}".format(median_price))
print ("Standard deviation of prices: Rub {:,.2f}".format(std_price))
#added statistic for critica alfa = 0.05 double tailed cutover
#calculating standard error using correction for sample size n = 100
alfa_c = (std_price/99)*1.96
print ("critical alfa score for sample size of 100: Rub {:,.2f}".format(alfa_c))


# In[7]:


from sklearn.metrics import r2_score
def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """
    score = r2_score(y_true, y_predict) 
    return score


# In[8]:





# In[8]:


from sklearn.model_selection import train_test_split
#Shuffle and split the data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(features,prices, test_size =0.2, random_state = 33)
print ("Training and testing split was successful.")


# In[9]:


# Produce learning curves for varying training set sizes and maximum depths
from sklearn.model_selection import ShuffleSplit, train_test_split,
from sklearn.model_selection import learning_curve as curves

from sklearn.tree import DecisionTreeRegressor
def ModelLearning(X, y):
    """ Calculates the performance of several models with varying sizes of training data.
        The learning and testing scores for each model are then plotted. """
    
    # Create 10 cross-validation sets for training and testing
    cv = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 0)

    # Generate the training set sizes increasing by 50
    train_sizes = np.rint(np.linspace(1, X.shape[0]*0.8 - 1, 9)).astype(int)

    # Create the figure window
    fig = pl.figure(figsize=(10,7))

    # Create three different models based on max_depth
    for k, depth in enumerate([1,3,6,10]):
        
        # Create a Decision tree regressor at max_depth = depth
        regressor = DecisionTreeRegressor(max_depth = depth)

        # Calculate the training and testing scores
        sizes, train_scores, test_scores = curves(regressor, X, y,             cv = cv, train_sizes = train_sizes, scoring = 'r2')
        
        # Find the mean and standard deviation for smoothing
        train_std = np.std(train_scores, axis = 1)
        train_mean = np.mean(train_scores, axis = 1)
        test_std = np.std(test_scores, axis = 1)
        test_mean = np.mean(test_scores, axis = 1)

        # Subplot the learning curve 
        ax = fig.add_subplot(2, 2, k+1)
        ax.plot(sizes, train_mean, 'o-', color = 'r', label = 'Training Score')
        ax.plot(sizes, test_mean, 'o-', color = 'g', label = 'Testing Score')
        ax.fill_between(sizes, train_mean - train_std,             train_mean + train_std, alpha = 0.15, color = 'r')
        ax.fill_between(sizes, test_mean - test_std,             test_mean + test_std, alpha = 0.15, color = 'g')
        
        # Labels
        ax.set_title('max_depth = %s'%(depth))
        ax.set_xlabel('Number of Training Points')
        ax.set_ylabel('Score')
        ax.set_xlim([0, X.shape[0]*0.8])
        ax.set_ylim([-0.05, 1.05])
    
    # Visual aesthetics
    ax.legend(bbox_to_anchor=(1.05, 2.05), loc='lower left', borderaxespad = 0.)
    fig.suptitle('Decision Tree Regressor Learning Performances', fontsize = 16, y = 1.03)
    fig.tight_layout()
    fig.show()


# In[10]:


ModelLearning(features, prices)


# In[11]:


from sklearn.model_selection import validation_curve
def ModelComplexity(X, y):
    """ Calculates the performance of the model as model complexity increases.
        The learning and testing errors rates are then plotted. """
    
    # Create 10 cross-validation sets for training and testing
    cv = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 0)

    # Vary the max_depth parameter from 1 to 10
    max_depth = np.arange(1,11)

    # Calculate the training and testing scores
    train_scores, test_scores = validation_curve(DecisionTreeRegressor(), X, y,         param_name = "max_depth", param_range = max_depth, cv = cv, scoring = 'r2')

    # Find the mean and standard deviation for smoothing
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot the validation curve
    pl.figure(figsize=(7, 5))
    pl.title('Decision Tree Regressor Complexity Performance')
    pl.plot(max_depth, train_mean, 'o-', color = 'r', label = 'Training Score')
    pl.plot(max_depth, test_mean, 'o-', color = 'g', label = 'Validation Score')
    pl.fill_between(max_depth, train_mean - train_std,         train_mean + train_std, alpha = 0.15, color = 'r')
    pl.fill_between(max_depth, test_mean - test_std,         test_mean + test_std, alpha = 0.15, color = 'g')
    
    # Visual aesthetics
    pl.legend(loc = 'lower right')
    pl.xlabel('Maximum Depth')
    pl.ylabel('Score')
    pl.ylim([-0.05,1.05])
    pl.show()


# In[12]:


ModelComplexity(X_train, y_train)


# In[13]:


xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}
dtrain = xgb.DMatrix(features, prices)


# In[14]:


cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
    verbose_eval=50, show_stdv=False)
cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()


# In[15]:


num_boost_rounds = len(cv_output)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)


# In[16]:


fig, ax = pl.subplots(1, 1, figsize=(8, 13))
xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)


# In[17]:


#prepare test data:
test_data = pd.read_csv("../input/test.csv", parse_dates=['timestamp'])
id_test = test_data.id
test_features_raw = test_data.drop(['id'], axis =1)
test_features_raw.fillna(0, inplace =True)
test_features_raw[numerical] = scaler.transform(test_features_raw[numerical])
for col in objects:
        lbl.fit(list(test_features_raw[col].values)) 
        test_features_raw[col] = lbl.transform(list(test_features_raw[col].values))
# remove boolean with _no
test_features = test_features_raw
test_features = test_features.drop(test_features.filter(regex='_no', axis=1),axis=1)
test_features.drop('timestamp',axis=1,inplace =True)
Test_matrix = xgb.DMatrix(test_features)


# In[18]:


test_features.head()


# In[19]:


y_pred = model.predict(Test_matrix)

df_submit = pd.DataFrame({'id': id_test, 'price_doc': y_pred})

df_submit.to_csv('submit.csv', index=False)

