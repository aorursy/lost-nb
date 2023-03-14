import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn import datasets, linear_model

# Mounting gdrive
#from google.colab import drive
#drive.mount('/content/gdrive')


# Load train data from gdrive. Review shape of dataset and number of features

dataset_train = pd.read_csv('../input/train.csv')
dataset_train.shape
dataset_train.iloc[:1,:132]

# Separate features from Loss value in train dataset.

pd.set_option('display.max_columns',None)
loss = dataset_train['loss']
features = dataset_train.drop('loss', axis=1)


# Analyse stats on continuous features
features.describe()

# Analyse skewness of all continous features

print(features['cont1'].skew())
print(features['cont2'].skew())
print(features['cont3'].skew())
print(features['cont4'].skew())
print(features['cont5'].skew())
print(features['cont6'].skew())
print(features['cont7'].skew())
print(features['cont8'].skew())
print(features['cont9'].skew())
print(features['cont10'].skew())
print(features['cont11'].skew())
print(features['cont12'].skew())
print(features['cont13'].skew())
print(features['cont14'].skew())


# Analyse Loss statistics 

#minimum loss
mini_loss = np.amin(loss)

#maxmium loss
max_loss = np.amax(loss)

#mean loss
mean_loss = np.mean(loss)

#standard deviation loss
std_loss = np.std(loss)

#median loss
median_loss = np.median(loss)

print("Minimum Loss {}".format(mini_loss))
print("Maximum Loss {}".format(max_loss))
print("Mean Loss {}".format(mean_loss))
print("Standard Deviation Loss {}".format(std_loss))
print("Median Loss {}".format(median_loss))


# Checking skewness of Loss data
loss.skew()

# Plot heat map to understand correlation between continous features 

plt.figure(figsize=(10, 6))
sns.set()
sns.heatmap(features.iloc[:,117:].corr())


# Plot scatter plots to understand correlation between continuous features

pd.plotting.scatter_matrix(features.iloc[:2000,117:], alpha=0.1, figsize = (40,25), diagonal='kde')

# Benchmark model. Running Linear Regression on features data with split on train and val in ratio of 9:1 to obtain MSE

# Hot encode categorical features
features_bench_train = pd.get_dummies(features)

# Drop Id column from feature data
features_bench_train = features_bench_train.drop('id',axis=1)

# Log transform Loss target
loss_log_bench = np.log(loss)

# Split train and test data in ratio of 9:1
X_train, X_test, y_train, y_test = train_test_split(features_bench_train, 
                                                    loss_log_bench, 
                                                    test_size = 0.1, 
                                                    random_state = 0)


# Run Linear Regression on input train and validation data. Compute Mean absolute error. 
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

# Make predictions using the test data
y_pred = regr.predict(X_test)

# Remove high outlier predictions
for i, num in enumerate(y_pred):
  if num > 1000:
    y_pred[i] = y_pred[i-1] 


#Compute MAE from prediction and truth     
print("Mean Absolute error: %.2f"
      % mean_absolute_error(np.exp(y_test), np.exp(y_pred)))


# Hot encode categorical features and review dimension of dataset after one hot encoding, 116 cat features are converted
# into 1139 + 14 cont features + 1 id feature = 1154 features  

features = pd.get_dummies(features)
print(features.shape)

# drop 'Id' from dataframe
features = features.drop('id',axis=1)

# Seperate continuous features
cont_features = features.iloc[:,:14]

# Seperate Categorical features
cat_features = features.iloc[:,14:]


# PCA on continuous features

list = []

# Instanstiate PCA with number of component = 11
pca = PCA(n_components=11)

# Perform fit on 11 features
pca.fit(cont_features)

# Perform transform
reduced_cont_feature = pca.transform(cont_features)

# print explained variance ratio to show variance for all 11 components 
list.append(pca.explained_variance_ratio_)
print(list)

# Perform PCA for dimensionality reduction. Run PCA for number of components = total number of features after hot encoding to 
#understand explained variance ratio for all dimensions

list = []

pca = PCA(n_components=1139)

# Perform fit on 1139 cat_features
pca.fit(cat_features)

# Perform transform
transformed_feature = pca.transform(cat_features)

# print explained variance ratio to show variance for all components
list.append(pca.explained_variance_ratio_)

# Review explained variance and derive number of dimensions to be considered for 99% variance.  

np_list = np.array(list)
np.set_printoptions(threshold=1500)
print(np_list)

list = []

pca = PCA(n_components=345)

#perform fit on 345 dimensions 
pca.fit(cat_features)

#perform transform
reduced_cat_feature = pca.transform(cat_features)

#print explained variance ratio. 345 dimensions represents 99% variance in catergorical features 
list.append(pca.explained_variance_ratio_)
print(list)

# After PCA combine Cat and Cont features into single dataset.  
reduced_feature = np.hstack((reduced_cat_feature,reduced_cont_feature))
reduced_feature.shape

# Review data skewness. Log transform to get normal or uniform distribution.

print("data skewness before log transform {}".format(loss.skew()))
loss_log = np.log(loss+1)
print("data skewness after log transform {}".format(loss_log.skew()))


# Mean absolute error performance metric. 

def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """
    
    # TODO: Calculate the performance score between 'y_true' and 'y_predict'
    mae = mean_absolute_error(y_true, y_predict)
    
    # Return the score
    return mae

# Split original data into train and test in ratio of 9:1

X_train, X_test, y_train, y_test = train_test_split(reduced_feature, 
                                                    loss_log, 
                                                    test_size = 0.1, 
                                                    random_state = 0)

#Running DecisionTreeRegressor with default parameters

#import DecisionTreeRegressor from sklearn library
from sklearn.tree import DecisionTreeRegressor

#Instantiate class
regressor = DecisionTreeRegressor()

# Fit train feature and target
regressor.fit(X_train,y_train)

# Predict on Test features
y_pred = regressor.predict(X_test)

#Print MAE from predicted and actual Loss
print("Mean Absolute error: %.2f"
      % performance_metric(np.exp(y_test), np.exp(y_pred)))

# Model complexity graph for max_depth parameter
from sklearn.model_selection import validation_curve

def ModelComplexity(X, y):

    # Vary the max_depth parameter from 1 to 10
    max_depth = np.arange(5,15)

    # Calculate the training and testing scores
    train_scores, test_scores = validation_curve(DecisionTreeRegressor(), X, y, \
        param_name = "max_depth", param_range = max_depth, cv = None, scoring = 'neg_mean_absolute_error')
    
    # Find the mean and standard deviation for smoothing
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
  
    print(train_mean,test_mean)
      
    # Plot the validation curve
    plt.figure(figsize=(7, 5))
    plt.title('Decision Tree Regressor Complexity Performance')
    plt.plot(max_depth, train_mean, 'o-', color = 'r', label = 'Training Score')
    plt.plot(max_depth, test_mean, 'o-', color = 'g', label = 'Validation Score')
    plt.fill_between(max_depth, train_mean - train_std, \
        train_mean + train_std, alpha = 0.15, color = 'r')
    plt.fill_between(max_depth, test_mean - test_std, \
        test_mean + test_std, alpha = 0.15, color = 'g')

    # Visual aesthetics
    plt.legend(loc = 'lower right')
    plt.xlabel('Max depth')
    plt.ylabel('Score')
    plt.ylim([-0.40,-0.60])
    plt.show()

ModelComplexity(X_train, y_train)

#Running DecisionTreeRegressor with max_depth parameters

# Initialize Regressor with max_depth=8
regressor = DecisionTreeRegressor(max_depth=8)

# perform fit on train dataset
regressor.fit(X_train,y_train)

# perform predict on test dataset
y_pred = regressor.predict(X_test)

# MAE computed from predicted and actual loss
print("Mean Absolute error: %.2f"
      % performance_metric(np.exp(y_test), np.exp(y_pred)))

#Running DecisionTreeRegressor with max_depth parameters

# Initialize Regressor with max_depth=9
regressor = DecisionTreeRegressor(max_depth=9)

# perform fit on train dataset
regressor.fit(X_train,y_train)

# perform predict on test dataset
y_pred = regressor.predict(X_test)

# MAE computed from predicted and actual loss
print("Mean Absolute error: %.2f"
      % performance_metric(np.exp(y_test), np.exp(y_pred)))

# Model complexity graph for min_samples_split parameter

def ModelComplexity(X, y):

    # Vary the max_depth parameter from 1 to 10
    min_samples_split = [350,400,450,500,550,600,650]

    # Calculate the training and testing scores
    train_scores, test_scores = validation_curve(DecisionTreeRegressor(), X, y, \
        param_name = "min_samples_split", param_range = min_samples_split, cv = None, scoring = 'neg_mean_absolute_error')
    
    # Find the mean and standard deviation for smoothing
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
  
    print(train_mean,test_mean)
      
    # Plot the validation curve
    plt.figure(figsize=(7, 5))
    plt.title('Decision Tree Regressor Complexity Performance')
    plt.plot(min_samples_split, train_mean, 'o-', color = 'r', label = 'Training Score')
    plt.plot(min_samples_split, test_mean, 'o-', color = 'g', label = 'Validation Score')
    plt.fill_between(min_samples_split, train_mean - train_std, \
        train_mean + train_std, alpha = 0.15, color = 'r')
    plt.fill_between(min_samples_split, test_mean - test_std, \
        test_mean + test_std, alpha = 0.15, color = 'g')

    # Visual aesthetics
    plt.legend(loc = 'lower right')
    plt.xlabel('Minimum Sample split')
    plt.ylabel('Score')
    plt.ylim([-0.40,-0.60])
    plt.show()

ModelComplexity(X_train, y_train)

#Running DecisionTreeRegressor with max_depth and min_samples_split parameters 

# Initialize Regressor with max_depth=9 and min_samples_split=500
regressor = DecisionTreeRegressor(max_depth=9, min_samples_split=500)

# perform fit on train dataset
regressor.fit(X_train,y_train)

# perform prediction on test dataset
y_pred = regressor.predict(X_test)

# MAE computed from predicted and actual loss values 
print("Mean Absolute error: %.2f"
      % performance_metric(np.exp(y_test), np.exp(y_pred)))


# Deep NN training using Keras 

#Import required libraries 
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint  
from keras import optimizers

# Split the original data into train and test in ratio of 9:1 
X_train, X_test, y_train, y_test = train_test_split(reduced_feature, 
                                                    loss_log, 
                                                    test_size = 0.1, 
                                                    random_state = 0)

model = Sequential()

# First Dense Layer
model.add(Dense(356, input_dim=reduced_feature.shape[1], kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.3))

# Hidden Layer
model.add(Dense(200, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.2))

# Output Layer
model.add(Dense(1, kernel_initializer='normal', activation='linear'))

# Initialize optimizer with lr
adam = optimizers.adam(lr=0.0001)

# Compile model
model.compile(loss='mean_absolute_error', optimizer=adam)

# Define model checkpoint with filepath to save trained params
checkpointer = ModelCheckpoint(filepath='weights.best.DeepNN.hdf5', 
                               verbose=1, save_best_only=True)

# Fit Train data with 10% data used for model inferencing. 
model.fit(X_train, y_train, 
          validation_split=0.1,
          epochs=50, batch_size=250, callbacks=[checkpointer], verbose=1)

# Load best weights from saved model checkpoint
model.load_weights('weights.best.DeepNN.hdf5')

# predict on test data
y_pred = model.predict(X_test)

# Compute MAE from predicted and actual loss
print("Mean Absolute error: %.2f"
      % performance_metric(np.exp(y_test), np.exp(y_pred)))

# For multiple iterations run model training with variations in train and test splits to test robustness of model

mae_list = []

for i in range(3):
  
  X_train, X_test, y_train, y_test = train_test_split(reduced_feature, 
                                                    loss_log, 
                                                    test_size = 0.1, 
                                                    random_state = 42)
  
  model = Sequential()
  model.add(Dense(356, input_dim=reduced_feature.shape[1], kernel_initializer='normal', activation='relu'))
  model.add(Dropout(0.3))
  model.add(Dense(200, kernel_initializer='normal', activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(1, kernel_initializer='normal', activation='linear'))
  # Compile model
  
  adam = optimizers.adam(lr=0.0001)
  
  model.compile(loss='mean_absolute_error', optimizer=adam)

  checkpointer = ModelCheckpoint(filepath='weights.shuffle.best.DeepNN.hdf5', 
                               verbose=1, save_best_only=True)

  model.fit(X_train, y_train, 
          validation_split=0.1,
          epochs=50, batch_size=250, callbacks=[checkpointer], verbose=1)

  model.load_weights('weights.shuffle.best.DeepNN.hdf5')

  y_pred = model.predict(X_test)
  
  mae = performance_metric(np.exp(y_test), np.exp(y_pred))
  
  mae_list.append(mae)
  
  print("Mean Absolute error: %.2f"
      % mae)
  
# Print all MAE scores from multiple iterations
print(mae_list)

# Print mean MAE score
print(np.mean(mae_list))

# Print standard deviation for MAE
print(np.std(mae_list))

# Import XGBRegressor from xgboost   
from xgboost import XGBRegressor

# Instantiate Regressor with high estimator and gblinear booster 
model = XGBRegressor(n_estimators=1000, booster='gblinear')

# fit model on train data
xgb = model.fit(X_train, y_train)

# predict model on test data 
y_pred = xgb.predict(X_test)

# Compute MAE from predicted and actual loss
print("Mean Absolute error: %.2f"
      % performance_metric(np.exp(y_test), np.exp(y_pred)))

# Instantiate Regressor with high estimator and subsample size of 0.5
model = XGBRegressor(n_estimators=1000, subsample=0.5)

# fit model on train data
xgb = model.fit(X_train, y_train)

# predict model on test data 
y_pred = xgb.predict(X_test)

# Compute MAE from predicted and actual loss
print("Mean Absolute error: %.2f"
      % performance_metric(np.exp(y_test), np.exp(y_pred)))

# Instantiate Regressor with high estimator
model = XGBRegressor(n_estimators=1000)

# fit model on train data
xgb = model.fit(X_train, y_train)

# predict model on test data 
y_pred = xgb.predict(X_test)

# Compute MAE from predicted and actual loss
print("Mean Absolute error: %.2f"
      % performance_metric(np.exp(y_test), np.exp(y_pred)))

# Load test data from gdrive and review shape

dataset_test = pd.read_csv('../input/test.csv')
dataset_test.shape

# Hot encode test features with pandas get_dummies function.
features_test = pd.get_dummies(dataset_test)

# drop Id column from test data 
features_test = features_test.drop('id',axis=1)

# separate categorical and continuous features into different datasets. 
cont_test_features = features_test.iloc[:,:14]
cat_test_features = features_test.iloc[:,14:]


# perform PCA on continuous features with 11 components
from sklearn.decomposition import PCA
list = []

pca = PCA(n_components=11)
pca.fit(cont_test_features)

reduced_cont_test_feature = pca.transform(cont_test_features)

# perform PCA on categorical features with 345 components
pca = PCA(n_components=345)
pca.fit(cat_test_features)

reduced_cat_test_feature = pca.transform(cat_test_features)

# combine cat and cont features
reduced_test_feature = np.hstack((reduced_cat_test_feature,reduced_cont_test_feature))
reduced_test_feature.shape


import scipy.stats as stats

model = Sequential()
model.add(Dense(356, input_dim=reduced_feature.shape[1], kernel_initializer='normal', activation='relu'))
#model.add(Dropout(0.3))
model.add(Dense(200, kernel_initializer='normal', activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(1, kernel_initializer='normal', activation='linear'))
# Compile model
adam = optimizers.adam(lr=0.0001)
model.compile(loss='mean_absolute_error', optimizer=adam)

model.load_weights('weights.best.DeepNN.hdf5')

y_pred = model.predict(reduced_test_feature)

h = y_pred
h.sort()
hmean = np.mean(h)
hstd = np.std(h)
hmin = np.amin(h)
hmax = np.amax(h)
hmedian = np.median(h)

print(hmean,hstd,hmin,hmax,hmedian)

sns.distplot(h); 

h = np.exp(y_pred)
h.sort()
hmean = np.mean(h)
hstd = np.std(h)
hmin = np.amin(h)
hmax = np.amax(h)
hmedian = np.median(h)

print(hmean,hstd,hmin,hmax,hmedian)

sns.distplot(h); 

h = np.array(loss_log)
h.sort()
hmean = np.mean(h)
hstd = np.std(h)
hmin = np.amin(h)
hmax = np.amax(h)
hmedian = np.median(h)

print(hmean,hstd,hmin,hmax,hmedian)

sns.distplot(h); 

h = np.array(loss)
h.sort()
hmean = np.mean(h)
hstd = np.std(h)
hmin = np.amin(h)
hmax = np.amax(h)
hmedian = np.median(h)

print(hmean,hstd,hmin,hmax,hmedian)

sns.distplot(h); 
