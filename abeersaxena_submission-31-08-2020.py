#!/usr/bin/env python
# coding: utf-8



# Pandas and numpy for data manipulation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from patsy import dmatrices
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

from pandas import DataFrame




data = pd.read_csv('../input/loan-default-prediction/train_v2.csv.zip',low_memory=False)




data.shape




data.head()




#Identification of variables and data types
data.info()




data.describe()




data.isnull().sum().sum()




mis_val=data.isnull().sum()

for i in range(len(data.index)) :
    print("Nan in row ", i , " : " ,  data.iloc[i].isnull().sum())



mis_val_percent = 100 * data.isnull().sum() / len(data)

Creating table for missing value and their percentage in various columns


def tableformissingvalues(df):
 
        missingvalues = df.isnull().sum()
        percent = 100 * df.isnull().sum() / len(df)
        table = pd.concat([missingvalues,percent], axis=1)
        tablerenamed = table.rename(
            columns = {0 : 'Missing Values', 1 : 'Percentage'})
        
#         # Sort the table by percentage of missing descending
#         tablerenamed= tablerenamed[
#             mtablerenamed.iloc[:,1] != 0].sort_values(
#         'Percentage', ascending=False).round(1)
        
        # Print some summary information
#         print ("Your selected dataframe has " + str(data.shape[1]) + " columns.\n"      
#             "There are " + str(mis_val_table_ren_columns.shape[0]) +
#               " columns that have missing values.")
        
        # Return the dataframe with missing information
        return tablerenamed




tableformissingvalues(data).head(50)




#filling missing values with mean
data.fillna(data.mean(), inplace=True)




data.shape




#Adding new binary clasifier column
data['Classifier'] = [0 if x ==0 else 1 for x in data['loss']] 




data.shape




data.head()




#dropping all the columns which still have missing values
data.dropna(inplace=True)




data.shape




data.describe()




plt.figure(figsize=(5,5))
data['loss'].plot(kind='density')




data.info()




#data.to_csv (r'C:\Users\abeer\Desktop\trainingdata.csv', index = False, header=True)




trainingdata = pd.read_csv('../input/training-data',low_memory=False)




trainingdata.shape




trainingdata.isnull().sum().sum()




plt.hist(trainingdata['Classifier'],color = 'yellow', edgecolor = 'black',  bins = int(100/5))




sns.countplot(x ='Classifier', data = trainingdata) 




correlations_data = trainingdata.corr()['Classifier'].sort_values()




plt.figure(figsize=(50,35))

sns.heatmap(correlations_data,fmt='.1g',vmin=-1, vmax=1, center= 0)




# # Correlations between Features and Target

# Find all correlations and sort 

# Print the most negative correlations
print(correlations_data.head(15), '\n')

# Print the most positive correlations
print(correlations_data.tail(15))




#removing the columns which have  smame value in all the rows
for i in trainingdata.columns:
    if len(set(trainingdata[i]))==1:
        trainingdata.drop(labels=[i], axis=1, inplace=True)




trainingdata.shape




#let’s now find out the number of columns that are of the object data type and figure out how we can make those values numeric.
print("Data types and their frequency\n{}".format(trainingdata.dtypes.value_counts()))




#The categprical input data in the data frame
trainingdata.select_dtypes(include=['object'])
#after execution of the above line the data seems to be redundant as it ranges from 0-96424400336838002265208913920, and if it is truly categorical, we wuld dummify it. but observing the range of values, we see that dummification would not be very useful for our features, so we will drop them




for i in trainingdata.select_dtypes(include=['object']).columns:
    trainingdata.drop(labels=i, axis=1, inplace=True)




trainingdata.shape




zeroes = trainingdata[trainingdata['Classifier'] == 0] 
zeroes.shape




#top 500 zeroes
zeroes=zeroes[:500]
zeroes.shape




ones = trainingdata[trainingdata['Classifier'] == 1] 
ones.shape




#top 500 ones
ones=ones[:500]
ones.shape




frames=[zeroes,ones]




#this data will be used for training model, and it is also balanced
train=pd.concat(frames)
train.shape




Y1 = train['Classifier'] # dependent variable
X1 = train.drop('Classifier', axis=1)

vif  = [variance_inflation_factor(X1.values, i) for i in range(X1.shape[1])]




vif




VIF = DataFrame (vif,columns=['VIF Score'])




VIF["features"] = X1.columns




VIF.shape




VIF1 = VIF[VIF['VIF Score'] > 5.0] 




VIF1.shape




list1=list(VIF1['features']) 
list1




train.shape




train=train.drop(list1, axis = 1)




train.shape




# # # Split Into Training and Testing Sets
# Splitting data into training and testing
from sklearn.model_selection import train_test_split
# Separate out the features and targets
features = train.drop(columns=['Classifier','loss'])

targets = pd.DataFrame(train['Classifier'])

# Split into 80% training and 20% testing set
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size = 0.2, random_state = 42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)




# # Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)




# Convert y to one-dimensional array (vector)
y_train = np.array(y_train).reshape((-1, ))
y_test = np.array(y_test).reshape((-1, ))




X_train




X_test




testdata=pd.read_csv('../input/loan-default-prediction/test_v2.csv.zip',low_memory=False)




testdata.fillna(testdata.mean(), inplace=True) 




#dropping all the columns which still have missing values
testdata.dropna(inplace=True)




testdata.isnull().sum().sum()




#removing the columns which have  smame value in all the rows
for i in testdata.columns:
    if len(set(testdata[i]))==1:
        testdata.drop(labels=[i], axis=1, inplace=True)




#let’s now find out the number of columns that are of the object data type and figure out how we can make those values numeric.
print("Data types and their frequency\n{}".format(testdata.dtypes.value_counts()))




for i in testdata.select_dtypes(include=['object']).columns:
    testdata.drop(labels=i, axis=1, inplace=True)




list2=list1




testdata=testdata.drop(list2, axis = 1)




testdata.head()
#Since testdata does not have loss and classifier column, we will eliminate loss column from train also




train.head()




# Function to calculate mean absolute error
def crossvalidation(X_train, y_train, model):
    # Applying k-Fold Cross Validation
    from sklearn.model_selection import cross_val_score
    accuracies = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 5)
    return accuracies.mean()

# Takes in a model, trains the model, and evaluates the model on the test set
def trainmodel(model):
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions and evalute
    model_pred = model.predict(testdata)
    model_cross = crossvalidation(X_train, y_train, model)
    
    # Return the performance metric
    return model_cross




# # Naive Bayes
from sklearn.naive_bayes import GaussianNB
naive = GaussianNB()
naive_cross = trainmodel(naive)

print('Naive Bayes Performance on the test set: Cross Validation Score = %0.4f' % naive_cross)




NBprediction=naive.predict(testdata)




# # Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
random = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
random_cross = trainmodel(random)

print('Random Forest Performance on the test set: Cross Validation Score = %0.4f' % random_cross)




RFprediction=random.predict(testdata)
RFprediction




from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression(random_state = 0) 
logistic_cross=trainmodel(random)

print('Logistic regression Performance on the test set: Cross Validation Score = %0.4f' % logistic_cross)




LRprediction=random.predict(testdata)
LRprediction




from sklearn import svm
#create a classifier
SVM = svm.SVC(kernel="linear")
svm_cross=trainmodel(random)
print('SVM Performance on the test set: Cross Validation Score = %0.4f' % svm_cross)




SVMprediction=random.predict(testdata)
SVMprediction




#Since the testdata does not contain target value,I have calculated cross validation scores manually.

