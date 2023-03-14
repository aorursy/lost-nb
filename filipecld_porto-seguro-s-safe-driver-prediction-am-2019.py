#!/usr/bin/env python
# coding: utf-8



# Importando bibliotecas
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn as sk # machine learning
from sklearn.impute import SimpleImputer

# verificando se as bases estão no diretório
import os
print(os.listdir("../input"))




train = pd.read_csv('../input/train.csv')




train.head()




train = train.drop(["id"], axis = 1)




train.head()




X_train = train.drop(["target"], axis = 1)
Y = train["target"]




X_train.head()




Y.head()




X_train.describe()




Y.isnull().any()




X_train.isnull().any()




X_train[X_train == -1].count()




X_train = X_train.drop(["ps_reg_03", "ps_car_03_cat", "ps_car_05_cat", "ps_car_14"], axis = 1)




X_train[X_train == -1].count()




X_columns = X_train.columns.tolist()




# Substituir os valores -1 pelo mais frequente 
imp = SimpleImputer(missing_values=-1, strategy='most_frequent')
imp.fit(X_train)




X_imp = imp.transform(X_train)




print(X_imp)




X = pd.DataFrame(X_imp,columns=X_columns)




X.head()




cat_cols = [col for col in X.columns if 'cat' in col]
bin_cols = [col for col in X.columns if 'bin' in col]
print(cat_cols, "\n")
print(bin_cols)




# Após a utilização da função Imputer() os dados ficaram float. 
# Temos que realizar a transformação das categóricas para category e as binárias para int 

for i in cat_cols: 
    X[i] = X[i].astype('category')

for j in bin_cols:
    X[j] = X[j].astype(int)




X.info()




X.head(10)




X.describe()




all_hists = X.hist(bins=20, figsize=(50,25))




# Agora vamos separar os dados em treinamento e teste
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)




X_train.head()




Y.head()




from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Create our first machine learning pipeline

ml_pipeline = Pipeline(pipeline_list)
ml_pipeline.fit(X_train, Y_train)
predictions = ml_pipeline.predict(X_test)
conf_mat = confusion_matrix(Y_test, predictions)
print("Matriz de Confusão\n", conf_mat)
test_acc = accuracy_score(predictions, Y_test)
print("Accuracy on the test set", test_acc)




# 1. Create a python list containing the classifiers
hypothesis_models = [
    KNeighborsClassifier(),
    Perceptron(),
    GaussianNB(),
    LinearSVC(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    LogisticRegression()
]


# 2. Uses the class StratifiedKFold to instanciate an object with n_splits=10
kfold = StratifiedKFold(n_splits=10, random_state=3)

# 3. Iterate over the list created in step 1 
for classifier in hypothesis_models:
  ml_pipeline.set_params(classifier=classifier)
  cv_results = cross_val_score(ml_pipeline, X_train, Y_train, cv=kfold, scoring="accuracy")
  print("--------------")
  print(str(classifier))
  print(cv_results)
  print(cv_results.mean())
  











