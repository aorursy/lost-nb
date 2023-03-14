#!/usr/bin/env python
# coding: utf-8



import warnings
warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import re as re

# Data Load
train = pd.read_csv('../input/train.csv', header = 0, dtype={'Age': np.float64})
test  = pd.read_csv('../input/test.csv' , header = 0, dtype={'Age': np.float64})
full_data = [train, test] # for Data Preprocessing




from IPython.display import Image
Image(url= "https://static1.squarespace.com/static/5006453fe4b09ef2252ba068/t/5090b249e4b047ba54dfd258/1351660113175/TItanic-Survival-Infographic.jpg?format=1500w")




get_ipython().run_line_magic('pinfo', 'train.describe')




print (train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean())
# 1등칸에 탄사람들의 생존률이 높으며, 3등칸 탑승객들은 대부분 죽었다.




print (train[["Sex", "Survived"]].groupby(['Sex'], as_index=True).mean())
# 남성보다 여성이 상대적으로 많이 살아남았다.




for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())
# 생존이 1로 표기되어 있어서 비율계산을 바로 해도된다.
# 생존이 0이 었다면, 1-value




for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
print (train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())




total_df_ = pd.concat([train, test])
total_df_['Embarked'].value_counts()




train['Embarked'].isnull().sum(), test['Embarked'].isnull().sum()




for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
print (train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())




fare_series = train['Fare'].copy()




fare_series.sort_values(inplace=True)
fare_series.reset_index(inplace=True, drop=True)




fare_series[int(len(fare_series)/2)]




fare_series.median()




fare_series.plot()




for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median()) # 중앙값
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
print (train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean())




for dataset in full_data:
    age_avg 	   = dataset['Age'].mean()
    age_std 	   = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count) ## ??
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
    
train['CategoricalAge'] = pd.cut(train['Age'], 5)

print (train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())




def get_title(name):
	title_search = re.search(' ([A-Za-z]+)\.', name)
	# If the title exists, extract and return it.
	if title_search: ## 예외 Series의 내장함수는 예외처리 하기 난해
		return title_search.group(1)
	return ""

for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)

print(pd.crosstab(train['Title'], train['Sex']))




for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

print (train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())




for dataset in full_data:
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
    # Mapping titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    
    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
    # Mapping Fare
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
    # Mapping Age
    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']                           = 4
    # Feature Selection
    
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp',                 'Parch', 'FamilySize']
train = train.drop(drop_elements, axis = 1)
train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)

test  = test.drop(drop_elements, axis = 1)

train_x = train.drop(['Survived'],axis=1).values
train_y = train['Survived'].values

test_x = test.values




train_x[0,:], train_y[0]




from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(max_iter=5, random_state=42)
sgd_clf.fit(train_x,train_y)




sgd_clf.predict([test_x[0]])




from sklearn.model_selection import cross_val_score
fold_score = cross_val_score(sgd_clf, train_x, train_y, cv=5, scoring="accuracy")
print(fold_score)
print(np.mean(fold_score))




# 모두 살았다고 내벹는 모델
from sklearn.base import BaseEstimator

class MyClassifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1),dtype=bool) # 모두 살았다고 내벹는 모델




my_classifier = MyClassifier()
fold_score = cross_val_score(my_classifier, train_x, train_y, cv=5, scoring="accuracy") # 모두 살았다고 내뱉어도 정확도가 61%정도 나온다.
print(fold_score)
print(np.mean(fold_score))




from sklearn.model_selection import cross_val_predict
# cross_val_score vs cross_val_predict
train_pred = cross_val_predict(sgd_clf, train_x, train_y, cv=5)
train_pred[:10]




train_pred.sum(), train_y.size, train_y.sum()




import time




from sklearn.metrics import confusion_matrix
get_ipython().run_line_magic('time', 'confusion_matrix(train_y, train_pred)')
# 452 : 죽은사람을 죽었다고 예측 ( TN )
# 97 : 산사람을 죽었다고 예측 ( FP )
# 165 : 산사람을 죽었다고 예측 ( FN )
# 177 : 산사람을 살았다고 예측 ( TP )




import numpy as np
get_ipython().run_line_magic('time', 'np.bincount(train_y * 2 + train_pred)')




get_ipython().run_line_magic('time', 'np.bincount(train_y * 2 + train_pred).reshape(2,2)')




from sklearn.metrics import precision_score, recall_score, f1_score
score_info = train_y, train_pred
print("precision : {:.2f}%".format(precision_score(*score_info)))
print("recall : {:.2f}%".format(recall_score(*score_info)))
print("f1-score : {:.2f}%".format(f1_score(*score_info)))




y_scores = sgd_clf.decision_function(test_x)
y_scores_df = pd.DataFrame(y_scores,columns=['Threshold'])
y_scores_df.describe() # min -77 / max 32 / mean -31




y_scores = cross_val_predict(sgd_clf, train_x, train_y, cv=5, method="decision_function")
y_scores




from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds  = precision_recall_curve(train_y, y_scores)




import matplotlib.pyplot as plt
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="precision")
    plt.plot(thresholds, recalls[:-1], "g--", label="recall")
    plt.xlabel("thresholds")
    plt.legend(loc="center left")
    plt.ylim([0,1])




plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()




from sklearn.metrics import average_precision_score
average_precision = average_precision_score(train_y, y_scores)




from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature
precisions, recalls, thresholds  = precision_recall_curve(train_y, y_scores)

# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recalls, precisions, color='b', alpha=0.2, where='post')
plt.fill_between(recalls, precisions, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))




pd.DataFrame({
    'precisions':precisions,
    'recalls':recalls
}).plot.line(x="recalls",y="precisions")
plt.show()




train_y_90 = ( y_scores > 40 )




train_y_90




precision_score(train_y, train_y_90)




recall_score(train_y, train_y_90)




from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(train_y, y_scores)




def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1],[0,1],'k--')
    plt.axis([0,1,0,1])
    plt.xlabel("fpr")
    plt.ylabel("tpr")
plot_roc_curve(fpr, tpr)
plt.show()




pd.DataFrame({
    'fpr':fpr,
    'tpr':tpr
}).plot(x="fpr",y="tpr")




from sklearn.metrics import average_precision_score
average_precision = average_precision_score(train_y, y_scores)
average_precision




from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, train_x, train_y, cv=5, method="predict_proba")
# An array containing the probabilities of belonging to a given class




y_probas_forest




y_probas_forest = y_probas_forest[:,-1] # Use probability for positive class as score
fpr_forest, tpr_forest, thresholds_forest = roc_curve(train_y, y_probas_forest)




plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "RandomForest")
plt.legend(loc="lower right")
plt.show()




from sklearn.metrics import roc_auc_score
print("SGD : {:.2f}".format(roc_auc_score(train_y,y_scores)))
print("RandomForest : {:.2f}".format(roc_auc_score(train_y,y_probas_forest)))
























































