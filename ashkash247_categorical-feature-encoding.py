#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/cat-in-the-dat/train.csv')
df_test = pd.read_csv('../input/cat-in-the-dat/test.csv')
df_SS = pd.read_csv('../input/cat-in-the-dat/sample_submission.csv')


# In[2]:


#importing libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import OneHotEncoder 
from sklearn import preprocessing 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  




from xgboost import XGBClassifier

from sklearn import svm

from sklearn.model_selection import GridSearchCV


# In[3]:


df_train["target"].value_counts()#No missing values


# In[4]:


#encoding categorical columns
df_train_bin= df_train.iloc[:,1:6]

#USING CUSTOM MAP FOR BIN_3 AND BIN_4
bin_3_mapper = {'F':0, 
                'T':1}

df_train_bin['bin_3C'] = df_train_bin['bin_3'].replace(bin_3_mapper)

bin_4_mapper = {'N':0, 
                'Y':1}

df_train_bin['bin_4C'] = df_train_bin['bin_4'].replace(bin_4_mapper)
df_train_bin_cleaned=  df_train_bin.iloc[:,[0,1,2,5,6]]

#USING ONE HOT ENCODING FOR NOMINAL FEATURES
df_train_nom=df_train.iloc[:,6:16]

onehotencoder = OneHotEncoder() 
  
df_train_nom_cleaned = pd.get_dummies(df_train_nom.iloc[:,[0,1,2,3,4,5,6,7,8,9]])  ##col 5,6,7,8,9 removed

#df_train_nom_cleaned['nom_7C'] = label_encoder.fit_transform(df_train_nom['nom_7'])  
#df_train_nom_cleaned['nom_8C'] = label_encoder.fit_transform(df_train_nom['nom_8'])  
#df_train_nom_cleaned['nom_9C'] = label_encoder.fit_transform(df_train_nom['nom_9'])  
df_train_nom_cleaned


# In[ ]:





# In[5]:


#USING CUSTOM MAP LABELS FOR ORDINAL FEATURES
df_train_ord_time = df_train.iloc[:,16:24]

#ord_1 mapping
ord_1_mapper = {'Novice':0, 
                'Contributor':1,
               'Expert':2,
                'Master':3,
               'Grandmaster':4}

df_train_ord_time['ord_1C'] = df_train_ord_time['ord_1'].replace(ord_1_mapper)

#ord_2 mapping
ord_2_mapper = {'Freezing':0, 
                'Cold':1,
               'Warm':2,
                'Hot':3,
               'Boiling Hot':4,
                'Lava Hot':5
               }

df_train_ord_time['ord_2C'] = df_train_ord_time['ord_2'].replace(ord_2_mapper)

#ord_3 mapping
ord_3_mapper = {'a':0, 'b':1,'c':2,'d':3,'e':4,'f':5,'g':6,'h':7,'i':8,'j':9,'k':10,'l':11,'m':12,'n':13,'o':14
               }

df_train_ord_time['ord_3C'] = df_train_ord_time['ord_3'].replace(ord_3_mapper)

#ord_4 mapping
df_train_ord_time['ord_4C'] = df_train_ord_time['ord_4'].apply(lambda x: ord(x)-65)

#ord_5 mapping
label_encoder = preprocessing.LabelEncoder() 
df_train_ord_time['ord_5C'] = label_encoder.fit_transform(df_train_ord_time['ord_5'])   

df_train_ord_time_cleaned = df_train_ord_time.iloc[:,[0,6,7,8,9,10,11,12]]#6,7 time

df_train_all =  pd.concat([df_train_bin_cleaned,df_train_ord_time_cleaned], axis=1) #, df_train_nom_cleaned

df_train_all


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(df_train_all,df_train.iloc[:,24].values , test_size=0.2, random_state=42)


# In[7]:


X_train1= X_train.iloc[0:1000,:] # (240000, 13)
y_train1=y_train[0:1000]


# In[8]:


clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train1,y_train1)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


# In[9]:


y_train


# In[10]:


import graphviz

data = export_graphviz(clf,out_file=None,feature_names=X_train.columns,class_names=['0','1'],   
                         filled=True, rounded=True,  
                         special_characters=True)
graph = graphviz.Source(data)
graph


# In[11]:


"""#create an instance and fit the model 
logmodel = LogisticRegression(C=0.1338,
                        solver="lbfgs",
                        tol=0.0003,
                        max_iter=5000)# cv: if integer then it is the numbe
logmodel.fit(X_train, y_train)
y_pred = logmodel.predict_proba(X_test)
roc_auc_score(y_test,y_pred[:, 1]) #7830551086693487 ,0.7829327708379799


# In[ ]:





# In[12]:


"""ogistic = LogisticRegression()
# Create regularization penalty space
penalty = ['l1', 'l2']

# Create regularization hyperparameter space
C = np.logspace(0, 4, 2)

# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)

# Create grid search using 5-fold cross validation
clf = GridSearchCV(logistic, hyperparameters, cv=2, verbose=1,scoring='roc_auc')

best_model = clf.fit(X_train, y_train)


# In[13]:


"""p=best_model.predict_proba(X_test)
roc_auc_score(y_test,p[:, 1]) #0.7831027261785495


# In[ ]:





# In[14]:


"""logit_roc_auc = roc_auc_score(y_test,y_pred )
fpr, tpr, thresholds = roc_curve(y_test, logmodel.predict_proba(X_test))
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()"""


# In[15]:


"""
model_xg = XGBClassifier(show_progress=True,verbose_eval=50,silent =1)
model_xg.fit(X_train, y_train)
y_pred_xg = model_xg.predict_proba(X_test)
roc_auc_score(y_test,y_pred_xg[:, 1])"""


# In[16]:


"""SVC_model = svm.SVC(kernel='linear', C = 1.0)
SVC_model.fit(X_train, y_train)

y_pred_svc = SVC_model.predict_proba(X_test)
roc_auc_score(y_test,y_pred_svc[:, 1])"""


# In[17]:


"""from keras import Sequential
from keras.layers import Dense


# In[18]:


"""classifier = Sequential()
#First Hidden Layer
classifier.add(Dense(8, activation='relu', kernel_initializer='random_normal', input_dim=13))
#Second  Hidden Layer
classifier.add(Dense(8, activation='relu', kernel_initializer='random_normal'))
#3  Hidden Layer
classifier.add(Dense(8, activation='relu', kernel_initializer='random_normal'))
#4  Hidden Layer
classifier.add(Dense(8, activation='relu', kernel_initializer='random_normal'))
#5  Hidden Layer
classifier.add(Dense(8, activation='relu', kernel_initializer='random_normal'))
#6  Hidden Layer
classifier.add(Dense(8, activation='relu', kernel_initializer='random_normal'))
#Output Layer
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])


# In[19]:


"""classifier.fit(X_train,y_train, batch_size=10, epochs=2) # 0.7261


# In[20]:


"""pred_neural = classifier.predict(X_test)

roc_auc_score(y_test,pred_neural) #0.7831027261785495


# In[ ]:





# In[21]:


#Predicting values
"""
#encoding categorical columns
df_test_bin= df_test.iloc[:,1:6]

#USING CUSTOM MAP FOR BIN_3 AND BIN_4


df_test_bin['bin_3C'] = df_test_bin['bin_3'].replace(bin_3_mapper)



df_test_bin['bin_4C'] = df_test_bin['bin_4'].replace(bin_4_mapper)
df_test_bin_cleaned=  df_test_bin.iloc[:,[0,1,2,5,6]]

#USING ONE HOT ENCODING FOR NOMINAL FEATURES
df_test_nom=df_test.iloc[:,6:16]

onehotencoder = OneHotEncoder() 
  
df_test_nom_cleaned = pd.get_dummies(df_test_nom.iloc[:,[0,1,2,3,4,5,6]] )  ##col 7,8,9 removed

#USING CUSTOM MAP LABELS FOR ORDINAL FEATURES
df_test_ord_time = df_test.iloc[:,16:24]

#ord_1 mapping


df_test_ord_time['ord_1C'] = df_test_ord_time['ord_1'].replace(ord_1_mapper)

#ord_2 mapping


df_test_ord_time['ord_2C'] = df_test_ord_time['ord_2'].replace(ord_2_mapper)

#ord_3 mapping

df_test_ord_time['ord_3C'] = df_test_ord_time['ord_3'].replace(ord_3_mapper)

#ord_4 mapping
df_test_ord_time['ord_4C'] = df_test_ord_time['ord_4'].apply(lambda x: ord(x)-65)

#ord_5 mapping
label_encoder = preprocessing.LabelEncoder() 
df_test_ord_time['ord_5C'] = label_encoder.fit_transform(df_test_ord_time['ord_5'])   

df_test_ord_time_cleaned = df_test_ord_time.iloc[:,[0,6,7,8,9,10,11,12]]

df_test_all =  pd.concat([df_test_bin_cleaned,df_test_nom_cleaned,df_test_ord_time_cleaned], axis=1) 


# In[22]:


"""y_pred_actual = logmodel.predict_proba(df_test_all)
y_pred_actual 


# In[23]:


"""Id =  pd.DataFrame(df_test['id'] , columns=['id'])
predictions_Log  = pd.DataFrame(y_pred_actual[:,1]  , columns=['target']) 
id_predictions_Log = pd.concat([Id,predictions_Log],axis=1)
id_predictions_Log.head()
pd.DataFrame(id_predictions_Log, columns=['id','target']).to_csv('id_predictions_Log.csv',index = False)

