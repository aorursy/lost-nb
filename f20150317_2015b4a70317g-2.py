#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


data_orig1 = pd.read_csv("/input/train.csv", sep=',')
df = data_orig1

data_orig2 = pd.read_csv("/input/test_1.csv", sep=',')
df_test = data_orig2


# In[3]:


#replacing '?' with nan

x = float('nan')
df.replace('?', x, inplace = True)

df_test.replace('?', x, inplace = True)


# In[4]:


df.info()


# In[5]:


df = df.drop('ID', axis=1)
df_test = df_test.drop('ID', axis=1)


# In[6]:


#In the below cells all categorical attributes values have been obtained so as to know the skewness of data and 
#remove the attributes accordingly


# In[7]:


df['Worker Class'].value_counts()


# In[8]:


df_test['Worker Class'].value_counts()


# In[9]:


df['Enrolled'].value_counts()


# In[10]:


df['Married_Life'].value_counts()


# In[11]:


df_test['Married_Life'].value_counts()


# In[12]:


df['Schooling'].value_counts()


# In[13]:


df['MIC'].value_counts()


# In[14]:


df['MOC'].value_counts()


# In[15]:


df['Cast'].value_counts()


# In[16]:


df['Hispanic'].value_counts()


# In[17]:


df_test['Hispanic'].value_counts()


# In[18]:


df['Sex'].value_counts()


# In[19]:


df['MLU'].value_counts()


# In[20]:


df_test['MLU'].value_counts()


# In[21]:


df_test['Reason'].value_counts()


# In[22]:


df['Reason'].value_counts()


# In[23]:


df['Full/Part'].value_counts()


# In[24]:


df['Tax Status'].value_counts()


# In[25]:


df['Area'].value_counts()


# In[26]:


df_test['Area'].value_counts()


# In[27]:


df['State'].value_counts()


# In[28]:


df_test['State'].value_counts()


# In[29]:


df['Detailed'].value_counts()


# In[30]:


df_test['Detailed'].value_counts()


# In[31]:


df['Summary'].value_counts()


# In[32]:


df['MSA'].value_counts()


# In[33]:


df_test['MSA'].value_counts()


# In[34]:


df['REG'].value_counts()


# In[35]:


df_test['REG'].value_counts()


# In[36]:


df['MOVE'].value_counts()


# In[37]:


df_test['MOVE'].value_counts()


# In[38]:


df['Live'].value_counts()


# In[39]:


df_test['Live'].value_counts()


# In[40]:


df['PREV'].value_counts()


# In[41]:


df['Teen'].value_counts()


# In[42]:


df_test['Teen'].value_counts()


# In[43]:


df['COB FATHER'].value_counts()


# In[44]:


df_test['COB FATHER'].value_counts()


# In[45]:


df['COB MOTHER'].value_counts()


# In[46]:


df_test['COB MOTHER'].value_counts()


# In[47]:


df['COB SELF'].value_counts()


# In[48]:


df_test['COB SELF'].value_counts()


# In[49]:


df['Citizen'].value_counts()


# In[50]:


df_test['Citizen'].value_counts()


# In[51]:


df['Fill'].value_counts()


# In[52]:


df_test['Fill'].value_counts()


# In[53]:


df['Class'].value_counts()


# In[54]:


#In the next few cells those attributes are dropped which are either very skewed or have large number 
#of rows with nan values


# In[55]:


df = df.drop('COB SELF', axis=1)


# In[56]:


df_test = df_test.drop('COB SELF', axis=1)


# In[57]:


df = df.drop('COB MOTHER', axis=1)


# In[58]:


df_test = df_test.drop('COB MOTHER', axis=1)


# In[59]:


df = df.drop('COB FATHER', axis=1)


# In[60]:


df_test = df_test.drop('COB FATHER', axis=1)


# In[61]:


df = df.drop('Detailed', axis=1) #in best


# In[62]:


df_test = df_test.drop('Detailed', axis=1) #in best


# In[63]:


df = df.drop('State', axis=1)


# In[64]:


df_test = df_test.drop('State', axis=1)


# In[65]:


df.fillna(df.mean(), inplace = True)


# In[66]:


df_test.fillna(df_test.mean(), inplace = True)


# In[67]:


df = df.drop('Enrolled', axis=1)
df = df.drop('MLU', axis=1)
df = df.drop('Fill', axis=1)
df = df.drop('Reason', axis=1)


# In[68]:


df_test = df_test.drop('Enrolled', axis=1)
df_test = df_test.drop('MLU', axis=1)
df_test = df_test.drop('Fill', axis=1)
df_test = df_test.drop('Reason', axis=1)


# In[69]:





# In[69]:


#Replacing nan with mode for categorical variables in both training and test data


# In[70]:


for column in ['Worker Class', 'Married_Life', 'Schooling', 'MIC', 'MOC', 'Sex', 'Area', 'Summary', 'MSA', 
               'REG', 'MOVE', 'PREV', 'Full/Part', 'Tax Status', 'Teen', 'Hispanic', 'Citizen', 'Cast', 'Live']:
    df[column].fillna(df[column].mode()[0], inplace=True) #after best


# In[71]:


for column in ['Worker Class', 'Married_Life', 'Schooling', 'MIC', 'MOC', 'Sex', 'Area', 'Summary', 'MSA', 
               'REG', 'MOVE', 'PREV', 'Full/Part', 'Tax Status', 'Teen', 'Hispanic', 'Citizen', 'Cast', 'Live']:
    df_test[column].fillna(df_test[column].mode()[0], inplace=True) #after best


# In[72]:





# In[72]:


#Label encoding for categorical variables


# In[73]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df['Worker Class'] = labelencoder.fit_transform(df['Worker Class'])


# In[74]:


df_test['Worker Class'] = labelencoder.fit_transform(df_test['Worker Class'])


# In[75]:


df['Married_Life'] = labelencoder.fit_transform(df['Married_Life'])


# In[76]:


df_test['Married_Life'] = labelencoder.fit_transform(df_test['Married_Life'])


# In[77]:


df['Schooling'] = labelencoder.fit_transform(df['Schooling'])


# In[78]:


df_test['Schooling'] = labelencoder.fit_transform(df_test['Schooling'])


# In[79]:


df['MIC'] = labelencoder.fit_transform(df['MIC'])


# In[80]:


df_test['MIC'] = labelencoder.fit_transform(df_test['MIC'])


# In[81]:


df['MOC'] = labelencoder.fit_transform(df['MOC'])


# In[82]:


df_test['MOC'] = labelencoder.fit_transform(df_test['MOC'])


# In[83]:


df['Cast'] = labelencoder.fit_transform(df['Cast'])


# In[84]:


df_test['Cast'] = labelencoder.fit_transform(df_test['Cast'])


# In[85]:


df['Sex'] = labelencoder.fit_transform(df['Sex'])


# In[86]:


df_test['Sex'] = labelencoder.fit_transform(df_test['Sex'])


# In[87]:


df['Area'] = labelencoder.fit_transform(df['Area'])


# In[88]:


df_test['Area'] = labelencoder.fit_transform(df_test['Area'])


# In[89]:


df['Summary'] = labelencoder.fit_transform(df['Summary'])


# In[90]:


df_test['Summary'] = labelencoder.fit_transform(df_test['Summary'])


# In[91]:


df['MSA'] = labelencoder.fit_transform(df['MSA'])


# In[92]:


df_test['MSA'] = labelencoder.fit_transform(df_test['MSA'])


# In[93]:


df['REG'] = labelencoder.fit_transform(df['REG'])


# In[94]:


df_test['REG'] = labelencoder.fit_transform(df_test['REG'])


# In[95]:


df['MOVE'] = labelencoder.fit_transform(df['MOVE'])


# In[96]:


df_test['MOVE'] = labelencoder.fit_transform(df_test['MOVE'])


# In[97]:


df['PREV'] = labelencoder.fit_transform(df['PREV'])


# In[98]:


df_test['PREV'] = labelencoder.fit_transform(df_test['PREV'])


# In[99]:


df['Full/Part'] = labelencoder.fit_transform(df['Full/Part'])


# In[100]:


df_test['Full/Part'] = labelencoder.fit_transform(df_test['Full/Part'])


# In[101]:


df['Tax Status'] = labelencoder.fit_transform(df['Tax Status'])


# In[102]:


df_test['Tax Status'] = labelencoder.fit_transform(df_test['Tax Status'])


# In[103]:


df['Teen'] = labelencoder.fit_transform(df['Teen'])


# In[104]:


df_test['Teen'] = labelencoder.fit_transform(df_test['Teen'])


# In[105]:


df['Hispanic'] = labelencoder.fit_transform(df['Hispanic']) #after best


# In[106]:


df_test['Hispanic'] = labelencoder.fit_transform(df_test['Hispanic']) #after best


# In[107]:


df['Citizen'] = labelencoder.fit_transform(df['Citizen']) #after best


# In[108]:


df_test['Citizen'] = labelencoder.fit_transform(df_test['Citizen']) #after best


# In[109]:





# In[109]:


df.info()


# In[110]:


df['Live'] = labelencoder.fit_transform(df['Live'])


# In[111]:


df_test['Live'] = labelencoder.fit_transform(df_test['Live'])


# In[112]:





# In[112]:


#In the next few cells heat maps have been obtained and attributes have been dropped accordingly


# In[113]:



import seaborn as sns
f, ax = plt.subplots(figsize=(30, 25))
corr = df.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), annot = True);


# In[114]:


f, ax = plt.subplots(figsize=(30, 25))
corr = df_test.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), annot = True);


# In[115]:


df = df.drop('NOP', axis=1)


# In[116]:


df_test = df_test.drop('NOP', axis=1)


# In[117]:


df = df.drop('Vet_Benefits', axis=1)


# In[118]:


df_test = df_test.drop('Vet_Benefits', axis=1)


# In[119]:


df = df.drop('MSA', axis=1)


# In[120]:


df_test = df_test.drop('MSA', axis=1)


# In[121]:





# In[121]:


import seaborn as sns
f, ax = plt.subplots(figsize=(30, 25))
corr = df.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), annot = True);


# In[122]:


import seaborn as sns
f, ax = plt.subplots(figsize=(30, 25))
corr = df_test.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), annot = True);


# In[123]:


df = df.drop('Weaks', axis=1) #remove


# In[124]:


df_test = df_test.drop('Weaks', axis=1) #remove


# In[125]:


df = df.drop('REG', axis=1)


# In[126]:


df_test = df_test.drop('REG', axis=1)


# In[127]:


import seaborn as sns
f, ax = plt.subplots(figsize=(30, 25))
corr = df.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), annot = True);


# In[128]:


import seaborn as sns
f, ax = plt.subplots(figsize=(30, 25))
corr = df_test.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), annot = True);


# In[129]:


df = df.drop('IC', axis = 1)


# In[130]:


df_test = df_test.drop('IC', axis = 1)


# In[131]:


import seaborn as sns
f, ax = plt.subplots(figsize=(30, 25))
corr = df.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), annot = True);


# In[132]:


import seaborn as sns
f, ax = plt.subplots(figsize=(30, 25))
corr = df_test.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), annot = True);


# In[133]:





# In[133]:


import seaborn as sns
f, ax = plt.subplots(figsize=(30, 25))
corr = df.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), annot = True);


# In[134]:


import seaborn as sns
f, ax = plt.subplots(figsize=(30, 25))
corr = df_test.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), annot = True);


# In[135]:


df = df.drop('Live', axis=1) #after best


# In[136]:


df_test = df_test.drop('Live', axis=1) #after best


# In[137]:


df =  df.drop('Tax Status', axis=1)


# In[138]:


df_test =  df_test.drop('Tax Status', axis=1)


# In[139]:


y=df['Class']
X=df.drop(['Class'],axis=1)
X.head()

#X_test = df_test


# In[140]:


df_test.info()


# In[141]:





# In[141]:


#splitting of training data for training the models

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)


# In[142]:


from sklearn import preprocessing
#Performing Min_Max Normalization
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(X_train)
X_train = pd.DataFrame(np_scaled)
np_scaled_val = min_max_scaler.transform(X_val)
X_val = pd.DataFrame(np_scaled_val)
X_train.head()


# In[143]:


min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(df_test)
df_test = pd.DataFrame(np_scaled)

df_test.head()


# In[144]:


#Training and running Naive Bayes on the test data. This method is chosen as it gives the highest AUC ROC score.

np.random.seed(42)


# In[145]:


from sklearn.naive_bayes import GaussianNB as NB


# In[146]:


nb = NB()
nb.fit(X_train,y_train)
nb.score(X_val,y_val)


# In[147]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

y_pred_NB = nb.predict(X_val)
print(confusion_matrix(y_val, y_pred_NB))


# In[148]:


print(classification_report(y_val, y_pred_NB))


# In[149]:


y_pred_NB


# In[150]:


y_ans = nb.predict(df_test)


# In[151]:


print(y_ans.tolist())


# In[152]:





# In[152]:


#Training and testing Logistic Regression model for classification. Only used for analysis

from sklearn.linear_model import LogisticRegression


# In[153]:


lg = LogisticRegression(solver = 'liblinear', C = 1, multi_class = 'ovr', random_state = 42)
lg.fit(X_train,y_train)
lg.score(X_val,y_val)


# In[154]:


y_pred_LR = lg.predict(X_val)
print(confusion_matrix(y_val, y_pred_LR))


# In[155]:


print(classification_report(y_val, y_pred_LR))


# In[156]:





# In[156]:


#Training and testing Decision Tree model for classification. Only used for analysis

from sklearn.tree import DecisionTreeClassifier


# In[157]:


train_acc = []
test_acc = []
for i in range(1,15):
    dTree = DecisionTreeClassifier(max_depth=i)
    dTree.fit(X_train,y_train)
    acc_train = dTree.score(X_train,y_train)
    train_acc.append(acc_train)
    acc_test = dTree.score(X_val,y_val)
    test_acc.append(acc_test)


# In[158]:


plt.figure(figsize=(10,6))
train_score,=plt.plot(range(1,15),train_acc,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='green', markersize=5)
test_score,=plt.plot(range(1,15),test_acc,color='red',linestyle='dashed',  marker='o',
         markerfacecolor='blue', markersize=5)
plt.legend( [train_score, test_score],["Train Accuracy", "Validation Accuracy"])
plt.title('Accuracy vs Max Depth')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')


# In[159]:


from sklearn.tree import DecisionTreeClassifier

train_acc = []
test_acc = []
for i in range(2,30):
    dTree = DecisionTreeClassifier(max_depth = 6, min_samples_split=i, random_state = 42)
    dTree.fit(X_train,y_train)
    acc_train = dTree.score(X_train,y_train)
    train_acc.append(acc_train)
    acc_test = dTree.score(X_val,y_val)
    test_acc.append(acc_test)


# In[160]:


plt.figure(figsize=(10,6))
train_score,=plt.plot(range(2,30),train_acc,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='green', markersize=5)
test_score,=plt.plot(range(2,30),test_acc,color='red',linestyle='dashed',  marker='o',
         markerfacecolor='blue', markersize=5)
plt.legend( [train_score, test_score],["Train Accuracy", "Validation Accuracy"])
plt.title('Accuracy vs min_samples_split')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')


# In[161]:


dTree = DecisionTreeClassifier(max_depth=6, random_state = 42)
dTree.fit(X_train,y_train)
dTree.score(X_val,y_val)


# In[162]:


y_pred_DT = dTree.predict(X_val)
print(confusion_matrix(y_val, y_pred_DT))


# In[163]:


print(classification_report(y_val, y_pred_DT))


# In[164]:





# In[164]:


#Training and testing Random Forest model for classification. Only used for analysis

from sklearn.ensemble import RandomForestClassifier


score_train_RF = []
score_test_RF = []

for i in range(1,18,1):
    rf = RandomForestClassifier(n_estimators=i, random_state = 42)
    rf.fit(X_train, y_train)
    sc_train = rf.score(X_train,y_train)
    score_train_RF.append(sc_train)
    sc_test = rf.score(X_val,y_val)
    score_test_RF.append(sc_test)


# In[165]:


plt.figure(figsize=(10,6))
train_score,=plt.plot(range(1,18,1),score_train_RF,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='green', markersize=5)
test_score,=plt.plot(range(1,18,1),score_test_RF,color='red',linestyle='dashed',  marker='o',
         markerfacecolor='blue', markersize=5)
plt.legend( [train_score,test_score],["Train Score","Test Score"])
plt.title('Fig4. Score vs. No. of Trees')
plt.xlabel('No. of Trees')
plt.ylabel('Score')


# In[166]:


rf = RandomForestClassifier(n_estimators=13, random_state = 42)
rf.fit(X_train, y_train)
rf.score(X_val,y_val)


# In[167]:


y_pred_RF = rf.predict(X_val)
confusion_matrix(y_val, y_pred_RF)


# In[168]:


print(classification_report(y_val, y_pred_RF))


# In[169]:





# In[169]:


#Obtaining AUC ROC value for each classification model to know which model gives the best prediction


# In[170]:


from sklearn.metrics import roc_curve, roc_auc_score


# In[171]:


roc_curve(y_val, y_pred_NB)


# In[172]:


roc_auc_score(y_val, y_pred_NB)


# In[173]:


roc_auc_score(y_val, y_pred_RF)


# In[174]:


roc_auc_score(y_val, y_pred_DT)


# In[175]:


roc_auc_score(y_val, y_pred_LR)


# In[176]:





# In[176]:


#Training and testing Logistic Regression model for classification. Only used for analysis

rom sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score

rf_temp = RandomForestClassifier(n_estimators = 13)        #Initialize the classifier object

parameters = {'max_depth':[3, 5, 8, 10],'min_samples_split':[2, 3, 4, 5]}    #Dictionary of parameters

scorer = make_scorer(f1_score, average = 'micro')         #Initialize the scorer using make_scorer

grid_obj = GridSearchCV(rf_temp, parameters, scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier

grid_fit = grid_obj.fit(X_train, y_train)        #Fit the gridsearch object with X_train,y_train

best_rf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object

print(grid_fit.best_params_)


# In[177]:


rf_best = RandomForestClassifier(n_estimators = 13, max_depth = 5, min_samples_split = 2)
rf_best.fit(X_train, y_train)
rf_best.score(X_val,y_val)


# In[178]:


y_pred_RF_best = rf_best.predict(X_val)
confusion_matrix(y_val, y_pred_RF_best)


# In[179]:


print(classification_report(y_val, y_pred_RF_best))


# In[180]:


roc_auc_score(y_val, y_pred_RF_best)


# In[181]:





# In[181]:


res1 = []
for i in range(len(y_ans)):
    if y_ans[i] == 0:
        res1.append(0)
    elif y_ans[i] == 1:
        res1.append(1)


# In[182]:


#Obtaining the final result

res2 = pd.DataFrame(res1)
final = pd.concat([data_orig2["ID"], res2], axis=1).reindex()
final = final.rename(columns={0: "Class"})
final


# In[183]:


final.to_csv('2015B4A70317G.csv', index = False)


# In[184]:


from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):
csv = df.to_csv(index=False)
b64 = base64.b64encode(csv.encode())
payload = b64.decode()
html
=
'<a
download="{filename}"
href="data:text/csv;base64,{payload}"
target="_blank">{title}</a>'
html = html.format(payload=payload,title=title,filename=filename)
return HTML(html)
create_download_link(final)


# In[185]:


#FINISH

