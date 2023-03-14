#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob #for searching files in directory
import warnings 
warnings.filterwarnings('ignore')


# In[2]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[3]:


base_dataset=glob.glob("../input/ieee-fraud-detection/*.csv")


# In[4]:


base_dataset[1].split("\\")[-1].split(".")[0]=10


# In[5]:


for i in base_dataset:
    print(i)


# In[6]:


""" iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.        
"""

for i in base_dataset:
    df=pd.read_csv(i)
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

i.split("\\")[-1].split(".")[0]=df
end_mem = df.memory_usage().sum() / 1024**2
print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
print("*******************************************************************************************")


# In[7]:


train=pd.read_csv("../input/ieee-fraud-detection/train_transaction.csv")


# In[8]:


train.head()


# In[9]:


train.shape


# In[10]:


null_value_table=(train.isna().sum()/train.shape[0])*100

retained_columns=null_value_table[null_value_table<30].index
drop_columns=null_value_table[null_value_table>30].index

train.drop(drop_columns,axis=1,inplace=True)

len(train.isna().sum().index)

cont=train.describe().columns

cat=[i for i in train.columns if i not in train.describe().columns]

for i in cat:
    train[i].fillna(train[i].value_counts().index[0],inplace=True)

for i in cont:
    train[i].fillna(train[i].median(),inplace=True)


# In[11]:


from sklearn.preprocessing import LabelEncoder
for i in cat:
    le=LabelEncoder()
    le.fit( train[i])
    x=le.transform( train[i])
    train[i]=x


# In[12]:


train=train.sample(200000)


# In[13]:


for i in train.var().sort_values(ascending=False).index[1:10]:
    x=np.array(train[i])
    qr1=np.quantile(x,0.25)
    qr3=np.quantile(x,0.75)
    iqr=qr3-qr1
    utv=qr3+(1.5*(iqr))
    ltv=qr1-(1.5*(iqr))
    y=[]
    for p in x:
        if p <ltv or p>utv:
            y.append(np.median(x))
        else:
            y.append(p)
    train[i]=y


# In[14]:


train["isFraud"].value_counts()


# In[15]:


X = np.array(train.iloc[:, train.columns != 'isFraud'])
y = np.array(train.iloc[:, train.columns == 'isFraud'])
print('Shape of X: {}'.format(X.shape))
print('Shape of y: {}'.format(y.shape))


# In[16]:


from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)


# In[17]:


print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))


# In[18]:


import seaborn as sns
import matplotlib.pyplot as plt
for i in  train.var().sort_values(ascending=False).index[1:10]:
    sns.boxplot( train[i])
    plt.show()


# In[19]:


import seaborn as sns
import matplotlib.pyplot as plt
for i in  train.var().sort_values(ascending=False).index[1:10]:
    for j in  train.var().sort_values(ascending=False).index[1:10]:
        plt.scatter( train[i], train[j])  
        plt.show()


# In[20]:


train.corr()


# In[21]:


sns.heatmap(train.corr())


# In[22]:


y=train['isFraud']
x=train.drop(['isFraud','M6'],axis=1)


# In[23]:


x.shape


# In[24]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=121)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[25]:


from sklearn.tree import DecisionTreeClassifier
ln=DecisionTreeClassifier()
ln.fit(X_train,y_train)
ln.predict(X_test)

from sklearn.metrics import confusion_matrix
print("\n",confusion_matrix(y_test,ln.predict(X_test)))

from sklearn.metrics import accuracy_score
print("\n\naccuracy_score using DT : ",accuracy_score(y_test,ln.predict(X_test)))


# In[26]:


ln.predict(X_test)[1:20]


# In[27]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(random_state=121)
rf.fit(X_train,y_train)
rf.predict(X_test)

from sklearn.metrics import confusion_matrix
print("\n",confusion_matrix(y_test,rf.predict(X_test)))

from sklearn.metrics import accuracy_score
print("\n\naccuracy_score using RF : ",accuracy_score(y_test,rf.predict(X_test)))


# In[28]:


rf.predict(X_test)[1:20] 


# In[29]:


test=pd.read_csv("../input/ieee-fraud-detection/test_transaction.csv")


# In[30]:


test.shape


# In[31]:


null_value_table=(test.isna().sum()/test.shape[0])*100

retained_columns=null_value_table[null_value_table<30].index
drop_columns=null_value_table[null_value_table>30].index

test.drop(drop_columns,axis=1,inplace=True)

len(test.isna().sum().index)

cont=test.describe().columns

cat=[i for i in test.columns if i not in test.describe().columns]

for i in cat:
    test[i].fillna(test[i].value_counts().index[0],inplace=True)

for i in cont:
    test[i].fillna(test[i].median(),inplace=True)


# In[32]:


from sklearn.preprocessing import LabelEncoder
for i in cat:
    le=LabelEncoder()
    le.fit(test[i])
    z=le.transform(test[i])
    test[i]=z


# In[33]:


pred=rf.predict(test)
pred=pred.astype(float)


# In[34]:


final_res=pd.DataFrame([test['TransactionID']]).T
final_res['isFraud']=pred


# In[35]:


final_res.head()


# In[36]:


final_res.to_csv('test_res.csv', index=False)


# In[37]:


sample=pd.read_csv("../input/ieee-fraud-detection/sample_submission.csv")
sample.head()


# In[38]:


sample=sample.drop('isFraud',axis=1)
sample = pd.merge(sample,final_res , on='TransactionID', how='inner')


# In[39]:


sample.head()


# In[40]:


from IPython.display import HTML

sample.to_csv('submission.csv', index=False)

def create_download_link(title = "Download CSV file", filename = "data.csv"):  
    html = '<a href={filename}>{title}</a>'
    html = html.format(title=title,filename=filename)
    return HTML(html)

# create a link to download the dataframe which was saved with .to_csv method
create_download_link(filename='submission.csv')


# In[41]:


create_download_link(filename='test_res.csv')


# In[ ]:




