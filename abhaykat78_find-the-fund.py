#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from google.colab import files
files.upload()


# In[3]:


get_ipython().system('pip install fastai==0.7.0')
from fastai.imports import*
from fastai.structured import *
from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from sklearn import metrics


# In[4]:


import pandas as pd


# In[5]:


df = pd.read_csv("train(2).csv")


# In[6]:


df = df.drop(["ID","comp_name","website","founded_on","hq_country_code","hq_state_code","hq_region","first_funding_date","last_funding_date"],axis=1)


# In[7]:


df.head()


# In[8]:


import matplotlib.pyplot as plt 
import seaborn as sns


# In[9]:


plt.figure(figsize=(10,7))
ax = sns.countplot(x="op_status", data=df)


# In[10]:


plt.figure(figsize=(10,7))
ax = sns.countplot(x="successful_investment", data=df)


# In[11]:


domains = df["domain"]


# In[12]:


domains = domains.values
domains = domains.astype(str)


# In[13]:


domains


# In[14]:


domains.reshape((1,len(domains)))
domains = list(domains)


# In[15]:



type(domains[0])


# In[16]:


unique=[]


# In[17]:


for x in domains:
  y= x.split("|")
  for z in y:
    if z not in unique:
      unique.append(z)
    else:
      continue

        


# In[18]:


unique


# In[19]:


dictOfWords = { i : unique[i] for i in range(0, len(unique) ) }


# In[20]:



dictOfWords


# In[21]:


b = dict([(value,key) for (key,value) in dictOfWords.items()])


# In[22]:


b


# In[23]:


df = df.values
df[:,1]=df[:,1].astype(str)


# In[24]:


multi_label = np.zeros((len(df[:,0]),len(unique)))


# In[25]:


for i in range(len(df[:,0])):
  x = df[i,1]
  l = x.split("|")
  for y in l:
    multi_label[i,b[y]]=1
  


# In[26]:


multi_label[0]


# In[27]:


df


# In[28]:


df = np.delete(df, 1, axis=1)


# In[29]:


df = np.hstack((multi_label,df))


# In[30]:



new = pd.DataFrame.from_records(df)


# In[31]:


new = new.drop([841,842],axis=1)


# In[32]:


new


# In[33]:


status = new[837].values


# In[34]:


status.shape


# In[35]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()


# In[36]:


status = labelencoder_X.fit_transform(status)


# In[ ]:





# In[37]:


import keras


# In[38]:


from keras.utils import to_categorical


# In[39]:


status_o = to_categorical(status)


# In[40]:


status_o.shape


# In[41]:


type(status_o)


# In[42]:


new1=new.drop([837],axis=1)
new1.shape


# In[43]:


new1 = new1.values


# In[44]:


print(status_o.shape)
new1.shape


# In[45]:


new1 = np.hstack((status_o,new1))


# In[46]:


new = pd.DataFrame.from_records(new1)


# In[47]:


new = new[]


# In[48]:


new[842] = new[842].fillna(new[842].mean())


# In[49]:


new


# In[50]:


new = new.drop([841],axis=1)


# In[51]:


new[843] = new[843].fillna(new[843].mean())


# In[52]:


for i in range(4):
  new[844+i] = new[844+i].fillna(new[844+i].mean())


# In[53]:


new


# In[54]:


new.to_csv("final.csv")


# In[55]:


get_ipython().system('ls')


# In[56]:


y = new.values[:,-1]
y


# In[57]:


X=new.values[:,0:-1]


# In[58]:


from keras.utils import to_categorical
y_o = to_categorical(y)


# In[59]:


X=(X-X.mean())/X.std()


# In[60]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_o, test_size=0.1, random_state=42)


# In[61]:


import tensorflow as tf
import keras


# In[62]:


from tensorflow.keras.layers import Input, Dense

input_layer = Input(shape = X.shape[1] )
hidden_layer = Dense(512, activation = 'relu',)(input_layer)
hidden_layer1 = Dense(512, activation = 'tanh',)(hidden_layer)
hidden_layer2 = Dense(512, activation = 'relu',)(hidden_layer1)
hidden_layer3 = Dense(512, activation = 'tanh',)(hidden_layer2)
output_layer = Dense(2, activation = 'softmax')(hidden_layer1)


# In[63]:


from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
model = Model(inputs=[input_layer], outputs=[output_layer])
model.compile(optimizer=optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# In[64]:


history = model.fit(X_train, y_train, validation_split = 0.1, epochs=100,batch_size=256)


# In[65]:



from fastai.imports import *
from fastai.structured import *
from sklearn.ensemble import RandomForestClassifier
from IPython.display import display
from sklearn import metrics


# In[66]:


new


# In[67]:


m = RandomForestRegressor(n_jobs=-1)
m.fit(new.drop([848], axis=1), new[848])


# In[68]:


m.score(new.drop([848], axis=1), new[848])


# In[69]:


from google.colab import files
files.upload()


# In[70]:


def preprocess(df,b):
    df = df.drop(["ID","comp_name","website","founded_on","hq_country_code","hq_state_code","hq_region"],axis=1)
    ''' domains = df["domain"]
    domains = domains.values
    domains = domains.astype(str)
    domains.reshape((1,len(domains)))
    domains = list(domains)
    unique=[]
    for x in domains:
        y= x.split("|")
        for z in y:
            if z not in unique:
              unique.append(z)
            else:
              continue
    dictOfWords = { i : unique[i] for i in range(0, len(unique) ) }
    b = dict([(value,key) for (key,value) in dictOfWords.items()])'''
    print(df.values.shape)
    df = df.values
    df[:,1]=df[:,1].astype(str)
    multi_label = np.zeros((len(df[:,0]),len(b)))
    print(multi_label.shape)
    for i in range(len(df[:,0])):
        x = df[i,1]
        l = x.split("|")
        for y in l:
          if i in b.keys():
            multi_label[i,b[y]]=1
    print(df.shape)
    df = np.delete(df, 1, axis=1)
    df = np.hstack((multi_label,df))
    print(df.shape)
    new = pd.DataFrame.from_records(df)
    new
    new = new.drop([841,842],axis=1)
    status = new[837].values
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder_X = LabelEncoder()
    status = labelencoder_X.fit_transform(status)
    import keras
    from keras.utils import to_categorical
    status_o = to_categorical(status)
    print(new[837])
    new1=new.drop([837],axis=1)
    new1 = new1.values
    new1 = np.hstack((status_o,new1))
    new = pd.DataFrame.from_records(new1)
    new[842] = new[842].fillna(new[842].mean())
    new = new.drop([841],axis=1)
    new[843] = new[843].fillna(new[843].mean())
    for i in range(4):
        new[844+i] = new[844+i].fillna(new[844+i].mean())

    return new


      


# In[71]:


final_test=preprocess(test,b)


# In[72]:


final_test


# In[73]:


np.unique(predic,return_counts=True)


# In[74]:


predic = m.predict(final_test)


# In[75]:


dd = pd.DataFrame(predic)
dd.to_csv('answer.csv')

