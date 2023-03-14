#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("../input/dmassign1/data.csv")


# In[3]:


data.info()


# In[4]:


data.head()


# In[5]:


data.Class.isna().count()
#No values are nan right now
#Missing values are being represented as '?'


# In[6]:


data.Class.count()


# In[7]:


data=data.drop_duplicates()


# In[8]:


data.info()


# In[9]:


categorical = []
for i in data.columns[1:]:
    if(data.dtypes[i]=='object'):
        categorical.append(i)
        
categorical
#categorical has all object type files


# In[10]:


#Numeric attributes are being stored as object type because they contain '?'ad
#Also removing all categorical attributes with missing values
numeric = categorical[:54]
alphabetical_to_remove = ['Col192','Col193','Col194','Col195','Col196','Col197']
numeric


# In[11]:


#Removing categorical attributes with missing values
data_alpha_rem = data.drop(alphabetical_to_remove,1)


# In[12]:


#Replace all ? in numeric columns to NAN
data_alpha_rem = data_alpha_rem.replace('?',np.nan)
            


# In[13]:


#Change all other values in numeric to float
for i in numeric:
    data_alpha_rem[i] = data_alpha_rem[i].astype(float)


# In[14]:


#Store all columns that contain NAN
null_valued = []
for i in data_alpha_rem.columns[1:-1]:
    if(data_alpha_rem.isna().any()[i]):
        null_valued.append(i)
        
null_valued


# In[15]:


#Replace NAN by mean
for i in null_valued:
    data_alpha_rem[i].fillna(data_alpha_rem[i].mean(),inplace=True)


# In[16]:


#Encode Col189 which has only 2 values
data_alpha_rem['Col189'].replace({
    'no':0,
    'yes':1
    },inplace=True)


# In[17]:


#Do onehot encoding of the other categorical variables
data_onehot = data_alpha_rem.copy()
data_onehot = pd.get_dummies(data_onehot, columns=['Col190','Col191'], prefix = ['Col190','Col191'])
data_onehot.head()


# In[18]:


#Find correlation for first 1300 data points
corr = data_onehot.head(1300).corr()


# In[19]:


#Sort columns with respect to correlation with Class
sorted_corr = abs(corr['Class']).sort_values()


# In[20]:


#If respectove correlation is less than 0.03, then remove
col_rem_corr = []
for i in data_onehot.columns[1:]:
    if(corr['Class'][i] < 0.03):
        col_rem_corr.append(i)
        
len(col_rem_corr)


# In[21]:


#Drop ID,CLass
data_id_rem = data_onehot.drop(['ID','Class'],1)


# In[22]:


#Drop low correlated attributes
data_rem = data_id_rem.drop(col_rem_corr,1)


# In[23]:


#many columns are removed
data_rem.info()


# In[24]:


#Scaling
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
#Using StandardScaler
scaler=StandardScaler()
scaled_data=scaler.fit(data_rem).transform(data_rem)
scaled_df=pd.DataFrame(scaled_data,columns=data_rem.columns)
scaled_df.tail()


# In[25]:


#Doing KMeans with 10 clusters, random state=42
from sklearn.cluster import KMeans

kmeans_clustering = KMeans(n_clusters=10, random_state=42)
cluster_labels = kmeans_clustering.fit_predict(scaled_df)

len(cluster_labels)


# In[26]:


#Calculating class frequency for each cluster
class_count_cluster=[[0 for i in range(5)] for j in range(10)]
for i in range(0,1300):
    cluster_ind = cluster_labels[i]
    class_ind = int(data['Class'][i])-1
    class_count_cluster[cluster_ind][class_ind] +=1

class_count_cluster


# In[27]:


#Denoting class label manually
class_label = [1,2,4,1,1,4,1,1,1,1]
res_data = []
for i in range(1300,13000):
    cluster_ind = cluster_labels[i]
    class_i = class_label[cluster_ind]
    res_data.append(class_i)
res_data 


# In[28]:


#writing out to csv file
out = [[data['ID'][i],res_data[i-1300]] for i in range(1300,13000)]
out_df = pd.DataFrame(data=out,columns=['ID','Class'])
out_df.to_csv(r'out_2_7.csv',index=False)


# In[29]:


from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):
 csv = df.to_csv(index=False)
 b64 = base64.b64encode(csv.encode())
 payload = b64.decode()
 html = '<a download="{filename}" href="data:text/csv;base64,{payload}"
target="_blank">{title}</a>'
 html = html.format(payload=payload,title=title,filename=filename)
 return HTML(html)
create_download_link(<submission_DataFrame_name>)

