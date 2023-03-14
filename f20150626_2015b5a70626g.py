#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

#import sklearn.preprocessing as sk
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm


# In[2]:


df=pd.read_csv('../input/dataset.csv')


# In[3]:


df.info()
df2=df.drop('Class',axis=1)


# In[4]:


df2.info
df2.dropna(inplace=True)
df2.info()


# In[5]:


df2.duplicated(keep='first')


# In[6]:


df2=df2.drop_duplicates()


# In[7]:


df2['Account1'].replace({'aa':0,
           'ab':1, 'ac':2,'ad':3
           },inplace=True)


df2['Account1'].replace({'?':(df2['Account1'].mode())[0]
           },inplace=True)


# In[8]:


df2['History'].replace({'c0':0,
           'c1':1, 'c2':2,'c3':3,'c4':4
           },inplace=True)

df2['History'].replace({'?':(df2['History'].mode())[0]
           },inplace=True)


# In[9]:


df2['Motive'].replace({'p0':0,'p1':1, 'p2':2,'p3':3,'p4':4,'p5':5, 'p6':6,'p8':8,'p9':9,'p10':10            
           },inplace=True)
df2['Motive'].replace({'?':(df2['Motive'].mode())[0]},inplace=True)


# In[10]:


df2['Monthly Period'].replace({'?':(df2['Monthly Period'].mode())[0]},inplace=True)


# In[11]:


df2['Credit1'].replace({'?':(df2['Credit1'].mode())[0]},inplace=True)


# In[12]:


df2['Account2'].replace({'sacc1':1,'sacc2':2, 'sacc3':3,'sacc4':4,'sacc5':5,'Sacc4':4            
           },inplace=True)


# In[13]:


df2['Employment Period'].replace({'time1':1,'time2':2, 'time3':3,'time4':4,'time5':5            
           },inplace=True)


# In[14]:


df2['InstallmentRate'].replace({'?':(df2['InstallmentRate'].mode())[0]            
           },inplace=True)


# In[15]:


df2['Gender&Type'].replace({'F0':0,'F1':1, 'M0':2,'M1':3           
           },inplace=True)


# In[16]:


df2['Sponsors'].replace({'G1':1,'g1':1, 'G2':2,'G3':3           
           },inplace=True)


# In[17]:


df2['Tenancy Period'].replace({'?':(df2['Tenancy Period'].mode())[0]},inplace=True)


# In[18]:


df2['Plotsize'].replace({'XL':0,'LA':1, 'ME':2,'me':2,'M.E.':2,'SM':3,'la':1,'sm':3           
           },inplace=True)


# In[19]:


df2['Age'].replace({'?':(df2['Age'].mode())[0]},inplace=True)


# In[20]:


df2['Plan'].replace({'PL1':1,'PL2':2, 'PL3':3           
           },inplace=True)


# In[21]:


df2['Housing'].replace({'H1':1,'H2':2, 'H3':3           
           },inplace=True)


# In[22]:


df2['Post'].replace({'Jb1':1,'Jb2':2, 'Jb3':3,'jb4':4 ,'Jb4':4          
           },inplace=True)


# In[23]:


df2['Phone'].replace({'Yes':1,'No':0,'yes':1,'no':0           
           },inplace=True)


# In[24]:


df2['Expatriate'].replace({True:1,False:0           
           },inplace=True)


# In[25]:


df2['InstallmentCredit'].unique()
df2['InstallmentCredit'].replace({'?':-3000},inplace=True)
df2['InstallmentCredit'].replace({-3000:df2['InstallmentCredit'].median()},inplace=True)


# In[26]:


df2['Yearly Period'].unique()
df2['Yearly Period'].replace({'?':-3000},inplace=True)
df2['Yearly Period'].replace({-3000:df2['Yearly Period'].median()},inplace=True)


# In[27]:


#MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

df3=df2.drop('id',axis=1)
scaler=MinMaxScaler()
scaled_data=scaler.fit(df3).transform(df3)
scaled_df=pd.DataFrame(scaled_data,columns=df3.columns)


# In[28]:


import seaborn as sns
f, ax = plt.subplots(figsize=(10, 8))
corr = scaled_df.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax, annot = True);


# In[29]:



df5=df3.drop('Monthly Period',axis=1)#removed after correlation matrix of normalised values


# In[30]:


data5 = pd.get_dummies(df5, columns=["Account1","History", "InstallmentRate","Account2","Employment Period","Motive","Gender&Type","Sponsors","Tenancy Period","Plotsize","Plan","Housing","#Credits","Post","#Authorities","Phone","Expatriate"])


# In[31]:


from sklearn import preprocessing
#Performing Min_Max Normalization
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(data5)
dataN5 = pd.DataFrame(np_scaled)
dataN5.head()


# In[32]:


from sklearn.decomposition import PCA
pca5 = PCA(n_components=4)
pca5.fit(dataN5)
T5 = pca5.transform(dataN5)


# In[33]:


from sklearn.cluster import AgglomerativeClustering as AC
aggclus = AC(n_clusters = 3,affinity='euclidean',linkage='ward',compute_full_tree='auto')
y_aggclus= aggclus.fit_predict(dataN5)
plt.scatter(T5[:, 0], T5[:, 1], c=y_aggclus)


# In[34]:


print(y_aggclus)


# In[35]:


T = [[0,0,0], [0,0,0], [0,0,0]]
for i in range(1,174):
  T[df['Class'][i].astype(int)][y_aggclus[i]]=T[df['Class'][i].astype(int)][y_aggclus[i]]+1
  
print(T) 


# In[36]:


#Observing that most class 0 tuples are in cluster 1,
#most class 1 tuples in cluster 2 and most class 2 tuples in cluster 0 
pred0=2
pred1=0
pred2=1


# In[37]:


finalresult = []
for i in range(175,1031):
    if y_aggclus[i] == pred0:
        finalresult.append(0)
    elif y_aggclus[i] == pred1:
        finalresult.append(1)
    elif y_aggclus[i] == pred2:
        finalresult.append(2)
    else:
        print(i)


# In[38]:


len(finalresult)


# In[39]:


Answer=pd.DataFrame(finalresult,df['id'][175:1031])
Answer = Answer.rename(columns={0: "Class"})
Answer.head(10)
Answer.to_csv('DMAssignmentaggluc0503bestverify.csv')


# In[40]:


from IPython.display import HTML
import pandas as pd
import numpy as np
import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

create_download_link(*'DMAssignmentaggluc0503bestverify.csv'*)

