#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

df = pd.read_csv("../input/dmassign1/data.csv")

# Any results you write to the current directory are saved as output.


# In[2]:


df.head()


# In[3]:


df.info(verbose = True,null_counts =True)


# In[4]:


column_objects = []
for name in df.columns:
    if df[name].dtype=='object':
        column_objects.append(name)


# In[5]:


df["Col197"].replace("me","ME",inplace = True)
df["Col197"].replace("M.E.","ME",inplace = True)
df["Col197"].replace("la","LA",inplace = True)
df["Col197"].replace("sm","SM",inplace = True)


# In[6]:


for i in range(1,179):
    colname = "Col"+str(i)
    if df[colname].dtype=='object':

        mean_val = int(df[df[colname]!='?'][colname].astype(int).mean())
        median_val = int(df[df[colname]!='?'][colname].astype(int).median())
        mode_val = int(df[df[colname]!='?'][colname].astype(int).mode()[0])

        print(colname," Found: ",len(df[df[colname]=='?']),"Replace with mean: ",str(mean_val),str(median_val),str(mode_val))
        df[colname].replace('?',mean_val,inplace = True)
        df[colname] = df[colname].astype(int)


# In[7]:


for i in range(179,189):
    colname = "Col"+str(i)
    if df[colname].dtype=='object':

        mean_val = float(df[df[colname]!='?'][colname].astype(float).mean())
        median_val = int(df[df[colname]!='?'][colname].astype(float).median())

        print(colname," Found: ",len(df[df[colname]=='?']),"Replace with mean: ",str(mean_val),str(median_val))
        df[colname].replace('?',mean_val,inplace = True)
        df[colname] = df[colname].astype(float)


# In[8]:


df_copy_safe = df


# In[9]:


for i in range(189,198):
    colname = "Col"+str(i)
    print(df[colname].describe())
    if df[colname].dtype=='object':

        mean_val = df[colname].mode().values[0]

        print(colname," Found: ",len(df[df[colname]=='?']),"Replace with mode: ",mean_val)
        df[colname].replace('?',mean_val,inplace = True)
        df[colname] = df[colname].astype('category')
    print(df[colname].describe())


# In[10]:


corr = df.corr()
corr
graph ,axis= plt.subplots()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),square=True, ax=axis, annot = False)


# In[11]:


print(corr["Class"])


# In[12]:


drop_columns = []
df = df.drop(drop_columns)
df_old = df.copy()
taken_cols = []

lastcol = 189
# lastcol = 189

for i in range(1,lastcol):
    colname = "Col"+str(i)
    taken_cols.append(colname)


# In[13]:


df = df[taken_cols]


# In[14]:


# corr_matrix = df.corr().abs()

# # Select upper triangle of correlation matrix
# upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# # Find index of feature columns with correlation greater than 0.95
# to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
# print(len(to_drop))
# df = df.drop(df[to_drop], axis=1)


# In[15]:


df.head()


# In[16]:


import seaborn as sns
f, ax = plt.subplots(figsize=(16, 10))
corr = data.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220,
square=True, ax=ax, annot = True);


# In[17]:


# df_n = pd.get_dummies(df, columns=["Col189","Col190","Col191","Col192","Col193","Col194","Col195","Col196","Col197"])


# In[18]:


# df_n.info()


# In[19]:


# df = df_n


# In[20]:


df.head()


# In[21]:


from sklearn.preprocessing import StandardScaler

x = df.loc[:].values
# Separating out the target
y = df_old.loc[:,['Class']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)


# In[22]:


from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA

pca = TSNE(n_components=2,n_jobs = 4,verbose = 1,perplexity=10, n_iter=2000)
# pca = PCA(n_components=2,svd_solver='full')

principalComponents = pca.fit_transform(x)


# In[23]:


X_labels = principalComponents[0:1300]
Y_labels = df_old["Class"][0:1300]
plt.figure(figsize=(16,10))
sns.scatterplot(
    x=X_labels[:,0], y=X_labels[:,1],
    hue=Y_labels,
    palette=sns.color_palette("hls", 5),
    legend="full",
    alpha=0.3
)


# In[24]:


# principledf = pd.DataFrame(X_labels)


# In[25]:


# centroid_center = []
# for i in range(5):
#     clust = []
#     for colname in principledf.columns:
#         clust.append(principledf[colname][Y_labels==i+1].mean())
#     centroid_center.append(clust)
# centroid_center = np.asarray(centroid_center)


# In[26]:


from sklearn.cluster import AgglomerativeClustering
labelsaglo = AgglomerativeClustering(n_clusters = 40).fit_predict(principalComponents)


# In[27]:


pd.DataFrame(labelsaglo)[0].value_counts()


# In[28]:


labelmapaglo = labelsaglo[0:1300]


# In[29]:


from scipy import stats as s

maplabelsaglo = dict()
for i in range(len(pd.DataFrame(labelsaglo)[0].value_counts())):
    maplabelsaglo[i] = int(s.mode(Y_labels[labelmapaglo==i])[0])


# In[30]:


totalmatch = 0
for i in range(1300):
    if (Y_labels[i] == maplabelsaglo[labelmapaglo[i]]):
        totalmatch += 1
totalmatch


# In[31]:


output = labelsaglo
for i in range(output.shape[0]):
    try:
        output[i] = maplabelsaglo[output[i]]
    except KeyError:
        output[i] = 2


# In[32]:


pd.DataFrame(output)[0].value_counts()


# In[33]:


tempdict = {"ID":df_old["ID"][1300:].values,"Class":output[1300:]}
df_submit = pd.DataFrame(tempdict)
df_submit.to_csv("666.csv",index = False)


# In[34]:


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
create_download_link(df_submit)


# In[35]:


#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
# END OF FILE
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################

