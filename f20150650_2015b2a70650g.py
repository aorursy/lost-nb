#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.




df=pd.read_csv("../input/dataset.csv",sep=",")




df.head()




df.tail()




df.info()




df = df.drop_duplicates()
df.info()




df.head()




df.info() 




df.columns = [c.replace(' ', '_') for c in df.columns]
df.info()




index_list = df[(df['Yearly_Period'] == '?')].index.tolist()
index_list
#index having ? returned




df1 =df




df2 = df.drop(df[df.Account1 == '?'].index)
df2 = df2.drop(df2[df2.Monthly_Period == '?'].index)
df2 = df2.drop(df2[df2.History == '?'].index)
df2 = df2.drop(df2[df2.Motive == '?'].index)
df2 = df2.drop(df2[df2.Credit1 == '?'].index)
df2 = df2.drop(df2[df2.InstallmentRate == '?'].index)
df2 = df2.drop(df2[df2.Tenancy_Period == '?'].index)
df2 = df2.drop(df2[df2.Age == '?'].index)
df2 = df2.drop(df2[df2.InstallmentCredit == '?'].index)
df2 = df2.drop(df2[df2.Yearly_Period == '?'].index)




df1.info()




df2.head()




df2.info()




df2['Monthly_Period']=df2.Monthly_Period.astype(int) 
df2['Credit1']=df2.Credit1.astype(int) 
df2['InstallmentRate']=df2.InstallmentRate.astype(int) 
df2['Tenancy_Period']=df2.Tenancy_Period.astype(int) 
df2['Age']=df2.Age.astype(int) 
df2['InstallmentCredit']=df2.InstallmentCredit.astype(float) 
df2['Yearly_Period']=df2.Yearly_Period.astype(float) 




df2.info()




df2['Sponsors'].unique() 




index_list1 = df2[(df2['Motive'] == 'p10')].index.tolist()
len(index_list1)




df1['Account1'].replace({    '?': 'ad' },inplace=True) 
df1['History'].replace({    '?': 'c2' },inplace=True) 
df1['Motive'].replace({    '?': 'p0' },inplace=True) 
df1.loc[1023]




df1.info()




replace = df2["Age"].mean()
df1['Age'].replace({    '?': replace },inplace=True) 
replace = df2["Monthly_Period"].mean()
df1['Monthly_Period'].replace({    '?': replace },inplace=True) 
replace = df2["Credit1"].mean()
df1['Credit1'].replace({    '?': replace },inplace=True) 
replace = df2["InstallmentRate"].mean()
df1['InstallmentRate'].replace({    '?': replace },inplace=True) 
replace = df2["Tenancy_Period"].mean()
df1['Tenancy_Period'].replace({    '?': replace },inplace=True) 
replace = df2["InstallmentCredit"].mean()
df1['InstallmentCredit'].replace({    '?': replace },inplace=True) 
replace = df2["Yearly_Period"].mean()
df1['Yearly_Period'].replace({    '?': replace },inplace=True) 
#df['Credit1'].unique() 
df1.head()




df2['Plotsize'].unique() 




df1['Sponsors'].replace({    'g1': 'G1' },inplace=True) 
df1['Plotsize'].replace({    'la' : 'LA',
                             'sm' : 'SM',
                             'M.E.' : 'ME',
                             'me' : 'ME'
                                  },inplace=True) 




df1['Sponsors'].unique()




df1['Monthly_Period']=df1.Monthly_Period.astype(int) 
df1['Credit1']=df1.Credit1.astype(int) 
df1['InstallmentRate']=df1.Monthly_Period.astype(int) 
df1['Tenancy_Period']=df1.Monthly_Period.astype(int) 
df1['Age']=df1.Monthly_Period.astype(int) 
df1['InstallmentCredit']=df1.Monthly_Period.astype(float) 
df1['Yearly_Period']=df1.Monthly_Period.astype(float)
df1.info()




import matplotlib.pyplot as plt
import seaborn as sns
f, ax = plt.subplots(figsize=(10, 8))
corr = df2.corr()
sns.heatmap(corr, cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax, annot = True);
#mask=np.zeros_like(corr, dtype=np.bool)




data = df2.drop(['id','Class'], 1)




data.info()




data.info()




data2 = data.drop(['Account1', 'Monthly_Period','History', 'Motive', 'Credit1','InstallmentRate','Tenancy_Period','Age','InstallmentCredit','Yearly_Period'],1)
data2.head()




data2 = pd.get_dummies(data2, columns=["Account2","Employment_Period","Gender&Type","Sponsors","Plotsize","Plan","Housing","Post","Phone","Expatriate"])
data2.head()




data2.info()




from sklearn import preprocessing
#Performing Min_Max Normalization
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(data2)
dataN2 = pd.DataFrame(np_scaled)
dataN2.head()




from sklearn.decomposition import PCA
pca2 = PCA(n_components=2)
pca2.fit(dataN2)
T2 = pca2.transform(dataN2)




df11 = df1.drop(['id','Class','Account1','Gender&Type','Monthly_Period','Credit1','Expatriate','Phone','Age','#Authorities','Yearly_Period'],1)
df11.head()




df11 = pd.get_dummies(df11, columns=["History","Motive","Account2","Employment_Period","Sponsors","Plotsize","Plan","Housing","Post"])
df11.info()
#Expatriate,Phone




from sklearn import preprocessing
#Performing Min_Max Normalization
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(df11)
dataN3 = pd.DataFrame(np_scaled)
dataN3.head()




from sklearn.decomposition import PCA
pca3 = PCA(n_components=2)
pca3.fit(dataN3)
T3 = pca3.transform(dataN3)




dataN3.info()




colors = ['red','green','blue']




from sklearn.cluster import AgglomerativeClustering as AC
aggclus = AC(n_clusters = 3,affinity='euclidean',linkage='ward',compute_full_tree='auto')
y_aggclus= aggclus.fit_predict(dataN3)
plt.scatter(T3[:, 0], T3[:, 1], c=y_aggclus)




plt.figure(figsize=(16, 8))

pred_pd = pd.DataFrame(y_aggclus)
arr = pred_pd[0].unique()

for i in arr:
    meanx = 0
    meany = 0
    count = 0
    for j in range(len(y_aggclus)):
        if i == y_aggclus[j]:
            count+=1
            meanx+=T3[j,0]
            meany+=T3[j,1]
            plt.scatter(T3[j, 0], T3[j, 1], c=colors[i])
    meanx = meanx/count
    meany = meany/count
    plt.annotate(i,(meanx, meany),size=30, weight='bold', color='black', backgroundcolor=colors[i])




y_aggclus




res3 = []
k = 0
for i in range(len(y_aggclus)):
    if y_aggclus[i] == 0:
        res3.append(2)
    elif y_aggclus[i] == 1:
        res3.append(1)
    elif y_aggclus[i] == 2:
        res3.append(0)
    
    #print(k , pred[i])
    #k=k+1   
res3




match = 0
for i in range(0,174):
    k = df.Class.values[i]
    if (res3[i] == k):
        match = match + 1
        
(match/175)*100    




resfinal = pd.DataFrame(res3)
finalagg = pd.concat([df["id"], resfinal], axis=1).reindex()
finalagg = finalagg.rename(columns={0: "Class"})
finalagg.tail()




finalagg.id.apply(str)




finall = finalagg.iloc[175:]




finall['Class']=finall.Class.astype(int) 




finall.info()




finall.to_csv('submissionAgg.csv', index = False)




from IPython.display import HTML
import pandas as pd
import numpy as np
import base64

def create_download_link(f, title = "Download CSV file", filename = "data.csv"):  
    csv = f.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

create_download_link(finall)

