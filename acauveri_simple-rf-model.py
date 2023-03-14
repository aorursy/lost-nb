#!/usr/bin/env python
# coding: utf-8



get_ipython().run_cell_magic('time', '', 'import pandas as pd\nfrom sklearn.ensemble import RandomForestClassifier')




get_ipython().run_cell_magic('time', '', '\nimport os\nprint(os.listdir("../input"))\n# Any results you write to the current directory are saved as output.\n\ndf_train = pd.read_csv(\'../input/train.csv\')\ndf_test = pd.read_csv(\'../input/test.csv\')')




df_train.shape




df_test.shape




df_train.head(2)




df_test.head(2)




df_train.isnull().sum()




df_test.isnull().sum()










df_train.date  = pd.to_datetime(df_train.date, format='%Y-%m-%d')




df_train['year'] = df_train.date.dt.year
df_train['month']=df_train.date.dt.month
df_train['day']=df_train.date.dt.day




df_train.head(2)





df_test.date  = pd.to_datetime(df_test.date, format='%Y-%m-%d')
df_test['year'] = df_test.date.dt.year
df_test['month']=df_test.date.dt.month
df_test['day']=df_test.date.dt.day




df_test.head(2)




df_train['year']=df_train['year'].astype('category')
df_train['month']=df_train['month'].astype('category')
df_train['day']=df_train['day'].astype('category')
df_train['store']=df_train['store'].astype('category')
df_train['item']=df_train['item'].astype('category')
df_train['sales']=df_train['sales'].astype('category')




y=pd.DataFrame()
y['sales']=df_train['sales']




df_train=df_train.drop(columns='date',axis=1)




df_train=df_train.drop(columns='sales',axis=1)




df_train.dtypes




df_test['year']=df_test['year'].astype('category')
df_test['month']=df_test['month'].astype('category')
df_test['day']=df_test['day'].astype('category')
df_test['store']=df_test['store'].astype('category')
df_test['item']=df_test['item'].astype('category')
df_test=df_test.drop(columns='date',axis=1)




df_test=df_test.drop(columns='id',axis=1)




df_test.dtypes




clf = RandomForestClassifier(max_depth=2, random_state=0)




x=df_train.iloc[:,0:5]  




clf=clf.fit(x,y)




get_ipython().run_cell_magic('time', '', 'output=clf.predict(df_test)\nresult=pd.DataFrame(output)\nresult')




test=pd.read_csv('../input/test.csv',usecols=['id'])
fin=pd.DataFrame(test)
fin['sales']=result
fin.to_csv('Sales.gz',index=False,compression='gzip')
 

